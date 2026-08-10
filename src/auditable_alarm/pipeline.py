"""End-to-end orchestration from raw time series to audited disposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .acsc import CandidatePolicy
from .audit import GateResult, SourceRoleLedger, route_reader_record
from .claims import ClaimDecision, ClaimGate, authorize_claim
from .detectors import DetectorEnsemble, SeriesMap
from .evidence import build_event_evidence
from .hashing import sha256
from .models import (
    CaseRecord,
    ContrastCard,
    Disposition,
    DispositionRecord,
    EventEvidence,
    ReaderRecord,
    SourceRole,
)
from .readers import AuditableReader
from .retrieval import retrieve_dual_channel


@dataclass(frozen=True)
class TimeSeriesEvent:
    event_id: str
    channels: SeriesMap
    signature: Mapping[str, Any]


@dataclass(frozen=True)
class DiagnosisResult:
    evidence: EventEvidence
    history: ContrastCard
    reader: ReaderRecord
    reader_gates: GateResult
    numeric_disposition: Disposition
    reader_disposition: Disposition
    disposition: DispositionRecord
    claims: tuple[ClaimDecision, ...]
    detector_backends: Mapping[str, str]


class DiagnosisPipeline:
    """Conservative integration of detectors, retrieval, reader, and ACSC policy."""

    def __init__(
        self,
        *,
        detectors: DetectorEnsemble,
        cases: Sequence[CaseRecord],
        reader: AuditableReader,
        routing_policy: CandidatePolicy,
        retrieval_k: int = 5,
        reader_confidence_threshold: float = 0.80,
    ) -> None:
        self.detectors = detectors
        self.cases = tuple(cases)
        self.reader = reader
        self.routing_policy = routing_policy
        self.retrieval_k = retrieval_k
        self.reader_confidence_threshold = reader_confidence_threshold

    def diagnose(
        self,
        event: TimeSeriesEvent,
        training_normal: Sequence[TimeSeriesEvent],
        *,
        agreement: int = 1,
        as_of: str | None = None,
    ) -> DiagnosisResult:
        if not training_normal:
            raise ValueError("at least one training-normal event is required")
        ledger = self._build_ledger(event, training_normal)
        if ledger.role_of(event.event_id) is not SourceRole.RETROSPECTIVE_EVALUATION:
            raise RuntimeError("query event does not have the evaluation role")

        event_scores = self.detectors.score(event.channels, event.event_id)
        reference_scores: dict[str, list[float]] = {name: [] for name in event_scores}
        for normal in training_normal:
            scored = self.detectors.score(normal.channels, normal.event_id)
            if set(scored) != set(reference_scores):
                raise ValueError("detector ensemble changed between reference and query scoring")
            for name, values in scored.items():
                reference_scores[name].extend(values)
        normal_binding = sha256(training_normal)
        calibration_refs = {
            name: f"{backend}:training-normal:{normal_binding}"
            for name, backend in self.detectors.calibration_refs.items()
        }
        evidence = build_event_evidence(
            event.event_id,
            event_scores,
            reference_scores,
            calibration_refs,
        )

        signature = dict(event.signature)
        signature["detector_percentiles"] = dict(evidence.detector_percentiles)
        signature["d_score"] = evidence.d_score
        history = retrieve_dual_channel(
            event.event_id,
            signature,
            self.cases,
            k=self.retrieval_k,
            as_of=as_of,
        )
        reader_record = self.reader.read(evidence, history)
        reader_route, gates = route_reader_record(
            reader_record,
            evidence,
            confidence_threshold=self.reader_confidence_threshold,
        )
        numeric = self.routing_policy.route(evidence.d_score, agreement)
        failed = list(gates.failed_gates)
        if numeric is Disposition.ABSTAIN:
            failed.append("P1_ACSC_ABSTAIN")
        if (
            numeric is not Disposition.ABSTAIN
            and reader_route.disposition is not Disposition.ABSTAIN
            and numeric is not reader_route.disposition
        ):
            failed.append("P2_READER_POLICY_DISAGREEMENT")
        final = (
            numeric
            if not failed and numeric is reader_route.disposition
            else Disposition.ABSTAIN
        )
        disposition = DispositionRecord(
            route_id=str(uuid4()),
            query_id=event.event_id,
            reader_record_hash=reader_record.record_hash,
            evidence_hash=evidence.content_hash,
            disposition=final,
            failed_gates=tuple(failed),
            policy_id=f"{self.routing_policy.policy_id}+reader-six-gate-v1",
        )
        gate_results = {
            ClaimGate.INTEGRITY: True,
            ClaimGate.DETECTOR: all(bool(values) for values in event_scores.values()),
            ClaimGate.RETRIEVAL: history.provenance_valid,
            ClaimGate.ROUTING: final is not Disposition.ABSTAIN,
            ClaimGate.CONFIRMATION: False,
        }
        claims = tuple(
            authorize_claim(f"{event.event_id}:{gate.value}", gate, gate_results)
            for gate in ClaimGate
        )
        return DiagnosisResult(
            evidence=evidence,
            history=history,
            reader=reader_record,
            reader_gates=gates,
            numeric_disposition=numeric,
            reader_disposition=reader_route.disposition,
            disposition=disposition,
            claims=claims,
            detector_backends=self.detectors.calibration_refs,
        )

    def _build_ledger(
        self, event: TimeSeriesEvent, training_normal: Sequence[TimeSeriesEvent]
    ) -> SourceRoleLedger:
        ledger = SourceRoleLedger()
        normal_ids = [normal.event_id for normal in training_normal]
        if len(normal_ids) != len(set(normal_ids)):
            raise ValueError("training-normal event IDs must be unique")
        source_ids = {normal.event_id for normal in training_normal} | {case.case_id for case in self.cases}
        if event.event_id in source_ids:
            raise ValueError("query event ID collides with a training or historical source")
        for normal in training_normal:
            ledger.assign(normal.event_id, SourceRole.TRAINING_NORMAL, "detector calibration reference")
        for case in self.cases:
            ledger.assign(case.case_id, SourceRole.HISTORICAL_CASE, "analogy-only retrieval source")
        ledger.assign(event.event_id, SourceRole.RETROSPECTIVE_EVALUATION, "current label-blind query")
        ledger.freeze()
        return ledger
