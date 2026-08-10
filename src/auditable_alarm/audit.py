"""Source ledger, reader invariants, six gates, and non-overwriting routing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping
from uuid import uuid4

from .models import Disposition, DispositionRecord, EventEvidence, ReaderLabel, ReaderRecord, SourceRole


@dataclass(frozen=True)
class LedgerEntry:
    source_id: str
    role: SourceRole
    assigned_at: str
    reason: str


class SourceRoleLedger:
    """Append-only role ledger with explicit reassignment before use."""

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._frozen = False
        self._confirmation_accesses: dict[str, int] = {}

    def assign(self, source_id: str, role: SourceRole, reason: str) -> LedgerEntry:
        if self._frozen:
            raise RuntimeError("ledger is frozen")
        entry = LedgerEntry(source_id, role, datetime.now(timezone.utc).isoformat(), reason)
        self._entries.append(entry)
        return entry

    def freeze(self) -> None:
        if not self._entries:
            raise RuntimeError("cannot freeze an empty ledger")
        self._frozen = True

    def role_of(self, source_id: str) -> SourceRole:
        entries = [entry for entry in self._entries if entry.source_id == source_id]
        if not entries:
            raise KeyError(source_id)
        return entries[-1].role

    def access_confirmation_once(self, source_id: str) -> None:
        if not self._frozen or self.role_of(source_id) is not SourceRole.CONFIRMATION:
            raise RuntimeError("source is not a frozen confirmation role")
        count = self._confirmation_accesses.get(source_id, 0) + 1
        self._confirmation_accesses[source_id] = count
        if count > 1:
            raise RuntimeError("sealed confirmation role has already been accessed")

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failed_gates: tuple[str, ...]
    checks: Mapping[str, bool]


def validate_reader_schema(parsed: Mapping[str, object]) -> bool:
    """Dependency-free strict validation of the frozen reader output contract."""
    required = {"primary_label", "confidence", "supporting_evidence", "contradictions", "action"}
    allowed = required | {"explanation", "missing_information"}
    confidence = parsed.get("confidence")
    return (
        required <= set(parsed) <= allowed
        and parsed.get("primary_label") in {item.value for item in ReaderLabel}
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0.0 <= float(confidence) <= 1.0
        and isinstance(parsed.get("supporting_evidence"), (list, tuple))
        and all(isinstance(item, str) for item in parsed.get("supporting_evidence", ()))
        and isinstance(parsed.get("contradictions"), (list, tuple))
        and all(isinstance(item, str) for item in parsed.get("contradictions", ()))
        and isinstance(parsed.get("action"), str)
    )


def check_six_gates(
    reader: ReaderRecord,
    evidence: EventEvidence,
    *,
    confidence_threshold: float = 0.80,
    low_risk_actions: Iterable[str] = ("REVIEW", "LOG", "MONITOR", "NOTIFY"),
) -> GateResult:
    parsed = reader.parsed
    support = tuple(str(item) for item in parsed.get("supporting_evidence", ()))
    contradictions = tuple(str(item) for item in parsed.get("contradictions", ()))
    valid_aliases = set(evidence.aliases)
    label = parsed.get("primary_label")
    checks = {
        "G1_SCHEMA_VALID": bool(reader.schema_valid) and validate_reader_schema(parsed),
        "G2_NOT_INSUFFICIENT": label in {ReaderLabel.FAULT.value, ReaderLabel.BENIGN.value},
        "G3_CONFIDENCE": isinstance(parsed.get("confidence"), (int, float))
        and float(parsed["confidence"]) >= confidence_threshold,
        "G4_CURRENT_SUPPORT": bool(support) and set(support) <= valid_aliases,
        "G5_DISJOINT_SETS": set(support).isdisjoint(contradictions),
        "G6_LOW_RISK_ACTION": str(parsed.get("action", "")).upper()
        in {item.upper() for item in low_risk_actions},
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return GateResult(not failed, failed, checks)


def route_reader_record(
    reader: ReaderRecord,
    evidence: EventEvidence,
    *,
    policy_id: str = "reader-six-gate-v1",
    **gate_options: object,
) -> tuple[DispositionRecord, GateResult]:
    """Append R(q;E,L) from saved L without invoking or rewriting a generator."""
    if reader.query_id != evidence.query_id or reader.evidence_hash != evidence.content_hash:
        raise ValueError("reader/evidence binding mismatch")
    gate_result = check_six_gates(reader, evidence, **gate_options)
    label = reader.parsed.get("primary_label")
    if not gate_result.passed:
        disposition = Disposition.ABSTAIN
    elif label == ReaderLabel.FAULT.value:
        disposition = Disposition.FAULT
    elif label == ReaderLabel.BENIGN.value:
        disposition = Disposition.BENIGN
    else:
        disposition = Disposition.ABSTAIN
    record = DispositionRecord(
        route_id=str(uuid4()),
        query_id=reader.query_id,
        reader_record_hash=reader.record_hash,
        evidence_hash=evidence.content_hash,
        disposition=disposition,
        failed_gates=gate_result.failed_gates,
        policy_id=policy_id,
    )
    return record, gate_result
