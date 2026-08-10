from __future__ import annotations

import unittest

from auditable_alarm.acsc import CandidatePolicy
from auditable_alarm.cli import _reject_inline_credentials
from auditable_alarm.detectors import (
    CallableDetectorAdapter,
    DetectorEnsemble,
    PrecomputedDetectorAdapter,
    build_reference_ensemble,
)
from auditable_alarm.models import CaseRecord, Disposition
from auditable_alarm.pipeline import DiagnosisPipeline, TimeSeriesEvent
from auditable_alarm.readers import AuditableReader, RuleBasedReaderBackend


def _event(event_id: str, spike: bool = False) -> TimeSeriesEvent:
    values = [1.0, 1.1, 1.0, 1.1, 8.0 if spike else 1.0, 8.5 if spike else 1.1, 1.0, 1.1]
    return TimeSeriesEvent(event_id, {"x": values}, {"subsystem": "a", "variables": ["x"]})


def _cases() -> list[CaseRecord]:
    base = dict(
        source_partition="history",
        source_interval="2025-01-01/2025-01-02",
        ranking_signature={"subsystem": "a", "variables": ["x"]},
        version=1,
        created_at="2026-01-01T00:00:00+00:00",
        transform_id="t1",
        model_id="case-v1",
    )
    return [
        CaseRecord(case_id="f1", outcome_channel="fault", resolution_text="inspect", **base),
        CaseRecord(case_id="b1", outcome_channel="benign", resolution_text="monitor", **base),
    ]


class DetectorAdapterTests(unittest.TestCase):
    def test_real_predictor_can_be_injected(self) -> None:
        detector = CallableDetectorAdapter("ModernTCN", lambda channels: [0.1, 0.9], "checkpoint:abc")
        ensemble = DetectorEnsemble([detector])
        self.assertEqual(ensemble.score({"x": [1, 2]}, "q")["ModernTCN"], (0.1, 0.9))
        self.assertEqual(ensemble.calibration_refs["ModernTCN"], "checkpoint:abc")

    def test_precomputed_service_scores_are_addressed_by_event(self) -> None:
        detector = PrecomputedDetectorAdapter(
            "TranAD", {"q": [0.2, 0.8]}, "service:tranad:model-v3"
        )
        ensemble = DetectorEnsemble([detector])
        self.assertEqual(ensemble.score({}, "q")["TranAD"], (0.2, 0.8))

    def test_inline_api_credentials_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be stored in JSON"):
            _reject_inline_credentials({"type": "openai-compatible", "api_key": ""})
        _reject_inline_credentials({"type": "openai-compatible", "api_key_env": "LLM_API_KEY"})


class FullPipelineTests(unittest.TestCase):
    def test_raw_series_reaches_audited_disposition(self) -> None:
        pipeline = DiagnosisPipeline(
            detectors=build_reference_ensemble(),
            cases=_cases(),
            reader=AuditableReader(RuleBasedReaderBackend()),
            routing_policy=CandidatePolicy("frozen", 0.62, 0.90),
            retrieval_k=1,
        )
        result = pipeline.diagnose(_event("q1", spike=True), [_event("n1"), _event("n2")])
        self.assertEqual(set(result.evidence.detector_percentiles), {"ModernTCN", "TimesNet", "TranAD"})
        self.assertEqual(len(result.history.fault_analogues), 1)
        self.assertEqual(len(result.history.benign_analogues), 1)
        self.assertTrue(result.reader.schema_valid)
        self.assertIs(result.numeric_disposition, Disposition.FAULT)
        self.assertIs(result.reader_disposition, Disposition.FAULT)
        self.assertIs(result.disposition.disposition, Disposition.FAULT)
        self.assertTrue(all(result.reader_gates.checks.values()))

    def test_policy_reader_disagreement_abstains(self) -> None:
        pipeline = DiagnosisPipeline(
            detectors=build_reference_ensemble(),
            cases=_cases(),
            reader=AuditableReader(RuleBasedReaderBackend(benign_threshold=1.0, fault_threshold=1.1)),
            routing_policy=CandidatePolicy("frozen", 0.62, 0.90),
        )
        result = pipeline.diagnose(_event("q2", spike=True), [_event("n1"), _event("n2")])
        self.assertIs(result.disposition.disposition, Disposition.ABSTAIN)
        self.assertIn("P2_READER_POLICY_DISAGREEMENT", result.disposition.failed_gates)


if __name__ == "__main__":
    unittest.main()
