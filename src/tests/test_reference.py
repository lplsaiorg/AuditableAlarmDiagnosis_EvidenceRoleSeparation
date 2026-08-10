from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from auditable_alarm.acsc import CandidatePolicy, select_policy
from auditable_alarm.audit import SourceRoleLedger, route_reader_record
from auditable_alarm.claims import ClaimGate, authorize_claim
from auditable_alarm.evidence import build_event_evidence
from auditable_alarm.hashing import canonical_bytes, sha256
from auditable_alarm.models import CaseRecord, Disposition, ReaderRecord, SourceRole
from auditable_alarm.retrieval import reciprocal_rank_fusion, retrieve_dual_channel
from auditable_alarm.statistics import clopper_pearson_interval, clopper_pearson_upper


class StatisticsTests(unittest.TestCase):
    def test_manuscript_confirmation_bounds(self) -> None:
        self.assertAlmostEqual(clopper_pearson_upper(9, 400), 0.0423, places=4)
        self.assertAlmostEqual(clopper_pearson_upper(4, 100), 0.0993, places=4)
        self.assertGreater(clopper_pearson_upper(5, 100), 0.10)

    def test_reported_zero_of_four_interval(self) -> None:
        low, high = clopper_pearson_interval(0, 4)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.6024, places=4)


class EvidenceAndRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = build_event_evidence(
            "q1",
            {"a": [0.9], "b": [0.8], "c": [0.7]},
            {"a": [0.1, 0.2], "b": [0.1, 0.2], "c": [0.1, 0.2]},
            {"a": "train-a", "b": "train-b", "c": "train-c"},
        )

    def _reader(self, parsed: dict) -> ReaderRecord:
        return ReaderRecord(
            "l1", "q1", b"input", canonical_bytes(parsed), parsed, True,
            self.evidence.content_hash, sha256("history"), sha256("prompt"), sha256("schema"),
            "reader", {"temperature": 0}, created_at="2026-01-01T00:00:00+00:00",
        )

    def test_six_gate_route_and_non_overwrite(self) -> None:
        parsed = {"primary_label": "FAULT_OR_ATTACK", "confidence": 0.9,
                  "supporting_evidence": ["e1"], "contradictions": [], "action": "REVIEW"}
        reader = self._reader(parsed)
        original_hash = reader.record_hash
        route, gates = route_reader_record(reader, self.evidence)
        self.assertTrue(gates.passed)
        self.assertIs(route.disposition, Disposition.FAULT)
        self.assertEqual(reader.record_hash, original_hash)
        with self.assertRaises(TypeError):
            reader.parsed["confidence"] = 0.1

    def test_gate_failure_forces_abstention(self) -> None:
        parsed = {"primary_label": "BENIGN_LOOKALIKE", "confidence": 0.79,
                  "supporting_evidence": ["historical-case-1"], "contradictions": [],
                  "action": "SHUTDOWN"}
        route, gates = route_reader_record(self._reader(parsed), self.evidence)
        self.assertIs(route.disposition, Disposition.ABSTAIN)
        self.assertIn("G3_CONFIDENCE", gates.failed_gates)
        self.assertIn("G4_CURRENT_SUPPORT", gates.failed_gates)
        self.assertIn("G6_LOW_RISK_ACTION", gates.failed_gates)


class RetrievalAndLedgerTests(unittest.TestCase):
    def test_label_blind_dual_channel_latest_version(self) -> None:
        base = dict(source_partition="history", source_interval="1:20",
                    ranking_signature={"subsystem": "A", "variables": ["x", "y"]},
                    resolution_text="inspect", created_at="2026-01-01T00:00:00+00:00",
                    transform_id="t1", model_id="m1")
        cases = [
            CaseRecord(case_id="f1", outcome_channel="fault", version=1, **base),
            CaseRecord(case_id="f1", outcome_channel="fault", version=2, **{**base, "resolution_text": "inspect v2"}),
            CaseRecord(case_id="b1", outcome_channel="benign", version=1, **base),
        ]
        card = retrieve_dual_channel("q", {"subsystem": "A", "variables": ["x"]}, cases, k=1)
        self.assertEqual(card.fault_analogues[0].case_version, 2)
        self.assertEqual(len(card.benign_analogues), 1)
        with self.assertRaises(ValueError):
            retrieve_dual_channel("q", {"ground_truth": "FAULT"}, cases)

    def test_rrf_formula(self) -> None:
        fused = reciprocal_rank_fusion({"a": ["x", "y"], "b": ["y", "x"]})
        self.assertEqual([item[0] for item in fused], ["x", "y"])
        self.assertAlmostEqual(fused[0][1], 1 / 61 + 1 / 62)

    def test_confirmation_is_one_shot(self) -> None:
        ledger = SourceRoleLedger()
        ledger.assign("sealed", SourceRole.CONFIRMATION, "prespecified")
        ledger.freeze()
        ledger.access_confirmation_once("sealed")
        with self.assertRaises(RuntimeError):
            ledger.access_confirmation_once("sealed")


class ACSCTests(unittest.TestCase):
    def test_no_go_and_selection(self) -> None:
        # Larger sample sizes make zero-error candidates pass the manuscript limits.
        scores = [0.05] * 100 + [0.95] * 400
        labels = ["NORMAL"] * 100 + ["FAULT"] * 400
        safe = CandidatePolicy("safe", 0.1, 0.9)
        result = select_policy([safe], scores, labels)
        self.assertEqual(result.selected.policy.policy_id, "safe")
        self.assertEqual(result.selected.coverage, 1.0)
        unsafe = CandidatePolicy("unsafe", 0.96, 0.97)
        no_go = select_policy([unsafe], scores, labels)
        self.assertIsNone(no_go.selected)
        self.assertIsNotNone(no_go.no_go_reason)

    def test_failed_gate_blocks_only_stronger_claim(self) -> None:
        decision = authorize_claim("routing-safe", ClaimGate.ROUTING, {ClaimGate.INTEGRITY: True})
        self.assertFalse(decision.released)
        self.assertIn("descriptive", decision.reason)


if __name__ == "__main__":
    unittest.main()
