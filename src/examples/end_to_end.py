"""Synthetic protocol demonstration; it does not reconstruct benchmark results."""

from __future__ import annotations

import json

from auditable_alarm.audit import route_reader_record
from auditable_alarm.evidence import build_event_evidence
from auditable_alarm.hashing import canonical_bytes, sha256
from auditable_alarm.models import ReaderRecord


def main() -> None:
    evidence = build_event_evidence(
        "query-001",
        event_scores={"ModernTCN": [0.4, 0.9], "TimesNet": [0.3, 0.8], "TranAD": [0.2, 0.7]},
        training_normal_scores={
            "ModernTCN": [0.1, 0.2, 0.3, 0.4],
            "TimesNet": [0.1, 0.2, 0.3, 0.4],
            "TranAD": [0.1, 0.2, 0.3, 0.4],
        },
        calibration_refs={"ModernTCN": "train:m", "TimesNet": "train:t", "TranAD": "train:a"},
    )
    parsed = {
        "primary_label": "FAULT_OR_ATTACK",
        "confidence": 0.91,
        "supporting_evidence": ["e1", "e2"],
        "contradictions": [],
        "action": "REVIEW",
    }
    input_bytes = canonical_bytes({"evidence_hash": evidence.content_hash, "history": "contrast-card-001"})
    output_bytes = canonical_bytes(parsed)
    reader = ReaderRecord(
        record_id="reader-001",
        query_id=evidence.query_id,
        input_bytes=input_bytes,
        raw_output_bytes=output_bytes,
        parsed=parsed,
        schema_valid=True,
        evidence_hash=evidence.content_hash,
        history_hash=sha256("contrast-card-001"),
        prompt_hash=sha256("prompt-v1"),
        schema_hash=sha256("schema-v1"),
        model_id="local-reader",
        decoding={"temperature": 0},
    )
    route, gates = route_reader_record(reader, evidence)
    print(json.dumps({"d_score": evidence.d_score, "reader_hash": reader.record_hash,
                      "disposition": route.disposition.value, "gates": dict(gates.checks)}, indent=2))


if __name__ == "__main__":
    main()
