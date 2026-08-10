"""Small JSON CLI for auditable ACSC selection and exact risk checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .acsc import CandidatePolicy, select_policy
from .detectors import DetectorEnsemble, PrecomputedDetectorAdapter, build_reference_ensemble
from .hashing import canonical_bytes
from .models import CaseRecord
from .pipeline import DiagnosisPipeline, TimeSeriesEvent
from .readers import AuditableReader, OpenAICompatibleReaderBackend, RuleBasedReaderBackend
from .statistics import clopper_pearson_interval, clopper_pearson_upper


def _read_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _risk(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "errors": args.errors,
        "total": args.total,
        "one_sided_upper": clopper_pearson_upper(args.errors, args.total, args.confidence),
    }
    if args.two_sided:
        result["two_sided_interval"] = clopper_pearson_interval(
            args.errors, args.total, args.two_sided_confidence
        )
    return result


def _select(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(args.input)
    candidates = [CandidatePolicy(**item) for item in payload["candidates"]]
    result = select_policy(
        candidates,
        payload["scores"],
        payload["labels"],
        payload.get("agreements"),
        epsilon_fb=float(payload.get("epsilon_fb", 0.05)),
        epsilon_nf=float(payload.get("epsilon_nf", 0.10)),
        bound_confidence=float(payload.get("bound_confidence", 0.975)),
    )
    return {
        "status": "PASS" if result.selected else "NO_GO",
        "selected": asdict(result.selected) if result.selected else None,
        "no_go_reason": result.no_go_reason,
        "evaluations": [asdict(item) for item in result.evaluations],
    }


def _event(payload: dict[str, Any]) -> TimeSeriesEvent:
    return TimeSeriesEvent(
        event_id=str(payload["event_id"]),
        channels=payload["channels"],
        signature=payload.get("signature", {}),
    )


def _detectors(payload: dict[str, Any]) -> DetectorEnsemble:
    config = payload.get("detectors")
    if not isinstance(config, dict):
        raise ValueError("detectors configuration is required; choose reference-proxy or precomputed")
    detector_type = config.get("type")
    if detector_type == "reference-proxy":
        return build_reference_ensemble()
    if detector_type == "precomputed":
        return DetectorEnsemble(
            PrecomputedDetectorAdapter(
                name=str(item["name"]),
                scores_by_event=item["scores_by_event"],
                backend_id=str(item["backend_id"]),
            )
            for item in config["models"]
        )
    raise ValueError("detectors.type must be 'reference-proxy' or 'precomputed'")


def _reject_inline_credentials(config: dict[str, Any]) -> None:
    forbidden = {"api_key", "apikey", "token", "access_token", "secret", "password"}
    supplied = sorted(key for key in config if str(key).lower() in forbidden)
    if supplied:
        raise ValueError(
            "reader credentials must not be stored in JSON; use api_key_env instead "
            f"(forbidden fields: {', '.join(supplied)})"
        )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(args.input)
    reader_config = payload.get("reader", {"type": "rule"})
    _reject_inline_credentials(reader_config)
    reader_type = reader_config.get("type", "rule")
    if reader_type == "rule":
        backend = RuleBasedReaderBackend(
            benign_threshold=float(reader_config.get("benign_threshold", 0.62)),
            fault_threshold=float(reader_config.get("fault_threshold", 0.90)),
        )
    elif reader_type == "openai-compatible":
        backend = OpenAICompatibleReaderBackend(
            model=str(reader_config["model"]),
            base_url=str(reader_config["base_url"]),
            api_key_env=str(reader_config.get("api_key_env", "OPENAI_API_KEY")),
            temperature=float(reader_config.get("temperature", 0.0)),
        )
    else:
        raise ValueError("reader.type must be 'rule' or 'openai-compatible'")
    cases = [CaseRecord(**item) for item in payload["cases"]]
    policy = CandidatePolicy(**payload["routing_policy"])
    pipeline = DiagnosisPipeline(
        detectors=_detectors(payload),
        cases=cases,
        reader=AuditableReader(backend),
        routing_policy=policy,
        retrieval_k=int(payload.get("retrieval_k", 5)),
    )
    result = pipeline.diagnose(
        _event(payload["query"]),
        [_event(item) for item in payload["training_normal"]],
        agreement=int(payload.get("agreement", 1)),
        as_of=payload.get("as_of"),
    )
    return {
        "query_id": result.evidence.query_id,
        "d_score": result.evidence.d_score,
        "detector_percentiles": dict(result.evidence.detector_percentiles),
        "detector_backends": dict(result.detector_backends),
        "evidence_hash": result.evidence.content_hash,
        "history_hash": result.reader.history_hash,
        "fault_analogues": [item.case_id for item in result.history.fault_analogues],
        "benign_analogues": [item.case_id for item in result.history.benign_analogues],
        "reader_model": result.reader.model_id,
        "reader_output": dict(result.reader.parsed),
        "reader_hash": result.reader.record_hash,
        "reader_gates": dict(result.reader_gates.checks),
        "numeric_disposition": result.numeric_disposition.value,
        "reader_disposition": result.reader_disposition.value,
        "final_disposition": result.disposition.disposition.value,
        "failed_gates": list(result.disposition.failed_gates),
        "route_hash": result.disposition.content_hash,
        "claims": [json.loads(canonical_bytes(item)) for item in result.claims],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auditable-alarm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    risk = subparsers.add_parser("risk-bound", help="compute an exact binomial bound")
    risk.add_argument("errors", type=int)
    risk.add_argument("total", type=int)
    risk.add_argument("--confidence", type=float, default=0.975)
    risk.add_argument("--two-sided", action="store_true")
    risk.add_argument("--two-sided-confidence", type=float, default=0.95)
    risk.set_defaults(handler=_risk)
    select = subparsers.add_parser("select", help="select ACSC policy from frozen JSON candidates")
    select.add_argument("input")
    select.set_defaults(handler=_select)
    run = subparsers.add_parser("run", help="run raw time series through the full audited pipeline")
    run.add_argument("input")
    run.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(args.handler(args), ensure_ascii=False, indent=2))
    return 0
