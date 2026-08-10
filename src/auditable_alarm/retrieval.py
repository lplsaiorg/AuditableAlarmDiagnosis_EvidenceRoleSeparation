"""Version-aware, label-blind dual-channel QAKB retrieval."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from .models import Analogue, CaseRecord, ContrastCard


FORBIDDEN_QUERY_KEYS = {"label", "truth", "ground_truth", "outcome", "y_q", "attack_label"}


def _assert_label_blind(value: Any, path: str = "query") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_QUERY_KEYS:
                raise ValueError(f"query truth is forbidden in retrieval input: {path}.{key}")
            _assert_label_blind(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_label_blind(item, f"{path}[{index}]")


def structured_similarity(query: Mapping[str, Any], case: Mapping[str, Any]) -> float:
    """Transparent mixed-type similarity over common ranking-visible fields."""
    _assert_label_blind(query)
    keys = sorted(set(query) & set(case))
    if not keys:
        return 0.0
    scores: list[float] = []
    for key in keys:
        left, right = query[key], case[key]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            scale = max(abs(float(left)), abs(float(right)), 1.0)
            scores.append(max(0.0, 1.0 - abs(float(left) - float(right)) / scale))
        elif isinstance(left, (list, tuple, set)) and isinstance(right, (list, tuple, set)):
            a, b = set(left), set(right)
            scores.append(len(a & b) / len(a | b) if a | b else 1.0)
        else:
            scores.append(float(left == right))
    return sum(scores) / len(scores)


def _latest_versions(cases: Iterable[CaseRecord], as_of: str | None) -> list[CaseRecord]:
    eligible: dict[str, CaseRecord] = {}
    cutoff = datetime.fromisoformat(as_of.replace("Z", "+00:00")) if as_of else None
    for case in cases:
        created = datetime.fromisoformat(case.created_at.replace("Z", "+00:00"))
        if not case.quality_passed or (cutoff is not None and created > cutoff):
            continue
        previous = eligible.get(case.case_id)
        if previous is None or case.version > previous.version:
            eligible[case.case_id] = case
    return list(eligible.values())


def retrieve_dual_channel(
    query_id: str,
    query_signature: Mapping[str, Any],
    cases: Iterable[CaseRecord],
    *,
    k: int = 5,
    as_of: str | None = None,
) -> ContrastCard:
    if k < 1:
        raise ValueError("k must be positive")
    _assert_label_blind(query_signature)
    ranked: dict[str, list[tuple[float, CaseRecord]]] = {"fault": [], "benign": []}
    for case in _latest_versions(cases, as_of):
        score = structured_similarity(query_signature, case.ranking_signature)
        ranked[case.outcome_channel].append((score, case))
    analogues: dict[str, tuple[Analogue, ...]] = {}
    for channel in ("fault", "benign"):
        chosen = sorted(ranked[channel], key=lambda item: (-item[0], item[1].case_id))[:k]
        analogues[channel] = tuple(
            Analogue(
                case_id=case.case_id,
                case_version=case.version,
                score=score,
                outcome_analogy=case.outcome_channel,
                resolution_analogy=case.resolution_text,
                source_interval=case.source_interval,
                case_hash=case.content_hash,
            )
            for score, case in chosen
        )
    selected_cases = [
        sorted(ranked[channel], key=lambda item: (-item[0], item[1].case_id))[0][1]
        for channel in ("fault", "benign") if ranked[channel]
    ]
    commonality = tuple(
        key for key in sorted(query_signature)
        if selected_cases and all(case.ranking_signature.get(key) == query_signature[key] for case in selected_cases)
    )
    differences = tuple(
        key for key in sorted(query_signature)
        if selected_cases and any(case.ranking_signature.get(key) != query_signature[key] for case in selected_cases)
    )
    limitations = ["Historical outcomes are analogies, not current-event evidence."]
    if not analogues["fault"] or not analogues["benign"]:
        limitations.append("One or more outcome channels are unavailable.")
    return ContrastCard(
        query_id=query_id,
        fault_analogues=analogues["fault"],
        benign_analogues=analogues["benign"],
        commonality=commonality,
        differences=differences,
        limitations=tuple(limitations),
        provenance_valid=bool(analogues["fault"] and analogues["benign"])
        and all(item.case_hash for channel in analogues.values() for item in channel),
    )


def reciprocal_rank_fusion(
    branch_rankings: Mapping[str, Sequence[str]],
    weights: Mapping[str, float] | None = None,
    constant: int = 60,
) -> list[tuple[str, float]]:
    if constant <= 0:
        raise ValueError("constant must be positive")
    weights = weights or {name: 1.0 for name in branch_rankings}
    if set(weights) != set(branch_rankings):
        raise ValueError("weights must cover exactly the supplied branches")
    totals: defaultdict[str, float] = defaultdict(float)
    for branch, ranking in branch_rankings.items():
        weight = float(weights[branch])
        if not isfinite(weight) or weight < 0:
            raise ValueError("RRF weights must be finite and non-negative")
        for rank, case_id in enumerate(ranking, start=1):
            totals[case_id] += weight / (constant + rank)
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))
