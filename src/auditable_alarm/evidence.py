"""Training-normal percentile calibration and label-free D-score evidence."""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
from math import ceil
from statistics import median
from typing import Mapping, Sequence

from .hashing import sha256
from .models import EventEvidence


def empirical_percentile(reference: Sequence[float], value: float) -> float:
    """Right-continuous empirical CDF F_hat(value)."""
    if not reference:
        raise ValueError("training-normal reference cannot be empty")
    ordered = sorted(float(item) for item in reference)
    return bisect_right(ordered, float(value)) / len(ordered)


def build_event_evidence(
    query_id: str,
    event_scores: Mapping[str, Sequence[float]],
    training_normal_scores: Mapping[str, Sequence[float]],
    calibration_refs: Mapping[str, str],
) -> EventEvidence:
    """Build E(q) without accepting a query label or outcome argument."""
    if set(event_scores) != set(training_normal_scores) or set(event_scores) != set(calibration_refs):
        raise ValueError("detector names must match across event, reference, and provenance inputs")
    if not event_scores:
        raise ValueError("at least one detector is required")
    summaries: dict[str, dict[str, float]] = {}
    percentiles: dict[str, float] = {}
    aliases: dict[str, str] = {}
    for index, detector in enumerate(sorted(event_scores), start=1):
        scores = [float(value) for value in event_scores[detector]]
        if not scores:
            raise ValueError(f"event scores are empty for {detector}")
        reference = training_normal_scores[detector]
        maximum = max(scores)
        if not reference:
            raise ValueError(f"training-normal reference is empty for {detector}")
        p99 = sorted(float(x) for x in reference)[ceil(0.99 * len(reference)) - 1]
        summaries[detector] = {
            "maximum": maximum,
            "exceedance_fraction_p99": sum(value > p99 for value in scores) / len(scores),
        }
        percentiles[detector] = empirical_percentile(reference, maximum)
        aliases[f"e{index}"] = detector
    d_score = float(median(percentiles.values()))
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "query_id": query_id,
        "detector_percentiles": percentiles,
        "detector_summaries": summaries,
        "aliases": aliases,
        "calibration_refs": {name: calibration_refs[name] for name in sorted(calibration_refs)},
        "d_score": d_score,
        "created_at": created_at,
    }
    return EventEvidence(**payload, content_hash=sha256(payload))
