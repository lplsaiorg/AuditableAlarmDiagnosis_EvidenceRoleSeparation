"""Detector adapters and dependency-free reference backends.

The reference backends make the protocol executable without model weights.  They
are deliberately labelled as proxies; production callers should wrap the real
ModernTCN, TimesNet, and TranAD predictors with ``CallableDetectorAdapter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import median
from typing import Callable, Iterable, Mapping, Protocol, Sequence


SeriesMap = Mapping[str, Sequence[float]]


class DetectorAdapter(Protocol):
    name: str
    backend_id: str

    def score(self, channels: SeriesMap) -> tuple[float, ...]: ...


def _matrix(channels: SeriesMap) -> list[list[float]]:
    if not channels:
        raise ValueError("at least one signal channel is required")
    ordered = [[float(value) for value in channels[name]] for name in sorted(channels)]
    lengths = {len(values) for values in ordered}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise ValueError("all channels must have the same length of at least two")
    return ordered


def _robust_scale(values: Sequence[float]) -> float:
    center = median(values)
    mad = median(abs(value - center) for value in values)
    if mad > 0:
        return 1.4826 * mad
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return max(sqrt(variance), 1e-12)


@dataclass(frozen=True)
class CallableDetectorAdapter:
    """Adapter for a real exported model or an existing inference function."""

    name: str
    predictor: Callable[[SeriesMap], Sequence[float]]
    backend_id: str

    def score(self, channels: SeriesMap) -> tuple[float, ...]:
        _matrix(channels)
        scores = tuple(float(value) for value in self.predictor(channels))
        if not scores:
            raise ValueError(f"{self.name} returned no anomaly scores")
        return scores


@dataclass(frozen=True)
class PrecomputedDetectorAdapter:
    """Adapter for scores exported by a separately deployed detector."""

    name: str
    scores_by_event: Mapping[str, Sequence[float]]
    backend_id: str = "precomputed-v1"

    def score_event(self, event_id: str) -> tuple[float, ...]:
        if event_id not in self.scores_by_event:
            raise KeyError(f"no precomputed scores for {event_id!r} from {self.name}")
        scores = tuple(float(value) for value in self.scores_by_event[event_id])
        if not scores:
            raise ValueError(f"precomputed scores are empty for {event_id!r}")
        return scores

    def score(self, channels: SeriesMap) -> tuple[float, ...]:
        raise RuntimeError("PrecomputedDetectorAdapter requires score_event(event_id)")


@dataclass(frozen=True)
class ReferenceDetectorAdapter:
    """Transparent statistical proxy used only for runnable examples and tests."""

    name: str
    mode: str
    backend_id: str = "reference-proxy-v1"

    def score(self, channels: SeriesMap) -> tuple[float, ...]:
        matrix = _matrix(channels)
        length = len(matrix[0])
        per_channel: list[list[float]] = []
        for values in matrix:
            scale = _robust_scale(values)
            if self.mode == "multiscale_residual":
                residuals = []
                for index, value in enumerate(values):
                    left = max(0, index - 2)
                    history = values[left:index] or values[:1]
                    residuals.append(abs(value - sum(history) / len(history)) / scale)
            elif self.mode == "periodic_difference":
                lags = (1, 2, 4)
                residuals = [
                    sum(abs(values[index] - values[max(0, index - lag)]) for lag in lags) /
                    (len(lags) * scale)
                    for index in range(length)
                ]
            elif self.mode == "adaptive_reconstruction":
                estimate = values[0]
                residuals = []
                for value in values:
                    residuals.append(abs(value - estimate) / scale)
                    estimate = 0.8 * estimate + 0.2 * value
            else:
                raise ValueError(f"unknown reference detector mode: {self.mode}")
            per_channel.append(residuals)
        return tuple(max(channel[index] for channel in per_channel) for index in range(length))


class DetectorEnsemble:
    """Runs one score stream per detector and preserves backend provenance."""

    def __init__(self, detectors: Iterable[DetectorAdapter]) -> None:
        self.detectors = tuple(detectors)
        names = [detector.name for detector in self.detectors]
        if not names or len(names) != len(set(names)):
            raise ValueError("detector names must be non-empty and unique")

    def score(self, channels: SeriesMap, event_id: str | None = None) -> dict[str, tuple[float, ...]]:
        result: dict[str, tuple[float, ...]] = {}
        for detector in self.detectors:
            if isinstance(detector, PrecomputedDetectorAdapter):
                if event_id is None:
                    raise ValueError("event_id is required for precomputed detector scores")
                result[detector.name] = detector.score_event(event_id)
            else:
                result[detector.name] = detector.score(channels)
        return result

    @property
    def calibration_refs(self) -> dict[str, str]:
        return {detector.name: detector.backend_id for detector in self.detectors}


def build_reference_ensemble() -> DetectorEnsemble:
    """Return runnable proxies under the three paper detector identities."""
    return DetectorEnsemble(
        (
            ReferenceDetectorAdapter("ModernTCN", "multiscale_residual"),
            ReferenceDetectorAdapter("TimesNet", "periodic_difference"),
            ReferenceDetectorAdapter("TranAD", "adaptive_reconstruction"),
        )
    )
