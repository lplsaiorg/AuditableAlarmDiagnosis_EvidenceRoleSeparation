"""Immutable protocol objects E(q), H(q), L(q;E,H), and R(q;E,L)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .hashing import sha256


class SourceRole(str, Enum):
    TRAINING_NORMAL = "training_normal"
    HISTORICAL_CASE = "historical_case"
    DEVELOPMENT = "development"
    RETROSPECTIVE_EVALUATION = "retrospective_evaluation"
    NEGATIVE_CALIBRATION = "negative_calibration"
    STABILITY = "stability"
    CONFIRMATION = "confirmation"


class Disposition(str, Enum):
    FAULT = "FAULT"
    BENIGN = "BENIGN"
    ABSTAIN = "ABSTAIN"


class ReaderLabel(str, Enum):
    FAULT = "FAULT_OR_ATTACK"
    BENIGN = "BENIGN_LOOKALIKE"
    INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _frozen_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class EventEvidence:
    """Current-event evidence. Ground-truth fields are deliberately absent."""

    query_id: str
    detector_percentiles: Mapping[str, float]
    detector_summaries: Mapping[str, Mapping[str, float]]
    aliases: Mapping[str, str]
    calibration_refs: Mapping[str, str]
    d_score: float
    created_at: str
    content_hash: str

    def __post_init__(self) -> None:
        payload = {
            "query_id": self.query_id,
            "detector_percentiles": self.detector_percentiles,
            "detector_summaries": self.detector_summaries,
            "aliases": self.aliases,
            "calibration_refs": self.calibration_refs,
            "d_score": self.d_score,
            "created_at": self.created_at,
        }
        if self.content_hash != sha256(payload):
            raise ValueError("evidence content_hash does not match its content")
        object.__setattr__(self, "detector_percentiles", _frozen_mapping(self.detector_percentiles))
        object.__setattr__(self, "detector_summaries", _frozen_mapping(self.detector_summaries))
        object.__setattr__(self, "aliases", _frozen_mapping(self.aliases))
        object.__setattr__(self, "calibration_refs", _frozen_mapping(self.calibration_refs))


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    source_partition: str
    source_interval: str
    outcome_channel: str
    ranking_signature: Mapping[str, Any]
    resolution_text: str
    version: int
    created_at: str
    transform_id: str
    model_id: str
    quality_passed: bool = True
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.outcome_channel not in {"fault", "benign"}:
            raise ValueError("outcome_channel must be 'fault' or 'benign'")
        if self.version < 1:
            raise ValueError("version must be positive")
        payload = {
            "case_id": self.case_id,
            "source_partition": self.source_partition,
            "source_interval": self.source_interval,
            "outcome_channel": self.outcome_channel,
            "ranking_signature": self.ranking_signature,
            "resolution_text": self.resolution_text,
            "version": self.version,
            "created_at": self.created_at,
            "transform_id": self.transform_id,
            "model_id": self.model_id,
            "quality_passed": self.quality_passed,
        }
        expected = sha256(payload)
        if self.content_hash and self.content_hash != expected:
            raise ValueError("case content_hash does not match its content")
        object.__setattr__(self, "ranking_signature", _frozen_mapping(self.ranking_signature))
        object.__setattr__(self, "content_hash", expected)


@dataclass(frozen=True)
class Analogue:
    """Retrieved history; outcome remains explicitly analogy-scoped."""

    case_id: str
    case_version: int
    score: float
    outcome_analogy: str
    resolution_analogy: str
    source_interval: str
    case_hash: str


@dataclass(frozen=True)
class ContrastCard:
    query_id: str
    fault_analogues: tuple[Analogue, ...]
    benign_analogues: tuple[Analogue, ...]
    commonality: tuple[str, ...]
    differences: tuple[str, ...]
    limitations: tuple[str, ...]
    provenance_valid: bool


@dataclass(frozen=True)
class ReaderRecord:
    """Immutable L(q;E,H), including exact bytes and runtime bindings."""

    record_id: str
    query_id: str
    input_bytes: bytes
    raw_output_bytes: bytes
    parsed: Mapping[str, Any]
    schema_valid: bool
    evidence_hash: str
    history_hash: str
    prompt_hash: str
    schema_hash: str
    model_id: str
    decoding: Mapping[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    record_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parsed", _frozen_mapping(self.parsed))
        object.__setattr__(self, "decoding", _frozen_mapping(self.decoding))
        payload = {
            "record_id": self.record_id,
            "query_id": self.query_id,
            "input_sha256": sha256(self.input_bytes),
            "raw_output_sha256": sha256(self.raw_output_bytes),
            "parsed": self.parsed,
            "schema_valid": self.schema_valid,
            "evidence_hash": self.evidence_hash,
            "history_hash": self.history_hash,
            "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash,
            "model_id": self.model_id,
            "decoding": self.decoding,
            "created_at": self.created_at,
        }
        expected = sha256(payload)
        if self.record_hash and self.record_hash != expected:
            raise ValueError("reader record_hash does not match its immutable content")
        object.__setattr__(self, "record_hash", expected)


@dataclass(frozen=True)
class DispositionRecord:
    """Separate R(q;E,L); never mutates or embeds a rewritten reader output."""

    route_id: str
    query_id: str
    reader_record_hash: str
    evidence_hash: str
    disposition: Disposition
    failed_gates: tuple[str, ...]
    policy_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def content_hash(self) -> str:
        return sha256(self)
