"""Reference implementation for auditable industrial alarm disposition."""

from .acsc import CandidatePolicy, SelectionResult, evaluate_policy, select_policy
from .audit import GateResult, route_reader_record
from .claims import ClaimDecision, ClaimGate, authorize_claim
from .evidence import build_event_evidence, empirical_percentile
from .detectors import (
    CallableDetectorAdapter,
    DetectorEnsemble,
    PrecomputedDetectorAdapter,
    ReferenceDetectorAdapter,
    build_reference_ensemble,
)
from .models import (
    CaseRecord,
    Disposition,
    DispositionRecord,
    EventEvidence,
    ReaderRecord,
    SourceRole,
)
from .statistics import clopper_pearson_interval, clopper_pearson_upper
from .pipeline import DiagnosisPipeline, DiagnosisResult, TimeSeriesEvent
from .readers import (
    AuditableReader,
    ChatModelReaderBackend,
    OpenAICompatibleReaderBackend,
    RuleBasedReaderBackend,
)

__all__ = [
    "CandidatePolicy",
    "CallableDetectorAdapter",
    "CaseRecord",
    "ClaimDecision",
    "ClaimGate",
    "Disposition",
    "DispositionRecord",
    "DetectorEnsemble",
    "DiagnosisPipeline",
    "DiagnosisResult",
    "EventEvidence",
    "GateResult",
    "ReaderRecord",
    "PrecomputedDetectorAdapter",
    "ReferenceDetectorAdapter",
    "RuleBasedReaderBackend",
    "OpenAICompatibleReaderBackend",
    "ChatModelReaderBackend",
    "AuditableReader",
    "SelectionResult",
    "SourceRole",
    "TimeSeriesEvent",
    "build_event_evidence",
    "build_reference_ensemble",
    "authorize_claim",
    "clopper_pearson_interval",
    "clopper_pearson_upper",
    "empirical_percentile",
    "evaluate_policy",
    "route_reader_record",
    "select_policy",
]
