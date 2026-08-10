"""Audit-constrained selective calibration (ACSC), manuscript Equation (7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import Disposition
from .statistics import clopper_pearson_upper


@dataclass(frozen=True)
class CandidatePolicy:
    policy_id: str
    benign_threshold: float
    fault_threshold: float
    model_id: str = "frozen-score"
    min_agreement: int = 1
    complexity: int = 0

    def __post_init__(self) -> None:
        if self.benign_threshold >= self.fault_threshold:
            raise ValueError("benign_threshold must be lower than fault_threshold")
        if self.min_agreement < 1:
            raise ValueError("min_agreement must be positive")

    def route(self, score: float, agreement: int = 1) -> Disposition:
        if agreement < self.min_agreement:
            return Disposition.ABSTAIN
        if score <= self.benign_threshold:
            return Disposition.BENIGN
        if score >= self.fault_threshold:
            return Disposition.FAULT
        return Disposition.ABSTAIN


@dataclass(frozen=True)
class PolicyEvaluation:
    policy: CandidatePolicy
    fault_to_benign: int
    normal_to_fault: int
    n_fault: int
    n_normal: int
    automatic: int
    total: int
    coverage: float
    fault_to_benign_upper: float
    normal_to_fault_upper: float
    admissible: bool


@dataclass(frozen=True)
class SelectionResult:
    selected: PolicyEvaluation | None
    evaluations: tuple[PolicyEvaluation, ...]
    no_go_reason: str | None


def evaluate_policy(
    policy: CandidatePolicy,
    scores: Sequence[float],
    labels: Sequence[str],
    agreements: Sequence[int] | None = None,
    *,
    epsilon_fb: float = 0.05,
    epsilon_nf: float = 0.10,
    bound_confidence: float = 0.975,
) -> PolicyEvaluation:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have equal, non-zero length")
    if agreements is None:
        agreements = [1] * len(scores)
    if len(agreements) != len(scores):
        raise ValueError("agreements must match scores")
    normalized = [str(label).upper() for label in labels]
    if any(label not in {"FAULT", "NORMAL", "BENIGN"} for label in normalized):
        raise ValueError("development labels must be FAULT, NORMAL, or BENIGN")
    routes = [policy.route(float(score), int(agreement)) for score, agreement in zip(scores, agreements)]
    fault_indices = [i for i, label in enumerate(normalized) if label == "FAULT"]
    normal_indices = [i for i, label in enumerate(normalized) if label in {"NORMAL", "BENIGN"}]
    if not fault_indices or not normal_indices:
        raise ValueError("ACSC requires both fault and normal development units")
    fb = sum(routes[i] is Disposition.BENIGN for i in fault_indices)
    nf = sum(routes[i] is Disposition.FAULT for i in normal_indices)
    automatic = sum(route is not Disposition.ABSTAIN for route in routes)
    fb_upper = clopper_pearson_upper(fb, len(fault_indices), bound_confidence)
    nf_upper = clopper_pearson_upper(nf, len(normal_indices), bound_confidence)
    return PolicyEvaluation(
        policy=policy,
        fault_to_benign=fb,
        normal_to_fault=nf,
        n_fault=len(fault_indices),
        n_normal=len(normal_indices),
        automatic=automatic,
        total=len(scores),
        coverage=automatic / len(scores),
        fault_to_benign_upper=fb_upper,
        normal_to_fault_upper=nf_upper,
        admissible=fb_upper <= epsilon_fb and nf_upper <= epsilon_nf,
    )


def select_policy(
    candidates: Iterable[CandidatePolicy],
    scores: Sequence[float],
    labels: Sequence[str],
    agreements: Sequence[int] | None = None,
    **risk_options: float,
) -> SelectionResult:
    """Maximize coverage subject to both exact risk bounds; return no-go if empty."""
    evaluations = tuple(
        evaluate_policy(candidate, scores, labels, agreements, **risk_options)
        for candidate in candidates
    )
    if not evaluations:
        raise ValueError("candidate family cannot be empty")
    feasible = [item for item in evaluations if item.admissible]
    if not feasible:
        return SelectionResult(None, evaluations, "no candidate satisfies both prespecified risk bounds")
    # Frozen tie-break: coverage, lower dangerous-error bound, simpler policy, stable ID.
    selected = sorted(
        feasible,
        key=lambda item: (-item.coverage, item.fault_to_benign_upper, item.policy.complexity, item.policy.policy_id),
    )[0]
    return SelectionResult(selected, evaluations, None)
