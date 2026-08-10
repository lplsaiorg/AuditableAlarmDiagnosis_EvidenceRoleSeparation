"""I4 claim admissibility: descriptive evidence survives, stronger claims are gated."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ClaimGate(str, Enum):
    INTEGRITY = "integrity"
    DETECTOR = "detector"
    RETRIEVAL = "retrieval"
    ROUTING = "routing"
    CONFIRMATION = "confirmation"


@dataclass(frozen=True)
class ClaimDecision:
    claim_id: str
    required_gate: ClaimGate
    released: bool
    reason: str


def authorize_claim(
    claim_id: str,
    required_gate: ClaimGate,
    gate_results: Mapping[ClaimGate | str, bool],
) -> ClaimDecision:
    """Release only a claim whose named gate passed; never discard description."""
    normalized = {
        key if isinstance(key, ClaimGate) else ClaimGate(str(key)): bool(value)
        for key, value in gate_results.items()
    }
    passed = normalized.get(required_gate, False)
    reason = (
        f"{required_gate.value} gate passed"
        if passed else
        f"{required_gate.value} gate failed or was not evaluated; retain descriptive results only"
    )
    return ClaimDecision(claim_id, required_gate, passed, reason)
