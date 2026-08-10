"""Exact binomial intervals with no mandatory third-party dependency."""

from __future__ import annotations

import math


def _binomial_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logs = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        + j * math.log(p) + (n - j) * math.log1p(-p)
        for j in range(k + 1)
    ]
    pivot = max(logs)
    return math.exp(pivot) * math.fsum(math.exp(item - pivot) for item in logs)


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.975) -> float:
    """One-sided exact upper confidence bound for a binomial proportion."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("require n > 0 and 0 <= k <= n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if k == n:
        return 1.0
    target = 1.0 - confidence
    low, high = k / n, 1.0
    for _ in range(70):
        mid = (low + high) / 2.0
        if _binomial_cdf(k, n, mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def clopper_pearson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided equal-tail exact Clopper-Pearson interval."""
    alpha = 1.0 - confidence
    upper = clopper_pearson_upper(k, n, 1.0 - alpha / 2.0)
    if k == 0:
        return 0.0, upper
    # Symmetry: lower(k,n) = 1 - upper(n-k,n).
    lower = 1.0 - clopper_pearson_upper(n - k, n, 1.0 - alpha / 2.0)
    return lower, upper
