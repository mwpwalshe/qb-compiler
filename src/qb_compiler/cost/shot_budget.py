"""Shot-budget estimation: how many shots before a quantity is resolved.

Signals only: these are textbook sample-size formulas exposed as a convenience,
so "how many shots do I need" gets an answer before any QPU time is spent.
"""

from __future__ import annotations

import math

_Z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def _z(confidence: float) -> float:
    if confidence not in _Z:
        raise ValueError(f"confidence must be one of {sorted(_Z)}, got {confidence}")
    return _Z[confidence]


def shots_for_expectation(
    epsilon: float,
    *,
    observable_l1: float = 1.0,
    confidence: float = 0.95,
) -> int:
    """Shots to estimate an observable expectation to within +-epsilon.

    Uses the worst-case variance bound for an observable with coefficient
    1-norm ``observable_l1`` (per-shot values bounded by it), normal
    approximation. Real circuits often need fewer; this is the budget-safe
    upper estimate.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if observable_l1 <= 0:
        raise ValueError("observable_l1 must be positive")
    z = _z(confidence)
    return math.ceil((z * observable_l1 / epsilon) ** 2)


def shots_for_rate(
    rate: float,
    *,
    rel_width: float = 0.2,
    confidence: float = 0.95,
) -> int:
    """Shots to resolve a small event rate (e.g. an error rate) to a relative
    confidence half-width ``rel_width`` (0.2 means +-20 percent of the rate).

    Normal approximation to the binomial; for very small rates the answer is
    dominated by 1/rate, which is the honest cost of measuring rare events.
    """
    if not 0 < rate < 1:
        raise ValueError("rate must be in (0, 1)")
    if rel_width <= 0:
        raise ValueError("rel_width must be positive")
    z = _z(confidence)
    return math.ceil(z * z * (1.0 - rate) / (rate * rel_width * rel_width))
