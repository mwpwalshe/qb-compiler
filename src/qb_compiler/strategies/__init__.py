"""Compilation strategies.

Provides a :func:`get_strategy` factory for looking up strategies by name.
"""

from __future__ import annotations

from qb_compiler.strategies.base import (
    CompilationStrategy,
    PassConfig,
    PassManager,
)
from qb_compiler.strategies.budget_aware import BudgetAwareStrategy
from qb_compiler.strategies.cost_optimal import CostOptimalStrategy
from qb_compiler.strategies.depth_optimal import DepthOptimalStrategy
from qb_compiler.strategies.fidelity_optimal import FidelityOptimalStrategy
from qb_compiler.strategies.speed_optimal import SpeedOptimalStrategy

__all__ = [
    "BudgetAwareStrategy",
    "CompilationStrategy",
    "CostOptimalStrategy",
    "DepthOptimalStrategy",
    "FidelityOptimalStrategy",
    "PassConfig",
    "PassManager",
    "SpeedOptimalStrategy",
    "get_strategy",
]

_STRATEGY_REGISTRY: dict[str, type[CompilationStrategy]] = {
    "speed_optimal": SpeedOptimalStrategy,
    "speed": SpeedOptimalStrategy,
    "fidelity_optimal": FidelityOptimalStrategy,
    "fidelity": FidelityOptimalStrategy,
    "depth_optimal": DepthOptimalStrategy,
    "depth": DepthOptimalStrategy,
    "cost_optimal": CostOptimalStrategy,
    "cost": CostOptimalStrategy,
}


def get_strategy(name: str, **kwargs: object) -> CompilationStrategy:
    """Look up and instantiate a compilation strategy by name.

    Parameters
    ----------
    name:
        Strategy identifier.  Accepted values: ``"speed_optimal"`` (or
        ``"speed"``), ``"fidelity_optimal"`` (or ``"fidelity"``),
        ``"depth_optimal"`` (or ``"depth"``), ``"cost_optimal"`` (or
        ``"cost"``), ``"budget_aware"``.
    **kwargs:
        Forwarded to the strategy constructor (only used by strategies
        that accept parameters, e.g. ``BudgetAwareStrategy``).

    Returns
    -------
    CompilationStrategy
        An instance of the requested strategy.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    if name in ("budget_aware", "budget"):
        return BudgetAwareStrategy(**kwargs)  # type: ignore[arg-type]

    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        available = sorted({*_STRATEGY_REGISTRY, "budget_aware", "budget"})
        raise ValueError(f"Unknown strategy {name!r}. Available: {available}")
    return cls()
