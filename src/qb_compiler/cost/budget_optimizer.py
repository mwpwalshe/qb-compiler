"""Budget-constrained compilation optimizer.

:class:`BudgetOptimizer` finds the best compilation settings (shots,
strategy, qubit selection) that fit within a given USD budget while
maximising expected fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

from qb_compiler.config import BACKEND_CONFIGS
from qb_compiler.cost.pricing import VENDOR_PRICING, cost_per_shot


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Result of a budget optimization.

    Parameters
    ----------
    backend:
        Recommended backend.
    recommended_shots:
        Number of shots that fit within the budget while maximising
        statistical power.
    estimated_fidelity:
        Rough fidelity estimate based on backend median error rates.
    estimated_cost_usd:
        Estimated total cost in USD.
    strategy:
        Recommended compilation strategy name.
    notes:
        Human-readable notes about trade-offs made.
    """

    backend: str
    recommended_shots: int
    estimated_fidelity: float
    estimated_cost_usd: float
    strategy: str
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(backend={self.backend!r}, "
            f"shots={self.recommended_shots:,}, "
            f"fidelity~{self.estimated_fidelity:.3f}, "
            f"cost=${self.estimated_cost_usd:.4f})"
        )


class BudgetOptimizer:
    """Finds optimal compilation settings within a budget.

    Given a target backend and budget, determines the best number of
    shots and compilation strategy.  Can also recommend alternative
    backends if the target is too expensive.

    Parameters
    ----------
    min_shots:
        Minimum acceptable number of shots.  If the budget cannot
        afford this many shots, the optimizer raises
        :class:`~qb_compiler.exceptions.BudgetExceededError`.
    """

    def __init__(self, *, min_shots: int = 100) -> None:
        if min_shots < 1:
            raise ValueError(f"min_shots must be >= 1, got {min_shots}")
        self._min_shots = min_shots

    def optimize(
        self,
        backend: str,
        budget_usd: float,
        *,
        target_shots: int | None = None,
        circuit_depth: int = 50,
    ) -> OptimizationResult:
        """Find the best compilation settings within *budget_usd*.

        Parameters
        ----------
        backend:
            Target backend identifier.
        budget_usd:
            Maximum allowed spend in USD.
        target_shots:
            Desired number of shots.  If *None*, maximises shots within
            the budget.
        circuit_depth:
            Estimated circuit depth (used for fidelity estimation).

        Returns
        -------
        OptimizationResult
            Recommended settings.

        Raises
        ------
        ~qb_compiler.exceptions.BudgetExceededError
            If even :attr:`min_shots` exceeds the budget.
        ValueError
            If the backend is not in the pricing table.
        """
        from qb_compiler.exceptions import BudgetExceededError

        if budget_usd <= 0:
            raise BudgetExceededError(
                estimated_usd=0.01,
                budget_usd=budget_usd,
                shots=1,
            )

        try:
            cps = cost_per_shot(backend)
        except KeyError:
            raise ValueError(
                f"No pricing data for backend '{backend}'. "
                f"Known backends: {sorted(VENDOR_PRICING.keys())}"
            ) from None

        # Maximum affordable shots
        max_shots = int(budget_usd / cps) if cps > 0 else target_shots or 10_000

        if max_shots < self._min_shots:
            raise BudgetExceededError(
                estimated_usd=cps * self._min_shots,
                budget_usd=budget_usd,
                shots=self._min_shots,
            )

        # Determine recommended shots
        recommended_shots = min(target_shots, max_shots) if target_shots is not None else max_shots

        recommended_shots = max(recommended_shots, self._min_shots)

        # Estimate fidelity from backend specs
        estimated_fidelity = self._estimate_fidelity(backend, circuit_depth)

        # Choose strategy based on cost
        strategy = self._recommend_strategy(cps)

        estimated_cost = cps * recommended_shots
        notes = self._build_notes(
            backend, target_shots, recommended_shots, max_shots, cps,
        )

        return OptimizationResult(
            backend=backend,
            recommended_shots=recommended_shots,
            estimated_fidelity=estimated_fidelity,
            estimated_cost_usd=estimated_cost,
            strategy=strategy,
            notes=notes,
        )

    def find_cheapest_backend(
        self,
        budget_usd: float,
        *,
        min_qubits: int = 1,
        target_shots: int = 1000,
    ) -> OptimizationResult | None:
        """Find the cheapest backend that can run *target_shots* within budget.

        Parameters
        ----------
        budget_usd:
            Maximum budget in USD.
        min_qubits:
            Minimum required qubit count.
        target_shots:
            Desired number of shots.

        Returns
        -------
        OptimizationResult | None
            Best option, or *None* if no backend fits.
        """
        candidates: list[OptimizationResult] = []

        for name, spec in BACKEND_CONFIGS.items():
            if spec.n_qubits < min_qubits:
                continue
            if name not in VENDOR_PRICING:
                continue
            cps = VENDOR_PRICING[name].cost_per_shot_usd
            total = cps * target_shots
            if total <= budget_usd:
                fidelity = self._estimate_fidelity(name, 50)
                candidates.append(OptimizationResult(
                    backend=name,
                    recommended_shots=target_shots,
                    estimated_fidelity=fidelity,
                    estimated_cost_usd=total,
                    strategy=self._recommend_strategy(cps),
                ))

        if not candidates:
            return None

        # Sort by fidelity (descending), then by cost (ascending)
        candidates.sort(key=lambda r: (-r.estimated_fidelity, r.estimated_cost_usd))
        return candidates[0]

    @staticmethod
    def _estimate_fidelity(backend: str, circuit_depth: int) -> float:
        """Rough fidelity estimate from backend median error rates.

        Uses ``(1 - cx_error) ^ depth * (1 - readout_error)`` as a
        simple multiplicative error model.
        """
        spec = BACKEND_CONFIGS.get(backend)
        if spec is None:
            return 0.5  # conservative default for unknown backends

        # Per-layer fidelity from 2q gate error
        layer_fidelity = (1.0 - spec.median_cx_error) ** circuit_depth
        readout_fidelity = 1.0 - spec.median_readout_error
        return max(0.0, min(1.0, layer_fidelity * readout_fidelity))

    @staticmethod
    def _recommend_strategy(cost_per_shot_usd: float) -> str:
        """Recommend strategy based on per-shot cost.

        Expensive backends benefit more from fidelity-optimal strategies
        since each wasted shot is costly.  Cheap backends can afford
        more shots, so speed-optimal is fine.
        """
        if cost_per_shot_usd >= 0.10:
            return "fidelity_optimal"
        if cost_per_shot_usd >= 0.001:
            return "cost_optimal"
        return "speed_optimal"

    @staticmethod
    def _build_notes(
        backend: str,
        target_shots: int | None,
        recommended_shots: int,
        max_shots: int,
        cps: float,
    ) -> str:
        parts: list[str] = []
        if target_shots is not None and recommended_shots < target_shots:
            parts.append(
                f"Reduced shots from {target_shots} to {recommended_shots} "
                f"to fit within budget"
            )
        if cps >= 0.10:
            parts.append(
                f"High per-shot cost (${cps:.2f}); "
                f"fidelity_optimal strategy recommended"
            )
        return "; ".join(parts) if parts else "Within budget"
