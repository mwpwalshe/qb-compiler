"""Execution cost estimation."""

from __future__ import annotations

from dataclasses import dataclass

from qb_compiler.cost.pricing import VENDOR_PRICING, cost_per_shot
from qb_compiler.exceptions import BackendNotSupportedError, BudgetExceededError


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Result of a cost estimation.

    Parameters
    ----------
    backend:
        Backend identifier.
    shots:
        Number of shots requested.
    cost_per_shot:
        Per-shot cost in USD.
    total_cost_usd:
        ``shots * cost_per_shot``.
    currency:
        Always ``"USD"``.
    """

    backend: str
    shots: int
    cost_per_shot: float
    total_cost_usd: float
    currency: str = "USD"

    def __repr__(self) -> str:
        return (
            f"CostEstimate(backend={self.backend!r}, shots={self.shots:,}, "
            f"total=${self.total_cost_usd:.4f})"
        )


class CostEstimator:
    """Estimates execution cost for quantum circuits.

    Parameters
    ----------
    budget_usd:
        Optional spending cap.  If set, :meth:`estimate` will raise
        :class:`BudgetExceededError` when the estimate exceeds this.
    """

    def __init__(self, *, budget_usd: float | None = None) -> None:
        self._budget = budget_usd

    def estimate(self, backend: str, shots: int) -> CostEstimate:
        """Estimate execution cost for *shots* on *backend*.

        Parameters
        ----------
        backend:
            Backend identifier (must be in :data:`VENDOR_PRICING`).
        shots:
            Number of shots to run.

        Returns
        -------
        CostEstimate
            Breakdown of the estimated cost.

        Raises
        ------
        BackendNotSupportedError
            If *backend* is not in the pricing table.
        BudgetExceededError
            If the estimate exceeds the configured budget.
        """
        if shots < 0:
            raise ValueError(f"shots must be non-negative, got {shots}")

        try:
            cps = cost_per_shot(backend)
        except KeyError:
            raise BackendNotSupportedError(
                backend, list(VENDOR_PRICING.keys())
            ) from None

        total = cps * shots
        estimate = CostEstimate(
            backend=backend,
            shots=shots,
            cost_per_shot=cps,
            total_cost_usd=total,
        )

        if self._budget is not None and total > self._budget:
            raise BudgetExceededError(
                estimated_usd=total,
                budget_usd=self._budget,
                shots=shots,
            )

        return estimate

    def max_shots_within_budget(self, backend: str, budget_usd: float | None = None) -> int:
        """Return the maximum number of shots affordable within *budget_usd*.

        Parameters
        ----------
        backend:
            Backend identifier.
        budget_usd:
            Budget cap.  If *None*, uses the estimator's configured budget.
            If neither is set, raises :class:`ValueError`.

        Returns
        -------
        int
            Maximum whole shots that fit within the budget.
        """
        budget = budget_usd if budget_usd is not None else self._budget
        if budget is None:
            raise ValueError("No budget specified")
        if budget <= 0:
            return 0

        try:
            cps = cost_per_shot(backend)
        except KeyError:
            raise BackendNotSupportedError(
                backend, list(VENDOR_PRICING.keys())
            ) from None

        if cps <= 0:
            # Free backend — effectively unlimited
            return 2**31 - 1
        return int(budget / cps)

    def compare_backends(
        self, shots: int, backends: list[str] | None = None
    ) -> list[CostEstimate]:
        """Estimate cost across multiple backends, sorted cheapest first.

        Parameters
        ----------
        shots:
            Number of shots per backend.
        backends:
            List of backend identifiers.  If *None*, uses all known backends.

        Returns
        -------
        list[CostEstimate]
            Sorted from cheapest to most expensive.
        """
        if backends is None:
            backends = list(VENDOR_PRICING.keys())

        estimates = []
        for b in backends:
            try:
                estimates.append(self.estimate(b, shots))
            except BackendNotSupportedError:
                continue  # skip unknown backends silently
            except BudgetExceededError as exc:
                # Still include in comparison, just mark the overage
                estimates.append(CostEstimate(
                    backend=b,
                    shots=shots,
                    cost_per_shot=exc.estimated_usd / shots if shots > 0 else 0,
                    total_cost_usd=exc.estimated_usd,
                ))

        estimates.sort(key=lambda e: e.total_cost_usd)
        return estimates
