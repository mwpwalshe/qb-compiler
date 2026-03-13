"""Tests for :class:`qb_compiler.cost.budget_optimizer.BudgetOptimizer`."""

from __future__ import annotations

import pytest

from qb_compiler.cost.budget_optimizer import BudgetOptimizer, OptimizationResult
from qb_compiler.exceptions import BudgetExceededError


class TestBudgetOptimizer:
    def test_optimize_ibm_fez_within_budget(self) -> None:
        """Should return a valid result for a cheap backend with ample budget."""
        optimizer = BudgetOptimizer(min_shots=100)
        result = optimizer.optimize("ibm_fez", budget_usd=10.0)
        assert isinstance(result, OptimizationResult)
        assert result.backend == "ibm_fez"
        assert result.recommended_shots >= 100
        assert result.estimated_cost_usd <= 10.0
        assert 0.0 <= result.estimated_fidelity <= 1.0

    def test_optimize_reduces_shots_to_fit(self) -> None:
        """When target_shots exceeds budget, should reduce shots."""
        optimizer = BudgetOptimizer(min_shots=10)
        result = optimizer.optimize(
            "ionq_aria",
            budget_usd=10.0,
            target_shots=1000,
        )
        # IonQ Aria: $0.30/shot, budget $10 -> max 33 shots
        assert result.recommended_shots <= 33
        assert result.recommended_shots >= 10
        assert result.estimated_cost_usd <= 10.0

    def test_optimize_raises_on_tiny_budget(self) -> None:
        """Should raise BudgetExceededError when min_shots can't be afforded."""
        optimizer = BudgetOptimizer(min_shots=100)
        with pytest.raises(BudgetExceededError):
            optimizer.optimize("ionq_aria", budget_usd=1.0)
            # 100 shots * $0.30 = $30 > $1

    def test_optimize_raises_on_zero_budget(self) -> None:
        """Zero budget should raise BudgetExceededError."""
        optimizer = BudgetOptimizer(min_shots=1)
        with pytest.raises(BudgetExceededError):
            optimizer.optimize("ibm_fez", budget_usd=0.0)

    def test_optimize_strategy_recommendation(self) -> None:
        """Expensive backends should recommend fidelity_optimal."""
        optimizer = BudgetOptimizer(min_shots=1)
        result = optimizer.optimize("ionq_aria", budget_usd=100.0)
        assert result.strategy == "fidelity_optimal"

    def test_optimize_cheap_backend_strategy(self) -> None:
        """Cheap backends should recommend speed_optimal or cost_optimal."""
        optimizer = BudgetOptimizer(min_shots=100)
        result = optimizer.optimize("ibm_fez", budget_usd=100.0)
        assert result.strategy in ("speed_optimal", "cost_optimal")

    def test_find_cheapest_backend(self) -> None:
        """Should find a backend that fits within budget."""
        optimizer = BudgetOptimizer(min_shots=100)
        result = optimizer.find_cheapest_backend(
            budget_usd=1.0,
            target_shots=1000,
        )
        assert result is not None
        assert result.estimated_cost_usd <= 1.0
        assert result.recommended_shots == 1000

    def test_find_cheapest_backend_none_fits(self) -> None:
        """Should return None when no backend fits."""
        optimizer = BudgetOptimizer(min_shots=100)
        result = optimizer.find_cheapest_backend(
            budget_usd=0.001,
            target_shots=1000,
            min_qubits=100,
        )
        assert result is None

    def test_invalid_min_shots(self) -> None:
        """min_shots < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="min_shots"):
            BudgetOptimizer(min_shots=0)

    def test_fidelity_estimation_decreases_with_depth(self) -> None:
        """Deeper circuits should have lower estimated fidelity."""
        optimizer = BudgetOptimizer(min_shots=100)
        shallow = optimizer.optimize("ibm_fez", budget_usd=100.0, circuit_depth=10)
        deep = optimizer.optimize("ibm_fez", budget_usd=100.0, circuit_depth=500)
        assert shallow.estimated_fidelity > deep.estimated_fidelity
