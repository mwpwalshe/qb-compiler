"""Tests for all compilation strategies and the get_strategy factory."""

from __future__ import annotations

import pytest

from qb_compiler.config import CompilerConfig
from qb_compiler.exceptions import BudgetExceededError
from qb_compiler.strategies import (
    BudgetAwareStrategy,
    CompilationStrategy,
    CostOptimalStrategy,
    DepthOptimalStrategy,
    FidelityOptimalStrategy,
    PassManager,
    SpeedOptimalStrategy,
    get_strategy,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _default_config(**overrides: object) -> CompilerConfig:
    """Create a CompilerConfig with sensible defaults for testing."""
    defaults: dict = {
        "backend": "ibm_fez",
        "optimization_level": 2,
    }
    defaults.update(overrides)
    return CompilerConfig(**defaults)


# ── SpeedOptimalStrategy ──────────────────────────────────────────────


class TestSpeedOptimalStrategy:
    def test_name(self) -> None:
        strategy = SpeedOptimalStrategy()
        assert strategy.name == "speed_optimal"

    def test_minimal_passes(self) -> None:
        """Speed strategy should have very few passes."""
        strategy = SpeedOptimalStrategy()
        config = _default_config()
        pm = strategy.build_pass_manager(config)
        assert isinstance(pm, PassManager)
        # Should have at most: trivial_layout + basis_translation (no coupling map)
        assert len(pm) <= 3

    def test_ignores_calibration(self) -> None:
        """Speed strategy should not use calibration-aware passes."""
        strategy = SpeedOptimalStrategy()
        config = _default_config()
        pm = strategy.build_pass_manager(config, calibration=None, noise_model=None)
        pass_names = [p.name for p in pm]
        assert "noise_aware_layout" not in pass_names
        assert "t2_aware_scheduling" not in pass_names

    def test_with_coupling_map(self) -> None:
        """Should add routing pass when coupling map is present."""
        strategy = SpeedOptimalStrategy()
        config = _default_config(
            coupling_map=[(0, 1), (1, 2), (2, 3)],
        )
        pm = strategy.build_pass_manager(config)
        pass_names = [p.name for p in pm]
        assert "swap_routing" in pass_names


# ── FidelityOptimalStrategy ───────────────────────────────────────────


class TestFidelityOptimalStrategy:
    def test_name(self) -> None:
        strategy = FidelityOptimalStrategy()
        assert strategy.name == "fidelity_optimal"

    def test_has_fidelity_estimation(self) -> None:
        """Fidelity strategy should include fidelity estimation pass."""
        strategy = FidelityOptimalStrategy()
        config = _default_config()
        pm = strategy.build_pass_manager(config)
        pass_names = [p.name for p in pm]
        assert "fidelity_estimation" in pass_names

    def test_more_passes_than_speed(self) -> None:
        """Fidelity strategy should have more passes than speed."""
        config = _default_config()
        speed_pm = SpeedOptimalStrategy().build_pass_manager(config)
        fidelity_pm = FidelityOptimalStrategy().build_pass_manager(config)
        assert len(fidelity_pm) > len(speed_pm)


# ── DepthOptimalStrategy ─────────────────────────────────────────────


class TestDepthOptimalStrategy:
    def test_name(self) -> None:
        strategy = DepthOptimalStrategy()
        assert strategy.name == "depth_optimal"

    def test_aggressive_cancellation(self) -> None:
        """Depth strategy should use aggressive gate cancellation."""
        strategy = DepthOptimalStrategy()
        config = _default_config()
        pm = strategy.build_pass_manager(config)
        cancel_passes = [p for p in pm if p.name == "gate_cancellation"]
        assert len(cancel_passes) >= 1
        # Should have high max_iterations
        assert cancel_passes[0].options.get("max_iterations", 0) >= 5

    def test_has_depth_analysis(self) -> None:
        """Should include a depth analysis pass."""
        strategy = DepthOptimalStrategy()
        config = _default_config()
        pm = strategy.build_pass_manager(config)
        pass_names = [p.name for p in pm]
        assert "depth_analysis" in pass_names

    def test_has_circuit_simplification(self) -> None:
        """Should include circuit simplification pass."""
        strategy = DepthOptimalStrategy()
        config = _default_config()
        pm = strategy.build_pass_manager(config)
        pass_names = [p.name for p in pm]
        assert "circuit_simplification" in pass_names


# ── CostOptimalStrategy ──────────────────────────────────────────────


class TestCostOptimalStrategy:
    def test_name(self) -> None:
        strategy = CostOptimalStrategy()
        assert strategy.name == "cost_optimal"

    def test_includes_cost_analysis(self) -> None:
        """Cost strategy should include cost analysis pass."""
        strategy = CostOptimalStrategy()
        config = _default_config()
        pm = strategy.build_pass_manager(config)
        pass_names = [p.name for p in pm]
        assert "cost_analysis" in pass_names

    def test_cost_per_shot_in_options(self) -> None:
        """Cost analysis pass should contain the backend's cost per shot."""
        strategy = CostOptimalStrategy()
        config = _default_config(backend="ibm_fez")
        pm = strategy.build_pass_manager(config)
        cost_pass = next(p for p in pm if p.name == "cost_analysis")
        assert cost_pass.options["cost_per_shot"] == pytest.approx(0.00016)


# ── BudgetAwareStrategy ──────────────────────────────────────────────


class TestBudgetAwareStrategy:
    def test_name(self) -> None:
        strategy = BudgetAwareStrategy(budget_usd=10.0)
        assert strategy.name == "budget_aware"

    def test_within_budget(self) -> None:
        """Should build pass manager when cost is within budget."""
        strategy = BudgetAwareStrategy(budget_usd=10.0, shots=1000)
        config = _default_config(backend="ibm_fez")
        pm = strategy.build_pass_manager(config)
        assert isinstance(pm, PassManager)
        assert len(pm) > 0
        # IBM Fez: 1000 * 0.00016 = $0.16, well within $10
        assert strategy.effective_shots == 1000

    def test_reduces_shots_to_fit_budget(self) -> None:
        """Should reduce shots when they exceed budget."""
        # IonQ Aria: $0.30/shot, 1000 shots = $300
        strategy = BudgetAwareStrategy(budget_usd=10.0, shots=1000)
        config = _default_config(backend="ionq_aria")
        strategy.build_pass_manager(config)
        # Budget $10 / $0.30 = 33 shots
        assert strategy.effective_shots == 33
        assert strategy.effective_shots < 1000

    def test_raises_on_impossible_budget(self) -> None:
        """Should raise BudgetExceededError when even 1 shot exceeds budget."""
        # IonQ Aria: $0.30/shot, budget $0.10 -> can't even afford 1 shot
        strategy = BudgetAwareStrategy(budget_usd=0.10, shots=100)
        config = _default_config(backend="ionq_aria")
        with pytest.raises(BudgetExceededError):
            strategy.build_pass_manager(config)

    def test_negative_budget_raises(self) -> None:
        """Negative budget should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            BudgetAwareStrategy(budget_usd=-5.0)

    def test_budget_constraint_pass_present(self) -> None:
        """Should include a budget_constraint analysis pass."""
        strategy = BudgetAwareStrategy(budget_usd=100.0, shots=500)
        config = _default_config(backend="ibm_fez")
        pm = strategy.build_pass_manager(config)
        pass_names = [p.name for p in pm]
        assert "budget_constraint" in pass_names


# ── get_strategy factory ─────────────────────────────────────────────


class TestGetStrategy:
    def test_speed_by_name(self) -> None:
        s = get_strategy("speed_optimal")
        assert isinstance(s, SpeedOptimalStrategy)

    def test_speed_short_name(self) -> None:
        s = get_strategy("speed")
        assert isinstance(s, SpeedOptimalStrategy)

    def test_fidelity_by_name(self) -> None:
        s = get_strategy("fidelity_optimal")
        assert isinstance(s, FidelityOptimalStrategy)

    def test_depth_by_name(self) -> None:
        s = get_strategy("depth")
        assert isinstance(s, DepthOptimalStrategy)

    def test_cost_by_name(self) -> None:
        s = get_strategy("cost_optimal")
        assert isinstance(s, CostOptimalStrategy)

    def test_budget_by_name(self) -> None:
        s = get_strategy("budget_aware", budget_usd=50.0)
        assert isinstance(s, BudgetAwareStrategy)
        assert s.budget_usd == 50.0

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent")

    def test_all_strategies_are_compilation_strategy(self) -> None:
        """Every strategy returned by the factory must be a CompilationStrategy."""
        names = ["speed", "fidelity", "depth", "cost"]
        for name in names:
            s = get_strategy(name)
            assert isinstance(s, CompilationStrategy), f"{name} is not a CompilationStrategy"
