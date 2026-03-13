"""Tests for the FidelityOptimalStrategy."""

from __future__ import annotations

import pytest

from qb_compiler.config import CompilerConfig
from qb_compiler.strategies import FidelityOptimalStrategy, PassManager


def _config(**overrides: object) -> CompilerConfig:
    defaults: dict = {"backend": "ibm_fez", "optimization_level": 2}
    defaults.update(overrides)
    return CompilerConfig(**defaults)


class TestFidelityOptimalStrategy:
    """Tests for FidelityOptimalStrategy pass manager construction."""

    def test_name(self) -> None:
        strategy = FidelityOptimalStrategy()
        assert strategy.name == "fidelity_optimal"

    def test_returns_pass_manager(self) -> None:
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config())
        assert isinstance(pm, PassManager)
        assert len(pm) > 0

    def test_includes_fidelity_estimation(self) -> None:
        """Fidelity strategy must include a fidelity estimation pass."""
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config())
        pass_names = [p.name for p in pm]
        assert "fidelity_estimation" in pass_names

    def test_includes_circuit_analysis(self) -> None:
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config())
        pass_names = [p.name for p in pm]
        assert "circuit_analysis" in pass_names

    def test_includes_gate_cancellation_at_opt_level_1(self) -> None:
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config(optimization_level=1))
        pass_names = [p.name for p in pm]
        assert "gate_cancellation" in pass_names

    def test_no_gate_cancellation_at_opt_level_0(self) -> None:
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config(optimization_level=0))
        pass_names = [p.name for p in pm]
        assert "gate_cancellation" not in pass_names

    def test_includes_basis_translation_with_backend(self) -> None:
        """When a backend is configured, basis translation should be present."""
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config())
        pass_names = [p.name for p in pm]
        assert "basis_translation" in pass_names

    def test_no_basis_translation_without_backend(self) -> None:
        """Without a backend, there is no basis to translate to."""
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config(backend=None))
        pass_names = [p.name for p in pm]
        assert "basis_translation" not in pass_names

    def test_trivial_layout_without_calibration(self) -> None:
        """Without calibration, should fall back to trivial layout."""
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config(), calibration=None, noise_model=None)
        pass_names = [p.name for p in pm]
        assert "trivial_layout" in pass_names
        assert "noise_aware_layout" not in pass_names

    def test_swap_routing_with_coupling_map(self) -> None:
        """When coupling_map is given, swap routing should be present."""
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(
            _config(coupling_map=[(0, 1), (1, 2)])
        )
        pass_names = [p.name for p in pm]
        assert "swap_routing" in pass_names

    def test_no_swap_routing_without_coupling_map(self) -> None:
        """Without coupling_map, no swap routing needed."""
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config())
        pass_names = [p.name for p in pm]
        assert "swap_routing" not in pass_names

    def test_post_route_cancellation_at_opt_level_2(self) -> None:
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config(optimization_level=2))
        pass_names = [p.name for p in pm]
        assert "post_route_cancellation" in pass_names

    def test_peephole_optimization_at_opt_level_3(self) -> None:
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config(optimization_level=3))
        pass_names = [p.name for p in pm]
        assert "peephole_optimization" in pass_names

    def test_no_peephole_at_opt_level_2(self) -> None:
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config(optimization_level=2))
        pass_names = [p.name for p in pm]
        assert "peephole_optimization" not in pass_names

    def test_scheduling_pass_present(self) -> None:
        """Some scheduling pass should always be present."""
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config())
        pass_names = [p.name for p in pm]
        has_scheduling = any(
            name in pass_names
            for name in ("alap_scheduling", "t2_aware_scheduling")
        )
        assert has_scheduling

    def test_alap_scheduling_without_noise_model(self) -> None:
        """Without noise model, should use regular ALAP scheduling."""
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config(), noise_model=None)
        pass_names = [p.name for p in pm]
        assert "alap_scheduling" in pass_names
        assert "t2_aware_scheduling" not in pass_names

    def test_gate_cancellation_iterations_higher_at_opt_level_3(self) -> None:
        """At opt level 3, gate cancellation should have more iterations."""
        strategy = FidelityOptimalStrategy()

        pm_2 = strategy.build_pass_manager(_config(optimization_level=2))
        pm_3 = strategy.build_pass_manager(_config(optimization_level=3))

        gc_2 = next(p for p in pm_2 if p.name == "gate_cancellation")
        gc_3 = next(p for p in pm_3 if p.name == "gate_cancellation")

        assert gc_3.options["max_iterations"] > gc_2.options["max_iterations"]

    def test_fidelity_estimation_is_last(self) -> None:
        """Fidelity estimation should be the last pass in the pipeline."""
        strategy = FidelityOptimalStrategy()
        pm = strategy.build_pass_manager(_config())
        passes = list(pm)
        assert passes[-1].name == "fidelity_estimation"
