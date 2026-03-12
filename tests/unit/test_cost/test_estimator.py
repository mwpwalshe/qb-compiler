"""Tests for :class:`qb_compiler.compiler.CostEstimator`."""

from __future__ import annotations

import pytest

from qb_compiler.compiler import CostEstimator
from qb_compiler.config import get_backend_spec


class TestCostEstimator:
    """Cost estimation for various backends."""

    def test_ibm_fez_cost(self) -> None:
        """IBM Fez cost should reflect $0.00016/shot baseline."""
        spec = get_backend_spec("ibm_fez")
        estimator = CostEstimator(spec)

        # depth=50, 4 qubits, 1000 shots
        # depth_factor = max(1.0, 50/100) = 1.0
        # per_shot = 0.00016 * 1.0 = 0.00016
        # total = 0.00016 * 1000 = 0.16
        result = estimator.estimate(depth=50, n_qubits=4, shots=1000)

        assert result.total_usd == pytest.approx(0.16, rel=0.01)
        assert result.cost_per_shot_usd == pytest.approx(0.00016, rel=0.01)
        assert result.shots == 1000
        assert result.depth == 50
        assert result.n_qubits == 4

    def test_ibm_fez_deep_circuit(self) -> None:
        """Deeper circuits should cost more (depth factor > 1)."""
        spec = get_backend_spec("ibm_fez")
        estimator = CostEstimator(spec)

        # depth=200 -> depth_factor = 200/100 = 2.0
        result = estimator.estimate(depth=200, n_qubits=4, shots=1000)

        assert result.total_usd == pytest.approx(0.32, rel=0.01)

    def test_ionq_aria_cost(self) -> None:
        """IonQ Aria at $0.30/shot should be much more expensive."""
        spec = get_backend_spec("ionq_aria")
        estimator = CostEstimator(spec)

        # depth=50, 4 qubits, 100 shots
        # depth_factor = max(1.0, 50/100) = 1.0
        # per_shot = 0.30 * 1.0 = 0.30
        # total = 0.30 * 100 = 30.0
        result = estimator.estimate(depth=50, n_qubits=4, shots=100)

        assert result.total_usd == pytest.approx(30.0, rel=0.01)
        assert result.cost_per_shot_usd == pytest.approx(0.30, rel=0.01)

    def test_zero_shots(self) -> None:
        """Zero shots should result in zero cost."""
        spec = get_backend_spec("ibm_fez")
        estimator = CostEstimator(spec)

        result = estimator.estimate(depth=50, n_qubits=4, shots=0)

        assert result.total_usd == 0.0
        assert result.shots == 0

    def test_within_budget(self) -> None:
        """CostEstimate.within_budget should correctly compare."""
        spec = get_backend_spec("ibm_fez")
        estimator = CostEstimator(spec)

        result = estimator.estimate(depth=50, n_qubits=4, shots=1000)

        assert result.within_budget(1.0) is True
        assert result.within_budget(0.01) is False
