"""Tests for circuit viability checking."""

from __future__ import annotations

from qiskit import QuantumCircuit

from qb_compiler.viability import (
    ViabilityResult,
    _estimate_viable_depth,
    check_viability,
)

# ── viable depth estimation ──────────────────────────────────────


class TestEstimateViableDepth:
    def test_low_error_gives_high_depth(self):
        # Very low error → can run deep circuits
        d = _estimate_viable_depth(0.001, 0.005, 5)
        assert d > 200

    def test_high_error_gives_low_depth(self):
        # High error → shallow circuits only
        d = _estimate_viable_depth(0.05, 0.03, 5)
        assert d < 60

    def test_more_qubits_lower_depth(self):
        # More qubits → noise floor drops → viable depth increases
        d5 = _estimate_viable_depth(0.005, 0.01, 5)
        d15 = _estimate_viable_depth(0.005, 0.01, 15)
        # With more qubits, noise floor is lower so we can tolerate
        # more gate errors before hitting it
        assert d15 > d5

    def test_zero_error_gives_large_depth(self):
        d = _estimate_viable_depth(0.0, 0.01, 5)
        assert d >= 1000

    def test_extreme_error(self):
        d = _estimate_viable_depth(1.0, 0.5, 5)
        assert d >= 1000  # fallback for invalid error


# ── viability check ──────────────────────────────────────────────


class TestCheckViability:
    def test_bell_state_is_viable(self):
        qc = QuantumCircuit(2, 2, name="Bell")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))

        result = check_viability(qc, backend="ibm_fez", n_seeds=2)
        assert isinstance(result, ViabilityResult)
        assert result.viable is True
        assert result.status == "VIABLE"
        assert result.estimated_fidelity > 0.8
        assert result.signal_to_noise > 2.0
        assert result.two_qubit_gate_count >= 1
        assert "proceed" in result.suggestions[0].lower() or "good" in result.suggestions[0].lower()

    def test_ghz_8_is_viable(self):
        qc = QuantumCircuit(8, 8, name="GHZ-8")
        qc.h(0)
        for i in range(7):
            qc.cx(i, i + 1)
        qc.measure(range(8), range(8))

        result = check_viability(qc, backend="ibm_fez", n_seeds=2)
        assert result.viable is True
        assert result.estimated_fidelity > 0.5

    def test_deep_circuit_not_viable(self):
        import random

        random.seed(99)
        qc = QuantumCircuit(15, 15, name="Deep-15q")
        for _ in range(500):
            q1, q2 = random.sample(range(15), 2)
            qc.cx(q1, q2)
        qc.measure(range(15), range(15))

        result = check_viability(qc, backend="ibm_fez", n_seeds=2)
        assert result.viable is False
        assert result.status == "NOT VIABLE"
        assert result.estimated_fidelity < 0.01
        assert any(
            "waste" in s.lower() or "not submit" in s.lower() or "Do not" in s
            for s in result.suggestions
        )

    def test_result_has_cost_estimate(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))

        result = check_viability(qc, backend="ibm_fez", n_seeds=1)
        assert result.cost_estimate_usd is not None
        assert result.cost_estimate_usd > 0

    def test_result_str_is_readable(self):
        qc = QuantumCircuit(2, 2, name="Test")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))

        result = check_viability(qc, backend="ibm_fez", n_seeds=1)
        text = str(result)
        assert "Test" in text
        assert "ibm_fez" in text
        assert "VIABLE" in text

    def test_unknown_backend_still_works(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))

        # Unknown backend with no calibration → uses rough estimate
        result = check_viability(qc, backend="fake_backend", n_seeds=1)
        assert isinstance(result, ViabilityResult)
        assert result.backend == "fake_backend"

    def test_viable_depth_in_result(self):
        qc = QuantumCircuit(5, 5, name="Test")
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)
        qc.measure(range(5), range(5))

        result = check_viability(qc, backend="ibm_fez", n_seeds=1)
        assert result.viable_depth > 0
        assert result.depth > 0
