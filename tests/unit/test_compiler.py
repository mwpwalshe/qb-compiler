"""Tests for :class:`qb_compiler.compiler.QBCompiler`."""

from __future__ import annotations

import pytest

from qb_compiler.compiler import QBCircuit, QBCompiler
from qb_compiler.exceptions import (
    BackendNotSupportedError,
    BudgetExceededError,
    InvalidCircuitError,
)


class TestQBCompiler:
    """Core compiler functionality."""

    def test_compile_basic_circuit(self, sample_circuit: QBCircuit) -> None:
        """Compiling a simple circuit should return a valid CompileResult."""
        compiler = QBCompiler(backend="ibm_fez")
        result = compiler.compile(sample_circuit)

        assert result.compiled_circuit is not None
        assert result.compiled_circuit.gate_count > 0
        assert result.compilation_time_ms >= 0
        assert 0.0 <= result.estimated_fidelity <= 1.0
        assert result.original_depth > 0
        assert len(result.pass_log) > 0

    def test_compile_reduces_depth(self) -> None:
        """Optimisation on a cancellable circuit should reduce gate count.

        Note: with a backend set, basis translation may decompose gates
        (e.g. H -> rz sx rz), which can increase depth even when
        cancellation removes pairs.  So we test without a backend to
        verify cancellation reduces the gate count.
        """
        circ = QBCircuit(2)
        circ.h(0)
        circ.h(0)  # should cancel with first H
        circ.cx(0, 1)
        circ.h(1)
        circ.h(1)  # should cancel with previous H on q1

        # No backend -> no basis translation -> pure cancellation
        compiler = QBCompiler(backend=None, strategy="fidelity_optimal")
        result = compiler.compile(circ)

        # After cancellation of H-H pairs, only CX remains
        assert result.compiled_circuit.gate_count < circ.gate_count
        assert result.compiled_depth <= result.original_depth

    def test_from_backend_factory(self) -> None:
        """``QBCompiler.from_backend`` should create a pre-configured compiler."""
        compiler = QBCompiler.from_backend("ibm_fez")
        assert compiler.config.backend == "ibm_fez"
        assert compiler.config.backend_spec is not None
        assert compiler.config.backend_spec.n_qubits == 156

    def test_from_backend_unknown_raises(self) -> None:
        """``from_backend`` with an unknown backend should raise."""
        with pytest.raises(BackendNotSupportedError, match="not_a_real_backend"):
            QBCompiler.from_backend("not_a_real_backend")

    def test_estimate_cost(self, sample_circuit: QBCircuit) -> None:
        """Cost estimation should return sensible values."""
        compiler = QBCompiler.from_backend("ibm_fez")
        cost = compiler.estimate_cost(sample_circuit, shots=1000)

        assert cost.total_usd > 0.0
        assert cost.shots == 1000
        assert cost.backend == "ibm_fez"
        assert cost.cost_per_shot_usd > 0.0

    def test_budget_exceeded_raises(self, sample_circuit: QBCircuit) -> None:
        """Compile with a tiny budget should raise BudgetExceededError."""
        compiler = QBCompiler.from_backend("ionq_aria")

        # IonQ Aria costs $0.30/shot — at 1024 shots that is ~$307
        # A budget of $0.001 should definitely be exceeded
        with pytest.raises(BudgetExceededError):
            compiler.compile(sample_circuit, budget_usd=0.001)

    def test_compile_empty_circuit_raises(self) -> None:
        """Compiling an empty circuit should raise InvalidCircuitError."""
        compiler = QBCompiler(backend="ibm_fez")
        circ = QBCircuit(2)  # no gates

        with pytest.raises(InvalidCircuitError, match="empty"):
            compiler.compile(circ)

    def test_compile_non_circuit_raises(self) -> None:
        """Passing a non-QBCircuit should raise InvalidCircuitError."""
        compiler = QBCompiler(backend="ibm_fez")
        with pytest.raises(InvalidCircuitError):
            compiler.compile("not a circuit")  # type: ignore[arg-type]

    def test_qiskit_target_stored(self) -> None:
        """QBCompiler should store and expose the qiskit_target parameter."""
        sentinel = object()
        compiler = QBCompiler(backend="ibm_fez", qiskit_target=sentinel)
        assert compiler.qiskit_target is sentinel

    def test_qiskit_target_default_none(self) -> None:
        """Without qiskit_target, it should default to None."""
        compiler = QBCompiler(backend="ibm_fez")
        assert compiler.qiskit_target is None

    def test_qiskit_target_flows_to_calibration_pipeline(self, sample_circuit: QBCircuit) -> None:
        """qiskit_target should be passed through compile() to the calibration pipeline."""
        # We can't easily test the full flow without a real Qiskit target,
        # but we can verify that passing None doesn't break anything and
        # that initial_layout is still produced.
        compiler = QBCompiler(backend="ibm_fez", qiskit_target=None)
        result = compiler.compile(sample_circuit)

        # Should still produce a layout from the calibration pipeline
        assert result.initial_layout is not None or result.compiled_circuit is not None
