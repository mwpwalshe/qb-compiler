"""Tests for the error budget estimator analysis pass."""

from __future__ import annotations

import pytest

from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.analysis.error_budget_estimator import ErrorBudgetEstimator


def _make_props(
    n: int,
    t1: float = 100.0,
    t2: float = 50.0,
    readout_error: float = 0.01,
) -> list[QubitProperties]:
    """Create n identical QubitProperties."""
    return [
        QubitProperties(qubit_id=i, t1_us=t1, t2_us=t2, readout_error=readout_error)
        for i in range(n)
    ]


class TestErrorBudgetEstimator:
    """Tests for ErrorBudgetEstimator."""

    def test_perfect_circuit_fidelity_one(self) -> None:
        """A circuit with no errors and no idle time should have fidelity ~1."""
        circ = QBCircuit(1, n_clbits=1)
        circ.add_gate(QBGate(name="x", qubits=(0,)))
        circ.add_measurement(0, 0)

        props = [QubitProperties(qubit_id=0, t1_us=1e6, t2_us=1e6, readout_error=0.0)]
        estimator = ErrorBudgetEstimator(
            qubit_properties=props,
            gate_error_rates={},  # all gates perfect
            gate_duration_us=0.001,
        )
        context: dict = {}
        estimator.run(circ, context)

        assert context["estimated_fidelity"] == pytest.approx(1.0, abs=1e-6)
        assert context["error_budget"]["gate"] == pytest.approx(0.0)
        assert context["error_budget"]["readout"] == pytest.approx(0.0)

    def test_gate_errors_reduce_fidelity(self) -> None:
        """Gate errors should reduce estimated fidelity."""
        circ = QBCircuit(2)
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))

        props = _make_props(2, t1=1e6, readout_error=0.0)
        estimator = ErrorBudgetEstimator(
            qubit_properties=props,
            gate_error_rates={"cx": 0.01},
            gate_duration_us=0.001,
        )
        context: dict = {}
        estimator.run(circ, context)

        # Two CX gates each with 1% error: fidelity ~ 0.99 * 0.99 = 0.9801
        assert context["estimated_fidelity"] == pytest.approx(0.99 * 0.99, abs=1e-4)
        assert context["error_budget"]["gate"] == pytest.approx(0.02, abs=1e-6)

    def test_readout_errors_in_budget(self) -> None:
        """Readout errors should appear in the error budget."""
        circ = QBCircuit(2, n_clbits=2)
        circ.add_gate(QBGate(name="x", qubits=(0,)))
        circ.add_measurement(0, 0)
        circ.add_measurement(1, 1)

        props = _make_props(2, t1=1e6, readout_error=0.05)
        estimator = ErrorBudgetEstimator(
            qubit_properties=props,
            gate_error_rates={},
            gate_duration_us=0.001,
        )
        context: dict = {}
        estimator.run(circ, context)

        # Two measured qubits each with 5% readout error
        assert context["error_budget"]["readout"] == pytest.approx(0.10, abs=1e-6)
        # Fidelity should be reduced by readout
        assert context["estimated_fidelity"] < 1.0

    def test_decoherence_from_idle_time(self) -> None:
        """Qubits with idle time should accumulate decoherence error."""
        circ = QBCircuit(2)
        # Only act on qubit 0 twice -- qubit 1 is idle but used
        circ.add_gate(QBGate(name="x", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))
        circ.add_gate(QBGate(name="x", qubits=(0,)))

        props = _make_props(2, t1=10.0, readout_error=0.0)
        estimator = ErrorBudgetEstimator(
            qubit_properties=props,
            gate_error_rates={},
            gate_duration_us=1.0,  # 1us per gate slot
        )
        context: dict = {}
        estimator.run(circ, context)

        # Qubit 1 has depth=2, busy=1 -> 1 idle slot -> exp(-1/10) contribution
        assert context["error_budget"]["decoherence"] > 0.0
        assert context["estimated_fidelity"] < 1.0

    def test_empty_circuit(self) -> None:
        """An empty circuit should have perfect fidelity."""
        circ = QBCircuit(2)
        props = _make_props(2)
        estimator = ErrorBudgetEstimator(qubit_properties=props)
        context: dict = {}
        estimator.run(circ, context)

        assert context["estimated_fidelity"] == pytest.approx(1.0)
        assert context["error_budget"]["gate"] == 0.0
        assert context["error_budget"]["decoherence"] == 0.0
        assert context["error_budget"]["readout"] == 0.0
