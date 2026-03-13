"""Tests for the circuit simplification pass."""

from __future__ import annotations

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.transformation.circuit_simplification import CircuitSimplifier


class TestCircuitSimplifier:
    """Tests for CircuitSimplifier."""

    def test_cancel_x_x(self) -> None:
        """Two adjacent X gates on the same qubit should cancel."""
        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="x", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(0,)))
        circ.add_gate(QBGate(name="h", qubits=(1,)))

        simplifier = CircuitSimplifier()
        ctx: dict = {}
        result = simplifier.run(circ, ctx)

        assert result.modified is True
        assert result.circuit.gate_count == 1
        assert result.circuit.gates[0].name == "h"

    def test_cancel_cx_cx(self) -> None:
        """Two adjacent CX gates on the same qubits should cancel."""
        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))

        simplifier = CircuitSimplifier()
        ctx: dict = {}
        result = simplifier.run(circ, ctx)

        assert result.modified is True
        assert result.circuit.gate_count == 0

    def test_h_cx_h_simplification(self) -> None:
        """H(t) CX(c,t) H(t) should simplify to CX(t,c)."""
        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="h", qubits=(1,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="h", qubits=(1,)))

        simplifier = CircuitSimplifier()
        ctx: dict = {}
        result = simplifier.run(circ, ctx)

        assert result.modified is True
        assert result.circuit.gate_count == 1
        gate = result.circuit.gates[0]
        assert gate.name == "cx"
        assert gate.qubits == (1, 0)  # control/target swapped

    def test_no_simplification_needed(self) -> None:
        """Circuit with no simplifiable patterns should pass through."""
        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="rz", qubits=(1,), params=(0.5,)))

        simplifier = CircuitSimplifier()
        ctx: dict = {}
        result = simplifier.run(circ, ctx)

        assert result.modified is False
        assert result.circuit.gate_count == 3

    def test_metadata_reports_count(self) -> None:
        """Metadata should report simplification count."""
        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="x", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(0,)))

        simplifier = CircuitSimplifier()
        ctx: dict = {}
        result = simplifier.run(circ, ctx)

        assert result.metadata["simplifications_applied"] == 1
