"""Tests for :class:`qb_compiler.ir.circuit.QBCircuit`."""

from __future__ import annotations

import pytest

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate


class TestQBCircuit:
    """Tests for the IR-level QBCircuit (in qb_compiler.ir.circuit)."""

    def test_create_empty_circuit(self) -> None:
        """A fresh circuit should have zero operations and zero depth."""
        circ = QBCircuit(4)
        assert circ.n_qubits == 4
        assert circ.n_clbits == 0
        assert circ.gate_count == 0
        assert circ.depth == 0
        assert len(circ) == 0

    def test_create_circuit_invalid_qubits(self) -> None:
        """Creating a circuit with < 1 qubit should raise ValueError."""
        with pytest.raises(ValueError, match="n_qubits"):
            QBCircuit(0)

    def test_add_gates(self) -> None:
        """Adding gates should increase gate count and length."""
        circ = QBCircuit(3)
        circ.add_gate(QBGate("h", (0,)))
        circ.add_gate(QBGate("cx", (0, 1)))
        circ.add_gate(QBGate("rz", (2,), params=(1.57,)))

        assert circ.gate_count == 3
        assert len(circ) == 3
        assert len(circ.gates) == 3

    def test_add_gate_out_of_range(self) -> None:
        """Adding a gate with a qubit index out of range should raise."""
        circ = QBCircuit(2)
        with pytest.raises(IndexError):
            circ.add_gate(QBGate("h", (5,)))

    def test_depth_calculation(self) -> None:
        """Depth should track the longest chain per qubit.

        Circuit layout:
            q0: H ─ CX ─ ─
            q1: ─ ─ CX ─ H
            q2: H ─ ─ ─ ─ ─

        H on q0 = layer 1, CX(0,1) = layer 2, H on q1 = layer 3
        H on q2 = layer 1
        => depth = 3
        """
        circ = QBCircuit(3)
        circ.add_gate(QBGate("h", (0,)))  # q0: layer 1
        circ.add_gate(QBGate("h", (2,)))  # q2: layer 1
        circ.add_gate(QBGate("cx", (0, 1)))  # q0,q1: layer 2
        circ.add_gate(QBGate("h", (1,)))  # q1: layer 3

        assert circ.depth == 3

    def test_two_qubit_gate_count(self) -> None:
        """two_qubit_gate_count should count only multi-qubit gates."""
        circ = QBCircuit(3)
        circ.add_gate(QBGate("h", (0,)))
        circ.add_gate(QBGate("cx", (0, 1)))
        circ.add_gate(QBGate("h", (1,)))
        circ.add_gate(QBGate("cx", (1, 2)))
        circ.add_gate(QBGate("rz", (2,), params=(0.5,)))

        assert circ.two_qubit_gate_count == 2

    def test_copy_is_independent(self) -> None:
        """Modifying a copy should not affect the original."""
        circ = QBCircuit(2)
        circ.add_gate(QBGate("h", (0,)))
        circ.add_gate(QBGate("cx", (0, 1)))

        copied = circ.copy()
        assert copied == circ

        # Mutate the copy
        copied.add_gate(QBGate("h", (1,)))

        assert copied.gate_count == 3
        assert circ.gate_count == 2  # original unchanged
        assert copied != circ

    def test_measurements(self) -> None:
        """add_measurement should store QBMeasure operations."""
        circ = QBCircuit(2, n_clbits=2)
        circ.add_measurement(0, 0)
        circ.add_measurement(1, 1)

        assert len(circ.measurements) == 2
        assert len(circ) == 2
        assert circ.gate_count == 0  # measurements are not gates

    def test_gate_counts(self) -> None:
        """gate_counts should return a Counter of gate names."""
        circ = QBCircuit(3)
        circ.add_gate(QBGate("h", (0,)))
        circ.add_gate(QBGate("h", (1,)))
        circ.add_gate(QBGate("cx", (0, 1)))

        counts = circ.gate_counts
        assert counts["h"] == 2
        assert counts["cx"] == 1
