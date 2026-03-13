"""Tests for the commutation optimisation pass."""

from __future__ import annotations

import math

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.transformation.commutation_analysis import CommutationOptimizer


class TestCommutationOptimizer:
    """Tests for CommutationOptimizer."""

    def test_merge_adjacent_rz(self) -> None:
        """Two adjacent RZ gates on the same qubit should merge."""
        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="rz", qubits=(0,), params=(0.5,)))
        circ.add_gate(QBGate(name="rz", qubits=(0,), params=(0.3,)))

        opt = CommutationOptimizer()
        ctx: dict = {}
        result = opt.run(circ, ctx)

        assert result.modified is True
        assert result.circuit.gate_count == 1
        gate = result.circuit.gates[0]
        assert gate.name == "rz"
        assert abs(gate.params[0] - 0.8) < 1e-10

    def test_no_merge_different_axes(self) -> None:
        """RZ followed by RX should not merge (different rotation axes)."""
        circ = QBCircuit(n_qubits=1)
        circ.add_gate(QBGate(name="rz", qubits=(0,), params=(0.5,)))
        circ.add_gate(QBGate(name="rx", qubits=(0,), params=(0.3,)))

        opt = CommutationOptimizer()
        ctx: dict = {}
        result = opt.run(circ, ctx)

        assert result.modified is False
        assert result.circuit.gate_count == 2

    def test_eliminate_zero_rotation(self) -> None:
        """RZ(pi) + RZ(pi) = RZ(2pi) ≈ identity should be eliminated."""
        circ = QBCircuit(n_qubits=1)
        circ.add_gate(QBGate(name="rz", qubits=(0,), params=(math.pi,)))
        circ.add_gate(QBGate(name="rz", qubits=(0,), params=(math.pi,)))

        opt = CommutationOptimizer()
        ctx: dict = {}
        result = opt.run(circ, ctx)

        assert result.modified is True
        assert result.circuit.gate_count == 0

    def test_no_merge_different_qubits(self) -> None:
        """RZ on different qubits should not merge."""
        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="rz", qubits=(0,), params=(0.5,)))
        circ.add_gate(QBGate(name="rz", qubits=(1,), params=(0.3,)))

        opt = CommutationOptimizer()
        ctx: dict = {}
        result = opt.run(circ, ctx)

        assert result.modified is False
        assert result.circuit.gate_count == 2

    def test_empty_circuit_unchanged(self) -> None:
        """Empty circuit should pass through."""
        circ = QBCircuit(n_qubits=1)

        opt = CommutationOptimizer()
        ctx: dict = {}
        result = opt.run(circ, ctx)

        assert result.modified is False
