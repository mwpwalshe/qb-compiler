"""Tests for the gate decomposition pass."""

from __future__ import annotations

import math

import pytest

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.transformation.gate_decomposition import GateDecompositionPass


class TestGateDecomposition:
    """Tests for GateDecompositionPass."""

    def test_h_decomposition_ibm(self) -> None:
        """H gate decomposes to rz-sx-rz for IBM basis."""
        circ = QBCircuit(1)
        circ.add_gate(QBGate(name="h", qubits=(0,)))

        decomp = GateDecompositionPass(target_basis=("cx", "rz", "sx", "x", "id"))
        context: dict = {}
        result = decomp.run(circ, context)

        gates = result.circuit.gates
        names = [g.name for g in gates]
        assert names == ["rz", "sx", "rz"]
        assert gates[0].params == (math.pi,)
        assert gates[2].params == (math.pi,)
        assert context["decomposed_gates"] == 1

    def test_h_decomposition_rigetti(self) -> None:
        """H gate decomposes to rx-rz-rx for Rigetti basis."""
        circ = QBCircuit(1)
        circ.add_gate(QBGate(name="h", qubits=(0,)))

        decomp = GateDecompositionPass(target_basis=("cz", "rx", "rz"))
        context: dict = {}
        result = decomp.run(circ, context)

        gates = result.circuit.gates
        names = [g.name for g in gates]
        assert names == ["rx", "rz", "rx"]
        assert context["decomposed_gates"] == 1

    def test_already_in_basis_passthrough(self) -> None:
        """Gates already in the target basis should pass through unchanged."""
        circ = QBCircuit(2)
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="rz", qubits=(0,), params=(0.5,)))

        decomp = GateDecompositionPass(target_basis=("cx", "rz", "sx", "x", "id"))
        context: dict = {}
        result = decomp.run(circ, context)

        assert result.circuit.gate_count == 2
        assert not result.modified
        assert context["decomposed_gates"] == 0

    def test_swap_decomposition(self) -> None:
        """SWAP decomposes into 3 CX gates."""
        circ = QBCircuit(2)
        circ.add_gate(QBGate(name="swap", qubits=(0, 1)))

        decomp = GateDecompositionPass(target_basis=("cx", "rz", "sx", "x", "id"))
        context: dict = {}
        result = decomp.run(circ, context)

        gates = result.circuit.gates
        assert len(gates) == 3
        assert all(g.name == "cx" for g in gates)
        # Verify the CX pattern: (0,1), (1,0), (0,1)
        assert gates[0].qubits == (0, 1)
        assert gates[1].qubits == (1, 0)
        assert gates[2].qubits == (0, 1)
        assert context["decomposed_gates"] == 1

    def test_gate_count_changes_correctly(self) -> None:
        """Decomposition should increase gate count appropriately."""
        circ = QBCircuit(2)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))  # already in basis
        circ.add_gate(QBGate(name="swap", qubits=(0, 1)))

        decomp = GateDecompositionPass(target_basis=("cx", "rz", "sx", "x", "id"))
        context: dict = {}
        result = decomp.run(circ, context)

        # h -> 3 gates, cx -> 1 gate (passthrough), swap -> 3 gates = 7 total
        assert result.circuit.gate_count == 7
        assert context["decomposed_gates"] == 2

    def test_unknown_gate_raises_error(self) -> None:
        """A gate with no decomposition rule should raise ValueError."""
        circ = QBCircuit(2)
        circ.add_gate(QBGate(name="cswap", qubits=(0, 1, 2) if False else (0, 1)))

        # Use a gate that has no decomposition rule and is not in basis
        circ2 = QBCircuit(2)
        circ2.add_gate(QBGate(name="rzz", qubits=(0, 1), params=(0.5,)))

        decomp = GateDecompositionPass(target_basis=("cx", "rz", "sx", "x", "id"))
        context: dict = {}

        with pytest.raises(ValueError, match="No decomposition rule"):
            decomp.run(circ2, context)

    def test_ccx_decomposition(self) -> None:
        """Toffoli gate should decompose into CX and single-qubit gates."""
        circ = QBCircuit(3)
        circ.add_gate(QBGate(name="ccx", qubits=(0, 1, 2)))

        decomp = GateDecompositionPass(target_basis=("cx", "rz", "sx", "x", "id"))
        context: dict = {}
        result = decomp.run(circ, context)

        gates = result.circuit.gates
        # All output gates should be in the target basis
        for g in gates:
            assert g.name in ("cx", "rz", "sx", "x", "id"), \
                f"Gate '{g.name}' not in target basis"
        assert context["decomposed_gates"] == 1
        # Toffoli should produce multiple gates
        assert len(gates) > 1
