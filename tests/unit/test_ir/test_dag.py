"""Tests for :class:`qb_compiler.ir.dag.QBDag`."""

from __future__ import annotations

import pytest

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.dag import QBDag
from qb_compiler.ir.operations import QBGate


class TestQBDag:
    """DAG construction and traversal."""

    def _make_bell_circuit(self) -> QBCircuit:
        """H(0), CX(0,1) — simple Bell state."""
        circ = QBCircuit(2)
        circ.add_gate(QBGate("h", (0,)))
        circ.add_gate(QBGate("cx", (0, 1)))
        return circ

    def test_from_circuit_roundtrip(self) -> None:
        """Converting circuit -> DAG -> circuit should preserve gates."""
        original = self._make_bell_circuit()
        dag = QBDag.from_circuit(original)
        restored = dag.to_circuit()

        # Same number of gates
        assert restored.gate_count == original.gate_count
        # Same gate names in order
        orig_names = [g.name for g in original.gates]
        rest_names = [g.name for g in restored.gates]
        assert rest_names == orig_names

    def test_topological_order(self) -> None:
        """Topological iteration should respect qubit dependencies.

        H(0) must come before CX(0,1) since they share qubit 0.
        """
        circ = self._make_bell_circuit()
        dag = QBDag.from_circuit(circ)
        ops = list(dag.topological_ops())

        assert len(ops) == 2
        # H must precede CX
        names = [op.name for op in ops if isinstance(op, QBGate)]
        assert names == ["h", "cx"]

    def test_topological_order_independent_qubits(self) -> None:
        """Gates on independent qubits have no forced order."""
        circ = QBCircuit(4)
        circ.add_gate(QBGate("h", (0,)))
        circ.add_gate(QBGate("h", (2,)))
        circ.add_gate(QBGate("cx", (0, 1)))
        circ.add_gate(QBGate("cx", (2, 3)))

        dag = QBDag.from_circuit(circ)
        ops = list(dag.topological_ops())
        assert len(ops) == 4

        # Both H gates must come before their respective CX gates
        names_qubits = [(op.name, op.qubits) for op in ops if isinstance(op, QBGate)]
        h0_idx = next(i for i, (n, q) in enumerate(names_qubits) if n == "h" and 0 in q)
        cx01_idx = next(i for i, (n, q) in enumerate(names_qubits) if n == "cx" and q == (0, 1))
        h2_idx = next(i for i, (n, q) in enumerate(names_qubits) if n == "h" and 2 in q)
        cx23_idx = next(i for i, (n, q) in enumerate(names_qubits) if n == "cx" and q == (2, 3))
        assert h0_idx < cx01_idx
        assert h2_idx < cx23_idx

    def test_layers(self) -> None:
        """Layers should group independent operations together.

        H(0) and H(2) are independent -> layer 0
        CX(0,1) and CX(2,3) are independent -> layer 1
        """
        circ = QBCircuit(4)
        circ.add_gate(QBGate("h", (0,)))
        circ.add_gate(QBGate("h", (2,)))
        circ.add_gate(QBGate("cx", (0, 1)))
        circ.add_gate(QBGate("cx", (2, 3)))

        dag = QBDag.from_circuit(circ)
        layers = dag.layers()

        # Should have exactly 2 layers
        assert len(layers) == 2

        # Layer 0: two H gates
        layer0_names = sorted(op.name for op in layers[0] if isinstance(op, QBGate))
        assert layer0_names == ["h", "h"]

        # Layer 1: two CX gates
        layer1_names = sorted(op.name for op in layers[1] if isinstance(op, QBGate))
        assert layer1_names == ["cx", "cx"]

    def test_empty_circuit_dag(self) -> None:
        """DAG from an empty circuit should have zero nodes."""
        circ = QBCircuit(2)
        dag = QBDag.from_circuit(circ)
        assert dag.node_count == 0
        assert dag.layers() == []

    def test_node_and_edge_counts(self) -> None:
        """Verify node/edge counts for a known circuit."""
        circ = self._make_bell_circuit()
        dag = QBDag.from_circuit(circ)

        assert dag.node_count == 2
        # H(0) -> CX(0,1): one edge on qubit 0
        assert dag.edge_count == 1
