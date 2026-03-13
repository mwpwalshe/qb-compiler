"""Tests for the connectivity check analysis pass."""

from __future__ import annotations

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.analysis.connectivity_check import ConnectivityCheck


class TestConnectivityCheck:
    """Tests for ConnectivityCheck."""

    def test_all_gates_connected(self) -> None:
        """A circuit with 2Q gates only on connected qubits should pass."""
        coupling = [(0, 1), (1, 2)]
        check = ConnectivityCheck(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="cx", qubits=(1, 2)))

        ctx: dict = {}
        check.run(circ, ctx)

        assert ctx["connectivity_satisfied"] is True
        assert ctx["violating_gates"] == []

    def test_violating_gate_detected(self) -> None:
        """A CX on non-adjacent qubits should be flagged."""
        coupling = [(0, 1), (1, 2)]
        check = ConnectivityCheck(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 2)))

        ctx: dict = {}
        check.run(circ, ctx)

        assert ctx["connectivity_satisfied"] is False
        assert ctx["violating_gates"] == [0]

    def test_mixed_connected_and_violating(self) -> None:
        """Mix of connected and non-connected gates."""
        coupling = [(0, 1), (1, 2)]
        check = ConnectivityCheck(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))  # connected
        circ.add_gate(QBGate(name="h", qubits=(0,)))  # 1Q, ignored
        circ.add_gate(QBGate(name="cx", qubits=(0, 2)))  # violating

        ctx: dict = {}
        check.run(circ, ctx)

        assert ctx["connectivity_satisfied"] is False
        # gate index 0: cx(0,1), gate index 1: h(0), gate index 2: cx(0,2)
        assert ctx["violating_gates"] == [2]

    def test_single_qubit_gates_only(self) -> None:
        """Circuit with only 1Q gates should always satisfy connectivity."""
        coupling = [(0, 1)]
        check = ConnectivityCheck(coupling_map=coupling)

        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))

        ctx: dict = {}
        check.run(circ, ctx)

        assert ctx["connectivity_satisfied"] is True
        assert ctx["violating_gates"] == []

    def test_empty_circuit(self) -> None:
        """Empty circuit should satisfy connectivity."""
        coupling = [(0, 1)]
        check = ConnectivityCheck(coupling_map=coupling)

        circ = QBCircuit(n_qubits=2)

        ctx: dict = {}
        check.run(circ, ctx)

        assert ctx["connectivity_satisfied"] is True

    def test_undirected_coupling(self) -> None:
        """CX(1,0) should be connected if coupling has (0,1)."""
        coupling = [(0, 1)]
        check = ConnectivityCheck(coupling_map=coupling)

        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="cx", qubits=(1, 0)))

        ctx: dict = {}
        check.run(circ, ctx)

        assert ctx["connectivity_satisfied"] is True
