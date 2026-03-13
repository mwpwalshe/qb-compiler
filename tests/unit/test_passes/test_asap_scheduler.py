"""Tests for the ASAP scheduler pass."""

from __future__ import annotations

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.scheduling.asap_scheduler import ASAPScheduler


class TestASAPScheduler:
    """Tests for ASAPScheduler."""

    def test_empty_circuit(self) -> None:
        """Empty circuit should pass through without error."""
        circ = QBCircuit(n_qubits=2)
        scheduler = ASAPScheduler()
        ctx: dict = {}
        result = scheduler.run(circ, ctx)

        assert result.circuit.gate_count == 0
        assert result.modified is False

    def test_preserves_dependencies(self) -> None:
        """Gates with dependencies must maintain their relative order."""
        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))

        scheduler = ASAPScheduler()
        ctx: dict = {}
        result = scheduler.run(circ, ctx)

        gates = result.circuit.gates
        names = [g.name for g in gates]
        assert names.index("h") < names.index("cx")
        assert names.index("cx") < names.index("x")

    def test_context_populated(self) -> None:
        """Context should contain depth information."""
        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))

        scheduler = ASAPScheduler()
        ctx: dict = {}
        scheduler.run(circ, ctx)

        assert "asap_depth_before" in ctx
        assert "asap_depth_after" in ctx

    def test_independent_gates_minimize_depth(self) -> None:
        """Independent gates on different qubits should be in the same layer."""
        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))
        circ.add_gate(QBGate(name="z", qubits=(2,)))

        scheduler = ASAPScheduler()
        ctx: dict = {}
        result = scheduler.run(circ, ctx)

        # All three gates are independent — depth should be 1
        assert result.circuit.depth == 1
        assert result.circuit.gate_count == 3

    def test_preserves_gate_count(self) -> None:
        """Scheduling should not add or remove gates."""
        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="rz", qubits=(2,), params=(1.0,)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))

        scheduler = ASAPScheduler()
        ctx: dict = {}
        result = scheduler.run(circ, ctx)

        assert result.circuit.gate_count == 4
