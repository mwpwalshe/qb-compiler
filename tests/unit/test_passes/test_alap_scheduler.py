"""Tests for the ALAP scheduler pass."""

from __future__ import annotations

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.scheduling.alap_scheduler import ALAPScheduler


class TestALAPScheduler:
    """Tests for ALAPScheduler."""

    def test_empty_circuit(self) -> None:
        """Empty circuit should pass through without error."""
        circ = QBCircuit(n_qubits=2)
        scheduler = ALAPScheduler()
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

        scheduler = ALAPScheduler()
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

        scheduler = ALAPScheduler()
        ctx: dict = {}
        scheduler.run(circ, ctx)

        assert "alap_depth_before" in ctx
        assert "alap_depth_after" in ctx

    def test_preserves_gate_count(self) -> None:
        """Scheduling should not add or remove gates."""
        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="rz", qubits=(2,), params=(1.0,)))

        scheduler = ALAPScheduler()
        ctx: dict = {}
        result = scheduler.run(circ, ctx)

        assert result.circuit.gate_count == 4
