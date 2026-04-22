"""Tests for the noise-aware ALAP scheduler pass."""

from __future__ import annotations

from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.scheduling.noise_aware_scheduler import NoiseAwareScheduler


def _make_props(*t2_values: float) -> list[QubitProperties]:
    """Create QubitProperties with given T2 values (index = qubit_id)."""
    return [QubitProperties(qubit_id=i, t1_us=100.0, t2_us=t2) for i, t2 in enumerate(t2_values)]


class TestNoiseAwareScheduler:
    """Tests for NoiseAwareScheduler."""

    def test_empty_circuit_unchanged(self) -> None:
        """An empty circuit should pass through without error."""
        circ = QBCircuit(2)
        props = _make_props(50.0, 100.0)
        scheduler = NoiseAwareScheduler(qubit_properties=props)
        context: dict = {}
        result = scheduler.run(circ, context)

        assert result.circuit.gate_count == 0
        assert "estimated_decoherence_reduction" in context

    def test_independent_gates_reordered_by_urgency(self) -> None:
        """Independent gates on different qubits should be reordered so
        that high-urgency (low T2) qubits appear later in the schedule."""
        circ = QBCircuit(3)
        # q0: T2=10 (high urgency), q1: T2=100 (low urgency), q2: T2=50 (medium)
        circ.add_gate(QBGate(name="x", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))
        circ.add_gate(QBGate(name="x", qubits=(2,)))

        props = _make_props(10.0, 100.0, 50.0)
        scheduler = NoiseAwareScheduler(qubit_properties=props)
        context: dict = {}
        result = scheduler.run(circ, context)

        # All three gates are independent (same layer), should be sorted
        # by urgency: q1 (lowest urgency) first, q2 middle, q0 (highest) last
        gates = result.circuit.gates
        assert len(gates) == 3
        assert gates[0].qubits == (1,)  # T2=100, urgency=0.01
        assert gates[1].qubits == (2,)  # T2=50, urgency=0.02
        assert gates[2].qubits == (0,)  # T2=10, urgency=0.1

    def test_dependencies_preserved(self) -> None:
        """Gates with data dependencies must maintain their relative order."""
        circ = QBCircuit(2)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))

        props = _make_props(10.0, 100.0)
        scheduler = NoiseAwareScheduler(qubit_properties=props)
        context: dict = {}
        result = scheduler.run(circ, context)

        gates = result.circuit.gates
        assert len(gates) == 3
        # h must come before cx (dependency on q0)
        # cx must come before x(1) (dependency on q1)
        gate_names = [g.name for g in gates]
        assert gate_names.index("h") < gate_names.index("cx")
        assert gate_names.index("cx") < gate_names.index("x")

    def test_context_has_decoherence_reduction(self) -> None:
        """The pass should populate 'estimated_decoherence_reduction' in context."""
        circ = QBCircuit(2)
        circ.add_gate(QBGate(name="x", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))

        props = _make_props(50.0, 100.0)
        scheduler = NoiseAwareScheduler(qubit_properties=props)
        context: dict = {}
        scheduler.run(circ, context)

        assert "estimated_decoherence_reduction" in context
        assert isinstance(context["estimated_decoherence_reduction"], float)
        assert context["estimated_decoherence_reduction"] >= 0.0
