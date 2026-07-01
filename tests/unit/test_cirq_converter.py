"""Tests for the Cirq converter (cirq is installed, so these RUN)."""

from __future__ import annotations

import math

import pytest

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from tests.conftest import requires_cirq


@requires_cirq
class TestFromCirq:
    """cirq.Circuit -> QBCircuit."""

    def test_bell_circuit(self) -> None:
        import cirq

        from qb_compiler.ir.converters.cirq_converter import from_cirq

        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1])])

        circ = from_cirq(circuit)
        assert circ.n_qubits == 2
        assert circ.gate_count == 2
        names = [g.name for g in circ.gates]
        assert names == ["h", "cx"]

    def test_parametrized_gates(self) -> None:
        import cirq

        from qb_compiler.ir.converters.cirq_converter import from_cirq

        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.rx(0.7)(q[0]), cirq.rz(1.2)(q[1]), cirq.CZ(q[0], q[1])])

        circ = from_cirq(circuit)
        by_name = {g.name: g for g in circ.gates}
        assert set(by_name) == {"rx", "rz", "cz"}
        assert by_name["rx"].params[0] == pytest.approx(0.7)
        assert by_name["rz"].params[0] == pytest.approx(1.2)
        assert by_name["cz"].qubits == (0, 1)

    def test_measurement_consumes_clbits(self) -> None:
        import cirq

        from qb_compiler.ir.converters.cirq_converter import from_cirq

        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            [cirq.H(q[0]), cirq.measure(q[0], key="a"), cirq.measure(q[1], key="b")]
        )

        circ = from_cirq(circuit)
        assert circ.n_clbits == 2
        assert len(circ.measurements) == 2


@requires_cirq
class TestToCirq:
    """QBCircuit -> cirq.Circuit."""

    def test_basic_gates(self) -> None:
        import cirq

        from qb_compiler.ir.converters.cirq_converter import to_cirq

        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))

        out = to_cirq(circ)
        assert isinstance(out, cirq.Circuit)
        gate_types = {type(op.gate).__name__ for op in out.all_operations()}
        # HPowGate (H) and CXPowGate (CNOT)
        assert "HPowGate" in gate_types
        assert "CXPowGate" in gate_types

    def test_unsupported_gate_raises(self) -> None:
        from qb_compiler.ir.converters.cirq_converter import to_cirq

        circ = QBCircuit(n_qubits=1)
        circ.add_gate(QBGate(name="ecr", qubits=(0,)))
        with pytest.raises(ValueError, match="Unsupported gate"):
            to_cirq(circ)


@requires_cirq
class TestCirqRoundTrip:
    """QBCircuit -> cirq -> QBCircuit preserves gate counts/types."""

    def test_roundtrip_bell(self) -> None:
        from qb_compiler.ir.converters.cirq_converter import from_cirq, to_cirq

        original = QBCircuit(n_qubits=2)
        original.add_gate(QBGate(name="h", qubits=(0,)))
        original.add_gate(QBGate(name="cx", qubits=(0, 1)))

        rebuilt = from_cirq(to_cirq(original))
        assert rebuilt.n_qubits == original.n_qubits
        assert [g.name for g in rebuilt.gates] == [g.name for g in original.gates]

    def test_roundtrip_parametrized(self) -> None:
        from qb_compiler.ir.converters.cirq_converter import from_cirq, to_cirq

        original = QBCircuit(n_qubits=2)
        original.add_gate(QBGate(name="rx", qubits=(0,), params=(math.pi / 5,)))
        original.add_gate(QBGate(name="rz", qubits=(1,), params=(0.83,)))
        original.add_gate(QBGate(name="cz", qubits=(0, 1)))

        rebuilt = from_cirq(to_cirq(original))
        assert [g.name for g in rebuilt.gates] == ["rx", "rz", "cz"]
        by_name = {g.name: g for g in rebuilt.gates}
        assert by_name["rx"].params[0] == pytest.approx(math.pi / 5)
        assert by_name["rz"].params[0] == pytest.approx(0.83)
