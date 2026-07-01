"""Tests for the PennyLane converter.

PennyLane is an optional dependency and is not installed in the base
environment, so the conversion tests are gated behind ``importorskip``.
The ImportError-hint test runs without PennyLane.
"""

from __future__ import annotations

import math

import pytest

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate


def test_from_pennylane_missing_sdk_raises_importerror() -> None:
    """Without PennyLane installed, the converter raises a helpful ImportError.

    Skips cleanly if PennyLane happens to be present in the environment.
    """
    try:
        import pennylane  # noqa: F401

        pytest.skip("pennylane is installed; ImportError path not exercised")
    except ImportError:
        pass

    from qb_compiler.ir.converters.pennylane_converter import from_pennylane

    with pytest.raises(ImportError, match=r"qb-compiler\[pennylane\]"):
        from_pennylane([])


class TestFromPennyLane:
    """pennylane tape/ops -> QBCircuit (requires pennylane)."""

    def test_bell_from_ops_list(self) -> None:
        qml = pytest.importorskip("pennylane")
        from qb_compiler.ir.converters.pennylane_converter import from_pennylane

        ops = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])]
        circ = from_pennylane(ops)
        assert circ.n_qubits == 2
        assert [g.name for g in circ.gates] == ["h", "cx"]

    def test_parametrized_from_tape(self) -> None:
        qml = pytest.importorskip("pennylane")
        from qb_compiler.ir.converters.pennylane_converter import from_pennylane

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.7, wires=0)
            qml.RZ(1.2, wires=1)
            qml.CZ(wires=[0, 1])

        circ = from_pennylane(tape)
        by_name = {g.name: g for g in circ.gates}
        assert set(by_name) == {"rx", "rz", "cz"}
        assert by_name["rx"].params[0] == pytest.approx(0.7)
        assert by_name["rz"].params[0] == pytest.approx(1.2)


class TestToPennyLane:
    """QBCircuit -> pennylane tape (requires pennylane)."""

    def test_roundtrip(self) -> None:
        pytest.importorskip("pennylane")
        from qb_compiler.ir.converters.pennylane_converter import (
            from_pennylane,
            to_pennylane,
        )

        original = QBCircuit(n_qubits=2)
        original.add_gate(QBGate(name="h", qubits=(0,)))
        original.add_gate(QBGate(name="rx", qubits=(1,), params=(math.pi / 4,)))
        original.add_gate(QBGate(name="cx", qubits=(0, 1)))

        tape = to_pennylane(original)
        rebuilt = from_pennylane(tape)
        assert [g.name for g in rebuilt.gates] == ["h", "rx", "cx"]
        by_name = {g.name: g for g in rebuilt.gates}
        assert by_name["rx"].params[0] == pytest.approx(math.pi / 4)
