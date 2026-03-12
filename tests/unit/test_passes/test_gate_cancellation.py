"""Tests for the gate cancellation pass."""

from __future__ import annotations

import pytest

from qb_compiler.compiler import QBCircuit, GateOp, _GateCancellationPass
from qb_compiler.config import CompilerConfig


class TestGateCancellation:
    """Tests for _GateCancellationPass."""

    def _run_pass(self, circ: QBCircuit) -> QBCircuit:
        """Helper: run gate cancellation with default config."""
        config = CompilerConfig(optimization_level=1)
        return _GateCancellationPass().run(circ, config)

    def test_cancel_adjacent_hadamards(self) -> None:
        """Two adjacent H gates on the same qubit should cancel."""
        circ = QBCircuit(2)
        circ.h(0)
        circ.h(0)
        circ.cx(0, 1)

        result = self._run_pass(circ)

        names = [op.name for op in result.ops]
        assert names == ["cx"]

    def test_cancel_adjacent_cx(self) -> None:
        """Two adjacent CX gates on the same qubit pair should cancel."""
        circ = QBCircuit(2)
        circ.cx(0, 1)
        circ.cx(0, 1)

        result = self._run_pass(circ)

        assert result.gate_count == 0

    def test_no_cancel_different_gates(self) -> None:
        """H followed by X should NOT cancel (different gates)."""
        circ = QBCircuit(2)
        circ.h(0)
        circ.x(0)
        circ.cx(0, 1)

        result = self._run_pass(circ)

        names = [op.name for op in result.ops]
        assert names == ["h", "x", "cx"]

    def test_empty_circuit_unchanged(self) -> None:
        """Running the pass on a circuit with no gates returns empty ops."""
        circ = QBCircuit(2)

        result = self._run_pass(circ)

        assert result.gate_count == 0

    def test_no_cancel_different_qubits(self) -> None:
        """H on q0 then H on q1 should NOT cancel."""
        circ = QBCircuit(2)
        circ.h(0)
        circ.h(1)

        result = self._run_pass(circ)

        assert result.gate_count == 2

    def test_cancel_multiple_pairs(self) -> None:
        """Multiple cancellable pairs should all be removed."""
        circ = QBCircuit(2)
        circ.h(0)
        circ.h(0)
        circ.x(1)
        circ.x(1)

        result = self._run_pass(circ)

        assert result.gate_count == 0

    def test_non_self_inverse_not_cancelled(self) -> None:
        """rz-rz (not self-inverse) should NOT be cancelled."""
        circ = QBCircuit(1)
        circ.rz(0, 0.5)
        circ.rz(0, 0.5)

        result = self._run_pass(circ)

        # rz is not in the _SELF_INVERSE set, so both remain
        assert result.gate_count == 2
