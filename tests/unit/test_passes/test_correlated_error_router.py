"""Tests for the correlated error router pass."""

from __future__ import annotations

import pytest

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.correlated_error_router import CorrelatedErrorRouter


def _linear_chain(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _count_cx(circuit: QBCircuit) -> int:
    return sum(1 for g in circuit.gates if g.name == "cx")


class TestCorrelatedErrorRouter:
    """Tests for CorrelatedErrorRouter."""

    def test_adjacent_no_swaps(self) -> None:
        """Adjacent qubits should not require SWAPs."""
        coupling = [(0, 1), (1, 2)]
        router = CorrelatedErrorRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 0
        assert result.modified is False

    def test_non_adjacent_inserts_swaps(self) -> None:
        """Non-adjacent qubits should trigger SWAP insertion."""
        coupling = _linear_chain(3)
        router = CorrelatedErrorRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 2)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 1
        assert result.modified is True
        assert _count_cx(result.circuit) == 4

    def test_avoids_correlated_path(self) -> None:
        """Router should avoid high-correlation edges.

        Topology:
            0 --- 1 --- 4
            |           |
            2 --- 3 ----+

        High correlation on 0-1 and 1-4 edges.
        Low correlation on 0-2, 2-3, 3-4 edges.
        """
        coupling = [(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)]
        correlations = {
            (0, 1): 0.9,
            (1, 4): 0.9,
            (0, 2): 0.01,
            (2, 3): 0.01,
            (3, 4): 0.01,
        }
        router = CorrelatedErrorRouter(
            coupling_map=coupling,
            correlation_matrix=correlations,
            correlation_weight=5.0,
        )

        circ = QBCircuit(n_qubits=5)
        circ.add_gate(QBGate(name="cx", qubits=(0, 4)))

        ctx: dict = {}
        router.run(circ, ctx)

        # Should take the longer path 0->2->3->4 (2 SWAPs) to avoid
        # the high-correlation short path 0->1->4 (1 SWAP).
        assert ctx["swaps_inserted"] == 2
        assert ctx["correlation_aware"] is True

    def test_no_correlation_data_falls_back(self) -> None:
        """Without correlation data, should behave like standard router."""
        coupling = _linear_chain(3)
        router = CorrelatedErrorRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 2)))

        ctx: dict = {}
        router.run(circ, ctx)

        assert ctx["correlation_aware"] is False
        assert ctx["swaps_inserted"] == 1

    def test_single_qubit_gates_pass_through(self) -> None:
        """Single-qubit gates should not be affected by routing."""
        coupling = _linear_chain(3)
        router = CorrelatedErrorRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(2,)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 0
        assert result.circuit.gate_count == 2

    def test_no_path_raises(self) -> None:
        """Disconnected graph should raise ValueError."""
        coupling = [(0, 1), (2, 3)]
        router = CorrelatedErrorRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=4)
        circ.add_gate(QBGate(name="cx", qubits=(0, 3)))

        ctx: dict = {}
        with pytest.raises(ValueError, match="No path"):
            router.run(circ, ctx)
