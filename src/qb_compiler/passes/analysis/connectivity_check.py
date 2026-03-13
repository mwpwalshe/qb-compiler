"""Connectivity check analysis pass.

Checks whether all 2-qubit gates in the circuit operate on physically
connected qubits.  Populates ``context["connectivity_satisfied"]`` and
``context["violating_gates"]`` so downstream passes can decide whether
routing is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.base import AnalysisPass

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qb_compiler.ir.circuit import QBCircuit


class ConnectivityCheck(AnalysisPass):
    """Check if all 2-qubit gates use connected qubits.

    Parameters
    ----------
    coupling_map :
        Edges ``[(q0, q1), ...]`` describing physical connectivity.
        Treated as undirected.
    """

    def __init__(self, coupling_map: Sequence[tuple[int, int]]) -> None:
        self._adjacency: set[tuple[int, int]] = set()
        for q0, q1 in coupling_map:
            self._adjacency.add((min(q0, q1), max(q0, q1)))

    @property
    def name(self) -> str:
        return "connectivity_check"

    def analyze(self, circuit: QBCircuit, context: dict) -> None:
        violating: list[int] = []
        gate_index = 0

        for op in circuit.iter_ops():
            if isinstance(op, QBGate):
                if op.num_qubits >= 2:
                    q0, q1 = op.qubits[0], op.qubits[1]
                    edge = (min(q0, q1), max(q0, q1))
                    if edge not in self._adjacency:
                        violating.append(gate_index)
                gate_index += 1

        context["connectivity_satisfied"] = len(violating) == 0
        context["violating_gates"] = violating
