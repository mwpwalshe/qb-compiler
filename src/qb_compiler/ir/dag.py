"""DAG (directed acyclic graph) representation of a quantum circuit.

Converts a sequential :class:`QBCircuit` into a dependency graph where edges
represent qubit-level data dependencies.  This is the form most useful for
passes that reorder, parallelise, or cancel operations.

Requires ``rustworkx`` (listed as a core dependency in ``pyproject.toml``).
"""

from __future__ import annotations

from typing import Iterator, Union

import rustworkx as rx

from qb_compiler.ir.circuit import Operation, QBCircuit
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure


class QBDag:
    """DAG wrapper around a :class:`rustworkx.PyDAG`.

    Each node payload is an :class:`Operation` (QBGate | QBMeasure | QBBarrier).
    Edges carry the qubit index that creates the dependency.
    """

    __slots__ = ("_dag", "_n_qubits", "_n_clbits", "_name")

    def __init__(self, n_qubits: int, n_clbits: int = 0, name: str = "") -> None:
        self._dag: rx.PyDAG = rx.PyDAG()
        self._n_qubits = n_qubits
        self._n_clbits = n_clbits
        self._name = name

    # ── properties ────────────────────────────────────────────────────

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_clbits(self) -> int:
        return self._n_clbits

    @property
    def name(self) -> str:
        return self._name

    @property
    def node_count(self) -> int:
        return self._dag.num_nodes()

    @property
    def edge_count(self) -> int:
        return self._dag.num_edges()

    @property
    def dag(self) -> rx.PyDAG:
        """Direct access to the underlying ``rustworkx.PyDAG``."""
        return self._dag

    # ── construction ──────────────────────────────────────────────────

    @classmethod
    def from_circuit(cls, circuit: QBCircuit) -> QBDag:
        """Build a DAG from a :class:`QBCircuit`.

        For every qubit, the most recent node that touched that qubit is
        tracked.  When a new operation arrives, edges are added from all
        predecessors on the involved qubits.
        """
        dag = cls(
            n_qubits=circuit.n_qubits,
            n_clbits=circuit.n_clbits,
            name=circuit.name,
        )

        # last node index that touched each qubit
        last_on_qubit: dict[int, int] = {}

        for op in circuit.iter_ops():
            qubits = _op_qubits(op)
            node_id = dag._dag.add_node(op)

            for q in qubits:
                pred = last_on_qubit.get(q)
                if pred is not None:
                    dag._dag.add_edge(pred, node_id, q)
                last_on_qubit[q] = node_id

        return dag

    # ── linearisation ─────────────────────────────────────────────────

    def to_circuit(self) -> QBCircuit:
        """Linearise the DAG back into a :class:`QBCircuit`.

        Operations are emitted in topological order.
        """
        circ = QBCircuit(
            n_qubits=self._n_qubits,
            n_clbits=self._n_clbits,
            name=self._name,
        )
        for op in self.topological_ops():
            if isinstance(op, QBGate):
                circ.add_gate(op)
            elif isinstance(op, QBMeasure):
                circ.add_measurement(op.qubit, op.clbit)
            elif isinstance(op, QBBarrier):
                circ.add_barrier(op.qubits)
        return circ

    # ── traversal ─────────────────────────────────────────────────────

    def topological_ops(self) -> Iterator[Operation]:
        """Yield operations in a valid topological order."""
        for node_id in rx.topological_sort(self._dag):
            yield self._dag[node_id]

    def layers(self) -> list[list[Operation]]:
        """Partition operations into parallel layers.

        Two operations are in the same layer iff they share no qubit
        dependency.  Computed by assigning each node a depth equal to its
        longest path from any root.
        """
        if self._dag.num_nodes() == 0:
            return []

        # Compute the longest-path depth for each node
        node_ids = list(self._dag.node_indices())
        in_degree: dict[int, int] = {nid: 0 for nid in node_ids}
        for nid in node_ids:
            for pred in self._dag.predecessor_indices(nid):
                in_degree[nid] += 1

        # BFS-style topological layering
        depth_of: dict[int, int] = {}
        queue: list[int] = [nid for nid, deg in in_degree.items() if deg == 0]
        for nid in queue:
            depth_of[nid] = 0

        visited = set(queue)
        head = 0
        while head < len(queue):
            nid = queue[head]
            head += 1
            for sid in self._dag.successor_indices(nid):
                new_d = depth_of[nid] + 1
                depth_of[sid] = max(depth_of.get(sid, 0), new_d)
                if sid not in visited:
                    # Check if all predecessors visited
                    all_pred_done = all(
                        p in visited for p in self._dag.predecessor_indices(sid)
                    )
                    if all_pred_done:
                        visited.add(sid)
                        queue.append(sid)

        # Group by depth
        max_depth = max(depth_of.values(), default=-1)
        result: list[list[Operation]] = [[] for _ in range(max_depth + 1)]
        for nid in node_ids:
            d = depth_of.get(nid, 0)
            result[d].append(self._dag[nid])

        return [layer for layer in result if layer]

    # ── mutation helpers ──────────────────────────────────────────────

    def remove_node(self, node_id: int) -> None:
        """Remove a node and re-wire edges around it.

        Predecessors are connected directly to successors so dependency
        ordering is preserved.
        """
        preds = self._dag.predecessor_indices(node_id)
        succs = self._dag.successor_indices(node_id)
        # Re-wire: connect each predecessor to each successor on shared qubits
        pred_qubits: dict[int, set[int]] = {}
        for pid in preds:
            edge_data = self._dag.get_edge_data(pid, node_id)
            pred_qubits.setdefault(pid, set()).add(edge_data)
        succ_qubits: dict[int, set[int]] = {}
        for sid in succs:
            edge_data = self._dag.get_edge_data(node_id, sid)
            succ_qubits.setdefault(sid, set()).add(edge_data)

        for pid, pqs in pred_qubits.items():
            for sid, sqs in succ_qubits.items():
                shared = pqs & sqs
                for q in shared:
                    # avoid duplicate edges
                    if not self._dag.has_edge(pid, sid):
                        self._dag.add_edge(pid, sid, q)

        self._dag.remove_node(node_id)

    # ── dunder ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"QBDag(qubits={self._n_qubits}, nodes={self.node_count}, "
            f"edges={self.edge_count})"
        )


# ── module-level helpers ──────────────────────────────────────────────


def _op_qubits(op: Operation) -> tuple[int, ...]:
    """Extract the qubit indices from any operation type."""
    if isinstance(op, QBGate):
        return op.qubits
    if isinstance(op, QBMeasure):
        return (op.qubit,)
    if isinstance(op, QBBarrier):
        return op.qubits
    raise TypeError(f"Unknown operation type: {type(op)}")  # pragma: no cover
