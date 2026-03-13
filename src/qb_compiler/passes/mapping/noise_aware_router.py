"""Noise-aware SWAP router.

Inserts SWAP gates to route 2-qubit operations to physically adjacent qubits
on the coupling map, choosing paths that minimise accumulated gate error
rather than simply minimising SWAP count.

Algorithm
---------
1. Build a weighted ``rustworkx.PyGraph`` from the coupling map where each
   edge weight is ``-log(1 - gate_error)``.  Shortest path in this metric
   corresponds to highest-fidelity routing path.
2. Iterate over circuit operations in order.  For each 2-qubit gate:
   a. Look up the *current* physical locations of its logical qubits
      (tracking the permutation induced by prior SWAPs).
   b. If the two physical qubits are adjacent on the coupling graph, emit the
      gate directly.
   c. Otherwise, compute the Dijkstra shortest (lowest-error) path and insert
      SWAPs along the path to bring the qubits together.  Update the
      permutation after each SWAP.
3. Single-qubit gates and measurements are re-mapped through the current
   permutation and emitted unchanged.

Each SWAP decomposes into 3 CX gates, so the error contribution of a single
SWAP on edge ``(a, b)`` is ``3 * cx_error(a, b)``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import rustworkx as rx

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Default 2-qubit gate error used when calibration data is missing for an edge.
_DEFAULT_EDGE_ERROR = 0.01


class NoiseAwareRouter(TransformationPass):
    """Route 2-qubit gates through lowest-error SWAP paths.

    Parameters
    ----------
    coupling_map :
        Edges ``[(q0, q1), ...]`` describing physical connectivity.  Treated
        as undirected.
    gate_errors :
        Mapping ``(q0, q1) -> error_rate`` for the native 2-qubit gate on
        each edge.  Both directions ``(a, b)`` and ``(b, a)`` are accepted
        and merged.  Missing edges default to *default_error*.
    default_error :
        Fallback 2-qubit gate error for edges absent from *gate_errors*.
    """

    def __init__(
        self,
        coupling_map: Sequence[tuple[int, int]],
        gate_errors: dict[tuple[int, int], float] | None = None,
        default_error: float = _DEFAULT_EDGE_ERROR,
    ) -> None:
        self._coupling_map = list(coupling_map)
        self._gate_errors = gate_errors or {}
        self._default_error = default_error

        # Build undirected adjacency set for quick neighbour checks.
        self._adjacency: dict[int, set[int]] = {}
        for q0, q1 in self._coupling_map:
            self._adjacency.setdefault(q0, set()).add(q1)
            self._adjacency.setdefault(q1, set()).add(q0)

        # Build rustworkx weighted graph.
        self._graph, self._node_to_qubit, self._qubit_to_node = (
            self._build_weighted_graph()
        )

    # ── BasePass interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "noise_aware_router"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        """Route *circuit* and return the routed version."""
        # logical -> physical mapping (identity at start)
        log_to_phys: list[int] = list(range(circuit.n_qubits))
        # physical -> logical (inverse)
        phys_to_log: list[int] = list(range(circuit.n_qubits))

        routed = QBCircuit(
            n_qubits=circuit.n_qubits,
            n_clbits=circuit.n_clbits,
            name=circuit.name,
        )

        swaps_inserted = 0
        total_fidelity_cost = 0.0  # sum of -log(1-err) for every SWAP edge

        for op in circuit.iter_ops():
            if isinstance(op, QBBarrier):
                # Remap barrier qubits through current permutation.
                new_qubits = tuple(log_to_phys[q] for q in op.qubits)
                routed.add_barrier(new_qubits)

            elif isinstance(op, QBMeasure):
                phys = log_to_phys[op.qubit]
                routed.add_measurement(phys, op.clbit)

            elif isinstance(op, QBGate):
                if op.num_qubits == 1:
                    phys = log_to_phys[op.qubits[0]]
                    routed.add_gate(
                        QBGate(
                            name=op.name,
                            qubits=(phys,),
                            params=op.params,
                            condition=op.condition,
                        )
                    )
                elif op.num_qubits == 2:
                    phys0 = log_to_phys[op.qubits[0]]
                    phys1 = log_to_phys[op.qubits[1]]

                    if self._are_adjacent(phys0, phys1):
                        # Already adjacent — emit directly.
                        routed.add_gate(
                            QBGate(
                                name=op.name,
                                qubits=(phys0, phys1),
                                params=op.params,
                                condition=op.condition,
                            )
                        )
                    else:
                        # Route through lowest-error path.
                        path = self._shortest_path(phys0, phys1)
                        # path = [phys0, ..., phys1]
                        # Insert SWAPs along path[0:-2] to move phys0 next to phys1.
                        # After SWAPs along (path[0],path[1]), ..., (path[-3],path[-2]),
                        # the logical qubit originally at phys0 ends up at path[-2],
                        # which is adjacent to path[-1] == phys1.
                        for i in range(len(path) - 2):
                            a, b = path[i], path[i + 1]
                            self._emit_swap(routed, a, b)
                            swaps_inserted += 1
                            total_fidelity_cost += self._edge_weight(a, b) * 3

                            # Update permutation.
                            log_a = phys_to_log[a]
                            log_b = phys_to_log[b]
                            # Swap physical assignments.
                            log_to_phys[log_a] = b
                            log_to_phys[log_b] = a
                            phys_to_log[a] = log_b
                            phys_to_log[b] = log_a

                        # Now the logical qubit at op.qubits[0] is at path[-2],
                        # and op.qubits[1] is still at phys1 = path[-1].
                        new_phys0 = log_to_phys[op.qubits[0]]
                        new_phys1 = log_to_phys[op.qubits[1]]
                        routed.add_gate(
                            QBGate(
                                name=op.name,
                                qubits=(new_phys0, new_phys1),
                                params=op.params,
                                condition=op.condition,
                            )
                        )
                else:
                    # 3+ qubit gates: pass through with remapped qubits.
                    new_qubits = tuple(log_to_phys[q] for q in op.qubits)
                    routed.add_gate(
                        QBGate(
                            name=op.name,
                            qubits=new_qubits,
                            params=op.params,
                            condition=op.condition,
                        )
                    )

        # Populate context (property_set).
        context["swaps_inserted"] = swaps_inserted
        context["routing_fidelity_cost"] = total_fidelity_cost

        modified = swaps_inserted > 0
        return PassResult(
            circuit=routed,
            metadata={
                "swaps_inserted": swaps_inserted,
                "routing_fidelity_cost": round(total_fidelity_cost, 8),
            },
            modified=modified,
        )

    # ── internal helpers ──────────────────────────────────────────────

    def _build_weighted_graph(
        self,
    ) -> tuple[rx.PyGraph, dict[int, int], dict[int, int]]:
        """Build an undirected weighted graph for Dijkstra routing.

        Returns (graph, node_id->qubit, qubit->node_id).
        """
        graph = rx.PyGraph()
        qubit_to_node: dict[int, int] = {}
        node_to_qubit: dict[int, int] = {}

        # Collect all qubit indices that appear in the coupling map.
        all_qubits: set[int] = set()
        for q0, q1 in self._coupling_map:
            all_qubits.add(q0)
            all_qubits.add(q1)

        for q in sorted(all_qubits):
            nid = graph.add_node(q)
            qubit_to_node[q] = nid
            node_to_qubit[nid] = q

        # Add edges (undirected — deduplicate).
        seen_edges: set[tuple[int, int]] = set()
        for q0, q1 in self._coupling_map:
            key = (min(q0, q1), max(q0, q1))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            w = self._edge_weight(q0, q1)
            graph.add_edge(qubit_to_node[q0], qubit_to_node[q1], w)

        return graph, node_to_qubit, qubit_to_node

    def _edge_weight(self, q0: int, q1: int) -> float:
        """Return ``-log(1 - error)`` for the edge ``(q0, q1)``."""
        err = self._gate_errors.get((q0, q1))
        if err is None:
            err = self._gate_errors.get((q1, q0))
        if err is None:
            err = self._default_error
        # Clamp to avoid log(0) or negative weights.
        err = max(min(err, 0.999), 1e-12)
        return -math.log(1.0 - err)

    def _are_adjacent(self, q0: int, q1: int) -> bool:
        return q1 in self._adjacency.get(q0, set())

    def _shortest_path(self, source: int, target: int) -> list[int]:
        """Return the list of physical qubits on the lowest-error path.

        Uses Dijkstra on the weighted coupling graph.
        """
        src_node = self._qubit_to_node[source]
        tgt_node = self._qubit_to_node[target]

        # rx.dijkstra_shortest_paths returns {target_node: [node_path]}
        paths = rx.dijkstra_shortest_paths(
            self._graph, src_node, target=tgt_node, weight_fn=float
        )

        if tgt_node not in paths:
            raise ValueError(
                f"No path between physical qubit {source} and {target} "
                f"in the coupling graph"
            )

        node_path = paths[tgt_node]
        return [self._node_to_qubit[nid] for nid in node_path]

    @staticmethod
    def _emit_swap(circuit: QBCircuit, q0: int, q1: int) -> None:
        """Decompose a SWAP into 3 CX gates and append to *circuit*."""
        circuit.add_gate(QBGate(name="cx", qubits=(q0, q1)))
        circuit.add_gate(QBGate(name="cx", qubits=(q1, q0)))
        circuit.add_gate(QBGate(name="cx", qubits=(q0, q1)))
