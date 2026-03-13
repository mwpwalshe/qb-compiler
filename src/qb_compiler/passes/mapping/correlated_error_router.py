"""Correlated-error-aware SWAP router.

Extends the standard noise-aware routing strategy by penalising qubit pairs
that exhibit high temporal error correlation (as measured by SafetyGate /
QubitBoost).  When correlation data is unavailable, falls back to standard
error-weighted routing.
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

_DEFAULT_EDGE_ERROR = 0.01


class CorrelatedErrorRouter(TransformationPass):
    """Route 2-qubit gates while avoiding temporally correlated qubit pairs.

    Parameters
    ----------
    coupling_map :
        Edges ``[(q0, q1), ...]`` describing physical connectivity.
    gate_errors :
        Mapping ``(q0, q1) -> error_rate`` for 2-qubit gates.
    correlation_matrix :
        Mapping ``(q0, q1) -> correlation_score`` (0-1) indicating
        temporal error correlation between qubit pairs.  Higher values
        mean the pair is more likely to experience correlated errors.
        If ``None``, no correlation penalty is applied.
    correlation_weight :
        How much to weight correlation data relative to gate error.
    correlation_threshold :
        Correlation values below this threshold are ignored.
    default_error :
        Fallback gate error for edges not in *gate_errors*.
    """

    def __init__(
        self,
        coupling_map: Sequence[tuple[int, int]],
        gate_errors: dict[tuple[int, int], float] | None = None,
        correlation_matrix: dict[tuple[int, int], float] | None = None,
        correlation_weight: float = 1.0,
        correlation_threshold: float = 0.1,
        default_error: float = _DEFAULT_EDGE_ERROR,
    ) -> None:
        self._coupling_map = list(coupling_map)
        self._gate_errors = gate_errors or {}
        self._correlations = correlation_matrix or {}
        self._correlation_weight = correlation_weight
        self._correlation_threshold = correlation_threshold
        self._default_error = default_error

        # Build adjacency set
        self._adjacency: dict[int, set[int]] = {}
        for q0, q1 in self._coupling_map:
            self._adjacency.setdefault(q0, set()).add(q1)
            self._adjacency.setdefault(q1, set()).add(q0)

        self._graph, self._node_to_qubit, self._qubit_to_node = (
            self._build_weighted_graph()
        )

    @property
    def name(self) -> str:
        return "correlated_error_router"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        log_to_phys = list(range(circuit.n_qubits))
        phys_to_log = list(range(circuit.n_qubits))

        routed = QBCircuit(
            n_qubits=circuit.n_qubits,
            n_clbits=circuit.n_clbits,
            name=circuit.name,
        )

        swaps_inserted = 0
        total_fidelity_cost = 0.0

        for op in circuit.iter_ops():
            if isinstance(op, QBBarrier):
                new_qubits = tuple(log_to_phys[q] for q in op.qubits)
                routed.add_barrier(new_qubits)

            elif isinstance(op, QBMeasure):
                routed.add_measurement(log_to_phys[op.qubit], op.clbit)

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
                        routed.add_gate(
                            QBGate(
                                name=op.name,
                                qubits=(phys0, phys1),
                                params=op.params,
                                condition=op.condition,
                            )
                        )
                    else:
                        path = self._shortest_path(phys0, phys1)
                        for idx in range(len(path) - 2):
                            a, b = path[idx], path[idx + 1]
                            self._emit_swap(routed, a, b)
                            swaps_inserted += 1
                            total_fidelity_cost += self._edge_weight(a, b) * 3

                            log_a = phys_to_log[a]
                            log_b = phys_to_log[b]
                            log_to_phys[log_a] = b
                            log_to_phys[log_b] = a
                            phys_to_log[a] = log_b
                            phys_to_log[b] = log_a

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
                    new_qubits = tuple(log_to_phys[q] for q in op.qubits)
                    routed.add_gate(
                        QBGate(
                            name=op.name,
                            qubits=new_qubits,
                            params=op.params,
                            condition=op.condition,
                        )
                    )

        context["swaps_inserted"] = swaps_inserted
        context["routing_fidelity_cost"] = total_fidelity_cost
        context["correlation_aware"] = bool(self._correlations)

        modified = swaps_inserted > 0
        return PassResult(
            circuit=routed,
            metadata={
                "swaps_inserted": swaps_inserted,
                "routing_fidelity_cost": round(total_fidelity_cost, 8),
                "correlation_aware": bool(self._correlations),
            },
            modified=modified,
        )

    # ── internal helpers ──────────────────────────────────────────────

    def _build_weighted_graph(
        self,
    ) -> tuple[rx.PyGraph, dict[int, int], dict[int, int]]:
        graph = rx.PyGraph()
        qubit_to_node: dict[int, int] = {}
        node_to_qubit: dict[int, int] = {}

        all_qubits: set[int] = set()
        for q0, q1 in self._coupling_map:
            all_qubits.add(q0)
            all_qubits.add(q1)

        for q in sorted(all_qubits):
            nid = graph.add_node(q)
            qubit_to_node[q] = nid
            node_to_qubit[nid] = q

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
        """Compute edge weight combining gate error and correlation penalty."""
        err = self._gate_errors.get((q0, q1))
        if err is None:
            err = self._gate_errors.get((q1, q0))
        if err is None:
            err = self._default_error
        err = max(min(err, 0.999), 1e-12)

        base_weight = -math.log(1.0 - err)

        # Add correlation penalty
        corr = self._correlations.get((q0, q1))
        if corr is None:
            corr = self._correlations.get((q1, q0))
        if corr is not None and corr >= self._correlation_threshold:
            base_weight += self._correlation_weight * corr

        return base_weight

    def _are_adjacent(self, q0: int, q1: int) -> bool:
        return q1 in self._adjacency.get(q0, set())

    def _shortest_path(self, source: int, target: int) -> list[int]:
        src_node = self._qubit_to_node[source]
        tgt_node = self._qubit_to_node[target]

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
        circuit.add_gate(QBGate(name="cx", qubits=(q0, q1)))
        circuit.add_gate(QBGate(name="cx", qubits=(q1, q0)))
        circuit.add_gate(QBGate(name="cx", qubits=(q0, q1)))
