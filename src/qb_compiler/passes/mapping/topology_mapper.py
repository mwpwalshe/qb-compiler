"""Topology-only qubit mapping pass.

Maps logical qubits to physical qubits using only the coupling map
(no calibration data).  Uses ``rustworkx.vf2_mapping`` for subgraph
isomorphism and picks the first valid layout.  This is a simpler
fallback for when calibration data is not available.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import rustworkx as rx

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class TopologyMapper(TransformationPass):
    """Map logical qubits to physical qubits using coupling map only.

    Unlike :class:`CalibrationMapper`, this pass does not use gate error
    rates or coherence data.  It picks the first valid VF2 subgraph
    isomorphism, or falls back to a trivial identity mapping if no 2-qubit
    interactions exist or VF2 finds no match.

    Parameters
    ----------
    coupling_map :
        Edges ``[(q0, q1), ...]`` describing physical connectivity.
        Treated as undirected.
    max_candidates :
        Maximum number of VF2 candidate mappings to evaluate.
    """

    def __init__(
        self,
        coupling_map: Sequence[tuple[int, int]],
        max_candidates: int = 100,
    ) -> None:
        self._coupling_map = list(coupling_map)
        self._max_candidates = max_candidates

        # Derive n_physical from coupling map
        all_qubits: set[int] = set()
        for q0, q1 in self._coupling_map:
            all_qubits.add(q0)
            all_qubits.add(q1)
        self._n_physical = max(all_qubits) + 1 if all_qubits else 0

    @property
    def name(self) -> str:
        return "topology_mapper"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        n_logical = circuit.n_qubits

        if n_logical > self._n_physical:
            raise ValueError(
                f"Circuit requires {n_logical} qubits but coupling map "
                f"only covers {self._n_physical} physical qubits"
            )

        interactions = self._extract_interactions(circuit)

        if not interactions:
            # No 2Q gates — use identity mapping
            layout = {i: i for i in range(n_logical)}
        else:
            layout = self._find_layout(circuit, interactions)

        mapped_circuit = self._apply_layout(circuit, layout)

        context["initial_layout"] = dict(layout)

        return PassResult(
            circuit=mapped_circuit,
            metadata={"initial_layout": dict(layout)},
            modified=True,
        )

    @staticmethod
    def _extract_interactions(circuit: QBCircuit) -> dict[tuple[int, int], int]:
        interactions: dict[tuple[int, int], int] = defaultdict(int)
        for op in circuit.iter_ops():
            if isinstance(op, QBGate) and op.num_qubits >= 2:
                q_sorted = sorted(op.qubits[:2])
                interactions[(q_sorted[0], q_sorted[1])] += 1
        return dict(interactions)

    def _find_layout(
        self,
        circuit: QBCircuit,
        interactions: dict[tuple[int, int], int],
    ) -> dict[int, int]:
        # Build logical interaction graph
        logical_graph = rx.PyGraph()
        logical_nodes: dict[int, int] = {}
        for log_a, log_b in interactions:
            for q in (log_a, log_b):
                if q not in logical_nodes:
                    logical_nodes[q] = logical_graph.add_node(q)
            logical_graph.add_edge(logical_nodes[log_a], logical_nodes[log_b], None)

        # Build physical coupling graph
        physical_graph = rx.PyGraph()
        physical_nodes: dict[int, int] = {}
        seen_edges: set[tuple[int, int]] = set()
        for q0, q1 in self._coupling_map:
            for q in (q0, q1):
                if q not in physical_nodes:
                    physical_nodes[q] = physical_graph.add_node(q)
            edge_key = (min(q0, q1), max(q0, q1))
            if edge_key not in seen_edges:
                physical_graph.add_edge(physical_nodes[q0], physical_nodes[q1], None)
                seen_edges.add(edge_key)

        phys_node_to_qubit = {nid: q for q, nid in physical_nodes.items()}
        log_node_to_qubit = {nid: q for q, nid in logical_nodes.items()}

        vf2_iter = rx.vf2_mapping(
            physical_graph,
            logical_graph,
            subgraph=True,
            induced=False,
            id_order=False,
            call_limit=10_000,
        )

        best_layout: dict[int, int] | None = None
        n_evaluated = 0

        for mapping in vf2_iter:
            if n_evaluated >= self._max_candidates:
                break
            layout: dict[int, int] = {}
            for phys_node, log_node in mapping.items():
                layout[log_node_to_qubit[log_node]] = phys_node_to_qubit[phys_node]
            best_layout = layout
            break  # Take the first valid mapping

        if best_layout is None:
            logger.warning("TopologyMapper: VF2 found no mapping, using identity")
            best_layout = {i: i for i in range(circuit.n_qubits)}
        else:
            # Fill in unmapped logical qubits
            used = set(best_layout.values())
            for q in range(circuit.n_qubits):
                if q not in best_layout:
                    for p in range(self._n_physical):
                        if p not in used:
                            best_layout[q] = p
                            used.add(p)
                            break

        return best_layout

    @staticmethod
    def _apply_layout(circuit: QBCircuit, layout: dict[int, int]) -> QBCircuit:
        max_phys = max(layout.values()) if layout else 0
        new_n_qubits = max(max_phys + 1, circuit.n_qubits)

        mapped = QBCircuit(
            n_qubits=new_n_qubits,
            n_clbits=circuit.n_clbits,
            name=circuit.name,
        )

        for op in circuit.iter_ops():
            if isinstance(op, QBGate):
                new_qubits = tuple(layout[q] for q in op.qubits)
                mapped.add_gate(
                    QBGate(
                        name=op.name,
                        qubits=new_qubits,
                        params=op.params,
                        condition=op.condition,
                    )
                )
            elif isinstance(op, QBMeasure):
                mapped.add_measurement(layout[op.qubit], op.clbit)
            elif isinstance(op, QBBarrier):
                mapped.add_barrier(tuple(layout[q] for q in op.qubits))

        return mapped
