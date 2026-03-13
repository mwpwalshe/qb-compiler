"""As-Late-As-Possible (ALAP) scheduler — no noise awareness.

Delays each gate as late as possible while respecting data dependencies.
This is the simpler counterpart of :class:`NoiseAwareScheduler` for use
when no calibration data is available.
"""

from __future__ import annotations

from qb_compiler.ir.circuit import Operation, QBCircuit
from qb_compiler.ir.dag import QBDag
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass


class ALAPScheduler(TransformationPass):
    """Schedule gates as late as possible while respecting dependencies.

    Operations are partitioned into layers via the DAG and emitted in
    reverse-dependency order — each gate is placed in the latest possible
    layer.  The output preserves correctness (topological order) while
    pushing operations toward the end of the circuit.
    """

    @property
    def name(self) -> str:
        return "alap_scheduler"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        if circuit.gate_count == 0:
            return PassResult(circuit=circuit, modified=False)

        dag = QBDag.from_circuit(circuit)
        layers = dag.layers()

        # ALAP: emit layers in order but reverse intra-layer ordering
        # so that within each layer, operations that could move later do.
        # The DAG layers already represent the earliest possible schedule;
        # for a true ALAP we reverse the topological order and assign
        # each op to its latest valid layer.
        scheduled_ops: list[Operation] = []

        # Compute ALAP layer assignment using reverse traversal
        inner_dag = dag.dag
        node_ids = list(inner_dag.node_indices())

        if not node_ids:
            return PassResult(circuit=circuit, modified=False)

        # Find the maximum depth from forward layering
        max_depth = len(layers) - 1

        # For ALAP: compute latest start for each node
        # latest[n] = max_depth - longest_path_from_n_to_any_sink
        # We compute this by reverse BFS from sinks
        out_degree: dict[int, int] = {}
        for nid in node_ids:
            out_degree[nid] = len(list(inner_dag.successor_indices(nid)))

        sinks = [nid for nid in node_ids if out_degree[nid] == 0]

        # Reverse BFS to compute distance from sinks
        dist_to_sink: dict[int, int] = {}
        queue = list(sinks)
        for nid in queue:
            dist_to_sink[nid] = 0

        visited = set(queue)
        head = 0
        while head < len(queue):
            nid = queue[head]
            head += 1
            for pid in inner_dag.predecessor_indices(nid):
                new_d = dist_to_sink[nid] + 1
                dist_to_sink[pid] = max(dist_to_sink.get(pid, 0), new_d)
                if pid not in visited:
                    all_succ_done = all(
                        s in visited for s in inner_dag.successor_indices(pid)
                    )
                    if all_succ_done:
                        visited.add(pid)
                        queue.append(pid)

        # ALAP layer = max_depth - dist_to_sink
        alap_layers: list[list[Operation]] = [[] for _ in range(max_depth + 1)]
        for nid in node_ids:
            d = dist_to_sink.get(nid, 0)
            alap_layer = max_depth - d
            alap_layers[alap_layer].append(inner_dag[nid])

        for layer in alap_layers:
            scheduled_ops.extend(layer)

        new_circuit = QBCircuit(
            n_qubits=circuit.n_qubits,
            n_clbits=circuit.n_clbits,
            name=circuit.name,
        )
        for op in scheduled_ops:
            if isinstance(op, QBGate):
                new_circuit.add_gate(op)
            elif isinstance(op, QBMeasure):
                new_circuit.add_measurement(op.qubit, op.clbit)
            elif isinstance(op, QBBarrier):
                new_circuit.add_barrier(op.qubits)

        depth_before = circuit.depth
        depth_after = new_circuit.depth

        context["alap_depth_before"] = depth_before
        context["alap_depth_after"] = depth_after

        modified = new_circuit != circuit
        return PassResult(
            circuit=new_circuit,
            metadata={
                "depth_before": depth_before,
                "depth_after": depth_after,
            },
            modified=modified,
        )
