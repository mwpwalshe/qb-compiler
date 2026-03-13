"""As-Soon-As-Possible (ASAP) scheduler.

Schedules each gate as early as its dependencies allow, useful for
minimising overall circuit depth.
"""

from __future__ import annotations

from qb_compiler.ir.circuit import Operation, QBCircuit
from qb_compiler.ir.dag import QBDag
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass


class ASAPScheduler(TransformationPass):
    """Schedule gates as soon as possible while respecting dependencies.

    Operations are partitioned into layers via the DAG; each gate is placed
    in the earliest layer allowed by its data dependencies.  This minimises
    circuit depth.
    """

    @property
    def name(self) -> str:
        return "asap_scheduler"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        if circuit.gate_count == 0:
            return PassResult(circuit=circuit, modified=False)

        dag = QBDag.from_circuit(circuit)
        layers = dag.layers()

        # ASAP: the DAG layers() method already computes the earliest-
        # possible layer for each operation, so we just linearise them.
        scheduled_ops: list[Operation] = []
        for layer in layers:
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

        context["asap_depth_before"] = depth_before
        context["asap_depth_after"] = depth_after

        modified = new_circuit != circuit
        return PassResult(
            circuit=new_circuit,
            metadata={
                "depth_before": depth_before,
                "depth_after": depth_after,
            },
            modified=modified,
        )
