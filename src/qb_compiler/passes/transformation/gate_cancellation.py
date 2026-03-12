"""Gate cancellation transformation pass.

Cancels adjacent pairs of self-inverse gates: H-H, X-X, Y-Y, Z-Z, CX-CX
(same qubits), CZ-CZ, SWAP-SWAP, etc.
"""

from __future__ import annotations

from qb_compiler.ir.circuit import Operation, QBCircuit
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass

# Gates that are their own inverse (G * G = I)
_SELF_INVERSE: frozenset[str] = frozenset(
    {"h", "x", "y", "z", "cx", "cy", "cz", "swap", "id", "ecr", "ccx", "cswap"}
)


class GateCancellationPass(TransformationPass):
    """Cancel adjacent pairs of self-inverse gates.

    Scans the operation list and removes consecutive identical gate
    applications (same name, same qubits, no parameters or identical
    parameters) when the gate is known to be self-inverse.

    Barriers and measurements break adjacency.
    """

    @property
    def name(self) -> str:
        return "gate_cancellation"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        ops = list(circuit.iter_ops())
        result_ops: list[Operation] = []
        cancelled = 0
        i = 0

        while i < len(ops):
            op = ops[i]

            # Only try to cancel QBGate ops
            if isinstance(op, QBGate) and op.name in _SELF_INVERSE:
                # Look ahead for the next gate on the same qubits
                j = i + 1
                found_cancel = False

                while j < len(ops):
                    next_op = ops[j]

                    if isinstance(next_op, QBBarrier) or isinstance(next_op, QBMeasure):
                        # Barriers and measurements break adjacency
                        break

                    if isinstance(next_op, QBGate):
                        # Check if qubits overlap
                        if set(next_op.qubits) & set(op.qubits):
                            # Same gate on same qubits? Cancel both.
                            if (
                                next_op.name == op.name
                                and next_op.qubits == op.qubits
                                and next_op.params == op.params
                                and next_op.condition is None
                                and op.condition is None
                            ):
                                # Cancel: skip both i and j, but keep
                                # anything between them
                                for k in range(i + 1, j):
                                    result_ops.append(ops[k])
                                cancelled += 2
                                i = j + 1
                                found_cancel = True
                                break
                            else:
                                # Different gate on overlapping qubits — stop
                                break
                        # else: non-overlapping gate, keep scanning
                    j += 1

                if found_cancel:
                    continue

            result_ops.append(op)
            i += 1

        if cancelled == 0:
            return PassResult(circuit=circuit, modified=False)

        new_circuit = QBCircuit(
            n_qubits=circuit.n_qubits,
            n_clbits=circuit.n_clbits,
            name=circuit.name,
        )
        for op in result_ops:
            if isinstance(op, QBGate):
                new_circuit.add_gate(op)
            elif isinstance(op, QBMeasure):
                new_circuit.add_measurement(op.qubit, op.clbit)
            elif isinstance(op, QBBarrier):
                new_circuit.add_barrier(op.qubits)

        return PassResult(
            circuit=new_circuit,
            metadata={"gates_cancelled": cancelled},
            modified=True,
        )
