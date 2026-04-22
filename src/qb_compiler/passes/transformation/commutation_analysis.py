"""Commutation-based gate optimisation pass.

Identifies and exploits gate commutativity to merge consecutive gates.
Currently handles:

- **RZ + RZ merge**: ``RZ(a)`` followed by ``RZ(b)`` on the same qubit
  collapses to ``RZ(a + b)``.  Applies to all same-axis rotation gates
  (``rz``, ``rx``, ``ry``, ``p``, ``u1``, ``rzz``, ``rxx``, ``ryy``).
- **Identity elimination**: merged rotations that evaluate to angle 0
  (within tolerance) are removed entirely.
"""

from __future__ import annotations

import math

from qb_compiler.ir.circuit import Operation, QBCircuit
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass

# Gates whose consecutive pairs on the same qubits can be merged by
# summing their parameters.
_MERGEABLE_ROTATIONS: frozenset[str] = frozenset(
    {"rz", "rx", "ry", "p", "u1", "rzz", "rxx", "ryy", "crz", "crx", "cry", "cp"}
)

# Tolerance for treating a rotation angle as zero.
_ZERO_TOL = 1e-10


class CommutationOptimizer(TransformationPass):
    """Merge consecutive same-axis rotation gates.

    Scans the operation list and merges adjacent rotation gates on the
    same qubits by summing their parameters.  If the resulting rotation
    angle is effectively zero, the gate is eliminated entirely.

    Barriers and measurements break adjacency.
    """

    @property
    def name(self) -> str:
        return "commutation_optimizer"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        ops = list(circuit.iter_ops())
        result_ops: list[Operation] = []
        merges = 0
        eliminations = 0
        i = 0

        while i < len(ops):
            op = ops[i]

            if (
                isinstance(op, QBGate)
                and op.name in _MERGEABLE_ROTATIONS
                and op.condition is None
                and len(op.params) > 0
            ):
                # Look ahead for the next gate on the same qubits with
                # the same rotation type.
                j = i + 1
                found_merge = False

                while j < len(ops):
                    next_op = ops[j]

                    if isinstance(next_op, (QBBarrier, QBMeasure)):
                        break

                    if isinstance(next_op, QBGate) and set(next_op.qubits) & set(op.qubits):
                        if (
                            next_op.name == op.name
                            and next_op.qubits == op.qubits
                            and next_op.condition is None
                            and len(next_op.params) == len(op.params)
                        ):
                            # Merge: sum all parameters
                            merged_params = tuple(
                                a + b for a, b in zip(op.params, next_op.params, strict=True)
                            )

                            # Keep any ops between i and j
                            for k in range(i + 1, j):
                                result_ops.append(ops[k])

                            # Check if result is effectively zero
                            if all(
                                abs(p % (2 * math.pi)) < _ZERO_TOL
                                or abs(p % (2 * math.pi) - 2 * math.pi) < _ZERO_TOL
                                for p in merged_params
                            ):
                                eliminations += 1
                            else:
                                result_ops.append(
                                    QBGate(
                                        name=op.name,
                                        qubits=op.qubits,
                                        params=merged_params,
                                    )
                                )

                            merges += 1
                            i = j + 1
                            found_merge = True
                            break
                        else:
                            # Different gate on overlapping qubits — stop
                            break

                    j += 1

                if found_merge:
                    continue

            result_ops.append(op)
            i += 1

        total_changes = merges
        if total_changes == 0:
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
            metadata={
                "rotations_merged": merges,
                "rotations_eliminated": eliminations,
            },
            modified=True,
        )
