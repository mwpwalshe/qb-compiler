"""Template-based circuit simplification pass.

Applies known circuit identities to reduce gate count:

- **X-X cancellation**: Two consecutive X gates on the same qubit → identity
- **CX-CX cancellation**: Two consecutive CX gates on the same qubit pair → identity
- **H-CX-H simplification**: H(t) CX(c,t) H(t) → CX(t,c) (control/target swap)

Reports how many simplifications were applied.
"""

from __future__ import annotations

from qb_compiler.ir.circuit import Operation, QBCircuit
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass

# Self-inverse gates handled by GateCancellationPass are also caught here
# for completeness, but the primary value-add is the H-CX-H template.


class CircuitSimplifier(TransformationPass):
    """Template-based circuit simplification.

    Applies the following rewrite rules in a single forward scan:

    1. **X-X → identity** (same qubit, no intervening ops on that qubit)
    2. **CX-CX → identity** (same qubit pair)
    3. **H-CX-H → CX with swapped control/target**
       When ``H(t), CX(c, t), H(t)`` appears and no other operation
       touches qubit ``t`` in between, it is replaced by ``CX(t, c)``.
    """

    @property
    def name(self) -> str:
        return "circuit_simplification"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        ops = list(circuit.iter_ops())
        result_ops: list[Operation] = []
        simplifications = 0
        skip: set[int] = set()

        for i, op in enumerate(ops):
            if i in skip:
                continue

            if not isinstance(op, QBGate) or op.condition is not None:
                result_ops.append(op)
                continue

            # Try H-CX-H template: H(t), CX(c,t), H(t) → CX(t,c)
            if op.name == "h" and op.num_qubits == 1:
                matched = self._try_h_cx_h(ops, i, skip)
                if matched is not None:
                    result_ops.append(matched)
                    simplifications += 1
                    continue

            # Try self-inverse cancellation (X-X, CX-CX, etc.)
            if op.name in _CANCEL_PAIRS:
                cancelled = self._try_cancel(ops, i, skip)
                if cancelled:
                    simplifications += 1
                    continue

            result_ops.append(op)

        if simplifications == 0:
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
            metadata={"simplifications_applied": simplifications},
            modified=True,
        )

    # ── template matchers ─────────────────────────────────────────────

    @staticmethod
    def _try_h_cx_h(ops: list[Operation], i: int, skip: set[int]) -> QBGate | None:
        """Try to match H(t), CX(c, t), H(t) starting at index *i*.

        Returns a replacement CX(t, c) gate if matched, else None.
        """
        h1 = ops[i]
        if not isinstance(h1, QBGate) or h1.name != "h" or h1.num_qubits != 1:
            return None

        target = h1.qubits[0]

        # Find the next op that touches target
        j = i + 1
        while j < len(ops):
            if j in skip:
                j += 1
                continue
            nop = ops[j]
            if isinstance(nop, (QBBarrier, QBMeasure)):
                return None
            if isinstance(nop, QBGate) and set(nop.qubits) & {target}:
                break
            j += 1
        else:
            return None

        cx = ops[j]
        if (
            not isinstance(cx, QBGate)
            or cx.name != "cx"
            or cx.num_qubits != 2
            or cx.qubits[1] != target
            or cx.condition is not None
        ):
            return None

        control = cx.qubits[0]

        # Find the next op after cx that touches target
        k = j + 1
        while k < len(ops):
            if k in skip:
                k += 1
                continue
            nop = ops[k]
            if isinstance(nop, (QBBarrier, QBMeasure)):
                return None
            if isinstance(nop, QBGate) and set(nop.qubits) & {target}:
                break
            k += 1
        else:
            return None

        h2 = ops[k]
        if (
            not isinstance(h2, QBGate)
            or h2.name != "h"
            or h2.num_qubits != 1
            or h2.qubits[0] != target
            or h2.condition is not None
        ):
            return None

        # Match! Mark j and k as consumed.
        skip.add(j)
        skip.add(k)
        # i is consumed by the caller (not added to result_ops)
        return QBGate(name="cx", qubits=(target, control))

    @staticmethod
    def _try_cancel(ops: list[Operation], i: int, skip: set[int]) -> bool:
        """Try to cancel ops[i] with the next identical gate on the same qubits.

        Returns True if cancelled (marks both i and the match in *skip*).
        """
        op = ops[i]
        if not isinstance(op, QBGate):
            return False

        target_qubits = set(op.qubits)

        j = i + 1
        while j < len(ops):
            if j in skip:
                j += 1
                continue
            nop = ops[j]
            if isinstance(nop, (QBBarrier, QBMeasure)):
                return False
            if isinstance(nop, QBGate) and set(nop.qubits) & target_qubits:
                if (
                    nop.name == op.name
                    and nop.qubits == op.qubits
                    and nop.params == op.params
                    and nop.condition is None
                ):
                    skip.add(i)
                    skip.add(j)
                    return True
                else:
                    return False
            j += 1

        return False


_CANCEL_PAIRS: frozenset[str] = frozenset({"x", "y", "z", "h", "cx", "cz", "swap", "cy"})
