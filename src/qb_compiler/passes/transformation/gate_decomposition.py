"""Gate decomposition transformation pass.

Decomposes gates into a target native basis gate set, applying known
decomposition rules for standard gates.
"""

from __future__ import annotations

import math

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass


class GateDecompositionPass(TransformationPass):
    """Decompose gates into a target native basis gate set.

    Parameters
    ----------
    target_basis : tuple[str, ...]
        Gate names in the target native basis (e.g. ``("cx", "rz", "sx", "x", "id")``
        for IBM Eagle, or ``("cz", "rx", "rz")`` for Rigetti).
    """

    def __init__(self, target_basis: tuple[str, ...]) -> None:
        self._target_basis = frozenset(target_basis)

    @property
    def name(self) -> str:
        return "gate_decomposition"

    # ── decomposition rules ────────────────────────────────────────────

    def _decompose_h(self, qubits: tuple[int, ...]) -> list[QBGate]:
        """H -> rz(pi) sx rz(pi)  [IBM-style] or rx(pi/2) rz(pi/2) rx(pi/2) [Rigetti]."""
        q = qubits[0]
        if "sx" in self._target_basis and "rz" in self._target_basis:
            # IBM decomposition: H = Rz(pi) . SX . Rz(pi)
            return [
                QBGate(name="rz", qubits=(q,), params=(math.pi,)),
                QBGate(name="sx", qubits=(q,), params=()),
                QBGate(name="rz", qubits=(q,), params=(math.pi,)),
            ]
        elif "rx" in self._target_basis and "rz" in self._target_basis:
            # Rigetti decomposition: H = Rx(pi/2) . Rz(pi/2) . Rx(pi/2)
            return [
                QBGate(name="rx", qubits=(q,), params=(math.pi / 2,)),
                QBGate(name="rz", qubits=(q,), params=(math.pi / 2,)),
                QBGate(name="rx", qubits=(q,), params=(math.pi / 2,)),
            ]
        raise ValueError(
            f"Cannot decompose 'h' into basis {set(self._target_basis)}: need (sx, rz) or (rx, rz)"
        )

    def _decompose_cx(self, qubits: tuple[int, ...]) -> list[QBGate]:
        """CX -> ECR + single-qubit gates (for ECR-native backends)."""
        if "ecr" in self._target_basis:
            ctrl, tgt = qubits
            # CX = (I x Rz(-pi/2)) . ECR . (SX x I)
            # Simplified: rz on target, ecr, sx on control
            return [
                QBGate(name="rz", qubits=(tgt,), params=(-math.pi / 2,)),
                QBGate(name="ecr", qubits=(ctrl, tgt), params=()),
                QBGate(name="sx", qubits=(ctrl,), params=()),
            ]
        raise ValueError(
            f"Cannot decompose 'cx' into basis {set(self._target_basis)}: need 'ecr' in basis"
        )

    def _decompose_swap(self, qubits: tuple[int, ...]) -> list[QBGate]:
        """SWAP -> CX CX CX."""
        a, b = qubits
        cx_name = "cx" if "cx" in self._target_basis else None
        if cx_name is None:
            raise ValueError(
                f"Cannot decompose 'swap' into basis {set(self._target_basis)}: need 'cx'"
            )
        return [
            QBGate(name="cx", qubits=(a, b), params=()),
            QBGate(name="cx", qubits=(b, a), params=()),
            QBGate(name="cx", qubits=(a, b), params=()),
        ]

    def _decompose_ccx(self, qubits: tuple[int, ...]) -> list[QBGate]:
        """Toffoli (CCX) decomposition into CX + single-qubit gates.

        Uses the standard 6-CX decomposition with H and T/Tdg gates,
        further decomposable if needed.
        """
        a, b, c = qubits  # control1, control2, target

        if "cx" not in self._target_basis:
            raise ValueError(
                f"Cannot decompose 'ccx' into basis {set(self._target_basis)}: need 'cx'"
            )

        # Standard Toffoli decomposition: H, CX, T, Tdg gates
        # We'll use the basis-compatible version with rz for T/Tdg and
        # the H decomposition from above if h is not in basis
        pi4 = math.pi / 4
        gates: list[QBGate] = []

        def _h(q: int) -> list[QBGate]:
            if "h" in self._target_basis:
                return [QBGate(name="h", qubits=(q,), params=())]
            return self._decompose_h((q,))

        def _t(q: int) -> QBGate:
            return QBGate(name="rz", qubits=(q,), params=(pi4,))

        def _tdg(q: int) -> QBGate:
            return QBGate(name="rz", qubits=(q,), params=(-pi4,))

        # Toffoli = sequence of H, CNOT, T, Tdg
        gates.extend(_h(c))
        gates.append(QBGate(name="cx", qubits=(b, c), params=()))
        gates.append(_tdg(c))
        gates.append(QBGate(name="cx", qubits=(a, c), params=()))
        gates.append(_t(c))
        gates.append(QBGate(name="cx", qubits=(b, c), params=()))
        gates.append(_tdg(c))
        gates.append(QBGate(name="cx", qubits=(a, c), params=()))
        gates.append(_t(b))
        gates.append(_t(c))
        gates.extend(_h(c))
        gates.append(QBGate(name="cx", qubits=(a, b), params=()))
        gates.append(_t(a))
        gates.append(_tdg(b))
        gates.append(QBGate(name="cx", qubits=(a, b), params=()))

        return gates

    # ── main logic ─────────────────────────────────────────────────────

    def _decompose_gate(self, gate: QBGate) -> list[QBGate]:
        """Decompose a single gate. Returns list of basis gates."""
        if gate.name in self._target_basis:
            return [gate]

        decomposers = {
            "h": self._decompose_h,
            "cx": self._decompose_cx,
            "swap": self._decompose_swap,
            "ccx": self._decompose_ccx,
        }

        decomposer = decomposers.get(gate.name)
        if decomposer is None:
            raise ValueError(
                f"No decomposition rule for gate '{gate.name}' into basis {set(self._target_basis)}"
            )

        result = decomposer(gate.qubits)

        # Recursively decompose if any result gates are still not in basis
        final: list[QBGate] = []
        for g in result:
            if g.name in self._target_basis:
                final.append(g)
            else:
                final.extend(self._decompose_gate(g))

        return final

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        new_circuit = QBCircuit(
            n_qubits=circuit.n_qubits,
            n_clbits=circuit.n_clbits,
            name=circuit.name,
        )
        decomposed_count = 0

        for op in circuit.iter_ops():
            if isinstance(op, QBGate):
                if op.name in self._target_basis:
                    new_circuit.add_gate(op)
                else:
                    decomposed_gates = self._decompose_gate(op)
                    for g in decomposed_gates:
                        new_circuit.add_gate(g)
                    decomposed_count += 1
            elif isinstance(op, QBMeasure):
                new_circuit.add_measurement(op.qubit, op.clbit)
            elif isinstance(op, QBBarrier):
                new_circuit.add_barrier(op.qubits)

        context["decomposed_gates"] = decomposed_count

        return PassResult(
            circuit=new_circuit,
            metadata={"decomposed_gates": decomposed_count},
            modified=decomposed_count > 0,
        )
