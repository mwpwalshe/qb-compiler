"""Noise-aware ALAP scheduler.

Reorders gates within dependency constraints so that qubits with shorter
T1/T2 times have their gates packed as close to measurement as possible,
minimising idle-time-weighted decoherence.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qb_compiler.ir.circuit import Operation, QBCircuit
from qb_compiler.ir.dag import QBDag
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qb_compiler.calibration.models.qubit_properties import QubitProperties


class NoiseAwareScheduler(TransformationPass):
    """ALAP scheduler that accounts for per-qubit T1/T2 decay.

    Within each parallel layer of the DAG, operations are sorted so that
    high-urgency qubits (short T2) are placed later -- closer to
    measurement.  The overall topological ordering is preserved; only the
    *intra-layer* ordering changes.

    Parameters
    ----------
    qubit_properties : Sequence[QubitProperties]
        Per-qubit calibration data.  Must cover every qubit in the circuit.
    default_gate_duration_us : float
        Assumed gate duration in microseconds (used for decoherence
        estimation when no per-gate durations are available).
    """

    def __init__(
        self,
        qubit_properties: Sequence[QubitProperties],
        default_gate_duration_us: float = 0.1,
    ) -> None:
        self._qprops = {qp.qubit_id: qp for qp in qubit_properties}
        self._default_gate_duration_us = default_gate_duration_us

    @property
    def name(self) -> str:
        return "noise_aware_scheduler"

    # ── helpers ────────────────────────────────────────────────────────

    def _urgency(self, qubit: int) -> float:
        """Urgency = 1/T2.  Higher means qubit decoheres faster."""
        qp = self._qprops.get(qubit)
        if qp is None or qp.t2_us is None or qp.t2_us <= 0:
            return 0.0
        return 1.0 / qp.t2_us

    def _op_urgency(self, op: Operation) -> float:
        """Maximum urgency across all qubits an operation touches."""
        if isinstance(op, QBGate):
            qubits = op.qubits
        elif isinstance(op, QBMeasure):
            qubits = (op.qubit,)
        elif isinstance(op, QBBarrier):
            return 0.0  # barriers don't move
        else:
            return 0.0
        if not qubits:
            return 0.0
        return max(self._urgency(q) for q in qubits)

    def _estimate_idle_decoherence(
        self, circuit: QBCircuit
    ) -> float:
        """Rough estimate of total decoherence factor for a circuit.

        For each qubit, estimate idle time as (depth - gates_on_qubit) *
        gate_duration and compute exp(-idle/T2).  Returns the product of
        all per-qubit survival probabilities.
        """
        depth = circuit.depth
        if depth == 0:
            return 1.0

        # Count how many layers each qubit is busy
        qubit_busy: dict[int, int] = {}
        for op in circuit.iter_ops():
            if isinstance(op, QBGate):
                for q in op.qubits:
                    qubit_busy[q] = qubit_busy.get(q, 0) + 1
            elif isinstance(op, QBMeasure):
                qubit_busy[op.qubit] = qubit_busy.get(op.qubit, 0) + 1

        survival = 1.0
        dt = self._default_gate_duration_us
        for q in range(circuit.n_qubits):
            qp = self._qprops.get(q)
            if qp is None or qp.t2_us is None or qp.t2_us <= 0:
                continue
            busy = qubit_busy.get(q, 0)
            idle_slots = max(0, depth - busy)
            idle_time = idle_slots * dt
            survival *= math.exp(-idle_time / qp.t2_us)

        return survival

    # ── main logic ─────────────────────────────────────────────────────

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        # Estimate decoherence before scheduling
        survival_before = self._estimate_idle_decoherence(circuit)

        dag = QBDag.from_circuit(circuit)
        layers = dag.layers()

        # Sort each layer: low-urgency ops first, high-urgency ops last
        # (ALAP -- high-urgency gates are pushed later / closer to measurement)
        scheduled_ops: list[Operation] = []
        for layer in layers:
            layer_sorted = sorted(layer, key=lambda op: self._op_urgency(op))
            scheduled_ops.extend(layer_sorted)

        # Build the new circuit
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

        # Estimate decoherence after scheduling
        survival_after = self._estimate_idle_decoherence(new_circuit)

        # Compute improvement fraction (0 means no change, >0 means better)
        if survival_before > 0:
            reduction = (
                (survival_after - survival_before) / (1.0 - survival_before)
                if survival_before < 1.0
                else 0.0
            )
        else:
            reduction = 0.0
        reduction = max(0.0, reduction)

        context["estimated_decoherence_reduction"] = reduction

        modified = new_circuit != circuit
        return PassResult(
            circuit=new_circuit,
            metadata={
                "estimated_decoherence_reduction": reduction,
                "survival_before": survival_before,
                "survival_after": survival_after,
            },
            modified=modified,
        )
