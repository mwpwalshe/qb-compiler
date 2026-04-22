"""Error budget estimation analysis pass.

Estimates the total expected infidelity of a circuit given per-gate error
rates and per-qubit T1/T2 calibration data.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qb_compiler.ir.operations import QBGate, QBMeasure
from qb_compiler.passes.base import AnalysisPass

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qb_compiler.calibration.models.qubit_properties import QubitProperties
    from qb_compiler.ir.circuit import QBCircuit


class ErrorBudgetEstimator(AnalysisPass):
    """Estimate total circuit fidelity from a noise model.

    Accumulates three error sources:

    1. **Gate errors** -- per-gate infidelity ``(1 - fidelity)``.
    2. **Decoherence** -- idle-time decay ``exp(-t/T1)`` per qubit.
    3. **Readout errors** -- measurement error for each measured qubit.

    Results are written to ``context`` as ``"estimated_fidelity"`` (float)
    and ``"error_budget"`` (dict with ``gate``, ``decoherence``, ``readout``
    breakdown).

    Parameters
    ----------
    qubit_properties : Sequence[QubitProperties]
        Per-qubit calibration data (T1, T2, readout error).
    gate_error_rates : dict[str, float]
        Mapping from gate name to error rate (probability of error per
        application).  Missing gates are assumed perfect (error = 0).
    gate_duration_us : float
        Assumed uniform gate duration in microseconds.
    """

    def __init__(
        self,
        qubit_properties: Sequence[QubitProperties],
        gate_error_rates: dict[str, float] | None = None,
        gate_duration_us: float = 0.1,
    ) -> None:
        self._qprops = {qp.qubit_id: qp for qp in qubit_properties}
        self._gate_errors = gate_error_rates or {}
        self._gate_duration_us = gate_duration_us

    @property
    def name(self) -> str:
        return "error_budget_estimator"

    def analyze(self, circuit: QBCircuit, context: dict) -> None:
        gate_fidelity = 1.0
        decoherence_fidelity = 1.0
        readout_fidelity = 1.0

        # --- Gate errors ---
        gate_infidelity_total = 0.0
        for op in circuit.iter_ops():
            if isinstance(op, QBGate):
                err = self._gate_errors.get(op.name, 0.0)
                gate_fidelity *= 1.0 - err
                gate_infidelity_total += err

        # --- Decoherence (idle time) ---
        # Compute per-qubit idle slots: circuit depth minus busy slots
        depth = circuit.depth
        qubit_busy: dict[int, int] = {}
        for op in circuit.iter_ops():
            if isinstance(op, QBGate):
                for q in op.qubits:
                    qubit_busy[q] = qubit_busy.get(q, 0) + 1
            elif isinstance(op, QBMeasure):
                qubit_busy[op.qubit] = qubit_busy.get(op.qubit, 0) + 1

        decoherence_infidelity_total = 0.0
        for q in circuit.qubits_used():
            qp = self._qprops.get(q)
            if qp is None:
                continue
            busy = qubit_busy.get(q, 0)
            idle_slots = max(0, depth - busy)
            idle_time = idle_slots * self._gate_duration_us

            # T1 decay contribution
            if qp.t1_us is not None and qp.t1_us > 0:
                t1_survival = math.exp(-idle_time / qp.t1_us)
                decoherence_fidelity *= t1_survival
                decoherence_infidelity_total += 1.0 - t1_survival

        # --- Readout errors ---
        readout_infidelity_total = 0.0
        measured_qubits: set[int] = set()
        for op in circuit.iter_ops():
            if isinstance(op, QBMeasure):
                measured_qubits.add(op.qubit)

        for q in measured_qubits:
            qp = self._qprops.get(q)
            if qp is None:
                continue
            ro_err = qp.readout_error or 0.0
            readout_fidelity *= 1.0 - ro_err
            readout_infidelity_total += ro_err

        # --- Total fidelity ---
        estimated_fidelity = gate_fidelity * decoherence_fidelity * readout_fidelity

        error_budget = {
            "gate": gate_infidelity_total,
            "decoherence": decoherence_infidelity_total,
            "readout": readout_infidelity_total,
        }

        context["estimated_fidelity"] = estimated_fidelity
        context["error_budget"] = error_budget
