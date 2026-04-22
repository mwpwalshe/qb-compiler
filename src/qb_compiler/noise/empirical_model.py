"""Noise model built from real calibration data."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qb_compiler.noise.noise_model import NoiseModel

if TYPE_CHECKING:
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties
    from qb_compiler.calibration.provider import CalibrationProvider


# Conservative fallback values when calibration data is missing
_DEFAULT_SINGLE_QUBIT_ERROR = 0.005
_DEFAULT_TWO_QUBIT_ERROR = 0.02
_DEFAULT_READOUT_ERROR = 0.015
_DEFAULT_T1_US = 100.0
_DEFAULT_T2_US = 80.0
_DEFAULT_GATE_TIME_NS = 40.0  # typical 1q gate on superconducting


class EmpiricalNoiseModel(NoiseModel):
    """Noise model derived from a :class:`CalibrationProvider`.

    Translates per-qubit T1/T2 and per-gate error rates into the
    unified :class:`NoiseModel` interface.  Where calibration data is
    missing, conservative defaults are used.

    Parameters
    ----------
    qubit_props:
        Mapping from qubit index to its calibration properties.
    gate_props:
        Mapping from ``(gate_type, qubits)`` to gate calibration.
    """

    def __init__(
        self,
        qubit_props: dict[int, QubitProperties],
        gate_props: dict[tuple[str, tuple[int, ...]], GateProperties],
    ) -> None:
        self._qubits = qubit_props
        self._gates = gate_props

    # ── factory ──────────────────────────────────────────────────────

    @classmethod
    def from_calibration(cls, provider: CalibrationProvider) -> EmpiricalNoiseModel:
        """Build an :class:`EmpiricalNoiseModel` from a calibration provider."""
        qubit_props = {qp.qubit_id: qp for qp in provider.get_all_qubit_properties()}
        gate_props = {(gp.gate_type, gp.qubits): gp for gp in provider.get_all_gate_properties()}
        return cls(qubit_props=qubit_props, gate_props=gate_props)

    # ── NoiseModel interface ─────────────────────────────────────────

    def qubit_error(self, qubit: int) -> float:
        """Combined error: average of best single-qubit gate error, readout
        error, and a T2-based idling decoherence term over one gate time."""
        self._qubits.get(qubit)
        gate_err = self._best_single_qubit_gate_error(qubit)
        ro_err = self.readout_error(qubit)
        dec_err = self.decoherence_factor(qubit, _DEFAULT_GATE_TIME_NS)
        # Combine via depolarising channel composition:
        # P(no error) = (1-gate_err)(1-ro_err)(1-dec_err)
        p_ok = (1.0 - gate_err) * (1.0 - ro_err) * (1.0 - dec_err)
        return 1.0 - p_ok

    def gate_error(self, gate: str, qubits: tuple[int, ...]) -> float:
        gp = self._gates.get((gate, qubits))
        if gp is not None and gp.error_rate is not None:
            return min(gp.error_rate, 1.0)
        # Fallback: try reverse qubit order for symmetric gates
        if len(qubits) == 2:
            gp = self._gates.get((gate, (qubits[1], qubits[0])))
            if gp is not None and gp.error_rate is not None:
                return min(gp.error_rate, 1.0)
        # Conservative default
        if len(qubits) >= 2:
            return _DEFAULT_TWO_QUBIT_ERROR
        return _DEFAULT_SINGLE_QUBIT_ERROR

    def readout_error(self, qubit: int) -> float:
        qp = self._qubits.get(qubit)
        if qp is not None and qp.readout_error is not None:
            return min(qp.readout_error, 1.0)
        return _DEFAULT_READOUT_ERROR

    def decoherence_factor(self, qubit: int, gate_time_ns: float) -> float:
        """Decoherence error from T1 relaxation.

        Uses the formula:  p_decay = 1 - exp(-t / T1)

        where *t* is the gate time.  T2-based dephasing is also
        incorporated:  p_dephase = 1 - exp(-t / T2).  The combined
        decoherence error is:

            p_err = 1 - (1 - p_decay) * (1 - p_dephase)
        """
        qp = self._qubits.get(qubit)
        t1_us = qp.t1_us if qp and qp.t1_us else _DEFAULT_T1_US
        t2_us = qp.t2_us if qp and qp.t2_us else _DEFAULT_T2_US

        # Convert gate time from ns to us
        t_us = gate_time_ns / 1000.0

        # T1 amplitude damping
        p_decay = 1.0 - math.exp(-t_us / t1_us)
        # T2 dephasing (T2 <= 2*T1 by physics, but we don't enforce)
        p_dephase = 1.0 - math.exp(-t_us / t2_us)

        # Combined: P(no decoherence) = (1-p_decay)*(1-p_dephase)
        return 1.0 - (1.0 - p_decay) * (1.0 - p_dephase)

    # ── helpers ──────────────────────────────────────────────────────

    def _best_single_qubit_gate_error(self, qubit: int) -> float:
        """Find the lowest error 1-qubit gate on *qubit*."""
        best = _DEFAULT_SINGLE_QUBIT_ERROR
        for (_gtype, gqubits), gp in self._gates.items():
            if gqubits == (qubit,) and gp.error_rate is not None:
                best = min(best, gp.error_rate)
        return best

    def gate_time_ns(self, gate: str, qubits: tuple[int, ...]) -> float:
        """Gate duration in nanoseconds, or default if unknown."""
        gp = self._gates.get((gate, qubits))
        if gp is not None and gp.gate_time_ns is not None:
            return gp.gate_time_ns
        return _DEFAULT_GATE_TIME_NS

    @property
    def n_qubits(self) -> int:
        """Number of qubits with calibration data."""
        return len(self._qubits)

    def __repr__(self) -> str:
        return f"EmpiricalNoiseModel(n_qubits={len(self._qubits)}, n_gates={len(self._gates)})"
