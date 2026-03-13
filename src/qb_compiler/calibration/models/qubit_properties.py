"""Per-qubit calibration properties."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QubitProperties:
    """Calibration snapshot for a single physical qubit.

    Parameters
    ----------
    qubit_id:
        Physical qubit index on the device.
    t1_us:
        Energy relaxation time (T1) in microseconds, or *None* if unavailable.
    t2_us:
        Dephasing time (T2) in microseconds, or *None* if unavailable.
    readout_error:
        Combined (symmetrised) readout error probability, or *None*.
    frequency_ghz:
        Qubit drive frequency in GHz, or *None*.
    readout_error_0to1:
        P(1|0) — probability of reading 1 when state is 0, or *None*.
    readout_error_1to0:
        P(0|1) — probability of reading 0 when state is 1, or *None*.
    """

    qubit_id: int
    t1_us: float | None = None
    t2_us: float | None = None
    readout_error: float | None = None
    frequency_ghz: float | None = None
    readout_error_0to1: float | None = None
    readout_error_1to0: float | None = None

    @property
    def t1_asymmetry_ratio(self) -> float:
        """Ratio of |1⟩→|0⟩ decay error to |0⟩→|1⟩ excitation error.

        On superconducting qubits, thermal excitation P(1|0) is typically
        much smaller than relaxation P(0|1).  A high ratio means the qubit
        loses |1⟩ states disproportionately — circuits that hold qubits in
        |1⟩ (after X gates, CNOT targets, etc.) suffer more on these qubits.

        Returns 1.0 when asymmetry data is unavailable, meaning no penalty.
        """
        if (
            self.readout_error_0to1 is not None
            and self.readout_error_1to0 is not None
            and self.readout_error_0to1 > 1e-9
        ):
            return self.readout_error_1to0 / self.readout_error_0to1
        return 1.0

    @property
    def t1_asymmetry_penalty(self) -> float:
        """Readout-scaled penalty for T1 asymmetry.

        Returns ``readout_error * ln(ratio)`` so the penalty is proportional
        to both the *magnitude* of the readout error and the asymmetry.
        This keeps the penalty in the same units as readout error, ensuring
        it doesn't overwhelm other scoring terms.

        - ratio=1  → penalty=0 (symmetric, no penalty)
        - ratio=10, ro=0.01 → penalty≈0.023
        - ratio=24, ro=0.01 → penalty≈0.032

        Clamped so ratios below 1 produce zero penalty (those qubits
        are actually *better* at holding |1⟩ states).
        """
        ratio = self.t1_asymmetry_ratio
        if ratio <= 1.0:
            return 0.0
        ro = self.readout_error if self.readout_error is not None else 0.01
        return ro * math.log(ratio)

    @staticmethod
    def symmetrise_readout(
        err_0to1: float | None,
        err_1to0: float | None,
    ) -> float | None:
        """Average the two asymmetric readout errors into one number."""
        if err_0to1 is not None and err_1to0 is not None:
            return (err_0to1 + err_1to0) / 2.0
        return err_0to1 or err_1to0

    @classmethod
    def from_qubitboost_dict(cls, d: dict) -> QubitProperties:
        """Parse a single entry from the QubitBoost ``qubit_properties`` list.

        Expected keys: qubit, T1, T2, frequency, readout_error_0to1,
        readout_error_1to0.
        """
        err_01 = d.get("readout_error_0to1")
        err_10 = d.get("readout_error_1to0")
        return cls(
            qubit_id=int(d["qubit"]),
            t1_us=d.get("T1"),
            t2_us=d.get("T2"),
            frequency_ghz=d.get("frequency"),
            readout_error=cls.symmetrise_readout(err_01, err_10),
            readout_error_0to1=err_01,
            readout_error_1to0=err_10,
        )
