"""Per-qubit calibration properties."""

from __future__ import annotations

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
