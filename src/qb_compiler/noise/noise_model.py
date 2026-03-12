"""Abstract base class for noise models."""

from __future__ import annotations

import abc


class NoiseModel(abc.ABC):
    """Interface for objects that estimate noise characteristics.

    Implementations translate raw calibration numbers into error
    estimates that the compiler can use for routing, scheduling,
    and fidelity prediction.
    """

    @abc.abstractmethod
    def qubit_error(self, qubit: int) -> float:
        """Combined single-qubit error estimate (gate + readout + decoherence).

        Returns a value in [0, 1] where 0 = perfect, 1 = maximally noisy.
        """

    @abc.abstractmethod
    def gate_error(self, gate: str, qubits: tuple[int, ...]) -> float:
        """Error rate for *gate* applied to *qubits*.

        Returns a value in [0, 1].  For gates not found in calibration,
        implementations should return a conservative default.
        """

    @abc.abstractmethod
    def readout_error(self, qubit: int) -> float:
        """Measurement error probability for *qubit*.

        Returns a value in [0, 1].
        """

    @abc.abstractmethod
    def decoherence_factor(self, qubit: int, gate_time_ns: float) -> float:
        """Decoherence-induced error for *qubit* over *gate_time_ns*.

        Based on T1/T2 relaxation.  Returns a value in [0, 1] where
        0 = no decoherence and 1 = fully decohered.
        """
