"""Abstract base class for calibration data providers."""

from __future__ import annotations

import abc
from datetime import datetime, timezone

from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties


class CalibrationProvider(abc.ABC):
    """Interface for objects that supply device calibration data.

    Concrete implementations may pull from static files, live APIs,
    or cached wrappers around other providers.
    """

    # ── per-element lookups ──────────────────────────────────────────

    @abc.abstractmethod
    def get_qubit_properties(self, qubit: int) -> QubitProperties | None:
        """Return calibration data for *qubit*, or *None* if unknown."""

    @abc.abstractmethod
    def get_gate_properties(
        self, gate: str, qubits: tuple[int, ...]
    ) -> GateProperties | None:
        """Return calibration data for *gate* on *qubits*, or *None*."""

    # ── bulk access ──────────────────────────────────────────────────

    @abc.abstractmethod
    def get_all_qubit_properties(self) -> list[QubitProperties]:
        """Return calibration data for every qubit on the device."""

    @abc.abstractmethod
    def get_all_gate_properties(self) -> list[GateProperties]:
        """Return calibration data for every characterised gate."""

    # ── metadata ─────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier (e.g. ``'ibm_fez'``)."""

    @property
    @abc.abstractmethod
    def timestamp(self) -> datetime:
        """UTC datetime when the calibration data was taken."""

    @property
    def age_hours(self) -> float:
        """Hours elapsed since the calibration was taken."""
        delta = datetime.now(timezone.utc) - self.timestamp
        return delta.total_seconds() / 3600.0
