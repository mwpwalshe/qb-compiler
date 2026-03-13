"""Abstract base class for calibration data providers."""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties

ALLOWED_CALIBRATION_HOSTS = frozenset({
    "api.quantum-computing.ibm.com",
    "quantum.ibm.com",
    "api.qubitboost.io",
    "calibration.qubitboost.io",
})


def validate_calibration_url(url: str) -> None:
    """Validate that *url* points to an approved calibration host over HTTPS.

    Parameters
    ----------
    url:
        The URL to validate.

    Raises
    ------
    CalibrationError
        If the host is not in the allowlist or the scheme is not HTTPS.
    """
    from qb_compiler.exceptions import CalibrationError

    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_CALIBRATION_HOSTS:
        raise CalibrationError(
            f"Refusing to fetch calibration from untrusted host: {parsed.hostname}. "
            f"Allowed hosts: {', '.join(sorted(ALLOWED_CALIBRATION_HOSTS))}"
        )
    if parsed.scheme != "https":
        raise CalibrationError(
            f"Calibration endpoints must use HTTPS, got: {parsed.scheme}"
        )


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
