"""Calibration provider backed by a static :class:`BackendProperties` snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.provider import CalibrationProvider

if TYPE_CHECKING:
    from pathlib import Path

    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties


class StaticCalibrationProvider(CalibrationProvider):
    """Serves calibration data from an in-memory :class:`BackendProperties`.

    This is the simplest provider — no network calls, no caching logic.
    Useful for offline analysis, unit tests, and replaying historical
    calibration snapshots from the QubitBoost ``calibration_hub``.

    Parameters
    ----------
    props:
        Pre-loaded backend calibration snapshot.
    """

    def __init__(self, props: BackendProperties) -> None:
        self._props = props
        # Build lookup indices for O(1) access
        self._qubit_map: dict[int, QubitProperties] = {
            qp.qubit_id: qp for qp in props.qubit_properties
        }
        self._gate_map: dict[tuple[str, tuple[int, ...]], GateProperties] = {
            (gp.gate_type, gp.qubits): gp for gp in props.gate_properties
        }
        # Parse timestamp once
        self._ts = _parse_timestamp(props.timestamp)

    # ── CalibrationProvider interface ────────────────────────────────

    def get_qubit_properties(self, qubit: int) -> QubitProperties | None:
        return self._qubit_map.get(qubit)

    def get_gate_properties(
        self, gate: str, qubits: tuple[int, ...]
    ) -> GateProperties | None:
        return self._gate_map.get((gate, qubits))

    def get_all_qubit_properties(self) -> list[QubitProperties]:
        return list(self._props.qubit_properties)

    def get_all_gate_properties(self) -> list[GateProperties]:
        return list(self._props.gate_properties)

    @property
    def backend_name(self) -> str:
        return self._props.backend

    @property
    def timestamp(self) -> datetime:
        return self._ts

    # ── convenience constructors ─────────────────────────────────────

    @classmethod
    def from_json(cls, path: str | Path) -> StaticCalibrationProvider:
        """Load from a QubitBoost ``calibration_hub`` JSON file.

        Parameters
        ----------
        path:
            Filesystem path to a JSON file matching the QubitBoost
            calibration format (``backend_name``, ``qubit_properties``,
            ``gate_properties``, ``coupling_map``, etc.).
        """
        props = BackendProperties.from_qubitboost_json(path)
        return cls(props)

    @classmethod
    def from_dict(cls, data: dict) -> StaticCalibrationProvider:
        """Build from an already-parsed calibration dict."""
        props = BackendProperties.from_qubitboost_dict(data)
        return cls(props)

    # ── extras ───────────────────────────────────────────────────────

    @property
    def properties(self) -> BackendProperties:
        """The underlying :class:`BackendProperties` snapshot."""
        return self._props

    def __repr__(self) -> str:
        return (
            f"StaticCalibrationProvider(backend={self._props.backend!r}, "
            f"n_qubits={self._props.n_qubits}, "
            f"timestamp={self._props.timestamp!r})"
        )


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp, assuming UTC when no timezone is given."""
    if not ts:
        return datetime.now(timezone.utc)
    # Strip trailing 'Z' that some serialisers use
    cleaned = ts.rstrip("Z")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        # Last resort: try a common format without microseconds
        dt = datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
