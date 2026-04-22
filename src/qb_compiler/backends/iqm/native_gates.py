"""IQM backend native gate definitions."""

from __future__ import annotations

IQM_GARNET_BASIS: tuple[str, ...] = ("cz", "prx", "id")
"""Native gates for IQM Garnet processors.

- ``cz``: controlled-Z (native 2q entangling gate)
- ``prx``: parameterised single-qubit rotation (combines Rx and Ry)
- ``id``: identity
"""

IQM_TYPICAL_GATE_TIMES_NS: dict[str, float] = {
    "id": 20.0,
    "prx": 32.0,  # fast single-qubit gate
    "cz": 60.0,  # native 2q gate (faster than IBM/Rigetti)
    "reset": 1500.0,
    "measure": 1200.0,
}

# Typical error rates for IQM Garnet (superconducting, transmon-based)
IQM_TYPICAL_1Q_ERROR: float = 0.001
IQM_TYPICAL_2Q_ERROR: float = 0.015
IQM_TYPICAL_READOUT_ERROR: float = 0.02
