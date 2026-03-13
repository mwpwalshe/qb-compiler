"""Parse IBM calibration data from the QubitBoost calibration_hub format."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties

if TYPE_CHECKING:
    from pathlib import Path


def parse_ibm_calibration(data: dict) -> BackendProperties:
    """Parse a QubitBoost ``calibration_hub/heron/*.json`` dict into
    :class:`BackendProperties`.

    The IBM-specific format stores:
    - ``qubit_properties[].T1`` / ``T2`` in microseconds
    - ``qubit_properties[].readout_error_0to1`` / ``readout_error_1to0``
    - ``qubit_properties[].frequency`` in GHz (often null for Heron)
    - ``gate_properties[].parameters.gate_error``
    - ``gate_properties[].parameters.gate_length`` in nanoseconds
    - ``coupling_map`` as list of [q1, q2] pairs
    - ``basis_gates`` as list of gate name strings

    Parameters
    ----------
    data:
        Parsed JSON from a QubitBoost IBM calibration file.

    Returns
    -------
    BackendProperties
        Fully populated properties object.
    """
    qubit_props: list[QubitProperties] = []
    for entry in data.get("qubit_properties", []):
        err_01 = entry.get("readout_error_0to1")
        err_10 = entry.get("readout_error_1to0")
        # Symmetrise readout error
        readout_err = None
        if err_01 is not None and err_10 is not None:
            readout_err = (err_01 + err_10) / 2.0
        elif err_01 is not None:
            readout_err = err_01
        elif err_10 is not None:
            readout_err = err_10

        qubit_props.append(
            QubitProperties(
                qubit_id=int(entry["qubit"]),
                t1_us=entry.get("T1"),
                t2_us=entry.get("T2"),
                frequency_ghz=entry.get("frequency"),
                readout_error=readout_err,
                readout_error_0to1=err_01,
                readout_error_1to0=err_10,
            )
        )

    gate_props: list[GateProperties] = []
    for entry in data.get("gate_properties", []):
        params = entry.get("parameters", {})
        gate_error = params.get("gate_error")
        gate_length = params.get("gate_length")

        # Sanitise: some IBM calibrations report gate_error = 1.0 for
        # broken qubits/gates.  We keep them (the compiler should route
        # around them) but don't let NaN through.
        if gate_error is not None and not _is_finite(gate_error):
            gate_error = 1.0
        if gate_length is not None and not _is_finite(gate_length):
            gate_length = None

        gate_props.append(
            GateProperties(
                gate_type=entry["gate"],
                qubits=tuple(int(q) for q in entry["qubits"]),
                error_rate=gate_error,
                gate_time_ns=gate_length,
            )
        )

    coupling = [
        (int(e[0]), int(e[1])) for e in data.get("coupling_map", [])
    ]
    basis = tuple(data.get("basis_gates", []))

    return BackendProperties(
        backend=data.get("backend_name", "unknown_ibm"),
        provider="ibm",
        n_qubits=int(data.get("n_qubits", len(qubit_props))),
        basis_gates=basis,
        coupling_map=coupling,
        qubit_properties=qubit_props,
        gate_properties=gate_props,
        timestamp=data.get("timestamp", ""),
    )


def load_ibm_calibration(path: str | Path) -> BackendProperties:
    """Convenience: load JSON file and parse in one call."""
    with open(path) as fh:
        data = json.load(fh)
    return parse_ibm_calibration(data)


def _is_finite(v: float) -> bool:
    """Check for finite float (not NaN or inf)."""
    import math
    return math.isfinite(v)
