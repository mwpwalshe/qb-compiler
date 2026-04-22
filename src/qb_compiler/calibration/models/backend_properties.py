"""Aggregate backend calibration snapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class BackendProperties:
    """Complete calibration snapshot for a quantum backend.

    Parameters
    ----------
    backend:
        Backend identifier (e.g. ``"ibm_fez"``).
    provider:
        Vendor / cloud provider (e.g. ``"ibm"``, ``"rigetti"``).
    n_qubits:
        Number of physical qubits on the device.
    basis_gates:
        Native gate set advertised by the backend.
    coupling_map:
        Directional adjacency list ``[(ctrl, tgt), ...]``.
    qubit_properties:
        Per-qubit calibration data.
    gate_properties:
        Per-gate calibration data.
    timestamp:
        ISO-8601 string indicating when calibration was taken.
    """

    backend: str
    provider: str
    n_qubits: int
    basis_gates: tuple[str, ...]
    coupling_map: list[tuple[int, int]]
    qubit_properties: list[QubitProperties]
    gate_properties: list[GateProperties]
    timestamp: str

    # ── factory ──────────────────────────────────────────────────────

    @classmethod
    def from_qubitboost_json(cls, path: str | Path) -> BackendProperties:
        """Load from a QubitBoost ``calibration_hub`` JSON file.

        The expected top-level keys are ``backend_name``, ``n_qubits``,
        ``timestamp``, ``qubit_properties``, ``gate_properties``,
        ``coupling_map``, ``basis_gates``.
        """
        with open(path) as fh:
            data = json.load(fh)
        return cls.from_qubitboost_dict(data)

    @classmethod
    def from_qubitboost_dict(cls, data: dict) -> BackendProperties:
        """Build from an already-parsed QubitBoost calibration dict."""
        qprops = [QubitProperties.from_qubitboost_dict(q) for q in data.get("qubit_properties", [])]
        gprops = [GateProperties.from_qubitboost_dict(g) for g in data.get("gate_properties", [])]
        coupling = [(int(e[0]), int(e[1])) for e in data.get("coupling_map", [])]
        basis = tuple(data.get("basis_gates", []))

        # Infer provider from backend name if not explicitly present
        backend_name = data.get("backend_name", "unknown")
        provider = data.get("provider", _infer_provider(backend_name))

        return cls(
            backend=backend_name,
            provider=provider,
            n_qubits=int(data.get("n_qubits", len(qprops))),
            basis_gates=basis,
            coupling_map=coupling,
            qubit_properties=qprops,
            gate_properties=gprops,
            timestamp=data.get("timestamp", ""),
        )

    # ── helpers ──────────────────────────────────────────────────────

    def qubit(self, qubit_id: int) -> QubitProperties | None:
        """Return :class:`QubitProperties` for *qubit_id*, or *None*."""
        for qp in self.qubit_properties:
            if qp.qubit_id == qubit_id:
                return qp
        return None

    def gate(self, gate_type: str, qubits: tuple[int, ...]) -> GateProperties | None:
        """Return :class:`GateProperties` matching *gate_type* and *qubits*."""
        for gp in self.gate_properties:
            if gp.gate_type == gate_type and gp.qubits == qubits:
                return gp
        return None


def _infer_provider(backend_name: str) -> str:
    """Best-effort provider inference from backend name."""
    name = backend_name.lower()
    if name.startswith("ibm"):
        return "ibm"
    if name.startswith("rigetti") or name.startswith("ankaa"):
        return "rigetti"
    if name.startswith("ionq"):
        return "ionq"
    if name.startswith("iqm"):
        return "iqm"
    return "unknown"
