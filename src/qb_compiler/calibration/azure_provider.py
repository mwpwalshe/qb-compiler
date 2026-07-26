"""Calibration / device-info provider for Azure Quantum.

Azure Quantum is an aggregator: it brokers access to IonQ, Quantinuum, Rigetti (and others) and
exposes per-target *metadata*, name, provider, current availability/queue, and capability info
(qubit count, native gate set). It does **not** publish rich per-qubit error calibration the way IBM
or Braket do. So this provider is honest about scope: it captures device capabilities + availability
and leaves per-qubit/per-gate error fields sparse (read only when a target actually publishes them).

It is registered as a *secondary* access path in the calibration registry, a redundant route to
IonQ/Quantinuum/Rigetti used only if the primary (Braket / pytket) provider is unavailable.

``parse_azure_target`` is a pure function over a target-metadata dict, so it is fully unit-testable
offline without the ``azure-quantum`` SDK or an Azure subscription.

Requires::

    pip install "qb-compiler[azure]"   # pulls azure-quantum
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.provider import CalibrationProvider
from qb_compiler.calibration.static_provider import StaticCalibrationProvider

if TYPE_CHECKING:
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties


def _f(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def parse_azure_target(
    target: dict[str, Any],
    *,
    backend: str,
    provider: str | None = None,
    timestamp: str | None = None,
) -> BackendProperties:
    """Parse an Azure Quantum target-metadata dict into a :class:`BackendProperties` snapshot.

    Pure function (no network). Reads capability info (qubit count, native gates) when present and
    any published per-qubit error/readout; otherwise degrades to a structurally-valid,
    calibration-sparse snapshot (Azure usually does not publish per-qubit calibration).
    """
    from qb_compiler.calibration.models.qubit_properties import QubitProperties

    prov = provider or str(target.get("provider_id") or target.get("provider") or "azure")
    caps = target.get("capabilities") or target.get("capability") or {}
    n_qubits = int(_f(caps.get("qubitCount") or caps.get("n_qubits")) or 0)
    native = caps.get("nativeGateSet") or caps.get("gateSet") or caps.get("basis_gates") or []
    basis_gates = tuple(str(g).lower() for g in native)

    qubit_props: list[QubitProperties] = []
    gate_props: list[GateProperties] = []

    # Only populate calibration if a target actually publishes it (rare on Azure).
    cal = target.get("calibration") or {}
    per_qubit = cal.get("qubits") or {}
    for q_key, vals in per_qubit.items():
        try:
            q = int(q_key)
        except (TypeError, ValueError):
            continue
        qubit_props.append(
            QubitProperties(
                qubit_id=q,
                t1_us=_f(vals.get("t1_us")),
                t2_us=_f(vals.get("t2_us")),
                readout_error=_f(vals.get("readout_error")),
            )
        )
    if not qubit_props and n_qubits:
        qubit_props = [QubitProperties(qubit_id=q) for q in range(n_qubits)]

    return BackendProperties(
        backend=backend,
        provider=prov,
        n_qubits=n_qubits,
        basis_gates=basis_gates,
        coupling_map=[],  # Azure does not publish connectivity uniformly
        qubit_properties=qubit_props,
        gate_properties=gate_props,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
    )


#: qb-compiler backend name -> Azure Quantum target id.
AZURE_TARGET_IDS: dict[str, str] = {
    "ionq_aria": "ionq.qpu.aria-1",
    "ionq_forte": "ionq.qpu.forte-1",
    "quantinuum_h2": "quantinuum.qpu.h2-1",
    "rigetti_ankaa": "rigetti.qpu.ankaa-3",
}


class AzureQuantumCalibrationProvider(CalibrationProvider):
    """Device-info provider backed by Azure Quantum (secondary access path).

    The live path (``Workspace.get_targets()``) needs an Azure subscription and is not yet
    hardware-validated; use :meth:`from_target` for offline / replay.
    """

    def __init__(
        self,
        backend: str,
        *,
        workspace: Any | None = None,
        resource_id: str | None = None,
        location: str | None = None,
    ) -> None:
        try:
            from azure.quantum import Workspace
        except ImportError as exc:
            raise ImportError(
                "Azure Quantum support requires azure-quantum. "
                "Install: pip install 'qb-compiler[azure]', then configure an Azure workspace."
            ) from exc
        if workspace is None:
            workspace = Workspace(resource_id=resource_id, location=location)
        target_id = AZURE_TARGET_IDS.get(backend, backend)
        target = workspace.get_targets(name=target_id)
        meta = {
            "provider_id": getattr(target, "provider_id", None),
            "capabilities": getattr(target, "capability", {}) or {},
        }
        snapshot = parse_azure_target(meta, backend=backend)
        self._snapshot = snapshot
        self._delegate = StaticCalibrationProvider(snapshot)
        self._backend = backend
        self._status = getattr(target, "current_availability", None)

    # ── CalibrationProvider interface (delegated) ─────────────────────
    def get_qubit_properties(self, qubit: int) -> QubitProperties | None:
        return self._delegate.get_qubit_properties(qubit)

    def get_gate_properties(self, gate: str, qubits: tuple[int, ...]) -> GateProperties | None:
        return self._delegate.get_gate_properties(gate, qubits)

    def get_all_qubit_properties(self) -> list[QubitProperties]:
        return self._delegate.get_all_qubit_properties()

    def get_all_gate_properties(self) -> list[GateProperties]:
        return self._delegate.get_all_gate_properties()

    @property
    def backend_name(self) -> str:
        return self._backend

    @property
    def backend_properties(self) -> BackendProperties:
        return self._snapshot

    @property
    def timestamp(self) -> datetime:
        return self._delegate.timestamp

    @property
    def device_status(self) -> str | None:
        return self._status

    # ── offline constructor (no network; for testing / replay) ────────
    @classmethod
    def from_target(
        cls, target: dict[str, Any], *, backend: str, provider: str | None = None
    ) -> AzureQuantumCalibrationProvider:
        """Build directly from an Azure target-metadata dict (no network)."""
        obj = cls.__new__(cls)
        snapshot = parse_azure_target(target, backend=backend, provider=provider)
        obj._snapshot = snapshot
        obj._delegate = StaticCalibrationProvider(snapshot)
        obj._backend = backend
        obj._status = None
        return obj
