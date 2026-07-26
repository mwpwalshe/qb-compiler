"""Live calibration provider backed by IBM Quantum (qiskit-ibm-runtime).

Fetches *live* device calibration straight from IBM via ``qiskit-ibm-runtime``, qubit
T1/T2/frequency, per-gate error/duration, and readout error: from the backend's qiskit ``Target``.
This is the open-source IBM live path: it needs only ``qiskit-ibm-runtime`` + a saved IBM Quantum
account, with no proprietary QubitBoost hub.

Design follows the Braket provider: the network fetch is kept separate from parsing.
``parse_ibm_target`` is a pure function over a qiskit ``Target`` (qiskit is a core dependency), so
it is fully unit-testable offline with a hand-built ``Target`` and no IBM credentials. Lookups
delegate to
:class:`StaticCalibrationProvider`.

Requires the IBM extra::

    pip install "qb-compiler[ibm]"   # pulls qiskit-ibm-runtime

then a saved account, e.g.::

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...")
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

#: Target operations that are not calibratable gates (skipped when reading gate errors).
_NON_GATE_OPS = frozenset({"measure", "reset", "delay", "barrier"})


def _f(value: Any) -> float | None:
    """Best-effort float coercion; None on non-numeric / None."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _to_us(value: Any) -> float | None:
    """Convert an SI-seconds coherence time to microseconds; None if not numeric."""
    s = _f(value)
    return s * 1e6 if s is not None else None


def _to_ns(value: Any) -> float | None:
    """Convert an SI-seconds gate duration to nanoseconds; None if not numeric."""
    s = _f(value)
    return s * 1e9 if s is not None else None


def _to_ghz(value: Any) -> float | None:
    """Convert an Hz frequency to GHz; None if not numeric."""
    hz = _f(value)
    return hz / 1e9 if hz is not None else None


def parse_ibm_target(
    target: Any,
    *,
    backend: str,
    provider: str = "ibm",
    timestamp: str | None = None,
) -> BackendProperties:
    """Parse a qiskit ``Target`` into a :class:`BackendProperties` snapshot.

    Pure function (no network). Robust to missing fields: ``qubit_properties`` may be ``None`` or
    ``None`` entries; t1/t2/frequency may be ``None``; ``build_coupling_map()`` may return ``None``;
    instruction-property entries may be ``None``.
    """
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties

    n_qubits = int(getattr(target, "num_qubits", 0) or 0)
    op_names = list(getattr(target, "operation_names", []) or [])
    basis_gates = tuple(op for op in op_names if op not in _NON_GATE_OPS)

    # coupling map (may be None / raise)
    coupling_map: list[tuple[int, int]] = []
    try:
        cmap = target.build_coupling_map()
        if cmap is not None:
            coupling_map = [(int(u), int(v)) for u, v in cmap.get_edges()]
    except Exception:
        coupling_map = []

    # readout error per qubit from the measure instruction
    readout: dict[int, float] = {}
    if "measure" in op_names:
        try:
            for qtuple, instprop in (target["measure"] or {}).items():
                if instprop is not None and len(qtuple) == 1:
                    err = _f(getattr(instprop, "error", None))
                    if err is not None:
                        readout[int(qtuple[0])] = err
        except Exception:
            pass

    qubit_props_src = getattr(target, "qubit_properties", None) or []
    qprops: list[QubitProperties] = []
    for q in range(n_qubits):
        qp = qubit_props_src[q] if q < len(qubit_props_src) else None
        qprops.append(
            QubitProperties(
                qubit_id=q,
                t1_us=_to_us(getattr(qp, "t1", None)) if qp is not None else None,
                t2_us=_to_us(getattr(qp, "t2", None)) if qp is not None else None,
                readout_error=readout.get(q),
                frequency_ghz=_to_ghz(getattr(qp, "frequency", None)) if qp is not None else None,
            )
        )

    gprops: list[GateProperties] = []
    for op_name in op_names:
        if op_name in _NON_GATE_OPS:
            continue
        try:
            inst_map = target[op_name] or {}
        except Exception:
            continue
        for qtuple, instprop in inst_map.items():
            if instprop is None:
                continue
            err = _f(getattr(instprop, "error", None))
            dur = _to_ns(getattr(instprop, "duration", None))
            if err is None and dur is None:
                continue
            gprops.append(
                GateProperties(
                    gate_type=op_name,
                    qubits=tuple(int(q) for q in qtuple),
                    error_rate=err,
                    gate_time_ns=dur,
                )
            )

    return BackendProperties(
        backend=backend,
        provider=provider,
        n_qubits=n_qubits,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        qubit_properties=qprops,
        gate_properties=gprops,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
    )


class IBMRuntimeCalibrationProvider(CalibrationProvider):
    """Calibration provider that pulls live device data from IBM via qiskit-ibm-runtime.

    Parameters
    ----------
    backend:
        IBM backend name (e.g. ``"ibm_fez"``).
    service:
        An existing ``QiskitRuntimeService`` to reuse. When ``None``, one is created from the saved
        account (optionally overridden by ``channel`` / ``instance`` / ``token``).
    """

    def __init__(
        self,
        backend: str,
        *,
        service: Any | None = None,
        channel: str | None = None,
        instance: str | None = None,
        token: str | None = None,
    ) -> None:
        if service is None:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
            except ImportError as exc:
                raise ImportError(
                    "IBM live calibration requires qiskit-ibm-runtime. "
                    "Install: pip install 'qb-compiler[ibm]', then save an IBM Quantum account "
                    "(QiskitRuntimeService.save_account(...))."
                ) from exc
            kwargs: dict[str, Any] = {}
            if channel is not None:
                kwargs["channel"] = channel
            if instance is not None:
                kwargs["instance"] = instance
            if token is not None:
                kwargs["token"] = token
            service = QiskitRuntimeService(**kwargs)

        ibm_backend = service.backend(backend)
        snapshot = parse_ibm_target(ibm_backend.target, backend=backend)
        self._snapshot = snapshot
        self._delegate = StaticCalibrationProvider(snapshot)
        self._backend = backend
        self._status: str | None = None
        self._pending_jobs: int | None = None
        try:
            status = ibm_backend.status()
            self._status = "ONLINE" if getattr(status, "operational", None) else "OFFLINE"
            self._pending_jobs = getattr(status, "pending_jobs", None)
        except Exception:
            pass

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
        """``"ONLINE"`` / ``"OFFLINE"`` (None if unknown)."""
        return self._status

    @property
    def pending_jobs(self) -> int | None:
        """IBM queue depth (pending jobs) for the backend, if available."""
        return self._pending_jobs

    # ── offline constructors (no network; for testing / replay) ───────
    @classmethod
    def from_target(
        cls, target: Any, *, backend: str, provider: str = "ibm"
    ) -> IBMRuntimeCalibrationProvider:
        """Build directly from a qiskit ``Target`` (no network)."""
        obj = cls.__new__(cls)
        snapshot = parse_ibm_target(target, backend=backend, provider=provider)
        obj._snapshot = snapshot
        obj._delegate = StaticCalibrationProvider(snapshot)
        obj._backend = backend
        obj._status = None
        obj._pending_jobs = None
        return obj

    @classmethod
    def from_backend(cls, ibm_backend: Any) -> IBMRuntimeCalibrationProvider:
        """Build from an already-obtained qiskit ``BackendV2`` (no service call)."""
        name = getattr(ibm_backend, "name", "ibm_unknown")
        return cls.from_target(ibm_backend.target, backend=str(name))
