"""Live calibration provider backed by AWS Braket device properties.

Fetches *live* device data from an Amazon Braket QPU — qubit count, connectivity, native gates, and
(where the provider publishes it) per-qubit/per-gate fidelities, T1/T2, readout error — plus the
device's online/offline status and queue depth.  The AWS-Braket analogue of the IBM live-calibration
path.

Design: the network fetch (``AwsDevice``) is kept separate from parsing. The parser is a pure
function over the device ``.properties`` dict, unit-testable offline with no AWS credentials.
Lookups delegate to :class:`StaticCalibrationProvider` over the parsed snapshot.

Requires the Braket SDK::

    pip install "qb-compiler[ionq]"   # pulls amazon-braket-sdk

then standard AWS credentials (e.g. via ``aws configure`` or environment variables).
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

#: Known qb-compiler backend names -> Braket device ARNs.  ARNs can change as devices are
#: retired/replaced; pass an explicit ARN to override.
BRAKET_DEVICE_ARNS: dict[str, str] = {
    "ionq_aria": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    "ionq_forte": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
    "rigetti_ankaa": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
    "iqm_garnet": "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
    "iqm_emerald": "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald",
}


def _f(value: Any) -> float | None:
    """Best-effort float coercion; None on anything non-numeric."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_us(value: Any) -> float | None:
    """Convert a Braket coherence time (SI seconds) to microseconds; None if not numeric."""
    seconds = _f(value)
    return seconds * 1e6 if seconds is not None else None


def _coupling_from_connectivity(paradigm: dict[str, Any], n_qubits: int) -> list[tuple[int, int]]:
    """Build a coupling map from a Braket ``paradigm.connectivity`` block."""
    conn = paradigm.get("connectivity") or {}
    if conn.get("fullyConnected"):
        return [(i, j) for i in range(n_qubits) for j in range(n_qubits) if i != j]
    graph = conn.get("connectivityGraph") or {}
    edges: list[tuple[int, int]] = []
    for src, targets in graph.items():
        try:
            u = int(str(src).lstrip("q"))
        except ValueError:
            continue
        for tgt in targets or []:
            try:
                v = int(str(tgt).lstrip("q"))
            except ValueError:
                continue
            edges.append((u, v))
    return edges


def parse_braket_properties(
    properties: dict[str, Any],
    *,
    backend: str,
    provider: str,
    timestamp: str | None = None,
) -> BackendProperties:
    """Parse a Braket device ``.properties`` dict into a :class:`BackendProperties` snapshot.

    Pure function (no network).  Robust to missing provider calibration: standardized paradigm
    data (qubit count, connectivity, native gates) is always read; per-qubit/per-gate calibration is
    parsed best-effort and degrades to ``None`` when the provider does not publish it.
    """
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties

    paradigm = properties.get("paradigm") or {}
    n_qubits = int(paradigm.get("qubitCount") or 0)
    native = paradigm.get("nativeGateSet") or []
    basis_gates = tuple(str(g).lower() for g in native)
    coupling_map = _coupling_from_connectivity(paradigm, n_qubits)

    prov = properties.get("provider") or {}
    specs = prov.get("specs") or {}  # Rigetti/IQM-style per-element
    fidelity = prov.get("fidelity") or {}  # IonQ-style device averages
    timing = prov.get("timing") or {}

    qubit_props: list[QubitProperties] = []
    gate_props: list[GateProperties] = []

    one_q = specs.get("1Q") or {}
    two_q = specs.get("2Q") or {}

    if one_q:
        # per-qubit calibration (Rigetti/IQM)
        for q_key, vals in one_q.items():
            try:
                q = int(str(q_key).lstrip("q"))
            except ValueError:
                continue
            ro = _f(vals.get("fRO"))
            qubit_props.append(
                QubitProperties(
                    qubit_id=q,
                    t1_us=_to_us(vals.get("T1")),
                    t2_us=_to_us(vals.get("T2")),
                    readout_error=(1.0 - ro) if ro is not None else None,
                )
            )
            f1q = _f(vals.get("f1QRB") or vals.get("f1Q_simultaneous_RB"))
            if f1q is not None:
                gate_props.append(GateProperties(gate_type="1q", qubits=(q,), error_rate=1.0 - f1q))
        for edge_key, vals in two_q.items():
            qs = [int(x.lstrip("q")) for x in str(edge_key).split("-") if x.lstrip("q").isdigit()]
            f2q = _f(vals.get("fCZ") or vals.get("fCPHASE") or vals.get("f2Q"))
            if len(qs) == 2 and f2q is not None:
                gate_props.append(
                    GateProperties(gate_type="2q", qubits=tuple(qs), error_rate=1.0 - f2q)
                )
    elif fidelity or timing:
        # device-average calibration (IonQ): apply uniformly across qubits (honestly an average)
        f1 = _f((fidelity.get("1Q") or {}).get("mean"))
        f2 = _f((fidelity.get("2Q") or {}).get("mean"))
        spam = _f((fidelity.get("spam") or {}).get("mean"))
        t1_us = _to_us(timing.get("T1"))
        t2_us = _to_us(timing.get("T2"))
        ro_err = (1.0 - spam) if spam is not None else None
        for q in range(n_qubits):
            qubit_props.append(
                QubitProperties(qubit_id=q, t1_us=t1_us, t2_us=t2_us, readout_error=ro_err)
            )
            if f1 is not None:
                gate_props.append(GateProperties(gate_type="1q", qubits=(q,), error_rate=1.0 - f1))
        if f2 is not None:
            for u, v in coupling_map:
                gate_props.append(
                    GateProperties(gate_type="2q", qubits=(u, v), error_rate=1.0 - f2)
                )

    return BackendProperties(
        backend=backend,
        provider=provider,
        n_qubits=n_qubits,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        qubit_properties=qubit_props,
        gate_properties=gate_props,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
    )


class BraketCalibrationProvider(CalibrationProvider):
    """Calibration provider that pulls live device data from AWS Braket.

    Parameters
    ----------
    device:
        A known qb-compiler backend name (see :data:`BRAKET_DEVICE_ARNS`) or a Braket device ARN.
    provider:
        Vendor name for the snapshot (inferred from the backend name when omitted).
    """

    def __init__(self, device: str, *, provider: str | None = None) -> None:
        arn = BRAKET_DEVICE_ARNS.get(device, device)
        backend = device if device in BRAKET_DEVICE_ARNS else arn.rsplit("/", 1)[-1]
        provider = provider or _infer_provider(arn)

        try:
            from braket.aws import AwsDevice
        except ImportError as exc:
            raise ImportError(
                "Braket calibration requires the Braket SDK. "
                "Install with: pip install 'qb-compiler[ionq]' (pulls amazon-braket-sdk), "
                "then configure AWS credentials."
            ) from exc

        aws_device = AwsDevice(arn)
        props_dict = aws_device.properties.dict() if aws_device.properties is not None else {}
        snapshot = parse_braket_properties(props_dict, backend=backend, provider=provider)

        self._snapshot = snapshot
        self._delegate = StaticCalibrationProvider(snapshot)
        self._backend = backend
        self._arn = arn
        self._status = getattr(aws_device, "status", None)
        try:
            self._queue_depth: Any = aws_device.queue_depth()
        except Exception:
            self._queue_depth = None

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
        """The parsed live-device calibration snapshot (for passing to ``check_viability``)."""
        return self._snapshot

    @property
    def timestamp(self) -> datetime:
        return self._delegate.timestamp

    # ── live availability signals ─────────────────────────────────────
    @property
    def device_status(self) -> str | None:
        """Device availability, e.g. ``"ONLINE"`` / ``"OFFLINE"`` (None if unknown)."""
        return str(self._status) if self._status is not None else None

    @property
    def queue_depth(self) -> Any:
        """Braket queue depth for the device (None if unavailable)."""
        return self._queue_depth

    @classmethod
    def from_device_properties(
        cls,
        properties: dict[str, Any],
        *,
        backend: str,
        provider: str | None = None,
        status: str | None = None,
    ) -> BraketCalibrationProvider:
        """Build a provider directly from a device ``.properties`` dict (no network).

        Useful for offline use / testing and for replaying a captured device snapshot.
        """
        obj = cls.__new__(cls)
        provider = provider or _infer_provider(backend)
        snapshot = parse_braket_properties(properties, backend=backend, provider=provider)
        obj._snapshot = snapshot
        obj._delegate = StaticCalibrationProvider(snapshot)
        obj._backend = backend
        obj._arn = BRAKET_DEVICE_ARNS.get(backend, backend)
        obj._status = status
        obj._queue_depth = None
        return obj


def _infer_provider(name: str) -> str:
    n = name.lower()
    if "ionq" in n:
        return "ionq"
    if "rigetti" in n:
        return "rigetti"
    if "iqm" in n:
        return "iqm"
    return "braket"
