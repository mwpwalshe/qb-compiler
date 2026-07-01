"""Calibration provider for Quantinuum H-series via pytket-quantinuum.

Quantinuum publishes device calibration through pytket's ``BackendInfo`` (gate set, per-qubit /
per-edge gate errors, readout errors). This provider parses that into a :class:`BackendProperties`
snapshot. Following the Braket template, ``parse_quantinuum_info`` is a pure function over a pytket
``BackendInfo`` (a real, importable type), so it is fully unit-testable offline with a hand-built
``BackendInfo`` and no Quantinuum account.

Status: the offline parser is tested; the live network path (fetching ``backend_info`` from a real
Quantinuum device) requires authentication and has NOT yet been validated on hardware. Quantinuum's
``BackendInfo`` does not report T1/T2/frequency, so those fields are ``None`` (honest).

Requires::

    pip install "qb-compiler[quantinuum]"   # pulls pytket-quantinuum
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


def _node_id(node: Any, fallback: int) -> int:
    """Best-effort integer qubit id from a pytket Node (``index`` is a tuple like ``(3,)``)."""
    idx = getattr(node, "index", None)
    if idx:
        try:
            return int(idx[0])
        except (TypeError, ValueError, IndexError):
            pass
    return fallback


def _readout_err_from_matrix(matrix: Any) -> float | None:
    """1 - mean(diagonal) of a confusion matrix [[p00,p01],[p10,p11]]; None if unparsable."""
    try:
        diag = [float(matrix[i][i]) for i in range(len(matrix))]
        return 1.0 - sum(diag) / len(diag)
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def parse_quantinuum_info(
    info: Any,
    *,
    backend: str,
    provider: str = "quantinuum",
    timestamp: str | None = None,
) -> BackendProperties:
    """Parse a pytket ``BackendInfo`` into a :class:`BackendProperties` snapshot (pure, no network).

    Robust to missing calibration: H-series is all-to-all (empty coupling map = unconstrained), and
    T1/T2/frequency are not reported by Quantinuum (left ``None``). Readout + gate errors are read
    from the per-node/per-edge tables, falling back to the averaged tables.
    """
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties

    arch = getattr(info, "architecture", None)
    nodes = list(getattr(arch, "nodes", []) or [])
    # stable node -> qubit id map (sorted by pytket index)
    nodes_sorted = sorted(nodes, key=lambda n: _node_id(n, 0))
    node_to_id = {n: _node_id(n, i) for i, n in enumerate(nodes_sorted)}
    n_qubits = len(nodes_sorted)

    gate_set = getattr(info, "gate_set", None) or []
    basis_gates = tuple(sorted({str(getattr(g, "name", g)).lower() for g in gate_set}))

    # coupling: all-to-all H-series usually has no explicit coupling -> empty (unconstrained)
    coupling_map: list[tuple[int, int]] = []
    coupling = getattr(arch, "coupling", None)
    if coupling:
        for u, v in coupling:
            if u in node_to_id and v in node_to_id:
                coupling_map.append((node_to_id[u], node_to_id[v]))

    all_node = getattr(info, "all_node_gate_errors", None) or {}
    avg_node = getattr(info, "averaged_node_gate_errors", None) or {}
    all_ro = getattr(info, "all_readout_errors", None) or {}
    avg_ro = getattr(info, "averaged_readout_errors", None) or {}
    all_edge = getattr(info, "all_edge_gate_errors", None) or {}
    avg_edge = getattr(info, "averaged_edge_gate_errors", None) or {}

    qubit_props: list[QubitProperties] = []
    gate_props: list[GateProperties] = []

    for node in nodes_sorted:
        qid = node_to_id[node]
        # readout error: averaged table first, then confusion matrix
        ro = _f(avg_ro.get(node)) if avg_ro else None
        if ro is None and node in all_ro:
            ro = _readout_err_from_matrix(all_ro.get(node))
        qubit_props.append(QubitProperties(qubit_id=qid, t1_us=None, t2_us=None, readout_error=ro))
        # 1q gate error: averaged, else max across op types
        e1 = _f(avg_node.get(node)) if avg_node else None
        if e1 is None and node in all_node:
            raw1 = [_f(v) for v in (all_node.get(node) or {}).values()]
            clean1 = [v for v in raw1 if v is not None]
            e1 = max(clean1) if clean1 else None
        if e1 is not None:
            gate_props.append(GateProperties(gate_type="1q", qubits=(qid,), error_rate=e1))

    # 2q gate errors per edge (averaged table first)
    edge_src = avg_edge if avg_edge else all_edge
    for edge_key, val in edge_src.items():
        try:
            u, v = edge_key
        except (TypeError, ValueError):
            continue
        if u not in node_to_id or v not in node_to_id:
            continue
        if isinstance(val, dict):  # all_edge: {OpType: err}
            raw2 = [_f(x) for x in val.values()]
            clean2 = [x for x in raw2 if x is not None]
            e2 = max(clean2) if clean2 else None
        else:  # averaged_edge: float
            e2 = _f(val)
        if e2 is not None:
            gate_props.append(
                GateProperties(gate_type="2q", qubits=(node_to_id[u], node_to_id[v]), error_rate=e2)
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


class QuantinuumCalibrationProvider(CalibrationProvider):
    """Provider that pulls Quantinuum H-series calibration via pytket-quantinuum.

    The live path (``QuantinuumBackend(device).backend_info``) needs authentication and is not yet
    hardware-validated; use :meth:`from_backend_info` for offline / replay.
    """

    def __init__(self, backend: str, *, device_name: str | None = None) -> None:
        try:
            from pytket.extensions.quantinuum import QuantinuumBackend
        except ImportError as exc:
            raise ImportError(
                "Quantinuum calibration requires pytket-quantinuum. "
                "Install: pip install 'qb-compiler[quantinuum]', then log in to your "
                "Quantinuum account."
            ) from exc
        qb = QuantinuumBackend(device_name=device_name or backend)
        info = qb.backend_info
        snapshot = parse_quantinuum_info(info, backend=backend)
        self._snapshot = snapshot
        self._delegate = StaticCalibrationProvider(snapshot)
        self._backend = backend

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

    # ── offline constructor (no network; for testing / replay) ────────
    @classmethod
    def from_backend_info(
        cls, info: Any, *, backend: str, provider: str = "quantinuum"
    ) -> QuantinuumCalibrationProvider:
        """Build directly from a pytket ``BackendInfo`` (no network)."""
        obj = cls.__new__(cls)
        snapshot = parse_quantinuum_info(info, backend=backend, provider=provider)
        obj._snapshot = snapshot
        obj._delegate = StaticCalibrationProvider(snapshot)
        obj._backend = backend
        return obj
