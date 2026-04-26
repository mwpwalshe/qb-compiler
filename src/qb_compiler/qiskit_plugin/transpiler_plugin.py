"""Qiskit transpiler integration for qb-compiler.

Provides :class:`QBCalibrationLayout`, a Qiskit ``AnalysisPass`` that uses
calibration data (T1, T2, readout error, gate error) to assign virtual qubits
to the best-scoring physical qubits.  Also provides a convenience function
:func:`qb_transpile` that wires everything together.

Usage::

    from qb_compiler.qiskit_plugin import QBCalibrationLayout, qb_transpile

    # Option A: Drop into a Qiskit pass manager
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    pm.layout.append(QBCalibrationLayout(calibration_data))

    # Option B: One-liner
    compiled = qb_transpile(circuit, backend="ibm_fez", calibration_path="cal.json")
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.layout import Layout

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Qubit scoring
# ═══════════════════════════════════════════════════════════════════════


def _score_qubit(qubit_data: dict[str, Any]) -> float:
    """Score a physical qubit — *lower* is better.

    Combines T1, T2, readout error, and gate errors into a single quality
    metric. Single-qubit and two-qubit gate errors are tracked separately
    and weighted differently because they contribute to circuit fidelity at
    very different scales (typical 2q error ~5e-3 to 2e-2, typical 1q error
    ~1e-4 to 1e-3 — a 10-100x ratio). Pooling them into a single arithmetic
    mean (qb-compiler ≤0.5.0) caused the 1q signal to dilute the 2q signal
    when full-coverage calibration data was supplied, regressing chain
    selection on dense-1q workloads (UCCSD, HEA) by ~5-7% estimated fidelity
    on IBM Fez. v0.5.1 separates the tracks; weights tuned empirically against
    ``experiments/qb_compiler_v0_5_benchmarks/bench_layout_quality.py``.

    Missing values fall through to pessimistic defaults so qubits with
    incomplete calibration are deprioritised. Backward-compatible with the
    legacy ``gate_error`` key (treated as 2-qubit) for v0.4-fixture inputs.

    Score formula (v0.5.1)::

        score = w_ro  * readout_error
              + w_t1  * (1 / T1)
              + w_t2  * (1 / T2)
              + w_2q  * avg_2q_gate_error
              + w_1q  * avg_1q_gate_error

    Default weights: w_ro=0.35, w_t1=0.10, w_t2=0.10, w_2q=0.40, w_1q=0.05.
    """
    # Weights (v0.5.1) — 2q error is the only gate-error term that contributes
    # to chain selection. The 1q error track is captured but weighted at zero
    # because the current chain selector is connectivity-blind: it picks the
    # N best-scoring qubits regardless of whether they form a connected
    # subgraph on the device topology, and giving the 1q signal any weight
    # at all reduces the score's discrimination on the 2q dimension that
    # actually matters for chain choice. A connectivity-aware scorer that
    # can productively consume the 1q signal is scheduled for v0.6.
    w_ro = 0.40
    w_t1 = 0.10
    w_t2 = 0.10
    w_2q = 0.40
    w_1q = 0.00

    t1 = qubit_data.get("t1_us") or 50.0
    t2 = qubit_data.get("t2_us") or 25.0
    readout_err = qubit_data.get("readout_error") or 0.05

    # Backward compat: if caller passes legacy "gate_error" (single track,
    # v0.5.0 interface), treat it as a 2-qubit signal because that's what the
    # v0.4 fixture path supplied.
    err_2q = qubit_data.get("gate_error_2q")
    err_1q = qubit_data.get("gate_error_1q")
    if err_2q is None and "gate_error" in qubit_data:
        err_2q = qubit_data.get("gate_error")
    if err_2q is None:
        err_2q = 0.01
    if err_1q is None:
        err_1q = 0.001

    # Normalise T1/T2 into error-like quantities (larger T → smaller contribution)
    t1_contribution = 1.0 / max(t1, 1.0)
    t2_contribution = 1.0 / max(t2, 1.0)

    return (
        w_ro * readout_err
        + w_t1 * t1_contribution
        + w_t2 * t2_contribution
        + w_2q * err_2q
        + w_1q * err_1q
    )


def _load_calibration_dict(source: dict | str | Path) -> dict:
    """Normalise *source* into a parsed calibration dict."""
    if isinstance(source, (str, Path)):
        with open(source) as fh:
            return json.load(fh)  # type: ignore[no-any-return]
    if isinstance(source, dict):
        return source
    raise TypeError(f"calibration_data must be a dict or path, got {type(source)}")


def _provider_to_dict(provider: object, backend_hint: str | None = None) -> dict:
    """Materialise a CalibrationProvider snapshot into the dict format
    consumed by :class:`QBCalibrationLayout`.

    Used internally by :func:`qb_transpile` when called with
    ``calibration_provider=...``. Walks the provider's ``get_all_*``
    methods and emits a qb-compiler-compatible calibration dict
    (top-level ``qubit_properties``, ``gate_properties``,
    ``coupling_map``, ``basis_gates``).

    The provider's underlying ``BackendProperties`` (when accessible
    via ``_props`` per :class:`StaticCalibrationProvider`) is preferred
    because it preserves the coupling_map exactly. Otherwise we
    reconstruct from per-qubit / per-gate getters.
    """
    # Preferred path: provider wraps a BackendProperties (e.g. StaticCalibrationProvider).
    # LiveCalibrationProvider wraps a StaticCalibrationProvider in turn, so drill
    # one level deeper if needed.
    underlying = getattr(provider, "_props", None)
    if underlying is None:
        snapshot = getattr(provider, "_snapshot", None)
        if snapshot is not None:
            underlying = getattr(snapshot, "_props", None)
    if underlying is not None:
        return {
            "backend_name": getattr(underlying, "backend", backend_hint or "unknown"),
            "n_qubits": getattr(underlying, "n_qubits", None),
            "timestamp": getattr(underlying, "timestamp", None),
            "basis_gates": list(getattr(underlying, "basis_gates", [])),
            "coupling_map": [list(e) for e in getattr(underlying, "coupling_map", [])],
            "qubit_properties": [
                {
                    "qubit": q.qubit_id,
                    "T1": q.t1_us,
                    "T2": q.t2_us,
                    "frequency": q.frequency_ghz,
                    "readout_error": q.readout_error,
                    "readout_error_0to1": q.readout_error_0to1,
                    "readout_error_1to0": q.readout_error_1to0,
                }
                for q in underlying.qubit_properties
            ],
            "gate_properties": [
                {
                    "gate": g.gate_type,
                    "qubits": list(g.qubits),
                    "parameters": {
                        "gate_error": g.error_rate,
                        "gate_length": g.gate_time_ns,
                    },
                }
                for g in underlying.gate_properties
            ],
        }

    # Fallback path: walk the abstract provider interface
    return {
        "backend_name": getattr(provider, "backend_name", backend_hint or "unknown"),
        "timestamp": str(getattr(provider, "timestamp", "")),
        "qubit_properties": [
            {
                "qubit": q.qubit_id,
                "T1": q.t1_us,
                "T2": q.t2_us,
                "readout_error": q.readout_error,
                "readout_error_0to1": getattr(q, "readout_error_0to1", None),
                "readout_error_1to0": getattr(q, "readout_error_1to0", None),
            }
            for q in provider.get_all_qubit_properties()
        ],
        "gate_properties": [
            {
                "gate": g.gate_type,
                "qubits": list(g.qubits),
                "parameters": {
                    "gate_error": g.error_rate,
                    "gate_length": g.gate_time_ns,
                },
            }
            for g in provider.get_all_gate_properties()
        ],
    }


def _build_qubit_scores(cal_data: dict) -> dict[int, float]:
    """Build ``{physical_qubit: score}`` from a QubitBoost calibration dict.

    v0.5.1: separates single-qubit (1q) and two-qubit (2q) gate-error tracks
    so :func:`_score_qubit` can weight them at appropriate magnitudes. v0.4
    fixture format (2q-only ``gate_properties``) populates the 2q track only;
    1q track stays empty and falls through to default. v0.5 live-fetch format
    (full coverage) populates both. Either way the layout-selection ranking
    is dominated by the 2q signal, which is the load-bearing signal for
    chain selection on every benchmarked NISQ workload class.
    """
    qubit_props = cal_data.get("qubit_properties", [])

    # Track 1q and 2q gate errors separately
    gate_errors_1q: dict[int, list[float]] = {}
    gate_errors_2q: dict[int, list[float]] = {}
    for gp in cal_data.get("gate_properties", []):
        err = (gp.get("parameters") or {}).get("gate_error")
        if err is None:
            continue
        qubits = gp.get("qubits", [])
        if len(qubits) == 1:
            gate_errors_1q.setdefault(qubits[0], []).append(err)
        elif len(qubits) >= 2:
            for q in qubits:
                gate_errors_2q.setdefault(q, []).append(err)

    scores: dict[int, float] = {}
    for qp in qubit_props:
        qid = qp["qubit"]
        err_01 = qp.get("readout_error_0to1")
        err_10 = qp.get("readout_error_1to0")
        if err_01 is not None and err_10 is not None:
            ro_err = (err_01 + err_10) / 2.0
        else:
            ro_err = err_01 or err_10 or qp.get("readout_error") or None

        avg_2q = None
        if gate_errors_2q.get(qid):
            avg_2q = sum(gate_errors_2q[qid]) / len(gate_errors_2q[qid])
        avg_1q = None
        if gate_errors_1q.get(qid):
            avg_1q = sum(gate_errors_1q[qid]) / len(gate_errors_1q[qid])

        scores[qid] = _score_qubit(
            {
                "t1_us": qp.get("T1"),
                "t2_us": qp.get("T2"),
                "readout_error": ro_err,
                "gate_error_2q": avg_2q,
                "gate_error_1q": avg_1q,
            }
        )
    return scores


# ═══════════════════════════════════════════════════════════════════════
# QBCalibrationLayout — Qiskit AnalysisPass
# ═══════════════════════════════════════════════════════════════════════


class QBCalibrationLayout(AnalysisPass):
    """Calibration-aware layout pass for Qiskit.

    Scores every physical qubit using combined T1/T2/readout/gate-error
    metrics from a QubitBoost calibration snapshot, then maps virtual
    qubits to the best physical qubits.

    Parameters
    ----------
    calibration_data:
        Either a parsed calibration dict (matching the QubitBoost
        ``calibration_hub`` JSON schema) or a filesystem path to one.
    """

    def __init__(self, calibration_data: dict | str | Path) -> None:
        super().__init__()
        self._cal_data = _load_calibration_dict(calibration_data)
        self._scores = _build_qubit_scores(self._cal_data)

    def run(self, dag: Any) -> None:
        """Set ``property_set["layout"]`` to a calibration-optimal mapping.

        v0.5.1: connectivity-aware. Extracts the circuit's 2q interaction
        graph from the DAG, then uses ``rustworkx.vf2_mapping`` to enumerate
        candidate subgraph isomorphisms onto the device coupling map. Each
        candidate is scored as ``sum(per-qubit scores) + sum(per-edge gate
        errors weighted by interaction count)``; lowest-scoring layout wins.
        Falls back to topology-blind top-N selection if VF2 finds nothing
        (e.g. circuit interaction graph isn't a subgraph of the device
        coupling map, or rustworkx is unavailable).

        v0.5.0 used a topology-blind selector that picked the N best-scoring
        qubits regardless of connectivity. On dense-2q circuits (UCCSD, HEA)
        this often picked qubits scattered across the chip, forcing the
        downstream router to insert many SWAPs and crashing the post-routing
        fidelity. v0.5.1 closes that gap.
        """
        n_virtual = dag.num_qubits()

        if not self._scores:
            logger.warning("QBCalibrationLayout: no qubit scores available, skipping layout")
            return

        if len(self._scores) < n_virtual:
            logger.warning(
                "QBCalibrationLayout: circuit needs %d qubits but calibration "
                "only covers %d — skipping layout",
                n_virtual, len(self._scores),
            )
            return

        # Try connectivity-aware VF2 search. Falls through to topology-blind
        # selection on any failure (missing rustworkx, no isomorphism, etc.).
        layout_phys = self._vf2_calibration_aware(dag, n_virtual)
        if layout_phys is None:
            ranked = sorted(self._scores.items(), key=lambda kv: kv[1])
            layout_phys = [qid for qid, _ in ranked[:n_virtual]]
            logger.info(
                "QBCalibrationLayout: VF2 didn't find an isomorphism — falling "
                "back to topology-blind top-%d. Layout: %s", n_virtual, layout_phys,
            )
        else:
            logger.info(
                "QBCalibrationLayout: VF2 calibration-aware layout: %s", layout_phys,
            )

        layout_dict = {v_qubit: layout_phys[v_idx]
                       for v_idx, v_qubit in enumerate(dag.qubits)}
        self.property_set["layout"] = Layout(layout_dict)

    def _vf2_calibration_aware(self, dag: Any, n_virtual: int) -> list[int] | None:
        """VF2-based connectivity-aware layout selection. Returns None on failure."""
        try:
            import rustworkx as rx
        except ImportError:
            return None

        coupling = self._cal_data.get("coupling_map") or []
        if not coupling:
            return None

        # Build per-edge 2q gate error map (best across both directions)
        edge_err: dict[tuple[int, int], float] = {}
        for gp in self._cal_data.get("gate_properties", []):
            qubits = gp.get("qubits", [])
            if len(qubits) != 2:
                continue
            err = (gp.get("parameters") or {}).get("gate_error")
            if err is None:
                continue
            a, b = sorted((int(qubits[0]), int(qubits[1])))
            cur = edge_err.get((a, b), float("inf"))
            edge_err[(a, b)] = min(cur, float(err))

        # Build the device coupling graph (rustworkx undirected)
        nodes = sorted({int(q) for edge in coupling for q in edge})
        node_to_idx = {q: i for i, q in enumerate(nodes)}
        device = rx.PyGraph()
        device.add_nodes_from(nodes)
        seen_edges: set[tuple[int, int]] = set()
        for edge in coupling:
            a, b = int(edge[0]), int(edge[1])
            key = (min(a, b), max(a, b))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            device.add_edge(node_to_idx[a], node_to_idx[b], None)

        # Build the circuit's 2q interaction graph from the DAG
        interactions: dict[tuple[int, int], int] = {}
        for node in dag.op_nodes():
            qubits = [dag.find_bit(q).index for q in node.qargs]
            if len(qubits) == 2:
                a, b = sorted(qubits)
                interactions[(a, b)] = interactions.get((a, b), 0) + 1

        circuit_graph = rx.PyGraph()
        circuit_nodes = list(range(n_virtual))
        circuit_graph.add_nodes_from(circuit_nodes)
        for (a, b), _ in interactions.items():
            circuit_graph.add_edge(a, b, None)

        # If circuit has no 2q interactions, fall through to topology-blind
        # (no connectivity constraint to satisfy)
        if circuit_graph.num_edges() == 0:
            return None

        # Enumerate VF2 isomorphisms (subgraph isomorphism, undirected).
        # rustworkx convention: vf2_mapping(big, small, subgraph=True) returns
        # mappings {big_node_idx: small_node_idx}. We want to find circuit_graph
        # as a subgraph of device, so device is "big", circuit is "small".
        # Limit work; we want the best of a reasonable number of candidates.
        try:
            mappings = rx.vf2_mapping(
                device, circuit_graph, subgraph=True, induced=False, id_order=False,
            )
        except Exception:
            return None

        best_score = float("inf")
        best_layout: list[int] | None = None
        n_inspected = 0
        max_inspect = 256  # bound runtime
        for m in mappings:
            n_inspected += 1
            # m: dict[device_node_idx -> circuit_node_idx]. Invert to
            # circuit_node_idx -> physical qubit (via the nodes lookup).
            inv = {circuit_v: device_node_idx for device_node_idx, circuit_v in m.items()}
            phys = [nodes[inv[v]] for v in range(n_virtual)]
            # Per-qubit score sum
            score = sum(self._scores.get(p, 1.0) for p in phys)
            # Per-edge score
            for (la, lb), count in interactions.items():
                pa, pb = sorted((phys[la], phys[lb]))
                edge_e = edge_err.get((pa, pb))
                if edge_e is None:
                    score += 0.01 * count  # penalty for missing edge data
                else:
                    score += edge_e * count
            if score < best_score:
                best_score = score
                best_layout = phys
            if n_inspected >= max_inspect:
                break

        return best_layout


# ═══════════════════════════════════════════════════════════════════════
# QBCalibrationLayoutPlugin — Qiskit transpiler-stage plugin
# ═══════════════════════════════════════════════════════════════════════


class QBCalibrationLayoutPlugin:
    """Qiskit transpiler stage plugin for calibration-aware layout.

    Registered in ``pyproject.toml`` under
    ``[project.entry-points."qiskit.transpiler.layout"]`` as
    ``qb_calibration``.  Invoke via::

        pm = generate_preset_pass_manager(
            optimization_level=2,
            backend=backend,
            layout_method="qb_calibration",
        )

    At plugin-invoke time the calibration snapshot is loaded from the
    ``QB_CALIBRATION_PATH`` environment variable.  If the variable is
    unset, the plugin returns an empty ``PassManager`` (passthrough),
    which lets the preset pass manager fall through to its default
    layout finder.
    """

    def pass_manager(
        self,
        pass_manager_config: Any,  # qiskit.transpiler.PassManagerConfig
        optimization_level: int | None = None,
    ) -> Any:
        """Return a ``PassManager`` implementing the full layout stage.

        A layout-stage plugin must replace Qiskit's entire default layout
        stage, which finds a layout AND dilates the circuit with ancillas
        so that subsequent routing/translation passes have the correct
        number of physical qubits.  This plugin runs, in order:

        1. :class:`QBCalibrationLayout` — set ``property_set["layout"]``
           from calibration-aware scoring.  Falls back to
           :class:`~qiskit.transpiler.passes.TrivialLayout` if
           ``QB_CALIBRATION_PATH`` is not set.
        2. :class:`~qiskit.transpiler.passes.FullAncillaAllocation` —
           reserve physical qubits not touched by the layout.
        3. :class:`~qiskit.transpiler.passes.EnlargeWithAncilla` —
           extend the circuit with those ancillas.
        4. :class:`~qiskit.transpiler.passes.ApplyLayout` — rewrite
           the DAG onto physical qubits.
        """
        import os

        from qiskit.transpiler import PassManager as QiskitPM
        from qiskit.transpiler.passes import (
            ApplyLayout,
            EnlargeWithAncilla,
            FullAncillaAllocation,
            TrivialLayout,
        )

        coupling_map = pass_manager_config.coupling_map
        cal_path = os.environ.get("QB_CALIBRATION_PATH")

        if cal_path:
            layout_pass: Any = QBCalibrationLayout(cal_path)
        else:
            # Passthrough: behave like the default trivial layout so the
            # stage still produces a valid ancilla-dilated circuit when
            # no calibration is available.
            layout_pass = TrivialLayout(coupling_map)

        stage = [layout_pass]
        if coupling_map is not None:
            stage.extend(
                [
                    FullAncillaAllocation(coupling_map),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ]
            )
        return QiskitPM(stage)


class QBTranspilerPlugin(QBCalibrationLayoutPlugin):
    """Deprecated alias kept for backward compatibility.

    .. deprecated:: 0.3.0
        Use :class:`QBCalibrationLayoutPlugin` with
        ``generate_preset_pass_manager(layout_method="qb_calibration")``
        and ``QB_CALIBRATION_PATH`` env var, or call :func:`qb_transpile`
        directly.  ``QBTranspilerPlugin.get_pass_manager()`` will be
        removed in 0.4.0.
    """

    def get_pass_manager(
        self,
        *,
        calibration_data: dict | str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a ``PassManager`` with calibration-aware layout (legacy API)."""
        warnings.warn(
            "QBTranspilerPlugin.get_pass_manager() is deprecated; use "
            "generate_preset_pass_manager(layout_method='qb_calibration') "
            "with QB_CALIBRATION_PATH set, or call qb_transpile() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        from qiskit.transpiler import PassManager as QiskitPM

        if calibration_data is not None:
            return QiskitPM([QBCalibrationLayout(calibration_data)])
        return QiskitPM()


# ═══════════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════════


def qb_transpile(
    circuit: QuantumCircuit,
    backend: str | None = None,
    calibration_path: str | Path | None = None,
    calibration_data: dict | None = None,
    calibration_provider: object | None = None,
    optimization_level: int = 2,
) -> QuantumCircuit:
    """Transpile a Qiskit circuit with calibration-aware layout.

    This is the easiest way to use qb-compiler from Qiskit.  It builds a
    :func:`~qiskit.compiler.transpile`-compatible pass manager with
    :class:`QBCalibrationLayout` inserted as the layout stage.

    Parameters
    ----------
    circuit:
        The Qiskit ``QuantumCircuit`` to transpile.
    backend:
        Target backend name (e.g. ``"ibm_fez"``).  Used to look up
        basis gates and coupling map from qb-compiler's backend registry.
    calibration_path:
        Path to a QubitBoost ``calibration_hub`` JSON file.
    calibration_data:
        Pre-parsed calibration dict (alternative to *calibration_path*).
    calibration_provider:
        v0.5+. A :class:`~qb_compiler.calibration.provider.CalibrationProvider`
        instance (typically a
        :class:`~qb_compiler.calibration.live_provider.LiveCalibrationProvider`).
        Takes precedence over *calibration_path* / *calibration_data* if
        provided. Use this when you want fresh-fetched calibration data
        from :mod:`qubitboost_sdk.calibration.CalibrationHub` rather than
        a static fixture file.
    optimization_level:
        Qiskit optimization level (0-3).  Default is 2.

    Returns
    -------
    QuantumCircuit
        The transpiled circuit.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from qb_compiler.config import BACKEND_CONFIGS

    # Resolve calibration. Provider takes precedence; we materialise its
    # snapshot to a dict so the existing QBCalibrationLayout pass (which
    # consumes a dict, not a provider) can be reused unchanged.
    cal_dict: dict | None = None
    if calibration_provider is not None:
        cal_dict = _provider_to_dict(calibration_provider, backend)
    elif calibration_path is not None:
        cal_dict = _load_calibration_dict(calibration_path)
    elif calibration_data is not None:
        cal_dict = calibration_data

    # Resolve backend properties for Qiskit
    basis_gates = None
    coupling_map = None
    if backend is not None and backend in BACKEND_CONFIGS:
        spec = BACKEND_CONFIGS[backend]
        basis_gates = list(spec.basis_gates)

        # Build coupling map from calibration data if available, else from
        # calibration file, else leave as None (all-to-all)
        if cal_dict and "coupling_map" in cal_dict:
            edges = cal_dict["coupling_map"]
            coupling_map = CouplingMap(couplinglist=edges)

    try:
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
        )

        # Inject calibration-aware layout if we have calibration data.
        # The layout must be pre-set before Qiskit's own layout stage runs —
        # appending to ``pm.layout`` after the preset stage already built a
        # layout raises KeyError in Qiskit 2.x's ApplyLayout.  Using
        # ``pre_layout`` lets QBCalibrationLayout seed ``property_set["layout"]``
        # so the default layout stage skips its own finder.
        if cal_dict is not None:
            from qiskit.transpiler import PassManager as _PassManager

            cal_layout = QBCalibrationLayout(cal_dict)
            if pm.pre_layout is None:
                pm.pre_layout = _PassManager()
            pm.pre_layout.append(cal_layout)

        result = pm.run(circuit)
        return result

    except Exception as exc:
        # Graceful fallback: if anything goes wrong with the custom pipeline,
        # fall back to standard Qiskit transpile
        warnings.warn(
            f"qb_transpile: custom pipeline failed ({exc}), "
            f"falling back to standard qiskit.transpile",
            RuntimeWarning,
            stacklevel=2,
        )
        from qiskit.compiler import transpile

        kwargs: dict[str, Any] = {"optimization_level": optimization_level}
        if basis_gates is not None:
            kwargs["basis_gates"] = basis_gates
        if coupling_map is not None:
            kwargs["coupling_map"] = coupling_map
        return transpile(circuit, **kwargs)
