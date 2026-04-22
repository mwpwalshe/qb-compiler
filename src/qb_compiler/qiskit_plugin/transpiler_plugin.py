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

    Combines T1, T2, readout error, and (optionally) gate error into a
    single quality metric.  Missing values are replaced with pessimistic
    defaults so that qubits with incomplete calibration are deprioritised.

    Score formula::

        score = w_ro * readout_error
              + w_t1 * (1 / T1)
              + w_t2 * (1 / T2)
              + w_gate * avg_gate_error
    """
    # Weights — readout and gate errors matter most for NISQ circuits
    w_ro = 0.35
    w_t1 = 0.15
    w_t2 = 0.15
    w_gate = 0.35

    t1 = qubit_data.get("t1_us") or 50.0  # pessimistic default
    t2 = qubit_data.get("t2_us") or 25.0
    readout_err = qubit_data.get("readout_error") or 0.05
    gate_err = qubit_data.get("gate_error") or 0.01

    # Normalise T1/T2 into error-like quantities (larger T → smaller contribution)
    t1_contribution = 1.0 / max(t1, 1.0)
    t2_contribution = 1.0 / max(t2, 1.0)

    return w_ro * readout_err + w_t1 * t1_contribution + w_t2 * t2_contribution + w_gate * gate_err


def _load_calibration_dict(source: dict | str | Path) -> dict:
    """Normalise *source* into a parsed calibration dict."""
    if isinstance(source, (str, Path)):
        with open(source) as fh:
            return json.load(fh)  # type: ignore[no-any-return]
    if isinstance(source, dict):
        return source
    raise TypeError(f"calibration_data must be a dict or path, got {type(source)}")


def _build_qubit_scores(cal_data: dict) -> dict[int, float]:
    """Build ``{physical_qubit: score}`` from a QubitBoost calibration dict."""
    qubit_props = cal_data.get("qubit_properties", [])

    # Pre-compute per-qubit average gate error from gate_properties
    gate_errors: dict[int, list[float]] = {}
    for gp in cal_data.get("gate_properties", []):
        err = (gp.get("parameters") or {}).get("gate_error")
        if err is not None:
            for q in gp.get("qubits", []):
                gate_errors.setdefault(q, []).append(err)

    scores: dict[int, float] = {}
    for qp in qubit_props:
        qid = qp["qubit"]
        err_01 = qp.get("readout_error_0to1")
        err_10 = qp.get("readout_error_1to0")
        if err_01 is not None and err_10 is not None:
            ro_err = (err_01 + err_10) / 2.0
        else:
            ro_err = err_01 or err_10 or None

        avg_gate = None
        if gate_errors.get(qid):
            avg_gate = sum(gate_errors[qid]) / len(gate_errors[qid])

        scores[qid] = _score_qubit(
            {
                "t1_us": qp.get("T1"),
                "t2_us": qp.get("T2"),
                "readout_error": ro_err,
                "gate_error": avg_gate,
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
        """Set ``property_set["layout"]`` to a calibration-optimal mapping."""
        n_virtual = dag.num_qubits()

        if not self._scores:
            logger.warning("QBCalibrationLayout: no qubit scores available, skipping layout")
            return

        # Rank physical qubits by score (best first)
        ranked = sorted(self._scores.items(), key=lambda kv: kv[1])

        if len(ranked) < n_virtual:
            logger.warning(
                "QBCalibrationLayout: circuit needs %d qubits but calibration "
                "only covers %d — falling back",
                n_virtual,
                len(ranked),
            )
            return

        # Take the top-n_virtual physical qubits
        best_physical = [qid for qid, _ in ranked[:n_virtual]]

        # Build a Qiskit Layout: virtual qubit index -> physical qubit

        virtual_qubits = dag.qubits
        layout_dict = {}
        for v_idx, v_qubit in enumerate(virtual_qubits):
            layout_dict[v_qubit] = best_physical[v_idx]

        layout = Layout(layout_dict)
        self.property_set["layout"] = layout
        logger.info(
            "QBCalibrationLayout: mapped %d virtual qubits to physical %s",
            n_virtual,
            best_physical,
        )


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
    optimization_level:
        Qiskit optimization level (0-3).  Default is 2.

    Returns
    -------
    QuantumCircuit
        The transpiled circuit.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from qb_compiler.config import BACKEND_CONFIGS

    # Resolve calibration
    cal_dict: dict | None = None
    if calibration_path is not None:
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
