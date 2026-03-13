#!/usr/bin/env python3
"""Generate consistent benchmark tables for the README.

Produces three markdown tables:

1. **Core compiler (no ML)** — Greedy vs VF2 calibration-aware mapping
2. **With ML acceleration** — Greedy vs VF2 vs ML+VF2 vs GNN+VF2
3. **T1 asymmetry** — Symmetric vs asymmetric fidelity estimation

All tables use the SAME baseline (Greedy: max_candidates=1, vf2_call_limit=1)
and the SAME ``estimate_fidelity`` function.

Usage:
    python scripts/benchmark_readme.py
"""
from __future__ import annotations

import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qb_compiler.compiler import QBCircuit, _load_calibration_fixture, _to_ir_circuit
from qb_compiler.calibration.models.backend_properties import BackendProperties


# ── Circuit builders ─────────────────────────────────────────────────


def build_bell() -> QBCircuit:
    return QBCircuit(2).h(0).cx(0, 1).measure_all()


def build_ghz(n: int) -> QBCircuit:
    c = QBCircuit(n).h(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    return c.measure_all()


def build_qaoa_ring(n: int) -> QBCircuit:
    c = QBCircuit(n)
    for i in range(n - 1):
        c.cx(i, i + 1)
        c.rz(i + 1, 0.5)
        c.cx(i, i + 1)
    for i in range(n):
        c.rx(i, 0.7)
    return c.measure_all()


# ── Shared fidelity estimator ────────────────────────────────────────


def _get_calibration_stats(
    cal: BackendProperties,
) -> tuple[float, float, float]:
    """Return (median_2q_error, median_readout_error, median_1q_error)."""
    two_q_errors = [
        gp.error_rate
        for gp in cal.gate_properties
        if len(gp.qubits) == 2
        and gp.error_rate is not None
        and gp.error_rate < 0.5
    ]
    readout_errors = [
        qp.readout_error
        for qp in cal.qubit_properties
        if qp.readout_error is not None
    ]
    median_2q = statistics.median(two_q_errors) if two_q_errors else 0.005
    median_ro = statistics.median(readout_errors) if readout_errors else 0.01
    median_1q = median_2q / 10.0
    return median_2q, median_ro, median_1q


def estimate_fidelity(
    circuit: QBCircuit,
    cal: BackendProperties,
    layout: dict[int, int] | None = None,
) -> float:
    """Estimate fidelity using symmetric readout error.

    This is the ONE fidelity function used for ALL tables.
    If *layout* is None, uses median device errors (baseline).
    """
    gate_map: dict[tuple[int, int], float] = {}
    qubit_readout: dict[int, float] = {}
    if layout is not None:
        for gp in cal.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                gate_map[(gp.qubits[0], gp.qubits[1])] = gp.error_rate
        for qp in cal.qubit_properties:
            if qp.readout_error is not None:
                qubit_readout[qp.qubit_id] = qp.readout_error

    median_2q, median_ro, median_1q = _get_calibration_stats(cal)

    fidelity = 1.0
    for op in circuit.ops:
        if op.name == "measure":
            if layout:
                phys_q = layout.get(op.qubits[0], op.qubits[0])
                err = qubit_readout.get(phys_q, median_ro)
            else:
                err = median_ro
            fidelity *= 1.0 - err
        elif op.is_two_qubit:
            if layout:
                phys_0 = layout.get(op.qubits[0], op.qubits[0])
                phys_1 = layout.get(op.qubits[1], op.qubits[1])
                err = gate_map.get((phys_0, phys_1))
                if err is None:
                    err = gate_map.get((phys_1, phys_0))
                if err is None:
                    err = median_2q
            else:
                err = median_2q
            fidelity *= 1.0 - err
        elif op.name not in ("barrier", "reset"):
            fidelity *= 1.0 - median_1q
    return fidelity


def estimate_fidelity_asymmetric(
    circuit: QBCircuit,
    cal: BackendProperties,
    layout: dict[int, int] | None = None,
) -> float:
    """Estimate fidelity using ASYMMETRIC readout errors.

    Uses P(0|1) and P(1|0) separately instead of the symmetrised
    readout error.  Circuits that place qubits in |1> are penalised
    by the higher P(0|1) relaxation error.
    """
    gate_map: dict[tuple[int, int], float] = {}
    qubit_ro_1to0: dict[int, float] = {}
    qubit_ro_0to1: dict[int, float] = {}
    if layout is not None:
        for gp in cal.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                gate_map[(gp.qubits[0], gp.qubits[1])] = gp.error_rate
        for qp in cal.qubit_properties:
            if qp.readout_error_1to0 is not None:
                qubit_ro_1to0[qp.qubit_id] = qp.readout_error_1to0
            if qp.readout_error_0to1 is not None:
                qubit_ro_0to1[qp.qubit_id] = qp.readout_error_0to1

    median_2q, median_ro, median_1q = _get_calibration_stats(cal)

    fidelity = 1.0
    for op in circuit.ops:
        if op.name == "measure":
            if layout:
                phys_q = layout.get(op.qubits[0], op.qubits[0])
                err_10 = qubit_ro_1to0.get(phys_q, median_ro)
                err_01 = qubit_ro_0to1.get(phys_q, median_ro)
                err = 0.6 * err_10 + 0.4 * err_01
            else:
                err = median_ro
            fidelity *= 1.0 - err
        elif op.is_two_qubit:
            if layout:
                phys_0 = layout.get(op.qubits[0], op.qubits[0])
                phys_1 = layout.get(op.qubits[1], op.qubits[1])
                err = gate_map.get((phys_0, phys_1))
                if err is None:
                    err = gate_map.get((phys_1, phys_0))
                if err is None:
                    err = median_2q
            else:
                err = median_2q
            fidelity *= 1.0 - err
        elif op.name not in ("barrier", "reset"):
            fidelity *= 1.0 - median_1q
    return fidelity


# ── Mapper helpers ───────────────────────────────────────────────────


def _run_mapper(mapper, circuit: QBCircuit) -> dict[int, int]:
    """Run a CalibrationMapper and return the layout dict."""
    ir_circ = _to_ir_circuit(circuit)
    ctx: dict = {}
    mapper.run(ir_circ, ctx)
    return ctx.get("initial_layout", {})


# ── Main benchmark ───────────────────────────────────────────────────


def run_benchmark() -> None:
    import logging

    logging.basicConfig(level=logging.WARNING)

    from qb_compiler.passes.mapping.calibration_mapper import (
        CalibrationMapper,
        CalibrationMapperConfig,
    )

    # Load calibration data
    cal = _load_calibration_fixture("ibm_fez")
    if cal is None:
        print("ERROR: Could not load IBM Fez calibration data.")
        sys.exit(1)

    median_2q, median_ro, _median_1q = _get_calibration_stats(cal)
    print(f"<!-- Benchmark on IBM Fez ({cal.n_qubits} qubits), "
          f"median CZ error={median_2q:.6f}, median readout={median_ro:.4f} -->")
    print()

    # ── Define circuits ──────────────────────────────────────────────

    circuits = [
        ("Bell (2q)", build_bell()),
        ("GHZ-5", build_ghz(5)),
        ("GHZ-8", build_ghz(8)),
        ("QAOA-4", build_qaoa_ring(4)),
        ("QAOA-8", build_qaoa_ring(8)),
    ]

    # ── Define mapper configs ────────────────────────────────────────

    config_greedy = CalibrationMapperConfig(
        max_candidates=1,
        vf2_call_limit=1,
    )

    config_vf2 = CalibrationMapperConfig(
        max_candidates=500,
        vf2_call_limit=50_000,
    )

    mapper_greedy = CalibrationMapper(cal, config=config_greedy)
    mapper_vf2 = CalibrationMapper(cal, config=config_vf2)

    # ── Table 1: Core compiler (no ML) ───────────────────────────────

    print("### Table 1: Core Compiler (no ML)")
    print()
    print("| Circuit | Qubits | 2Q Gates | Greedy | VF2 | Improvement |")
    print("|---------|-------:|---------:|-------:|----:|------------:|")

    # Collect results for reuse in Table 2
    greedy_fidelities: dict[str, float] = {}
    vf2_fidelities: dict[str, float] = {}
    greedy_layouts: dict[str, dict[int, int]] = {}
    vf2_layouts: dict[str, dict[int, int]] = {}

    for name, circ in circuits:
        n_2q = circ.two_qubit_count

        try:
            layout_g = _run_mapper(mapper_greedy, circ)
            fid_g = estimate_fidelity(circ, cal, layout=layout_g)
        except Exception:
            fid_g = 0.0
            layout_g = {}

        try:
            layout_v = _run_mapper(mapper_vf2, circ)
            fid_v = estimate_fidelity(circ, cal, layout=layout_v)
        except Exception:
            fid_v = 0.0
            layout_v = {}

        greedy_fidelities[name] = fid_g
        vf2_fidelities[name] = fid_v
        greedy_layouts[name] = layout_g
        vf2_layouts[name] = layout_v

        if fid_g > 0:
            imp = ((fid_v - fid_g) / fid_g) * 100.0
        else:
            imp = 0.0

        sign = "+" if imp >= 0 else ""
        print(
            f"| {name:<9s} | {circ.n_qubits:>5d} | {n_2q:>7d} "
            f"| {fid_g:.4f} | {fid_v:.4f} | {sign}{imp:.1f}% |"
        )

    print()
    print("*Greedy = edge-ranked placement (max_candidates=1). "
          "VF2 = full subgraph isomorphism search (max_candidates=500, call_limit=50k).*")
    print()

    # ── Table 2: With ML acceleration ────────────────────────────────

    # Try loading ML predictors
    ml_predictor = None
    gnn_predictor = None
    ml_available = False
    gnn_available = False

    try:
        from qb_compiler.ml import is_available as _ml_avail

        if _ml_avail():
            from qb_compiler.ml.layout_predictor import MLLayoutPredictor

            try:
                ml_predictor = MLLayoutPredictor.load_bundled("ibm_heron")
                ml_available = True
            except (FileNotFoundError, Exception):
                pass
    except ImportError:
        pass

    try:
        from qb_compiler.ml import is_gnn_available as _gnn_avail

        if _gnn_avail():
            from qb_compiler.ml.gnn_router import GNNLayoutPredictor

            try:
                gnn_predictor = GNNLayoutPredictor.load_bundled("ibm_heron")
                gnn_available = True
            except (FileNotFoundError, Exception):
                pass
    except ImportError:
        pass

    mapper_ml = None
    mapper_gnn = None
    if ml_available and ml_predictor is not None:
        mapper_ml = CalibrationMapper(
            cal, config=config_vf2, layout_predictor=ml_predictor
        )
    if gnn_available and gnn_predictor is not None:
        mapper_gnn = CalibrationMapper(
            cal, config=config_vf2, layout_predictor=gnn_predictor
        )

    print("### Table 2: With ML Acceleration")
    print()
    print(
        "| Circuit | Qubits | Greedy | VF2 | ML+VF2 | GNN+VF2 | Best vs Greedy |"
    )
    print(
        "|---------|-------:|-------:|----:|-------:|--------:|---------------:|"
    )

    for name, circ in circuits:
        fid_g = greedy_fidelities[name]
        fid_v = vf2_fidelities[name]

        # ML+VF2
        if mapper_ml is not None:
            try:
                layout_m = _run_mapper(mapper_ml, circ)
                fid_m = estimate_fidelity(circ, cal, layout=layout_m)
                ml_str = f"{fid_m:.4f}"
            except Exception:
                fid_m = 0.0
                ml_str = "N/A"
        else:
            fid_m = 0.0
            ml_str = "N/A"

        # GNN+VF2
        if mapper_gnn is not None:
            try:
                layout_gn = _run_mapper(mapper_gnn, circ)
                fid_gn = estimate_fidelity(circ, cal, layout=layout_gn)
                gnn_str = f"{fid_gn:.4f}"
            except Exception:
                fid_gn = 0.0
                gnn_str = "N/A"
        else:
            fid_gn = 0.0
            gnn_str = "N/A"

        # Best vs Greedy
        best = max(fid_v, fid_m, fid_gn)
        if fid_g > 0:
            best_imp = ((best - fid_g) / fid_g) * 100.0
        else:
            best_imp = 0.0
        sign = "+" if best_imp >= 0 else ""

        print(
            f"| {name:<9s} | {circ.n_qubits:>5d} "
            f"| {fid_g:.4f} | {fid_v:.4f} "
            f"| {ml_str:>6s} | {gnn_str:>7s} "
            f"| {sign}{best_imp:.1f}% |"
        )

    print()
    if not ml_available:
        print("*ML+VF2: XGBoost not available "
              "(install with `pip install 'qb-compiler[ml]'` and train model).*")
    if not gnn_available:
        print("*GNN+VF2: PyTorch not available "
              "(install with `pip install 'qb-compiler[gnn]'` and train model).*")
    print()

    # ── Table 3: T1 asymmetry ────────────────────────────────────────

    print("### Table 3: T1 Asymmetry Impact")
    print()
    print(
        "| Circuit | Qubits | Symmetric | Asymmetric | Delta |"
    )
    print(
        "|---------|-------:|----------:|-----------:|------:|"
    )

    for name, circ in circuits:
        layout = vf2_layouts[name]
        if not layout:
            continue
        fid_sym = estimate_fidelity(circ, cal, layout=layout)
        fid_asym = estimate_fidelity_asymmetric(circ, cal, layout=layout)

        if fid_sym > 0:
            delta = ((fid_asym - fid_sym) / fid_sym) * 100.0
        else:
            delta = 0.0
        sign = "+" if delta >= 0 else ""

        print(
            f"| {name:<9s} | {circ.n_qubits:>5d} "
            f"| {fid_sym:.6f} | {fid_asym:.6f} "
            f"| {sign}{delta:.2f}% |"
        )

    print()
    print("*Symmetric = standard (P(0|1)+P(1|0))/2 readout model. "
          "Asymmetric = qb-compiler T1-aware model weighting P(0|1) at 60%.*")
    print("*Negative delta = symmetric model overestimates fidelity "
          "(hides T1 relaxation error that Qiskit's default transpiler misses).*")
    print()
    print(f"*Calibration: IBM Fez, {cal.n_qubits} qubits, "
          f"timestamp={cal.timestamp}*")


if __name__ == "__main__":
    run_benchmark()
