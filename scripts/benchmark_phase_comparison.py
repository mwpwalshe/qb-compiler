#!/usr/bin/env python3
"""Comprehensive Phase 1-4 Comparative Analysis.

Compares all ML phases across multiple dimensions:
- Layout fidelity (estimated)
- Compilation speed
- Model size & complexity
- Routing quality (Phase 4)
- Scaling behaviour (circuit size)

Usage:
    python scripts/benchmark_phase_comparison.py
"""
from __future__ import annotations

import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qb_compiler.compiler import QBCircuit, _load_calibration_fixture, _to_ir_circuit
from qb_compiler.calibration.models.backend_properties import BackendProperties


# ── Circuit builders ──────────────────────────────────────────────────


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


def build_qft(n: int) -> QBCircuit:
    c = QBCircuit(n)
    for i in range(n):
        c.h(i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            c.cx(j, i)
            c.rz(i, -angle / 2)
            c.cx(j, i)
            c.rz(i, angle / 2)
    return c.measure_all()


def build_star(n: int) -> QBCircuit:
    c = QBCircuit(n).h(0)
    for i in range(1, n):
        c.cx(0, i)
    return c.measure_all()


# ── Fidelity estimator ───────────────────────────────────────────────


def estimate_fidelity(
    circuit: QBCircuit,
    cal: BackendProperties,
    layout: dict[int, int] | None = None,
) -> float:
    gate_map: dict[tuple[int, int], float] = {}
    qubit_readout: dict[int, float] = {}
    if layout is not None:
        for gp in cal.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                gate_map[(gp.qubits[0], gp.qubits[1])] = gp.error_rate
        for qp in cal.qubit_properties:
            if qp.readout_error is not None:
                qubit_readout[qp.qubit_id] = qp.readout_error

    two_q_errors = [
        gp.error_rate for gp in cal.gate_properties
        if len(gp.qubits) == 2 and gp.error_rate is not None and gp.error_rate < 0.5
    ]
    readout_errors = [
        qp.readout_error for qp in cal.qubit_properties
        if qp.readout_error is not None
    ]
    median_2q = statistics.median(two_q_errors) if two_q_errors else 0.005
    median_ro = statistics.median(readout_errors) if readout_errors else 0.01
    median_1q = median_2q / 10.0

    fidelity = 1.0
    for op in circuit.ops:
        if op.name == "measure":
            if layout:
                phys_q = layout.get(op.qubits[0], op.qubits[0])
                err = qubit_readout.get(phys_q, median_ro)
            else:
                err = median_ro
            fidelity *= (1.0 - err)
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
            fidelity *= (1.0 - err)
        elif op.name not in ("barrier", "reset"):
            fidelity *= (1.0 - median_1q)
    return fidelity


def run_benchmark():
    import logging
    logging.basicConfig(level=logging.WARNING)

    from qb_compiler.passes.mapping.calibration_mapper import (
        CalibrationMapper,
        CalibrationMapperConfig,
    )
    from qb_compiler.ml import is_available, is_gnn_available

    cal = _load_calibration_fixture("ibm_fez")
    if cal is None:
        print("ERROR: No calibration data")
        sys.exit(1)

    # ── Load all models ──
    xgb_pred = None
    gnn_pred = None
    rl_router = None
    xgb_size = 0
    gnn_size = 0
    rl_size = 0
    xgb_meta = {}
    gnn_meta = {}
    rl_meta = {}

    if is_available():
        try:
            from qb_compiler.ml.layout_predictor import MLLayoutPredictor, _WEIGHTS_DIR
            w = _WEIGHTS_DIR / "ibm_heron_v1.json"
            if w.exists():
                xgb_pred = MLLayoutPredictor(model_path=w, min_candidates=20)
                xgb_size = w.stat().st_size
                xgb_meta = xgb_pred.metadata
        except Exception as e:
            print(f"WARNING: XGBoost failed: {e}")

    if is_gnn_available():
        try:
            from qb_compiler.ml.gnn_router import GNNLayoutPredictor, _WEIGHTS_DIR as GNN_W
            w = GNN_W / "gnn_heron_v1.pt"
            if w.exists():
                gnn_pred = GNNLayoutPredictor(model_path=w, min_candidates=20)
                gnn_size = w.stat().st_size
                gnn_meta = gnn_pred.metadata
        except Exception as e:
            print(f"WARNING: GNN failed: {e}")

        try:
            from qb_compiler.ml.rl_router import RLRouter, _WEIGHTS_DIR as RL_W
            w = RL_W / "rl_router_v1.pt"
            if w.exists():
                rl_router = RLRouter(model_path=w, backend=cal)
                rl_size = w.stat().st_size
                rl_meta = rl_router.metadata
        except Exception as e:
            print(f"WARNING: RL failed: {e}")

    # ── Mappers ──
    config_greedy = CalibrationMapperConfig(max_candidates=1, vf2_call_limit=1)
    config_vf2 = CalibrationMapperConfig(max_candidates=500, vf2_call_limit=50_000)

    mapper_greedy = CalibrationMapper(cal, config=config_greedy)
    mapper_vf2 = CalibrationMapper(cal, config=config_vf2)
    mapper_xgb = CalibrationMapper(cal, config=config_vf2, layout_predictor=xgb_pred) if xgb_pred else None
    mapper_gnn = CalibrationMapper(cal, config=config_vf2, layout_predictor=gnn_pred) if gnn_pred else None

    # ── Circuits ──
    circuits = [
        ("Bell", build_ghz(2)),
        ("GHZ-3", build_ghz(3)),
        ("GHZ-5", build_ghz(5)),
        ("GHZ-8", build_ghz(8)),
        ("QAOA-3", build_qaoa_ring(3)),
        ("QAOA-4", build_qaoa_ring(4)),
        ("QAOA-6", build_qaoa_ring(6)),
        ("QAOA-8", build_qaoa_ring(8)),
        ("QFT-3", build_qft(3)),
        ("QFT-4", build_qft(4)),
        ("Star-4", build_star(4)),
        ("Star-6", build_star(6)),
    ]

    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  COMPREHENSIVE ML PIPELINE COMPARISON — IBM Fez (156 qubits)")
    print("  Phases 1-4: Data Gen → XGBoost → GNN → RL Router")
    print("=" * 120)

    # ── Section 1: Layout Fidelity ──
    print()
    print("  SECTION 1: ESTIMATED FIDELITY (layout-only, higher = better)")
    print("  " + "-" * 116)
    print(f"  {'Circuit':<10} {'Q':>3} {'2Q':>4} "
          f"{'Baseline':>10} {'Greedy':>10} {'VF2':>10} {'XGB+VF2':>10} {'GNN+VF2':>10} "
          f"{'XGB vs BL':>10} {'GNN vs BL':>10} {'GNN vs XGB':>11}")
    print("  " + "-" * 116)

    results = []
    for name, circ in circuits:
        n_2q = circ.two_qubit_count
        ir_circ = _to_ir_circuit(circ)

        # Baseline (median device error)
        fid_bl = estimate_fidelity(circ, cal, layout=None)

        # Greedy
        ctx_g: dict = {}
        t0 = time.perf_counter()
        mapper_greedy.run(ir_circ.copy(), ctx_g)
        t_greedy = (time.perf_counter() - t0) * 1000
        fid_g = estimate_fidelity(circ, cal, layout=ctx_g.get("initial_layout", {}))

        # VF2
        ctx_v: dict = {}
        t0 = time.perf_counter()
        mapper_vf2.run(ir_circ.copy(), ctx_v)
        t_vf2 = (time.perf_counter() - t0) * 1000
        fid_v = estimate_fidelity(circ, cal, layout=ctx_v.get("initial_layout", {}))

        # XGBoost
        fid_x = 0.0
        t_xgb = 0.0
        layout_x = {}
        if mapper_xgb:
            ctx_x: dict = {}
            t0 = time.perf_counter()
            mapper_xgb.run(ir_circ.copy(), ctx_x)
            t_xgb = (time.perf_counter() - t0) * 1000
            layout_x = ctx_x.get("initial_layout", {})
            fid_x = estimate_fidelity(circ, cal, layout=layout_x)

        # GNN
        fid_n = 0.0
        t_gnn = 0.0
        layout_n = {}
        if mapper_gnn:
            ctx_n: dict = {}
            t0 = time.perf_counter()
            mapper_gnn.run(ir_circ.copy(), ctx_n)
            t_gnn = (time.perf_counter() - t0) * 1000
            layout_n = ctx_n.get("initial_layout", {})
            fid_n = estimate_fidelity(circ, cal, layout=layout_n)

        xgb_vs_bl = ((fid_x - fid_bl) / fid_bl * 100) if fid_bl > 0 and fid_x > 0 else 0
        gnn_vs_bl = ((fid_n - fid_bl) / fid_bl * 100) if fid_bl > 0 and fid_n > 0 else 0
        gnn_vs_xgb = ((fid_n - fid_x) / fid_x * 100) if fid_x > 0 and fid_n > 0 else 0

        def _s(v):
            return "+" if v > 0 else ""

        print(
            f"  {name:<10} {circ.n_qubits:>3} {n_2q:>4} "
            f"{fid_bl:>10.4f} {fid_g:>10.4f} {fid_v:>10.4f} {fid_x:>10.4f} {fid_n:>10.4f} "
            f"{_s(xgb_vs_bl)}{xgb_vs_bl:>9.1f}% {_s(gnn_vs_bl)}{gnn_vs_bl:>9.1f}% {_s(gnn_vs_xgb)}{gnn_vs_xgb:>10.1f}%"
        )

        results.append({
            "name": name, "n_q": circ.n_qubits, "n_2q": n_2q,
            "fid_bl": fid_bl, "fid_g": fid_g, "fid_v": fid_v,
            "fid_x": fid_x, "fid_n": fid_n,
            "t_greedy": t_greedy, "t_vf2": t_vf2, "t_xgb": t_xgb, "t_gnn": t_gnn,
            "layout_v": ctx_v.get("initial_layout", {}),
            "layout_x": layout_x, "layout_n": layout_n,
        })

    # ── Section 2: Compilation Speed ──
    print()
    print("  SECTION 2: COMPILATION SPEED (milliseconds, lower = better)")
    print("  " + "-" * 90)
    print(f"  {'Circuit':<10} {'Greedy':>10} {'VF2':>10} {'XGB+VF2':>10} {'GNN+VF2':>10} "
          f"{'XGB Speedup':>12} {'GNN Speedup':>12}")
    print("  " + "-" * 90)

    for r in results:
        xgb_speedup = r["t_vf2"] / r["t_xgb"] if r["t_xgb"] > 0 else 0
        gnn_speedup = r["t_vf2"] / r["t_gnn"] if r["t_gnn"] > 0 else 0
        print(
            f"  {r['name']:<10} {r['t_greedy']:>9.1f} {r['t_vf2']:>9.1f} "
            f"{r['t_xgb']:>9.1f} {r['t_gnn']:>9.1f} "
            f"{xgb_speedup:>11.1f}x {gnn_speedup:>11.1f}x"
        )

    # ── Section 3: Routing Quality (Phase 4) ──
    if rl_router:
        print()
        print("  SECTION 3: RL ROUTING QUALITY (Phase 4)")
        print("  " + "-" * 60)
        print(f"  {'Circuit':<10} {'Layout':>10} {'SWAPs':>8} {'Error':>10} {'Error/2Q':>10}")
        print("  " + "-" * 60)

        for r in results:
            layout = r["layout_n"] if r["layout_n"] else r["layout_v"]
            if not layout:
                continue
            circ_name = r["name"]
            circ = [c for n, c in circuits if n == circ_name][0]
            ir = _to_ir_circuit(circ)

            final, swaps, error = rl_router.route(ir, layout)
            err_per_2q = error / r["n_2q"] if r["n_2q"] > 0 else 0
            src = "GNN" if r["layout_n"] else "VF2"
            print(
                f"  {circ_name:<10} {src:>10} {len(swaps):>8} "
                f"{error:>10.4f} {err_per_2q:>10.4f}"
            )

    # ── Section 4: Scaling Analysis ──
    print()
    print("  SECTION 4: SCALING — GNN ADVANTAGE vs XGB by CIRCUIT SIZE")
    print("  " + "-" * 70)
    print(f"  {'Qubits':>8} {'2Q Gates':>10} {'XGB Fidelity':>14} {'GNN Fidelity':>14} {'Delta':>10}")
    print("  " + "-" * 70)

    for r in results:
        if r["fid_x"] > 0 and r["fid_n"] > 0:
            delta = r["fid_n"] - r["fid_x"]
            marker = " ***" if delta > 0.005 else " *" if delta > 0.001 else ""
            print(
                f"  {r['n_q']:>8} {r['n_2q']:>10} {r['fid_x']:>14.4f} {r['fid_n']:>14.4f} "
                f"{'+' if delta > 0 else ''}{delta:>9.4f}{marker}"
            )

    print()
    print("  Key: *** = GNN advantage > 0.5%, * = GNN advantage > 0.1%")

    # ── Section 5: Model Comparison Summary ──
    print()
    print("  " + "=" * 80)
    print("  SECTION 5: MODEL ARCHITECTURE COMPARISON")
    print("  " + "=" * 80)
    print()
    print(f"  {'Metric':<28} {'Phase 2':>16} {'Phase 3':>16} {'Phase 4':>16}")
    print(f"  {'':28s} {'XGBoost':>16} {'GNN (GCN)':>16} {'RL (PPO)':>16}")
    print("  " + "-" * 76)
    print(f"  {'Model type':<28} {'Gradient boost':>16} {'Graph neural':>16} {'Actor-critic':>16}")
    print(f"  {'Task':<28} {'Qubit scoring':>16} {'Qubit scoring':>16} {'SWAP routing':>16}")
    print(f"  {'Input':<28} {'Flat features':>16} {'Graph struct':>16} {'Routing state':>16}")

    # Parameters
    xgb_params = "N/A"
    gnn_params = str(gnn_meta.get("n_parameters", "N/A"))
    rl_params = str(rl_meta.get("n_parameters", "N/A"))
    print(f"  {'Parameters':<28} {xgb_params:>16} {gnn_params:>16} {rl_params:>16}")

    print(f"  {'Model size':<28} {xgb_size/1024:>15.0f}K {gnn_size/1024:>15.0f}K {rl_size/1024:>15.0f}K")

    xgb_auc = f"{xgb_meta.get('auc', 0):.4f}" if xgb_meta else "N/A"
    gnn_auc = f"{gnn_meta.get('training_auc', 0):.4f}" if gnn_meta else "N/A"
    print(f"  {'Training AUC':<28} {xgb_auc:>16} {gnn_auc:>16} {'N/A':>16}")

    print(f"  {'Graph-aware':<28} {'No':>16} {'Yes':>16} {'Yes':>16}")
    print(f"  {'Dependency':<28} {'xgboost':>16} {'torch':>16} {'torch':>16}")
    print(f"  {'License':<28} {'Apache 2.0':>16} {'Apache 2.0':>16} {'Proprietary':>16}")
    print(f"  {'Retrainable nightly':<28} {'Yes':>16} {'Yes':>16} {'Yes (prod)':>16}")

    # ── Summary statistics ──
    print()
    print("  " + "=" * 80)
    print("  SUMMARY")
    print("  " + "=" * 80)

    # Average improvement over baseline
    xgb_improvements = [(r["fid_x"] - r["fid_bl"]) / r["fid_bl"] * 100
                        for r in results if r["fid_x"] > 0 and r["fid_bl"] > 0]
    gnn_improvements = [(r["fid_n"] - r["fid_bl"]) / r["fid_bl"] * 100
                        for r in results if r["fid_n"] > 0 and r["fid_bl"] > 0]
    gnn_vs_xgb = [(r["fid_n"] - r["fid_x"]) / r["fid_x"] * 100
                  for r in results if r["fid_x"] > 0 and r["fid_n"] > 0]

    if xgb_improvements:
        print(f"  XGBoost avg improvement over baseline: {sum(xgb_improvements)/len(xgb_improvements):>+.2f}%")
        print(f"  XGBoost max improvement over baseline: {max(xgb_improvements):>+.2f}%")
    if gnn_improvements:
        print(f"  GNN avg improvement over baseline:     {sum(gnn_improvements)/len(gnn_improvements):>+.2f}%")
        print(f"  GNN max improvement over baseline:     {max(gnn_improvements):>+.2f}%")
    if gnn_vs_xgb:
        print(f"  GNN avg improvement over XGBoost:      {sum(gnn_vs_xgb)/len(gnn_vs_xgb):>+.2f}%")
        print(f"  GNN max improvement over XGBoost:      {max(gnn_vs_xgb):>+.2f}%")
        gnn_wins = sum(1 for d in gnn_vs_xgb if d > 0)
        print(f"  GNN wins vs XGBoost:                   {gnn_wins}/{len(gnn_vs_xgb)} circuits")

    # Speed comparison
    xgb_speedups = [r["t_vf2"] / r["t_xgb"] for r in results if r["t_xgb"] > 0]
    gnn_speedups = [r["t_vf2"] / r["t_gnn"] for r in results if r["t_gnn"] > 0]
    if xgb_speedups:
        print(f"  XGBoost avg speedup over VF2:          {sum(xgb_speedups)/len(xgb_speedups):.1f}x")
    if gnn_speedups:
        print(f"  GNN avg speedup over VF2:              {sum(gnn_speedups)/len(gnn_speedups):.1f}x")

    print()


if __name__ == "__main__":
    run_benchmark()
