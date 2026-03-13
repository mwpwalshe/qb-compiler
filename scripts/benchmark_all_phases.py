#!/usr/bin/env python3
"""Benchmark: All 4 ML phases compared.

Phase 1: Training data generation (not a model)
Phase 2: XGBoost layout predictor
Phase 3: GNN layout predictor
Phase 4: RL SWAP router (routing quality comparison)

Usage:
    python scripts/benchmark_all_phases.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qb_compiler.compiler import _load_calibration_fixture


def run_benchmark():
    import logging
    logging.basicConfig(level=logging.WARNING)

    from qb_compiler.ir.circuit import QBCircuit
    from qb_compiler.ir.operations import QBGate
    from qb_compiler.ml import is_available, is_gnn_available
    from qb_compiler.passes.mapping.calibration_mapper import (
        CalibrationMapper, CalibrationMapperConfig
    )

    cal = _load_calibration_fixture("ibm_fez")
    if cal is None:
        print("ERROR: No calibration data")
        sys.exit(1)

    def ghz(n):
        c = QBCircuit(n_qubits=n, n_clbits=0)
        c.add_gate(QBGate("h", (0,)))
        for i in range(n - 1):
            c.add_gate(QBGate("cx", (i, i + 1)))
        return c

    def qaoa(n):
        c = QBCircuit(n_qubits=n, n_clbits=0)
        for i in range(n - 1):
            c.add_gate(QBGate("cx", (i, i + 1)))
            c.add_gate(QBGate("rz", (i + 1,), (0.5,)))
            c.add_gate(QBGate("cx", (i, i + 1)))
        return c

    circuits = [
        ("GHZ-5", ghz(5)),
        ("GHZ-8", ghz(8)),
        ("QAOA-4", qaoa(4)),
        ("QAOA-8", qaoa(8)),
    ]

    print("=" * 100)
    print("ML PIPELINE COMPARISON — All 4 Phases")
    print("=" * 100)
    print()

    # Load models
    xgb_pred = None
    gnn_pred = None
    rl_router = None

    if is_available():
        try:
            from qb_compiler.ml.layout_predictor import MLLayoutPredictor, _WEIGHTS_DIR
            w = _WEIGHTS_DIR / "ibm_heron_v1.json"
            if w.exists():
                xgb_pred = MLLayoutPredictor(model_path=w, min_candidates=20)
        except Exception:
            pass

    if is_gnn_available():
        try:
            from qb_compiler.ml.gnn_router import GNNLayoutPredictor, _WEIGHTS_DIR as GNN_W
            w = GNN_W / "gnn_heron_v1.pt"
            if w.exists():
                gnn_pred = GNNLayoutPredictor(model_path=w, min_candidates=20)
        except Exception:
            pass

        try:
            from qb_compiler.ml.rl_router import RLRouter, _WEIGHTS_DIR as RL_W
            w = RL_W / "rl_router_v1.pt"
            if w.exists():
                rl_router = RLRouter(model_path=w, backend=cal)
        except Exception:
            pass

    # ── Layout Quality Comparison ──
    config = CalibrationMapperConfig(max_candidates=500, vf2_call_limit=50_000)
    mapper_vf2 = CalibrationMapper(cal, config=config)
    mapper_xgb = CalibrationMapper(cal, config=config, layout_predictor=xgb_pred) if xgb_pred else None
    mapper_gnn = CalibrationMapper(cal, config=config, layout_predictor=gnn_pred) if gnn_pred else None

    print("1. LAYOUT QUALITY (lower score = better)")
    print(f"{'Circuit':<10} {'VF2':>10} {'XGB+VF2':>10} {'GNN+VF2':>10}")
    print("-" * 50)

    layout_results = {}
    for name, circ in circuits:
        ctx_v = {}
        mapper_vf2.run(circ.copy(), ctx_v)
        score_v = ctx_v.get("calibration_score", 0)

        score_x = 0
        ctx_x = {}
        if mapper_xgb:
            mapper_xgb.run(circ.copy(), ctx_x)
            score_x = ctx_x.get("calibration_score", 0)

        score_g = 0
        ctx_g = {}
        if mapper_gnn:
            mapper_gnn.run(circ.copy(), ctx_g)
            score_g = ctx_g.get("calibration_score", 0)

        print(f"{name:<10} {score_v:>10.4f} {score_x:>10.4f} {score_g:>10.4f}")
        layout_results[name] = {
            "vf2_layout": ctx_v.get("initial_layout", {}),
            "xgb_layout": ctx_x.get("initial_layout", {}),
            "gnn_layout": ctx_g.get("initial_layout", {}),
        }

    # ── Routing Quality (Phase 4 vs greedy routing) ──
    if rl_router:
        print()
        print("2. ROUTING QUALITY (total error, lower = better)")
        print(f"{'Circuit':<10} {'RL SWAPs':>10} {'RL Error':>10}")
        print("-" * 35)

        for name, circ in circuits:
            layout = layout_results[name]["gnn_layout"] or layout_results[name]["vf2_layout"]
            if layout:
                final_layout, swaps, error = rl_router.route(circ, layout)
                print(f"{name:<10} {len(swaps):>10} {error:>10.4f}")

    # ── Model Sizes ──
    print()
    print("3. MODEL COMPARISON")
    print(f"{'':>20} {'Phase 2':>15} {'Phase 3':>15} {'Phase 4':>15}")
    print(f"{'':>20} {'XGBoost':>15} {'GNN':>15} {'RL (PPO)':>15}")
    print("-" * 70)

    if xgb_pred:
        xm = xgb_pred.metadata
        print(f"{'AUC':>20} {xm.get('auc', 0):>14.4f}", end="")
    else:
        print(f"{'AUC':>20} {'N/A':>15}", end="")
    if gnn_pred:
        gm = gnn_pred.metadata
        print(f" {gm.get('training_auc', 0):>14.4f}", end="")
    else:
        print(f" {'N/A':>14}", end="")
    print(f" {'N/A':>14}")

    sizes = []
    for pred, label in [(xgb_pred, "Phase 2"), (gnn_pred, "Phase 3")]:
        if pred:
            s = pred.metadata.get("model_size_bytes", 0)
            sizes.append(f"{s/1024:.0f} KB")
        else:
            sizes.append("N/A")
    if rl_router:
        s = rl_router.metadata.get("model_size_bytes", 0)
        sizes.append(f"{s/1024:.0f} KB")
    else:
        sizes.append("N/A")

    print(f"{'Model size':>20} {sizes[0]:>15} {sizes[1]:>15} {sizes[2]:>15}")
    print(f"{'Task':>20} {'Layout pred':>15} {'Layout pred':>15} {'SWAP routing':>15}")
    print(f"{'Graph-aware':>20} {'No':>15} {'Yes':>15} {'Yes':>15}")
    print(f"{'License':>20} {'Apache 2.0':>15} {'Apache 2.0':>15} {'Proprietary':>15}")
    print()


if __name__ == "__main__":
    run_benchmark()
