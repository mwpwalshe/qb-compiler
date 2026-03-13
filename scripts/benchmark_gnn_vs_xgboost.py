#!/usr/bin/env python3
"""Benchmark: GNN (Phase 3) vs XGBoost (Phase 2) vs Greedy/VF2 layout prediction.

Compares four approaches:
1. Greedy — no VF2, just edge-ranked placement
2. VF2 (standard) — full VF2 search over all physical qubits
3. XGBoost+VF2 (Phase 2) — XGBoost narrows candidates, then VF2
4. GNN+VF2 (Phase 3) — GNN narrows candidates, then VF2

Shows: layout quality (fidelity), compilation speed, model size.

Usage:
    python scripts/benchmark_gnn_vs_xgboost.py
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
    from qb_compiler.ml import is_available as ml_available, is_gnn_available

    cal = _load_calibration_fixture("ibm_fez")
    if cal is None:
        print("ERROR: Could not load IBM Fez calibration data.")
        sys.exit(1)

    # Check ML availability
    has_xgb = ml_available()
    has_gnn = is_gnn_available()

    xgb_predictor = None
    gnn_predictor = None

    if has_xgb:
        from qb_compiler.ml.layout_predictor import MLLayoutPredictor, _WEIGHTS_DIR as XGB_WEIGHTS
        xgb_weights = XGB_WEIGHTS / "ibm_heron_v1.json"
        if xgb_weights.exists():
            xgb_predictor = MLLayoutPredictor(model_path=xgb_weights, min_candidates=20)
            xgb_size = xgb_weights.stat().st_size / 1024
        else:
            print("WARNING: XGBoost weights not found")
            xgb_size = 0
    else:
        print("WARNING: XGBoost not installed")
        xgb_size = 0

    if has_gnn:
        from qb_compiler.ml.gnn_router import GNNLayoutPredictor, _WEIGHTS_DIR as GNN_WEIGHTS
        gnn_weights = GNN_WEIGHTS / "gnn_heron_v1.pt"
        if gnn_weights.exists():
            gnn_predictor = GNNLayoutPredictor(model_path=gnn_weights, min_candidates=20)
            gnn_size = gnn_weights.stat().st_size / 1024
        else:
            print("WARNING: GNN weights not found. Run: python -c 'from qb_compiler.ml.gnn_router import train_gnn_model; train_gnn_model()'")
            gnn_size = 0
    else:
        print("WARNING: PyTorch not installed")
        gnn_size = 0

    print(f"IBM Fez: {cal.n_qubits} qubits")
    if xgb_predictor:
        meta = xgb_predictor.metadata
        print(f"XGBoost: v{meta.get('version', '?')}, AUC={meta.get('auc', 0):.4f}, {xgb_size:.0f} KB")
    if gnn_predictor:
        meta = gnn_predictor.metadata
        print(f"GNN: v{meta.get('version', '?')}, AUC={meta.get('training_auc', 0):.4f}, {gnn_size:.0f} KB")
    print()

    # Mapper configs
    config_greedy = CalibrationMapperConfig(max_candidates=1, vf2_call_limit=1)
    config_vf2 = CalibrationMapperConfig(max_candidates=500, vf2_call_limit=50_000)

    mapper_greedy = CalibrationMapper(cal, config=config_greedy)
    mapper_vf2 = CalibrationMapper(cal, config=config_vf2)
    mapper_xgb = CalibrationMapper(cal, config=config_vf2, layout_predictor=xgb_predictor) if xgb_predictor else None
    mapper_gnn = CalibrationMapper(cal, config=config_vf2, layout_predictor=gnn_predictor) if gnn_predictor else None

    circuits = [
        ("GHZ-5", build_ghz(5)),
        ("GHZ-8", build_ghz(8)),
        ("QAOA-4", build_qaoa_ring(4)),
        ("QAOA-8", build_qaoa_ring(8)),
        ("QFT-4", build_qft(4)),
    ]

    print("=" * 120)
    print("BENCHMARK: Greedy vs VF2 vs XGBoost+VF2 (Phase 2) vs GNN+VF2 (Phase 3)")
    print("=" * 120)
    print()
    print(f"{'Circuit':<10} {'Q':>3} {'2Q':>4} "
          f"{'Greedy':>8} {'VF2':>8} {'XGB+VF2':>8} {'GNN+VF2':>8} "
          f"{'Greedy ms':>10} {'VF2 ms':>10} {'XGB ms':>10} {'GNN ms':>10} "
          f"{'XGB vs VF2':>10} {'GNN vs VF2':>10}")
    print("-" * 120)

    for name, circ in circuits:
        n_2q = circ.two_qubit_count
        ir_circ = _to_ir_circuit(circ)

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
        if mapper_xgb:
            ctx_x: dict = {}
            t0 = time.perf_counter()
            mapper_xgb.run(ir_circ.copy(), ctx_x)
            t_xgb = (time.perf_counter() - t0) * 1000
            fid_x = estimate_fidelity(circ, cal, layout=ctx_x.get("initial_layout", {}))

        # GNN
        fid_n = 0.0
        t_gnn = 0.0
        if mapper_gnn:
            ctx_n: dict = {}
            t0 = time.perf_counter()
            mapper_gnn.run(ir_circ.copy(), ctx_n)
            t_gnn = (time.perf_counter() - t0) * 1000
            fid_n = estimate_fidelity(circ, cal, layout=ctx_n.get("initial_layout", {}))

        xgb_vs = ((fid_x - fid_v) / fid_v * 100) if fid_v > 0 and fid_x > 0 else 0
        gnn_vs = ((fid_n - fid_v) / fid_v * 100) if fid_v > 0 and fid_n > 0 else 0

        def _sign(v):
            return "+" if v > 0 else ""

        print(
            f"{name:<10} {circ.n_qubits:>3} {n_2q:>4} "
            f"{fid_g:>8.4f} {fid_v:>8.4f} {fid_x:>8.4f} {fid_n:>8.4f} "
            f"{t_greedy:>9.1f} {t_vf2:>9.1f} {t_xgb:>9.1f} {t_gnn:>9.1f} "
            f"{_sign(xgb_vs)}{xgb_vs:>9.1f}% {_sign(gnn_vs)}{gnn_vs:>9.1f}%"
        )

    print()
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<30} {'XGBoost (Phase 2)':>20} {'GNN (Phase 3)':>20}")
    print("-" * 70)
    print(f"{'Architecture':<30} {'Gradient Boosted Trees':>20} {'Dual-Graph GCN':>20}")
    print(f"{'Model size':<30} {f'{xgb_size:.0f} KB':>20} {f'{gnn_size:.0f} KB':>20}")
    if xgb_predictor:
        xm = xgb_predictor.metadata
        print(f"{'Training AUC':<30} {xm.get('auc', 0):>20.4f}", end="")
    else:
        print(f"{'Training AUC':<30} {'N/A':>20}", end="")
    if gnn_predictor:
        gm = gnn_predictor.metadata
        print(f" {gm.get('training_auc', 0):>19.4f}")
    else:
        print(f" {'N/A':>19}")
    print(f"{'Uses graph structure':<30} {'No':>20} {'Yes':>20}")
    print(f"{'Dependency':<30} {'xgboost':>20} {'torch':>20}")
    print()


if __name__ == "__main__":
    run_benchmark()
