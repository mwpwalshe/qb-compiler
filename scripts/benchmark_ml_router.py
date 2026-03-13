#!/usr/bin/env python3
"""Benchmark: ML-accelerated layout prediction vs greedy/VF2.

Compares three approaches:
1. Greedy (fallback) — no VF2, just edge-ranked placement
2. VF2 (standard) — full VF2 search over all physical qubits
3. ML+VF2 — XGBoost narrows candidates, then VF2 on reduced graph

Shows: layout quality (score), compilation speed, and fidelity estimate.

Usage:
    python scripts/benchmark_ml_router.py
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
    """Estimate fidelity using layout and calibration data."""
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

    # Check ML availability
    from qb_compiler.ml import is_available as ml_available
    if not ml_available():
        print("ERROR: XGBoost not installed. Run: pip install 'qb-compiler[ml]'")
        sys.exit(1)

    from qb_compiler.ml.layout_predictor import MLLayoutPredictor, _WEIGHTS_DIR
    weights = _WEIGHTS_DIR / "ibm_heron_v1.json"
    if not weights.exists():
        print("ERROR: Model weights not found. Run: python -m qb_compiler.ml.train")
        sys.exit(1)

    # Load calibration
    cal = _load_calibration_fixture("ibm_fez")
    if cal is None:
        print("ERROR: Could not load IBM Fez calibration data.")
        sys.exit(1)

    predictor = MLLayoutPredictor(model_path=weights, min_candidates=20)
    meta = predictor.metadata

    print(f"IBM Fez: {cal.n_qubits} qubits")
    print(f"ML model: v{meta.get('version', '?')}, "
          f"AUC={meta.get('auc', 0):.4f}, "
          f"size={meta.get('model_size_bytes', 0) / 1024:.0f} KB")
    print()

    # Three mapper configs
    config_greedy = CalibrationMapperConfig(
        max_candidates=1,    # force greedy fallback
        vf2_call_limit=1,
    )

    config_vf2 = CalibrationMapperConfig(
        max_candidates=500,
        vf2_call_limit=50_000,
    )

    config_ml = CalibrationMapperConfig(
        max_candidates=500,
        vf2_call_limit=50_000,
    )

    mapper_greedy = CalibrationMapper(cal, config=config_greedy)
    mapper_vf2 = CalibrationMapper(cal, config=config_vf2)
    mapper_ml = CalibrationMapper(cal, config=config_ml, layout_predictor=predictor)

    circuits = [
        ("Bell (2q)", build_bell()),
        ("GHZ-5", build_ghz(5)),
        ("GHZ-8", build_ghz(8)),
        ("QAOA-4", build_qaoa_ring(4)),
        ("QAOA-8", build_qaoa_ring(8)),
        ("QFT-4", build_qft(4)),
    ]

    print("=" * 100)
    print("BENCHMARK: Greedy vs VF2 vs ML+VF2 Layout Selection")
    print("=" * 100)
    print()
    print(f"{'Circuit':<12} {'Qubits':>6} {'2Q':>4} "
          f"{'Greedy':>10} {'VF2':>10} {'ML+VF2':>10} "
          f"{'Greedy ms':>10} {'VF2 ms':>10} {'ML ms':>10} "
          f"{'ML vs VF2':>10}")
    print("-" * 100)

    for name, circ in circuits:
        n_2q = circ.two_qubit_count
        ir_circ = _to_ir_circuit(circ)

        # Greedy
        ctx_g: dict = {}
        t0 = time.perf_counter()
        mapper_greedy.run(ir_circ.copy(), ctx_g)
        t_greedy = (time.perf_counter() - t0) * 1000
        layout_g = ctx_g.get("initial_layout", {})
        fid_g = estimate_fidelity(circ, cal, layout=layout_g)

        # VF2 (full)
        ctx_v: dict = {}
        t0 = time.perf_counter()
        mapper_vf2.run(ir_circ.copy(), ctx_v)
        t_vf2 = (time.perf_counter() - t0) * 1000
        layout_v = ctx_v.get("initial_layout", {})
        fid_v = estimate_fidelity(circ, cal, layout=layout_v)

        # ML+VF2
        ctx_m: dict = {}
        t0 = time.perf_counter()
        mapper_ml.run(ir_circ.copy(), ctx_m)
        t_ml = (time.perf_counter() - t0) * 1000
        layout_m = ctx_m.get("initial_layout", {})
        fid_m = estimate_fidelity(circ, cal, layout=layout_m)

        # ML vs VF2 improvement
        if fid_v > 0:
            ml_vs_vf2 = ((fid_m - fid_v) / fid_v) * 100
        else:
            ml_vs_vf2 = 0

        sign = "+" if ml_vs_vf2 > 0 else ""
        print(
            f"{name:<12} {circ.n_qubits:>6} {n_2q:>4} "
            f"{fid_g:>10.4f} {fid_v:>10.4f} {fid_m:>10.4f} "
            f"{t_greedy:>9.1f} {t_vf2:>9.1f} {t_ml:>9.1f} "
            f"{sign}{ml_vs_vf2:>9.1f}%"
        )

    print()
    print("Greedy  = edge-ranked placement (no VF2 search)")
    print("VF2     = full VF2 subgraph isomorphism over 156 qubits")
    print("ML+VF2  = XGBoost narrows to ~20 candidates, then VF2")
    print("ML vs VF2 = fidelity difference (ML relative to VF2)")
    print()

    # === Feature importance ===
    print("=" * 60)
    print("XGBoost Feature Importance (top 10 by gain)")
    print("=" * 60)
    print()
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(str(weights))
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    for fname, gain in sorted_imp[:10]:
        bar = "█" * int(gain / sorted_imp[0][1] * 30)
        print(f"  {fname:>30s}: {gain:>8.2f}  {bar}")


if __name__ == "__main__":
    run_benchmark()
