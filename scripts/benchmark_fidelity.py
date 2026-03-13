#!/usr/bin/env python3
"""Benchmark: qb-compiler calibration-aware mapping vs baseline.

Loads real IBM Fez calibration data and compares fidelity estimates for
circuits compiled with and without calibration-aware qubit mapping.

Shows impact of:
1. Calibration-aware mapping (gate error + coherence + readout)
2. T1 asymmetry awareness (penalise qubits with high |1⟩ decay)
3. Temporal correlation detection (penalise correlated qubit pairs)

Usage:
    python scripts/benchmark_fidelity.py
"""
from __future__ import annotations

import math
import statistics
import sys
import os

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


def build_qaoa_ring(n: int) -> QBCircuit:
    """QAOA-style circuit: nearest-neighbor ZZ interactions on a ring + X mixer."""
    c = QBCircuit(n)
    for i in range(n - 1):  # linear chain, not ring (ring needs routing)
        c.cx(i, i + 1)
        c.rz(i + 1, 0.5)
        c.cx(i, i + 1)
    for i in range(n):
        c.rx(i, 0.7)
    return c.measure_all()


def _get_calibration_stats(cal: BackendProperties):
    """Extract median error rates from calibration data."""
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
    return median_2q, median_ro, median_1q


def estimate_fidelity_asymmetric(
    circuit: QBCircuit,
    cal: BackendProperties,
    layout: dict[int, int] | None = None,
) -> float:
    """Estimate fidelity using ASYMMETRIC readout errors.

    Unlike the standard estimator that uses symmetrised readout error,
    this uses the T1 asymmetry data: P(0|1) for qubits likely in |1⟩
    and P(1|0) for qubits likely in |0⟩.

    For circuits with H gates and CNOT targets, qubits spend significant
    time in |1⟩, so P(0|1) dominates the error.
    """
    gate_map: dict[tuple[int, int], float] = {}
    qubit_ro_1to0: dict[int, float] = {}  # P(0|1) — relaxation error
    qubit_ro_0to1: dict[int, float] = {}  # P(1|0) — excitation error
    qubit_ro_sym: dict[int, float] = {}   # symmetrised
    if layout is not None:
        for gp in cal.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                gate_map[(gp.qubits[0], gp.qubits[1])] = gp.error_rate
        for qp in cal.qubit_properties:
            if qp.readout_error_1to0 is not None:
                qubit_ro_1to0[qp.qubit_id] = qp.readout_error_1to0
            if qp.readout_error_0to1 is not None:
                qubit_ro_0to1[qp.qubit_id] = qp.readout_error_0to1
            if qp.readout_error is not None:
                qubit_ro_sym[qp.qubit_id] = qp.readout_error

    median_2q, median_ro, median_1q = _get_calibration_stats(cal)

    fidelity = 1.0
    for op in circuit.ops:
        if op.name == "measure":
            if layout:
                phys_q = layout.get(op.qubits[0], op.qubits[0])
                # Use P(0|1) as the dominant error — this is the T1 asymmetry
                # effect. In most quantum circuits, measurement follows gates
                # that put qubits in superposition, so ~50% of the time the
                # qubit is in |1⟩ and P(0|1) is the relevant error.
                err_10 = qubit_ro_1to0.get(phys_q, median_ro)
                err_01 = qubit_ro_0to1.get(phys_q, median_ro)
                # Weight: ~60% P(0|1) + ~40% P(1|0) for typical circuits
                err = 0.6 * err_10 + 0.4 * err_01
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


def estimate_fidelity(
    circuit: QBCircuit,
    cal: BackendProperties,
    layout: dict[int, int] | None = None,
) -> float:
    """Estimate fidelity with symmetric readout. If layout is None, use medians."""
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


def run_full_pipeline(circuit, cal, mapper):
    """Run CalibrationMapper + NoiseAwareRouter."""
    from qb_compiler.passes.mapping.noise_aware_router import NoiseAwareRouter
    from qb_compiler.ir.operations import QBGate

    ir_circ = _to_ir_circuit(circuit)
    context: dict = {}

    result = mapper.run(ir_circ, context)
    mapped_circ = result.circuit
    layout = context.get("initial_layout", {})
    breakdown = context.get("score_breakdown", {})

    gate_errors: dict[tuple[int, int], float] = {}
    for gp in cal.gate_properties:
        if len(gp.qubits) == 2 and gp.error_rate is not None:
            gate_errors[gp.qubits] = gp.error_rate

    router = NoiseAwareRouter(
        coupling_map=cal.coupling_map,
        gate_errors=gate_errors,
    )
    route_result = router.run(mapped_circ, context)
    routed_circ = route_result.circuit
    n_swaps = route_result.metadata.get("swaps_inserted", 0)

    # Fidelity on routed circuit
    gate_map: dict[tuple[int, int], float] = {}
    for gp in cal.gate_properties:
        if len(gp.qubits) == 2 and gp.error_rate is not None:
            gate_map[(gp.qubits[0], gp.qubits[1])] = gp.error_rate

    qubit_readout: dict[int, float] = {}
    for qp in cal.qubit_properties:
        if qp.readout_error is not None:
            qubit_readout[qp.qubit_id] = qp.readout_error

    median_2q, median_ro, median_1q = _get_calibration_stats(cal)

    fidelity = 1.0
    for op in routed_circ.iter_ops():
        if isinstance(op, QBGate):
            if op.num_qubits >= 2:
                q0, q1 = op.qubits[0], op.qubits[1]
                err = gate_map.get((q0, q1))
                if err is None:
                    err = gate_map.get((q1, q0))
                if err is None:
                    err = median_2q
                fidelity *= (1.0 - err)
            else:
                fidelity *= (1.0 - median_1q)

    for op in circuit.ops:
        if op.name == "measure":
            phys_q = layout.get(op.qubits[0], op.qubits[0])
            err = qubit_readout.get(phys_q, median_ro)
            fidelity *= (1.0 - err)

    return fidelity, n_swaps, layout, breakdown


def run_benchmark():
    from qb_compiler.passes.mapping.calibration_mapper import (
        CalibrationMapper,
        CalibrationMapperConfig,
    )

    # Load calibration data
    cal = _load_calibration_fixture("ibm_fez")
    if cal is None:
        print("ERROR: Could not load IBM Fez calibration data.")
        sys.exit(1)

    median_2q, median_ro, median_1q = _get_calibration_stats(cal)
    print(f"IBM Fez calibration: {cal.n_qubits} qubits, timestamp={cal.timestamp}")
    print(f"Median CZ error: {median_2q:.6f}, median readout: {median_ro:.4f}")

    # Show T1 asymmetry statistics
    ratios = [qp.t1_asymmetry_ratio for qp in cal.qubit_properties
              if qp.readout_error_0to1 is not None and qp.readout_error_1to0 is not None]
    if ratios:
        ratios.sort()
        print(f"T1 asymmetry: min={ratios[0]:.1f}x, median={ratios[len(ratios)//2]:.1f}x, "
              f"max={ratios[-1]:.1f}x")

    # Load temporal correlation data if available
    correlation_analyzer = None
    try:
        from qb_compiler.calibration.static_provider import StaticCalibrationProvider
        from qb_compiler.passes.mapping.temporal_correlation import (
            TemporalCorrelationAnalyzer,
        )

        import glob
        fixture_dir = os.path.join(
            os.path.dirname(__file__), "..", "tests", "fixtures", "calibration_snapshots"
        )
        fez_files = sorted(glob.glob(os.path.join(fixture_dir, "ibm_fez*.json")))
        if len(fez_files) >= 2:
            snapshots = []
            for f in fez_files:
                prov = StaticCalibrationProvider.from_json(f)
                snapshots.append(prov.properties)
            correlation_analyzer = TemporalCorrelationAnalyzer.from_snapshots(snapshots)
            res = correlation_analyzer.result
            print(f"Temporal correlation: {res.n_snapshots} snapshots, "
                  f"{len(res.qubit_volatility)} qubits tracked, "
                  f"{len(res.edge_correlation)} edges scored")
            # Show top volatile qubits
            top_vol = sorted(res.qubit_volatility.items(), key=lambda x: -x[1])[:5]
            if top_vol:
                print(f"Most volatile qubits: "
                      + ", ".join(f"Q{q}({v:.4f})" for q, v in top_vol))
        else:
            print(f"Temporal correlation: only {len(fez_files)} snapshot(s), need 2+")
    except Exception as e:
        print(f"Temporal correlation: skipped ({e})")

    # === Benchmark 1: Standard vs Asymmetry-Aware ===
    print()
    print("=" * 90)
    print("BENCHMARK: Impact of T1 asymmetry + temporal correlation awareness")
    print("=" * 90)
    print()

    # Config WITHOUT T1 asymmetry or correlation
    config_basic = CalibrationMapperConfig(
        gate_error_weight=10.0,
        coherence_weight=0.3,
        readout_weight=5.0,
        t1_asymmetry_weight=0.0,      # disabled
        correlation_weight=0.0,        # disabled
        max_candidates=500,
        vf2_call_limit=50_000,
    )

    # Config WITH T1 asymmetry + correlation
    config_full = CalibrationMapperConfig(
        gate_error_weight=10.0,
        coherence_weight=0.3,
        readout_weight=5.0,
        t1_asymmetry_weight=5.0,       # enabled
        correlation_weight=2.0,        # enabled
        max_candidates=500,
        vf2_call_limit=50_000,
    )

    mapper_basic = CalibrationMapper(cal, config=config_basic)
    mapper_full = CalibrationMapper(
        cal, config=config_full,
        correlation_analyzer=correlation_analyzer,
    )

    circuits = [
        ("Bell (2q)", build_bell()),
        ("GHZ-5", build_ghz(5)),
        ("GHZ-8", build_ghz(8)),
        ("QAOA-4", build_qaoa_ring(4)),
        ("QAOA-8", build_qaoa_ring(8)),
        ("QFT-4", build_qft(4)),
    ]

    # === Benchmark Table 1: Layout quality comparison ===
    print(f"{'Circuit':<12} {'Qubits':>6} {'2Q':>4} "
          f"{'Baseline':>10} {'Calibrated':>10} {'Impr':>8}")
    print("-" * 56)

    for name, circ in circuits:
        n_2q = circ.two_qubit_count
        baseline = estimate_fidelity(circ, cal, layout=None)

        try:
            ir_circ = _to_ir_circuit(circ)
            ctx: dict = {}
            mapper_full.run(ir_circ, ctx)
            layout = ctx.get("initial_layout", {})
            fid_mapped = estimate_fidelity(circ, cal, layout=layout)
        except Exception as e:
            print(f"{name:<12} {circ.n_qubits:>6} {n_2q:>4}  ERROR: {e}")
            continue

        if baseline > 0:
            imp = ((fid_mapped - baseline) / baseline) * 100.0
        else:
            imp = 0.0

        sign = "+" if imp > 0 else ""
        print(
            f"{name:<12} {circ.n_qubits:>6} {n_2q:>4} "
            f"{baseline:>10.4f} {fid_mapped:>10.4f} {sign}{imp:>7.1f}%"
        )

    print()
    print("Baseline   = median device errors (topology-only mapping)")
    print("Calibrated = qb-compiler with calibration + T1 asymmetry + correlation")
    print("Impr       = fidelity improvement over baseline")

    # === Benchmark Table 2: Asymmetric readout model ===
    print()
    print("=" * 90)
    print("BENCHMARK: Symmetric vs asymmetric readout fidelity model")
    print("=" * 90)
    print()
    print("Standard transpilers use symmetric readout error: (P(0|1) + P(1|0)) / 2")
    print("qb-compiler models the asymmetry: P(0|1) dominates for circuits holding |1⟩")
    print()

    print(f"{'Circuit':<12} {'Qubits':>6} "
          f"{'Symmetric':>12} {'Asymmetric':>12} {'Delta':>8}  Note")
    print("-" * 70)

    for name, circ in circuits:
        try:
            ir_circ = _to_ir_circuit(circ)
            ctx: dict = {}
            mapper_full.run(ir_circ, ctx)
            layout = ctx.get("initial_layout", {})
            fid_sym = estimate_fidelity(circ, cal, layout=layout)
            fid_asym = estimate_fidelity_asymmetric(circ, cal, layout=layout)
        except Exception as e:
            continue

        delta = (fid_asym - fid_sym) / fid_sym * 100 if fid_sym > 0 else 0
        sign = "+" if delta > 0 else ""
        note = ""
        if delta < -0.5:
            note = "<-- T1 asymmetry hurts here"
        elif delta > 0.5:
            note = "<-- asymmetry helps here"

        print(
            f"{name:<12} {circ.n_qubits:>6} "
            f"{fid_sym:>12.6f} {fid_asym:>12.6f} {sign}{delta:>7.2f}%  {note}"
        )

    print()
    print("Negative delta = symmetric model overestimates fidelity (hides T1 decay)")
    print("This is the error Qiskit's default transpiler makes.")

    # === Benchmark 2: T1 Asymmetry Impact Detail ===
    print()
    print("=" * 90)
    print("DETAIL: T1 asymmetry impact on qubit selection")
    print("=" * 90)
    print()

    # Compare layouts for GHZ-8
    ghz8 = build_ghz(8)
    ir_ghz8 = _to_ir_circuit(ghz8)

    ctx_basic: dict = {}
    mapper_basic.run(ir_ghz8, ctx_basic)
    layout_basic = ctx_basic["initial_layout"]

    ctx_full: dict = {}
    mapper_full.run(ir_ghz8, ctx_full)
    layout_full = ctx_full["initial_layout"]

    print("GHZ-8 qubit selection comparison:")
    print(f"  {'Logical':>8} {'Basic':>8} {'Full':>8}")
    for lq in range(8):
        pb = layout_basic.get(lq, -1)
        pf = layout_full.get(lq, -1)
        marker = " *" if pb != pf else ""
        print(f"  q{lq:>6d} -> Q{pb:>5d}  Q{pf:>5d}{marker}")

    # Show asymmetry of selected qubits
    print()
    print("T1 asymmetry of selected qubits:")
    print(f"  {'Qubit':>8} {'Ratio':>8} {'P(0|1)':>10} {'P(1|0)':>10} {'Readout':>10}")
    all_selected = set(layout_basic.values()) | set(layout_full.values())
    for pq in sorted(all_selected):
        qp = next((q for q in cal.qubit_properties if q.qubit_id == pq), None)
        if qp:
            in_basic = "B" if pq in layout_basic.values() else " "
            in_full = "F" if pq in layout_full.values() else " "
            print(
                f"  Q{pq:>5d} [{in_basic}{in_full}] "
                f"{qp.t1_asymmetry_ratio:>7.1f}x "
                f"{qp.readout_error_1to0 or 0:>10.6f} "
                f"{qp.readout_error_0to1 or 0:>10.6f} "
                f"{qp.readout_error or 0:>10.6f}"
            )

    # Score breakdown
    if "score_breakdown" in ctx_full:
        bd = ctx_full["score_breakdown"]
        print()
        print("Full mapper score breakdown (GHZ-8):")
        for key, val in bd.items():
            print(f"  {key:>15s}: {val:.4f}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    run_benchmark()
