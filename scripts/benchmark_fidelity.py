#!/usr/bin/env python3
"""Benchmark: qb-compiler calibration-aware mapping vs baseline.

Loads real IBM Fez calibration data and compares fidelity estimates for
circuits compiled with and without calibration-aware qubit mapping.

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


def estimate_fidelity(
    circuit: QBCircuit,
    cal: BackendProperties,
    layout: dict[int, int] | None = None,
) -> float:
    """Estimate fidelity. If layout is None, use median errors (baseline)."""
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


def run_mapper_only(circuit, cal, mapper):
    """Run CalibrationMapper and return layout."""
    ir_circ = _to_ir_circuit(circuit)
    context: dict = {}
    result = mapper.run(ir_circ, context)
    return context.get("initial_layout", {})


def run_full_pipeline(circuit, cal, mapper):
    """Run CalibrationMapper + NoiseAwareRouter and return layout + routed circuit."""
    from qb_compiler.passes.mapping.noise_aware_router import NoiseAwareRouter
    from qb_compiler.ir.operations import QBGate

    ir_circ = _to_ir_circuit(circuit)
    context: dict = {}

    # Step 1: Map
    result = mapper.run(ir_circ, context)
    mapped_circ = result.circuit
    layout = context.get("initial_layout", {})

    # Step 2: Route (inserts SWAPs for non-adjacent 2Q gates)
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

    # Count SWAPs inserted
    n_swaps = route_result.metadata.get("swaps_inserted", 0)

    # Build the fidelity estimate on the routed circuit.
    # The routed circuit has physical qubit indices already applied.
    # We need to estimate fidelity directly on the physical operations.
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

    # Add readout error for original measurement qubits
    for op in circuit.ops:
        if op.name == "measure":
            phys_q = layout.get(op.qubits[0], op.qubits[0])
            err = qubit_readout.get(phys_q, median_ro)
            fidelity *= (1.0 - err)

    return fidelity, n_swaps, layout


def run_benchmark():
    cal = _load_calibration_fixture("ibm_fez")
    if cal is None:
        print("ERROR: Could not load IBM Fez calibration data.")
        sys.exit(1)

    median_2q, median_ro, median_1q = _get_calibration_stats(cal)
    print(f"IBM Fez calibration: {cal.n_qubits} qubits, timestamp={cal.timestamp}")
    print(f"Median CZ error: {median_2q:.6f}, median readout: {median_ro:.4f}")

    from qb_compiler.passes.mapping.calibration_mapper import (
        CalibrationMapper,
        CalibrationMapperConfig,
    )

    mapper_config = CalibrationMapperConfig(
        gate_error_weight=10.0,
        coherence_weight=0.3,
        readout_weight=5.0,
        max_candidates=50000,
        vf2_call_limit=500_000,
    )
    mapper = CalibrationMapper(cal, config=mapper_config)

    # Focus on circuits that map to nearest-neighbor (no routing needed)
    # These show pure benefit of calibration-aware qubit selection
    circuits = [
        ("Bell (2q)", build_bell()),
        ("GHZ-5", build_ghz(5)),
        ("GHZ-8", build_ghz(8)),
        ("GHZ-16", build_ghz(16)),
        ("GHZ-32", build_ghz(32)),
        ("QAOA-4", build_qaoa_ring(4)),
        ("QAOA-8", build_qaoa_ring(8)),
        ("QAOA-12", build_qaoa_ring(12)),
        ("QFT-4", build_qft(4)),
    ]

    print()
    print(f"{'Circuit':<12} {'Qubits':>6} {'2Q':>4} {'Baseline':>10} "
          f"{'qb-comp':>10} {'Improv.':>10} {'SWAPs':>6}")
    print("-" * 64)

    for name, circ in circuits:
        n_2q = circ.two_qubit_count
        baseline = estimate_fidelity(circ, cal, layout=None)

        try:
            mapped_fid, n_swaps, layout = run_full_pipeline(circ, cal, mapper)
        except Exception as e:
            print(f"{name:<12} {circ.n_qubits:>6} {n_2q:>4} {baseline:>10.4f} "
                  f"{'ERROR':>10} {'':>10} {'':>6}  {e}")
            continue

        if baseline > 0:
            improvement = ((mapped_fid - baseline) / baseline) * 100.0
        else:
            improvement = 0.0

        sign = "+" if improvement > 0 else ""
        print(f"{name:<12} {circ.n_qubits:>6} {n_2q:>4} {baseline:>10.4f} "
              f"{mapped_fid:>10.4f} {sign}{improvement:>9.1f}% {n_swaps:>6}")

    print()
    print("Baseline = median device errors (topology-only mapping)")
    print("qb-comp  = calibration-aware mapping + noise-aware routing")
    print("SWAPs    = additional SWAP gates inserted for non-adjacent interactions")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    run_benchmark()
