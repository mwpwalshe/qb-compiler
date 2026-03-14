#!/usr/bin/env python3
"""Final local validation: all fixes active on March 14 calibration.

Tests tiebreaker + Qiskit seed injection + post-routing rescoring.
Outputs one clean comparison table for GHZ-5, GHZ-8, GHZ-10.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CZGate, CXGate, HGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.calibration_mapper import (
    CalibrationMapper,
    CalibrationMapperConfig,
)

GHZ_SIZES = [5, 8, 10]
N_QISKIT_SEEDS = 20


def build_target(props: BackendProperties) -> Target:
    n_q = props.n_qubits
    target = Target(num_qubits=n_q)
    gate_types = {gp.gate_type for gp in props.gate_properties if len(gp.qubits) == 2}
    gate_2q_cls = CZGate if "cz" in gate_types else CXGate
    twoq_props = {}
    for gp in props.gate_properties:
        if len(gp.qubits) == 2:
            twoq_props[gp.qubits] = InstructionProperties(
                error=gp.error_rate,
                duration=gp.gate_time_ns * 1e-9 if gp.gate_time_ns else 68e-9,
            )
    target.add_instruction(gate_2q_cls(), twoq_props)
    sq_props = {(q,): None for q in range(n_q)}
    theta = Parameter("theta")
    target.add_instruction(RZGate(theta), sq_props)
    target.add_instruction(SXGate(), sq_props)
    target.add_instruction(XGate(), sq_props)
    target.add_instruction(HGate(), sq_props)
    target.add_instruction(Measure(), sq_props)
    return target


def build_ghz_ir(n: int) -> QBCircuit:
    c = QBCircuit(n_qubits=n, n_clbits=n, name=f"ghz_{n}")
    c.add_gate(QBGate("h", (0,)))
    for i in range(n - 1):
        c.add_gate(QBGate("cx", (i, i + 1)))
    for i in range(n):
        c.add_measurement(i, i)
    return c


def build_ghz_qiskit(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def est_fidelity(layout: dict[int, int], props: BackendProperties, n: int) -> float:
    """Estimated fidelity using actual gate + readout errors."""
    fid = 1.0
    for i in range(n - 1):
        pa, pb = layout[i], layout[i + 1]
        err = 0.01
        for gp in props.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                if set(gp.qubits) == {pa, pb}:
                    err = min(err, gp.error_rate)
        fid *= (1.0 - err)
    for i in range(n):
        qp = props.qubit(layout[i])
        ro = qp.readout_error if qp and qp.readout_error is not None else 0.015
        fid *= (1.0 - ro)
    return fid


def qiskit_best_of_seeds(qc: QuantumCircuit, target: Target, n: int, props: BackendProperties):
    """Run Qiskit opt_level=3 with N seeds, return best by estimated fidelity."""
    best_fid = -1.0
    best_layout = {}
    best_phys = []
    best_seed = -1
    for seed in range(N_QISKIT_SEEDS):
        tc = transpile(qc, target=target, optimization_level=3, seed_transpiler=seed)
        ql = tc.layout
        if ql and ql.initial_layout:
            phys = [ql.initial_layout[qc.qubits[i]] for i in range(n)]
        else:
            phys = list(range(n))
        layout = {i: phys[i] for i in range(n)}
        fid = est_fidelity(layout, props, n)
        if fid > best_fid:
            best_fid = fid
            best_layout = layout
            best_phys = phys
            best_seed = seed
    return best_fid, best_phys, best_seed


def main():
    cal_file = Path("tests/fixtures/calibration_snapshots/ibm_fez_2026_03_14.json")
    if not cal_file.exists():
        print(f"ERROR: {cal_file} not found")
        sys.exit(1)

    props = BackendProperties.from_qubitboost_json(str(cal_file))
    target = build_target(props)
    print(f"Calibration: {cal_file.name} ({props.n_qubits} qubits)\n")

    results = []
    for n in GHZ_SIZES:
        circ_ir = build_ghz_ir(n)
        qc = build_ghz_qiskit(n)

        # qb-compiler: all fixes active
        mapper = CalibrationMapper(
            props,
            config=CalibrationMapperConfig(
                max_candidates=500, vf2_call_limit=50_000,
                top_k=20, max_per_region=3,
            ),
            qiskit_target=target,
        )
        ctx: dict[str, Any] = {}
        mapper.run(circ_ir, ctx)
        qb_layout = ctx["initial_layout"]
        qb_phys = [qb_layout[i] for i in range(n)]
        qb_fid = est_fidelity(qb_layout, props, n)
        qb_region = f"{min(qb_phys)}-{max(qb_phys)}"

        # Qiskit best of 20 seeds
        qi_fid, qi_phys, qi_seed = qiskit_best_of_seeds(qc, target, n, props)

        delta = qb_fid - qi_fid
        results.append({
            "circuit": f"GHZ-{n}",
            "qb_region": qb_region,
            "qb_qubits": qb_phys,
            "qb_fid": qb_fid,
            "qi_fid": qi_fid,
            "qi_phys": qi_phys,
            "qi_seed": qi_seed,
            "delta": delta,
        })

        # Show rescore details
        rescore = getattr(mapper, '_last_rescore_details', [])
        qiskit_injected = getattr(mapper, '_last_qiskit_injected', 0)
        print(f"  GHZ-{n}: {len(rescore)} candidates rescored, {qiskit_injected} Qiskit seeds injected")

    # Clean table
    print(f"\n{'='*85}")
    print(f"{'Circuit':>10} | {'qb-compiler region':>20} | {'qb-compiler fid':>15} | {'Qiskit best fid':>15} | {'Delta':>8}")
    print(f"{'-'*10}-+-{'-'*20}-+-{'-'*15}-+-{'-'*15}-+-{'-'*8}")

    all_positive = True
    for r in results:
        sign = "+" if r["delta"] >= 0 else ""
        marker = " OK" if r["delta"] >= 0 else " FAIL"
        print(
            f"{r['circuit']:>10} | {r['qb_region']:>20} | {r['qb_fid']:>15.6f} | {r['qi_fid']:>15.6f} | {sign}{r['delta']:.6f}{marker}"
        )
        if r["delta"] < -0.0001:
            all_positive = False

    print(f"{'='*85}")
    print()

    for r in results:
        print(f"  {r['circuit']}: qb={r['qb_qubits']}  qi={r['qi_phys']} (seed {r['qi_seed']})")

    print()
    if all_positive:
        print("ALL CIRCUITS SHOW POSITIVE OR NEUTRAL DELTA — READY FOR HARDWARE")
    else:
        print("WARNING: Some circuits show negative delta")

    # Save results
    out = Path("results/final_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
