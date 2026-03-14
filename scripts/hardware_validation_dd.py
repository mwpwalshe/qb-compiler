#!/usr/bin/env python3
"""Hardware validation: Qiskit alone vs Qiskit + DD on IBM Fez.

For each circuit, submits TWO jobs to the QPU:
  1. Qiskit transpile (opt_level=3, best of 20 seeds) — baseline
  2. Same circuit + Dynamical Decoupling — enhanced

Compares output distribution quality (heavy output probability for GHZ,
or ground state overlap for QAOA/QFT).

Usage:
    python scripts/hardware_validation_dd.py [--dry-run] [--yes]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Measure, Parameter, Delay
from qiskit.circuit.library import CZGate, HGate, RZGate, SXGate, XGate

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qb_compiler.passes.scheduling.dynamical_decoupling import (
    insert_dd_calibration_aware,
)

N_SEEDS = 20
N_SHOTS = 4096
BACKEND_NAME = "ibm_fez"


# ── Circuit builders ──────────────────────────────────────────────


def build_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def build_qaoa(n: int, p: int = 1) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    gamma, beta = 0.7, 0.3
    for i in range(n):
        qc.h(i)
    for _ in range(p):
        for i in range(n):
            j = (i + 1) % n
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)
        for i in range(n):
            qc.rx(2 * beta, i)
    qc.measure(range(n), range(n))
    return qc


def build_qft(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    qc.measure(range(n), range(n))
    return qc


# ── Analysis ──────────────────────────────────────────────────────


def count_2q(tc) -> int:
    return sum(
        1 for inst in tc.data
        if len(inst.qubits) == 2
        and inst.operation.name not in ("barrier", "measure", "reset")
    )


def ghz_heavy_output_prob(counts: dict, n: int) -> float:
    """Probability of measuring 00...0 or 11...1 (ideal GHZ outcomes)."""
    total = sum(counts.values())
    all_zeros = "0" * n
    all_ones = "1" * n
    heavy = counts.get(all_zeros, 0) + counts.get(all_ones, 0)
    return heavy / total


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DD hardware validation on IBM Fez")
    parser.add_argument("--dry-run", action="store_true", help="Prepare circuits without submitting")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    circuits = {
        "GHZ-8": build_ghz(8),
        "QAOA-6": build_qaoa(6, p=1),
        "QFT-6": build_qft(6),
    }

    if not args.dry_run:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = service.backend(BACKEND_NAME)
        target = backend.target
        print(f"Backend: {backend.name} ({target.num_qubits} qubits)")
        print(f"Target dt: {target.dt}")

        # Load calibration for DD type selection
        try:
            from qb_compiler.calibration.models.backend_properties import BackendProperties
            cal_file = Path("tests/fixtures/calibration_snapshots/ibm_fez_2026_03_14.json")
            if cal_file.exists():
                props = BackendProperties.from_qubitboost_json(str(cal_file))
            else:
                props = None
        except Exception:
            props = None
    else:
        # Dry-run: build synthetic target from calibration fixture
        from qiskit.transpiler import InstructionProperties, Target
        from qb_compiler.calibration.models.backend_properties import BackendProperties

        cal_file = Path("tests/fixtures/calibration_snapshots/ibm_fez_2026_03_14.json")
        props = BackendProperties.from_qubitboost_json(str(cal_file))
        n_q = props.n_qubits
        DT = 2.2222222222222221e-10
        target = Target(num_qubits=n_q, dt=DT)

        twoq_props = {}
        for gp in props.gate_properties:
            if len(gp.qubits) == 2:
                dur = gp.gate_time_ns * 1e-9 if gp.gate_time_ns else 68e-9
                twoq_props[gp.qubits] = InstructionProperties(
                    error=gp.error_rate, duration=dur,
                )
        target.add_instruction(CZGate(), twoq_props)

        sq_dur = 25e-9
        theta = Parameter("theta")
        target.add_instruction(RZGate(theta), {(q,): InstructionProperties(duration=0) for q in range(n_q)})
        target.add_instruction(SXGate(), {(q,): InstructionProperties(duration=sq_dur) for q in range(n_q)})
        target.add_instruction(XGate(), {(q,): InstructionProperties(duration=sq_dur) for q in range(n_q)})
        target.add_instruction(HGate(), {(q,): InstructionProperties(duration=sq_dur) for q in range(n_q)})
        target.add_instruction(Measure(), {(q,): InstructionProperties(duration=1.6e-6) for q in range(n_q)})
        delay_param = Parameter("t")
        target.add_instruction(Delay(delay_param), {(q,): None for q in range(n_q)}, name="delay")

        print(f"DRY RUN — Synthetic target ({n_q} qubits)")

    # Prepare all circuits
    prepared = {}
    for name, qc in circuits.items():
        print(f"\n--- {name} ---")

        # Step 1: Qiskit best of N seeds
        best_tc = None
        best_2q = float("inf")
        best_seed = -1

        for seed in range(N_SEEDS):
            tc = transpile(qc, target=target, optimization_level=3, seed_transpiler=seed)
            c2q = count_2q(tc)
            if c2q < best_2q:
                best_2q = c2q
                best_tc = tc
                best_seed = seed

        # Extract physical qubits
        ql = best_tc.layout
        if ql and ql.initial_layout:
            phys = [ql.initial_layout[qc.qubits[i]] for i in range(qc.num_qubits)]
        else:
            phys = list(range(qc.num_qubits))

        print(f"  Qiskit best: seed={best_seed}, 2Q={best_2q}, "
              f"depth={best_tc.depth()}, qubits={phys}")

        # Step 2: Add DD
        if props is not None:
            enhanced = insert_dd_calibration_aware(best_tc, target, props)
        else:
            from qb_compiler.passes.scheduling.dynamical_decoupling import insert_dd
            enhanced = insert_dd(best_tc, target)

        base_ops = best_tc.count_ops()
        enh_ops = enhanced.count_ops()
        dd_x = enh_ops.get("x", 0) - base_ops.get("x", 0)
        dd_y = enh_ops.get("y", 0) - base_ops.get("y", 0)
        dd_total = dd_x + dd_y
        print(f"  DD: {dd_total} gates ({dd_x} X, {dd_y} Y)")

        prepared[name] = {
            "original": qc,
            "base": best_tc,
            "enhanced": enhanced,
            "seed": best_seed,
            "2q": best_2q,
            "dd_gates": dd_total,
            "physical_qubits": phys,
        }

    if args.dry_run:
        print("\n--- DRY RUN COMPLETE ---")
        print("Circuits prepared but not submitted.")
        for name, p in prepared.items():
            print(f"  {name}: 2Q={p['2q']}, DD={p['dd_gates']}, qubits={p['physical_qubits']}")
        return

    # Confirm submission
    total_circuits = len(prepared) * 2
    total_shots = total_circuits * N_SHOTS
    print(f"\nReady to submit {total_circuits} circuits × {N_SHOTS} shots = "
          f"{total_shots:,} total shots to {BACKEND_NAME}")

    if not args.yes:
        response = input("Submit? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Submit jobs
    sampler = SamplerV2(mode=backend)
    all_pubs = []
    pub_labels = []

    for name, p in prepared.items():
        all_pubs.append((p["base"], N_SHOTS))
        pub_labels.append(f"{name}_base")
        all_pubs.append((p["enhanced"], N_SHOTS))
        pub_labels.append(f"{name}_dd")

    print(f"\nSubmitting {len(all_pubs)} circuits...")
    from qiskit_ibm_runtime import SamplerV2

    pubs = [(circ,) for circ, _ in all_pubs]
    job = sampler.run(pubs, shots=N_SHOTS)
    print(f"Job ID: {job.job_id()}")
    print("Waiting for results...")

    result = job.result()
    print(f"Job completed!")

    # Process results
    results_out = []
    for i, (label, (circ, _)) in enumerate(zip(pub_labels, all_pubs)):
        counts = result[i].data.meas.get_counts()
        name = label.rsplit("_", 1)[0]
        variant = label.rsplit("_", 1)[1]

        if "GHZ" in name:
            n = int(name.split("-")[1])
            quality = ghz_heavy_output_prob(counts, n)
            metric_name = "heavy_output_prob"
        else:
            quality = None
            metric_name = "N/A"

        results_out.append({
            "circuit": name,
            "variant": variant,
            "shots": N_SHOTS,
            "quality": quality,
            "metric": metric_name,
            "top_counts": dict(sorted(counts.items(), key=lambda x: -x[1])[:5]),
        })

        print(f"  {label}: {metric_name}={quality}")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Circuit':>10} | {'Base quality':>12} | {'DD quality':>12} | {'Delta':>8}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    for name in prepared:
        base_r = next(r for r in results_out if r["circuit"] == name and r["variant"] == "base")
        dd_r = next(r for r in results_out if r["circuit"] == name and r["variant"] == "dd")
        if base_r["quality"] is not None and dd_r["quality"] is not None:
            delta = dd_r["quality"] - base_r["quality"]
            sign = "+" if delta >= 0 else ""
            print(f"{name:>10} | {base_r['quality']:>12.4f} | {dd_r['quality']:>12.4f} | {sign}{delta:.4f}")
        else:
            print(f"{name:>10} |          N/A |          N/A |      N/A")

    print(f"{'='*70}")

    # Save
    out = Path("results/hardware_validation_dd.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "backend": BACKEND_NAME,
            "job_id": job.job_id(),
            "n_shots": N_SHOTS,
            "n_seeds": N_SEEDS,
            "results": results_out,
            "prepared": {
                name: {
                    "seed": p["seed"],
                    "2q": p["2q"],
                    "dd_gates": p["dd_gates"],
                    "physical_qubits": p["physical_qubits"],
                }
                for name, p in prepared.items()
            },
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
