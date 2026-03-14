#!/usr/bin/env python3
"""Hardware validation v3: Rich circuits + DD on IBM Fez.

Tests circuits where layout MATTERS (QAOA, QFT, EfficientSU2) +
dynamical decoupling as post-routing optimization.

Strategy: Use Qiskit opt_level=3 for routing (including VF2PostLayout),
but seed it with our calibration-aware layout. Then add DD on top.
This means we get Qiskit's best routing AND our best layout AND DD.

Comparison:
  A) Qiskit opt_level=3 (no DD) — what users get by default
  B) qb-compiler: our layout + Qiskit routing + DD

Usage::

    python scripts/hardware_validation_v3.py --dry-run
    python scripts/hardware_validation_v3.py --yes
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CZGate, CXGate, HGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, PassManager, Target
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.calibration_mapper import (
    CalibrationMapper,
    CalibrationMapperConfig,
)

console = Console()

BACKEND_NAME = "ibm_fez"
SHOTS = 4096
QISKIT_SEEDS = list(range(20))


# ── Circuit builders ──────────────────────────────────────────────────


def build_qaoa_maxcut_6() -> tuple[QuantumCircuit, QBCircuit, str]:
    """QAOA MaxCut on 6-node graph: ring + 2 cross edges."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4)]
    gamma, beta = 0.7, 0.5
    n = 6

    qc = QuantumCircuit(n, n)
    ir = QBCircuit(n_qubits=n, n_clbits=n, name="qaoa_6")
    for i in range(n):
        qc.h(i)
        ir.add_gate(QBGate("h", (i,)))
    for i, j in edges:
        qc.cx(i, j)
        qc.rz(gamma, j)
        qc.cx(i, j)
        ir.add_gate(QBGate("cx", (i, j)))
        ir.add_gate(QBGate("rz", (j,), params=(gamma,)))
        ir.add_gate(QBGate("cx", (i, j)))
    for i in range(n):
        qc.rx(2 * beta, i)
        ir.add_gate(QBGate("rx", (i,), params=(2 * beta,)))
    qc.measure(range(n), range(n))
    for i in range(n):
        ir.add_measurement(i, i)

    return qc, ir, "qaoa_6"


def build_qft_6() -> tuple[QuantumCircuit, QBCircuit, str]:
    """QFT on 6 qubits."""
    n = 6
    qc = QuantumCircuit(n, n)
    ir = QBCircuit(n_qubits=n, n_clbits=n, name="qft_6")

    for i in range(n):
        qc.h(i)
        ir.add_gate(QBGate("h", (i,)))
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            qc.cx(j, i)
            qc.rz(angle, i)
            qc.cx(j, i)
            ir.add_gate(QBGate("cx", (j, i)))
            ir.add_gate(QBGate("rz", (i,), params=(angle,)))
            ir.add_gate(QBGate("cx", (j, i)))

    qc.measure(range(n), range(n))
    for i in range(n):
        ir.add_measurement(i, i)

    return qc, ir, "qft_6"


def build_efficientsu2_8() -> tuple[QuantumCircuit, QBCircuit, str]:
    """EfficientSU2 8q, reps=2, full entanglement."""
    n = 8
    reps = 2
    rng = np.random.default_rng(42)

    qc = QuantumCircuit(n, n)
    ir = QBCircuit(n_qubits=n, n_clbits=n, name="su2_8")

    for rep in range(reps + 1):
        for i in range(n):
            ty = float(rng.uniform(0, 2 * math.pi))
            tz = float(rng.uniform(0, 2 * math.pi))
            qc.ry(ty, i)
            qc.rz(tz, i)
            ir.add_gate(QBGate("ry", (i,), params=(ty,)))
            ir.add_gate(QBGate("rz", (i,), params=(tz,)))
        if rep < reps:
            for i in range(n):
                for j in range(i + 1, n):
                    qc.cx(i, j)
                    ir.add_gate(QBGate("cx", (i, j)))

    qc.measure(range(n), range(n))
    for i in range(n):
        ir.add_measurement(i, i)

    return qc, ir, "su2_8"


def count_2q(qc: QuantumCircuit) -> int:
    return sum(
        1 for inst in qc.data
        if len(inst.qubits) == 2
        and inst.operation.name not in ("barrier", "measure", "reset")
    )


def try_insert_dd(circuit: QuantumCircuit, target: Any) -> QuantumCircuit:
    """Try to insert DD. Returns original circuit if DD fails."""
    try:
        dd_pm = PassManager([
            ALAPScheduleAnalysis(target=target),
            PadDynamicalDecoupling(
                target=target,
                dd_sequence=[XGate(), XGate()],
            ),
        ])
        result = dd_pm.run(circuit)
        dd_x = result.count_ops().get("x", 0) - circuit.count_ops().get("x", 0)
        if dd_x > 0:
            console.print(f"    DD: inserted {dd_x} X gates")
        return result
    except Exception as e:
        console.print(f"    [dim]DD skipped: {e}[/dim]")
        return circuit


def est_fidelity_routed(tc: QuantumCircuit, props: BackendProperties) -> float:
    """Fidelity from routed circuit's actual gate placements."""
    fid = 1.0
    for inst in tc.data:
        if (len(inst.qubits) == 2
                and inst.operation.name not in ("barrier", "measure", "reset")):
            q0 = tc.find_bit(inst.qubits[0]).index
            q1 = tc.find_bit(inst.qubits[1]).index
            err = 0.02
            for gp in props.gate_properties:
                if len(gp.qubits) == 2 and gp.error_rate is not None:
                    if set(gp.qubits) == {q0, q1}:
                        err = min(err, gp.error_rate)
            fid *= (1.0 - err)
    measured = set()
    for inst in tc.data:
        if inst.operation.name == "measure":
            measured.add(tc.find_bit(inst.qubits[0]).index)
    for q in measured:
        qp = props.qubit(q)
        ro = qp.readout_error if qp and qp.readout_error is not None else 0.015
        fid *= (1.0 - ro)
    return fid


def props_from_backend_target(backend: Any) -> BackendProperties:
    """Build BackendProperties from a live IBM backend."""
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties

    target = backend.target
    n_qubits = target.num_qubits

    qubit_props_list = []
    for q in range(n_qubits):
        t1, t2, ro_err, freq = None, None, None, None
        try:
            bp = backend.properties()
            if bp:
                t1 = bp.t1(q)
                t2 = bp.t2(q)
                ro_err = bp.readout_error(q)
                freq = bp.frequency(q)
        except Exception:
            pass
        if t1 is None:
            try:
                tgt_qp = target.qubit_properties
                if tgt_qp and q < len(tgt_qp) and tgt_qp[q]:
                    t1 = getattr(tgt_qp[q], "t1", None)
                    t2 = getattr(tgt_qp[q], "t2", None)
                    freq = getattr(tgt_qp[q], "frequency", None)
            except Exception:
                pass
        qubit_props_list.append(QubitProperties(
            qubit_id=q,
            t1_us=(t1 * 1e6) if t1 else 100.0,
            t2_us=(t2 * 1e6) if t2 else 80.0,
            readout_error=ro_err if ro_err is not None else 0.015,
            frequency_ghz=(freq * 1e-9) if freq else 5.0,
        ))

    gate_props_list = []
    coupling_map = []
    for op_name in target.operation_names:
        qargs = target.qargs_for_operation_name(op_name)
        if qargs is None:
            continue
        for qarg in qargs:
            if len(qarg) == 2:
                pe = target[op_name].get(qarg)
                gate_props_list.append(GateProperties(
                    gate_type=op_name,
                    qubits=(qarg[0], qarg[1]),
                    error_rate=pe.error if pe and pe.error else None,
                    gate_time_ns=(pe.duration * 1e9) if pe and pe.duration else 68.0,
                ))
                coupling_map.append((qarg[0], qarg[1]))

    basis_2q = {gp.gate_type for gp in gate_props_list}
    return BackendProperties(
        backend=backend.name,
        provider="ibm",
        n_qubits=n_qubits,
        basis_gates=tuple(sorted(basis_2q)) or ("cz",),
        qubit_properties=qubit_props_list,
        gate_properties=gate_props_list,
        coupling_map=coupling_map,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


def get_qb_layout(ir: QBCircuit, props: BackendProperties, target: Target):
    """Get qb-compiler layout."""
    mapper = CalibrationMapper(
        props,
        config=CalibrationMapperConfig(
            max_candidates=500, vf2_call_limit=50_000,
            top_k=20, max_per_region=3,
        ),
        qiskit_target=target,
    )
    ctx: dict[str, Any] = {}
    mapper.run(ir, ctx)
    return ctx["initial_layout"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--channel", type=str, default=None)
    parser.add_argument("--instance", type=str, default=None)
    args = parser.parse_args()

    console.print("[bold]═══ qb-compiler Hardware Validation v3 ═══[/bold]")
    console.print(f"Backend: {BACKEND_NAME} | Shots: {SHOTS}")
    console.print("Circuits: QAOA-6, QFT-6, SU2-8")
    console.print("Strategy: qb layout + Qiskit routing + DD vs Qiskit default")
    console.print()

    # Connect
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError:
        console.print("[red]pip install qiskit-ibm-runtime[/red]")
        sys.exit(1)

    console.print("[bold]Connecting...[/bold]")
    service_kwargs: dict[str, Any] = {}
    if args.channel:
        service_kwargs["channel"] = args.channel
    if args.instance:
        service_kwargs["instance"] = args.instance
    if not service_kwargs:
        service = QiskitRuntimeService(name="default-ibm-quantum-platform")
    else:
        service = QiskitRuntimeService(**service_kwargs)

    backend = service.backend(BACKEND_NAME)
    target = backend.target
    console.print(f"  Connected: {backend.name}, {target.num_qubits} qubits")

    props = props_from_backend_target(backend)
    console.print(f"  Calibration: {props.timestamp}")

    # Build circuits
    circuit_builders = [build_qaoa_maxcut_6, build_qft_6, build_efficientsu2_8]
    all_circuits = [builder() for builder in circuit_builders]

    console.print("\n[bold]Computing layouts...[/bold]\n")

    plan: list[dict[str, Any]] = []
    circuits_to_submit: list[tuple[str, QuantumCircuit]] = []

    for qc, ir, name in all_circuits:
        n = qc.num_qubits
        console.print(f"  {name}: {n}q, {count_2q(qc)} original 2Q gates")

        # A) Qiskit best of 20 seeds (NO DD — what users get by default)
        best_qi_tc = None
        best_qi_fid = -1.0
        best_qi_seed = -1
        for seed in QISKIT_SEEDS:
            tc = transpile(qc, target=target, optimization_level=3, seed_transpiler=seed)
            fid = est_fidelity_routed(tc, props)
            if fid > best_qi_fid:
                best_qi_fid = fid
                best_qi_tc = tc
                best_qi_seed = seed

        qi_2q = count_2q(best_qi_tc)
        ql = best_qi_tc.layout
        qi_phys = ([ql.initial_layout[qc.qubits[i]] for i in range(n)]
                   if ql and ql.initial_layout else list(range(n)))

        # B) qb-compiler layout + Qiskit opt_level=3 routing + DD
        qb_layout = get_qb_layout(ir, props, target)
        qb_phys = [qb_layout[i] for i in range(n)]

        # Use opt_level=3 so VF2PostLayout can refine our layout
        tc_qb = transpile(
            qc, target=target, optimization_level=3,
            initial_layout=qb_phys, seed_transpiler=42,
        )
        qb_2q = count_2q(tc_qb)
        qb_fid = est_fidelity_routed(tc_qb, props)

        # Add DD to qb-compiler circuit (uses real backend target with durations)
        tc_qb_dd = try_insert_dd(tc_qb, target)

        entry = {
            "circuit": name,
            "n_qubits": n,
            "original_2q": count_2q(qc),
            "qiskit": {
                "seed": best_qi_seed,
                "qubits": qi_phys,
                "region": f"{min(qi_phys)}-{max(qi_phys)}",
                "post_routing_2q": qi_2q,
                "depth": best_qi_tc.depth(),
                "est_fidelity": round(best_qi_fid, 6),
            },
            "qb_compiler": {
                "qubits": qb_phys,
                "region": f"{min(qb_phys)}-{max(qb_phys)}",
                "post_routing_2q": qb_2q,
                "depth": tc_qb.depth(),
                "est_fidelity": round(qb_fid, 6),
                "dd_applied": tc_qb_dd is not tc_qb,
            },
        }
        plan.append(entry)

        circuits_to_submit.append((f"{name}_qiskit_s{best_qi_seed}", best_qi_tc))
        circuits_to_submit.append((f"{name}_qb_dd", tc_qb_dd))

        console.print(f"    Qiskit:  {qi_2q} 2Q gates, fid={best_qi_fid:.4f}, qubits={qi_phys[:5]}...")
        console.print(f"    qb+DD:   {qb_2q} 2Q gates, fid={qb_fid:.4f}, qubits={qb_phys[:5]}...")

    # Dry-run table
    console.print(f"\n[bold]═══ {'DRY-RUN ' if args.dry_run else ''}PLAN ═══[/bold]\n")

    tbl = Table(title=f"{'DRY-RUN: ' if args.dry_run else ''}6 Circuits to Submit")
    tbl.add_column("Circuit")
    tbl.add_column("Approach")
    tbl.add_column("2Q Gates", justify="right")
    tbl.add_column("Depth", justify="right")
    tbl.add_column("Est Fidelity", justify="right")
    tbl.add_column("Region")

    for entry in plan:
        qi = entry["qiskit"]
        qb = entry["qb_compiler"]
        delta_fid = qb["est_fidelity"] - qi["est_fidelity"]
        delta_color = "green" if delta_fid > 0.001 else ("red" if delta_fid < -0.001 else "dim")

        tbl.add_row(entry["circuit"], f"Qiskit (s={qi['seed']})",
                     str(qi["post_routing_2q"]), str(qi["depth"]),
                     f"{qi['est_fidelity']:.4f}", qi["region"])
        tbl.add_row("", "qb+DD",
                     str(qb["post_routing_2q"]), str(qb["depth"]),
                     f"[{delta_color}]{qb['est_fidelity']:.4f} ({delta_fid:+.4f})[/{delta_color}]",
                     qb["region"])
        tbl.add_row()

    console.print(tbl)

    if args.dry_run:
        console.print("\n[yellow bold]DRY RUN — no submission[/yellow bold]")
        out = Path("results/hardware_validation_v3_dryrun.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"mode": "dry_run", "plan": plan,
                       "timestamp": datetime.now(tz=timezone.utc).isoformat()}, f, indent=2)
        console.print(f"  Saved: {out}")
        return

    # Confirm
    console.print(f"\n[bold yellow]Submit {len(circuits_to_submit)} circuits to {BACKEND_NAME}?[/bold yellow]")
    if not args.yes:
        if input("  Proceed? (yes/no): ").strip().lower() not in ("yes", "y"):
            console.print("[red]Aborted.[/red]")
            return
    else:
        console.print("  --yes flag set, proceeding...")

    # Submit
    console.print("\n[bold]Submitting...[/bold]")
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = SHOTS

    all_tc = [tc for _, tc in circuits_to_submit]
    labels = [label for label, _ in circuits_to_submit]
    for i, label in enumerate(labels):
        console.print(f"  [{i}] {label}")

    t0 = time.monotonic()
    job = sampler.run(all_tc)
    console.print(f"  Job: {job.job_id()}")
    console.print("  Waiting...")
    result = job.result()
    elapsed = time.monotonic() - t0
    console.print(f"  Done in {elapsed:.1f}s")

    # Results
    console.print("\n[bold]═══ HARDWARE RESULTS ═══[/bold]\n")

    all_counts = []
    for pub_result in result:
        creg = list(pub_result.data.keys())[0]
        all_counts.append(getattr(pub_result.data, creg).get_counts())

    res_tbl = Table(title="HARDWARE: qb-compiler+DD vs Qiskit Default")
    res_tbl.add_column("Circuit", style="bold")
    res_tbl.add_column("Qiskit HW Fid", justify="right")
    res_tbl.add_column("qb+DD HW Fid", justify="right")
    res_tbl.add_column("Delta", justify="right")
    res_tbl.add_column("Winner")

    hw_results = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "backend": BACKEND_NAME,
        "shots": SHOTS,
        "elapsed_s": elapsed,
        "circuits": [],
    }

    wins, losses, ties = 0, 0, 0
    idx = 0

    for entry in plan:
        n = entry["n_qubits"]
        counts_qi = all_counts[idx]
        counts_qb = all_counts[idx + 1]
        idx += 2

        # Generic fidelity: fraction of most common bitstring
        # (not GHZ-specific — works for any circuit)
        total_qi = sum(counts_qi.values())
        total_qb = sum(counts_qb.values())
        # Heavy output probability: fraction of outputs in top 50% of ideal distribution
        # For general circuits, use entropy-based metric
        # Simple metric: probability of most common output
        top_qi = max(counts_qi.values()) / total_qi if total_qi else 0
        top_qb = max(counts_qb.values()) / total_qb if total_qb else 0

        delta = top_qb - top_qi
        if delta > 0.005:
            wins += 1
            winner = "[green bold]qb+DD[/green bold]"
        elif delta < -0.005:
            losses += 1
            winner = "[red bold]Qiskit[/red bold]"
        else:
            ties += 1
            winner = "[dim]TIE[/dim]"

        dc = "green" if delta > 0.005 else ("red" if delta < -0.005 else "dim")
        res_tbl.add_row(
            entry["circuit"],
            f"{top_qi:.4f}",
            f"{top_qb:.4f}",
            f"[{dc}]{delta:+.4f}[/{dc}]",
            winner,
        )
        hw_results["circuits"].append({
            **entry,
            "hardware": {
                "qiskit_top_prob": round(top_qi, 6),
                "qb_top_prob": round(top_qb, 6),
                "delta": round(delta, 6),
                "counts_qiskit_top5": dict(sorted(counts_qi.items(), key=lambda x: -x[1])[:5]),
                "counts_qb_top5": dict(sorted(counts_qb.items(), key=lambda x: -x[1])[:5]),
            },
        })

    console.print(res_tbl)
    console.print(f"\n  Wins: {wins}, Losses: {losses}, Ties: {ties}")

    hw_results["summary"] = {"wins": wins, "losses": losses, "ties": ties}

    for out in [
        Path("results/hardware_validation_v3.json"),
        Path("/mnt/d/hardware_validation_v3.json"),
    ]:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(hw_results, f, indent=2, default=str)
        console.print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
