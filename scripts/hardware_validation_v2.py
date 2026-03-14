#!/usr/bin/env python3
"""Hardware validation v2: qb-compiler vs Qiskit BEST on IBM Fez.

6 circuits total: GHZ-5, GHZ-8, GHZ-10 × 2 approaches.
Uses FRESH calibration from live backend, NOT cached data.
Qiskit baseline = BEST of 20 seeds.

Usage::

    python scripts/hardware_validation_v2.py           # full run
    python scripts/hardware_validation_v2.py --dry-run  # dry run only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CZGate, CXGate, HGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target
from rich.console import Console
from rich.table import Table

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.calibration_mapper import (
    CalibrationMapper,
    CalibrationMapperConfig,
)

console = Console()

BACKEND_NAME = "ibm_fez"
GHZ_SIZES = [5, 8, 10]
SHOTS = 4096
QISKIT_SEEDS = list(range(20))
EST_SECONDS_PER_CIRCUIT = 1.5


# ── Circuit builders ──────────────────────────────────────────────────


def build_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n, name=f"ghz_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def build_ghz_ir(n: int) -> QBCircuit:
    circ = QBCircuit(n_qubits=n, n_clbits=n, name=f"ghz_{n}")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    for i in range(n - 1):
        circ.add_gate(QBGate(name="cx", qubits=(i, i + 1)))
    for i in range(n):
        circ.add_measurement(i, i)
    return circ


def count_2q(qc: QuantumCircuit) -> int:
    return sum(
        1 for inst in qc.data
        if len(inst.qubits) == 2
        and inst.operation.name not in ("barrier", "measure", "reset")
    )


def ghz_fidelity(counts: dict[str, int], n: int) -> float:
    """GHZ fidelity = P(000...0) + P(111...1)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (counts.get("0" * n, 0) + counts.get("1" * n, 0)) / total


def est_fidelity(layout: dict[int, int], props: BackendProperties, n: int) -> float:
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


# ── Calibration from live backend ─────────────────────────────────────


def props_from_backend_target(backend: Any) -> BackendProperties:
    """Build BackendProperties from a live IBM backend's target.

    This fetches FRESH calibration — not cached fixture data.
    """
    target = backend.target
    n_qubits = target.num_qubits

    from qb_compiler.calibration.models.qubit_properties import QubitProperties
    from qb_compiler.calibration.models.coupling_properties import GateProperties

    qubit_props_list = []
    for q in range(n_qubits):
        # Get qubit properties from target
        t1 = None
        t2 = None
        ro_err = None
        freq = None

        # Try to get from backend.properties() if available
        try:
            bp = backend.properties()
            if bp:
                t1 = bp.t1(q)
                t2 = bp.t2(q)
                ro_err = bp.readout_error(q)
                freq = bp.frequency(q)
        except Exception:
            pass

        # Fallback: try target qubit properties
        if t1 is None:
            try:
                tgt_qp = target.qubit_properties
                if tgt_qp and q < len(tgt_qp) and tgt_qp[q]:
                    t1 = getattr(tgt_qp[q], 't1', None)
                    t2 = getattr(tgt_qp[q], 't2', None)
                    freq = getattr(tgt_qp[q], 'frequency', None)
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
                props_entry = target[op_name].get(qarg)
                error = props_entry.error if props_entry and props_entry.error else None
                duration = props_entry.duration if props_entry and props_entry.duration else None
                gate_props_list.append(GateProperties(
                    gate_type=op_name,
                    qubits=(qarg[0], qarg[1]),
                    error_rate=error,
                    gate_time_ns=(duration * 1e9) if duration else 68.0,
                ))
                coupling_map.append((qarg[0], qarg[1]))

    # Detect basis gates from target
    basis_2q = set()
    for gp in gate_props_list:
        basis_2q.add(gp.gate_type)
    basis_gates = tuple(sorted(basis_2q)) or ("cz",)

    return BackendProperties(
        backend=backend.name,
        provider="ibm",
        n_qubits=n_qubits,
        basis_gates=basis_gates,
        qubit_properties=qubit_props_list,
        gate_properties=gate_props_list,
        coupling_map=coupling_map,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


# ── Layout selection ──────────────────────────────────────────────────


def get_qb_layout(
    n: int, props: BackendProperties, target: Target,
) -> tuple[dict[int, int], CalibrationMapper]:
    """Get qb-compiler layout with post-routing rescoring."""
    circ_ir = build_ghz_ir(n)
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
    return ctx["initial_layout"], mapper


def get_qiskit_best(
    qc: QuantumCircuit, target: Target, n: int, props: BackendProperties,
) -> dict[str, Any]:
    """Run Qiskit with 20 seeds, return the best by estimated fidelity."""
    best: dict[str, Any] | None = None

    for seed in QISKIT_SEEDS:
        tc = transpile(qc, target=target, optimization_level=3, seed_transpiler=seed)
        ql = tc.layout
        if ql and ql.initial_layout:
            phys = [ql.initial_layout[qc.qubits[i]] for i in range(n)]
        else:
            phys = list(range(n))
        layout = {i: phys[i] for i in range(n)}
        fid = est_fidelity(layout, props, n)

        entry = {
            "seed": seed,
            "physical_qubits": phys,
            "region": f"{min(phys)}-{max(phys)}",
            "post_routing_2q_gates": count_2q(tc),
            "post_routing_depth": tc.depth(),
            "estimated_fidelity": fid,
            "transpiled_circuit": tc,
        }
        if best is None or fid > best["estimated_fidelity"]:
            best = entry

    return best  # type: ignore[return-value]


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware validation v2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without QPU submission")
    parser.add_argument("--channel", type=str, default=None)
    parser.add_argument("--instance", type=str, default=None)
    parser.add_argument("--calibration", type=str, default=None,
                        help="Use cached calibration JSON instead of live (for dry-run)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    console.print("[bold]═══ qb-compiler Hardware Validation v2 ═══[/bold]")
    console.print(f"Backend: {BACKEND_NAME} | Shots: {SHOTS}")
    console.print(f"Circuits: GHZ-{', GHZ-'.join(str(n) for n in GHZ_SIZES)}")
    console.print(f"Approaches: A) Qiskit best-of-{len(QISKIT_SEEDS)} seeds  B) qb-compiler layout")
    console.print(f"Total circuits: {len(GHZ_SIZES) * 2}")
    console.print()

    # ── Connect to backend ─────────────────────────────────────────

    backend = None
    target = None
    props = None

    if not args.dry_run or args.calibration is None:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        except ImportError:
            console.print("[red]pip install qiskit-ibm-runtime[/red]")
            sys.exit(1)

        console.print("[bold]Connecting to IBM Quantum...[/bold]")
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

        # FRESH calibration from live backend
        console.print("  Fetching FRESH calibration...")
        props = props_from_backend_target(backend)
        console.print(f"  Calibration timestamp: {props.timestamp}")
        console.print(f"  Qubits: {props.n_qubits}, Gates: {len(props.gate_properties)}")

        # Diagnostic: how many qubits have real calibration?
        real_t1 = sum(1 for qp in props.qubit_properties if qp.t1_us != 100.0)
        real_ro = sum(1 for qp in props.qubit_properties if qp.readout_error != 0.015)
        real_gate_err = sum(1 for gp in props.gate_properties if gp.error_rate is not None)
        console.print(f"  Calibration coverage: T1={real_t1}/{props.n_qubits}, "
                       f"readout={real_ro}/{props.n_qubits}, "
                       f"gate_err={real_gate_err}/{len(props.gate_properties)}")

        # Also save calibration snapshot
        snap_path = Path("tests/fixtures/calibration_snapshots")
        snap_path.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now(tz=timezone.utc).strftime("%Y_%m_%d")
        snap_file = snap_path / f"ibm_fez_{ts_str}_live.json"
        # We'd need to serialize here — skip for now, focus on validation

    elif args.calibration:
        cal_path = Path(args.calibration)
        props = BackendProperties.from_qubitboost_json(cal_path)
        console.print(f"  Using cached calibration: {cal_path}")
        console.print(f"  Timestamp: {props.timestamp}")
        # Build target from cached props for dry-run
        n_q = props.n_qubits
        target = Target(num_qubits=n_q)
        gate_types = {gp.gate_type for gp in props.gate_properties if len(gp.qubits) == 2}
        gate_2q_cls = CZGate if "cz" in gate_types else CXGate
        twoq_props: dict[tuple[int, int], InstructionProperties | None] = {}
        for gp in props.gate_properties:
            if len(gp.qubits) == 2:
                twoq_props[gp.qubits] = InstructionProperties(
                    error=gp.error_rate,
                    duration=gp.gate_time_ns * 1e-9 if gp.gate_time_ns else 68e-9,
                )
        target.add_instruction(gate_2q_cls(), twoq_props)
        sq_props: dict[tuple[int, ...], InstructionProperties | None] = {
            (q,): None for q in range(n_q)
        }
        theta = Parameter("theta")
        target.add_instruction(RZGate(theta), sq_props)
        target.add_instruction(SXGate(), sq_props)
        target.add_instruction(XGate(), sq_props)
        target.add_instruction(HGate(), sq_props)
        target.add_instruction(Measure(), sq_props)

    assert props is not None
    assert target is not None

    # ── Phase 1: Compute layouts ───────────────────────────────────

    console.print("\n[bold]Phase 1: Computing layouts[/bold]\n")

    circuits_to_submit: list[tuple[str, QuantumCircuit]] = []
    plan: list[dict[str, Any]] = []

    for n in GHZ_SIZES:
        qc = build_ghz(n)

        # Approach A: Qiskit BEST of 20 seeds
        console.print(f"  GHZ-{n}: Testing {len(QISKIT_SEEDS)} Qiskit seeds...")
        qiskit_best = get_qiskit_best(qc, target, n, props)

        # Approach B: qb-compiler layout + Qiskit routing
        console.print(f"  GHZ-{n}: Running qb-compiler mapper...")
        qb_layout, mapper = get_qb_layout(n, props, target)
        qb_phys = [qb_layout[i] for i in range(n)]

        # Transpile with our layout using same optimization_level=3
        tc_qb = transpile(
            qc, target=target, optimization_level=3,
            initial_layout=qb_phys, seed_transpiler=42,
        )

        qb_fid_est = est_fidelity(qb_layout, props, n)

        entry = {
            "circuit": f"GHZ-{n}",
            "n_qubits": n,
            "qb_compiler": {
                "physical_qubits": qb_phys,
                "region": f"{min(qb_phys)}-{max(qb_phys)}",
                "post_routing_2q_gates": count_2q(tc_qb),
                "post_routing_depth": tc_qb.depth(),
                "estimated_fidelity": round(qb_fid_est, 6),
            },
            "qiskit_best": {
                "seed": qiskit_best["seed"],
                "physical_qubits": qiskit_best["physical_qubits"],
                "region": qiskit_best["region"],
                "post_routing_2q_gates": qiskit_best["post_routing_2q_gates"],
                "post_routing_depth": qiskit_best["post_routing_depth"],
                "estimated_fidelity": round(qiskit_best["estimated_fidelity"], 6),
            },
            "delta_estimated": round(qb_fid_est - qiskit_best["estimated_fidelity"], 6),
        }
        plan.append(entry)

        # Queue circuits: Qiskit best first, then qb-compiler
        circuits_to_submit.append((f"ghz_{n}_qiskit_seed{qiskit_best['seed']}", qiskit_best["transpiled_circuit"]))
        circuits_to_submit.append((f"ghz_{n}_qb", tc_qb))

    # ── Dry-run summary ───────────────────────────────────────────

    console.print("\n[bold]═══ DRY-RUN SUMMARY ═══[/bold]\n")

    dry_table = Table(title="Pre-submission Plan: 6 Circuits")
    dry_table.add_column("Circuit", style="bold")
    dry_table.add_column("Approach")
    dry_table.add_column("Physical Qubits")
    dry_table.add_column("Region")
    dry_table.add_column("2Q Gates\n(post-route)", justify="right")
    dry_table.add_column("Depth\n(post-route)", justify="right")
    dry_table.add_column("Est. Fidelity", justify="right")

    for entry in plan:
        qb = entry["qb_compiler"]
        qs = entry["qiskit_best"]

        # Qiskit row
        dry_table.add_row(
            entry["circuit"],
            f"Qiskit (seed={qs['seed']})",
            " ".join(str(q) for q in qs["physical_qubits"][:6]) + ("..." if len(qs["physical_qubits"]) > 6 else ""),
            qs["region"],
            str(qs["post_routing_2q_gates"]),
            str(qs["post_routing_depth"]),
            f"{qs['estimated_fidelity']:.4f}",
        )

        # qb-compiler row
        delta = entry["delta_estimated"]
        delta_color = "green" if delta > 0.001 else ("red" if delta < -0.001 else "dim")
        dry_table.add_row(
            "",
            "qb-compiler",
            " ".join(str(q) for q in qb["physical_qubits"][:6]) + ("..." if len(qb["physical_qubits"]) > 6 else ""),
            qb["region"],
            str(qb["post_routing_2q_gates"]),
            str(qb["post_routing_depth"]),
            f"[{delta_color}]{qb['estimated_fidelity']:.4f} ({delta:+.4f})[/{delta_color}]",
        )
        dry_table.add_row("", "", "", "", "", "", "")

    console.print(dry_table)

    n_circuits = len(circuits_to_submit)
    est_min = (n_circuits * SHOTS * 1e-6 * 50 + n_circuits * 2) / 60  # rough estimate
    console.print(f"\n  Total circuits: {n_circuits}")
    console.print(f"  Shots per circuit: {SHOTS}")
    console.print(f"  Estimated QPU time: ~{n_circuits * EST_SECONDS_PER_CIRCUIT:.0f}s")
    console.print(f"  Calibration timestamp: {props.timestamp}")

    if args.dry_run:
        console.print("\n[yellow bold]DRY RUN COMPLETE — no circuits submitted[/yellow bold]")

        # Save dry-run results
        dry_results = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "mode": "dry_run",
            "calibration_timestamp": props.timestamp,
            "plan": plan,
        }
        for out in [
            Path("results/hardware_validation_v2_dryrun.json"),
            Path("/mnt/d/hardware_validation_v2_dryrun.json"),
        ]:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(dry_results, f, indent=2)
            console.print(f"  Saved: {out}")

        return

    # ── Confirmation ──────────────────────────────────────────────

    console.print("\n[bold yellow]>>> CONFIRMATION REQUIRED <<<[/bold yellow]")
    console.print(f"  About to submit {n_circuits} circuits to {BACKEND_NAME}")
    console.print(f"  This will use ~{n_circuits * EST_SECONDS_PER_CIRCUIT:.0f}s of QPU time")
    if not args.yes:
        confirm = input("\n  Mike — proceed? (yes/no): ").strip().lower()
        if confirm not in ("yes", "y"):
            console.print("[red]Aborted.[/red]")
            sys.exit(0)
    else:
        console.print("  --yes flag set, proceeding...")

    # ── Phase 2: Submit to QPU ────────────────────────────────────

    console.print("\n[bold]Phase 2: Submitting to QPU[/bold]\n")

    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = SHOTS

    # Batch all 6 circuits into one job
    transpiled_circuits = [tc for _, tc in circuits_to_submit]
    labels = [label for label, _ in circuits_to_submit]

    console.print(f"  Submitting {len(transpiled_circuits)} circuits as single batch...")
    for i, label in enumerate(labels):
        console.print(f"    [{i}] {label}")

    t_start = time.monotonic()
    job = sampler.run(transpiled_circuits)
    console.print(f"  Job ID: {job.job_id()}")
    console.print("  Waiting for results...")

    result = job.result()
    t_elapsed = time.monotonic() - t_start
    console.print(f"  Completed in {t_elapsed:.1f}s")

    # ── Phase 3: Extract and analyze results ──────────────────────

    console.print("\n[bold]Phase 3: Results[/bold]\n")

    all_counts: list[dict[str, int]] = []
    for pub_result in result:
        creg_name = list(pub_result.data.keys())[0]
        bitarray = getattr(pub_result.data, creg_name)
        all_counts.append(bitarray.get_counts())

    # Build results table
    results_table = Table(title="HARDWARE RESULTS: qb-compiler vs Qiskit Best")
    results_table.add_column("Circuit", style="bold")
    results_table.add_column("Qiskit Best", justify="right")
    results_table.add_column("qb-compiler", justify="right")
    results_table.add_column("Delta", justify="right")
    results_table.add_column("Winner", justify="center")

    hw_results: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "backend": BACKEND_NAME,
        "calibration_timestamp": props.timestamp,
        "shots": SHOTS,
        "qiskit_seeds_tested": len(QISKIT_SEEDS),
        "total_execution_time_s": t_elapsed,
        "circuits": [],
    }

    wins = 0
    losses = 0
    ties = 0
    idx = 0

    for i, n in enumerate(GHZ_SIZES):
        counts_qiskit = all_counts[idx]
        counts_qb = all_counts[idx + 1]
        idx += 2

        fid_qiskit = ghz_fidelity(counts_qiskit, n)
        fid_qb = ghz_fidelity(counts_qb, n)
        delta = fid_qb - fid_qiskit

        if delta > 0.005:
            wins += 1
            winner = "qb-compiler"
            winner_rich = "[green bold]qb-compiler[/green bold]"
        elif delta < -0.005:
            losses += 1
            winner = "qiskit"
            winner_rich = "[red bold]Qiskit[/red bold]"
        else:
            ties += 1
            winner = "tie"
            winner_rich = "[dim]TIE[/dim]"

        delta_color = "green" if delta > 0.005 else ("red" if delta < -0.005 else "dim")

        results_table.add_row(
            f"GHZ-{n}",
            f"{fid_qiskit:.4f}",
            f"{fid_qb:.4f}",
            f"[{delta_color}]{delta:+.4f} ({delta*100:+.1f}%)[/{delta_color}]",
            winner_rich,
        )

        circuit_result = {
            **plan[i],
            "hardware": {
                "qiskit_fidelity": round(fid_qiskit, 6),
                "qb_compiler_fidelity": round(fid_qb, 6),
                "delta": round(delta, 6),
                "winner": winner,
                "counts_qiskit": counts_qiskit,
                "counts_qb": counts_qb,
            },
        }
        hw_results["circuits"].append(circuit_result)

    console.print(results_table)

    hw_results["summary"] = {
        "qb_compiler_wins": wins,
        "qiskit_wins": losses,
        "ties": ties,
        "total": len(GHZ_SIZES),
        "success": wins >= 2,
    }

    # Verdict
    console.print()
    if wins >= 2:
        console.print("[bold green]SUCCESS: qb-compiler wins on >=2 of 3 circuits[/bold green]")
    elif wins + ties >= 2:
        console.print("[bold yellow]PARTIAL: qb-compiler competitive but not clearly winning[/bold yellow]")
    else:
        console.print("[bold red]FAIL: qb-compiler not beating Qiskit[/bold red]")

    console.print(f"\n  Wins: {wins}/{len(GHZ_SIZES)}")
    console.print(f"  Losses: {losses}/{len(GHZ_SIZES)}")
    console.print(f"  Ties: {ties}/{len(GHZ_SIZES)}")

    # ── Save results ─────────────────────────────────────────────

    for out in [
        Path("results/hardware_validation_v2.json"),
        Path("/mnt/d/hardware_validation_v2.json"),
    ]:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(hw_results, f, indent=2, default=str)
        console.print(f"  Results saved: {out}")


if __name__ == "__main__":
    main()
