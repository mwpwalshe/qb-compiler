#!/usr/bin/env python3
"""Local post-routing validation using real calibration data.

Compares qb-compiler (NEW mapper with windowed VF2 + post-routing rescoring)
against Qiskit's BEST of 20 seeds (not just seed=42).

Also diagnoses whether the multi-region search and post-routing rescoring
are actually producing differentiated candidates.

Usage::

    python scripts/local_post_routing_validation.py
"""

from __future__ import annotations

import json
import logging
import sys
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

GHZ_SIZES = [3, 5, 8, 10]
QISKIT_SEEDS = list(range(20))

# Enable CalibrationMapper logging so we can see diagnostics
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
cal_logger = logging.getLogger("qb_compiler.passes.mapping.calibration_mapper")
cal_logger.setLevel(logging.INFO)


def build_target(props: BackendProperties) -> Target:
    n_qubits = props.n_qubits
    target = Target(num_qubits=n_qubits)
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
        (q,): None for q in range(n_qubits)
    }
    theta = Parameter("theta")
    target.add_instruction(RZGate(theta), sq_props)
    target.add_instruction(SXGate(), sq_props)
    target.add_instruction(XGate(), sq_props)
    target.add_instruction(HGate(), sq_props)
    target.add_instruction(Measure(), sq_props)
    return target


def build_ghz_qiskit(n: int) -> QuantumCircuit:
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


def get_qb_layout(circ_ir: QBCircuit, props: BackendProperties, target: Target) -> tuple[dict[int, int], CalibrationMapper]:
    """Returns (layout, mapper) so we can inspect mapper diagnostics."""
    mapper = CalibrationMapper(
        props,
        config=CalibrationMapperConfig(
            max_candidates=500, vf2_call_limit=50_000, top_k=20, max_per_region=3,
        ),
        qiskit_target=target,
    )
    ctx: dict[str, Any] = {}
    mapper.run(circ_ir, ctx)
    return ctx["initial_layout"], mapper


def get_qb_layout_old(circ_ir: QBCircuit, props: BackendProperties) -> dict[int, int]:
    mapper = CalibrationMapper(
        props,
        config=CalibrationMapperConfig(
            max_candidates=500, vf2_call_limit=50_000, top_k=1,
        ),
        qiskit_target=None,
    )
    ctx: dict[str, Any] = {}
    mapper.run(circ_ir, ctx)
    return ctx["initial_layout"]


def run_qiskit_multi_seed(
    qc: QuantumCircuit, target: Target, n: int, props: BackendProperties
) -> list[dict[str, Any]]:
    """Run Qiskit transpile with multiple seeds, return results sorted by fidelity."""
    results = []
    for seed in QISKIT_SEEDS:
        tc = transpile(qc, target=target, optimization_level=3, seed_transpiler=seed)
        ql = tc.layout
        if ql and ql.initial_layout:
            phys = [ql.initial_layout[qc.qubits[i]] for i in range(n)]
        else:
            phys = list(range(n))
        layout = {i: phys[i] for i in range(n)}
        fid = est_fidelity(layout, props, n)
        results.append({
            "seed": seed,
            "physical_qubits": phys,
            "region": f"{min(phys)}-{max(phys)}",
            "post_routing_2q_gates": count_2q(tc),
            "post_routing_depth": tc.depth(),
            "estimated_fidelity": fid,
        })
    results.sort(key=lambda r: -r["estimated_fidelity"])
    return results


def main() -> None:
    fixture_dir = Path("tests/fixtures/calibration_snapshots")
    snapshots = sorted(fixture_dir.glob("ibm_fez_*.json"))

    full_snapshots: list[tuple[str, BackendProperties]] = []
    for path in snapshots:
        try:
            props = BackendProperties.from_qubitboost_json(path)
            if len(props.qubit_properties) >= 50:
                full_snapshots.append((path.stem, props))
        except Exception:
            pass

    if not full_snapshots:
        console.print("[red]No full-chip calibration snapshots found[/red]")
        sys.exit(1)

    console.print(f"[bold]Validation: qb-compiler vs Qiskit BEST of {len(QISKIT_SEEDS)} seeds[/bold]\n")

    all_results: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "description": (
            f"qb-compiler (windowed VF2 + post-routing rescoring) vs "
            f"Qiskit opt_level=3 BEST of {len(QISKIT_SEEDS)} seeds. "
            "No QPU time — estimated fidelity from calibration data."
        ),
        "snapshots": [],
    }

    # Summary table
    summary_table = Table(title="qb-compiler vs Qiskit BEST seed")
    summary_table.add_column("Snapshot", style="bold")
    summary_table.add_column("GHZ", justify="right")
    summary_table.add_column("qb Fidelity", justify="right")
    summary_table.add_column("Qiskit Best Seed", justify="right")
    summary_table.add_column("Qiskit Best Fidelity", justify="right")
    summary_table.add_column("Delta", justify="right")
    summary_table.add_column("Winner")

    wins = 0
    losses = 0
    ties = 0
    total = 0

    for snap_name, props in full_snapshots:
        target = build_target(props)
        snap_result: dict[str, Any] = {
            "snapshot": snap_name,
            "n_qubits": props.n_qubits,
            "circuits": [],
        }

        console.print(f"\n[bold underline]{snap_name}[/bold underline]\n")

        for n in GHZ_SIZES:
            qc = build_ghz_qiskit(n)
            circ_ir = build_ghz_ir(n)

            # ── qb-compiler (NEW mapper) ──
            console.print(f"[dim]Running qb-compiler for GHZ-{n}...[/dim]")
            new_layout, mapper = get_qb_layout(circ_ir, props, target)
            new_phys = [new_layout[i] for i in range(n)]
            tc_new = transpile(qc, target=target, optimization_level=3,
                               initial_layout=new_phys, seed_transpiler=42)
            new_fid = est_fidelity(new_layout, props, n)

            # ── qb-compiler (OLD mapper) ──
            old_layout = get_qb_layout_old(circ_ir, props)
            old_phys = [old_layout[i] for i in range(n)]
            old_fid = est_fidelity(old_layout, props, n)

            # ── Diagnostics: what did the mapper's pipeline produce? ──
            raw_cands = getattr(mapper, '_last_raw_candidates', '?')
            raw_regions = getattr(mapper, '_last_raw_regions', '?')
            div_cands = getattr(mapper, '_last_diversified_candidates', '?')
            div_regions = getattr(mapper, '_last_diversified_regions', '?')
            rescore_details = getattr(mapper, '_last_rescore_details', [])

            console.print(f"\n[bold]GHZ-{n} Mapper Diagnostics:[/bold]")
            console.print(f"  Raw candidates: {raw_cands} ({raw_regions} distinct regions)")
            console.print(f"  After diversity filter: {div_cands} ({div_regions} regions)")
            console.print(f"  Post-routing rescore candidates:")

            unique_2q_counts = set()
            for d in rescore_details:
                unique_2q_counts.add(d["post_routing_2q"])
                console.print(
                    f"    #{d['index']}: qubits={d['physical_qubits']}  "
                    f"region={d['region']}  centroid={d['centroid']:.1f}  "
                    f"2Q={d['post_routing_2q']}  cal_score={d['cal_score']:.4f}"
                )

            if len(unique_2q_counts) <= 1:
                console.print(
                    f"  [yellow]WARNING: All candidates have identical 2Q count "
                    f"({unique_2q_counts}) — rescore provides no signal![/yellow]"
                )
            else:
                console.print(
                    f"  [green]Post-routing rescore differentiated: "
                    f"2Q counts = {sorted(unique_2q_counts)}[/green]"
                )

            console.print(f"  OLD layout: {old_phys} (fid={old_fid:.4f})")
            console.print(f"  NEW layout: {new_phys} (fid={new_fid:.4f})")
            console.print(f"  Same layout? {'YES' if old_phys == new_phys else 'NO'}")

            # ── Qiskit: best of 20 seeds ──
            console.print(f"[dim]Running Qiskit with {len(QISKIT_SEEDS)} seeds for GHZ-{n}...[/dim]")
            qiskit_results = run_qiskit_multi_seed(qc, target, n, props)
            best_qiskit = qiskit_results[0]

            console.print(f"\n  Qiskit top 5 seeds:")
            for r in qiskit_results[:5]:
                console.print(
                    f"    seed={r['seed']:2d}: qubits={r['physical_qubits']}  "
                    f"region={r['region']}  fid={r['estimated_fidelity']:.4f}"
                )

            # ── Comparison ──
            delta = new_fid - best_qiskit["estimated_fidelity"]
            total += 1
            if delta > 0.001:
                wins += 1
                winner = "qb-compiler"
                winner_rich = "[green]qb-compiler[/green]"
            elif delta < -0.001:
                losses += 1
                winner = "qiskit"
                winner_rich = "[red]Qiskit[/red]"
            else:
                ties += 1
                winner = "tie"
                winner_rich = "[dim]TIE[/dim]"

            delta_color = "green" if delta > 0.001 else ("red" if delta < -0.001 else "dim")

            summary_table.add_row(
                snap_name if n == GHZ_SIZES[0] else "",
                str(n),
                f"{new_fid:.4f}",
                str(best_qiskit["seed"]),
                f"{best_qiskit['estimated_fidelity']:.4f}",
                f"[{delta_color}]{delta:+.4f}[/{delta_color}]",
                winner_rich,
            )

            # Save detailed results
            snap_result["circuits"].append({
                "circuit": f"GHZ-{n}",
                "n_qubits": n,
                "qb_compiler": {
                    "physical_qubits": new_phys,
                    "region": f"{min(new_phys)}-{max(new_phys)}",
                    "post_routing_2q_gates": count_2q(tc_new),
                    "post_routing_depth": tc_new.depth(),
                    "estimated_fidelity": round(new_fid, 6),
                    "old_mapper_same_layout": old_phys == new_phys,
                    "old_mapper_fidelity": round(old_fid, 6),
                },
                "diagnostics": {
                    "raw_candidates": raw_cands,
                    "raw_regions": raw_regions,
                    "diversified_candidates": div_cands,
                    "diversified_regions": div_regions,
                    "rescore_all_same_2q": len(unique_2q_counts) <= 1,
                    "rescore_details": rescore_details,
                },
                "qiskit_all_seeds": [
                    {
                        "seed": r["seed"],
                        "physical_qubits": r["physical_qubits"],
                        "region": r["region"],
                        "estimated_fidelity": round(r["estimated_fidelity"], 6),
                        "post_routing_2q_gates": r["post_routing_2q_gates"],
                    }
                    for r in qiskit_results
                ],
                "qiskit_best": {
                    "seed": best_qiskit["seed"],
                    "physical_qubits": best_qiskit["physical_qubits"],
                    "region": best_qiskit["region"],
                    "estimated_fidelity": round(best_qiskit["estimated_fidelity"], 6),
                },
                "delta_vs_qiskit_best": round(delta, 6),
                "winner": winner,
            })

        all_results["snapshots"].append(snap_result)

    console.print()
    console.print(summary_table)

    win_rate = wins / total * 100 if total > 0 else 0
    all_results["summary"] = {
        "total_comparisons": total,
        "qb_compiler_wins": wins,
        "qiskit_wins": losses,
        "ties": ties,
        "win_rate_pct": round(win_rate, 1),
        "note": f"Qiskit tested with {len(QISKIT_SEEDS)} seeds per circuit, best seed used for comparison",
    }

    console.print(f"\n[bold]Summary: qb-compiler vs Qiskit BEST of {len(QISKIT_SEEDS)} seeds[/bold]")
    console.print(f"  qb-compiler wins: [green]{wins}[/green]/{total}")
    console.print(f"  Qiskit wins:      [red]{losses}[/red]/{total}")
    console.print(f"  Ties:             [dim]{ties}[/dim]/{total}")
    console.print(f"  Win rate:         {win_rate:.0f}%")

    # Write results to both local and D:
    for out_path in [
        Path("results/validation_vs_qiskit_best_seed.json"),
        Path("/mnt/d/validation_vs_qiskit_best_seed.json"),
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"[dim]Results written to {out_path}[/dim]")


if __name__ == "__main__":
    main()
