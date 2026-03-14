#!/usr/bin/env python3
"""Hardware validation for qb-compiler on IBM Fez.

Tests the core hypothesis: does qb-compiler's CalibrationMapper find
better qubit layouts than Qiskit's built-in SabreLayout?

**Test design**: For each GHZ circuit, compile TWO ways using the
SAME Qiskit routing (optimization_level=3):

  A) Pure Qiskit — ``transpile(circuit, target, optimization_level=3)``
  B) qb-compiler layout + Qiskit routing —
     ``transpile(circuit, target, optimization_level=3, initial_layout=qb_layout)``

Both paths use Qiskit's SabreSwap for routing. The ONLY difference is
initial qubit placement. This isolates layout quality.

Three outcomes:

1. **B beats A** — CalibrationMapper finds better qubits. Ship it.
2. **B ties A** — Layout isn't helping. Scoring weights need tuning.
3. **B loses to A** — Layout is picking worse qubits. Fundamental issue.

Also runs a T1 asymmetry test comparing high vs low asymmetry qubits.

Usage::

    # Dry run — no QPU time
    python scripts/hardware_validation.py --dry-run

    # Real execution (uses latest calibration fixture)
    python scripts/hardware_validation.py

    # With specific calibration snapshot
    python scripts/hardware_validation.py --calibration tests/fixtures/calibration_snapshots/ibm_fez_2026_03_14.json

Requires:
    pip install qb-compiler qiskit-ibm-runtime
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
from rich.console import Console
from rich.table import Table

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.passes.mapping.calibration_mapper import (
    CalibrationMapper,
    CalibrationMapperConfig,
)

console = Console()

# ── Constants ─────────────────────────────────────────────────────────

BACKEND_NAME = "ibm_fez"
GHZ_SIZES = [3, 5, 8, 10]
SHOTS = 4096
T1_DELAY_NS = 5000  # 5 μs delay for T1 asymmetry test
T1_NUM_QUBITS = 5   # Test top-5 and bottom-5 asymmetry qubits

EST_SECONDS_PER_CIRCUIT = 1.5


# ── Circuit builders ──────────────────────────────────────────────────

def build_ghz_circuit(n_qubits: int) -> QuantumCircuit:
    """Build a GHZ circuit: H on q0, then CNOT chain."""
    qc = QuantumCircuit(n_qubits, n_qubits, name=f"ghz_{n_qubits}")
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def build_t1_asymmetry_circuit(qubit: int, delay_ns: int) -> QuantumCircuit:
    """Prepare |1⟩ on a single logical qubit, delay, then measure."""
    qc = QuantumCircuit(1, 1, name=f"t1_asym_q{qubit}")
    qc.x(0)
    if delay_ns > 0:
        qc.delay(delay_ns, 0, unit="ns")
    qc.measure(0, 0)
    return qc


# ── CalibrationMapper layout extraction ───────────────────────────────

def get_qb_layout(
    qc: QuantumCircuit,
    props: BackendProperties,
    qiskit_target: Any = None,
) -> dict[int, int]:
    """Run CalibrationMapper on a Qiskit circuit, return initial_layout.

    When ``qiskit_target`` is provided, the mapper performs post-routing
    rescoring: it trial-transpiles top-K candidate layouts and picks the
    one that needs the fewest SWAPs.
    """
    from qb_compiler.ir.circuit import QBCircuit
    from qb_compiler.ir.operations import QBGate

    n_q = qc.num_qubits
    circ = QBCircuit(n_qubits=n_q, n_clbits=n_q, name=qc.name)

    for instruction in qc.data:
        op = instruction.operation
        qubit_indices = [qc.find_bit(q).index for q in instruction.qubits]

        if op.name == "h":
            circ.add_gate(QBGate(name="h", qubits=(qubit_indices[0],)))
        elif op.name == "cx":
            circ.add_gate(
                QBGate(name="cx", qubits=(qubit_indices[0], qubit_indices[1]))
            )
        elif op.name == "measure":
            clbit_idx = qc.find_bit(instruction.clbits[0]).index
            circ.add_measurement(qubit_indices[0], clbit_idx)

    mapper = CalibrationMapper(
        props,
        config=CalibrationMapperConfig(
            max_candidates=500,
            vf2_call_limit=50_000,
            top_k=20,
        ),
        qiskit_target=qiskit_target,
    )
    context: dict[str, Any] = {}
    mapper.run(circ, context)
    return context["initial_layout"]


# ── T1 asymmetry qubit selection ──────────────────────────────────────

def select_asymmetry_qubits(
    props: BackendProperties, n: int
) -> tuple[list[int], list[int]]:
    """Select the n highest and n lowest T1-asymmetry qubits."""
    scored: list[tuple[float, int]] = []
    for qp in props.qubit_properties:
        ratio = qp.t1_asymmetry_ratio
        if ratio > 0:
            scored.append((ratio, qp.qubit_id))

    scored.sort(key=lambda x: x[0])
    low_asym = [q for _, q in scored[:n]]
    high_asym = [q for _, q in scored[-n:]]
    return high_asym, low_asym


# ── Fidelity calculation ─────────────────────────────────────────────

def ghz_fidelity(counts: dict[str, int], n_qubits: int) -> float:
    """Fidelity = P(000…0) + P(111…1)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    zeros = "0" * n_qubits
    ones = "1" * n_qubits
    return (counts.get(zeros, 0) + counts.get(ones, 0)) / total


def p1_from_counts(counts: dict[str, int]) -> float:
    """P(1) from a single-qubit measurement."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts.get("1", 0) / total


# ── Main validation logic ────────────────────────────────────────────

def run_validation(
    props: BackendProperties,
    dry_run: bool = False,
    output_path: str | None = None,
    channel: str | None = None,
    instance: str | None = None,
) -> dict[str, Any]:
    """Run the full hardware validation suite."""

    results: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "backend": BACKEND_NAME,
        "calibration_timestamp": props.timestamp,
        "shots": SHOTS,
        "dry_run": dry_run,
        "test_description": (
            "qb-compiler layout + Qiskit routing vs pure Qiskit. "
            "Both use optimization_level=3 and SabreSwap. "
            "Only difference is initial qubit placement."
        ),
        "ghz_results": [],
        "t1_asymmetry_results": {},
    }

    # ── Phase 1: Prepare all circuits ─────────────────────────────

    console.print("\n[bold]Phase 1: Preparing circuits[/bold]\n")

    # For each GHZ size, we submit TWO circuits:
    #   ghz_{n}_qiskit  — pure Qiskit layout + routing
    #   ghz_{n}_qb      — qb-compiler layout + Qiskit routing
    ghz_circuits: list[QuantumCircuit] = []
    ghz_labels: list[str] = []
    ghz_layouts: dict[int, dict[int, int]] = {}

    # On dry-run we don't have a Qiskit target, so post-routing rescore
    # is skipped.  On real runs, the target is passed in Phase 2 below.
    # For Phase 1 (preparation) we do a preliminary layout search without
    # post-routing rescoring.
    for n in GHZ_SIZES:
        qc = build_ghz_circuit(n)
        layout = get_qb_layout(qc, props)
        ghz_layouts[n] = layout

        ghz_circuits.append(qc.copy())
        ghz_labels.append(f"ghz_{n}_qiskit")

        ghz_circuits.append(qc.copy())
        ghz_labels.append(f"ghz_{n}_qb")

        phys = ", ".join(str(layout[i]) for i in range(n))
        console.print(f"  GHZ-{n}: qb-compiler layout = [{phys}]")

    # T1 asymmetry circuits
    high_asym, low_asym = select_asymmetry_qubits(props, T1_NUM_QUBITS)

    console.print(f"\n  T1 asymmetry — high qubits: {high_asym}")
    console.print(f"  T1 asymmetry — low qubits:  {low_asym}")
    for label, group in [("High", high_asym), ("Low", low_asym)]:
        parts = []
        for q in group:
            qp = props.qubit(q)
            ratio = qp.t1_asymmetry_ratio if qp else 0.0
            parts.append(f"q{q}={ratio:.1f}x")
        console.print(f"  {label} ratios: {', '.join(parts)}")

    t1_circuits: list[QuantumCircuit] = []
    t1_labels: list[str] = []
    for q in high_asym + low_asym:
        t1_circuits.append(build_t1_asymmetry_circuit(q, T1_DELAY_NS))
        t1_labels.append(f"t1_asym_q{q}")

    results["t1_asymmetry_results"]["high_asymmetry_qubits"] = high_asym
    results["t1_asymmetry_results"]["low_asymmetry_qubits"] = low_asym
    results["t1_asymmetry_results"]["delay_ns"] = T1_DELAY_NS

    all_circuits = ghz_circuits + t1_circuits
    all_labels = ghz_labels + t1_labels

    # ── QPU time estimate ─────────────────────────────────────────

    n_circuits = len(all_circuits)
    est_minutes = (n_circuits * EST_SECONDS_PER_CIRCUIT) / 60.0

    console.print(f"\n[bold]QPU Time Estimate[/bold]")
    console.print(f"  Circuits: {n_circuits}")
    console.print(f"  Shots/circuit: {SHOTS}")
    console.print(f"  Estimated: {est_minutes:.1f} min")
    console.print(
        f"  Fits in 10 min: "
        f"{'[green]YES[/green]' if est_minutes <= 10 else '[red]NO[/red]'}"
    )

    # ── Dry run ───────────────────────────────────────────────────

    if dry_run:
        console.print("\n[yellow bold]DRY RUN — no circuits submitted[/yellow bold]\n")

        table = Table(title="Circuits that would be submitted")
        table.add_column("#", style="dim")
        table.add_column("Label")
        table.add_column("Qubits", justify="right")
        table.add_column("Method")

        for i, (qc_item, label) in enumerate(zip(all_circuits, all_labels)):
            if label.endswith("_qb"):
                n = int(label.split("_")[1])
                layout = ghz_layouts[n]
                method = f"qb layout [{', '.join(str(layout[j]) for j in range(n))}] + Qiskit routing"
            elif label.endswith("_qiskit"):
                method = "Pure Qiskit (layout + routing)"
            else:
                q_id = label.split("_q")[1]
                method = f"Physical qubit {q_id}"
            table.add_row(str(i), label, str(qc_item.num_qubits), method)

        console.print(table)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"\nDry-run results saved to {output_path}")
        return results

    # ── Phase 2: Connect, transpile, execute ──────────────────────

    console.print("\n[bold]Phase 2: Connecting to IBM Quantum[/bold]\n")

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError:
        console.print("[red]pip install qiskit-ibm-runtime[/red]")
        sys.exit(1)

    service_kwargs: dict[str, Any] = {}
    if channel:
        service_kwargs["channel"] = channel
    if instance:
        service_kwargs["instance"] = instance
    if not service_kwargs:
        service = QiskitRuntimeService(name="default-ibm-quantum-platform")
    else:
        service = QiskitRuntimeService(**service_kwargs)

    backend = service.backend(BACKEND_NAME)
    target = backend.target
    console.print(f"  Connected to {backend.name}")

    # Re-run layout selection with post-routing rescoring now that we
    # have the real Qiskit target.
    console.print("\n  Re-running layout selection with post-routing rescoring...")
    for n in GHZ_SIZES:
        qc = build_ghz_circuit(n)
        layout = get_qb_layout(qc, props, qiskit_target=target)
        ghz_layouts[n] = layout
        phys = ", ".join(str(layout[i]) for i in range(n))
        console.print(f"    GHZ-{n}: post-routing layout = [{phys}]")

    # Transpile all circuits
    console.print("\n  Transpiling...")
    transpiled: list[QuantumCircuit] = []

    for qc_item, label in zip(all_circuits, all_labels):
        if label.startswith("ghz_"):
            n = int(label.split("_")[1])
            if label.endswith("_qb"):
                # qb-compiler layout + Qiskit routing
                layout = ghz_layouts[n]
                initial_layout = [layout[i] for i in range(n)]
                tc = transpile(
                    qc_item, target=target,
                    optimization_level=3,
                    initial_layout=initial_layout,
                )
            else:
                # Pure Qiskit
                tc = transpile(qc_item, target=target, optimization_level=3)
        else:
            # T1 asymmetry — map to specific physical qubit
            q_id = int(label.split("_q")[1])
            tc = transpile(
                qc_item, target=target,
                optimization_level=0,
                initial_layout=[q_id],
            )
        transpiled.append(tc)
        console.print(f"    {label}: depth={tc.depth()}")

    # ── Phase 3: Execute ──────────────────────────────────────────

    console.print("\n[bold]Phase 3: Executing on hardware[/bold]\n")

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = SHOTS

    console.print(f"  Submitting {len(transpiled)} circuits...")
    t_start = time.monotonic()
    job = sampler.run(transpiled)
    console.print(f"  Job ID: {job.job_id()}")
    console.print("  Waiting for results...")

    result = job.result()
    t_elapsed = time.monotonic() - t_start
    console.print(f"  Completed in {t_elapsed:.1f}s")

    # ── Phase 4: Analyse ──────────────────────────────────────────

    console.print("\n[bold]Phase 4: Results[/bold]\n")

    # Extract counts
    all_counts: list[dict[str, int]] = []
    for pub_result in result:
        creg_name = list(pub_result.data.keys())[0]
        bitarray = getattr(pub_result.data, creg_name)
        all_counts.append(bitarray.get_counts())

    # ── GHZ results ───────────────────────────────────────────────

    ghz_table = Table(
        title="GHZ Fidelity: qb-compiler Layout + Qiskit Routing vs Pure Qiskit"
    )
    ghz_table.add_column("Circuit", style="bold")
    ghz_table.add_column("Qubits", justify="right")
    ghz_table.add_column("Pure Qiskit", justify="right")
    ghz_table.add_column("qb Layout", justify="right")
    ghz_table.add_column("Delta", justify="right")
    ghz_table.add_column("qb Physical Qubits")

    idx = 0
    for n in GHZ_SIZES:
        counts_qiskit = all_counts[idx]
        counts_qb = all_counts[idx + 1]
        idx += 2

        fid_qiskit = ghz_fidelity(counts_qiskit, n)
        fid_qb = ghz_fidelity(counts_qb, n)
        delta = fid_qb - fid_qiskit

        results["ghz_results"].append({
            "n_qubits": n,
            "pure_qiskit_fidelity": fid_qiskit,
            "qb_layout_fidelity": fid_qb,
            "delta": delta,
            "qb_layout": ghz_layouts[n],
            "counts_qiskit": counts_qiskit,
            "counts_qb": counts_qb,
        })

        color = "green" if delta > 0.001 else ("red" if delta < -0.001 else "dim")
        delta_str = f"[{color}]{delta:+.4f} ({delta * 100:+.1f}%)[/{color}]"

        layout = ghz_layouts[n]
        phys_str = "→".join(str(layout[i]) for i in range(n))

        ghz_table.add_row(
            f"GHZ-{n}", str(n),
            f"{fid_qiskit:.4f}", f"{fid_qb:.4f}",
            delta_str, phys_str,
        )

    console.print(ghz_table)

    # Verdict
    deltas = [r["delta"] for r in results["ghz_results"]]
    avg_delta = sum(deltas) / len(deltas)
    if avg_delta > 0.005:
        verdict = "[bold green]LAYOUT WINS — CalibrationMapper finds better qubits[/bold green]"
    elif avg_delta > -0.005:
        verdict = "[bold yellow]TIE — Layout not significantly different from Qiskit[/bold yellow]"
    else:
        verdict = "[bold red]LAYOUT LOSES — CalibrationMapper picking worse qubits[/bold red]"
    console.print(f"\n  Average delta: {avg_delta:+.4f}")
    console.print(f"  Verdict: {verdict}")

    # ── T1 asymmetry results ──────────────────────────────────────

    t1_table = Table(title="T1 Asymmetry Test: P(1) After 5μs Delay")
    t1_table.add_column("Group", style="bold")
    t1_table.add_column("Qubit", justify="right")
    t1_table.add_column("Asym. Ratio", justify="right")
    t1_table.add_column("P(1)", justify="right")
    t1_table.add_column("Loss", justify="right")

    t1_data_high: list[dict[str, Any]] = []
    t1_data_low: list[dict[str, Any]] = []

    for i, q in enumerate(high_asym):
        counts = all_counts[idx + i]
        p1 = p1_from_counts(counts)
        qp = props.qubit(q)
        ratio = qp.t1_asymmetry_ratio if qp else 0.0
        t1_data_high.append({
            "qubit": q, "asymmetry_ratio": ratio, "p1": p1,
            "fidelity_loss": 1.0 - p1, "counts": counts,
        })
        t1_table.add_row(
            "[red]HIGH[/red]", str(q), f"{ratio:.1f}x",
            f"{p1:.4f}", f"{1.0 - p1:.4f}",
        )
    idx += len(high_asym)

    for i, q in enumerate(low_asym):
        counts = all_counts[idx + i]
        p1 = p1_from_counts(counts)
        qp = props.qubit(q)
        ratio = qp.t1_asymmetry_ratio if qp else 0.0
        t1_data_low.append({
            "qubit": q, "asymmetry_ratio": ratio, "p1": p1,
            "fidelity_loss": 1.0 - p1, "counts": counts,
        })
        t1_table.add_row(
            "[green]LOW[/green]", str(q), f"{ratio:.1f}x",
            f"{p1:.4f}", f"{1.0 - p1:.4f}",
        )

    console.print(t1_table)

    results["t1_asymmetry_results"]["high"] = t1_data_high
    results["t1_asymmetry_results"]["low"] = t1_data_low

    avg_p1_high = sum(d["p1"] for d in t1_data_high) / len(t1_data_high)
    avg_p1_low = sum(d["p1"] for d in t1_data_low) / len(t1_data_low)
    results["t1_asymmetry_results"]["avg_p1_high"] = avg_p1_high
    results["t1_asymmetry_results"]["avg_p1_low"] = avg_p1_low
    results["total_execution_time_s"] = t_elapsed

    console.print(f"\n  Avg P(1) high-asymmetry: {avg_p1_high:.4f}")
    console.print(f"  Avg P(1) low-asymmetry:  {avg_p1_low:.4f}")
    diff = avg_p1_low - avg_p1_high
    label = "[green]low-asym better[/green]" if diff > 0 else "[red]unexpected[/red]"
    console.print(f"  Difference: {diff:+.4f} ({label})")

    # ── Save ──────────────────────────────────────────────────────

    if output_path is None:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"results/validation_{ts}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[bold green]Results saved to {output_path}[/bold green]")

    return results


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hardware validation for qb-compiler on IBM Fez",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without using QPU time")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to calibration JSON snapshot")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--channel", type=str, default=None)
    parser.add_argument("--instance", type=str, default=None)
    args = parser.parse_args()

    console.print("[bold]qb-compiler Hardware Validation[/bold]")
    console.print(f"Backend: {BACKEND_NAME} | Shots: {SHOTS}")
    console.print(f"Test: qb-compiler layout + Qiskit routing vs pure Qiskit")

    # Load calibration
    if args.calibration:
        cal_path = Path(args.calibration)
    else:
        fixture_dir = Path("tests/fixtures/calibration_snapshots")
        candidates = sorted(fixture_dir.glob("ibm_fez_*.json"))
        if not candidates:
            console.print("[red]No calibration fixtures. Use --calibration.[/red]")
            sys.exit(1)
        cal_path = candidates[-1]

    console.print(f"Calibration: {cal_path}")
    props = BackendProperties.from_qubitboost_json(cal_path)
    console.print(f"  {props.n_qubits} qubits, timestamp={props.timestamp}")

    run_validation(
        props=props,
        dry_run=args.dry_run,
        output_path=args.output,
        channel=args.channel,
        instance=args.instance,
    )


if __name__ == "__main__":
    main()
