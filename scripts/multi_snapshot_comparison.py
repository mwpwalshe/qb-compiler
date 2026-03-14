#!/usr/bin/env python3
"""Compare qb-compiler vs Qiskit across multiple calibration snapshots.

Shows whether our layout advantage is stable across different days'
calibration data, or if it only works on one snapshot.

Usage::

    python scripts/multi_snapshot_comparison.py
"""

from __future__ import annotations

import sys
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


def build_target(props: BackendProperties) -> Target:
    """Build Qiskit Target from calibration, auto-detecting gate type."""
    n_qubits = props.n_qubits
    target = Target(num_qubits=n_qubits)

    # Detect 2Q gate type from calibration data
    gate_types = {gp.gate_type for gp in props.gate_properties if len(gp.qubits) == 2}
    gate_2q_name = "cz" if "cz" in gate_types else "cx"
    gate_2q_cls = CZGate if gate_2q_name == "cz" else CXGate

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


def get_qb_layout(circ_ir: QBCircuit, props: BackendProperties, target: Target) -> dict[int, int]:
    mapper = CalibrationMapper(
        props,
        config=CalibrationMapperConfig(
            max_candidates=500, vf2_call_limit=50_000, top_k=20, max_per_region=3,
        ),
        qiskit_target=target,
    )
    ctx: dict[str, Any] = {}
    mapper.run(circ_ir, ctx)
    return ctx["initial_layout"]


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


def main() -> None:
    fixture_dir = Path("tests/fixtures/calibration_snapshots")
    snapshots = sorted(fixture_dir.glob("ibm_fez_*.json"))

    # Filter to full-chip snapshots (> 50 qubit_properties)
    full_snapshots: list[tuple[str, BackendProperties]] = []
    for path in snapshots:
        try:
            props = BackendProperties.from_qubitboost_json(path)
            if len(props.qubit_properties) >= 50:
                full_snapshots.append((path.stem, props))
        except Exception as e:
            console.print(f"[dim]Skipping {path.name}: {e}[/dim]")

    if not full_snapshots:
        console.print("[red]No full-chip calibration snapshots found[/red]")
        sys.exit(1)

    console.print(f"[bold]Multi-Snapshot Comparison: qb-compiler vs Qiskit[/bold]")
    console.print(f"Snapshots: {len(full_snapshots)}\n")

    for snap_name, props in full_snapshots:
        console.print(f"[dim]{snap_name}: {len(props.qubit_properties)} qubits, "
                       f"{len(props.gate_properties)} gates[/dim]")

    console.print()

    # Results table
    table = Table(title="qb-compiler (windowed VF2 + post-routing) vs Qiskit opt=3")
    table.add_column("Snapshot", style="bold")
    table.add_column("GHZ", justify="right")
    table.add_column("qb Qubits")
    table.add_column("Qiskit Qubits")
    table.add_column("qb Fidelity", justify="right")
    table.add_column("Qiskit Fidelity", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Winner")

    wins = 0
    ties = 0
    losses = 0
    total = 0

    for snap_name, props in full_snapshots:
        target = build_target(props)

        for n in GHZ_SIZES:
            qc = build_ghz_qiskit(n)
            circ_ir = build_ghz_ir(n)

            # qb-compiler layout
            qb_layout = get_qb_layout(circ_ir, props, target)
            qb_phys = [qb_layout[i] for i in range(n)]
            qb_fid = est_fidelity(qb_layout, props, n)

            # Qiskit layout
            tc_qiskit = transpile(qc, target=target, optimization_level=3, seed_transpiler=42)
            ql = tc_qiskit.layout
            if ql and ql.initial_layout:
                qiskit_phys = [ql.initial_layout[qc.qubits[i]] for i in range(n)]
            else:
                qiskit_phys = list(range(n))
            qiskit_layout = {i: qiskit_phys[i] for i in range(n)}
            qiskit_fid = est_fidelity(qiskit_layout, props, n)

            delta = qb_fid - qiskit_fid
            total += 1

            if delta > 0.001:
                winner = "[green]qb-compiler[/green]"
                wins += 1
            elif delta < -0.001:
                winner = "[red]Qiskit[/red]"
                losses += 1
            else:
                winner = "[dim]TIE[/dim]"
                ties += 1

            delta_color = "green" if delta > 0.001 else ("red" if delta < -0.001 else "dim")

            table.add_row(
                snap_name if n == GHZ_SIZES[0] else "",
                str(n),
                " ".join(str(q) for q in qb_phys[:5]) + ("..." if n > 5 else ""),
                " ".join(str(q) for q in qiskit_phys[:5]) + ("..." if n > 5 else ""),
                f"{qb_fid:.4f}",
                f"{qiskit_fid:.4f}",
                f"[{delta_color}]{delta:+.4f}[/{delta_color}]",
                winner,
            )

        table.add_row("", "", "", "", "", "", "", "")

    console.print(table)

    console.print(f"\n[bold]Summary across {len(full_snapshots)} snapshots, {len(GHZ_SIZES)} circuits each:[/bold]")
    console.print(f"  qb-compiler wins: [green]{wins}[/green]/{total}")
    console.print(f"  Qiskit wins:      [red]{losses}[/red]/{total}")
    console.print(f"  Ties:             [dim]{ties}[/dim]/{total}")

    win_rate = wins / total * 100 if total > 0 else 0
    if win_rate >= 75:
        console.print(f"\n  [bold green]WIN RATE: {win_rate:.0f}% — READY FOR HARDWARE[/bold green]")
    elif win_rate >= 50:
        console.print(f"\n  [bold yellow]WIN RATE: {win_rate:.0f}% — Promising but needs tuning[/bold yellow]")
    else:
        console.print(f"\n  [bold red]WIN RATE: {win_rate:.0f}% — Still losing[/bold red]")


if __name__ == "__main__":
    main()
