#!/usr/bin/env python3
"""Save local validation + multi-snapshot results to results/ as JSON."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CZGate, CXGate, HGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.calibration_mapper import (
    CalibrationMapper,
    CalibrationMapperConfig,
)

GHZ_SIZES = [3, 5, 8, 10]


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

    full_snapshots: list[tuple[str, BackendProperties]] = []
    for path in snapshots:
        try:
            props = BackendProperties.from_qubitboost_json(path)
            if len(props.qubit_properties) >= 50:
                full_snapshots.append((path.stem, props))
        except Exception:
            pass

    all_results: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "description": (
            "Local validation comparing OLD mapper (single-region VF2, pre-routing scoring) "
            "vs NEW mapper (windowed VF2 + post-routing rescoring) vs Qiskit opt_level=3. "
            "No QPU time used — estimated fidelity from calibration data."
        ),
        "method": {
            "old_mapper": "Single VF2 run, top_k=1, pre-routing calibration scoring only",
            "new_mapper": "Windowed VF2 (multi-region), top_k=20, diversity filter, post-routing rescoring",
            "qiskit": "transpile(optimization_level=3, seed_transpiler=42)",
        },
        "snapshots": [],
    }

    wins = 0
    losses = 0
    ties = 0
    total = 0

    for snap_name, props in full_snapshots:
        target = build_target(props)
        snap_result: dict[str, Any] = {
            "snapshot": snap_name,
            "timestamp": props.timestamp,
            "n_qubits": props.n_qubits,
            "n_gates": len(props.gate_properties),
            "circuits": [],
        }

        for n in GHZ_SIZES:
            qc = build_ghz_qiskit(n)
            circ_ir = build_ghz_ir(n)

            # OLD mapper
            old_layout = get_qb_layout_old(circ_ir, props)
            old_phys = [old_layout[i] for i in range(n)]
            tc_old = transpile(qc, target=target, optimization_level=3,
                               initial_layout=old_phys, seed_transpiler=42)
            old_fid = est_fidelity(old_layout, props, n)

            # NEW mapper
            new_layout = get_qb_layout(circ_ir, props, target)
            new_phys = [new_layout[i] for i in range(n)]
            tc_new = transpile(qc, target=target, optimization_level=3,
                               initial_layout=new_phys, seed_transpiler=42)
            new_fid = est_fidelity(new_layout, props, n)

            # Qiskit
            tc_qiskit = transpile(qc, target=target, optimization_level=3, seed_transpiler=42)
            ql = tc_qiskit.layout
            if ql and ql.initial_layout:
                qiskit_phys = [ql.initial_layout[qc.qubits[i]] for i in range(n)]
            else:
                qiskit_phys = list(range(n))
            qiskit_layout = {i: qiskit_phys[i] for i in range(n)}
            qiskit_fid = est_fidelity(qiskit_layout, props, n)

            delta_new_vs_qiskit = new_fid - qiskit_fid
            delta_new_vs_old = new_fid - old_fid
            total += 1

            if delta_new_vs_qiskit > 0.001:
                wins += 1
                winner = "qb-compiler"
            elif delta_new_vs_qiskit < -0.001:
                losses += 1
                winner = "qiskit"
            else:
                ties += 1
                winner = "tie"

            snap_result["circuits"].append({
                "circuit": f"GHZ-{n}",
                "n_qubits": n,
                "old_mapper": {
                    "physical_qubits": old_phys,
                    "region": f"{min(old_phys)}-{max(old_phys)}",
                    "post_routing_2q_gates": count_2q(tc_old),
                    "post_routing_depth": tc_old.depth(),
                    "estimated_fidelity": round(old_fid, 6),
                },
                "new_mapper": {
                    "physical_qubits": new_phys,
                    "region": f"{min(new_phys)}-{max(new_phys)}",
                    "post_routing_2q_gates": count_2q(tc_new),
                    "post_routing_depth": tc_new.depth(),
                    "estimated_fidelity": round(new_fid, 6),
                },
                "qiskit_opt3": {
                    "physical_qubits": qiskit_phys,
                    "region": f"{min(qiskit_phys)}-{max(qiskit_phys)}",
                    "post_routing_2q_gates": count_2q(tc_qiskit),
                    "post_routing_depth": tc_qiskit.depth(),
                    "estimated_fidelity": round(qiskit_fid, 6),
                },
                "delta_new_vs_qiskit": round(delta_new_vs_qiskit, 6),
                "delta_new_vs_old": round(delta_new_vs_old, 6),
                "winner": winner,
            })

        all_results["snapshots"].append(snap_result)

    all_results["summary"] = {
        "total_comparisons": total,
        "qb_compiler_wins": wins,
        "qiskit_wins": losses,
        "ties": ties,
        "win_rate_pct": round(wins / total * 100, 1) if total > 0 else 0,
    }

    # Write results
    out_path = Path("results/validation_post_routing_local.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results written to {out_path}")
    print(f"Win rate: {wins}/{total} ({wins/total*100:.0f}%)")


if __name__ == "__main__":
    main()
