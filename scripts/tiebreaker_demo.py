"""Show GHZ-5 tiebreaker results for both chip regions on March 14 snapshot.

Demonstrates that routed_fidelity differentiates regions with equal 2Q gate counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.calibration_mapper import (
    CalibrationMapper,
    CalibrationMapperConfig,
)


def build_target_from_props(props: BackendProperties):
    """Build a Qiskit Target from BackendProperties."""
    from qiskit.circuit import Measure, Parameter
    from qiskit.circuit.library import CXGate, CZGate, HGate, RZGate, SXGate, XGate
    from qiskit.transpiler import InstructionProperties, Target

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


def build_ghz(n: int) -> QBCircuit:
    """Build a GHZ circuit."""
    c = QBCircuit(n_qubits=n, n_clbits=n, name=f"ghz_{n}")
    c.add_gate(QBGate("h", (0,)))
    for i in range(n - 1):
        c.add_gate(QBGate("cx", (i, i + 1)))
    for i in range(n):
        c.add_measurement(i, i)
    return c


def main():
    # Load March 14 calibration
    fixture_dir = Path(__file__).parents[1] / "tests" / "fixtures" / "calibration_snapshots"
    cal_file = fixture_dir / "ibm_fez_2026_03_14.json"
    if not cal_file.exists():
        # Fallback to any available
        cal_files = sorted(fixture_dir.glob("ibm_fez*.json"))
        if not cal_files:
            print("ERROR: No IBM Fez calibration fixtures found")
            sys.exit(1)
        cal_file = cal_files[-1]

    print(f"Using calibration: {cal_file.name}")
    props = BackendProperties.from_qubitboost_json(str(cal_file))
    target = build_target_from_props(props)

    # Build GHZ-5
    circuit = build_ghz(5)

    # Run CalibrationMapper with post-routing rescore
    config = CalibrationMapperConfig(
        top_k=30,
        max_per_region=5,
    )
    mapper = CalibrationMapper(props, config=config, qiskit_target=target)
    result = mapper.transform(circuit, {})

    # Show results
    print(f"\n{'='*90}")
    print(f"GHZ-5 Post-Routing Rescore — Tiebreaker Results")
    print(f"{'='*90}")

    details = mapper._last_rescore_details
    if not details:
        print("No rescore details available")
        return

    # Sort by (2Q gates, -routed_fidelity) to show ranking
    sorted_details = sorted(details, key=lambda d: (d["post_routing_2q"], -d.get("routed_fidelity", 0)))

    print(f"\n{'#':>3}  {'Region':>10}  {'Qubits':>30}  {'2Q Gates':>8}  {'Routed Fid':>10}  {'Cal Score':>10}  {'Depth':>5}")
    print(f"{'-'*3}  {'-'*10}  {'-'*30}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*5}")

    for rank, d in enumerate(sorted_details, 1):
        qubits_str = str(d["physical_qubits"])
        if len(qubits_str) > 30:
            qubits_str = qubits_str[:27] + "..."
        routed_fid = d.get("routed_fidelity", 0.0)
        marker = " <-- WINNER" if rank == 1 else ""
        print(
            f"{rank:>3}  {d['region']:>10}  {qubits_str:>30}  {d['post_routing_2q']:>8}  "
            f"{routed_fid:>10.6f}  {d['cal_score']:>10.4f}  {d['depth']:>5}{marker}"
        )

    # Summary
    winner = sorted_details[0]
    print(f"\n{'='*90}")
    print(f"WINNER: Region {winner['region']}, qubits={winner['physical_qubits']}")
    print(f"  2Q gates: {winner['post_routing_2q']}")
    print(f"  Routed fidelity: {winner.get('routed_fidelity', 0):.6f}")
    print(f"  Cal score (old tiebreaker): {winner['cal_score']:.4f}")

    # Show specific comparison between region ~16-23 and ~123-144 if both present
    print(f"\n{'='*90}")
    print("Region comparison (grouping by centroid):")
    low_region = [d for d in sorted_details if d["centroid"] < 50]
    high_region = [d for d in sorted_details if d["centroid"] > 100]

    if low_region:
        best_low = min(low_region, key=lambda d: (d["post_routing_2q"], -d.get("routed_fidelity", 0)))
        print(f"\n  Best in low region (~0-50):  qubits={best_low['physical_qubits']}")
        print(f"    2Q gates: {best_low['post_routing_2q']}, routed_fid: {best_low.get('routed_fidelity', 0):.6f}, cal_score: {best_low['cal_score']:.4f}")

    if high_region:
        best_high = min(high_region, key=lambda d: (d["post_routing_2q"], -d.get("routed_fidelity", 0)))
        print(f"\n  Best in high region (~100+): qubits={best_high['physical_qubits']}")
        print(f"    2Q gates: {best_high['post_routing_2q']}, routed_fid: {best_high.get('routed_fidelity', 0):.6f}, cal_score: {best_high['cal_score']:.4f}")

    if low_region and high_region:
        diff = best_high.get("routed_fidelity", 0) - best_low.get("routed_fidelity", 0)
        print(f"\n  Fidelity difference: {diff:+.6f} ({'high region better' if diff > 0 else 'low region better'})")

    # Show what the OLD tiebreaker would have picked
    old_sorted = sorted(details, key=lambda d: (d["post_routing_2q"], d["cal_score"]))
    old_winner = old_sorted[0]
    new_winner = sorted_details[0]
    if old_winner["physical_qubits"] != new_winner["physical_qubits"]:
        print(f"\n  OLD tiebreaker (cal_score) would have picked: {old_winner['physical_qubits']} (region {old_winner['region']})")
        print(f"  NEW tiebreaker (routed_fid) picks:             {new_winner['physical_qubits']} (region {new_winner['region']})")
    else:
        print(f"\n  Both tiebreakers pick the same layout: {new_winner['physical_qubits']}")


if __name__ == "__main__":
    main()
