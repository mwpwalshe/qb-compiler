"""Example 03: Noise-aware SWAP routing.

Demonstrates how NoiseAwareRouter inserts SWAPs along the lowest-error
paths when two-qubit gates target non-adjacent physical qubits. Compares
the routed circuit against the original to show inserted SWAP overhead.
"""

from pathlib import Path

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.noise_aware_router import NoiseAwareRouter

CAL_PATH = Path(__file__).resolve().parent.parent / (
    "tests/fixtures/calibration_snapshots/ibm_fez_2026_02_15.json"
)


def main() -> None:
    props = BackendProperties.from_qubitboost_json(CAL_PATH)

    # Extract coupling map and gate errors from calibration
    coupling_map = props.coupling_map
    gate_errors: dict[tuple[int, int], float] = {}
    for gp in props.gate_properties:
        if len(gp.qubits) == 2 and gp.error_rate is not None:
            gate_errors[gp.qubits] = gp.error_rate

    print(f"Coupling map: {len(coupling_map)} directed edges")
    print(f"Gate errors available for {len(gate_errors)} edges")

    # Build a circuit on physical qubits 0 and 5 (NOT adjacent on heavy-hex)
    n_qubits = max(max(e) for e in coupling_map) + 1
    circ = QBCircuit(n_qubits=n_qubits, name="non-adjacent-cx")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    circ.add_gate(QBGate(name="cx", qubits=(0, 5)))  # non-adjacent

    print(f"\n=== Before Routing ===")
    print(f"  Circuit: CX between physical q0 and q5")
    print(f"  Depth: {circ.depth}, Gates: {circ.gate_count}")

    # Route with noise awareness
    router = NoiseAwareRouter(
        coupling_map=coupling_map,
        gate_errors=gate_errors,
    )
    context: dict = {}
    result = router.run(circ, context)
    routed = result.circuit

    print(f"\n=== After Routing ===")
    print(f"  Depth: {routed.depth}, Gates: {routed.gate_count}")
    print(f"  SWAPs inserted: {context['swaps_inserted']}")
    print(f"  Routing fidelity cost: {context['routing_fidelity_cost']:.6f}")

    print(f"\n=== Routed Operations ===")
    for op in routed.gates:
        print(f"  {op}")


if __name__ == "__main__":
    main()
