"""Example 02: Calibration-aware qubit mapping.

Loads a real IBM Fez calibration snapshot and uses CalibrationMapper to
find the optimal physical qubit placement for a small circuit. Shows how
calibration data (gate errors, T1/T2, readout) drives layout selection.
"""

from pathlib import Path

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.calibration_mapper import (
    CalibrationMapper,
    CalibrationMapperConfig,
)

CAL_PATH = Path(__file__).resolve().parent.parent / (
    # Download sample calibration: https://github.com/mwpwalshe/qb-compiler/tree/main/tests/fixtures/calibration_snapshots
    "tests/fixtures/calibration_snapshots/ibm_fez_2026_02_15.json"  # Or use your own: qb calibrate fetch ibm_fez
)


def main() -> None:
    # Load calibration data
    props = BackendProperties.from_qubitboost_json(CAL_PATH)
    print(f"Backend: {props.backend} ({props.n_qubits} qubits)")
    print(f"Calibration timestamp: {props.timestamp}")

    # Build a 3-qubit GHZ circuit: H(0), CX(0,1), CX(1,2)
    circ = QBCircuit(n_qubits=3, name="GHZ-3")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
    circ.add_gate(QBGate(name="cx", qubits=(1, 2)))

    print(f"\nLogical circuit: {circ}")
    print(f"  Depth: {circ.depth}, Gates: {circ.gate_count}")

    # Run calibration-aware mapping
    mapper = CalibrationMapper(
        props,
        config=CalibrationMapperConfig(max_candidates=500),
    )
    context: dict = {}
    result = mapper.run(circ, context)

    layout = context["initial_layout"]
    score = context["calibration_score"]

    print(f"\n=== Calibration-Aware Layout ===")
    print(f"  Score (lower is better): {score:.6f}")
    for logical, physical in sorted(layout.items()):
        qp = props.qubit(physical)
        t1 = f"{qp.t1_us:.1f}" if qp and qp.t1_us else "N/A"
        ro = f"{qp.readout_error:.4f}" if qp and qp.readout_error else "N/A"
        print(f"  Logical q{logical} -> Physical q{physical}"
              f"  (T1={t1} us, readout_err={ro})")

    mapped = result.circuit
    print(f"\nMapped circuit: {mapped}")
    print(f"  Physical qubits used: {sorted(mapped.qubits_used())}")


if __name__ == "__main__":
    main()
