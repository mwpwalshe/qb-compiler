"""Example 05: Qiskit integration via qb_transpile().

Shows how to use qb-compiler's calibration-aware transpilation with
standard Qiskit QuantumCircuits. The qb_transpile() function injects
a calibration-aware layout pass into Qiskit's transpiler pipeline.
"""

from pathlib import Path

CAL_PATH = Path(__file__).resolve().parent.parent / (
    "tests/fixtures/calibration_snapshots/ibm_fez_2026_02_15.json"
)


def main() -> None:
    try:
        from qiskit.circuit import QuantumCircuit
    except ImportError:
        print("Qiskit is not installed. Install with: pip install qiskit")
        print("Skipping this example.")
        return

    from qb_compiler.qiskit_plugin.transpiler_plugin import qb_transpile

    # Build a 3-qubit GHZ circuit in Qiskit
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    print("=== Original Qiskit Circuit ===")
    print(qc.draw(output="text"))
    print(f"  Depth: {qc.depth()}")
    print(f"  Gate count: {qc.count_ops()}")

    # Transpile with calibration awareness
    compiled = qb_transpile(
        qc,
        backend="ibm_fez",
        calibration_path=str(CAL_PATH),
        optimization_level=2,
    )

    print("\n=== Compiled Circuit (IBM Fez basis) ===")
    print(compiled.draw(output="text", fold=120))
    print(f"  Depth:      {compiled.depth()}")
    print(f"  Gate count: {compiled.count_ops()}")
    print(f"  Width:      {compiled.num_qubits} qubits")

    if hasattr(compiled, "layout") and compiled.layout:
        print(f"\n=== Layout ===")
        layout = compiled.layout
        if hasattr(layout, "initial_layout") and layout.initial_layout:
            print(f"  Initial layout: {layout.initial_layout}")


if __name__ == "__main__":
    main()
