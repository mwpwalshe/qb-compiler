"""Example 09: Using qb-compiler with existing Qiskit circuits.

Shows how to take a Qiskit QuantumCircuit, compile with qb-compiler,
and get back a Qiskit QuantumCircuit ready for execution.
"""

from qb_compiler import QBCompiler


def main() -> None:
    # --- Option A: Use qb_transpile (drop-in replacement) ---
    print("=== Option A: qb_transpile (simplest) ===")
    from qb_compiler.qiskit_plugin import qb_transpile
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.measure_all()

    # One-liner compilation
    compiled_qc = qb_transpile(qc, backend="ibm_fez")
    print(f"Input:    {qc.num_qubits} qubits, depth {qc.depth()}")
    print(f"Compiled: {compiled_qc.num_qubits} qubits, depth {compiled_qc.depth()}")
    print()

    # --- Option B: Full QBCompiler API ---
    print("=== Option B: QBCompiler API (full control) ===")
    from qb_compiler import QBCircuit

    compiler = QBCompiler.from_backend("ibm_fez")

    # Build circuit with QBCircuit fluent API
    circ = QBCircuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3).measure_all()
    result = compiler.compile(circ)

    print(f"Compiled depth:      {result.compiled_depth}")
    print(f"Estimated fidelity:  {result.estimated_fidelity:.4f}")
    print(f"Depth reduction:     {result.depth_reduction_pct:.1f}%")

    # Estimate cost
    cost = compiler.estimate_cost(result.compiled_circuit, shots=4096)
    print(f"Estimated cost:      ${cost.total_usd:.4f}")
    print()

    # --- Inspect the pass log ---
    print("=== Pass log ===")
    for entry in result.pass_log:
        print(f"  {entry.pass_name:25s}  depth {entry.depth_before}->{entry.depth_after}  "
              f"gates {entry.gate_count_before}->{entry.gate_count_after}  "
              f"({entry.elapsed_ms:.2f} ms)")


if __name__ == "__main__":
    main()
