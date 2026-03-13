"""Example 01: Basic compilation with QBCompiler.

Demonstrates building a Bell state circuit using QBCircuit's fluent API,
compiling it with QBCompiler targeting IBM Fez, and inspecting the
before/after metrics (depth, gate count, estimated fidelity).
"""

from pathlib import Path

from qb_compiler import QBCircuit, QBCompiler


def main() -> None:
    # Build a Bell state: H on qubit 0, then CX(0, 1), measure both
    circuit = QBCircuit(2).h(0).cx(0, 1).measure_all()

    print("=== Before Compilation ===")
    print(f"  Circuit:    {circuit}")
    print(f"  Depth:      {circuit.depth}")
    print(f"  Gate count: {circuit.gate_count}")
    print(f"  Gates:      {[str(op) for op in circuit.ops]}")

    # Compile targeting IBM Fez with fidelity-optimal strategy
    compiler = QBCompiler.from_backend("ibm_fez")
    result = compiler.compile(circuit)

    print("\n=== After Compilation ===")
    print(f"  Circuit:          {result.compiled_circuit}")
    print(f"  Compiled depth:   {result.compiled_depth}")
    print(f"  Original depth:   {result.original_depth}")
    print(f"  Depth reduction:  {result.depth_reduction_pct:.1f}%")
    print(f"  Est. fidelity:    {result.estimated_fidelity:.4f}")
    print(f"  Compile time:     {result.compilation_time_ms:.2f} ms")

    print("\n=== Pass Log ===")
    for p in result.pass_log:
        print(f"  {p.pass_name:25s}  depth {p.depth_before}->{p.depth_after}  "
              f"gates {p.gate_count_before}->{p.gate_count_after}  "
              f"({p.elapsed_ms:.2f} ms)")

    # Also show cost estimation
    cost = compiler.estimate_cost(result.compiled_circuit, shots=1024)
    print(f"\n=== Cost Estimate (1024 shots) ===")
    print(f"  Backend:        {cost.backend}")
    print(f"  Per-shot:       ${cost.cost_per_shot_usd:.6f}")
    print(f"  Total:          ${cost.total_usd:.4f}")


if __name__ == "__main__":
    main()
