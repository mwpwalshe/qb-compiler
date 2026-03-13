"""Example 08: Compare compilation results across multiple backends.

Compiles the same circuit for all supported backends and compares
depth, gate count, estimated fidelity, and cost.
"""

import math

from qb_compiler import QBCompiler, QBCircuit


def main() -> None:
    # Build a QAOA-style circuit (4 qubits, alternating mixer + cost layers)
    circ = QBCircuit(4)
    gamma = math.pi / 4
    beta = math.pi / 6

    # Cost layer: ZZ interactions on edges (0-1, 1-2, 2-3)
    for q0, q1 in [(0, 1), (1, 2), (2, 3)]:
        circ.cx(q0, q1)
        circ.rz(q1, 2 * gamma)
        circ.cx(q0, q1)

    # Mixer layer: RX on all qubits
    for q in range(4):
        circ.h(q)
        circ.rz(q, 2 * beta)
        circ.h(q)

    circ.measure_all()

    print(f"Circuit: QAOA 4-qubit")
    print(f"  Qubits: {circ.n_qubits}")
    print(f"  Depth:  {circ.depth}")
    print(f"  Gates:  {circ.gate_count}")
    print(f"  2Q:     {circ.two_qubit_count}")
    print()

    # Compile for each backend
    backends = ["ibm_fez", "ibm_torino", "rigetti_ankaa", "ionq_aria", "iqm_garnet"]
    shots = 4096

    print(
        f"{'Backend':20s}  {'Depth':>6s}  {'Gates':>6s}  "
        f"{'Fidelity':>10s}  {'Cost/4096':>10s}"
    )
    print("-" * 60)

    for backend in backends:
        compiler = QBCompiler.from_backend(backend)
        result = compiler.compile(circ)
        cost = compiler.estimate_cost(result.compiled_circuit, shots=shots)

        print(
            f"{backend:20s}  "
            f"{result.compiled_depth:6d}  "
            f"{result.compiled_circuit.gate_count:6d}  "
            f"{result.estimated_fidelity:10.4f}  "
            f"${cost.total_usd:>8.4f}"
        )

    # Find the best fidelity/cost ratio
    print("\n--- Best Value Analysis ---")
    best_ratio = None
    best_backend = None

    for backend in backends:
        compiler = QBCompiler.from_backend(backend)
        result = compiler.compile(circ)
        cost = compiler.estimate_cost(result.compiled_circuit, shots=shots)
        if cost.total_usd > 0:
            ratio = result.estimated_fidelity / cost.total_usd
            if best_ratio is None or ratio > best_ratio:
                best_ratio = ratio
                best_backend = backend

    if best_backend:
        print(f"Best fidelity per dollar: {best_backend}")


if __name__ == "__main__":
    main()
