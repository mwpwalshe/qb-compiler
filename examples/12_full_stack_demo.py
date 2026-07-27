"""QubitBoost Full Stack Demo: Compiler + Gate Integration.

Shows the complete pipeline:
1. Circuit viability check
2. Circuit type detection and gate recommendation
3. Compilation with DD
4. Gate eligibility report
5. Cost estimation with explicit assumptions

This demo runs without the QubitBoost SDK installed.
Gate execution requires: pip install qubitboost-sdk
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from qb_compiler import QBCompiler, check_viability
from qb_compiler.integrations.qubitboost import (
    detect_circuit_type,
    is_sdk_available,
    recommend_gates,
)


def build_qaoa_maxcut(n_qubits: int, p: int = 2) -> QuantumCircuit:
    """Build a QAOA MaxCut circuit."""
    qc = QuantumCircuit(n_qubits, n_qubits, name=f"QAOA-MaxCut-{n_qubits}q-p{p}")

    # Cost layers + mixer layers for p rounds
    for layer in range(p):
        gamma = 0.5 + layer * 0.1
        beta = 0.3 + layer * 0.05

        # Cost layer: ZZ interactions on edges
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * gamma, i + 1)
            qc.cx(i, i + 1)

        # Mixer layer: X rotations
        for i in range(n_qubits):
            qc.rx(2 * beta, i)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> None:
    circuit = build_qaoa_maxcut(n_qubits=8, p=2)
    backend = "ibm_fez"

    # Step 1: Viability check
    print("=" * 60)
    print("Step 1: Viability Check")
    print("=" * 60)
    result = check_viability(circuit, backend=backend, n_seeds=2)
    print(result)
    print()

    # Step 2: Circuit type detection
    print("=" * 60)
    print("Step 2: Circuit Type Detection")
    print("=" * 60)
    circuit_type, confidence = detect_circuit_type(circuit)
    print(f"Circuit type: {circuit_type.upper()} ({confidence.value} confidence)")
    print()

    # Step 3: Gate eligibility
    print("=" * 60)
    print("Step 3: QubitBoost Gate Eligibility")
    print("=" * 60)
    recs = recommend_gates(circuit_type, confidence)
    for r in recs:
        print(f"  * {r.gate:14s} {r.status}: {r.headline}")
        if r.validated_claim:
            print(
                f"    {'':14s} Hardware-validated: "
                f"{r.validated_claim} {r.qualifier}"
            )
    print()

    # Step 4: Cost estimation with explicit assumptions
    print("=" * 60)
    print("Step 4: Cost Estimation (assumptions shown)")
    print("=" * 60)
    shots = 4096
    cost_per_shot = 0.00016  # IBM Fez standard pricing
    standard_cost = shots * cost_per_shot

    print(f"  Shots: {shots:,}")
    print(f"  Backend: {backend}")
    print(f"  Price/shot: ${cost_per_shot} (IBM standard)")
    print(f"  Standard cost: ${standard_cost:.2f}")
    print()

    if circuit_type == "qaoa" and confidence.value in ("high", "medium"):
        print("  Adaptive shot-reduction gates for QAOA are available in the")
        print("  QubitBoost SDK. The achievable reduction depends on circuit")
        print("  structure, backend calibration and problem instance, so this")
        print("  demo projects no figure: the SDK reports the shots actually")
        print("  used against the baseline above once the job has run.")
    print()

    # Step 5: SDK status
    print("=" * 60)
    print("Step 5: Execution")
    print("=" * 60)
    if is_sdk_available():
        print("QubitBoost SDK installed. Gates are ready for execution.")
    else:
        print("To execute with QubitBoost gates: pip install qubitboost-sdk")
        print("Learn more: https://qubitboost.io")
    print()


if __name__ == "__main__":
    main()
