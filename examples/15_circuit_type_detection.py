#!/usr/bin/env python3
"""Example 15: Circuit type detection and gate eligibility.

Automatically detect whether your circuit is QAOA, VQE, QEC,
or general — and see which QubitBoost gates are applicable.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from qb_compiler.integrations.qubitboost import (
    detect_circuit_type,
    recommend_gates,
)


def show_detection(name: str, qc: QuantumCircuit) -> None:
    """Detect circuit type and show gate recommendations."""
    circuit_type, confidence = detect_circuit_type(qc)
    recs = recommend_gates(circuit_type, confidence)

    print(f"=== {name} ===")
    print(f"Type:       {circuit_type.upper()}")
    print(f"Confidence: {confidence.value}")
    print(f"Gates:      {qc.size()}  Depth: {qc.depth()}")
    print()

    if recs:
        print("QubitBoost Gate Eligibility:")
        for r in recs:
            print(f"  {r.gate:14s}  {r.status} — {r.headline}")
            if r.validated_claim:
                print(f"  {'':14s}  Hardware-validated: {r.validated_claim} {r.qualifier}")
    print()


# --- QAOA circuit (detected by name) ---
qaoa = QuantumCircuit(6, 6, name="QAOA-MaxCut")
for _ in range(2):
    for i in range(5):
        qaoa.cx(i, i + 1)
        qaoa.rz(0.5, i + 1)
        qaoa.cx(i, i + 1)
    for i in range(6):
        qaoa.rx(0.3, i)
qaoa.measure(range(6), range(6))
show_detection("QAOA MaxCut (name-based, HIGH confidence)", qaoa)

# --- VQE circuit (detected by name) ---
vqe = QuantumCircuit(4, 4, name="VQE-H2")
for i in range(4):
    vqe.ry(0.5, i)
    vqe.rz(0.3, i)
vqe.cx(0, 1)
vqe.cx(2, 3)
vqe.measure(range(4), range(4))
show_detection("VQE H2 (name-based, HIGH confidence)", vqe)

# --- QEC circuit (detected by name) ---
qec = QuantumCircuit(5, 5, name="surface-code-d3")
qec.h(0)
for i in range(4):
    qec.cx(0, i + 1)
qec.measure(range(5), range(5))
show_detection("Surface Code d=3 (name-based, HIGH confidence)", qec)

# --- General circuit (Bell state) ---
bell = QuantumCircuit(2, 2, name="Bell")
bell.h(0)
bell.cx(0, 1)
bell.measure([0, 1], [0, 1])
show_detection("Bell State (general, LOW confidence)", bell)

# --- Structural QAOA detection (no QAOA in name) ---
struct_qaoa = QuantumCircuit(4, 4, name="test-circuit")
for _ in range(3):
    for i in range(3):
        struct_qaoa.cx(i, i + 1)
        struct_qaoa.rz(0.5, i + 1)
        struct_qaoa.cx(i, i + 1)
    for i in range(4):
        struct_qaoa.rx(0.3, i)
struct_qaoa.measure(range(4), range(4))
show_detection("Structural QAOA (no name hint, MEDIUM confidence)", struct_qaoa)
