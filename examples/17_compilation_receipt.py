#!/usr/bin/env python3
"""Example 17: Compilation with receipt.

Compile a circuit and get a full audit trail as a JSON receipt.
"""
from __future__ import annotations

import json

from qb_compiler import QBCompiler, QBCircuit

# Build a GHZ-10 circuit using fluent API
circuit = QBCircuit(10)
circuit.h(0)
for i in range(9):
    circuit.cx(i, i + 1)
circuit.measure_all()

# Compile
compiler = QBCompiler.from_backend("ibm_fez")
result = compiler.compile(circuit)

print("=== Compilation Result ===")
print(f"Circuit:    GHZ-10")
print(f"Backend:    ibm_fez")
print(f"Input depth:  {result.original_depth}")
print(f"Output depth: {result.compiled_depth}")
print(f"Reduction:    {result.depth_reduction_pct:.1f}%")
print(f"Fidelity:     {result.estimated_fidelity:.4f}")
print(f"Time:         {result.compilation_time_ms:.1f} ms")
print()

# Build receipt
cost = compiler.estimate_cost(result.compiled_circuit, shots=4096)
receipt = {
    "circuit": "GHZ-10",
    "backend": "ibm_fez",
    "input_depth": result.original_depth,
    "output_depth": result.compiled_depth,
    "depth_reduction_pct": round(result.depth_reduction_pct, 1),
    "estimated_fidelity": round(result.estimated_fidelity, 4),
    "compilation_time_ms": round(result.compilation_time_ms, 1),
    "cost_4096_shots_usd": round(cost.total_usd, 4),
    "passes": [
        {
            "name": p.pass_name,
            "depth_before": p.depth_before,
            "depth_after": p.depth_after,
            "elapsed_ms": round(p.elapsed_ms, 1),
        }
        for p in result.pass_log
    ],
}

print("=== Receipt (JSON) ===")
print(json.dumps(receipt, indent=2))
