#!/usr/bin/env python3
"""Example 11: ML-accelerated layout prediction.

Shows how to use XGBoost (Phase 2) and GNN (Phase 3) predictors
to speed up calibration-aware qubit mapping on large devices.

Install extras:
    pip install "qb-compiler[ml]"   # XGBoost
    pip install "qb-compiler[gnn]"  # GNN (PyTorch)
"""
from __future__ import annotations

from qb_compiler import QBCompiler, QBCircuit

# Build a circuit
circuit = QBCircuit(8).h(0)
for i in range(7):
    circuit.cx(i, i + 1)
circuit.measure_all()

# Compile — ML predictor is used automatically if installed
compiler = QBCompiler.from_backend("ibm_fez")
result = compiler.compile(circuit)

print(f"Depth: {result.compiled_depth}")
print(f"Estimated fidelity: {result.estimated_fidelity:.4f}")

# Or use the ML predictor directly
from qb_compiler.ml import is_available, is_gnn_available

if is_available():
    from qb_compiler.ml.layout_predictor import MLLayoutPredictor

    predictor = MLLayoutPredictor.load_bundled("ibm_heron")
    print(f"\nXGBoost model: AUC={predictor.metadata.get('auc', 'N/A')}")

if is_gnn_available():
    from qb_compiler.ml.gnn_router import GNNLayoutPredictor

    try:
        predictor = GNNLayoutPredictor.load_bundled("ibm_heron")
        print(f"GNN model: AUC={predictor.metadata.get('training_auc', 'N/A')}")
    except FileNotFoundError:
        print("GNN weights not found — train with: python -m qb_compiler.ml.train_gnn")
