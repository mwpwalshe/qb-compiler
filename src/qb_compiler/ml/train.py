"""Train an XGBoost layout predictor from calibration data.

Usage::

    python -m qb_compiler.ml.train

This generates training data from benchmark circuits + calibration
snapshots, trains an XGBoost binary classifier, and saves the model
weights to ``src/qb_compiler/ml/_weights/``.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = Path(__file__).parent / "_weights"


def _build_training_circuits() -> list[tuple[str, "QBCircuit"]]:
    """Build a diverse set of training circuits at various qubit counts."""
    from qb_compiler.ir.circuit import QBCircuit
    from qb_compiler.ir.operations import QBGate

    circuits: list[tuple[str, QBCircuit]] = []

    # GHZ circuits (linear chain)
    for n in [2, 3, 4, 5, 6, 8, 10, 12]:
        c = QBCircuit(n_qubits=n, n_clbits=0, name=f"ghz_{n}")
        c.add_gate(QBGate("h", (0,)))
        for i in range(n - 1):
            c.add_gate(QBGate("cx", (i, i + 1)))
        circuits.append((f"GHZ-{n}", c))

    # QAOA ring circuits
    for n in [3, 4, 5, 6, 8, 10]:
        c = QBCircuit(n_qubits=n, n_clbits=0, name=f"qaoa_{n}")
        for i in range(n - 1):
            c.add_gate(QBGate("cx", (i, i + 1)))
            c.add_gate(QBGate("rz", (i + 1,), params=(0.5,)))
            c.add_gate(QBGate("cx", (i, i + 1)))
        for i in range(n):
            c.add_gate(QBGate("rx", (i,), params=(0.7,)))
        circuits.append((f"QAOA-{n}", c))

    # Star topology (one hub qubit connected to all others)
    for n in [3, 4, 5, 6]:
        c = QBCircuit(n_qubits=n, n_clbits=0, name=f"star_{n}")
        for i in range(1, n):
            c.add_gate(QBGate("cx", (0, i)))
        circuits.append((f"Star-{n}", c))

    # Grid-like interactions (for 4-qubit square)
    c = QBCircuit(n_qubits=4, n_clbits=0, name="grid_4")
    c.add_gate(QBGate("cx", (0, 1)))
    c.add_gate(QBGate("cx", (1, 3)))
    c.add_gate(QBGate("cx", (0, 2)))
    c.add_gate(QBGate("cx", (2, 3)))
    circuits.append(("Grid-4", c))

    # Dense interaction (all-to-all for small circuits)
    for n in [3, 4, 5]:
        c = QBCircuit(n_qubits=n, n_clbits=0, name=f"dense_{n}")
        for i in range(n):
            for j in range(i + 1, n):
                c.add_gate(QBGate("cx", (i, j)))
        circuits.append((f"Dense-{n}", c))

    # QFT-like (long-range interactions)
    for n in [3, 4, 5, 6]:
        c = QBCircuit(n_qubits=n, n_clbits=0, name=f"qft_{n}")
        for i in range(n):
            c.add_gate(QBGate("h", (i,)))
            for j in range(i + 1, n):
                c.add_gate(QBGate("cx", (j, i)))
                c.add_gate(QBGate("rz", (i,), params=(math.pi / 2 ** (j - i),)))
                c.add_gate(QBGate("cx", (j, i)))
        circuits.append((f"QFT-{n}", c))

    return circuits


def _load_calibration_snapshots() -> list["BackendProperties"]:
    """Load all available calibration snapshots."""
    import glob

    from qb_compiler.calibration.static_provider import StaticCalibrationProvider

    fixture_dir = Path(__file__).parents[3] / "tests" / "fixtures" / "calibration_snapshots"
    snapshots = []

    # Try IBM Fez snapshots
    fez_files = sorted(glob.glob(str(fixture_dir / "ibm_fez*.json")))
    for f in fez_files:
        try:
            prov = StaticCalibrationProvider.from_json(f)
            snapshots.append(prov.properties)
            logger.info("Loaded calibration snapshot: %s", Path(f).name)
        except Exception as e:
            logger.warning("Failed to load %s: %s", f, e)

    # Also try loading via compiler helper
    if not snapshots:
        from qb_compiler.compiler import _load_calibration_fixture

        cal = _load_calibration_fixture("ibm_fez")
        if cal is not None:
            snapshots.append(cal)
            logger.info("Loaded IBM Fez calibration via compiler helper")

    return snapshots


def train_model(
    output_path: Path | None = None,
    n_trials: int = 300,
    top_fraction: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Train an XGBoost layout predictor and save weights.

    Returns
    -------
    dict
        Training metrics (accuracy, AUC, model size, etc.).
    """
    import numpy as np

    try:
        import xgboost as xgb
    except ImportError:
        print("ERROR: xgboost is required. Install with: pip install 'qb-compiler[ml]'")
        sys.exit(1)

    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("ERROR: scikit-learn is required. Install with: pip install scikit-learn")
        sys.exit(1)

    if output_path is None:
        output_path = _WEIGHTS_DIR / "ibm_heron_v1.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load calibration data
    snapshots = _load_calibration_snapshots()
    if not snapshots:
        print("ERROR: No calibration snapshots found.")
        sys.exit(1)

    if verbose:
        print(f"Loaded {len(snapshots)} calibration snapshot(s)")

    # Build training circuits
    circuits = _build_training_circuits()
    if verbose:
        print(f"Built {len(circuits)} training circuits")

    # Generate training data
    from qb_compiler.ml.data_generator import TrainingDataGenerator
    from qb_compiler.ml.features import feature_names

    all_features: list[list[float]] = []
    all_labels: list[int] = []

    start = time.perf_counter()

    for snap_idx, snapshot in enumerate(snapshots):
        gen = TrainingDataGenerator(
            snapshot,
            n_trials=n_trials,
            top_fraction=top_fraction,
            seed=seed + snap_idx,
        )

        for name, circ in circuits:
            batch = gen.generate_from_circuit(circ)
            all_features.extend(batch.features)
            all_labels.extend(batch.labels)
            if verbose:
                print(
                    f"  {name:>12s}: {batch.n_positive:>4d} positive, "
                    f"{batch.n_negative:>4d} negative "
                    f"({batch.n_trials_total} trials)"
                )

    gen_time = time.perf_counter() - start

    if verbose:
        n_pos = sum(all_labels)
        n_neg = len(all_labels) - n_pos
        print(
            f"\nTotal: {len(all_labels)} samples "
            f"({n_pos} positive, {n_neg} negative, "
            f"ratio={n_pos / len(all_labels):.1%}) "
            f"in {gen_time:.1f}s"
        )

    # Train/validation split
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Handle class imbalance
    n_pos_train = y_train.sum()
    n_neg_train = len(y_train) - n_pos_train
    scale_pos = n_neg_train / max(n_pos_train, 1)

    # Train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names())

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(scale_pos),
        "seed": seed,
        "verbosity": 0,
    }

    if verbose:
        print(f"\nTraining XGBoost (scale_pos_weight={scale_pos:.1f})...")

    evals_result: dict = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=150,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=20,
        verbose_eval=10 if verbose else 0,
    )

    # Evaluate
    y_pred_prob = model.predict(dval)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = float(accuracy_score(y_val, y_pred))
    auc = float(roc_auc_score(y_val, y_pred_prob))

    if verbose:
        print(f"\nValidation accuracy: {accuracy:.4f}")
        print(f"Validation AUC:      {auc:.4f}")

        # Feature importance
        importance = model.get_score(importance_type="gain")
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
        print("\nTop features by gain:")
        for fname, gain in sorted_imp[:10]:
            print(f"  {fname:>30s}: {gain:.2f}")

    # Save model
    model.save_model(str(output_path))
    model_size = output_path.stat().st_size

    if verbose:
        print(f"\nModel saved to {output_path} ({model_size / 1024:.1f} KB)")

    # Save metadata
    meta = {
        "version": "1.0.0",
        "backend_family": "ibm_heron",
        "n_training_samples": len(y_train),
        "n_validation_samples": len(y_val),
        "n_circuits": len(circuits),
        "n_calibration_snapshots": len(snapshots),
        "n_trials_per_circuit": n_trials,
        "top_fraction": top_fraction,
        "accuracy": accuracy,
        "auc": auc,
        "model_size_bytes": model_size,
        "feature_names": feature_names(),
        "xgb_params": params,
        "best_iteration": model.best_iteration,
    }

    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"Metadata saved to {meta_path}")

    return meta


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_model(verbose=True)


if __name__ == "__main__":
    main()
