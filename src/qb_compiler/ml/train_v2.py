"""Train XGBoost v2: regression on post-routing 2Q gate count.

Old target: binary classification (is this qubit in top 10% layouts?)
New target: regression (predict post-routing 2Q gate count for a layout)

Usage::

    python -m qb_compiler.ml.train_v2
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = Path(__file__).parent / "_weights"


def _build_training_circuits() -> list[tuple[str, Any]]:
    """Build diverse circuits: GHZ, QAOA, QFT, random."""
    from qb_compiler.ir.circuit import QBCircuit
    from qb_compiler.ir.operations import QBGate

    circuits: list[tuple[str, QBCircuit]] = []

    # GHZ (linear chain)
    for n in [3, 4, 5, 6, 8, 10, 12]:
        c = QBCircuit(n_qubits=n, n_clbits=n, name=f"ghz_{n}")
        c.add_gate(QBGate("h", (0,)))
        for i in range(n - 1):
            c.add_gate(QBGate("cx", (i, i + 1)))
        for i in range(n):
            c.add_measurement(i, i)
        circuits.append((f"GHZ-{n}", c))

    # QAOA ring
    for n in [4, 5, 6, 8, 10]:
        c = QBCircuit(n_qubits=n, n_clbits=n, name=f"qaoa_{n}")
        for i in range(n - 1):
            c.add_gate(QBGate("cx", (i, i + 1)))
            c.add_gate(QBGate("rz", (i + 1,), params=(0.5,)))
            c.add_gate(QBGate("cx", (i, i + 1)))
        # Close the ring
        c.add_gate(QBGate("cx", (n - 1, 0)))
        c.add_gate(QBGate("rz", (0,), params=(0.5,)))
        c.add_gate(QBGate("cx", (n - 1, 0)))
        for i in range(n):
            c.add_gate(QBGate("rx", (i,), params=(0.7,)))
        for i in range(n):
            c.add_measurement(i, i)
        circuits.append((f"QAOA-{n}", c))

    # QFT (long-range interactions)
    for n in [4, 5, 6, 8]:
        c = QBCircuit(n_qubits=n, n_clbits=n, name=f"qft_{n}")
        for i in range(n):
            c.add_gate(QBGate("h", (i,)))
            for j in range(i + 1, n):
                c.add_gate(QBGate("cx", (j, i)))
                c.add_gate(QBGate("rz", (i,), params=(math.pi / 2 ** (j - i),)))
                c.add_gate(QBGate("cx", (j, i)))
        for i in range(n):
            c.add_measurement(i, i)
        circuits.append((f"QFT-{n}", c))

    # Star topology
    for n in [4, 5, 6]:
        c = QBCircuit(n_qubits=n, n_clbits=n, name=f"star_{n}")
        for i in range(1, n):
            c.add_gate(QBGate("cx", (0, i)))
        for i in range(n):
            c.add_measurement(i, i)
        circuits.append((f"Star-{n}", c))

    return circuits


def _build_target_from_props(props: Any) -> Any:
    """Build a Qiskit Target from BackendProperties."""
    from qiskit.circuit import Measure, Parameter
    from qiskit.circuit.library import CXGate, CZGate, HGate, RZGate, SXGate, XGate
    from qiskit.transpiler import InstructionProperties, Target

    n_q = props.n_qubits
    target = Target(num_qubits=n_q)
    gate_types = {gp.gate_type for gp in props.gate_properties if len(gp.qubits) == 2}
    gate_2q_cls = CZGate if "cz" in gate_types else CXGate
    twoq_props = {}
    for gp in props.gate_properties:
        if len(gp.qubits) == 2:
            twoq_props[gp.qubits] = InstructionProperties(
                error=gp.error_rate,
                duration=gp.gate_time_ns * 1e-9 if gp.gate_time_ns else 68e-9,
            )
    target.add_instruction(gate_2q_cls(), twoq_props)
    sq_props = {(q,): None for q in range(n_q)}
    theta = Parameter("theta")
    target.add_instruction(RZGate(theta), sq_props)
    target.add_instruction(SXGate(), sq_props)
    target.add_instruction(XGate(), sq_props)
    target.add_instruction(HGate(), sq_props)
    target.add_instruction(Measure(), sq_props)
    return target


def _load_calibration_snapshots() -> list[Any]:
    """Load all available calibration snapshots."""
    import glob

    from qb_compiler.calibration.models.backend_properties import BackendProperties

    fixture_dir = Path(__file__).parents[3] / "tests" / "fixtures" / "calibration_snapshots"
    snapshots = []

    fez_files = sorted(glob.glob(str(fixture_dir / "ibm_fez*.json")))
    for f in fez_files:
        try:
            props = BackendProperties.from_qubitboost_json(f)
            if len(props.qubit_properties) >= 50:
                snapshots.append(props)
                logger.info("Loaded: %s (%d qubits)", Path(f).name, props.n_qubits)
        except Exception as e:
            logger.warning("Failed to load %s: %s", f, e)

    return snapshots


def train_model_v2(
    output_path: Path | None = None,
    n_trials: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Train XGBoost v2 regressor on post-routing 2Q gate count."""
    import numpy as np

    try:
        import xgboost as xgb
    except ImportError:
        logger.error(" xgboost required. pip install xgboost")
        sys.exit(1)

    try:
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.error(" scikit-learn required. pip install scikit-learn")
        sys.exit(1)

    if output_path is None:
        output_path = _WEIGHTS_DIR / "ibm_heron_v2.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load calibration
    snapshots = _load_calibration_snapshots()
    if not snapshots:
        logger.error(" No calibration snapshots found.")
        sys.exit(1)

    if verbose:
        logger.info(f"Loaded {len(snapshots)} calibration snapshot(s)")

    # Build circuits
    circuits = _build_training_circuits()
    if verbose:
        logger.info(f"Built {len(circuits)} training circuits")

    # Generate training data
    from qb_compiler.ml.data_generator_v2 import TrainingDataGeneratorV2, v2_feature_names

    all_features: list[list[float]] = []
    all_targets: list[float] = []
    all_targets_depth: list[float] = []
    all_targets_fid: list[float] = []

    start = time.perf_counter()

    for snap_idx, props in enumerate(snapshots):
        target = _build_target_from_props(props)
        if verbose:
            logger.info(f"Snapshot {snap_idx + 1}/{len(snapshots)}: {props.n_qubits} qubits")

        gen = TrainingDataGeneratorV2(
            props,
            qiskit_target=target,
            n_trials=n_trials,
            seed=seed + snap_idx,
        )

        data = gen.generate(circuits)
        all_features.extend(data.features)
        all_targets.extend(data.targets_2q_gates)
        all_targets_depth.extend(data.targets_depth)
        all_targets_fid.extend(data.targets_fidelity)

        if verbose:
            logger.info(f"  Generated {data.n_samples} samples")

    gen_time = time.perf_counter() - start

    if verbose:
        logger.info(f"Total: {len(all_targets)} samples in {gen_time:.1f}s")
        targets_arr = np.array(all_targets)
        logger.info(
            f"  2Q gates: mean={targets_arr.mean():.1f}, "
            f"std={targets_arr.std():.1f}, "
            f"min={targets_arr.min():.0f}, max={targets_arr.max():.0f}"
        )

    # Train/val split
    x_data = np.array(all_features, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)

    x_train, x_val, y_train, y_val = train_test_split(
        x_data,
        y,
        test_size=0.2,
        random_state=seed,
    )

    # Train XGBoost regressor
    fnames = v2_feature_names()
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=fnames)
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=fnames)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "verbosity": 0,
    }

    if verbose:
        logger.info("Training XGBoost regressor...")

    evals_result: dict = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=20,
        verbose_eval=20 if verbose else 0,
    )

    # Evaluate
    y_pred = model.predict(dval)
    mae = float(mean_absolute_error(y_val, y_pred))
    correlation = float(np.corrcoef(y_val, y_pred)[0, 1])

    if verbose:
        logger.info(f"Validation MAE: {mae:.3f}")
        logger.info(f"Correlation with actual: {correlation:.4f}")

        # Feature importance
        importance = model.get_score(importance_type="gain")
        sorted_imp = sorted(
            importance.items(),
            key=lambda x: -float(x[1]) if isinstance(x[1], (int, float)) else 0.0,
        )
        logger.info("Top features by gain:")
        for fname, gain in sorted_imp[:10]:
            logger.info(f"  {fname:>30s}: {gain:.2f}")

    # Save model
    model.save_model(str(output_path))
    model_size = output_path.stat().st_size

    if verbose:
        logger.info(f"Model saved to {output_path} ({model_size / 1024:.1f} KB)")

    # Save metadata
    meta = {
        "version": "2.0.0",
        "model_type": "regression",
        "target": "post_routing_2q_gate_count",
        "backend_family": "ibm_heron",
        "n_training_samples": len(y_train),
        "n_validation_samples": len(y_val),
        "n_circuits": len(circuits),
        "n_calibration_snapshots": len(snapshots),
        "n_trials_per_circuit": n_trials,
        "mae": mae,
        "correlation": correlation,
        "model_size_bytes": model_size,
        "feature_names": fnames,
        "xgb_params": params,
        "best_iteration": model.best_iteration,
        "generation_time_s": gen_time,
    }

    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        logger.info(f"Metadata saved to {meta_path}")

    # Save training data as parquet if pyarrow available
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = {name: [row[i] for row in all_features] for i, name in enumerate(fnames)}
        columns["target_2q_gates"] = all_targets
        columns["target_depth"] = all_targets_depth
        columns["target_fidelity"] = all_targets_fid

        table = pa.table(columns)
        parquet_path = output_path.with_suffix(".parquet")
        pq.write_table(table, parquet_path)
        if verbose:
            logger.info(f"Training data saved to {parquet_path}")
    except ImportError:
        if verbose:
            logger.info("pyarrow not available — skipping parquet export")

    return meta


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    train_model_v2(verbose=True)


if __name__ == "__main__":
    main()
