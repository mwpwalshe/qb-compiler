"""XGBoost-based layout prediction for calibration-aware qubit mapping.

Predicts which physical qubits are most likely to appear in high-quality
layouts, narrowing the VF2 search space from O(n_physical) to O(top_k).

Requires: ``pip install "qb-compiler[ml]"``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from qb_compiler.ml.features import build_feature_matrix, feature_names

if TYPE_CHECKING:
    from qb_compiler.calibration.models.backend_properties import BackendProperties
    from qb_compiler.ir.circuit import QBCircuit

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = Path(__file__).parent / "_weights"


class MLLayoutPredictor:
    """Predict promising physical qubits for layout mapping.

    Uses a trained XGBoost model to score all physical qubits and return
    the top-K most likely to appear in a high-quality layout.  This
    narrows the VF2 search space, making ``CalibrationMapper`` faster
    on large backends (e.g. IBM Fez with 156 qubits).

    Parameters
    ----------
    model_path :
        Path to a saved XGBoost model file (JSON format).
        If None, attempts to load bundled weights.
    top_k_factor :
        Return ``n_logical * top_k_factor`` candidate qubits.
    min_candidates :
        Minimum number of candidates to return (overrides top_k_factor
        for very small circuits).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        top_k_factor: float = 3.0,
        min_candidates: int = 20,
    ) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for ML layout prediction. "
                "Install with: pip install 'qb-compiler[ml]'"
            ) from exc

        self._top_k_factor = top_k_factor
        self._min_candidates = min_candidates

        if model_path is None:
            model_path = _WEIGHTS_DIR / "ibm_heron_v1.json"

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self._model_path}. "
                f"Train a model with: python -m qb_compiler.ml.train"
            )

        self._model = xgb.Booster()
        self._model.load_model(str(self._model_path))

        # Load metadata if available
        meta_path = self._model_path.with_suffix(".meta.json")
        self._metadata: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

        logger.info(
            "Loaded ML layout predictor from %s (version=%s)",
            self._model_path.name,
            self._metadata.get("version", "unknown"),
        )

    def predict_candidate_qubits(
        self,
        circuit: QBCircuit,
        backend: BackendProperties,
    ) -> list[int]:
        """Return physical qubits most likely to appear in good layouts.

        Parameters
        ----------
        circuit :
            The circuit to be mapped (IR level).
        backend :
            Backend calibration data.

        Returns
        -------
        list[int]
            Physical qubit IDs, sorted by predicted probability
            (highest first).
        """
        import numpy as np
        import xgboost as xgb

        # Build feature matrix for all physical qubits
        feature_matrix, qubit_ids = build_feature_matrix(circuit, backend)

        if not feature_matrix:
            return qubit_ids  # fallback: return all

        # Predict probabilities
        dmatrix = xgb.DMatrix(
            np.array(feature_matrix, dtype=np.float32),
            feature_names=feature_names(),
        )
        probabilities = self._model.predict(dmatrix)

        # Sort by probability (highest first)
        scored = sorted(
            zip(qubit_ids, probabilities, strict=False),
            key=lambda x: -x[1],
        )

        # Determine how many to return
        n_logical = circuit.n_qubits
        top_k = max(
            int(n_logical * self._top_k_factor),
            self._min_candidates,
        )
        top_k = min(top_k, len(scored))  # don't exceed available

        candidates = [qid for qid, _prob in scored[:top_k]]

        logger.debug(
            "ML predictor: %d candidates from %d physical qubits "
            "(top prob=%.3f, cutoff prob=%.3f)",
            len(candidates),
            len(qubit_ids),
            scored[0][1] if scored else 0,
            scored[top_k - 1][1] if scored else 0,
        )

        return candidates

    @classmethod
    def load_bundled(
        cls, backend_family: str = "ibm_heron"
    ) -> MLLayoutPredictor:
        """Load pre-trained model for a backend family.

        Parameters
        ----------
        backend_family :
            One of "ibm_heron".  More families will be added.
        """
        model_map = {
            "ibm_heron": "ibm_heron_v1.json",
        }
        filename = model_map.get(backend_family)
        if filename is None:
            raise ValueError(
                f"No bundled model for {backend_family!r}. "
                f"Available: {list(model_map.keys())}"
            )
        return cls(model_path=_WEIGHTS_DIR / filename)

    @property
    def metadata(self) -> dict:
        """Training metadata (version, AUC, feature names, etc.)."""
        return dict(self._metadata)
