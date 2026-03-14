"""XGBoost-based layout prediction for calibration-aware qubit mapping.

v1: binary classification — predicts per-qubit probability of appearing
    in a high-quality layout.  Trained on pre-routing calibration scores.

v2: regression — predicts post-routing 2Q gate count for a given layout.
    Trained on actual trial-transpilation outcomes.  Scores layouts
    directly instead of individual qubits.  For candidate qubit filtering,
    samples random layouts and aggregates which qubits appear in the
    lowest-gate-count predictions.

Requires: ``pip install "qb-compiler[ml]"``
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from qb_compiler.ml.features import build_feature_matrix, extract_circuit_features, feature_names

if TYPE_CHECKING:
    from qb_compiler.calibration.models.backend_properties import BackendProperties
    from qb_compiler.ir.circuit import QBCircuit

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = Path(__file__).parent / "_weights"


class MLLayoutPredictor:
    """Predict promising physical qubits for layout mapping (v1).

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


class MLLayoutPredictorV2:
    """Predict post-routing 2Q gate count for candidate layouts (v2).

    Unlike v1 which scores individual qubits, v2 scores entire layouts
    by predicting how many 2Q gates will remain after routing.  For
    candidate qubit filtering, it samples random connected layouts and
    identifies which qubits consistently appear in low-gate-count
    predictions.

    Can also directly score a specific layout via :meth:`score_layout`.

    Parameters
    ----------
    model_path :
        Path to a saved XGBoost v2 model file (JSON format).
        If None, loads bundled ``ibm_heron_v2.json``.
    min_candidates :
        Minimum number of candidate qubits to return.
    n_probe_layouts :
        Number of random layouts to sample for qubit scoring.
    seed :
        Random seed for layout sampling.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_candidates: int = 30,
        n_probe_layouts: int = 200,
        seed: int = 42,
    ) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for ML layout prediction. "
                "Install with: pip install 'qb-compiler[ml]'"
            ) from exc

        self._min_candidates = min_candidates
        self._n_probe_layouts = n_probe_layouts
        self._rng = random.Random(seed)

        if model_path is None:
            model_path = _WEIGHTS_DIR / "ibm_heron_v2.json"

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self._model_path}. "
                f"Train with: python -m qb_compiler.ml.train_v2"
            )

        self._model = xgb.Booster()
        self._model.load_model(str(self._model_path))

        meta_path = self._model_path.with_suffix(".meta.json")
        self._metadata: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

        self._feature_names = self._metadata.get("feature_names")
        if not self._feature_names:
            from qb_compiler.ml.data_generator_v2 import v2_feature_names
            self._feature_names = v2_feature_names()

        logger.info(
            "Loaded ML layout predictor v2 from %s (MAE=%.3f, r=%.4f)",
            self._model_path.name,
            self._metadata.get("mae", -1),
            self._metadata.get("correlation", -1),
        )

    def score_layout(
        self,
        circuit: QBCircuit,
        layout: dict[int, int],
        backend: BackendProperties,
    ) -> float:
        """Predict post-routing 2Q gate count for a specific layout.

        Returns the predicted gate count (lower = better).
        """
        import numpy as np
        import xgboost as xgb

        feats = self._extract_layout_features(circuit, layout, backend)
        dmatrix = xgb.DMatrix(
            np.array([feats], dtype=np.float32),
            feature_names=self._feature_names,
        )
        pred = self._model.predict(dmatrix)
        return float(pred[0])

    def score_layouts(
        self,
        circuit: QBCircuit,
        layouts: list[dict[int, int]],
        backend: BackendProperties,
    ) -> list[float]:
        """Predict post-routing 2Q gate count for multiple layouts (batched)."""
        import numpy as np
        import xgboost as xgb

        if not layouts:
            return []

        feature_matrix = [
            self._extract_layout_features(circuit, layout, backend)
            for layout in layouts
        ]
        dmatrix = xgb.DMatrix(
            np.array(feature_matrix, dtype=np.float32),
            feature_names=self._feature_names,
        )
        preds = self._model.predict(dmatrix)
        return [float(p) for p in preds]

    def predict_candidate_qubits(
        self,
        circuit: QBCircuit,
        backend: BackendProperties,
    ) -> list[int]:
        """Return physical qubits that appear in low-gate-count layouts.

        Samples random connected layouts, scores each with the v2 model,
        and returns qubits that frequently appear in the best-scoring
        (lowest predicted gate count) layouts.
        """

        n_logical = circuit.n_qubits

        # Build adjacency for random layout sampling
        adjacency: dict[int, list[int]] = defaultdict(list)
        seen_edges: set[tuple[int, int]] = set()
        for q0, q1 in backend.coupling_map:
            key = (min(q0, q1), max(q0, q1))
            if key not in seen_edges:
                seen_edges.add(key)
                adjacency[q0].append(q1)
                adjacency[q1].append(q0)
        all_qubits = sorted(adjacency.keys())

        if n_logical > len(all_qubits):
            return all_qubits

        # Extract interactions for degree-aware mapping
        interactions = self._extract_interactions(circuit)

        # Sample random connected layouts
        layouts: list[dict[int, int]] = []
        for _ in range(self._n_probe_layouts):
            layout = self._random_connected_layout(
                n_logical, all_qubits, adjacency, interactions
            )
            if layout is not None:
                layouts.append(layout)

        if not layouts:
            return all_qubits

        # Score all layouts in a single batch
        scores = self.score_layouts(circuit, layouts, backend)

        # Rank layouts by predicted gate count (lower = better)
        ranked = sorted(zip(scores, layouts, strict=False), key=lambda x: x[0])

        # Take top 25% of layouts
        n_top = max(len(ranked) // 4, 10)
        top_layouts = [layout for _, layout in ranked[:n_top]]

        # Count qubit appearances in top layouts
        qubit_counts: Counter[int] = Counter()
        for layout in top_layouts:
            for phys_q in layout.values():
                qubit_counts[phys_q] += 1

        # Sort by frequency (most common first)
        sorted_qubits = [q for q, _ in qubit_counts.most_common()]

        # Ensure minimum candidates
        n_candidates = max(self._min_candidates, n_logical * 3)
        n_candidates = min(n_candidates, len(all_qubits))

        # Pad with remaining qubits if needed
        candidate_set = set(sorted_qubits[:n_candidates])
        if len(candidate_set) < n_candidates:
            for q in all_qubits:
                if len(candidate_set) >= n_candidates:
                    break
                candidate_set.add(q)

        candidates = sorted(candidate_set)

        logger.debug(
            "ML v2 predictor: %d candidates from %d probes "
            "(best predicted 2Q=%.1f, worst in top=%.1f)",
            len(candidates),
            len(layouts),
            ranked[0][0] if ranked else 0,
            ranked[n_top - 1][0] if ranked else 0,
        )

        return candidates

    def _extract_layout_features(
        self,
        circuit: QBCircuit,
        layout: dict[int, int],
        backend: BackendProperties,
    ) -> list[float]:
        """Extract v2 feature vector for a layout."""
        # Circuit features (8)
        cf = extract_circuit_features(circuit)
        feats = cf.to_list()

        n_logical = circuit.n_qubits
        phys_qubits = [layout[i] for i in range(n_logical)]
        phys_set = set(phys_qubits)

        # Build adjacency + gate error lookup
        adjacency: dict[int, list[int]] = defaultdict(list)
        seen_edges: set[tuple[int, int]] = set()
        for q0, q1 in backend.coupling_map:
            key = (min(q0, q1), max(q0, q1))
            if key not in seen_edges:
                seen_edges.add(key)
                adjacency[q0].append(q1)
                adjacency[q1].append(q0)

        gate_errors: dict[frozenset[int], float] = {}
        for gp in backend.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                gate_errors[frozenset(gp.qubits)] = gp.error_rate

        # Mean T1, T2, readout error
        t1s, t2s, ros = [], [], []
        for pq in phys_qubits:
            qp = backend.qubit(pq)
            t1s.append(qp.t1_us if qp and qp.t1_us else 100.0)
            t2s.append(qp.t2_us if qp and qp.t2_us else 80.0)
            ros.append(qp.readout_error if qp and qp.readout_error is not None else 0.015)

        feats.append(sum(t1s) / len(t1s))  # layout_mean_t1
        feats.append(sum(t2s) / len(t2s))  # layout_mean_t2
        feats.append(sum(ros) / len(ros))  # layout_mean_readout_error

        # Gate errors on edges required by circuit interactions
        interactions = self._extract_interactions(circuit)
        required_errors = []
        for (la, lb) in interactions:
            pa, pb = layout.get(la, 0), layout.get(lb, 0)
            err = gate_errors.get(frozenset({pa, pb}), 0.02)
            required_errors.append(err)

        if required_errors:
            feats.append(sum(required_errors) / len(required_errors))  # mean
            feats.append(max(required_errors))  # max (bottleneck)
        else:
            feats.append(0.02)
            feats.append(0.02)

        # Subgraph density
        internal_edges = 0
        for pq in phys_qubits:
            for nb in adjacency[pq]:
                if nb in phys_set and nb > pq:
                    internal_edges += 1
        possible = n_logical * (n_logical - 1) / 2
        feats.append(internal_edges / possible if possible > 0 else 0.0)

        # Boundary edges
        boundary = 0
        for pq in phys_qubits:
            for nb in adjacency[pq]:
                if nb not in phys_set:
                    boundary += 1
        feats.append(float(boundary))

        # Max shortest path (BFS within mapped subgraph)
        feats.append(float(self._max_shortest_path(phys_qubits, adjacency)))

        # Region mean error (2-hop neighborhood)
        neighborhood = set(phys_qubits)
        for pq in phys_qubits:
            for nb in adjacency[pq]:
                neighborhood.add(nb)
                for nb2 in adjacency.get(nb, []):
                    neighborhood.add(nb2)
        region_errors = []
        for q in neighborhood:
            for nb in adjacency[q]:
                if nb in neighborhood and nb > q:
                    err = gate_errors.get(frozenset({q, nb}))
                    if err is not None:
                        region_errors.append(err)
        feats.append(sum(region_errors) / len(region_errors) if region_errors else 0.02)

        return feats

    @staticmethod
    def _extract_interactions(circuit: QBCircuit) -> dict[tuple[int, int], int]:
        """Extract 2Q interaction graph from circuit."""
        from collections import defaultdict

        from qb_compiler.ir.operations import QBGate

        interactions: dict[tuple[int, int], int] = defaultdict(int)
        for op in circuit.iter_ops():
            if isinstance(op, QBGate) and op.num_qubits >= 2:
                q_sorted = sorted(op.qubits[:2])
                interactions[(q_sorted[0], q_sorted[1])] += 1
        return dict(interactions)

    @staticmethod
    def _max_shortest_path(qubits: list[int], adjacency: dict[int, list[int]]) -> int:
        """BFS max shortest path between mapped qubits."""
        qset = set(qubits)
        max_dist = 0
        for src in qubits:
            dist = {src: 0}
            queue = [src]
            idx = 0
            while idx < len(queue):
                cur = queue[idx]
                idx += 1
                for nb in adjacency[cur]:
                    if nb in qset and nb not in dist:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
            for tgt in qubits:
                if tgt in dist:
                    max_dist = max(max_dist, dist[tgt])
                else:
                    max_dist = max(max_dist, len(qubits) * 3)
        return max_dist

    def _random_connected_layout(
        self,
        n_logical: int,
        all_qubits: list[int],
        adjacency: dict[int, list[int]],
        interactions: dict[tuple[int, int], int],
    ) -> dict[int, int] | None:
        """Sample a random connected subgraph layout."""
        start = self._rng.choice(all_qubits)
        selected: list[int] = [start]
        visited = {start}
        frontier = list(adjacency[start])
        self._rng.shuffle(frontier)

        while len(selected) < n_logical and frontier:
            next_q = frontier.pop(0)
            if next_q in visited:
                continue
            visited.add(next_q)
            selected.append(next_q)
            neighbors = list(adjacency[next_q])
            self._rng.shuffle(neighbors)
            frontier.extend(n for n in neighbors if n not in visited)

        if len(selected) < n_logical:
            return None

        # Degree-aware mapping
        log_degree: Counter[int] = Counter()
        for (a, b), count in interactions.items():
            log_degree[a] += count
            log_degree[b] += count

        logical_order = sorted(range(n_logical), key=lambda q: -log_degree.get(q, 0))

        phys_degree: Counter[int] = Counter()
        sel_set = set(selected)
        for pq in selected:
            for nb in adjacency[pq]:
                if nb in sel_set:
                    phys_degree[pq] += 1

        physical_order = sorted(selected, key=lambda q: -phys_degree.get(q, 0))
        return {lg: p for lg, p in zip(logical_order, physical_order, strict=False)}

    @classmethod
    def load_bundled(
        cls, backend_family: str = "ibm_heron"
    ) -> MLLayoutPredictorV2:
        """Load pre-trained v2 model for a backend family."""
        model_map = {
            "ibm_heron": "ibm_heron_v2.json",
        }
        filename = model_map.get(backend_family)
        if filename is None:
            raise ValueError(
                f"No bundled v2 model for {backend_family!r}. "
                f"Available: {list(model_map.keys())}"
            )
        return cls(model_path=_WEIGHTS_DIR / filename)

    @property
    def metadata(self) -> dict:
        """Training metadata (MAE, correlation, feature names, etc.)."""
        return dict(self._metadata)
