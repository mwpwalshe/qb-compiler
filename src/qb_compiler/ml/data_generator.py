"""Training data generation for ML layout prediction.

Generates labelled training data by sampling many random layouts for
each (circuit, backend) pair, scoring them with the calibration-aware
scorer, and labelling the physical qubits that appear in top-scoring
layouts as positive examples.

No ML dependencies required — this module only needs numpy and rustworkx.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qb_compiler.ml.features import (
    extract_circuit_features,
    extract_qubit_features,
    to_feature_vector,
)

if TYPE_CHECKING:
    from qb_compiler.calibration.models.backend_properties import BackendProperties
    from qb_compiler.ir.circuit import QBCircuit

logger = logging.getLogger(__name__)


@dataclass
class TrainingBatch:
    """Labelled training data for XGBoost layout prediction."""

    features: list[list[float]]
    labels: list[int]
    n_circuits: int = 0
    n_trials_total: int = 0
    n_positive: int = 0
    n_negative: int = 0


@dataclass
class LayoutSample:
    """A single sampled layout with its score."""

    layout: dict[int, int]  # logical -> physical
    score: float
    physical_qubits: frozenset[int]


class TrainingDataGenerator:
    """Generate training data by sampling and scoring random layouts.

    For each circuit, generates *n_trials* random valid layouts (connected
    subgraphs of the coupling map), scores them using the calibration-aware
    scorer, and labels qubits in the top fraction as positive examples.

    Parameters
    ----------
    backend_properties :
        Calibration data for the target backend.
    n_trials :
        Number of random layouts to sample per circuit.
    top_fraction :
        Fraction of layouts to consider as "good" (default 10%).
    seed :
        Random seed for reproducibility.
    """

    def __init__(
        self,
        backend_properties: BackendProperties,
        n_trials: int = 200,
        top_fraction: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self._backend = backend_properties
        self._n_trials = n_trials
        self._top_fraction = top_fraction
        self._rng = random.Random(seed)

        # Pre-compute adjacency from coupling map
        self._adjacency: dict[int, list[int]] = defaultdict(list)
        seen_edges: set[tuple[int, int]] = set()
        for q0, q1 in backend_properties.coupling_map:
            key = (min(q0, q1), max(q0, q1))
            if key not in seen_edges:
                seen_edges.add(key)
                self._adjacency[q0].append(q1)
                self._adjacency[q1].append(q0)

        self._all_qubits = sorted(self._adjacency.keys())

    def generate_from_circuit(
        self,
        circuit: QBCircuit,
    ) -> TrainingBatch:
        """Generate labelled training data for a single circuit."""
        return self.generate_from_circuits([circuit])

    def generate_from_circuits(
        self,
        circuits: list[QBCircuit],
    ) -> TrainingBatch:
        """Generate labelled training data from multiple circuits."""
        from qb_compiler.passes.mapping.calibration_mapper import (
            CalibrationMapper,
            CalibrationMapperConfig,
        )

        # Create a scorer (mapper used only for scoring)
        config = CalibrationMapperConfig(
            max_candidates=1,
            vf2_call_limit=1,
        )
        mapper = CalibrationMapper(self._backend, config=config)

        all_features: list[list[float]] = []
        all_labels: list[int] = []
        total_trials = 0

        for circuit in circuits:
            n_logical = circuit.n_qubits
            if n_logical > len(self._all_qubits):
                logger.warning(
                    "Circuit needs %d qubits but backend has %d — skipping",
                    n_logical,
                    len(self._all_qubits),
                )
                continue

            # Extract interactions for scoring
            interactions = CalibrationMapper._extract_interactions(circuit)

            # Sample random layouts
            samples: list[LayoutSample] = []
            for _ in range(self._n_trials):
                layout = self._random_connected_layout(n_logical, interactions)
                if layout is None:
                    continue
                score = mapper._score_layout(layout, interactions, circuit)
                samples.append(
                    LayoutSample(
                        layout=layout,
                        score=score,
                        physical_qubits=frozenset(layout.values()),
                    )
                )
                total_trials += 1

            if not samples:
                continue

            # Also add the greedy layout as a sample
            greedy_layout = mapper._greedy_layout(circuit, interactions)
            greedy_score = mapper._score_layout(
                greedy_layout, interactions, circuit
            )
            samples.append(
                LayoutSample(
                    layout=greedy_layout,
                    score=greedy_score,
                    physical_qubits=frozenset(greedy_layout.values()),
                )
            )

            # Sort by score (lower is better), take top fraction
            samples.sort(key=lambda s: s.score)
            n_top = max(1, int(len(samples) * self._top_fraction))
            top_samples = samples[:n_top]

            # Collect positive qubits (appeared in any top layout)
            positive_qubits: set[int] = set()
            for s in top_samples:
                positive_qubits |= s.physical_qubits

            # Build feature vectors for all physical qubits
            circ_feats = extract_circuit_features(circuit)
            for qid in self._all_qubits:
                qf = extract_qubit_features(qid, self._backend)
                fv = to_feature_vector(circ_feats, qf)
                label = 1 if qid in positive_qubits else 0
                all_features.append(fv)
                all_labels.append(label)

        n_pos = sum(all_labels)
        n_neg = len(all_labels) - n_pos

        logger.info(
            "Generated %d samples (%d positive, %d negative) from %d circuits, "
            "%d trials",
            len(all_labels),
            n_pos,
            n_neg,
            len(circuits),
            total_trials,
        )

        return TrainingBatch(
            features=all_features,
            labels=all_labels,
            n_circuits=len(circuits),
            n_trials_total=total_trials,
            n_positive=n_pos,
            n_negative=n_neg,
        )

    def _random_connected_layout(
        self,
        n_logical: int,
        interactions: dict[tuple[int, int], int],
    ) -> dict[int, int] | None:
        """Sample a random connected subgraph layout.

        Starts from a random physical qubit and grows via BFS to collect
        n_logical connected qubits.  Maps logical qubits to physical qubits
        by aligning interaction-heavy logical qubits to well-connected
        physical qubits.
        """
        if not self._all_qubits:
            return None

        # Pick random starting qubit
        start = self._rng.choice(self._all_qubits)

        # BFS to collect n_logical connected qubits
        selected: list[int] = [start]
        visited = {start}
        frontier = list(self._adjacency[start])
        self._rng.shuffle(frontier)

        while len(selected) < n_logical and frontier:
            next_q = frontier.pop(0)
            if next_q in visited:
                continue
            visited.add(next_q)
            selected.append(next_q)
            neighbors = list(self._adjacency[next_q])
            self._rng.shuffle(neighbors)
            frontier.extend(n for n in neighbors if n not in visited)

        if len(selected) < n_logical:
            return None

        # Map logical qubits to physical qubits
        # Sort logical qubits by interaction degree (most connected first)
        log_degree: Counter[int] = Counter()
        for (a, b), count in interactions.items():
            log_degree[a] += count
            log_degree[b] += count

        logical_order = sorted(
            range(n_logical),
            key=lambda q: -log_degree.get(q, 0),
        )

        # Sort physical qubits by connectivity in the selected subgraph
        phys_subgraph_degree: Counter[int] = Counter()
        for pq in selected:
            for neighbor in self._adjacency[pq]:
                if neighbor in set(selected):
                    phys_subgraph_degree[pq] += 1

        physical_order = sorted(
            selected,
            key=lambda q: -phys_subgraph_degree.get(q, 0),
        )

        layout: dict[int, int] = {}
        for log_q, phys_q in zip(logical_order, physical_order, strict=False):
            layout[log_q] = phys_q

        return layout
