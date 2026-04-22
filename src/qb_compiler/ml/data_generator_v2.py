"""Training data generation v2: post-routing gate count as target.

Unlike v1 which labelled qubits by pre-routing calibration score,
v2 trial-transpiles each candidate layout and records the actual
post-routing 2Q gate count.  This is what correlates with real
hardware fidelity (fewer SWAPs = higher fidelity).

Generates per-layout regression data:
    features: circuit features + layout aggregate features
    target:   post_routing_2q_gate_count (lower = better)

Also records secondary targets: post_routing_depth, estimated_fidelity.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qb_compiler.calibration.models.backend_properties import BackendProperties
    from qb_compiler.ir.circuit import QBCircuit

logger = logging.getLogger(__name__)


@dataclass
class LayoutSampleV2:
    """A single layout sample with post-routing metrics."""

    layout: dict[int, int]
    physical_qubits: list[int]
    features: list[float]
    post_routing_2q_gates: int
    post_routing_depth: int
    estimated_fidelity: float
    calibration_score: float
    region: str  # e.g. "120-147"


@dataclass
class TrainingDataV2:
    """Regression training data for post-routing prediction."""

    features: list[list[float]]
    targets_2q_gates: list[float]
    targets_depth: list[float]
    targets_fidelity: list[float]
    calibration_scores: list[float]
    n_circuits: int = 0
    n_samples: int = 0
    n_calibrations: int = 0


# ── New features for v2 (added on top of v1 features) ────────────────

V2_FEATURE_NAMES = [
    # v1 circuit features (8)
    "n_logical_qubits",
    "n_gates",
    "n_2q_gates",
    "depth",
    "interaction_density",
    "max_interaction_degree",
    "mean_interaction_weight",
    "max_interaction_weight",
    # v2 layout-level aggregate features (9)
    "layout_mean_t1",
    "layout_mean_t2",
    "layout_mean_readout_error",
    "layout_mean_gate_error",
    "layout_max_gate_error",  # bottleneck — max error matters more than mean
    "layout_subgraph_density",  # edges / possible_edges in mapped subgraph
    "layout_boundary_edges",  # edges from mapped to unmapped (routing flexibility)
    "layout_max_shortest_path",  # longest shortest path between mapped qubits
    "layout_region_mean_error",  # average error in the surrounding 2-hop neighborhood
]


def v2_feature_names() -> list[str]:
    """Return feature names for v2 model."""
    return list(V2_FEATURE_NAMES)


class TrainingDataGeneratorV2:
    """Generate training data with post-routing gate counts as targets.

    Parameters
    ----------
    backend_properties :
        Calibration data for the target backend.
    qiskit_target :
        Qiskit Target for trial transpilation.
    n_trials :
        Number of random layouts to sample per circuit.
    seed :
        Random seed.
    """

    def __init__(
        self,
        backend_properties: BackendProperties,
        qiskit_target: Any,
        n_trials: int = 500,
        seed: int | None = None,
    ) -> None:
        self._backend = backend_properties
        self._target = qiskit_target
        self._n_trials = n_trials
        self._rng = random.Random(seed)

        # Pre-compute adjacency
        self._adjacency: dict[int, list[int]] = defaultdict(list)
        seen_edges: set[tuple[int, int]] = set()
        for q0, q1 in backend_properties.coupling_map:
            key = (min(q0, q1), max(q0, q1))
            if key not in seen_edges:
                seen_edges.add(key)
                self._adjacency[q0].append(q1)
                self._adjacency[q1].append(q0)

        self._all_qubits = sorted(self._adjacency.keys())

        # Pre-compute gate errors for fast lookup
        self._gate_errors: dict[frozenset[int], float] = {}
        for gp in backend_properties.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                self._gate_errors[frozenset(gp.qubits)] = gp.error_rate

    def generate(
        self,
        circuits: list[tuple[str, QBCircuit]],
    ) -> TrainingDataV2:
        """Generate training data from circuits."""
        from qiskit import transpile

        from qb_compiler.passes.mapping.calibration_mapper import (
            CalibrationMapper,
            CalibrationMapperConfig,
        )

        config = CalibrationMapperConfig(max_candidates=1, vf2_call_limit=1)
        mapper = CalibrationMapper(self._backend, config=config)

        all_features: list[list[float]] = []
        all_2q: list[float] = []
        all_depth: list[float] = []
        all_fid: list[float] = []
        all_cal: list[float] = []
        total = 0

        for name, circuit in circuits:
            n_logical = circuit.n_qubits
            if n_logical > len(self._all_qubits):
                continue

            interactions = CalibrationMapper._extract_interactions(circuit)
            qc = CalibrationMapper._ir_to_qiskit_circuit(circuit)
            if qc is None:
                logger.warning("Could not convert %s to Qiskit — skipping", name)
                continue

            successes = 0
            for _trial in range(self._n_trials):
                layout = self._random_connected_layout(n_logical, interactions)
                if layout is None:
                    continue

                phys = [layout[i] for i in range(n_logical)]

                # Trial transpile
                try:
                    tc = transpile(
                        qc,
                        target=self._target,
                        optimization_level=1,
                        initial_layout=phys,
                        seed_transpiler=42,
                    )
                except Exception:
                    continue

                # Count post-routing metrics
                n_2q = sum(
                    1
                    for inst in tc.data
                    if len(inst.qubits) == 2
                    and inst.operation.name not in ("barrier", "measure", "reset")
                )
                d = tc.depth()

                # Estimated fidelity
                fid = self._est_fidelity(layout, n_logical)

                # Calibration score (pre-routing)
                cal_score = mapper._score_layout(layout, interactions, circuit)

                # Extract features
                feats = self._extract_layout_features(circuit, layout, interactions)

                all_features.append(feats)
                all_2q.append(float(n_2q))
                all_depth.append(float(d))
                all_fid.append(fid)
                all_cal.append(cal_score)
                successes += 1
                total += 1

            logger.info(
                "%s: %d/%d layouts trial-transpiled",
                name,
                successes,
                self._n_trials,
            )

        return TrainingDataV2(
            features=all_features,
            targets_2q_gates=all_2q,
            targets_depth=all_depth,
            targets_fidelity=all_fid,
            calibration_scores=all_cal,
            n_circuits=len(circuits),
            n_samples=total,
        )

    def _extract_layout_features(
        self,
        circuit: QBCircuit,
        layout: dict[int, int],
        interactions: dict[tuple[int, int], int],
    ) -> list[float]:
        """Extract v2 feature vector for a layout."""
        from qb_compiler.ml.features import extract_circuit_features

        # Circuit features (8)
        cf = extract_circuit_features(circuit)
        feats = cf.to_list()

        n_logical = circuit.n_qubits
        phys_qubits = [layout[i] for i in range(n_logical)]
        phys_set = set(phys_qubits)

        # Layout aggregate features
        # Mean T1, T2, readout error across mapped qubits
        t1s, t2s, ros = [], [], []
        for pq in phys_qubits:
            qp = self._backend.qubit(pq)
            t1s.append(qp.t1_us if qp and qp.t1_us else 100.0)
            t2s.append(qp.t2_us if qp and qp.t2_us else 80.0)
            ros.append(qp.readout_error if qp and qp.readout_error is not None else 0.015)

        feats.append(sum(t1s) / len(t1s))  # layout_mean_t1
        feats.append(sum(t2s) / len(t2s))  # layout_mean_t2
        feats.append(sum(ros) / len(ros))  # layout_mean_readout_error

        # Gate errors on edges required by the circuit
        required_errors = []
        for la, lb in interactions:
            pa, pb = layout[la], layout[lb]
            err = self._gate_errors.get(frozenset({pa, pb}), 0.02)
            required_errors.append(err)

        if required_errors:
            feats.append(sum(required_errors) / len(required_errors))  # mean
            feats.append(max(required_errors))  # bottleneck (max)
        else:
            feats.append(0.02)
            feats.append(0.02)

        # Subgraph density: edges among mapped qubits / possible edges
        internal_edges = 0
        for pq in phys_qubits:
            for nb in self._adjacency[pq]:
                if nb in phys_set and nb > pq:
                    internal_edges += 1
        possible = n_logical * (n_logical - 1) / 2
        feats.append(internal_edges / possible if possible > 0 else 0.0)

        # Boundary edges: edges from mapped to unmapped neighbors
        boundary = 0
        for pq in phys_qubits:
            for nb in self._adjacency[pq]:
                if nb not in phys_set:
                    boundary += 1
        feats.append(float(boundary))

        # Max shortest path between mapped qubits (BFS)
        feats.append(float(self._max_shortest_path(phys_qubits)))

        # Region mean error: average gate error in 2-hop neighborhood
        neighborhood = set(phys_qubits)
        for pq in phys_qubits:
            for nb in self._adjacency[pq]:
                neighborhood.add(nb)
                for nb2 in self._adjacency.get(nb, []):
                    neighborhood.add(nb2)
        region_errors = []
        for q in neighborhood:
            for nb in self._adjacency[q]:
                if nb in neighborhood and nb > q:
                    region_err = self._gate_errors.get(frozenset({q, nb}), None)
                    if region_err is not None:
                        region_errors.append(region_err)
        feats.append(sum(region_errors) / len(region_errors) if region_errors else 0.02)

        return feats

    def _max_shortest_path(self, qubits: list[int]) -> int:
        """BFS shortest path between all pairs, return the maximum."""
        qset = set(qubits)
        max_dist = 0
        for src in qubits:
            # BFS from src within the subgraph
            dist = {src: 0}
            queue = [src]
            idx = 0
            while idx < len(queue):
                cur = queue[idx]
                idx += 1
                for nb in self._adjacency[cur]:
                    if nb in qset and nb not in dist:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
            for tgt in qubits:
                if tgt in dist:
                    max_dist = max(max_dist, dist[tgt])
                else:
                    # Not connected in subgraph — large penalty
                    max_dist = max(max_dist, len(qubits) * 3)
        return max_dist

    def _random_connected_layout(
        self,
        n_logical: int,
        interactions: dict[tuple[int, int], int],
    ) -> dict[int, int] | None:
        """Sample a random connected subgraph layout."""
        if not self._all_qubits:
            return None

        start = self._rng.choice(self._all_qubits)
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

        # Map logical to physical by degree alignment
        log_degree: Counter[int] = Counter()
        for (a, b), count in interactions.items():
            log_degree[a] += count
            log_degree[b] += count

        logical_order = sorted(range(n_logical), key=lambda q: -log_degree.get(q, 0))

        phys_degree: Counter[int] = Counter()
        sel_set = set(selected)
        for pq in selected:
            for nb in self._adjacency[pq]:
                if nb in sel_set:
                    phys_degree[pq] += 1

        physical_order = sorted(selected, key=lambda q: -phys_degree.get(q, 0))
        return {lg: p for lg, p in zip(logical_order, physical_order, strict=False)}

    def _est_fidelity(self, layout: dict[int, int], n: int) -> float:
        """Estimated GHZ-style fidelity from calibration data."""
        fid = 1.0
        for i in range(n - 1):
            pa, pb = layout[i], layout[i + 1]
            err = self._gate_errors.get(frozenset({pa, pb}), 0.01)
            fid *= 1.0 - err
        for i in range(n):
            qp = self._backend.qubit(layout[i])
            ro = qp.readout_error if qp and qp.readout_error is not None else 0.015
            fid *= 1.0 - ro
        return fid
