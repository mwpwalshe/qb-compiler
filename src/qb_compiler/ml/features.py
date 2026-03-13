"""Feature extraction for ML layout prediction.

Extracts numeric feature vectors from circuits and backend calibration
data.  These features are used by :class:`MLLayoutPredictor` to score
physical qubits without running VF2.

No ML dependencies required — this module uses only numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qb_compiler.calibration.models.backend_properties import BackendProperties
    from qb_compiler.ir.circuit import QBCircuit


# ── feature names (order must match to_feature_vector) ────────────────

_CIRCUIT_FEATURE_NAMES = [
    "n_logical_qubits",
    "n_gates",
    "n_2q_gates",
    "depth",
    "interaction_density",
    "max_interaction_degree",
    "mean_interaction_weight",
    "max_interaction_weight",
]

_QUBIT_FEATURE_NAMES = [
    "t1_us",
    "t2_us",
    "readout_error",
    "t1_asymmetry_penalty",
    "connectivity_degree",
    "min_adjacent_gate_error",
    "mean_adjacent_gate_error",
    "max_adjacent_gate_error",
    "frequency_ghz",
]


def feature_names() -> list[str]:
    """Return ordered feature names matching :func:`to_feature_vector`."""
    return _CIRCUIT_FEATURE_NAMES + _QUBIT_FEATURE_NAMES


N_FEATURES = len(_CIRCUIT_FEATURE_NAMES) + len(_QUBIT_FEATURE_NAMES)


# ── circuit features ─────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CircuitFeatures:
    """Numeric features describing circuit structure."""

    n_logical_qubits: int
    n_gates: int
    n_2q_gates: int
    depth: int
    interaction_density: float
    max_interaction_degree: int
    mean_interaction_weight: float
    max_interaction_weight: int

    def to_list(self) -> list[float]:
        return [
            float(self.n_logical_qubits),
            float(self.n_gates),
            float(self.n_2q_gates),
            float(self.depth),
            self.interaction_density,
            float(self.max_interaction_degree),
            self.mean_interaction_weight,
            float(self.max_interaction_weight),
        ]


@dataclass(frozen=True, slots=True)
class QubitFeatures:
    """Numeric features describing a single physical qubit."""

    t1_us: float
    t2_us: float
    readout_error: float
    t1_asymmetry_penalty: float
    connectivity_degree: int
    min_adjacent_gate_error: float
    mean_adjacent_gate_error: float
    max_adjacent_gate_error: float
    frequency_ghz: float

    def to_list(self) -> list[float]:
        return [
            self.t1_us,
            self.t2_us,
            self.readout_error,
            self.t1_asymmetry_penalty,
            float(self.connectivity_degree),
            self.min_adjacent_gate_error,
            self.mean_adjacent_gate_error,
            self.max_adjacent_gate_error,
            self.frequency_ghz,
        ]


# ── extraction functions ──────────────────────────────────────────────

def extract_circuit_features(circuit: QBCircuit) -> CircuitFeatures:
    """Extract circuit-level features from a QBCircuit (IR level)."""
    from collections import defaultdict

    from qb_compiler.ir.operations import QBGate

    n_qubits = circuit.n_qubits
    n_gates = circuit.gate_count
    n_2q = circuit.two_qubit_gate_count
    depth = circuit.depth

    # Build interaction graph
    interactions: dict[tuple[int, int], int] = defaultdict(int)
    for op in circuit.iter_ops():
        if isinstance(op, QBGate) and op.num_qubits >= 2:
            q_sorted = sorted(op.qubits[:2])
            interactions[(q_sorted[0], q_sorted[1])] += 1

    # Interaction graph density
    max_possible_edges = n_qubits * (n_qubits - 1) / 2
    density = len(interactions) / max_possible_edges if max_possible_edges > 0 else 0.0

    # Degree of each qubit in the interaction graph
    degree: dict[int, int] = defaultdict(int)
    for (a, b) in interactions:
        degree[a] += 1
        degree[b] += 1
    max_degree = max(degree.values()) if degree else 0

    # Interaction weights
    weights = list(interactions.values())
    mean_weight = sum(weights) / len(weights) if weights else 0.0
    max_weight = max(weights) if weights else 0

    return CircuitFeatures(
        n_logical_qubits=n_qubits,
        n_gates=n_gates,
        n_2q_gates=n_2q,
        depth=depth,
        interaction_density=density,
        max_interaction_degree=max_degree,
        mean_interaction_weight=mean_weight,
        max_interaction_weight=max_weight,
    )


def extract_qubit_features(
    qubit_id: int,
    backend: BackendProperties,
) -> QubitFeatures:
    """Extract features for a single physical qubit from calibration data."""
    # Find the QubitProperties for this qubit
    qp = None
    for q in backend.qubit_properties:
        if q.qubit_id == qubit_id:
            qp = q
            break

    t1 = qp.t1_us if (qp and qp.t1_us) else 100.0
    t2 = qp.t2_us if (qp and qp.t2_us) else 80.0
    ro = qp.readout_error if (qp and qp.readout_error is not None) else 0.015
    asym = qp.t1_asymmetry_penalty if qp else 0.0
    freq = qp.frequency_ghz if (qp and qp.frequency_ghz) else 5.0

    # Connectivity and adjacent gate errors
    neighbors: set[int] = set()
    adjacent_errors: list[float] = []
    for gp in backend.gate_properties:
        if len(gp.qubits) == 2 and gp.error_rate is not None:
            if gp.qubits[0] == qubit_id:
                neighbors.add(gp.qubits[1])
                adjacent_errors.append(gp.error_rate)
            elif gp.qubits[1] == qubit_id:
                neighbors.add(gp.qubits[0])
                adjacent_errors.append(gp.error_rate)

    degree = len(neighbors)

    if adjacent_errors:
        min_err = min(adjacent_errors)
        mean_err = sum(adjacent_errors) / len(adjacent_errors)
        max_err = max(adjacent_errors)
    else:
        min_err = mean_err = max_err = 0.02  # default

    return QubitFeatures(
        t1_us=t1,
        t2_us=t2,
        readout_error=ro,
        t1_asymmetry_penalty=asym,
        connectivity_degree=degree,
        min_adjacent_gate_error=min_err,
        mean_adjacent_gate_error=mean_err,
        max_adjacent_gate_error=max_err,
        frequency_ghz=freq,
    )


def to_feature_vector(
    circuit_feats: CircuitFeatures,
    qubit_feats: QubitFeatures,
) -> list[float]:
    """Flatten circuit + qubit features into a single numeric vector."""
    return circuit_feats.to_list() + qubit_feats.to_list()


def build_feature_matrix(
    circuit: QBCircuit,
    backend: BackendProperties,
) -> tuple[list[list[float]], list[int]]:
    """Build feature matrix for all physical qubits.

    Returns (feature_matrix, qubit_ids) where feature_matrix[i]
    corresponds to qubit_ids[i].
    """
    circ_feats = extract_circuit_features(circuit)

    qubit_ids: list[int] = sorted(
        {qp.qubit_id for qp in backend.qubit_properties}
    )

    matrix: list[list[float]] = []
    for qid in qubit_ids:
        qf = extract_qubit_features(qid, backend)
        matrix.append(to_feature_vector(circ_feats, qf))

    return matrix, qubit_ids
