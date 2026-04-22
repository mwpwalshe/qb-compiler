"""GNN-based layout prediction for calibration-aware qubit mapping.

Uses a dual-graph neural network architecture:

1. **Device graph encoder** — GCN over the device coupling graph, with
   per-qubit calibration features (T1, T2, readout error, gate error,
   connectivity, T1 asymmetry).
2. **Circuit graph encoder** — GCN over the circuit interaction graph,
   with per-qubit interaction features (degree, weight, depth).
3. **Cross-attention** — aligns circuit qubit needs with device qubit
   capabilities, producing a relevance-weighted device embedding.
4. **Scoring head** — MLP that scores each physical qubit as a
   candidate for the layout.

The GNN captures *structural* information (local connectivity patterns,
neighbourhood quality) that flat XGBoost features miss.  On large
devices with heterogeneous topology (e.g. IBM heavy-hex), this matters.

Requires: ``pip install "qb-compiler[gnn]"``
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qb_compiler.calibration.models.backend_properties import BackendProperties
    from qb_compiler.ir.circuit import QBCircuit

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = Path(__file__).parent / "_weights"

# ── Feature dimensions ────────────────────────────────────────────────

N_DEVICE_FEATURES = 9  # t1, t2, ro, asym, freq, connectivity, gate_err, headroom, nbr_err
N_CIRCUIT_FEATURES = 3  # degree, total_weight, is_measurement_qubit
GNN_HIDDEN_DIM = 32
GNN_N_LAYERS = 3


def _check_torch() -> None:
    """Raise ImportError with helpful message if torch is missing."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for GNN layout prediction. "
            "Install with: pip install 'qb-compiler[gnn]'"
        ) from exc


# ── Feature extraction ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DeviceGraphData:
    """Device coupling graph in tensor form."""

    node_features: list[list[float]]  # [n_qubits, N_DEVICE_FEATURES]
    edge_index: list[list[int]]  # [2, n_edges] (COO format)
    qubit_ids: list[int]  # physical qubit IDs in order


@dataclass(frozen=True, slots=True)
class CircuitGraphData:
    """Circuit interaction graph in tensor form."""

    node_features: list[list[float]]  # [n_logical, N_CIRCUIT_FEATURES]
    edge_index: list[list[int]]  # [2, n_edges] (COO format)


def extract_device_graph(backend: BackendProperties) -> DeviceGraphData:
    """Build device graph tensors from calibration data."""
    from qb_compiler.ml.features import extract_qubit_features

    qubit_ids = sorted({qp.qubit_id for qp in backend.qubit_properties})
    qid_to_idx = {qid: i for i, qid in enumerate(qubit_ids)}

    # Pre-compute adjacency for routing headroom and neighborhood error
    adjacency: dict[int, list[int]] = defaultdict(list)
    gate_errors: dict[frozenset[int], float] = {}
    seen_adj: set[tuple[int, int]] = set()
    for q0, q1 in backend.coupling_map:
        key = (min(q0, q1), max(q0, q1))
        if key not in seen_adj:
            seen_adj.add(key)
            adjacency[q0].append(q1)
            adjacency[q1].append(q0)
    for gp in backend.gate_properties:
        if len(gp.qubits) == 2 and gp.error_rate is not None:
            gate_errors[frozenset(gp.qubits)] = gp.error_rate

    # Find high-connectivity hub qubits (degree >= 3)
    hub_qubits = {q for q in qubit_ids if len(adjacency.get(q, [])) >= 3}

    node_features: list[list[float]] = []
    for qid in qubit_ids:
        qf = extract_qubit_features(qid, backend)

        # routing_headroom: avg shortest path to nearest hub qubit (BFS)
        if qid in hub_qubits:
            routing_headroom = 0.0
        elif hub_qubits:
            # BFS from qid to find nearest hub
            dist_to_hub = float("inf")
            visited_bfs = {qid}
            queue_bfs = [(qid, 0)]
            idx_bfs = 0
            while idx_bfs < len(queue_bfs):
                cur, d = queue_bfs[idx_bfs]
                idx_bfs += 1
                if cur in hub_qubits:
                    dist_to_hub = d
                    break
                for nb in adjacency.get(cur, []):
                    if nb not in visited_bfs:
                        visited_bfs.add(nb)
                        queue_bfs.append((nb, d + 1))
            routing_headroom = dist_to_hub if dist_to_hub < float("inf") else 5.0
        else:
            routing_headroom = 5.0

        # neighborhood_error: mean gate error within 2-hop radius
        neighborhood = {qid}
        for nb in adjacency.get(qid, []):
            neighborhood.add(nb)
            for nb2 in adjacency.get(nb, []):
                neighborhood.add(nb2)
        nb_errors = []
        for q in neighborhood:
            for nb in adjacency.get(q, []):
                if nb in neighborhood and nb > q:
                    err = gate_errors.get(frozenset({q, nb}))
                    if err is not None:
                        nb_errors.append(err)
        neighborhood_error = sum(nb_errors) / len(nb_errors) if nb_errors else 0.02

        node_features.append(
            [
                qf.t1_us / 300.0,
                qf.t2_us / 300.0,
                qf.readout_error * 10.0,
                qf.t1_asymmetry_penalty * 100.0,
                qf.frequency_ghz / 5.5,
                qf.connectivity_degree / 4.0,
                qf.mean_adjacent_gate_error * 100.0,
                routing_headroom / 5.0,  # normalise
                neighborhood_error * 100.0,  # scale up
            ]
        )

    # Build edge_index (COO format, undirected → add both directions)
    src_list: list[int] = []
    dst_list: list[int] = []
    seen: set[tuple[int, int]] = set()
    for q0, q1 in backend.coupling_map:
        if q0 in qid_to_idx and q1 in qid_to_idx:
            edge = (min(q0, q1), max(q0, q1))
            if edge not in seen:
                seen.add(edge)
                i0, i1 = qid_to_idx[q0], qid_to_idx[q1]
                src_list.extend([i0, i1])
                dst_list.extend([i1, i0])

    return DeviceGraphData(
        node_features=node_features,
        edge_index=[src_list, dst_list],
        qubit_ids=qubit_ids,
    )


def extract_circuit_graph(circuit: QBCircuit) -> CircuitGraphData:
    """Build circuit interaction graph tensors."""
    from qb_compiler.ir.operations import QBGate, QBMeasure

    n = circuit.n_qubits

    # Build interaction counts
    interactions: dict[tuple[int, int], int] = defaultdict(int)
    degree: dict[int, int] = defaultdict(int)
    meas_qubits: set[int] = set()

    for op in circuit.iter_ops():
        if isinstance(op, QBGate) and op.num_qubits >= 2:
            q_sorted = sorted(op.qubits[:2])
            pair = (q_sorted[0], q_sorted[1])
            interactions[pair] += 1
            degree[q_sorted[0]] += 1
            degree[q_sorted[1]] += 1
        elif isinstance(op, QBMeasure):
            meas_qubits.add(op.qubit)

    # Node features
    max_degree = max(degree.values()) if degree else 1
    node_features: list[list[float]] = []
    for q in range(n):
        d = degree.get(q, 0)
        # Total interaction weight for this qubit
        total_w = sum(c for (a, b), c in interactions.items() if a == q or b == q)
        node_features.append(
            [
                d / max(max_degree, 1),
                total_w / max(sum(interactions.values()), 1) if interactions else 0.0,
                1.0 if q in meas_qubits else 0.0,
            ]
        )

    # Edge index
    src_list: list[int] = []
    dst_list: list[int] = []
    for a, b in interactions:
        src_list.extend([a, b])
        dst_list.extend([b, a])

    return CircuitGraphData(
        node_features=node_features,
        edge_index=[src_list, dst_list],
    )


# ── GNN Model ────────────────────────────────────────────────────────


def _build_model() -> Any:
    """Construct the GNN model (requires torch)."""
    _check_torch()
    import torch
    import torch.nn as nn

    class GCNLayer(nn.Module):
        """Simple Graph Convolutional layer (Kipf & Welling, 2017)."""

        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            device = x.device

            if edge_index.size(1) == 0:
                # No edges: just apply linear (self-loop only)
                return self.linear(x)  # type: ignore[no-any-return]

            # Add self-loops
            self_loops = torch.arange(n, device=device).unsqueeze(0).expand(2, -1)
            edge_idx = torch.cat([edge_index, self_loops], dim=1)

            # Compute degree for normalisation
            row = edge_idx[0]
            deg = torch.zeros(n, device=device)
            deg.scatter_add_(0, row, torch.ones(edge_idx.size(1), device=device))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

            # Transform first, then aggregate
            xw = self.linear(x)  # [n, out_dim]
            h = xw.size(1)

            src, dst = edge_idx[0], edge_idx[1]
            norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]  # [n_edges]

            # Weighted messages
            messages = xw[src] * norm.unsqueeze(1)  # [n_edges, h]

            # Scatter-add into destination nodes
            out = torch.zeros(n, h, device=device)
            out.scatter_add_(0, dst.unsqueeze(1).expand(-1, h), messages)
            return out

    class GNNLayoutModel(nn.Module):
        """Dual-graph GNN for layout prediction.

        Device graph encoder → device embeddings [n_phys, hidden]
        Circuit graph encoder → circuit embedding [hidden] (global pool)
        Cross-attention → weighted device embeddings
        Scoring MLP → per-qubit score [n_phys, 1]
        """

        def __init__(
            self,
            device_in: int = N_DEVICE_FEATURES,
            circuit_in: int = N_CIRCUIT_FEATURES,
            hidden: int = GNN_HIDDEN_DIM,
            n_layers: int = GNN_N_LAYERS,
        ) -> None:
            super().__init__()
            # Device graph encoder
            self.device_layers = nn.ModuleList()
            self.device_layers.append(GCNLayer(device_in, hidden))
            for _ in range(n_layers - 1):
                self.device_layers.append(GCNLayer(hidden, hidden))

            # Circuit graph encoder
            self.circuit_layers = nn.ModuleList()
            self.circuit_layers.append(GCNLayer(circuit_in, hidden))
            for _ in range(n_layers - 1):
                self.circuit_layers.append(GCNLayer(hidden, hidden))

            # Cross-attention: project circuit embedding to query space
            self.query_proj = nn.Linear(hidden, hidden)
            self.key_proj = nn.Linear(hidden, hidden)

            # Scoring head
            self.score_head = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

            self.relu = nn.ReLU()

        def forward(
            self,
            device_x: torch.Tensor,
            device_edge: torch.Tensor,
            circuit_x: torch.Tensor,
            circuit_edge: torch.Tensor,
        ) -> torch.Tensor:
            """Score each physical qubit for layout suitability.

            Returns shape [n_physical_qubits] with logits (higher = better).
            """
            # Encode device graph
            h_dev = device_x
            for layer in self.device_layers:
                h_dev = self.relu(layer(h_dev, device_edge))

            # Encode circuit graph
            h_circ = circuit_x
            for layer in self.circuit_layers:
                h_circ = self.relu(layer(h_circ, circuit_edge))

            # Global pool circuit → single vector
            circuit_global = h_circ.mean(dim=0, keepdim=True)  # [1, hidden]

            # Cross-attention: how relevant is each device qubit to this circuit?
            query = self.query_proj(circuit_global)  # [1, hidden]
            keys = self.key_proj(h_dev)  # [n_phys, hidden]
            attn_scores = (keys @ query.T) / (keys.size(-1) ** 0.5)  # [n_phys, 1]
            attn_weights = torch.softmax(attn_scores, dim=0)

            # Weighted device embeddings
            h_dev_weighted = h_dev * attn_weights

            # Concatenate device embedding with broadcast circuit embedding
            circuit_expanded = circuit_global.expand(h_dev.size(0), -1)
            combined = torch.cat([h_dev_weighted, circuit_expanded], dim=1)

            # Score each physical qubit
            scores = self.score_head(combined).squeeze(-1)
            return scores  # type: ignore[no-any-return]

    return GNNLayoutModel()


# ── Inference wrapper ─────────────────────────────────────────────────


class GNNLayoutPredictor:
    """Predict promising physical qubits using a trained GNN.

    Drop-in replacement for :class:`MLLayoutPredictor` with the same
    ``predict_candidate_qubits`` interface, but uses graph structure
    instead of flat features.

    Parameters
    ----------
    model_path :
        Path to saved model weights (``.pt`` file).
        If None, attempts to load bundled weights.
    top_k_factor :
        Return ``n_logical * top_k_factor`` candidate qubits.
    min_candidates :
        Minimum number of candidates to return.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        top_k_factor: float = 3.0,
        min_candidates: int = 20,
    ) -> None:
        _check_torch()
        import torch

        self._top_k_factor = top_k_factor
        self._min_candidates = min_candidates

        if model_path is None:
            model_path = _WEIGHTS_DIR / "gnn_heron_v1.pt"

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"GNN model weights not found at {self._model_path}. "
                f"Train a model with: python -m qb_compiler.ml.train_gnn"
            )

        self._model = _build_model()
        state = torch.load(str(self._model_path), map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

        # Load metadata if available
        meta_path = self._model_path.with_suffix(".meta.json")
        self._metadata: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

        logger.info(
            "Loaded GNN layout predictor from %s (version=%s)",
            self._model_path.name,
            self._metadata.get("version", "unknown"),
        )

    def predict_candidate_qubits(
        self,
        circuit: QBCircuit,
        backend: BackendProperties,
    ) -> list[int]:
        """Return physical qubits most likely to appear in good layouts.

        Same interface as :meth:`MLLayoutPredictor.predict_candidate_qubits`.
        """
        import torch

        # Extract graph data
        dev_data = extract_device_graph(backend)
        circ_data = extract_circuit_graph(circuit)

        # Convert to tensors
        dev_x = torch.tensor(dev_data.node_features, dtype=torch.float32)
        circ_x = torch.tensor(circ_data.node_features, dtype=torch.float32)

        if dev_data.edge_index[0]:
            dev_edge = torch.tensor(dev_data.edge_index, dtype=torch.long)
        else:
            dev_edge = torch.zeros((2, 0), dtype=torch.long)

        if circ_data.edge_index[0]:
            circ_edge = torch.tensor(circ_data.edge_index, dtype=torch.long)
        else:
            circ_edge = torch.zeros((2, 0), dtype=torch.long)

        # Inference
        with torch.no_grad():
            scores = self._model(dev_x, dev_edge, circ_x, circ_edge)
            probabilities = torch.sigmoid(scores).numpy()

        # Sort by score (highest first)
        scored = sorted(
            zip(dev_data.qubit_ids, probabilities.tolist(), strict=False),
            key=lambda x: -x[1],
        )

        # Determine how many to return
        n_logical = circuit.n_qubits
        top_k = max(
            int(n_logical * self._top_k_factor),
            self._min_candidates,
        )
        top_k = min(top_k, len(scored))

        candidates = [qid for qid, _prob in scored[:top_k]]

        logger.debug(
            "GNN predictor: %d candidates from %d physical qubits "
            "(top score=%.3f, cutoff score=%.3f)",
            len(candidates),
            len(dev_data.qubit_ids),
            scored[0][1] if scored else 0,
            scored[top_k - 1][1] if scored else 0,
        )

        return candidates

    @classmethod
    def load_bundled(cls, backend_family: str = "ibm_heron") -> GNNLayoutPredictor:
        """Load pre-trained GNN model for a backend family."""
        model_map = {
            "ibm_heron": "gnn_heron_v1.pt",
        }
        filename = model_map.get(backend_family)
        if filename is None:
            raise ValueError(
                f"No bundled GNN model for {backend_family!r}. Available: {list(model_map.keys())}"
            )
        return cls(model_path=_WEIGHTS_DIR / filename)

    @property
    def metadata(self) -> dict:
        """Training metadata (version, AUC, architecture, etc.)."""
        return dict(self._metadata)

    @property
    def model(self) -> Any:
        """The underlying PyTorch model (for inspection/export)."""
        return self._model


# ── Training ──────────────────────────────────────────────────────────


def train_gnn_model(
    output_path: str | Path | None = None,
    n_trials: int = 200,
    top_fraction: float = 0.1,
    n_epochs: int = 50,
    lr: float = 0.005,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Train the GNN layout predictor from calibration snapshots.

    Uses the same training data pipeline as the XGBoost model (Phase 2)
    but trains the dual-graph GNN instead.

    Returns training metadata dict.
    """
    _check_torch()
    import torch
    import torch.nn as nn

    from qb_compiler.ir.circuit import QBCircuit
    from qb_compiler.ir.operations import QBGate
    from qb_compiler.ml.data_generator import TrainingDataGenerator

    torch.manual_seed(seed)

    if output_path is None:
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = _WEIGHTS_DIR / "gnn_heron_v1.pt"
    output_path = Path(output_path)

    # Load calibration data
    from qb_compiler.compiler import _load_calibration_fixture

    cal = _load_calibration_fixture("ibm_fez")
    if cal is None:
        cal = _load_calibration_fixture("ibm_torino")
    if cal is None:
        raise RuntimeError("No calibration fixture found for training")

    # Build circuits
    def _ghz(n: int) -> QBCircuit:
        c = QBCircuit(n_qubits=n, n_clbits=0)
        c.add_gate(QBGate("h", (0,)))
        for i in range(n - 1):
            c.add_gate(QBGate("cx", (i, i + 1)))
        return c

    def _qaoa_ring(n: int) -> QBCircuit:
        c = QBCircuit(n_qubits=n, n_clbits=0)
        for i in range(n - 1):
            c.add_gate(QBGate("cx", (i, i + 1)))
            c.add_gate(QBGate("rz", (i + 1,), (0.5,)))
            c.add_gate(QBGate("cx", (i, i + 1)))
        for i in range(n):
            c.add_gate(QBGate("rx", (i,), (0.7,)))
        return c

    def _star(n: int) -> QBCircuit:
        c = QBCircuit(n_qubits=n, n_clbits=0)
        for i in range(1, n):
            c.add_gate(QBGate("cx", (0, i)))
        return c

    circuits: list[QBCircuit] = []
    for n in range(2, 10):
        circuits.append(_ghz(n))
    for n in range(3, 8):
        circuits.append(_qaoa_ring(n))
    for n in range(3, 6):
        circuits.append(_star(n))

    if verbose:
        print(f"Training GNN on {len(circuits)} circuits, {cal.n_qubits} qubits")

    # Generate training labels using calibration-aware scorer
    gen = TrainingDataGenerator(cal, n_trials=n_trials, top_fraction=top_fraction, seed=seed)

    # For GNN training, we need per-circuit graph data + labels
    # We'll generate labels per qubit per circuit, then train
    model = _build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    dev_data = extract_device_graph(cal)
    dev_x = torch.tensor(dev_data.node_features, dtype=torch.float32)
    if dev_data.edge_index[0]:
        dev_edge = torch.tensor(dev_data.edge_index, dtype=torch.long)
    else:
        dev_edge = torch.zeros((2, 0), dtype=torch.long)

    qid_to_idx = {qid: i for i, qid in enumerate(dev_data.qubit_ids)}

    # Prepare per-circuit training data
    training_pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for circuit in circuits:
        batch = gen.generate_from_circuit(circuit)
        if not batch.labels:
            continue

        circ_data = extract_circuit_graph(circuit)
        circ_x = torch.tensor(circ_data.node_features, dtype=torch.float32)
        if circ_data.edge_index[0]:
            circ_edge = torch.tensor(circ_data.edge_index, dtype=torch.long)
        else:
            circ_edge = torch.zeros((2, 0), dtype=torch.long)

        # Labels: one per physical qubit
        n_phys = len(dev_data.qubit_ids)
        labels = torch.zeros(n_phys, dtype=torch.float32)

        # The batch has labels for all qubits in order of gen._all_qubits
        # Map them to our device graph ordering
        for i, qid in enumerate(gen._all_qubits):
            if qid in qid_to_idx and i < len(batch.labels):
                labels[qid_to_idx[qid]] = float(batch.labels[i])

        training_pairs.append((circ_x, circ_edge, labels))

    if not training_pairs:
        raise RuntimeError("No training data generated")

    if verbose:
        print(f"Generated training data for {len(training_pairs)} circuits")

    # Training loop
    model.train()
    best_loss = float("inf")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for circ_x, circ_edge, labels in training_pairs:
            optimizer.zero_grad()
            scores = model(dev_x, dev_edge, circ_x, circ_edge)
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(training_pairs)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.4f}")

    # Evaluate (compute AUC on training data as baseline)
    model.eval()
    all_preds: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for circ_x, circ_edge, labels in training_pairs:
            scores = model(dev_x, dev_edge, circ_x, circ_edge)
            probs = torch.sigmoid(scores)
            all_preds.extend(probs.tolist())
            all_labels.extend(labels.int().tolist())

    # Simple AUC calculation
    auc = _compute_auc(all_labels, all_preds)

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output_path))

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    model_size = output_path.stat().st_size

    metadata = {
        "version": "1.0.0",
        "architecture": "dual_graph_gcn",
        "hidden_dim": GNN_HIDDEN_DIM,
        "n_layers": GNN_N_LAYERS,
        "device_features": N_DEVICE_FEATURES,
        "circuit_features": N_CIRCUIT_FEATURES,
        "n_parameters": n_params,
        "model_size_bytes": model_size,
        "training_auc": round(auc, 4),
        "best_loss": round(best_loss, 6),
        "n_epochs": n_epochs,
        "n_circuits": len(training_pairs),
        "n_trials_per_circuit": n_trials,
        "backend": cal.backend,
    }

    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\nGNN model saved to {output_path}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Size: {model_size / 1024:.1f} KB")
        print(f"  Training AUC: {auc:.4f}")

    return metadata


def _compute_auc(labels: list[int], predictions: list[float]) -> float:
    """Compute AUC-ROC without sklearn dependency."""
    pairs = sorted(zip(predictions, labels, strict=False), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for _pred, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr

    return auc
