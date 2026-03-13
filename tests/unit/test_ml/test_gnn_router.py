"""Tests for GNN layout predictor (Phase 3)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.ml.gnn_router import (
    GNN_HIDDEN_DIM,
    GNNLayoutPredictor,
    N_CIRCUIT_FEATURES,
    N_DEVICE_FEATURES,
    _WEIGHTS_DIR,
    _build_model,
    _compute_auc,
    extract_circuit_graph,
    extract_device_graph,
)


# ── fixtures ──────────────────────────────────────────────────────────


def _make_backend(n: int = 10) -> BackendProperties:
    qubits = [
        QubitProperties(
            qubit_id=i,
            t1_us=100.0 + i * 10,
            t2_us=80.0 + i * 5,
            readout_error=0.01 + i * 0.002,
            frequency_ghz=5.0 + i * 0.05,
        )
        for i in range(n)
    ]
    coupling = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]
    gates = [
        GateProperties(gate_type="cx", qubits=(q0, q1), error_rate=0.005 + 0.001 * q0)
        for q0, q1 in coupling
    ]
    return BackendProperties(
        backend="test",
        provider="test",
        n_qubits=n,
        basis_gates=("cx",),
        coupling_map=coupling,
        qubit_properties=qubits,
        gate_properties=gates,
        timestamp="2026-01-01",
    )


def _make_bell() -> QBCircuit:
    c = QBCircuit(n_qubits=2, n_clbits=0)
    c.add_gate(QBGate("h", (0,)))
    c.add_gate(QBGate("cx", (0, 1)))
    return c


def _make_ghz(n: int) -> QBCircuit:
    c = QBCircuit(n_qubits=n, n_clbits=0)
    c.add_gate(QBGate("h", (0,)))
    for i in range(n - 1):
        c.add_gate(QBGate("cx", (i, i + 1)))
    return c


def _make_single_qubit() -> QBCircuit:
    """Circuit with only single-qubit gates (no interactions)."""
    c = QBCircuit(n_qubits=2, n_clbits=0)
    c.add_gate(QBGate("h", (0,)))
    c.add_gate(QBGate("x", (1,)))
    return c


# ── feature extraction tests ─────────────────────────────────────────


class TestDeviceGraphExtraction:
    def test_node_count(self):
        backend = _make_backend(10)
        data = extract_device_graph(backend)
        assert len(data.node_features) == 10
        assert len(data.qubit_ids) == 10

    def test_feature_dimension(self):
        backend = _make_backend(5)
        data = extract_device_graph(backend)
        assert all(len(f) == N_DEVICE_FEATURES for f in data.node_features)

    def test_edge_count_undirected(self):
        backend = _make_backend(5)
        data = extract_device_graph(backend)
        # 4 edges in chain, each direction = 8 entries in COO
        assert len(data.edge_index[0]) == 8
        assert len(data.edge_index[1]) == 8

    def test_qubit_ids_sorted(self):
        backend = _make_backend(10)
        data = extract_device_graph(backend)
        assert data.qubit_ids == list(range(10))

    def test_features_are_normalised(self):
        backend = _make_backend(5)
        data = extract_device_graph(backend)
        # T1 normalised by 300 → 100/300 ≈ 0.33 for first qubit
        assert 0.1 < data.node_features[0][0] < 2.0


class TestCircuitGraphExtraction:
    def test_bell_circuit(self):
        data = extract_circuit_graph(_make_bell())
        assert len(data.node_features) == 2
        assert all(len(f) == N_CIRCUIT_FEATURES for f in data.node_features)
        # One CX → 2 edges (both directions)
        assert len(data.edge_index[0]) == 2

    def test_ghz_5(self):
        data = extract_circuit_graph(_make_ghz(5))
        assert len(data.node_features) == 5
        # 4 CX gates → 4 edges → 8 entries in COO
        assert len(data.edge_index[0]) == 8

    def test_no_interaction_circuit(self):
        data = extract_circuit_graph(_make_single_qubit())
        assert len(data.node_features) == 2
        # No 2Q gates → no edges
        assert len(data.edge_index[0]) == 0

    def test_degree_normalisation(self):
        data = extract_circuit_graph(_make_ghz(3))
        # Qubit 1 has degree 2 (max), so normalised = 1.0
        assert data.node_features[1][0] == pytest.approx(1.0)
        # Qubit 0 has degree 1, so normalised = 0.5
        assert data.node_features[0][0] == pytest.approx(0.5)


# ── model architecture tests ─────────────────────────────────────────


class TestGNNModel:
    def test_build_model(self):
        model = _build_model()
        assert model is not None

    def test_parameter_count(self):
        model = _build_model()
        n_params = sum(p.numel() for p in model.parameters())
        # Should be reasonable (< 50K params for a small model)
        assert n_params < 50_000
        assert n_params > 100  # sanity

    def test_forward_pass(self):
        model = _build_model()
        backend = _make_backend(10)
        circuit = _make_bell()

        dev_data = extract_device_graph(backend)
        circ_data = extract_circuit_graph(circuit)

        dev_x = torch.tensor(dev_data.node_features, dtype=torch.float32)
        circ_x = torch.tensor(circ_data.node_features, dtype=torch.float32)
        dev_edge = torch.tensor(dev_data.edge_index, dtype=torch.long)
        circ_edge = torch.tensor(circ_data.edge_index, dtype=torch.long)

        with torch.no_grad():
            scores = model(dev_x, dev_edge, circ_x, circ_edge)

        assert scores.shape == (10,)
        assert torch.isfinite(scores).all()

    def test_forward_no_circuit_edges(self):
        """Model handles circuits with no 2Q gates."""
        model = _build_model()
        backend = _make_backend(5)
        circuit = _make_single_qubit()

        dev_data = extract_device_graph(backend)
        circ_data = extract_circuit_graph(circuit)

        dev_x = torch.tensor(dev_data.node_features, dtype=torch.float32)
        circ_x = torch.tensor(circ_data.node_features, dtype=torch.float32)
        dev_edge = torch.tensor(dev_data.edge_index, dtype=torch.long)
        circ_edge = torch.zeros((2, 0), dtype=torch.long)

        with torch.no_grad():
            scores = model(dev_x, dev_edge, circ_x, circ_edge)

        assert scores.shape == (5,)
        assert torch.isfinite(scores).all()

    def test_different_circuits_different_scores(self):
        """Model produces different outputs for structurally different circuits."""
        model = _build_model()
        backend = _make_backend(10)

        # Two different circuits
        bell = _make_bell()
        ghz5 = _make_ghz(5)

        dev_data = extract_device_graph(backend)
        dev_x = torch.tensor(dev_data.node_features, dtype=torch.float32)
        dev_edge = torch.tensor(dev_data.edge_index, dtype=torch.long)

        circ1 = extract_circuit_graph(bell)
        circ2 = extract_circuit_graph(ghz5)

        with torch.no_grad():
            scores1 = model(
                dev_x, dev_edge,
                torch.tensor(circ1.node_features, dtype=torch.float32),
                torch.tensor(circ1.edge_index, dtype=torch.long),
            )
            scores2 = model(
                dev_x, dev_edge,
                torch.tensor(circ2.node_features, dtype=torch.float32),
                torch.tensor(circ2.edge_index, dtype=torch.long),
            )

        # Scores should differ (model is randomly initialised)
        assert not torch.allclose(scores1, scores2)

    def test_model_is_trainable(self):
        """Verify gradients flow through the full model."""
        model = _build_model()
        backend = _make_backend(5)
        circuit = _make_bell()

        dev_data = extract_device_graph(backend)
        circ_data = extract_circuit_graph(circuit)

        dev_x = torch.tensor(dev_data.node_features, dtype=torch.float32)
        circ_x = torch.tensor(circ_data.node_features, dtype=torch.float32)
        dev_edge = torch.tensor(dev_data.edge_index, dtype=torch.long)
        circ_edge = torch.tensor(circ_data.edge_index, dtype=torch.long)

        scores = model(dev_x, dev_edge, circ_x, circ_edge)
        target = torch.zeros(5)
        target[0] = 1.0  # qubit 0 is "good"

        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, target)
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


# ── predictor tests ───────────────────────────────────────────────────


class TestGNNLayoutPredictor:
    @pytest.fixture()
    def predictor(self, tmp_path) -> GNNLayoutPredictor:
        """Create a predictor with random weights for testing."""
        model = _build_model()
        weights_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), str(weights_path))
        return GNNLayoutPredictor(model_path=weights_path)

    def test_returns_valid_qubits(self, predictor: GNNLayoutPredictor):
        backend = _make_backend(10)
        candidates = predictor.predict_candidate_qubits(_make_bell(), backend)
        valid_qubits = {qp.qubit_id for qp in backend.qubit_properties}
        assert all(q in valid_qubits for q in candidates)

    def test_returns_enough_candidates(self, predictor: GNNLayoutPredictor):
        backend = _make_backend(10)
        circ = _make_bell()
        candidates = predictor.predict_candidate_qubits(circ, backend)
        assert len(candidates) >= circ.n_qubits

    def test_returns_at_most_all_qubits(self, predictor: GNNLayoutPredictor):
        backend = _make_backend(10)
        candidates = predictor.predict_candidate_qubits(_make_bell(), backend)
        assert len(candidates) <= 10

    def test_missing_weights_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            GNNLayoutPredictor(model_path="/nonexistent/model.pt")

    def test_unknown_backend_family(self):
        with pytest.raises(ValueError, match="No bundled GNN model"):
            GNNLayoutPredictor.load_bundled("unknown_vendor")

    def test_metadata(self, predictor: GNNLayoutPredictor):
        # Random weights have no metadata
        meta = predictor.metadata
        assert isinstance(meta, dict)

    def test_model_property(self, predictor: GNNLayoutPredictor):
        assert predictor.model is not None


# ── mapper integration tests ─────────────────────────────────────────


class TestGNNMapperIntegration:
    @pytest.fixture()
    def predictor(self, tmp_path) -> GNNLayoutPredictor:
        model = _build_model()
        weights_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), str(weights_path))
        return GNNLayoutPredictor(model_path=weights_path, min_candidates=6)

    def test_mapper_with_gnn_predictor(self, predictor: GNNLayoutPredictor):
        from qb_compiler.passes.mapping.calibration_mapper import (
            CalibrationMapper,
            CalibrationMapperConfig,
        )

        backend = _make_backend(10)
        mapper = CalibrationMapper(
            backend,
            config=CalibrationMapperConfig(max_candidates=100, vf2_call_limit=5000),
            layout_predictor=predictor,
        )
        circ = _make_bell()
        ctx: dict = {}
        result = mapper.run(circ, ctx)
        assert "initial_layout" in ctx
        layout = ctx["initial_layout"]
        assert len(layout) >= 2
        assert all(0 <= v < 10 for v in layout.values())

    def test_gnn_mapper_ghz(self, predictor: GNNLayoutPredictor):
        from qb_compiler.passes.mapping.calibration_mapper import (
            CalibrationMapper,
            CalibrationMapperConfig,
        )

        backend = _make_backend(10)
        mapper = CalibrationMapper(
            backend,
            config=CalibrationMapperConfig(max_candidates=100, vf2_call_limit=5000),
            layout_predictor=predictor,
        )
        ctx: dict = {}
        mapper.run(_make_ghz(3), ctx)
        assert ctx["calibration_score"] > 0
        assert "score_breakdown" in ctx


# ── utility tests ─────────────────────────────────────────────────────


class TestAUCComputation:
    def test_perfect_auc(self):
        labels = [1, 1, 0, 0]
        preds = [0.9, 0.8, 0.2, 0.1]
        assert _compute_auc(labels, preds) == pytest.approx(1.0)

    def test_inverse_auc(self):
        labels = [0, 0, 1, 1]
        preds = [0.9, 0.8, 0.2, 0.1]
        auc = _compute_auc(labels, preds)
        assert auc == pytest.approx(0.0)

    def test_all_same_label(self):
        labels = [1, 1, 1]
        preds = [0.9, 0.8, 0.7]
        auc = _compute_auc(labels, preds)
        assert auc == pytest.approx(0.5)
