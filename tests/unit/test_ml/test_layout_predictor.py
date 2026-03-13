"""Tests for ML layout predictor."""

from __future__ import annotations

import pytest

xgb = pytest.importorskip("xgboost")

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.ml.layout_predictor import MLLayoutPredictor, _WEIGHTS_DIR


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
        GateProperties(gate_type="cx", qubits=(q0, q1), error_rate=0.005)
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


class TestMLLayoutPredictor:
    @pytest.fixture()
    def predictor(self) -> MLLayoutPredictor:
        weights = _WEIGHTS_DIR / "ibm_heron_v1.json"
        if not weights.exists():
            pytest.skip("Model weights not found; run training first")
        return MLLayoutPredictor(model_path=weights)

    def test_returns_valid_qubits(self, predictor: MLLayoutPredictor):
        backend = _make_backend(10)
        candidates = predictor.predict_candidate_qubits(_make_bell(), backend)
        valid_qubits = {qp.qubit_id for qp in backend.qubit_properties}
        assert all(q in valid_qubits for q in candidates)

    def test_returns_enough_candidates(self, predictor: MLLayoutPredictor):
        backend = _make_backend(10)
        circ = _make_bell()
        candidates = predictor.predict_candidate_qubits(circ, backend)
        # Should return at least n_logical qubits
        assert len(candidates) >= circ.n_qubits

    def test_returns_at_most_all_qubits(self, predictor: MLLayoutPredictor):
        backend = _make_backend(10)
        candidates = predictor.predict_candidate_qubits(_make_bell(), backend)
        assert len(candidates) <= 10

    def test_missing_weights_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            MLLayoutPredictor(model_path="/nonexistent/model.json")

    def test_load_bundled(self):
        weights = _WEIGHTS_DIR / "ibm_heron_v1.json"
        if not weights.exists():
            pytest.skip("Model weights not found; run training first")
        pred = MLLayoutPredictor.load_bundled("ibm_heron")
        assert pred.metadata.get("version") is not None

    def test_unknown_backend_family(self):
        with pytest.raises(ValueError, match="No bundled model"):
            MLLayoutPredictor.load_bundled("unknown_vendor")


class TestMLMapperIntegration:
    """Test CalibrationMapper with ML layout predictor."""

    @pytest.fixture()
    def predictor(self) -> MLLayoutPredictor:
        weights = _WEIGHTS_DIR / "ibm_heron_v1.json"
        if not weights.exists():
            pytest.skip("Model weights not found; run training first")
        return MLLayoutPredictor(model_path=weights, min_candidates=6)

    def test_mapper_with_predictor(self, predictor: MLLayoutPredictor):
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
        # Layout should map both logical qubits to valid physical qubits
        assert len(layout) >= 2
        assert all(0 <= v < 10 for v in layout.values())

    def test_ml_mapper_produces_valid_score(self, predictor: MLLayoutPredictor):
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
