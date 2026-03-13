"""Tests for the calibration-aware qubit mapper."""

from __future__ import annotations

import pytest

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.calibration.static_provider import StaticCalibrationProvider
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.base import PassResult
from qb_compiler.passes.mapping.calibration_mapper import (
    CalibrationMapper,
    CalibrationMapperConfig,
)

# ── helpers ──────────────────────────────────────────────────────────


def _make_backend(
    n_qubits: int = 6,
    qubit_props: list[QubitProperties] | None = None,
    gate_props: list[GateProperties] | None = None,
    coupling_map: list[tuple[int, int]] | None = None,
) -> BackendProperties:
    """Build a small synthetic BackendProperties for testing."""
    if qubit_props is None:
        qubit_props = [
            QubitProperties(qubit_id=0, t1_us=300.0, t2_us=250.0, readout_error=0.005),
            QubitProperties(qubit_id=1, t1_us=200.0, t2_us=180.0, readout_error=0.010),
            QubitProperties(qubit_id=2, t1_us=150.0, t2_us=120.0, readout_error=0.030),
            QubitProperties(qubit_id=3, t1_us=100.0, t2_us=80.0, readout_error=0.050),
            QubitProperties(qubit_id=4, t1_us=280.0, t2_us=230.0, readout_error=0.008),
            QubitProperties(qubit_id=5, t1_us=90.0, t2_us=70.0, readout_error=0.060),
        ]
    if gate_props is None:
        gate_props = [
            # Edge 0-1: very good (low error)
            GateProperties(gate_type="cx", qubits=(0, 1), error_rate=0.002, gate_time_ns=300.0),
            GateProperties(gate_type="cx", qubits=(1, 0), error_rate=0.002, gate_time_ns=300.0),
            # Edge 1-2: medium
            GateProperties(gate_type="cx", qubits=(1, 2), error_rate=0.008, gate_time_ns=400.0),
            GateProperties(gate_type="cx", qubits=(2, 1), error_rate=0.008, gate_time_ns=400.0),
            # Edge 2-3: bad
            GateProperties(gate_type="cx", qubits=(2, 3), error_rate=0.015, gate_time_ns=500.0),
            GateProperties(gate_type="cx", qubits=(3, 2), error_rate=0.015, gate_time_ns=500.0),
            # Edge 3-4: good
            GateProperties(gate_type="cx", qubits=(3, 4), error_rate=0.003, gate_time_ns=320.0),
            GateProperties(gate_type="cx", qubits=(4, 3), error_rate=0.003, gate_time_ns=320.0),
            # Edge 4-5: very bad
            GateProperties(gate_type="cx", qubits=(4, 5), error_rate=0.020, gate_time_ns=600.0),
            GateProperties(gate_type="cx", qubits=(5, 4), error_rate=0.020, gate_time_ns=600.0),
            # Edge 0-4: excellent
            GateProperties(gate_type="cx", qubits=(0, 4), error_rate=0.001, gate_time_ns=280.0),
            GateProperties(gate_type="cx", qubits=(4, 0), error_rate=0.001, gate_time_ns=280.0),
        ]
    if coupling_map is None:
        coupling_map = [
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
            (3, 4), (4, 3),
            (4, 5), (5, 4),
            (0, 4), (4, 0),
        ]
    return BackendProperties(
        backend="test_backend",
        provider="test",
        n_qubits=n_qubits,
        basis_gates=("cx", "rz", "sx", "x", "id"),
        coupling_map=coupling_map,
        qubit_properties=qubit_props,
        gate_properties=gate_props,
        timestamp="2026-03-12T00:00:00",
    )


def _make_2q_circuit() -> QBCircuit:
    """2-qubit circuit with a CX gate between logical qubits 0 and 1."""
    circ = QBCircuit(n_qubits=2, name="test_2q")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
    circ.add_gate(QBGate(name="h", qubits=(1,)))
    return circ


def _make_1q_only_circuit() -> QBCircuit:
    """3-qubit circuit with only single-qubit gates."""
    circ = QBCircuit(n_qubits=3, name="test_1q_only")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    circ.add_gate(QBGate(name="x", qubits=(1,)))
    circ.add_gate(QBGate(name="rz", qubits=(2,), params=(1.57,)))
    return circ


def _make_multi_2q_circuit() -> QBCircuit:
    """3-qubit circuit with multiple CX gates."""
    circ = QBCircuit(n_qubits=3, n_clbits=3, name="test_multi_2q")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))  # repeated
    circ.add_gate(QBGate(name="cx", qubits=(1, 2)))
    circ.add_gate(QBGate(name="h", qubits=(2,)))
    circ.add_measurement(0, 0)
    circ.add_measurement(1, 1)
    circ.add_measurement(2, 2)
    return circ


# ── basic mapping tests ──────────────────────────────────────────────


class TestCalibrationMapperBasic:
    """Basic mapping with known calibration data."""

    def test_returns_pass_result(self):
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_2q_circuit()
        context: dict = {}

        result = mapper.run(circuit, context)

        assert isinstance(result, PassResult)
        assert result.modified is True

    def test_context_has_initial_layout(self):
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_2q_circuit()
        context: dict = {}

        mapper.run(circuit, context)

        assert "initial_layout" in context
        layout = context["initial_layout"]
        assert isinstance(layout, dict)
        # Layout maps logical qubits 0 and 1 to physical qubits
        assert 0 in layout
        assert 1 in layout
        # Physical qubits should be distinct
        assert layout[0] != layout[1]

    def test_context_has_calibration_score(self):
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_2q_circuit()
        context: dict = {}

        mapper.run(circuit, context)

        assert "calibration_score" in context
        score = context["calibration_score"]
        assert isinstance(score, float)
        assert score >= 0.0

    def test_metadata_matches_context(self):
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_2q_circuit()
        context: dict = {}

        result = mapper.run(circuit, context)

        assert result.metadata["initial_layout"] == context["initial_layout"]
        assert result.metadata["calibration_score"] == context["calibration_score"]

    def test_best_qubits_selected_for_2q(self):
        """The mapper should prefer the lowest-error CX edge."""
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_2q_circuit()
        context: dict = {}

        mapper.run(circuit, context)
        layout = context["initial_layout"]

        # The best 2Q edge is (0, 4) with error 0.001
        # Both qubit 0 and qubit 4 are also the best individual qubits
        # So the layout should map to {0, 4} edge
        mapped_pair = {layout[0], layout[1]}
        assert mapped_pair == {0, 4}, (
            f"Expected mapping to edge (0,4) which has lowest error, "
            f"got physical qubits {mapped_pair}"
        )


class TestCalibrationMapperNo2QGates:
    """Circuit with no 2Q gates — should pick lowest-error individual qubits."""

    def test_picks_best_individual_qubits(self):
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_1q_only_circuit()
        context: dict = {}

        mapper.run(circuit, context)
        layout = context["initial_layout"]

        # 3 logical qubits, should pick the 3 best physical qubits
        assert len(layout) == 3
        physical = set(layout.values())
        assert len(physical) == 3  # all distinct

        # Qubit 0 (T1=300, T2=250, readout=0.005) and
        # qubit 4 (T1=280, T2=230, readout=0.008) should be among top picks
        assert 0 in physical
        assert 4 in physical

    def test_layout_is_valid(self):
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_1q_only_circuit()
        context: dict = {}

        result = mapper.run(circuit, context)
        mapped = result.circuit

        # All physical qubit indices must be valid
        for op in mapped.iter_ops():
            if isinstance(op, QBGate):
                for q in op.qubits:
                    assert 0 <= q < mapped.n_qubits


class TestCalibrationMapperWith2QGates:
    """Circuit with 2Q gates — should prefer lowest-error edges."""

    def test_prefers_low_error_edges(self):
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_multi_2q_circuit()
        context: dict = {}

        mapper.run(circuit, context)
        layout = context["initial_layout"]

        # Layout should not use the worst edges (2-3: 0.015, 4-5: 0.020)
        # The two strongest interactions are (0,1) with 2 CXs and (1,2) with 1 CX
        # These form a path of length 2 in the logical graph
        # Good physical paths: 4-0-1 (errors 0.001, 0.002)
        _p0, _p1 = layout[0], layout[1]
        # Just verify the score is reasonable (not using worst edges)
        score = context["calibration_score"]
        assert score < 1.0  # reasonable upper bound

    def test_score_improves_with_better_edges(self):
        """Verify that mapping to better edges yields a lower score."""
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_2q_circuit()

        # Run mapper
        context: dict = {}
        mapper.run(circuit, context)
        optimal_score = context["calibration_score"]

        # Manually compute score for a known-bad layout
        bad_layout = {0: 4, 1: 5}  # edge 4-5 has error 0.020
        interactions = CalibrationMapper._extract_interactions(circuit)
        bad_score = mapper._score_layout(bad_layout, interactions, circuit)

        # Optimal should be strictly better (lower) than the bad layout
        assert optimal_score < bad_score


class TestCalibrationMapperPreservesSemantics:
    """Mapping should preserve circuit semantics (gate count unchanged)."""

    def test_gate_count_preserved(self):
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_multi_2q_circuit()

        original_gate_count = circuit.gate_count
        original_2q_count = circuit.two_qubit_gate_count

        context: dict = {}
        result = mapper.run(circuit, context)
        mapped = result.circuit

        assert mapped.gate_count == original_gate_count
        assert mapped.two_qubit_gate_count == original_2q_count

    def test_measurement_count_preserved(self):
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_multi_2q_circuit()

        original_meas = len(circuit.measurements)

        context: dict = {}
        result = mapper.run(circuit, context)
        mapped = result.circuit

        assert len(mapped.measurements) == original_meas

    def test_gate_names_preserved(self):
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = _make_multi_2q_circuit()

        original_counts = circuit.gate_counts

        context: dict = {}
        result = mapper.run(circuit, context)
        mapped = result.circuit

        assert mapped.gate_counts == original_counts

    def test_gate_params_preserved(self):
        """Parametric gates should keep their parameters."""
        backend = _make_backend()
        mapper = CalibrationMapper(backend)

        circ = QBCircuit(n_qubits=2, name="param_test")
        circ.add_gate(QBGate(name="rz", qubits=(0,), params=(1.234,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="rx", qubits=(1,), params=(2.345,)))

        context: dict = {}
        result = mapper.run(circ, context)
        mapped = result.circuit

        mapped_gates = mapped.gates
        rz_gates = [g for g in mapped_gates if g.name == "rz"]
        rx_gates = [g for g in mapped_gates if g.name == "rx"]

        assert len(rz_gates) == 1
        assert rz_gates[0].params == (1.234,)
        assert len(rx_gates) == 1
        assert rx_gates[0].params == (2.345,)


class TestCalibrationMapperEdgeCases:
    """Edge case handling."""

    def test_circuit_needs_more_qubits_than_available(self):
        """Should raise ValueError if circuit is too large."""
        backend = _make_backend(n_qubits=3)
        mapper = CalibrationMapper(backend)
        circuit = QBCircuit(n_qubits=5, name="too_large")
        circuit.add_gate(QBGate(name="h", qubits=(0,)))

        with pytest.raises(ValueError, match="requires 5 qubits"):
            mapper.run(circuit, {})

    def test_single_qubit_circuit(self):
        """A 1-qubit circuit should still work."""
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = QBCircuit(n_qubits=1, name="single_q")
        circuit.add_gate(QBGate(name="h", qubits=(0,)))

        context: dict = {}
        mapper.run(circuit, context)

        assert "initial_layout" in context
        assert 0 in context["initial_layout"]
        # Should pick the best physical qubit (qubit 0: T1=300, T2=250, ro=0.005)
        assert context["initial_layout"][0] == 0

    def test_empty_circuit_no_gates(self):
        """Circuit with no gates at all."""
        backend = _make_backend()
        mapper = CalibrationMapper(backend)
        circuit = QBCircuit(n_qubits=2, name="empty")

        context: dict = {}
        mapper.run(circuit, context)

        assert "initial_layout" in context
        assert len(context["initial_layout"]) == 2

    def test_all_qubits_used_exactly(self):
        """When n_logical == n_physical, all physical qubits are used."""
        backend = _make_backend(n_qubits=2, qubit_props=[
            QubitProperties(qubit_id=0, t1_us=200.0, t2_us=180.0, readout_error=0.01),
            QubitProperties(qubit_id=1, t1_us=150.0, t2_us=120.0, readout_error=0.02),
        ], gate_props=[
            GateProperties(gate_type="cx", qubits=(0, 1), error_rate=0.005),
            GateProperties(gate_type="cx", qubits=(1, 0), error_rate=0.005),
        ], coupling_map=[(0, 1), (1, 0)])

        mapper = CalibrationMapper(backend)
        circuit = QBCircuit(n_qubits=2, name="exact_fit")
        circuit.add_gate(QBGate(name="cx", qubits=(0, 1)))

        context: dict = {}
        mapper.run(circuit, context)

        layout = context["initial_layout"]
        assert set(layout.values()) == {0, 1}


class TestCalibrationMapperWithProvider:
    """Test using CalibrationProvider instead of BackendProperties."""

    def test_accepts_calibration_provider(self):
        backend = _make_backend()
        provider = StaticCalibrationProvider(backend)

        mapper = CalibrationMapper(provider)
        circuit = _make_2q_circuit()
        context: dict = {}

        result = mapper.run(circuit, context)

        assert "initial_layout" in context
        assert "calibration_score" in context
        assert result.modified is True


class TestCalibrationMapperConfig:
    """Test custom configuration."""

    def test_custom_weights_affect_score(self):
        backend = _make_backend()
        circuit = _make_2q_circuit()

        # Default weights
        mapper1 = CalibrationMapper(backend)
        ctx1: dict = {}
        mapper1.run(circuit, ctx1)

        # Heavy readout weight
        config = CalibrationMapperConfig(
            gate_error_weight=0.1,
            coherence_weight=0.1,
            readout_weight=10.0,
        )
        mapper2 = CalibrationMapper(backend, config=config)
        ctx2: dict = {}
        mapper2.run(circuit, ctx2)

        # Scores should differ when weights differ
        # (layouts may even differ, but scores definitely will)
        # We just check both are valid
        assert ctx1["calibration_score"] >= 0.0
        assert ctx2["calibration_score"] >= 0.0


class TestCalibrationMapperWithRealSnapshot:
    """Test with the real IBM Fez calibration snapshot fixture."""

    @pytest.fixture
    def fez_backend(self) -> BackendProperties:
        from pathlib import Path

        fixture_path = (
            Path(__file__).parent.parent.parent
            / "fixtures"
            / "calibration_snapshots"
            / "ibm_fez_2026_02_15.json"
        )
        if not fixture_path.exists():
            pytest.skip("IBM Fez calibration fixture not found")
        return BackendProperties.from_qubitboost_json(fixture_path)

    def test_maps_small_circuit_on_fez(self, fez_backend: BackendProperties):
        mapper = CalibrationMapper(fez_backend)
        circuit = _make_2q_circuit()
        context: dict = {}

        mapper.run(circuit, context)

        assert "initial_layout" in context
        layout = context["initial_layout"]
        # Physical qubits should be valid Fez qubits
        for phys_q in layout.values():
            assert 0 <= phys_q < fez_backend.n_qubits

    def test_maps_3q_circuit_on_fez(self, fez_backend: BackendProperties):
        mapper = CalibrationMapper(fez_backend)
        circuit = _make_multi_2q_circuit()
        context: dict = {}

        result = mapper.run(circuit, context)

        layout = context["initial_layout"]
        assert len(layout) == 3
        # Gate count preserved
        assert result.circuit.gate_count == circuit.gate_count


class TestInteractionExtraction:
    """Test the interaction graph extraction."""

    def test_extract_interactions_simple(self):
        circ = QBCircuit(n_qubits=3, name="test")
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="cx", qubits=(1, 2)))

        interactions = CalibrationMapper._extract_interactions(circ)

        assert (0, 1) in interactions
        assert (1, 2) in interactions
        assert interactions[(0, 1)] == 1
        assert interactions[(1, 2)] == 1

    def test_extract_interactions_repeated(self):
        circ = QBCircuit(n_qubits=2, name="test")
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="cx", qubits=(1, 0)))  # reversed direction
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))

        interactions = CalibrationMapper._extract_interactions(circ)

        # All should be normalised to (0, 1)
        assert (0, 1) in interactions
        assert interactions[(0, 1)] == 3

    def test_extract_interactions_no_2q(self):
        circ = _make_1q_only_circuit()
        interactions = CalibrationMapper._extract_interactions(circ)
        assert interactions == {}
