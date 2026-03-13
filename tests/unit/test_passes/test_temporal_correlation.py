"""Tests for TemporalCorrelationAnalyzer."""

from __future__ import annotations

import pytest

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.passes.mapping.temporal_correlation import (
    TemporalCorrelationAnalyzer,
)


def _make_snapshot(
    qubit_readout: dict[int, float],
    gate_errors: dict[tuple[int, int], float] | None = None,
    timestamp: str = "2026-01-01T00:00:00Z",
) -> BackendProperties:
    """Build a minimal BackendProperties for testing."""
    qubits = [
        QubitProperties(
            qubit_id=qid,
            readout_error=ro,
            t1_us=200.0,
            t2_us=150.0,
        )
        for qid, ro in qubit_readout.items()
    ]

    coupling = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    gates = []
    ge = gate_errors or {}
    for q0, q1 in coupling:
        err = ge.get((q0, q1), 0.005)
        gates.append(
            GateProperties(gate_type="cx", qubits=(q0, q1), error_rate=err)
        )

    return BackendProperties(
        backend="test",
        provider="test",
        n_qubits=max(qubit_readout.keys()) + 1,
        basis_gates=("cx",),
        coupling_map=coupling,
        qubit_properties=qubits,
        gate_properties=gates,
        timestamp=timestamp,
    )


class TestTemporalCorrelationAnalyzer:
    """Tests for the temporal correlation analyzer."""

    def test_requires_at_least_2_snapshots(self):
        snap = _make_snapshot({0: 0.01, 1: 0.02})
        with pytest.raises(ValueError, match="at least 2"):
            TemporalCorrelationAnalyzer([snap])

    def test_qubit_volatility_stable_qubit(self):
        """A qubit with the same error across snapshots has zero volatility."""
        snap1 = _make_snapshot({0: 0.01, 1: 0.02}, timestamp="2026-01-01")
        snap2 = _make_snapshot({0: 0.01, 1: 0.02}, timestamp="2026-01-02")

        analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap1, snap2])
        assert analyzer.qubit_volatility(0) == pytest.approx(0.0)
        assert analyzer.qubit_volatility(1) == pytest.approx(0.0)

    def test_qubit_volatility_changing_qubit(self):
        """A qubit whose error changes has nonzero volatility."""
        snap1 = _make_snapshot({0: 0.01, 1: 0.02}, timestamp="2026-01-01")
        snap2 = _make_snapshot({0: 0.05, 1: 0.02}, timestamp="2026-01-02")

        analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap1, snap2])
        assert analyzer.qubit_volatility(0) > 0
        assert analyzer.qubit_volatility(1) == pytest.approx(0.0)

    def test_qubit_drift_positive_means_worse(self):
        """Positive drift means the error increased (got worse)."""
        snap1 = _make_snapshot({0: 0.01, 1: 0.05}, timestamp="2026-01-01")
        snap2 = _make_snapshot({0: 0.03, 1: 0.02}, timestamp="2026-01-02")

        analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap1, snap2])
        assert analyzer.qubit_drift(0) > 0   # got worse
        assert analyzer.qubit_drift(1) < 0   # got better

    def test_edge_correlation_co_moving(self):
        """Qubits whose errors move together should have positive correlation."""
        snap1 = _make_snapshot({0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01}, timestamp="t1")
        snap2 = _make_snapshot({0: 0.05, 1: 0.05, 2: 0.01, 3: 0.01}, timestamp="t2")
        snap3 = _make_snapshot({0: 0.02, 1: 0.02, 2: 0.01, 3: 0.01}, timestamp="t3")

        analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap1, snap2, snap3])
        # Q0 and Q1 move together: both up then both down
        corr_01 = analyzer.edge_correlation(0, 1)
        assert corr_01 > 0, f"Expected positive correlation, got {corr_01}"

    def test_edge_correlation_anti_moving(self):
        """Qubits whose errors move oppositely should have negative correlation."""
        snap1 = _make_snapshot({0: 0.01, 1: 0.05, 2: 0.01, 3: 0.01}, timestamp="t1")
        snap2 = _make_snapshot({0: 0.05, 1: 0.01, 2: 0.01, 3: 0.01}, timestamp="t2")
        snap3 = _make_snapshot({0: 0.02, 1: 0.04, 2: 0.01, 3: 0.01}, timestamp="t3")

        analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap1, snap2, snap3])
        corr_01 = analyzer.edge_correlation(0, 1)
        assert corr_01 < 0, f"Expected negative correlation, got {corr_01}"

    def test_unknown_qubit_returns_zero(self):
        """Querying a qubit not in the data returns 0."""
        snap1 = _make_snapshot({0: 0.01}, timestamp="t1")
        snap2 = _make_snapshot({0: 0.02}, timestamp="t2")

        analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap1, snap2])
        assert analyzer.qubit_volatility(999) == 0.0
        assert analyzer.edge_correlation(999, 998) == 0.0

    def test_n_snapshots_reported(self):
        snap1 = _make_snapshot({0: 0.01}, timestamp="t1")
        snap2 = _make_snapshot({0: 0.02}, timestamp="t2")

        analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap1, snap2])
        assert analyzer.result.n_snapshots == 2

    def test_gate_error_volatility_detected(self):
        """Volatile gate errors should contribute to edge correlation."""
        # Gate error changes dramatically between snapshots
        snap1 = _make_snapshot(
            {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01},
            gate_errors={(0, 1): 0.001, (1, 0): 0.001},
            timestamp="t1",
        )
        snap2 = _make_snapshot(
            {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01},
            gate_errors={(0, 1): 0.050, (1, 0): 0.050},
            timestamp="t2",
        )

        analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap1, snap2])
        # Edge (0,1) should have a nonzero correlation score from gate volatility
        corr = analyzer.edge_correlation(0, 1)
        assert corr > 0, f"Expected positive edge score from gate volatility, got {corr}"


class TestCalibrationMapperWithCorrelation:
    """Test CalibrationMapper integration with TemporalCorrelationAnalyzer."""

    def test_mapper_accepts_correlation_analyzer(self):
        """Mapper should accept and use a TemporalCorrelationAnalyzer."""
        from qb_compiler.ir.circuit import QBCircuit
        from qb_compiler.ir.operations import QBGate
        from qb_compiler.passes.mapping.calibration_mapper import (
            CalibrationMapper,
            CalibrationMapperConfig,
        )

        snap1 = _make_snapshot({0: 0.01, 1: 0.02, 2: 0.01, 3: 0.01}, timestamp="t1")
        snap2 = _make_snapshot({0: 0.05, 1: 0.06, 2: 0.01, 3: 0.01}, timestamp="t2")

        analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap1, snap2])

        mapper = CalibrationMapper(
            snap2,
            config=CalibrationMapperConfig(correlation_weight=5.0),
            correlation_analyzer=analyzer,
        )

        circ = QBCircuit(n_qubits=2, n_clbits=0)
        circ.add_gate(QBGate("cx", (0, 1)))

        ctx: dict = {}
        mapper.run(circ, ctx)

        assert "initial_layout" in ctx
        assert "score_breakdown" in ctx
        assert ctx["score_breakdown"]["correlation"] >= 0

    def test_volatile_qubits_are_avoided(self):
        """Mapper should avoid qubits with high error volatility."""
        # Q0 and Q1 are volatile; Q2 and Q3 are stable
        snap1 = _make_snapshot(
            {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01},
            timestamp="t1",
        )
        snap2 = _make_snapshot(
            {0: 0.10, 1: 0.10, 2: 0.01, 3: 0.01},
            timestamp="t2",
        )

        analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap1, snap2])

        from qb_compiler.ir.circuit import QBCircuit
        from qb_compiler.ir.operations import QBGate
        from qb_compiler.passes.mapping.calibration_mapper import (
            CalibrationMapper,
            CalibrationMapperConfig,
        )

        mapper = CalibrationMapper(
            snap2,
            config=CalibrationMapperConfig(
                gate_error_weight=1.0,
                coherence_weight=0.0,
                readout_weight=1.0,
                t1_asymmetry_weight=0.0,
                correlation_weight=50.0,  # very strong penalty
            ),
            correlation_analyzer=analyzer,
        )

        circ = QBCircuit(n_qubits=2, n_clbits=0)
        circ.add_gate(QBGate("cx", (0, 1)))

        ctx: dict = {}
        mapper.run(circ, ctx)
        layout = ctx["initial_layout"]

        # Should prefer stable qubits (Q2, Q3) over volatile ones (Q0, Q1)
        selected = set(layout.values())
        assert 2 in selected or 3 in selected, (
            f"Expected stable qubits (2 or 3) selected, got {layout}"
        )
