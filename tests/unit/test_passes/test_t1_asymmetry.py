"""Tests for T1 asymmetry awareness in CalibrationMapper."""

from __future__ import annotations

import math

import pytest

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.calibration_mapper import (
    CalibrationMapper,
    CalibrationMapperConfig,
)


def _make_5q_backend(
    asymmetry_ratios: dict[int, float] | None = None,
) -> BackendProperties:
    """Build a 5-qubit linear backend with controllable T1 asymmetry.

    asymmetry_ratios maps qubit_id -> P(0|1)/P(1|0) ratio.
    Default: all qubits symmetric (ratio=1).
    """
    ratios = asymmetry_ratios or {}
    base_err = 0.005  # P(1|0) for all qubits

    qubits = []
    for i in range(5):
        ratio = ratios.get(i, 1.0)
        err_0to1 = base_err
        err_1to0 = base_err * ratio
        qubits.append(
            QubitProperties(
                qubit_id=i,
                t1_us=200.0,
                t2_us=150.0,
                readout_error=QubitProperties.symmetrise_readout(err_0to1, err_1to0),
                frequency_ghz=5.0,
                readout_error_0to1=err_0to1,
                readout_error_1to0=err_1to0,
            )
        )

    # All CX gates have the same error — only asymmetry differs
    gates = []
    coupling = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]
    for q0, q1 in coupling:
        gates.append(
            GateProperties(gate_type="cx", qubits=(q0, q1), error_rate=0.005, gate_time_ns=300.0)
        )

    return BackendProperties(
        backend="test_5q",
        provider="test",
        n_qubits=5,
        basis_gates=("cx", "rz", "sx", "x", "id"),
        coupling_map=coupling,
        qubit_properties=qubits,
        gate_properties=gates,
        timestamp="2026-03-12T10:00:00Z",
    )


class TestQubitPropertiesAsymmetry:
    """Test QubitProperties.t1_asymmetry_ratio and penalty."""

    def test_symmetric_qubit_ratio_is_one(self):
        qp = QubitProperties(
            qubit_id=0,
            readout_error_0to1=0.005,
            readout_error_1to0=0.005,
        )
        assert qp.t1_asymmetry_ratio == pytest.approx(1.0)

    def test_asymmetric_qubit_ratio(self):
        qp = QubitProperties(
            qubit_id=0,
            readout_error_0to1=0.001,
            readout_error_1to0=0.020,
        )
        assert qp.t1_asymmetry_ratio == pytest.approx(20.0)

    def test_missing_data_ratio_is_one(self):
        qp = QubitProperties(qubit_id=0)
        assert qp.t1_asymmetry_ratio == 1.0

    def test_zero_excitation_error_ratio_is_one(self):
        qp = QubitProperties(
            qubit_id=0,
            readout_error_0to1=0.0,
            readout_error_1to0=0.01,
        )
        assert qp.t1_asymmetry_ratio == 1.0

    def test_symmetric_penalty_is_zero(self):
        qp = QubitProperties(
            qubit_id=0,
            readout_error_0to1=0.005,
            readout_error_1to0=0.005,
        )
        assert qp.t1_asymmetry_penalty == 0.0

    def test_asymmetric_penalty_positive(self):
        qp = QubitProperties(
            qubit_id=0,
            readout_error=0.0055,
            readout_error_0to1=0.001,
            readout_error_1to0=0.010,
        )
        # ratio=10, penalty=readout_error * ln(10) ≈ 0.0055 * 2.3 ≈ 0.01267
        assert qp.t1_asymmetry_penalty == pytest.approx(0.0055 * math.log(10), rel=1e-6)

    def test_inverted_ratio_penalty_is_zero(self):
        """Ratio < 1 means qubit is BETTER at holding |1⟩ — no penalty."""
        qp = QubitProperties(
            qubit_id=0,
            readout_error_0to1=0.010,
            readout_error_1to0=0.005,
        )
        assert qp.t1_asymmetry_penalty == 0.0


class TestCalibrationMapperAsymmetry:
    """Test that CalibrationMapper avoids high-asymmetry qubits."""

    def test_asymmetry_weight_changes_layout(self):
        """With one very asymmetric qubit, the mapper should avoid it."""
        # Q2 has 20x asymmetry; others are symmetric
        backend = _make_5q_backend(asymmetry_ratios={2: 20.0})

        # 2-qubit circuit: can pick from {0,1}, {1,2}, {2,3}, {3,4}
        # so avoiding Q2 is possible
        circ = QBCircuit(n_qubits=2, n_clbits=0)
        circ.add_gate(QBGate("cx", (0, 1)))

        # WITH asymmetry awareness — should avoid Q2
        mapper_asym = CalibrationMapper(
            backend,
            config=CalibrationMapperConfig(
                t1_asymmetry_weight=10.0,  # strong penalty
                correlation_weight=0.0,
            ),
        )
        ctx_yes: dict = {}
        mapper_asym.run(circ, ctx_yes)
        layout_yes = ctx_yes["initial_layout"]

        # The asymmetry-aware mapper should avoid Q2
        selected_with_asym = set(layout_yes.values())
        assert 2 not in selected_with_asym, (
            f"Asymmetry-aware mapper selected Q2 (20x asymmetry): {layout_yes}"
        )

    def test_score_breakdown_includes_asymmetry(self):
        """Score breakdown should include t1_asymmetry component."""
        backend = _make_5q_backend(asymmetry_ratios={0: 10.0})

        circ = QBCircuit(n_qubits=2, n_clbits=0)
        circ.add_gate(QBGate("cx", (0, 1)))

        mapper = CalibrationMapper(
            backend,
            config=CalibrationMapperConfig(t1_asymmetry_weight=3.0),
        )
        ctx: dict = {}
        mapper.run(circ, ctx)

        breakdown = ctx["score_breakdown"]
        assert "t1_asymmetry" in breakdown
        assert breakdown["t1_asymmetry"] >= 0

    def test_zero_asymmetry_weight_matches_original(self):
        """With t1_asymmetry_weight=0, behaviour matches original mapper."""
        backend = _make_5q_backend(asymmetry_ratios={2: 20.0})

        circ = QBCircuit(n_qubits=2, n_clbits=0)
        circ.add_gate(QBGate("cx", (0, 1)))

        mapper = CalibrationMapper(
            backend,
            config=CalibrationMapperConfig(
                t1_asymmetry_weight=0.0,
                correlation_weight=0.0,
            ),
        )
        ctx: dict = {}
        mapper.run(circ, ctx)

        breakdown = ctx["score_breakdown"]
        assert breakdown["t1_asymmetry"] == 0.0
        assert breakdown["correlation"] == 0.0
