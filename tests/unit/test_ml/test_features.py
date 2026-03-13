"""Tests for ML feature extraction."""

from __future__ import annotations

import pytest

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.ml.features import (
    N_FEATURES,
    build_feature_matrix,
    extract_circuit_features,
    extract_qubit_features,
    feature_names,
    to_feature_vector,
)


def _make_backend(n: int = 4) -> BackendProperties:
    qubits = [
        QubitProperties(
            qubit_id=i,
            t1_us=100.0 + i * 10,
            t2_us=80.0 + i * 5,
            readout_error=0.01 + i * 0.005,
            frequency_ghz=5.0 + i * 0.1,
            readout_error_0to1=0.005 + i * 0.001,
            readout_error_1to0=0.015 + i * 0.005,
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


class TestCircuitFeatures:
    def test_bell_features(self):
        feats = extract_circuit_features(_make_bell())
        assert feats.n_logical_qubits == 2
        assert feats.n_2q_gates == 1
        assert feats.n_gates == 2  # h + cx
        assert feats.interaction_density == pytest.approx(1.0)
        assert feats.max_interaction_degree == 1
        assert feats.max_interaction_weight == 1

    def test_ghz5_features(self):
        feats = extract_circuit_features(_make_ghz(5))
        assert feats.n_logical_qubits == 5
        assert feats.n_2q_gates == 4
        assert feats.max_interaction_degree == 2  # middle qubits

    def test_no_2q_gates(self):
        c = QBCircuit(n_qubits=3, n_clbits=0)
        c.add_gate(QBGate("h", (0,)))
        c.add_gate(QBGate("h", (1,)))
        feats = extract_circuit_features(c)
        assert feats.n_2q_gates == 0
        assert feats.interaction_density == 0.0
        assert feats.max_interaction_degree == 0

    def test_to_list_length(self):
        feats = extract_circuit_features(_make_bell())
        assert len(feats.to_list()) == 8


class TestQubitFeatures:
    def test_extract_from_backend(self):
        backend = _make_backend(4)
        feats = extract_qubit_features(0, backend)
        assert feats.t1_us == 100.0
        assert feats.t2_us == 80.0
        assert feats.readout_error == 0.01
        assert feats.connectivity_degree >= 1

    def test_unknown_qubit_defaults(self):
        backend = _make_backend(4)
        feats = extract_qubit_features(999, backend)
        assert feats.t1_us == 100.0  # default
        assert feats.readout_error == 0.015  # default

    def test_to_list_length(self):
        backend = _make_backend(4)
        feats = extract_qubit_features(0, backend)
        assert len(feats.to_list()) == 9


class TestFeatureVector:
    def test_vector_length(self):
        circ = _make_bell()
        backend = _make_backend(4)
        cf = extract_circuit_features(circ)
        qf = extract_qubit_features(0, backend)
        vec = to_feature_vector(cf, qf)
        assert len(vec) == N_FEATURES

    def test_feature_names_match(self):
        names = feature_names()
        assert len(names) == N_FEATURES

    def test_deterministic(self):
        circ = _make_bell()
        backend = _make_backend(4)
        cf = extract_circuit_features(circ)
        qf = extract_qubit_features(0, backend)
        v1 = to_feature_vector(cf, qf)
        v2 = to_feature_vector(cf, qf)
        assert v1 == v2


class TestBuildFeatureMatrix:
    def test_matrix_shape(self):
        circ = _make_bell()
        backend = _make_backend(4)
        matrix, qubit_ids = build_feature_matrix(circ, backend)
        assert len(matrix) == 4
        assert len(qubit_ids) == 4
        assert all(len(row) == N_FEATURES for row in matrix)

    def test_qubit_ids_sorted(self):
        backend = _make_backend(4)
        _, qubit_ids = build_feature_matrix(_make_bell(), backend)
        assert qubit_ids == sorted(qubit_ids)
