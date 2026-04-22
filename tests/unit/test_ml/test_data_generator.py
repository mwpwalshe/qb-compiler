"""Tests for ML training data generation."""

from __future__ import annotations

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.ml.data_generator import TrainingDataGenerator
from qb_compiler.ml.features import N_FEATURES


def _make_backend(n: int = 8) -> BackendProperties:
    qubits = [
        QubitProperties(
            qubit_id=i,
            t1_us=100.0 + i * 10,
            t2_us=80.0 + i * 5,
            readout_error=0.01 + i * 0.002,
            frequency_ghz=5.0,
        )
        for i in range(n)
    ]
    coupling = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]
    gates = [
        GateProperties(gate_type="cx", qubits=(q0, q1), error_rate=0.005) for q0, q1 in coupling
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


def _make_ghz(n: int) -> QBCircuit:
    c = QBCircuit(n_qubits=n, n_clbits=0)
    c.add_gate(QBGate("h", (0,)))
    for i in range(n - 1):
        c.add_gate(QBGate("cx", (i, i + 1)))
    return c


class TestTrainingDataGenerator:
    def test_generates_samples(self):
        backend = _make_backend(8)
        gen = TrainingDataGenerator(backend, n_trials=50, seed=42)
        circ = _make_ghz(3)
        batch = gen.generate_from_circuit(circ)
        assert len(batch.features) > 0
        assert len(batch.labels) == len(batch.features)

    def test_labels_are_binary(self):
        backend = _make_backend(8)
        gen = TrainingDataGenerator(backend, n_trials=50, seed=42)
        batch = gen.generate_from_circuit(_make_ghz(3))
        assert all(label in (0, 1) for label in batch.labels)

    def test_has_positive_labels(self):
        backend = _make_backend(8)
        gen = TrainingDataGenerator(backend, n_trials=50, seed=42)
        batch = gen.generate_from_circuit(_make_ghz(3))
        assert batch.n_positive > 0
        assert batch.n_negative > 0

    def test_feature_dimensions(self):
        backend = _make_backend(8)
        gen = TrainingDataGenerator(backend, n_trials=20, seed=42)
        batch = gen.generate_from_circuit(_make_ghz(3))
        assert all(len(row) == N_FEATURES for row in batch.features)

    def test_reproducibility(self):
        backend = _make_backend(8)
        gen1 = TrainingDataGenerator(backend, n_trials=30, seed=42)
        gen2 = TrainingDataGenerator(backend, n_trials=30, seed=42)
        b1 = gen1.generate_from_circuit(_make_ghz(3))
        b2 = gen2.generate_from_circuit(_make_ghz(3))
        assert b1.labels == b2.labels

    def test_multiple_circuits(self):
        backend = _make_backend(8)
        gen = TrainingDataGenerator(backend, n_trials=20, seed=42)
        circuits = [_make_ghz(2), _make_ghz(3), _make_ghz(4)]
        batch = gen.generate_from_circuits(circuits)
        # 3 circuits * 8 qubits = 24 samples
        assert len(batch.features) == 24
        assert batch.n_circuits == 3

    def test_circuit_too_large(self):
        backend = _make_backend(4)
        gen = TrainingDataGenerator(backend, n_trials=10, seed=42)
        # 5-qubit circuit on 4-qubit backend
        batch = gen.generate_from_circuit(_make_ghz(5))
        assert len(batch.features) == 0
