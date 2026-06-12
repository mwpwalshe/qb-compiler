"""Tests for verify mode (mirror-circuit fidelity check + accuracy record)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from qiskit import QuantumCircuit

# ── fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point QBC_DATA_DIR at tmp_path before anything touches the store."""
    monkeypatch.setenv("QBC_DATA_DIR", str(tmp_path))
    return tmp_path


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name="Bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))
    return qc


def _fake_runner(counts: dict[str, int]):
    def runner(circuit: Any, shots: int) -> dict[str, int]:
        return counts

    return runner


# ── build_mirror ─────────────────────────────────────────────────


class TestBuildMirror:
    def test_mirror_has_measurements_and_doubled_gates(self):
        from qb_compiler.verify import build_mirror

        qc = _bell_circuit()
        mirror = build_mirror(qc)

        ops = mirror.count_ops()
        # measure_all measures every qubit
        assert ops.get("measure", 0) == qc.num_qubits

        stripped = qc.remove_final_measurements(inplace=False)
        gate_count = sum(
            1 for inst in mirror.data if inst.operation.name not in ("measure", "barrier")
        )
        assert gate_count == 2 * len(stripped.data)
        # circuit + inverse is at least twice as deep as the stripped circuit
        assert mirror.depth() >= 2 * stripped.depth()

    def test_original_circuit_untouched(self):
        from qb_compiler.verify import build_mirror

        qc = _bell_circuit()
        before = len(qc.data)
        build_mirror(qc)
        assert len(qc.data) == before


# ── run_mirror ───────────────────────────────────────────────────


class TestRunMirror:
    def test_callable_runner_counts(self):
        from qb_compiler.verify import run_mirror

        result = run_mirror(_bell_circuit(), _fake_runner({"00": 250, "11": 6}), shots=256)
        assert result.shots == 256
        assert result.zeros_count == 250
        assert result.mirror_success == pytest.approx(250 / 256)

    def test_shot_mismatch_noted(self):
        from qb_compiler.verify import run_mirror

        result = run_mirror(_bell_circuit(), _fake_runner({"00": 90, "01": 10}), shots=256)
        assert result.shots == 100
        assert any("100" in note for note in result.notes)

    def test_keys_with_spaces_count_as_zeros(self):
        from qb_compiler.verify import run_mirror

        result = run_mirror(_bell_circuit(), _fake_runner({"0 0": 64}), shots=64)
        assert result.zeros_count == 64
        assert result.mirror_success == 1.0

    def test_bad_runner_raises(self):
        from qb_compiler.verify import run_mirror

        with pytest.raises(TypeError):
            run_mirror(_bell_circuit(), 42)

    def test_unknown_runner_string_raises(self):
        from qb_compiler.verify import run_mirror

        with pytest.raises(ValueError):
            run_mirror(_bell_circuit(), "qasm")


# ── verify_viability ─────────────────────────────────────────────


class TestVerifyViability:
    def test_records_one_line_with_sane_fields(self, tmp_path):
        from qb_compiler.verify import verify_viability

        result = verify_viability(
            _bell_circuit(),
            _fake_runner({"00": 250, "11": 6}),
            backend="ibm_fez",
            shots=256,
        )

        assert result.backend == "ibm_fez"
        assert result.n_qubits == 2
        assert 0.0 < result.predicted_fidelity <= 1.0
        assert result.predicted_squared == pytest.approx(result.predicted_fidelity**2)
        assert result.mirror_success == pytest.approx(250 / 256)
        assert result.shots == 256
        assert result.discrepancy == pytest.approx(result.mirror_success - result.predicted_squared)
        assert "predicted^2" in str(result)

        lines = (tmp_path / "verify_records.jsonl").read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["backend"] == "ibm_fez"
        assert record["n_qubits"] == 2
        assert record["shots"] == 256
        assert record["mirror_success"] == pytest.approx(250 / 256)
        assert "timestamp" in record

    def test_record_false_writes_nothing(self, tmp_path):
        from qb_compiler.verify import verify_viability

        verify_viability(
            _bell_circuit(),
            _fake_runner({"00": 200, "10": 56}),
            backend="ibm_fez",
            record=False,
        )
        assert not (tmp_path / "verify_records.jsonl").exists()


# ── accuracy_summary ─────────────────────────────────────────────


class TestAccuracySummary:
    def test_empty_log(self):
        from qb_compiler.verify import accuracy_summary

        summary = accuracy_summary()
        assert summary["n"] == 0
        assert summary["median_abs_discrepancy"] is None
        assert summary["mean_signed_discrepancy"] is None
        assert summary["per_backend_counts"] == {}

    def test_two_synthetic_records(self):
        from qb_compiler._store import append_jsonl
        from qb_compiler.verify import accuracy_summary

        append_jsonl(
            "verify_records.jsonl",
            {"backend": "ibm_fez", "discrepancy": 0.1},
        )
        append_jsonl(
            "verify_records.jsonl",
            {"backend": "ibm_torino", "discrepancy": -0.3},
        )

        summary = accuracy_summary()
        assert summary["n"] == 2
        assert summary["median_abs_discrepancy"] == pytest.approx(0.2)
        assert summary["mean_signed_discrepancy"] == pytest.approx(-0.1)
        assert summary["per_backend_counts"] == {"ibm_fez": 1, "ibm_torino": 1}


# ── aer path ─────────────────────────────────────────────────────


class TestAerRunner:
    def test_bell_mirror_end_to_end_on_aer(self):
        pytest.importorskip("qiskit_aer")
        from qb_compiler.verify import run_mirror

        result = run_mirror(_bell_circuit(), "aer", shots=128)
        # Noiseless simulator: the mirror returns all zeros every shot.
        assert result.shots == 128
        assert result.zeros_count == 128
        assert result.mirror_success == 1.0
        assert any("aer" in note for note in result.notes)
