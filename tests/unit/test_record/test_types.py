# SPDX-License-Identifier: Apache-2.0
"""Types, serialisation, and the promise that nothing here edits what it was handed."""

from __future__ import annotations

import json

import numpy as np
import pytest

from qb_compiler.record import validate
from qb_compiler.record.dem import build_repetition_dem
from qb_compiler.record.types import (
    PASS,
    Check,
    RecordDem,
    RecordSpec,
    ResidualReport,
    ValidationReport,
)

pytest.importorskip("pymatching")


class TestRecordSpecShapes:
    def test_a_two_dimensional_detector_block_is_refused(self):
        with pytest.raises(ValueError, match="3-D"):
            RecordSpec(detectors=np.zeros((10, 6), dtype=np.uint8))

    def test_a_label_count_that_does_not_match_is_refused(self):
        with pytest.raises(ValueError, match="labels must have 10 rows"):
            RecordSpec(
                detectors=np.zeros((10, 3, 2), dtype=np.uint8),
                labels=np.zeros(9, dtype=np.uint8),
            )

    def test_a_detector_index_of_the_wrong_shape_is_refused(self):
        with pytest.raises(ValueError, match="detector_index must have shape"):
            RecordSpec(
                detectors=np.zeros((10, 3, 2), dtype=np.uint8),
                detector_index=np.zeros((4, 2), dtype=np.int32),
            )

    def test_without_an_index_every_cell_is_a_detector(self):
        spec = RecordSpec(detectors=np.zeros((10, 3, 4), dtype=np.uint8))
        assert spec.n_detectors == 12
        assert spec.site_valid.all()
        assert spec.detector_matrix().shape == (10, 12)
        assert spec.round_of_detector().tolist() == [0] * 4 + [1] * 4 + [2] * 4

    def test_an_index_selects_and_orders_the_detectors(self):
        index = np.array([[-1, 0, 1], [2, 3, -1]], dtype=np.int32)
        dets = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3) % 2
        spec = RecordSpec(detectors=dets, detector_index=index)
        assert spec.n_detectors == 4
        flat = spec.detector_matrix()
        assert flat.shape == (2, 4)
        assert flat[0].tolist() == [dets[0, 0, 1], dets[0, 0, 2], dets[0, 1, 0], dets[0, 1, 1]]
        assert spec.round_of_detector().tolist() == [0, 0, 1, 1]

    def test_replace_detectors_keeps_everything_else(self, good_spec):
        replaced = good_spec.replace_detectors(np.zeros_like(good_spec.detectors))
        assert replaced.dem is good_spec.dem
        assert replaced.labels is good_spec.labels
        assert replaced.meta == good_spec.meta
        assert not np.asarray(replaced.detectors).any()

    def test_the_summary_says_what_is_present(self, good_spec):
        summary = good_spec.as_dict()
        assert summary["n_shots"] == good_spec.n_shots
        assert summary["n_detectors"] == good_spec.n_detectors
        assert "labels" in summary["present"]
        assert "loss" not in summary["present"]
        assert json.loads(good_spec.to_json())["schema"] == "qb.record_spec.v1"


class TestRecordDem:
    def test_shapes_must_agree(self):
        with pytest.raises(ValueError, match="observable must have shape"):
            RecordDem(
                check_matrix=np.zeros((4, 6), dtype=np.uint8),
                observable=np.zeros(5, dtype=np.uint8),
                weights=np.zeros(6),
            )

    def test_a_one_dimensional_check_matrix_is_refused(self):
        with pytest.raises(ValueError, match="2-D"):
            RecordDem(
                check_matrix=np.zeros(6, dtype=np.uint8),
                observable=np.zeros(6, dtype=np.uint8),
                weights=np.zeros(6),
            )

    def test_edge_qubits_must_match(self):
        with pytest.raises(ValueError, match="edge_qubits"):
            RecordDem(
                check_matrix=np.zeros((4, 6), dtype=np.uint8),
                observable=np.zeros(6, dtype=np.uint8),
                weights=np.zeros(6),
                edge_qubits=np.zeros(3, dtype=np.int32),
            )

    def test_counts_read_off_the_check_matrix(self):
        dem = build_repetition_dem(5, 5)
        assert dem.n_detectors == 24
        assert dem.n_mechanisms == 50
        assert dem.meta["observable"] == "data qubit 0"

    def test_impossible_parameters_are_refused(self):
        with pytest.raises(ValueError, match="distance"):
            build_repetition_dem(1, 3)
        with pytest.raises(ValueError, match="rounds"):
            build_repetition_dem(5, 0)
        with pytest.raises(ValueError, match="p_data"):
            build_repetition_dem(5, 3, p_data=0.0)
        with pytest.raises(ValueError, match="p_meas"):
            build_repetition_dem(5, 3, p_meas=1.0)


class TestSerialisation:
    def test_a_validation_report_round_trips(self, good_spec):
        report = validate(good_spec)
        restored = ValidationReport.from_json(report.to_json())
        assert restored.passed == report.passed
        assert restored.n_shots == report.n_shots
        assert restored.decoder == report.decoder
        assert [c.name for c in restored.checks] == [c.name for c in report.checks]
        assert [c.status for c in restored.checks] == [c.status for c in report.checks]
        assert restored.check("round_profile").measured == report.check("round_profile").measured
        assert restored.as_dict() == report.as_dict()

    def test_a_residual_report_round_trips(self):
        report = ResidualReport(
            residual_bits=0.0225,
            null_floor_mean=0.0001,
            null_floor_p95=0.0012,
            above_floor=True,
            auc_baseline=0.71,
            auc_augmented=0.74,
            loss_baseline_bits=0.3524,
            loss_augmented_bits=0.3299,
            feature_names=("gap",),
            baseline_names=("prediction", "weight", "total_events"),
            holdout="shot",
            n_folds=5,
            n_shots=5834,
            n_failures=481,
            seed=0,
            regularization_c=0.1,
            nulls={"shot_permutation": {"available": True, "n": 20}},
            warnings=("folds are stratified over shots",),
        )
        restored = ResidualReport.from_json(report.to_json())
        assert restored == report
        assert json.loads(report.to_json())["schema"] == "qb.record_residual.v1"

    def test_a_check_round_trips(self):
        check = Check("round_profile", PASS, True, "fine", {"a": 1.5}, {"b": 2})
        assert Check.from_dict(check.as_dict()) == check
        assert check.passed
        assert not check.skipped
        assert "round_profile" in str(check)

    def test_a_report_reads_as_text(self, good_spec):
        report = validate(good_spec)
        assert report.failures == ()
        assert report.skipped
        assert "record validation" in str(report)


class TestPurity:
    def test_validate_does_not_edit_the_record(self, good_spec):
        before = {
            name: np.asarray(getattr(good_spec, name)).copy()
            for name in ("detectors", "labels", "raw_syndromes", "final_parity", "logical_state")
        }
        meta_before = dict(good_spec.meta)
        validate(good_spec)
        for name, original in before.items():
            assert np.array_equal(np.asarray(getattr(good_spec, name)), original), name
        assert good_spec.meta == meta_before

    def test_residual_does_not_edit_the_record(self):
        pytest.importorskip("sklearn")
        from qb_compiler.record import residual

        rng = np.random.default_rng(0)
        dets = (rng.random((400, 4, 4)) < 0.1).astype(np.uint8)
        labels = np.zeros(400, dtype=np.uint8)
        spec = RecordSpec(detectors=dets, labels=labels)
        before = np.asarray(spec.detectors).copy()
        prediction = (rng.random(400) < 0.3).astype(np.uint8)

        def block(record: RecordSpec) -> np.ndarray:
            return np.asarray(record.detectors, dtype=np.float64).sum(axis=(1, 2))[:, None]

        residual(spec, {"prediction": prediction}, block, holdout="shot", n_nulls=3)
        assert np.array_equal(np.asarray(spec.detectors), before)

    def test_the_report_holds_its_own_copy_of_the_record_meta(self, good_spec):
        report = validate(good_spec)
        report.meta["record_meta"]["d"] = 999
        assert good_spec.meta["d"] != 999
