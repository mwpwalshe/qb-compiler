# SPDX-License-Identifier: Apache-2.0
"""Every way a check declines to run, and what it says when it does.

A check that cannot run has to say which input it wanted. Silently passing a check that never
looked at anything is the failure mode these tests exist to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from qb_compiler.record.residual import residual
from qb_compiler.record.types import SKIP, RecordSpec
from qb_compiler.record.validate import (
    DEFAULT_THRESHOLDS,
    check_endpoint_agreement,
    check_event_density,
    check_label_reconstruction,
    check_state_profile,
    check_type_consistency,
)

pytest.importorskip("pymatching")


def _dets(n_shots=50, rounds=4, sites=3):
    return np.zeros((n_shots, rounds, sites), dtype=np.uint8)


class TestEndpointAgreementSkips:
    def test_a_single_stored_round_cannot_be_compared(self):
        spec = RecordSpec(
            detectors=_dets(),
            raw_syndromes=np.zeros((50, 1, 3), dtype=np.uint8),
            final_parity=np.zeros((50, 3), dtype=np.uint8),
        )
        result = check_endpoint_agreement(spec, DEFAULT_THRESHOLDS["endpoint_agreement"])
        assert result.status == SKIP
        assert "n_rounds >= 2" in result.detail

    def test_a_final_parity_of_the_wrong_shape_is_named(self):
        spec = RecordSpec(
            detectors=_dets(),
            raw_syndromes=np.zeros((50, 3, 3), dtype=np.uint8),
            final_parity=np.zeros((50, 5), dtype=np.uint8),
        )
        result = check_endpoint_agreement(spec, DEFAULT_THRESHOLDS["endpoint_agreement"])
        assert result.status == SKIP
        assert "does not match" in result.detail

    def test_perfect_agreement_with_the_first_round_leaves_no_scale(self):
        spec = RecordSpec(
            detectors=_dets(),
            raw_syndromes=np.zeros((50, 3, 3), dtype=np.uint8),
            final_parity=np.zeros((50, 3), dtype=np.uint8),
        )
        result = check_endpoint_agreement(spec, DEFAULT_THRESHOLDS["endpoint_agreement"])
        assert result.status == SKIP
        assert "nothing to compare against" in result.detail


class TestEventDensitySkips:
    def test_a_record_with_no_detector_cells_is_skipped(self):
        spec = RecordSpec(
            detectors=np.zeros((10, 2, 2), dtype=np.uint8),
            detector_index=np.full((2, 2), -1, dtype=np.int32),
        )
        result = check_event_density(spec, DEFAULT_THRESHOLDS["event_density"])
        assert result.status == SKIP
        assert result.critical
        assert "no detectors" in result.detail


class TestTypeConsistencySkips:
    def test_an_empty_deterministic_site_list_is_skipped(self):
        spec = RecordSpec(
            detectors=_dets(),
            raw_syndromes=np.zeros((50, 3, 3), dtype=np.uint8),
            deterministic_sites=np.asarray([], dtype=np.int64),
        )
        result = check_type_consistency(spec, DEFAULT_THRESHOLDS["type_consistency"])
        assert result.status == SKIP
        assert "empty" in result.detail

    def test_too_few_shots_to_average_is_skipped(self):
        spec = RecordSpec(
            detectors=_dets(n_shots=20),
            raw_syndromes=np.zeros((20, 3, 3), dtype=np.uint8),
            deterministic_sites=np.arange(3),
        )
        result = check_type_consistency(spec, DEFAULT_THRESHOLDS["type_consistency"])
        assert result.status == SKIP
        assert "at least 100 shots" in result.detail

    def test_a_site_that_is_all_loss_is_left_out(self):
        rng = np.random.default_rng(0)
        raw = (rng.random((200, 3, 3)) < 0.05).astype(np.uint8)
        raw[:, :, 2] = 2
        spec = RecordSpec(
            detectors=_dets(n_shots=200),
            raw_syndromes=raw,
            deterministic_sites=np.arange(3),
        )
        result = check_type_consistency(spec, DEFAULT_THRESHOLDS["type_consistency"])
        assert result.status == "PASS"
        assert result.measured["n_means"] == 6

    def test_a_record_that_is_all_loss_is_skipped(self):
        spec = RecordSpec(
            detectors=_dets(n_shots=200),
            raw_syndromes=np.full((200, 3, 3), 2, dtype=np.uint8),
            deterministic_sites=np.arange(3),
        )
        result = check_type_consistency(spec, DEFAULT_THRESHOLDS["type_consistency"])
        assert result.status == SKIP
        assert "loss symbol" in result.detail


class TestStateProfileSkips:
    def test_a_single_stored_round_has_no_step(self):
        spec = RecordSpec(
            detectors=_dets(),
            raw_syndromes=np.zeros((50, 1, 3), dtype=np.uint8),
            logical_state=np.ones(50, dtype=np.uint8),
        )
        result = check_state_profile(spec, DEFAULT_THRESHOLDS["state_profile"])
        assert result.status == SKIP
        assert "at least 2 stored rounds" in result.detail


class TestLabelReconstructionSkips:
    def test_too_few_shots_without_loss_is_skipped(self):
        loss = np.ones((40, 6), dtype=np.uint8)
        loss[0] = 0
        spec = RecordSpec(
            detectors=_dets(n_shots=40),
            labels=np.zeros(40, dtype=np.uint8),
            final_data=np.zeros((40, 6), dtype=np.uint8),
            loss=loss,
        )
        result = check_label_reconstruction(spec, DEFAULT_THRESHOLDS["label_reconstruction"])
        assert result.status == SKIP
        assert "at least 2 shots" in result.detail


class TestResidualInputShapes:
    def test_a_one_dimensional_feature_block_becomes_one_column(self):
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(0)
        labels = (rng.random(600) < 0.5).astype(np.uint8)
        failed = (rng.random(600) < 0.25).astype(np.uint8)
        spec = RecordSpec(detectors=(rng.random((600, 3, 3)) < 0.1).astype(np.uint8), labels=labels)
        output = {"prediction": (labels ^ failed).astype(np.uint8)}
        report = residual(spec, output, rng.normal(size=600), holdout="shot", n_nulls=1)
        assert report.feature_names == ("feature_0",)

    def test_a_decoder_column_of_the_wrong_length_is_refused(self):
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(0)
        labels = (rng.random(600) < 0.5).astype(np.uint8)
        failed = (rng.random(600) < 0.25).astype(np.uint8)
        spec = RecordSpec(detectors=(rng.random((600, 3, 3)) < 0.1).astype(np.uint8), labels=labels)
        output = {"prediction": (labels ^ failed).astype(np.uint8), "weight": np.zeros(7)}
        with pytest.raises(ValueError, match="must have 600 rows"):
            residual(spec, output, rng.normal(size=(600, 2)), holdout="shot", n_nulls=0)

    def test_a_prediction_of_the_wrong_length_is_refused(self):
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(0)
        spec = RecordSpec(
            detectors=(rng.random((600, 3, 3)) < 0.1).astype(np.uint8),
            labels=np.zeros(600, dtype=np.uint8),
        )
        with pytest.raises(ValueError, match="must have 600 rows"):
            residual(
                spec,
                {"prediction": np.zeros(5, dtype=np.uint8)},
                rng.normal(size=(600, 2)),
                n_nulls=0,
            )


class TestDetectorMatrixShape:
    def test_a_replacement_block_of_the_wrong_shape_is_refused(self):
        spec = RecordSpec(detectors=_dets(n_shots=10, rounds=4, sites=3))
        with pytest.raises(ValueError, match=r"must be \(n_shots, 4, 3\)"):
            spec.detector_matrix(np.zeros((10, 2, 2), dtype=np.uint8))

    def test_a_replacement_block_with_a_different_shot_count_is_fine(self):
        spec = RecordSpec(detectors=_dets(n_shots=10, rounds=4, sites=3))
        assert spec.detector_matrix(np.zeros((3, 4, 3), dtype=np.uint8)).shape == (3, 12)
