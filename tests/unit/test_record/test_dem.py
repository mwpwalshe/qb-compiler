# SPDX-License-Identifier: Apache-2.0
"""Matching helpers and fingerprint comparison, and the refusals each one makes."""

from __future__ import annotations

import numpy as np
import pytest

from qb_compiler.record.dem import (
    build_matching,
    build_repetition_dem,
    decode_records,
    decode_with_weight,
    fingerprint,
    fingerprint_matches,
)
from qb_compiler.record.dem.repetition import expected_repetition_fingerprint

pytest.importorskip("pymatching")


@pytest.fixture(scope="module")
def dem():
    return build_repetition_dem(5, 3)


class TestMatching:
    def test_a_clean_record_decodes_to_no_flip(self, dem):
        dets = np.zeros((10, dem.n_detectors), dtype=np.uint8)
        assert not decode_records(dem, dets).any()

    def test_a_single_boundary_event_flips_the_observable(self, dem):
        dets = np.zeros((1, dem.n_detectors), dtype=np.uint8)
        dets[0, 0] = 1
        prediction, weight = decode_with_weight(dem, dets)
        assert prediction.shape == (1,)
        assert weight[0] > 0

    def test_decode_with_weight_agrees_with_the_batch_decode(self, dem):
        rng = np.random.default_rng(0)
        dets = (rng.random((50, dem.n_detectors)) < 0.08).astype(np.uint8)
        batch = decode_records(dem, dets)
        one_at_a_time, weights = decode_with_weight(dem, dets)
        assert np.array_equal(batch, one_at_a_time)
        assert weights.shape == (50,)
        assert (weights >= 0).all()

    def test_weights_of_the_wrong_length_are_refused(self, dem):
        with pytest.raises(ValueError, match="weights must have shape"):
            build_matching(dem, np.ones(3))

    def test_a_detector_matrix_of_the_wrong_width_is_refused(self, dem):
        with pytest.raises(ValueError, match="detector_matrix must be"):
            decode_records(dem, np.zeros((5, 3), dtype=np.uint8))

    def test_a_one_dimensional_detector_matrix_is_refused(self, dem):
        with pytest.raises(ValueError, match="detector_matrix must be"):
            decode_records(dem, np.zeros(dem.n_detectors, dtype=np.uint8))

    def test_replacement_weights_are_used(self, dem):
        rng = np.random.default_rng(1)
        dets = (rng.random((30, dem.n_detectors)) < 0.15).astype(np.uint8)
        free = np.zeros(dem.n_mechanisms, dtype=np.float64)
        cheap = build_matching(dem, free)
        assert cheap.decode_batch(dets).shape[0] == 30


class TestFingerprintComparison:
    def test_an_exact_match_reports_so(self, dem):
        ok, detail = fingerprint_matches(fingerprint(dem), expected_repetition_fingerprint(5, 3))
        assert ok
        assert "matches" in detail

    def test_a_missing_field_is_named(self, dem):
        ok, detail = fingerprint_matches(fingerprint(dem), {"n_hyperedges": 12})
        assert not ok
        assert "n_hyperedges" in detail
        assert "not measured" in detail

    def test_a_histogram_difference_names_the_bucket(self, dem):
        expected = dict(expected_repetition_fingerprint(5, 3))
        expected["degree_histogram"] = {**expected["degree_histogram"], "4": 999}
        ok, detail = fingerprint_matches(fingerprint(dem), expected)
        assert not ok
        assert "degree_histogram[4]" in detail
        assert "999" in detail

    def test_a_bucket_present_on_one_side_only_is_named(self, dem):
        expected = dict(expected_repetition_fingerprint(5, 3))
        expected["mechanism_size_histogram"] = {**expected["mechanism_size_histogram"], "3": 4}
        ok, detail = fingerprint_matches(fingerprint(dem), expected)
        assert not ok
        assert "mechanism_size_histogram[3]" in detail

    def test_only_the_named_fields_are_compared(self, dem):
        ok, _ = fingerprint_matches(fingerprint(dem), {"n_boundary": 8})
        assert ok


class TestExpectedFingerprintRefusals:
    @pytest.mark.parametrize(("d", "rounds"), [(1, 3), (5, 0), (0, 0)])
    def test_impossible_parameters_are_refused(self, d, rounds):
        with pytest.raises(ValueError, match="need d >= 2 and rounds >= 1"):
            expected_repetition_fingerprint(d, rounds)
