# SPDX-License-Identifier: Apache-2.0
"""The three controls: what each one preserves, and what each one destroys."""

from __future__ import annotations

import numpy as np
import pytest

from qb_compiler.record.controls import (
    geometry_shuffle,
    is_round_symmetric,
    shot_permutation,
    time_mirror,
)


class TestGeometryShuffle:
    def test_per_round_counts_survive_exactly(self):
        rng = np.random.default_rng(0)
        dets = (rng.random((200, 5, 7)) < 0.2).astype(np.uint8)
        shuffled = geometry_shuffle(dets, np.random.default_rng(1))
        assert np.array_equal(dets.sum(axis=2), shuffled.sum(axis=2))
        assert dets.sum() == shuffled.sum()

    def test_which_site_fired_does_not(self):
        rng = np.random.default_rng(0)
        dets = np.zeros((200, 4, 6), dtype=np.uint8)
        dets[:, :, 0] = 1
        shuffled = geometry_shuffle(dets, np.random.default_rng(2))
        assert not np.array_equal(dets, shuffled)
        assert np.array_equal(dets.sum(axis=2), shuffled.sum(axis=2))
        del rng

    def test_the_input_is_not_touched(self):
        rng = np.random.default_rng(0)
        dets = (rng.random((50, 3, 4)) < 0.3).astype(np.uint8)
        original = dets.copy()
        geometry_shuffle(dets, np.random.default_rng(3))
        assert np.array_equal(dets, original)

    def test_it_stays_inside_the_valid_sites(self):
        rng = np.random.default_rng(0)
        dets = (rng.random((100, 3, 5)) < 0.3).astype(np.uint8)
        valid = np.array(
            [[True, True, False, False, False], [True] * 5, [True, True, False, False, False]]
        )
        dets[:, 0, 2:] = 0
        dets[:, 2, 2:] = 0
        shuffled = geometry_shuffle(dets, np.random.default_rng(4), site_valid=valid)
        assert not shuffled[:, 0, 2:].any()
        assert not shuffled[:, 2, 2:].any()
        assert np.array_equal(dets.sum(axis=2), shuffled.sum(axis=2))

    def test_it_is_seeded(self):
        rng = np.random.default_rng(0)
        dets = (rng.random((80, 3, 5)) < 0.3).astype(np.uint8)
        first = geometry_shuffle(dets, np.random.default_rng(9))
        second = geometry_shuffle(dets, np.random.default_rng(9))
        assert np.array_equal(first, second)

    def test_a_two_dimensional_block_is_refused(self):
        with pytest.raises(ValueError, match="3-D"):
            geometry_shuffle(np.zeros((10, 6), dtype=np.uint8), np.random.default_rng(0))

    def test_a_mismatched_validity_mask_is_refused(self):
        with pytest.raises(ValueError, match="site_valid must have shape"):
            geometry_shuffle(
                np.zeros((10, 3, 4), dtype=np.uint8),
                np.random.default_rng(0),
                site_valid=np.ones((2, 4), dtype=bool),
            )

    def test_a_round_with_one_site_is_left_alone(self):
        dets = np.ones((10, 2, 1), dtype=np.uint8)
        assert np.array_equal(geometry_shuffle(dets, np.random.default_rng(0)), dets)


class TestTimeMirror:
    def test_it_reverses_the_round_axis(self):
        dets = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3) % 2
        mirrored = time_mirror(dets)
        assert np.array_equal(mirrored[:, 0], dets[:, 3])
        assert np.array_equal(mirrored[:, 3], dets[:, 0])

    def test_it_is_its_own_inverse(self):
        rng = np.random.default_rng(0)
        dets = (rng.random((20, 5, 3)) < 0.3).astype(np.uint8)
        assert np.array_equal(time_mirror(time_mirror(dets)), dets)

    def test_the_input_is_not_touched(self):
        rng = np.random.default_rng(0)
        dets = (rng.random((20, 5, 3)) < 0.3).astype(np.uint8)
        original = dets.copy()
        time_mirror(dets)
        assert np.array_equal(dets, original)


class TestShotPermutation:
    def test_it_is_a_permutation(self):
        order = shot_permutation(500, np.random.default_rng(0))
        assert sorted(order.tolist()) == list(range(500))

    def test_it_is_seeded(self):
        first = shot_permutation(100, np.random.default_rng(5))
        second = shot_permutation(100, np.random.default_rng(5))
        assert np.array_equal(first, second)

    def test_a_negative_count_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            shot_permutation(-1, np.random.default_rng(0))


class TestRoundSymmetry:
    def test_a_full_pattern_is_symmetric(self):
        assert is_round_symmetric(np.ones((5, 4), dtype=bool))

    def test_a_surface_code_pattern_is_symmetric(self):
        valid = np.array([[True, False], [True, True], [True, True], [True, False]])
        assert is_round_symmetric(valid)

    def test_a_ragged_pattern_is_not(self):
        valid = np.array([[True, False], [True, True], [True, True]])
        assert not is_round_symmetric(valid)
