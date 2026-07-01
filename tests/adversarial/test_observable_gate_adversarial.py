"""Hostile-input tests for the ObservableGate API (src/qb_compiler/observable_gate.py).

The pure-numpy core (``audit_matrices``) must reject malformed arrays with a clean,
typed ``ValueError`` rather than leaking a raw ``IndexError`` / numpy cast error. The
stim-backed helpers must survive empty and oversized DEMs without hanging or OOMing.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from qb_compiler.observable_gate import (
    ObservableMaskCollapseError,
    audit_matrices,
    canonicalize_dem,
    preflight_dem_gate,
)

stim = pytest.importorskip("stim")


# ── audit_matrices: malformed numpy inputs ──────────────────────────


class TestAuditMatricesHostile:
    def test_obs_fewer_columns_than_check_raises_valueerror(self) -> None:
        # FIX: previously raised a bare IndexError deep inside the loop.
        with pytest.raises(ValueError, match="column-count mismatch"):
            audit_matrices(np.ones((1, 5), np.uint8), np.ones((1, 2), np.uint8), np.ones(5))

    def test_priors_shorter_than_mechanisms_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="column-count mismatch"):
            audit_matrices(np.ones((1, 5), np.uint8), np.ones((1, 5), np.uint8), np.ones(2))

    def test_one_dimensional_check_matrix_raises_valueerror(self) -> None:
        # FIX: previously "IndexError: tuple index out of range".
        with pytest.raises(ValueError, match="2-D"):
            audit_matrices(
                np.array([1, 1], np.uint8), np.array([[0, 1]], np.uint8), np.array([0.5, 0.5])
            )

    def test_all_empty_flat_arrays_raise_valueerror(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            audit_matrices(np.array([]), np.array([]), np.array([]))

    def test_non_numeric_dtype_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            audit_matrices(
                np.array([["a", "b"]]), np.array([[0, 1]], np.uint8), np.array([0.5, 0.5])
            )

    def test_priors_two_dimensional_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            audit_matrices(np.ones((1, 2), np.uint8), np.ones((1, 2), np.uint8), np.ones((1, 2)))

    # ── benign edge cases that must NOT raise (locked-in behavior) ──

    def test_zero_mechanisms_is_pass(self) -> None:
        res = audit_matrices(np.zeros((1, 0), np.uint8), np.zeros((1, 0), np.uint8), np.zeros(0))
        assert res.status == "PASS"
        assert res.n_mechanisms == 0

    def test_nan_priors_do_not_crash(self) -> None:
        # A NaN probability is hostile but must not raise; the mask-collapse logic still runs.
        res = audit_matrices(
            np.array([[1, 1]], np.uint8), np.array([[0, 1]], np.uint8), np.array([np.nan, 0.5])
        )
        assert res.status in {"PASS", "WARN", "FAIL"}

    def test_negative_priors_do_not_crash(self) -> None:
        res = audit_matrices(
            np.array([[1, 1]], np.uint8), np.array([[0, 1]], np.uint8), np.array([-0.5, 0.5])
        )
        assert res.status == "FAIL"


# ── audit_dem / canonicalize: empty, degenerate, and oversized DEMs ──


class TestAuditDemHostile:
    def test_empty_dem_is_pass(self) -> None:
        res = preflight_dem_gate(stim.DetectorErrorModel())
        assert res.status == "PASS"
        assert res.n_mechanisms == 0

    def test_dem_with_no_detectors_or_observables(self) -> None:
        # An error mechanism that touches neither a detector nor an observable.
        dem = stim.DetectorErrorModel()
        dem.append("error", 0.1, [stim.target_logical_observable_id(0)])
        res = preflight_dem_gate(dem)
        assert res.status in {"PASS", "WARN", "FAIL"}

    def test_many_detector_identical_logical_distinct_mechanisms_fail_fast(self) -> None:
        # Thousands of collapse-risk pairs: must be detected (FAIL) and stay fast.
        dem = stim.DetectorErrorModel()
        for _ in range(2000):
            dem.append("error", 0.001, [stim.target_relative_detector_id(0)])
            dem.append(
                "error",
                0.001,
                [stim.target_relative_detector_id(0), stim.target_logical_observable_id(0)],
            )
        start = time.monotonic()
        with pytest.raises(ObservableMaskCollapseError):
            preflight_dem_gate(dem)
        assert time.monotonic() - start < 10.0

    def test_huge_dem_does_not_hang_or_oom(self) -> None:
        # 30k mechanisms — a size/time sanity check, capped to keep the suite fast.
        dem = stim.DetectorErrorModel()
        for i in range(30_000):
            dem.append("error", 0.001, [stim.target_relative_detector_id(i % 50)])
        start = time.monotonic()
        res = preflight_dem_gate(dem)
        assert time.monotonic() - start < 20.0
        assert res.status == "PASS"

    def test_canonicalize_empty_dem(self) -> None:
        out = canonicalize_dem(stim.DetectorErrorModel())
        assert out.num_errors == 0

    def test_canonicalize_preserves_distinct_masks(self) -> None:
        # Detector-identical, logical-distinct: must stay two mechanisms, never merge to one.
        dem = stim.DetectorErrorModel()
        dem.append("error", 0.01, [stim.target_relative_detector_id(0)])
        dem.append(
            "error",
            0.01,
            [stim.target_relative_detector_id(0), stim.target_logical_observable_id(0)],
        )
        out = canonicalize_dem(dem)
        assert out.num_errors == 2
