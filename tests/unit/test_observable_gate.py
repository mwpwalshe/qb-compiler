"""Tests for ObservableGate: the observable-mask-collapse decoder-input preflight.

The core invariant (:func:`audit_matrices`) is pure-numpy and runs in base CI without stim. The DEM
helpers are exercised only when stim is available (the ``[ising]`` extra).
"""

from __future__ import annotations

import numpy as np
import pytest

from qb_compiler.observable_gate import (
    ObservableMaskCollapseError,
    audit_matrices,
    preflight_dem_gate,
)


def test_witness_detector_identical_logical_distinct_is_fail():
    # two mechanisms, same detector, different logical mask -> genuine collapse risk
    check = np.array([[1, 1]], dtype=np.uint8)  # (n_det=1, n=2)
    obs = np.array([[0, 1]], dtype=np.uint8)  # (n_obs=1, n=2)
    priors = np.array([0.01, 0.01])
    result = audit_matrices(check, obs, priors)
    assert result.status == "FAIL"
    assert result.unique_detector_sigs == 1
    assert result.unique_detector_obs_sigs == 2
    assert result.mixed_groups == 1
    assert result.unsafe_to_merge_by_detector is True
    assert result.ok is False


def test_distinct_detector_signatures_is_pass():
    check = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    obs = np.array([[0, 1]], dtype=np.uint8)
    priors = np.array([0.01, 0.02])
    result = audit_matrices(check, obs, priors)
    assert result.status == "PASS"
    assert result.mixed_groups == 0
    assert result.ok is True


def test_decomposed_mixed_group_is_warn_not_fail():
    check = np.array([[1, 1]], dtype=np.uint8)
    obs = np.array([[0, 1]], dtype=np.uint8)
    priors = np.array([0.01, 0.01])
    result = audit_matrices(check, obs, priors, decomposed=True)
    assert result.status == "WARN"
    assert result.ok is True


def test_exact_mask_duplicates_do_not_trigger():
    check = np.array([[1, 1]], dtype=np.uint8)
    obs = np.array([[1, 1]], dtype=np.uint8)
    priors = np.array([0.01, 0.02])
    result = audit_matrices(check, obs, priors)
    assert result.status == "PASS"
    assert result.mixed_groups == 0


def test_worst_mask_ratio_equal_probs_is_one():
    check = np.array([[1, 1]], dtype=np.uint8)
    obs = np.array([[0, 1]], dtype=np.uint8)
    priors = np.array([0.01, 0.01])
    result = audit_matrices(check, obs, priors)
    assert result.worst_mask_ratio == pytest.approx(1.0)


# ── stim-dependent DEM-level tests (skipped if stim absent) ──────────────────


def test_preflight_gate_raises_on_fail_dem():
    stim = pytest.importorskip("stim")
    dem = stim.DetectorErrorModel("error(0.01) D0\nerror(0.01) D0 L0\ndetector D0\n")
    with pytest.raises(ObservableMaskCollapseError):
        preflight_dem_gate(dem)


def test_canonicalize_preserves_distinct_masks():
    stim = pytest.importorskip("stim")
    from qb_compiler.observable_gate import audit_dem, canonicalize_dem

    dem = stim.DetectorErrorModel("error(0.01) D0\nerror(0.01) D0 L0\ndetector D0\n")
    safe = canonicalize_dem(dem)
    # both distinct (detector, mask) mechanisms survive (the ambiguity is preserved, not erased)
    assert safe.num_errors == 2
    # still FAIL (intrinsic) -> the actionable signal is "don't merge by detector alone"
    assert audit_dem(safe).status == "FAIL"


def test_real_surface_dem_passes():
    stim = pytest.importorskip("stim")
    from qb_compiler.observable_gate import audit_dem

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.006,
        before_measure_flip_probability=0.006,
        after_reset_flip_probability=0.006,
        before_round_data_depolarization=0.006,
    )
    dem = circuit.detector_error_model(decompose_errors=False)
    assert audit_dem(dem).status == "PASS"
