"""ObservableGate — observable-mask-collapse preflight for QEC decoder inputs.

A stim Detector Error Model (DEM) error mechanism carries ``(detectors, logical-observables,
probability)``.  A decoder predicts the observable *frame* from the detector *symptom*.  If a
DEM-to-matrix canonicalization merges mechanisms by detector signature alone, two such
detector-identical but logical-distinct collapse and the logical mask is lost or arbitrarily chosen.

That is **not** a semantics-preserving operation.  Measured harm (research): on
``color_code:memory_xyz`` d3, naive detector-only merging inflates the logical error rate
8.2% -> 13.1% (~60% relative).  Standard production paths are safe — surface/repetition and the full
bivariate-bicycle / Gross family ([[72,12,6]] .. [[144,12,12]] .. [[288,12,18]], X and Z basis) have
no detector-identical / logical-distinct mechanisms; decomposed DEMs are XOR-benign.  This module
detects the unsafe condition before decoding and offers an observable-preserving canonicalization.

The core invariant (:func:`audit_matrices`) runs on numpy arrays and needs no stim.  The DEM helpers
require stim (the ``[ising]`` extra).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import stim


class ObservableMaskCollapseError(RuntimeError):
    """Raised by :func:`preflight_dem_gate` when a DEM is unsafe for detector-only merging."""


@dataclass(frozen=True)
class ObservableAuditResult:
    """Observable-mask-collapse audit of a DEM / check-matrix construction."""

    n_mechanisms: int
    unique_detector_sigs: int
    unique_detector_obs_sigs: int
    mixed_groups: int
    mixed_mass: float
    worst_mask_ratio: float
    max_group: int
    decomposed: bool
    status: str  # "PASS" | "WARN" | "FAIL"

    @property
    def ok(self) -> bool:
        """True unless the DEM has detector-identical / logical-distinct mechanisms (FAIL)."""
        return self.status != "FAIL"

    @property
    def unsafe_to_merge_by_detector(self) -> bool:
        return self.unique_detector_sigs < self.unique_detector_obs_sigs

    def recommendation(self) -> str:
        return {
            "PASS": "detector-only canonicalization appears observable-safe.",
            "WARN": "mixed groups exist but the DEM is decomposed (likely XOR-benign); "
            "review before merging by detector signature.",
            "FAIL": "detector-identical mechanisms carry conflicting masks; canonicalize by "
            "(detectors, observables) or preserve P(L|H) — never merge by detector alone.",
        }[self.status]

    def __str__(self) -> str:
        return (
            "ObservableGate DEM audit\n"
            f"  raw mechanisms             : {self.n_mechanisms}\n"
            f"  unique detector signatures : {self.unique_detector_sigs}\n"
            f"  unique detector+obs masks  : {self.unique_detector_obs_sigs}\n"
            f"  mixed detector groups      : {self.mixed_groups}\n"
            f"  mixed probability mass     : {self.mixed_mass:.6f}\n"
            f"  worst mask ratio           : {self.worst_mask_ratio:.4f}\n"
            f"  max collapse bucket        : {self.max_group}\n"
            f"  decomposed DEM             : {self.decomposed}\n"
            f"  status: {self.status}\n"
            f"  recommendation: {self.recommendation()}"
        )


def audit_matrices(
    check_matrix: np.ndarray,
    obs_matrix: np.ndarray,
    priors: np.ndarray,
    *,
    decomposed: bool = False,
) -> ObservableAuditResult:
    """Core invariant (no stim).

    ``check_matrix``: (n_detectors, n); ``obs_matrix``: (n_observables, n) logical masks;
    ``priors``: (n,) mechanism probabilities.  Flags detector-identical / logical-distinct columns.
    """
    try:
        check_matrix = np.asarray(check_matrix, dtype=np.uint8)
        obs_matrix = np.asarray(obs_matrix, dtype=np.uint8)
        priors = np.asarray(priors, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"check_matrix, obs_matrix and priors must be numeric arrays (failed to coerce: {exc})"
        ) from exc

    if check_matrix.ndim != 2 or obs_matrix.ndim != 2:
        raise ValueError(
            "check_matrix and obs_matrix must be 2-D arrays "
            f"(got ndim {check_matrix.ndim} and {obs_matrix.ndim}); "
            "expected (n_detectors, n) and (n_observables, n)"
        )
    if priors.ndim != 1:
        raise ValueError(f"priors must be a 1-D array, got ndim {priors.ndim}")

    n = check_matrix.shape[1]
    if obs_matrix.shape[1] != n or priors.shape[0] != n:
        raise ValueError(
            "column-count mismatch: check_matrix has "
            f"{n} mechanism column(s), obs_matrix has {obs_matrix.shape[1]}, "
            f"priors has {priors.shape[0]} — all three must agree on the mechanism count"
        )

    by_detector: dict[bytes, list[int]] = {}
    for j in range(n):
        by_detector.setdefault(check_matrix[:, j].tobytes(), []).append(j)
    unique_h = len(by_detector)
    unique_hl = len({(check_matrix[:, j].tobytes(), obs_matrix[:, j].tobytes()) for j in range(n)})

    mixed = 0
    mixed_mass = 0.0
    worst_ratio = 0.0
    max_group = 0
    for group in by_detector.values():
        masks: dict[bytes, float] = {}
        for j in group:
            key = obs_matrix[:, j].tobytes()
            masks[key] = masks.get(key, 0.0) + float(priors[j])
        if len(masks) > 1:
            mixed += 1
            mixed_mass += float(sum(priors[j] for j in group))
            max_group = max(max_group, len(group))
            ordered = sorted(masks.values(), reverse=True)
            worst_ratio = max(worst_ratio, ordered[1] / ordered[0] if ordered[0] > 0 else 0.0)

    if mixed == 0:
        status = "PASS"
    elif decomposed:
        status = "WARN"
    else:
        status = "FAIL"

    return ObservableAuditResult(
        n_mechanisms=n,
        unique_detector_sigs=unique_h,
        unique_detector_obs_sigs=unique_hl,
        mixed_groups=mixed,
        mixed_mass=mixed_mass,
        worst_mask_ratio=worst_ratio,
        max_group=max_group,
        decomposed=decomposed,
        status=status,
    )


def dem_to_matrices(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Extract ``(check_matrix, obs_matrix, priors, decomposed)`` -- one column per mechanism."""
    n_det, n_obs = dem.num_detectors, dem.num_observables
    h_cols: list[np.ndarray] = []
    l_cols: list[np.ndarray] = []
    probs: list[float] = []
    decomposed = False
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        det = np.zeros(n_det, np.uint8)
        obs = np.zeros(n_obs, np.uint8)
        for target in inst.targets_copy():
            if target.is_separator():
                decomposed = True
            elif target.is_relative_detector_id():
                det[target.val] = 1
            elif target.is_logical_observable_id():
                obs[target.val] = 1
        h_cols.append(det)
        l_cols.append(obs)
        probs.append(prob)
    check_matrix = np.array(h_cols, np.uint8).T if h_cols else np.zeros((n_det, 0), np.uint8)
    obs_matrix = np.array(l_cols, np.uint8).T if l_cols else np.zeros((n_obs, 0), np.uint8)
    return check_matrix, obs_matrix, np.array(probs), decomposed


def audit_dem(dem: stim.DetectorErrorModel) -> ObservableAuditResult:
    """Audit a stim DEM for observable-mask collapse."""
    check_matrix, obs_matrix, priors, decomposed = dem_to_matrices(dem)
    return audit_matrices(check_matrix, obs_matrix, priors, decomposed=decomposed)


def canonicalize_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Observable-preserving canonicalization.

    Merge only EXACT ``(detectors, observables)`` duplicates (XOR-combining probabilities); keep
    distinct pairs separate.  The duplicate-detector / distinct-mask ambiguity is intrinsic and is
    preserved, never erased.
    """
    import stim

    check_matrix, obs_matrix, priors, _ = dem_to_matrices(dem)
    merged: dict[tuple[bytes, bytes], float] = {}
    for j in range(check_matrix.shape[1]):
        key = (check_matrix[:, j].tobytes(), obs_matrix[:, j].tobytes())
        prev = merged.get(key, 0.0)
        prob = float(priors[j])
        merged[key] = prev + prob - 2.0 * prev * prob  # XOR-combine independent firings

    out = stim.DetectorErrorModel()
    for (h_key, l_key), prob in merged.items():
        det = np.frombuffer(h_key, np.uint8)
        obs = np.frombuffer(l_key, np.uint8)
        targets = [stim.target_relative_detector_id(int(i)) for i in np.flatnonzero(det)]
        targets += [stim.target_logical_observable_id(int(i)) for i in np.flatnonzero(obs)]
        if targets:
            out.append("error", float(min(max(prob, 0.0), 1.0)), targets)
    return out


def preflight_dem_gate(
    dem: stim.DetectorErrorModel, *, strict: bool = False
) -> ObservableAuditResult:
    """Gate: audit ``dem`` and raise on FAIL (and on WARN if ``strict``); else return the result.

    Raises :class:`ObservableMaskCollapseError` so a preflight / CI step can block decoding of a DEM
    whose detector-only canonicalization would erase logical-observable information.
    """
    result = audit_dem(dem)
    if result.status == "FAIL" or (strict and result.status == "WARN"):
        raise ObservableMaskCollapseError(
            f"observable-mask collapse risk ({result.status}): {result.mixed_groups} "
            f"detector-identical / logical-distinct group(s), {result.mixed_mass:.4f} probability "
            "mass. Canonicalize by (detectors, observables) or use an observable-preserving path."
        )
    return result
