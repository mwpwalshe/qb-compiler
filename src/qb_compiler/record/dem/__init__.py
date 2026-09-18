# SPDX-License-Identifier: Apache-2.0
"""Error models for records, and the matching helpers the checks run them through.

:mod:`~qb_compiler.record.dem.repetition` builds the standard repetition-code model,
:mod:`~qb_compiler.record.dem.fingerprint` reduces any model to a structural fingerprint, and the
functions here decode with one. Matching needs ``pymatching`` (the ``record`` extra); everything
else in this subpackage is numpy only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qb_compiler.record.dem.fingerprint import fingerprint, fingerprint_matches
from qb_compiler.record.dem.repetition import build_repetition_dem, expected_repetition_fingerprint
from qb_compiler.record.types import RecordDem

if TYPE_CHECKING:  # pragma: no cover
    import pymatching


def _require_pymatching() -> Any:
    try:
        import pymatching
    except ImportError:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "matching needs pymatching. Install with: pip install 'qb-compiler[record]'"
        ) from None
    return pymatching


def build_matching(dem: RecordDem, weights: np.ndarray | None = None) -> pymatching.Matching:
    """A ``pymatching.Matching`` over ``dem``, optionally with per-mechanism weights replaced.

    ``weights`` lets a caller decode with weights of their own without rebuilding the model
    description.
    """
    module = _require_pymatching()
    used = np.asarray(dem.weights if weights is None else weights, dtype=np.float64)
    if used.shape != (dem.n_mechanisms,):
        raise ValueError(f"weights must have shape ({dem.n_mechanisms},), got {used.shape}")
    faults = np.asarray(dem.observable, dtype=np.uint8).reshape(1, -1)
    matching: pymatching.Matching = module.Matching(
        np.asarray(dem.check_matrix, dtype=np.uint8), weights=used, faults_matrix=faults
    )
    return matching


def decode_records(dem: RecordDem, detector_matrix: np.ndarray) -> np.ndarray:
    """Predicted observable flip per shot. ``detector_matrix`` is ``(n_shots, n_detectors)``."""
    dets = np.asarray(detector_matrix, dtype=np.uint8)
    if dets.ndim != 2 or dets.shape[1] != dem.n_detectors:
        raise ValueError(f"detector_matrix must be (n_shots, {dem.n_detectors}), got {dets.shape}")
    matching = build_matching(dem)
    predictions: np.ndarray = matching.decode_batch(dets)[:, 0]
    return predictions.astype(np.uint8)


def decode_with_weight(
    dem: RecordDem, detector_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(prediction, solution_weight)`` per shot, decoded one shot at a time."""
    dets = np.asarray(detector_matrix, dtype=np.uint8)
    matching = build_matching(dem)
    n_shots = dets.shape[0]
    predictions = np.zeros(n_shots, dtype=np.uint8)
    weights = np.zeros(n_shots, dtype=np.float64)
    for shot in range(n_shots):
        correction, weight = matching.decode(dets[shot], return_weight=True)
        predictions[shot] = correction[0]
        weights[shot] = float(weight)
    return predictions, weights


__all__ = [
    "RecordDem",
    "build_matching",
    "build_repetition_dem",
    "decode_records",
    "decode_with_weight",
    "expected_repetition_fingerprint",
    "fingerprint",
    "fingerprint_matches",
]
