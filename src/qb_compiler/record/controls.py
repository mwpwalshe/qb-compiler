# SPDX-License-Identifier: Apache-2.0
"""Controls: rearrangements of a record that destroy structure while preserving counts.

Three of them, each answering a different question.

``geometry_shuffle`` permutes detector events within a round, across sites, independently per
shot. Per-round event counts survive exactly; which site fired does not. Anything that reads only
counts is unchanged by it, so it is the null for a claim that a quantity reads spatial structure.

``time_mirror`` reverses the round axis. A correctly built record decodes worse mirrored, because
the error model is not symmetric in time; a mirrored record decodes better mirrored, which is how
a storage-order problem announces itself.

``shot_permutation`` reorders shots, breaking the pairing between a feature row and its shot while
leaving both marginal distributions exactly as they were.
"""

from __future__ import annotations

import numpy as np


def _as_3d(detectors: np.ndarray) -> np.ndarray:
    dets = np.asarray(detectors, dtype=np.uint8)
    if dets.ndim != 3:
        raise ValueError(
            f"detectors must be 3-D (n_shots, n_rounds, n_sites), got {dets.ndim}-D "
            f"with shape {dets.shape}"
        )
    return dets


def geometry_shuffle(
    detectors: np.ndarray,
    rng: np.random.Generator,
    *,
    site_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Permute detector events within each round, independently per shot.

    Parameters
    ----------
    detectors :
        ``(n_shots, n_rounds, n_sites)`` uint8.
    rng :
        Generator, consumed once per round.
    site_valid :
        Optional ``(n_rounds, n_sites)`` bool. Only cells marked valid take part, so a round that
        holds detectors on a subset of sites keeps its events inside that subset. When ``None``
        every cell takes part.

    Returns
    -------
    A new array. The input is not touched.
    """
    dets = _as_3d(detectors)
    out = dets.copy()
    n_rounds, n_sites = dets.shape[1], dets.shape[2]
    valid = (
        np.ones((n_rounds, n_sites), dtype=bool) if site_valid is None else np.asarray(site_valid)
    )
    if valid.shape != (n_rounds, n_sites):
        raise ValueError(f"site_valid must have shape ({n_rounds}, {n_sites}), got {valid.shape}")
    for t in range(n_rounds):
        cols = np.flatnonzero(valid[t])
        if cols.size < 2:
            continue
        block = out[:, t, :][:, cols]
        order = np.argsort(rng.random(block.shape), axis=1)
        shuffled = np.take_along_axis(block, order, axis=1)
        round_slice = out[:, t, :]
        round_slice[:, cols] = shuffled
        out[:, t, :] = round_slice
    return out


def time_mirror(detectors: np.ndarray) -> np.ndarray:
    """Reverse the round axis. Returns a new array; the input is not touched."""
    dets = _as_3d(detectors)
    return np.ascontiguousarray(dets[:, ::-1, :])


def shot_permutation(n_shots: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of ``range(n_shots)``."""
    if n_shots < 0:
        raise ValueError(f"n_shots must be non-negative, got {n_shots}")
    return rng.permutation(int(n_shots))


def is_round_symmetric(site_valid: np.ndarray) -> bool:
    """True when reversing the round axis leaves the set of detector cells unchanged.

    :func:`time_mirror` only makes sense on a record whose validity pattern survives reversal;
    otherwise the mirrored record has events in cells that hold no detector.
    """
    valid = np.asarray(site_valid)
    return bool(np.array_equal(valid, valid[::-1]))


__all__ = ["geometry_shuffle", "is_round_symmetric", "shot_permutation", "time_mirror"]
