# SPDX-License-Identifier: Apache-2.0
"""A structural fingerprint of an error model.

Four numbers that do not depend on probabilities, ordering, or which library built the model:
how many mechanisms there are, how many of them touch a single detector, how many detectors sit at
each degree, and how many mechanisms have each detector count.

Two models with the same fingerprint are the same shape. Two with different fingerprints are
different models, whatever the file names say, and a decoder run against the wrong one produces
numbers that look ordinary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from qb_compiler.record.types import RecordDem


def _histogram(counts: np.ndarray) -> dict[str, int]:
    values, occurrences = np.unique(np.asarray(counts, dtype=np.int64), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, occurrences, strict=True)}


def fingerprint(dem: RecordDem) -> dict[str, Any]:
    """Structural fingerprint of ``dem``.

    Returns
    -------
    dict with ``n_mechanisms``, ``n_boundary``, ``degree_histogram`` (detector degree to how many
    detectors carry it) and ``mechanism_size_histogram`` (detector count to how many mechanisms
    carry it). Histogram keys are strings so the result survives a JSON round trip unchanged.
    """
    check = np.asarray(dem.check_matrix, dtype=np.uint8)
    mechanism_sizes = check.sum(axis=0)
    degrees = check.sum(axis=1)
    return {
        "n_mechanisms": int(check.shape[1]),
        "n_boundary": int((mechanism_sizes == 1).sum()),
        "degree_histogram": _histogram(degrees),
        "mechanism_size_histogram": _histogram(mechanism_sizes),
    }


def fingerprint_matches(measured: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    """Compare a measured fingerprint against an expected one.

    Only the keys present in ``expected`` are compared, so a caller can pin the mechanism count
    alone without having to write out both histograms. Returns ``(ok, detail)``; ``detail`` names
    every field that differs and by how much.
    """
    differences: list[str] = []
    for key, want in expected.items():
        if key not in measured:
            differences.append(f"{key}: expected {want}, not measured")
            continue
        got = measured[key]
        if isinstance(want, dict):
            keys = sorted(set(want) | set(got), key=lambda k: (len(k), k))
            for sub in keys:
                if int(want.get(sub, 0)) != int(got.get(sub, 0)):
                    differences.append(
                        f"{key}[{sub}]: expected {want.get(sub, 0)}, got {got.get(sub, 0)}"
                    )
        elif int(want) != int(got):
            differences.append(f"{key}: expected {want}, got {got}")
    if not differences:
        return True, "fingerprint matches the expected model on every compared field"
    return False, "; ".join(differences)


__all__ = ["fingerprint", "fingerprint_matches"]
