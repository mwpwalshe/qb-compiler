# SPDX-License-Identifier: Apache-2.0
"""The standard repetition-code error model.

``d`` data qubits in a chain, ``rounds`` rounds of stabilizer measurement, then a final data
readout. Detectors sit at ``(t, x)`` for ``t`` in ``0..rounds`` and ``x`` in ``0..d-2``: the first
``rounds`` come from the stabilizer readouts, the last from differencing the final data parity
against the last stabilizer round.

Two mechanism families:

* a data error on qubit ``q`` at round ``t``, lighting the detectors either side of it, so one
  detector at the ends of the chain and two in the middle. ``d(rounds+1)`` of them, of which
  ``2(rounds+1)`` are boundary mechanisms.
* a measurement error at ``(t, x)``, lighting ``(t, x)`` and ``(t+1, x)``. ``(d-1)rounds`` of them.

The observable is data qubit 0. A data error on qubit 0 flips it; nothing else does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qb_compiler.record.types import RecordDem

DATA_MECHANISM = 0
MEASUREMENT_MECHANISM = 1


def build_repetition_dem(
    d: int,
    rounds: int,
    *,
    p_data: float = 0.01,
    p_meas: float = 0.01,
) -> RecordDem:
    """Build the repetition-code error model for a distance-``d``, ``rounds``-round memory.

    ``p_data`` and ``p_meas`` are flat priors; the matching weight of a mechanism is
    ``-log(p / (1 - p))``, so a smaller probability costs more to use.

    The detector numbering is ``t * (d - 1) + x``, which is the row-major numbering a
    :class:`~qb_compiler.record.types.RecordSpec` with ``detector_index=None`` already uses.
    """
    if d < 2:
        raise ValueError(f"repetition distance must be at least 2, got {d}")
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    for name, prob in (("p_data", p_data), ("p_meas", p_meas)):
        if not 0.0 < prob < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1, got {prob}")

    n_layers = rounds + 1
    n_sites = d - 1
    n_detectors = n_layers * n_sites

    columns: list[list[int]] = []
    kinds: list[int] = []
    observable: list[int] = []
    owners: list[int] = []

    for t in range(n_layers):
        for q in range(d):
            lit = []
            if q - 1 >= 0:
                lit.append(t * n_sites + (q - 1))
            if q <= d - 2:
                lit.append(t * n_sites + q)
            columns.append(lit)
            kinds.append(DATA_MECHANISM)
            observable.append(1 if q == 0 else 0)
            owners.append(q)

    for t in range(n_layers - 1):
        for x in range(n_sites):
            columns.append([t * n_sites + x, (t + 1) * n_sites + x])
            kinds.append(MEASUREMENT_MECHANISM)
            observable.append(0)
            owners.append(d + x)

    n_mech = len(columns)
    check_matrix = np.zeros((n_detectors, n_mech), dtype=np.uint8)
    for j, lit in enumerate(columns):
        check_matrix[lit, j] = 1

    kind_array = np.asarray(kinds, dtype=np.int8)
    weight_data = -np.log(p_data / (1.0 - p_data))
    weight_meas = -np.log(p_meas / (1.0 - p_meas))
    weights = np.where(kind_array == DATA_MECHANISM, weight_data, weight_meas).astype(np.float64)

    meta: dict[str, Any] = {
        "family": "repetition",
        "d": int(d),
        "rounds": int(rounds),
        "p_data": float(p_data),
        "p_meas": float(p_meas),
        "observable": "data qubit 0",
        "n_data_mechanisms": int((kind_array == DATA_MECHANISM).sum()),
        "n_measurement_mechanisms": int((kind_array == MEASUREMENT_MECHANISM).sum()),
    }
    return RecordDem(
        check_matrix=check_matrix,
        observable=np.asarray(observable, dtype=np.uint8),
        weights=weights,
        edge_qubits=np.asarray(owners, dtype=np.int32),
        meta=meta,
    )


def expected_repetition_fingerprint(d: int, rounds: int) -> dict[str, Any]:
    """The fingerprint :func:`build_repetition_dem` must produce, derived from ``d`` and ``rounds``.

    Written out rather than computed from the model, so that handing it to
    :func:`~qb_compiler.record.validate.validate` checks the model against arithmetic rather than
    against itself.
    """
    if d < 2 or rounds < 1:
        raise ValueError(f"need d >= 2 and rounds >= 1, got d={d}, rounds={rounds}")
    n_layers = rounds + 1
    n_sites = d - 1
    n_data = d * n_layers
    n_meas = n_sites * rounds
    n_boundary = 2 * n_layers

    degrees: dict[str, int] = {}
    for t in range(n_layers):
        for _site in range(n_sites):
            deg = 2  # the two data mechanisms either side of this site, in this layer
            deg += 1 if t > 0 else 0
            deg += 1 if t < n_layers - 1 else 0
            degrees[str(deg)] = degrees.get(str(deg), 0) + 1

    return {
        "n_mechanisms": n_data + n_meas,
        "n_boundary": n_boundary,
        "mechanism_size_histogram": {"1": n_boundary, "2": n_data - n_boundary + n_meas},
        "degree_histogram": degrees,
    }


__all__ = [
    "DATA_MECHANISM",
    "MEASUREMENT_MECHANISM",
    "build_repetition_dem",
    "expected_repetition_fingerprint",
]
