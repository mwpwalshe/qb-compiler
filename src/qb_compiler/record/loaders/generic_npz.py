# SPDX-License-Identifier: Apache-2.0
"""The ``.npz`` record contract: one file, named keys, declared dtypes, no coercion.

Required
--------
=============== ======================== =========== =============================================
key             shape                    dtype       meaning
=============== ======================== =========== =============================================
``dets``        (n, rounds, sites)       uint8       detector events
``labels``      (n,)                     uint8       observable-flip truth
=============== ======================== =========== =============================================

Optional
--------
=============== ======================== =========== =============================================
key             shape                    dtype       meaning
=============== ======================== =========== =============================================
``site_coords`` (sites, k)               numeric     site coordinates
``groups``      (n,)                     any         grouping for held-out splits
``loss``        (n, rounds, sites)       uint8/bool  loss symbols
                or (n, measurements)
``det_index``   (rounds, sites)          int32       error-model detector number, -1 for no
                                                     detector. Absent means every cell is a
                                                     detector, numbered row major
``H``           (detectors, mechanisms)  uint8       error model check matrix
``L``           (mechanisms,)            uint8       which mechanisms flip the observable
``weights``     (mechanisms,)            float64     matching weights
``edge_qubits`` (mechanisms,)            int32       which qubit each mechanism belongs to
``raw``         (n, raw_rounds, sites)   uint8       stored readouts, loader order
``final_parity``(n, sites)               uint8       parity of the final data readout
``final_data``  (n, data_qubits)         uint8       raw final data readout
``state``       (n,)                     uint8       prepared logical state
``data0``       (n,)                     uint8       first data qubit of the final readout
``det_sites``   (k,)                     int64       deterministic stabilizer indices
``meta``        scalar                   str         JSON object of provenance
=============== ======================== =========== =============================================

The reader refuses a file whose dtypes or shapes do not match rather than coercing. A record
silently widened from uint8 to int64, or reshaped from three axes to two, still decodes, and the
number it produces is wrong in a way no later step can see.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from qb_compiler.record.types import RecordDem, RecordSpec

SCHEMA = "qb.record_npz.v1"

_REQUIRED = ("dets", "labels")


def _require(data: Any, key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise ValueError(
            f"{path}: required key {key!r} is missing. The contract needs {list(_REQUIRED)}"
        )
    return np.asarray(data[key])


def _check_dtype(array: np.ndarray, key: str, expected: str, path: Path) -> np.ndarray:
    if array.dtype != np.dtype(expected):
        raise ValueError(
            f"{path}: {key!r} has dtype {array.dtype}, expected {expected}. This reader does not "
            "coerce: a record quietly widened or narrowed still decodes and produces a wrong "
            "number nothing downstream can see. Cast it deliberately and write it again"
        )
    return array


def _check_ndim(array: np.ndarray, key: str, ndim: int, path: Path) -> np.ndarray:
    if array.ndim != ndim:
        raise ValueError(
            f"{path}: {key!r} has {array.ndim} axes with shape {array.shape}, expected {ndim}"
        )
    return array


def read_npz(path: str | Path) -> RecordSpec:
    """Read a record written to the contract above.

    Raises ``ValueError`` on a missing key, a wrong dtype or a wrong shape, naming the key and
    what was expected.
    """
    file_path = Path(path).expanduser()
    with np.load(file_path, allow_pickle=False) as data:
        dets = _check_ndim(
            _check_dtype(_require(data, "dets", file_path), "dets", "uint8", file_path),
            "dets",
            3,
            file_path,
        )
        n_shots, n_rounds, n_sites = dets.shape
        labels = _check_ndim(
            _check_dtype(_require(data, "labels", file_path), "labels", "uint8", file_path),
            "labels",
            1,
            file_path,
        )
        if labels.shape[0] != n_shots:
            raise ValueError(
                f"{file_path}: 'labels' has {labels.shape[0]} rows against {n_shots} shots in "
                "'dets'"
            )

        detector_index = None
        if "det_index" in data:
            detector_index = _check_dtype(
                np.asarray(data["det_index"]), "det_index", "int32", file_path
            )
            if detector_index.shape != (n_rounds, n_sites):
                raise ValueError(
                    f"{file_path}: 'det_index' has shape {detector_index.shape}, expected "
                    f"({n_rounds}, {n_sites})"
                )

        loss = None
        if "loss" in data:
            loss = np.asarray(data["loss"])
            if loss.dtype not in (np.dtype("uint8"), np.dtype("bool")):
                raise ValueError(
                    f"{file_path}: 'loss' has dtype {loss.dtype}, expected uint8 or bool"
                )
            if loss.shape[0] != n_shots:
                raise ValueError(
                    f"{file_path}: 'loss' has {loss.shape[0]} rows against {n_shots} shots"
                )
            if loss.ndim not in (2, 3):
                raise ValueError(
                    f"{file_path}: 'loss' has {loss.ndim} axes, expected "
                    "(n, rounds, sites) or (n, measurements)"
                )

        dem = None
        if "H" in data:
            check_matrix = _check_ndim(
                _check_dtype(np.asarray(data["H"]), "H", "uint8", file_path), "H", 2, file_path
            )
            n_mech = check_matrix.shape[1]
            if "L" not in data:
                raise ValueError(f"{file_path}: 'H' is present but 'L' is not; a model needs both")
            observable = _check_dtype(np.asarray(data["L"]), "L", "uint8", file_path).ravel()
            if observable.shape[0] != n_mech:
                raise ValueError(
                    f"{file_path}: 'L' has {observable.shape[0]} entries against {n_mech} "
                    "mechanisms in 'H'"
                )
            if "weights" in data:
                weights = _check_dtype(
                    np.asarray(data["weights"]), "weights", "float64", file_path
                ).ravel()
                if weights.shape[0] != n_mech:
                    raise ValueError(
                        f"{file_path}: 'weights' has {weights.shape[0]} entries against {n_mech} "
                        "mechanisms in 'H'"
                    )
            else:
                weights = np.ones(n_mech, dtype=np.float64)
            edge_qubits = None
            if "edge_qubits" in data:
                edge_qubits = _check_dtype(
                    np.asarray(data["edge_qubits"]), "edge_qubits", "int32", file_path
                ).ravel()
                if edge_qubits.shape[0] != n_mech:
                    raise ValueError(
                        f"{file_path}: 'edge_qubits' has {edge_qubits.shape[0]} entries against "
                        f"{n_mech} mechanisms in 'H'"
                    )
            dem = RecordDem(
                check_matrix=check_matrix,
                observable=observable,
                weights=weights,
                edge_qubits=edge_qubits,
            )

        meta: dict[str, Any] = {}
        if "meta" in data:
            raw_meta = data["meta"]
            text = raw_meta.item() if raw_meta.ndim == 0 else str(raw_meta)
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError(f"{file_path}: 'meta' must hold a JSON object, got {type(parsed)}")
            meta = parsed

        optional: dict[str, Any] = {}
        for key, field, ndim, dtype in (
            ("raw", "raw_syndromes", 3, "uint8"),
            ("final_parity", "final_parity", 2, "uint8"),
            ("final_data", "final_data", 2, "uint8"),
            ("state", "logical_state", 1, "uint8"),
            ("data0", "data0", 1, "uint8"),
            ("det_sites", "deterministic_sites", 1, "int64"),
        ):
            if key in data:
                optional[field] = _check_ndim(
                    _check_dtype(np.asarray(data[key]), key, dtype, file_path),
                    key,
                    ndim,
                    file_path,
                )

        return RecordSpec(
            detectors=dets,
            detector_index=detector_index,
            labels=labels,
            groups=np.asarray(data["groups"]) if "groups" in data else None,
            loss=loss,
            site_coords=np.asarray(data["site_coords"]) if "site_coords" in data else None,
            dem=dem,
            meta=meta,
            **optional,
        )


def write_npz(path: str | Path, spec: RecordSpec) -> Path:
    """Write ``spec`` to the contract above, compressed. Returns the path written."""
    file_path = Path(path).expanduser()
    payload: dict[str, Any] = {
        "dets": np.asarray(spec.detectors, dtype=np.uint8),
        "meta": np.array(json.dumps({**spec.meta, "schema": SCHEMA}, default=str)),
    }
    if spec.labels is None:
        raise ValueError(
            "the contract requires labels: a record without them cannot say whether a decoder "
            "failed on a shot"
        )
    payload["labels"] = np.asarray(spec.labels, dtype=np.uint8)
    if spec.detector_index is not None:
        payload["det_index"] = np.asarray(spec.detector_index, dtype=np.int32)
    if spec.groups is not None:
        payload["groups"] = np.asarray(spec.groups)
    if spec.loss is not None:
        loss = np.asarray(spec.loss)
        payload["loss"] = loss if loss.dtype == np.dtype("bool") else loss.astype(np.uint8)
    if spec.site_coords is not None:
        payload["site_coords"] = np.asarray(spec.site_coords)
    for key, field, dtype in (
        ("raw", "raw_syndromes", np.uint8),
        ("final_parity", "final_parity", np.uint8),
        ("final_data", "final_data", np.uint8),
        ("state", "logical_state", np.uint8),
        ("data0", "data0", np.uint8),
        ("det_sites", "deterministic_sites", np.int64),
    ):
        value = getattr(spec, field)
        if value is not None:
            payload[key] = np.asarray(value, dtype=dtype)
    if spec.dem is not None:
        payload["H"] = np.asarray(spec.dem.check_matrix, dtype=np.uint8)
        payload["L"] = np.asarray(spec.dem.observable, dtype=np.uint8)
        payload["weights"] = np.asarray(spec.dem.weights, dtype=np.float64)
        if spec.dem.edge_qubits is not None:
            payload["edge_qubits"] = np.asarray(spec.dem.edge_qubits, dtype=np.int32)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return file_path


__all__ = ["SCHEMA", "read_npz", "write_npz"]
