# SPDX-License-Identifier: Apache-2.0
"""Loader for the QuEra surface-code release, built through the publisher's own framework.

Dataset::

    QuEra Computing, surface code dataset, Zenodo record 15685795
    https://doi.org/10.5281/zenodo.15685795

Nothing is redistributed here and no vendor code is vendored. This reads the archive the
publisher serves and calls the decoding framework published alongside it, from a path the caller
supplies. ``qbc corpus verify quera-surface-code <your copy>`` checks the archive first.

The trap
--------
The archive stores ``measurement_events``: raw ancilla readouts, ternary, with ``2`` marking an
atom lost before it could be read. Raw readouts are not detectors. Differencing consecutive raw
readouts, the construction that looks obvious, produces an event rate of 0.448 on the d5 Z
memory, against 0.133 from the publisher's own construction. Detectors at 0.448 are coin flips.
A decoder still runs on them and still reports a logical error rate.

The reason is that the raw readouts were never referenced to a frame: the deterministic
stabilizers read anywhere between 0.487 and 0.633 under the naive construction, where the
publisher's construction puts them at 0.023 to 0.947. So ``detectors="naive"`` is refused rather
than offered, and ``detectors="vendor"`` calls ``MemorySimulator.get_detectors``.

Usage::

    from qb_compiler.record.loaders import quera_surface

    spec = quera_surface.load(
        "quera_surface_code.zip",
        "Zenodo/SurfaceCodeData/Data/Distance5/d5_Z_memory.npz",
        vendor_module_dir="/path/to/the/published/ML_decoder",
    )
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from qb_compiler.record.types import FIRST_LAYER_PREMISE, RecordSpec

LOADER_VERSION = "1.1"
LOSS_SYMBOL = 2

CITATION = (
    "QuEra Computing, surface code dataset, Zenodo record 15685795 "
    "(https://doi.org/10.5281/zenodo.15685795). Cite the record and the paper it accompanies, "
    "both named on the record page."
)

NAIVE_REFUSED = (
    "detectors='naive' is refused. Differencing consecutive raw readouts of this record produces "
    "an event rate near 0.45, against 0.13 from the publisher's own construction, because the raw "
    "readouts were never referenced to a frame: the deterministic stabilizers come out near 0.5 "
    "instead of near 0 or 1. Detectors built that way are close to coin flips and a decoder run "
    "on them reports a logical error rate that means nothing. Use detectors='vendor', which calls "
    "MemorySimulator.get_detectors from the framework published with the dataset."
)

_MEMBER_PATTERN = re.compile(r"d(?P<d>\d+)_(?P<basis>[ZX])_memory", re.IGNORECASE)


def _parse_member(member: str) -> tuple[int, str]:
    match = _MEMBER_PATTERN.search(Path(member).name)
    if not match:
        raise ValueError(
            f"cannot read the distance and basis out of {member!r}; expected a member named like "
            "d5_Z_memory.npz. Pass d= and basis= explicitly instead"
        )
    return int(match.group("d")), match.group("basis").upper()


def _check_vendor_dir(vendor_module_dir: str | Path) -> str:
    """Resolve and check the vendor path first, before anything large is opened."""
    directory = str(Path(vendor_module_dir).expanduser().resolve())
    if not Path(directory).is_dir():
        raise FileNotFoundError(
            f"{directory} is not a directory. vendor_module_dir must point at the decoding "
            "framework published with the dataset, the directory holding memory.py and "
            "noise_model.py. It is not bundled with this package"
        )
    return directory


def _vendor_simulator(vendor_module_dir: str | Path, d: int, basis: str, quadrant: Any) -> Any:
    directory = _check_vendor_dir(vendor_module_dir)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        from memory import MemorySimulator
        from noise_model import NoiseModel
    except ImportError as exc:
        raise ImportError(
            f"could not import MemorySimulator and NoiseModel from {directory}: {exc}. That "
            "directory should be the decoding framework published with the dataset"
        ) from exc
    return MemorySimulator(
        basis=basis,
        noise_model=NoiseModel(NoiseModel.DEFAULT_NOISE_PARAMS),
        d=d,
        quadrant=quadrant,
    )


def _spacetime_index(
    simulator: Any, n_detectors: int, n_ancilla: int, per_round: int, n_data: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(round, stabilizer)`` of each detector, read off the published circuit."""
    import stim

    circuit = stim.Circuit(
        "\n".join(str(i) for i in simulator.lc.cleanse_custom_instrs().instructions)
    )
    measure_ops = {"M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ"}
    targets_of: dict[int, list[int]] = {}
    measured = 0
    index = 0
    for instruction in circuit.flattened():
        if instruction.name in measure_ops:
            measured += sum(1 for t in instruction.targets_copy() if t.is_qubit_target)
        elif instruction.name == "DETECTOR":
            # A DETECTOR's targets are all measurement records, which count backwards from the
            # measurement made so far, so a negative value is what identifies one.
            targets_of[index] = [
                measured + t.value for t in instruction.targets_copy() if t.value < 0
            ]
            index += 1
    if index != n_detectors:
        raise ValueError(
            f"the published circuit declares {index} detectors but get_detectors returned "
            f"{n_detectors} columns"
        )
    det_round = np.zeros(n_detectors, dtype=np.int64)
    det_stab = np.zeros(n_detectors, dtype=np.int64)
    for detector in range(n_detectors):
        ancilla_targets = [i for i in targets_of[detector] if i < n_ancilla]
        if not ancilla_targets:
            raise ValueError(f"detector {detector} reads no ancilla measurement")
        touches_data = any(i >= n_ancilla for i in targets_of[detector])
        det_round[detector] = max(i // per_round for i in ancilla_targets) + int(touches_data)
        det_stab[detector] = max(ancilla_targets) % per_round
    del n_data
    return det_round, det_stab


def load(
    zip_path: str | Path,
    member: str,
    vendor_module_dir: str | Path,
    detectors: str = "vendor",
    *,
    d: int | None = None,
    basis: str | None = None,
    quadrant: Any = None,
) -> RecordSpec:
    """Load one memory record from the published archive.

    Parameters
    ----------
    zip_path :
        The archive as the publisher serves it. Read from, never extracted.
    member :
        Path inside the archive, e.g.
        ``Zenodo/SurfaceCodeData/Data/Distance5/d5_Z_memory.npz``.
    vendor_module_dir :
        Directory holding the decoding framework published with the dataset. Not bundled here.
    detectors :
        ``"vendor"`` calls the published ``MemorySimulator.get_detectors``. ``"naive"`` raises;
        see the module docstring.
    d, basis :
        Read out of ``member`` when not given.
    quadrant :
        Passed through to the published simulator.

    Returns
    -------
    A :class:`~qb_compiler.record.types.RecordSpec` with detectors laid out ``(n, rounds, sites)``
    and a ``detector_index`` marking which cells hold a detector, the loss symbols, the labels, the
    raw ancilla block, the final data readout, and the deterministic stabilizers when the archive's
    metadata member can be read.
    """
    if detectors == "naive":
        raise ValueError(NAIVE_REFUSED)
    if detectors != "vendor":
        raise ValueError(f"detectors must be 'vendor' or 'naive', got {detectors!r}")

    parsed_d, parsed_basis = _parse_member(member)
    d = parsed_d if d is None else d
    basis = parsed_basis if basis is None else basis
    _check_vendor_dir(vendor_module_dir)

    archive = zipfile.ZipFile(Path(zip_path).expanduser())
    with np.load(io.BytesIO(archive.read(member))) as data:
        events = np.asarray(data["measurement_events"])
        labels = np.asarray(data["observable_flips"], dtype=np.uint8)

    n_shots = int(events.shape[0])
    n_data = d * d
    n_ancilla = int(events.shape[1]) - n_data
    per_round = n_data - 1
    if n_ancilla <= 0 or n_ancilla % per_round:
        raise ValueError(
            f"{member}: {events.shape[1]} columns leave {n_ancilla} ancilla readings, which is not "
            f"a whole number of rounds of {per_round} stabilizers"
        )
    n_raw_rounds = n_ancilla // per_round

    simulator = _vendor_simulator(vendor_module_dir, d, basis, quadrant)
    vendor = np.asarray(simulator.get_detectors(events, randomise_loss=False, loss_state=0)).astype(
        np.uint8
    )
    n_detectors = int(vendor.shape[1])
    det_round, det_stab = _spacetime_index(simulator, n_detectors, n_ancilla, per_round, n_data)

    n_rounds = int(det_round.max()) + 1
    detector_index = np.full((n_rounds, per_round), -1, dtype=np.int32)
    for detector in range(n_detectors):
        cell = (int(det_round[detector]), int(det_stab[detector]))
        if detector_index[cell] >= 0:
            raise ValueError(
                f"two detectors claim round {cell[0]}, stabilizer {cell[1]}; the spacetime index "
                "is not a bijection on this record"
            )
        detector_index[cell] = detector
    grid = np.zeros((n_shots, n_rounds, per_round), dtype=np.uint8)
    for detector in range(n_detectors):
        grid[:, int(det_round[detector]), int(det_stab[detector])] = vendor[:, detector]

    deterministic = _deterministic_sites(archive, member, basis, per_round)
    raw = events[:, :n_ancilla].reshape(n_shots, n_raw_rounds, per_round).astype(np.uint8)

    meta: dict[str, Any] = {
        "loader": "quera_surface",
        "loader_version": LOADER_VERSION,
        "dataset": "quera-surface-code",
        "citation": CITATION,
        "zip_path": str(zip_path),
        "member": member,
        "d": int(d),
        "basis": basis,
        "detectors": detectors,
        "n_raw_rounds": int(n_raw_rounds),
        "per_round": int(per_round),
        "n_detectors": int(n_detectors),
        "loss_symbol": LOSS_SYMBOL,
        # The first round of this construction compares one ancilla reading against the prepared
        # stabilizer value, so it is quiet: 0.45 times the steady median on the released d5 Z
        # record. The round profile check runs on this platform because of that.
        FIRST_LAYER_PREMISE: {
            "source": (
                "the released d5 Z memory record built through the publisher's own construction: "
                "first round 0.45 times the steady median"
            )
        },
    }
    return RecordSpec(
        detectors=grid,
        detector_index=detector_index,
        labels=labels,
        loss=(events == LOSS_SYMBOL),
        raw_syndromes=raw,
        final_data=events[:, -n_data:].astype(np.uint8),
        deterministic_sites=deterministic,
        meta=meta,
    )


def _deterministic_sites(
    archive: zipfile.ZipFile, member: str, basis: str, per_round: int
) -> np.ndarray | None:
    """Which stabilizers are deterministic in this basis, from the archive's metadata member."""
    parent = str(Path(member).parent)
    match = _MEMBER_PATTERN.search(Path(member).name)
    if match is None:  # pragma: no cover - load() already parsed the member
        return None
    candidate = f"{parent}/d{int(match.group('d'))}_metadata.npz"
    try:
        payload = archive.read(candidate)
    except KeyError:
        return None
    with np.load(io.BytesIO(payload)) as metadata:
        if "ancilla_Zstab_mask" not in metadata:
            return None
        mask = np.asarray(metadata["ancilla_Zstab_mask"]).astype(bool)
    if mask.size != per_round:
        return None
    sites = np.flatnonzero(mask if basis.upper() == "Z" else ~mask)
    return sites.astype(np.int64)


__all__ = ["CITATION", "LOADER_VERSION", "LOSS_SYMBOL", "NAIVE_REFUSED", "load"]
