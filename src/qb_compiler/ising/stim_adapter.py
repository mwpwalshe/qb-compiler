"""Stim → 4-channel surface-code tensor builder.

Produces numpy tensors of shape ``(batch, 4, rounds, distance, distance)``
matching the semantics described in the NVIDIA Ising-Decoder-SurfaceCode-1
documentation (channels ``[x_type, z_type, x_present, z_present]``,
values = XOR of successive-round syndromes with presence masks encoding
both stabiliser weight and basis-round validity).

This is a clean-room reimplementation — it uses stim's native detector
coordinates and ``compile_detector_sampler`` rather than NVIDIA's
custom stab-index maps.  Tensors are shape-compatible with the
pretrained Ising-Decoder models; an orientation check is recommended
before feeding qb-compiler tensors to pretrained NVIDIA weights (see
``SurfaceCodeTensorLayout.orientation_fingerprint``).

This module depends only on ``stim`` and ``numpy``; ``torch`` is not
required here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import stim

from qb_compiler.ising.patch_spec import SurfaceCodePatchSpec

# ═══════════════════════════════════════════════════════════════════════
# Layout resolution
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SurfaceCodeTensorLayout:
    """Mapping from stim detector coordinates to (channel, round, row, col).

    Frozen by ``(distance, rounds, basis)`` — cached because the
    resolution is deterministic per stim task and gets expensive for
    large distances.

    Attributes
    ----------
    x_ancilla_cells:
        Array of shape ``(num_x_ancillas, 2)``; row ``i`` is the
        ``(row, col)`` grid cell assigned to X-ancilla ``i``.
    z_ancilla_cells:
        Same, for Z-type ancillas.
    x_ancilla_weights:
        Array of shape ``(num_x_ancillas,)`` float32.  Entry ``i`` is
        ``1.0`` if ancilla ``i`` has four data-qubit neighbours
        (bulk stabiliser) and ``0.5`` if it has two (boundary
        stabiliser).  Mirrors NVIDIA's ``normalized_weight_mapping``.
    z_ancilla_weights:
        Same, for Z-type ancillas.
    detector_assignments:
        Array of shape ``(num_detectors, 4)`` with dtype int32: columns
        are ``(channel, round, row, col)``.  ``channel`` is ``0`` for
        X-type and ``1`` for Z-type.
    orientation_fingerprint:
        Short hex string derived from the ancilla→grid mapping — useful
        for detecting orientation mismatches against a reference
        (e.g. NVIDIA pretrained weights).
    """

    distance: int
    rounds: int
    basis: str
    x_ancilla_cells: np.ndarray
    z_ancilla_cells: np.ndarray
    x_ancilla_weights: np.ndarray
    z_ancilla_weights: np.ndarray
    detector_assignments: np.ndarray
    orientation_fingerprint: str


@lru_cache(maxsize=64)
def _resolve_layout(
    distance: int,
    rounds: int,
    basis: str,
    p_error: float,
) -> SurfaceCodeTensorLayout:
    """Inspect a stim circuit and build the detector → grid-cell map."""
    spec = SurfaceCodePatchSpec(
        distance=distance, rounds=rounds, basis=basis, p_error=p_error
    )
    circuit = stim.Circuit.generated(
        spec.stim_task_name,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_error,
        after_reset_flip_probability=p_error,
        before_measure_flip_probability=p_error,
        before_round_data_depolarization=p_error,
    )

    # Collect detector coords: {det_idx: (x, y, t)}
    det_coords = circuit.get_detector_coordinates()
    # State-preparation round is round 0; the matching-basis detectors
    # are the only ones present there.  Use those to identify which
    # (x, y) positions are the matching-basis ancillas.
    matching_positions: set[tuple[float, float]] = set()
    for coord in det_coords.values():
        if len(coord) < 3:
            continue
        if round(coord[2]) == 0:
            matching_positions.add((float(coord[0]), float(coord[1])))

    # The other detectors (from rounds >= 1) include both bases.  The
    # non-matching positions must therefore be the opposite-basis
    # ancillas.
    other_positions: set[tuple[float, float]] = set()
    for coord in det_coords.values():
        if len(coord) < 3:
            continue
        other_positions.add((float(coord[0]), float(coord[1])))
    opposite_positions = other_positions - matching_positions

    if basis == "X":
        x_positions = sorted(matching_positions)
        z_positions = sorted(opposite_positions)
    else:
        z_positions = sorted(matching_positions)
        x_positions = sorted(opposite_positions)

    # Map each ancilla position to a (row, col) grid cell and record
    # its stabiliser weight.  Data qubits lie at stim coords
    # ``(2c+1, 2r+1)`` for ``r, c in 0..d-1``.  Each ancilla has up to
    # four neighbouring data qubits; we assign it to the smallest
    # in-bounds (row, col) neighbour.  A bulk stabiliser has four
    # in-bounds neighbours (weight 1.0); a boundary stabiliser has two
    # (weight 0.5).
    def _ancilla_to_cell_and_weight(
        ax: float, ay: float
    ) -> tuple[tuple[int, int], float]:
        neighbours: list[tuple[int, int]] = []
        for dx in (-1, +1):
            for dy in (-1, +1):
                col = (ax + dx - 1) // 2
                row = (ay + dy - 1) // 2
                if 0 <= row < distance and 0 <= col < distance:
                    neighbours.append((int(row), int(col)))
        if not neighbours:
            raise RuntimeError(
                f"Ancilla at ({ax}, {ay}) has no in-bounds neighbour for d={distance}"
            )
        neighbours.sort()
        weight = 1.0 if len(neighbours) == 4 else 0.5
        return neighbours[0], weight

    x_cells_list: list[tuple[int, int]] = []
    x_weights_list: list[float] = []
    for p in x_positions:
        cell, w = _ancilla_to_cell_and_weight(*p)
        x_cells_list.append(cell)
        x_weights_list.append(w)
    z_cells_list: list[tuple[int, int]] = []
    z_weights_list: list[float] = []
    for p in z_positions:
        cell, w = _ancilla_to_cell_and_weight(*p)
        z_cells_list.append(cell)
        z_weights_list.append(w)
    x_cells = np.array(x_cells_list, dtype=np.int32)
    z_cells = np.array(z_cells_list, dtype=np.int32)
    x_weights = np.array(x_weights_list, dtype=np.float32)
    z_weights = np.array(z_weights_list, dtype=np.float32)

    # Position → index lookup
    x_pos_to_idx = {p: i for i, p in enumerate(x_positions)}
    z_pos_to_idx = {p: i for i, p in enumerate(z_positions)}

    # Build detector assignments: (num_detectors, 4) = (channel, round, row, col)
    num_dets = len(det_coords)
    assignments = np.full((num_dets, 4), -1, dtype=np.int32)
    for det_idx in sorted(det_coords.keys()):
        coord = det_coords[det_idx]
        if len(coord) < 3:
            continue
        pos = (float(coord[0]), float(coord[1]))
        rnd = round(coord[2])
        if rnd < 0 or rnd >= rounds:
            # The final "logical reconciliation" detector can have
            # ``t == rounds``; push it into the last round slot so its
            # parity still reaches the tensor.
            rnd = rounds - 1
        if pos in x_pos_to_idx:
            idx = x_pos_to_idx[pos]
            row, col = x_cells[idx]
            assignments[det_idx] = (0, rnd, row, col)
        elif pos in z_pos_to_idx:
            idx = z_pos_to_idx[pos]
            row, col = z_cells[idx]
            assignments[det_idx] = (1, rnd, row, col)
        else:
            raise RuntimeError(
                f"Detector {det_idx} at {pos} was not classified as X or Z"
            )

    # Fingerprint: short digest of ancilla cell arrays so users can
    # detect orientation drift against a reference layout.
    import hashlib

    fp_bytes = (
        x_cells.tobytes()
        + b"|"
        + z_cells.tobytes()
        + b"|"
        + x_weights.tobytes()
        + b"|"
        + z_weights.tobytes()
    )
    fingerprint = hashlib.sha256(fp_bytes).hexdigest()[:16]

    return SurfaceCodeTensorLayout(
        distance=distance,
        rounds=rounds,
        basis=basis,
        x_ancilla_cells=x_cells,
        z_ancilla_cells=z_cells,
        x_ancilla_weights=x_weights,
        z_ancilla_weights=z_weights,
        detector_assignments=assignments,
        orientation_fingerprint=fingerprint,
    )


def resolve_layout(spec: SurfaceCodePatchSpec) -> SurfaceCodeTensorLayout:
    """Return the detector → grid-cell layout for *spec* (cached)."""
    return _resolve_layout(spec.distance, spec.rounds, spec.basis, spec.p_error)


# ═══════════════════════════════════════════════════════════════════════
# Presence masks
# ═══════════════════════════════════════════════════════════════════════


def _presence_mask_for_channel(
    ancilla_cells: np.ndarray,
    ancilla_weights: np.ndarray,
    distance: int,
    rounds: int,
    is_wrong_basis: bool,
) -> np.ndarray:
    """Build a ``(rounds, distance, distance)`` presence mask.

    Uses the per-ancilla weight computed during layout resolution: bulk
    stabilisers (four data-qubit neighbours) map to ``1.0`` and boundary
    stabilisers (two neighbours) to ``0.5``.  When ``is_wrong_basis`` is
    true the first and last rounds are zeroed — those rounds contain
    no reliable wrong-basis parity information in a memory experiment.
    """
    per_round = np.zeros((distance, distance), dtype=np.float32)
    for (row, col), w in zip(ancilla_cells, ancilla_weights, strict=True):
        per_round[int(row), int(col)] = float(w)

    mask = np.broadcast_to(per_round, (rounds, distance, distance)).copy()
    if is_wrong_basis:
        mask[0] = 0.0
        if rounds > 1:
            mask[-1] = 0.0
    return mask


# ═══════════════════════════════════════════════════════════════════════
# Public tensor builder
# ═══════════════════════════════════════════════════════════════════════


def build_ising_tensor(
    spec: SurfaceCodePatchSpec,
    detector_events: np.ndarray,
) -> np.ndarray:
    """Convert stim detector events into the 4-channel Ising-Decoder tensor.

    Parameters
    ----------
    spec:
        The :class:`SurfaceCodePatchSpec` describing the experiment.
    detector_events:
        Array of shape ``(batch, num_detectors)`` with ``bool`` or
        ``{0, 1}`` int dtype — the output of
        :meth:`stim.CompiledDetectorSampler.sample` for a circuit built
        from *spec*.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape
        ``(batch, 4, rounds, distance, distance)``.  Channels are
        ``[x_type, z_type, x_present, z_present]``; values already
        encode parity flips between consecutive rounds (stim detectors
        are XOR of successive measurements).

    Notes
    -----
    Stim's detector events are already XOR'd between rounds, so the
    channel-0/1 values map directly onto the decoder's expected input
    (no additional XOR-with-previous step is needed).
    """
    if detector_events.ndim != 2:
        raise ValueError(
            f"detector_events must be 2-D (batch, num_detectors); "
            f"got shape {detector_events.shape}"
        )

    layout = resolve_layout(spec)
    assignments = layout.detector_assignments
    batch = detector_events.shape[0]
    expected_dets = assignments.shape[0]
    if detector_events.shape[1] != expected_dets:
        raise ValueError(
            f"detector_events has {detector_events.shape[1]} detectors "
            f"but the layout for d={spec.distance}, T={spec.rounds}, "
            f"basis={spec.basis!r} expects {expected_dets}"
        )

    events = detector_events.astype(np.uint8)
    scratch = np.zeros(
        (batch, 4, spec.rounds, spec.distance, spec.distance),
        dtype=np.uint8,
    )

    # Scatter: for each detector, XOR-accumulate its batch column into
    # the (channel, round, row, col) slot.  Stim's detector-coordinate
    # convention produces rounds+1 logical time slots (the final
    # "reconciliation" detector at t=rounds carries the parity of the
    # last ancilla readout against the data-qubit measurements).
    # NVIDIA's tensor uses exactly ``rounds`` time slots and folds
    # reconciliation into the last slot.  XOR-accumulation is the
    # information-preserving merge for binary parity bits on the same
    # ancilla+round cell.
    rows_idx = assignments[:, 0]  # channel
    rnd_idx = assignments[:, 1]  # round
    r_idx = assignments[:, 2]  # row
    c_idx = assignments[:, 3]  # col
    for det in range(expected_dets):
        if rows_idx[det] < 0:
            continue
        scratch[
            :, rows_idx[det], rnd_idx[det], r_idx[det], c_idx[det]
        ] ^= events[:, det]

    out = np.zeros(
        (batch, 4, spec.rounds, spec.distance, spec.distance),
        dtype=np.float32,
    )
    out[:, :2] = scratch[:, :2].astype(np.float32)

    # Presence masks
    x_wrong = spec.basis != "X"
    z_wrong = spec.basis != "Z"
    x_presence = _presence_mask_for_channel(
        layout.x_ancilla_cells,
        layout.x_ancilla_weights,
        spec.distance,
        spec.rounds,
        x_wrong,
    )
    z_presence = _presence_mask_for_channel(
        layout.z_ancilla_cells,
        layout.z_ancilla_weights,
        spec.distance,
        spec.rounds,
        z_wrong,
    )
    out[:, 2] = x_presence[None, :, :, :]
    out[:, 3] = z_presence[None, :, :, :]

    # Basis-zeroing rule: in a basis=X memory experiment, z_type first
    # and last rounds carry no reliable parity — zero them explicitly.
    if spec.basis == "X":
        out[:, 1, 0] = 0.0
        if spec.rounds > 1:
            out[:, 1, -1] = 0.0
    else:
        out[:, 0, 0] = 0.0
        if spec.rounds > 1:
            out[:, 0, -1] = 0.0

    return out


def sample_and_build_tensor(
    spec: SurfaceCodePatchSpec,
    shots: int,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample detector events from a stim circuit and build the tensor.

    Returns
    -------
    tensor:
        Float32 array of shape ``(shots, 4, rounds, distance, distance)``.
    observables:
        Bool array of shape ``(shots, num_observables)``.  The
        corresponding logical-observable flips — used as ground-truth
        labels when comparing decoder accuracy.
    """
    circuit = stim.Circuit.generated(
        spec.stim_task_name,
        distance=spec.distance,
        rounds=spec.rounds,
        after_clifford_depolarization=spec.p_error,
        after_reset_flip_probability=spec.p_error,
        before_measure_flip_probability=spec.p_error,
        before_round_data_depolarization=spec.p_error,
    )
    sampler = circuit.compile_detector_sampler(seed=seed)
    det_events, obs_flips = sampler.sample(shots, separate_observables=True)
    tensor = build_ising_tensor(spec, det_events)
    return tensor, obs_flips
