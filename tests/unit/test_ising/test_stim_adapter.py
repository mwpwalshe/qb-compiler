"""Tests for the stim → 4-channel tensor adapter."""

from __future__ import annotations

import numpy as np
import pytest

from qb_compiler.ising import (
    SurfaceCodePatchSpec,
    build_ising_tensor,
    resolve_layout,
    sample_and_build_tensor,
)


class TestLayoutResolution:
    @pytest.mark.parametrize("d", [3, 5, 7, 9])
    @pytest.mark.parametrize("basis", ["X", "Z"])
    def test_detector_count_matches_stim(self, d: int, basis: str) -> None:
        spec = SurfaceCodePatchSpec(distance=d, rounds=d, basis=basis)
        layout = resolve_layout(spec)
        # Every detector must be assigned (channel != -1)
        assert (layout.detector_assignments[:, 0] >= 0).all()
        # Channel is 0 (X) or 1 (Z)
        assert set(np.unique(layout.detector_assignments[:, 0])) <= {0, 1}

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_ancilla_counts(self, d: int) -> None:
        spec = SurfaceCodePatchSpec(distance=d, rounds=d, basis="X")
        layout = resolve_layout(spec)
        expected = (d * d - 1) // 2
        assert layout.x_ancilla_cells.shape == (expected, 2)
        assert layout.z_ancilla_cells.shape == (expected, 2)
        assert layout.x_ancilla_weights.shape == (expected,)
        assert layout.z_ancilla_weights.shape == (expected,)

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_weights_split(self, d: int) -> None:
        """Bulk + boundary stabiliser counts match the theoretical values."""
        spec = SurfaceCodePatchSpec(distance=d, rounds=d, basis="X")
        layout = resolve_layout(spec)
        expected_bulk = (d - 1) * (d - 1) // 2
        expected_boundary = d - 1
        n_bulk = int((layout.x_ancilla_weights == 1.0).sum())
        n_boundary = int((layout.x_ancilla_weights == 0.5).sum())
        assert n_bulk == expected_bulk
        assert n_boundary == expected_boundary
        n_bulk_z = int((layout.z_ancilla_weights == 1.0).sum())
        n_boundary_z = int((layout.z_ancilla_weights == 0.5).sum())
        assert n_bulk_z == expected_bulk
        assert n_boundary_z == expected_boundary

    def test_orientation_fingerprint_stable(self) -> None:
        spec = SurfaceCodePatchSpec(distance=5, rounds=5, basis="X")
        fp1 = resolve_layout(spec).orientation_fingerprint
        fp2 = resolve_layout(spec).orientation_fingerprint
        assert fp1 == fp2
        assert isinstance(fp1, str)
        assert len(fp1) == 16


class TestTensorBuild:
    @pytest.mark.parametrize("d", [3, 5, 7])
    @pytest.mark.parametrize("basis", ["X", "Z"])
    def test_tensor_shape_matches_spec(self, d: int, basis: str) -> None:
        spec = SurfaceCodePatchSpec(distance=d, rounds=d, basis=basis)
        tensor, obs = sample_and_build_tensor(spec, shots=4, seed=0)
        assert tensor.shape == (4, 4, d, d, d)
        assert tensor.dtype == np.float32
        assert obs.shape == (4, 1)

    def test_wrong_basis_endpoint_rounds_zeroed(self) -> None:
        spec = SurfaceCodePatchSpec(distance=5, rounds=5, basis="X")
        tensor, _ = sample_and_build_tensor(spec, shots=16, seed=1)
        # X-basis: z_type (channel 1) must be zero on round 0 and round -1
        assert (tensor[:, 1, 0] == 0).all()
        assert (tensor[:, 1, -1] == 0).all()
        # Channel 3 presence-mask for z must also be zero on those rounds
        assert (tensor[:, 3, 0] == 0).all()
        assert (tensor[:, 3, -1] == 0).all()

    def test_presence_masks_are_time_invariant_for_matching_basis(self) -> None:
        spec = SurfaceCodePatchSpec(distance=5, rounds=5, basis="X")
        tensor, _ = sample_and_build_tensor(spec, shots=1, seed=0)
        # Matching-basis presence (channel 2 for X) identical across rounds
        x_presence = tensor[0, 2]
        for t in range(1, spec.rounds):
            np.testing.assert_array_equal(x_presence[t], x_presence[0])

    def test_presence_mask_totals(self) -> None:
        """Sum of presence mask matches bulk + boundary accounting."""
        for d in [3, 5, 7]:
            spec = SurfaceCodePatchSpec(distance=d, rounds=d, basis="X")
            tensor, _ = sample_and_build_tensor(spec, shots=1, seed=0)
            bulk = (d - 1) ** 2 // 2
            boundary = d - 1
            per_round = bulk * 1.0 + boundary * 0.5
            # X presence active in all rounds
            expected_x = per_round * d
            x_sum = tensor[0, 2].sum()
            assert np.isclose(x_sum, expected_x), f"d={d}: x {x_sum} != {expected_x}"
            # Z presence zeroed on first+last rounds (wrong basis)
            active_z_rounds = max(d - 2, 1)
            expected_z = per_round * active_z_rounds if d > 1 else per_round * d
            z_sum = tensor[0, 3].sum()
            assert np.isclose(z_sum, expected_z), f"d={d}: z {z_sum} != {expected_z}"

    def test_scatter_xors_collisions(self) -> None:
        """Collisions (rounds+1 stim vs rounds NVIDIA) XOR-accumulate.

        Stim emits ``rounds+1`` logical time slots (the last detector
        carries the reconciliation with data-qubit measurements).  The
        adapter folds that slot into round ``rounds-1`` via XOR, so if
        two detectors share the same ``(channel, round, row, col)`` the
        tensor value equals their XOR.  Synthesise one such collision
        and verify.
        """
        spec = SurfaceCodePatchSpec(distance=5, rounds=5, basis="Z")
        layout = resolve_layout(spec)
        assignments = layout.detector_assignments
        # Find a colliding pair (two detectors → same cell).
        from collections import defaultdict

        slot_to_dets: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
        for det, row in enumerate(assignments):
            slot_to_dets[tuple(int(x) for x in row)].append(det)
        colliding = [(slot, ds) for slot, ds in slot_to_dets.items() if len(ds) >= 2]
        assert colliding, "expected at least one collision for d=5, T=5"

        slot, dets = colliding[0]
        n_dets = assignments.shape[0]
        # Case 1: only the first detector flips → tensor == 1
        synth = np.zeros((1, n_dets), dtype=np.uint8)
        synth[0, dets[0]] = 1
        t1 = build_ising_tensor(spec, synth)
        ch, rnd, r, c = slot
        assert t1[0, ch, rnd, r, c] == 1.0
        # Case 2: both flip → XOR == 0
        synth[0, dets[1]] = 1
        t2 = build_ising_tensor(spec, synth)
        assert t2[0, ch, rnd, r, c] == 0.0

    def test_non_colliding_detectors_preserved_exactly(self) -> None:
        """Detectors whose slot has no collision recover exactly."""
        spec = SurfaceCodePatchSpec(distance=5, rounds=5, basis="Z")
        layout = resolve_layout(spec)
        assignments = layout.detector_assignments
        from collections import Counter

        slot_counts = Counter(tuple(int(x) for x in r) for r in assignments)
        unique_dets = [
            d for d, row in enumerate(assignments) if slot_counts[tuple(int(x) for x in row)] == 1
        ]
        assert unique_dets, "expected unique-slot detectors"

        rng = np.random.default_rng(2026)
        events = rng.integers(0, 2, size=(4, assignments.shape[0]), dtype=np.uint8)
        tensor = build_ising_tensor(spec, events)
        for det in unique_dets:
            ch, rnd, row, col = (int(x) for x in assignments[det])
            # Wrong-basis endpoint slots are zeroed by basis-zeroing rule.
            if spec.basis == "Z" and ch == 0 and rnd in (0, spec.rounds - 1):
                continue
            if spec.basis == "X" and ch == 1 and rnd in (0, spec.rounds - 1):
                continue
            np.testing.assert_array_equal(
                tensor[:, ch, rnd, row, col].astype(np.uint8),
                events[:, det],
            )

    def test_detector_events_shape_mismatch_raises(self) -> None:
        spec = SurfaceCodePatchSpec(distance=3, rounds=3, basis="X")
        bad = np.zeros((5, 999), dtype=np.uint8)
        with pytest.raises(ValueError, match="detectors"):
            build_ising_tensor(spec, bad)

    def test_detector_events_1d_raises(self) -> None:
        spec = SurfaceCodePatchSpec(distance=3, rounds=3, basis="X")
        bad = np.zeros(10, dtype=np.uint8)
        with pytest.raises(ValueError, match="2-D"):
            build_ising_tensor(spec, bad)
