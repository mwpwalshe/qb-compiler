"""Tests for :class:`qb_compiler.ising.SurfaceCodePatchSpec`."""

from __future__ import annotations

import pytest

from qb_compiler.ising import SurfaceCodePatchSpec


class TestSurfaceCodePatchSpec:
    def test_minimal_spec(self) -> None:
        spec = SurfaceCodePatchSpec(distance=3, rounds=3, basis="X")
        assert spec.distance == 3
        assert spec.rounds == 3
        assert spec.basis == "X"
        assert spec.p_error == 0.003
        assert spec.num_data_qubits == 9
        assert spec.num_ancillas_per_basis == 4
        assert spec.tensor_shape == (4, 3, 3, 3)
        assert spec.stim_task_name == "surface_code:rotated_memory_x"

    def test_z_basis_name(self) -> None:
        spec = SurfaceCodePatchSpec(distance=5, rounds=7, basis="Z", p_error=0.001)
        assert spec.stim_task_name == "surface_code:rotated_memory_z"
        assert spec.tensor_shape == (4, 7, 5, 5)

    @pytest.mark.parametrize("bad", [0, 1, 2, 4, -1])
    def test_even_or_small_distance_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="distance"):
            SurfaceCodePatchSpec(distance=bad, rounds=3, basis="X")

    def test_non_positive_rounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="rounds"):
            SurfaceCodePatchSpec(distance=3, rounds=0, basis="X")

    @pytest.mark.parametrize("bad", ["Y", "x", "z", "XZ", ""])
    def test_bad_basis_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="basis"):
            SurfaceCodePatchSpec(distance=3, rounds=3, basis=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_p_error_out_of_range_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="p_error"):
            SurfaceCodePatchSpec(distance=3, rounds=3, basis="X", p_error=bad)

    def test_frozen_dataclass(self) -> None:
        import dataclasses

        spec = SurfaceCodePatchSpec(distance=3, rounds=3, basis="X")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.distance = 5  # type: ignore[misc]
