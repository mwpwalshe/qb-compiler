"""Surface-code patch specification for the Ising-Decoder integration.

A :class:`SurfaceCodePatchSpec` captures everything needed to build a
stim surface-code memory experiment and prepare the 4-channel input
tensor consumed by NVIDIA's Ising-Decoder-SurfaceCode-1 family.

The spec is intentionally minimal — it carries only the physical
parameters the decoder's training data was parameterised over.  It
does not carry layout orientation flags beyond ``"rotated_memory_x"``
/ ``"rotated_memory_z"`` because those are the two code families the
pretrained weights cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Basis = Literal["X", "Z"]


@dataclass(frozen=True)
class SurfaceCodePatchSpec:
    """Parameters of a rotated surface-code memory experiment.

    Parameters
    ----------
    distance:
        Code distance ``d``.  Must be odd and ``>= 3``.  The resulting
        patch has ``d * d`` data qubits and ``d * d - 1`` ancilla qubits
        (``(d**2 - 1) // 2`` X-type and as many Z-type).
    rounds:
        Number of stabiliser-measurement rounds ``T``.  Must be ``>= 1``.
        Round 0 is the state-preparation round; round ``T-1`` closes the
        experiment with data-qubit measurements.
    basis:
        Logical basis of the memory experiment.  ``"X"`` prepares
        ``|+>_L`` and measures in ``X``; ``"Z"`` prepares ``|0>_L`` and
        measures in ``Z``.
    p_error:
        Physical gate-error rate to inject at the circuit level (used as
        ``after_clifford_depolarization`` and matching readout/reset
        errors in :func:`stim.Circuit.generated`).  Must be in ``[0, 1]``.

    Notes
    -----
    The pretrained Ising-Decoder weights were trained on
    circuit-level noise; ``p_error`` should typically lie in
    ``[1e-4, 5e-3]`` for results to be meaningful.  At ``p_error > 0.02``
    the decoder has not been validated.

    Examples
    --------
    >>> spec = SurfaceCodePatchSpec(distance=5, rounds=5, basis="X", p_error=0.003)
    >>> spec.num_data_qubits
    25
    >>> spec.num_ancillas_per_basis
    12
    """

    distance: int
    rounds: int
    basis: Basis
    p_error: float = 0.003

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError(f"distance must be an odd integer >= 3, got {self.distance}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")
        if self.basis not in ("X", "Z"):
            raise ValueError(f"basis must be 'X' or 'Z', got {self.basis!r}")
        if not (0.0 <= self.p_error <= 1.0):
            raise ValueError(f"p_error must be in [0, 1], got {self.p_error}")

    @property
    def num_data_qubits(self) -> int:
        return self.distance * self.distance

    @property
    def num_ancillas_per_basis(self) -> int:
        return (self.distance * self.distance - 1) // 2

    @property
    def stim_task_name(self) -> str:
        """Return the stim ``Circuit.generated`` task name."""
        return f"surface_code:rotated_memory_{self.basis.lower()}"

    @property
    def tensor_shape(self) -> tuple[int, int, int, int]:
        """Shape of the 4-channel input tensor, excluding the batch dim.

        Returns
        -------
        (channels, rounds, distance, distance)
        """
        return (4, self.rounds, self.distance, self.distance)
