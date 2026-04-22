"""Compiler configuration and backend definitions.

:class:`CompilerConfig` holds all tuneable parameters for a compilation run.
:data:`BACKEND_CONFIGS` provides per-backend hardware metadata used for
noise-aware scheduling, basis-gate targeting, and cost estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from qb_compiler.exceptions import BackendNotSupportedError

# ── per-backend hardware metadata ────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Immutable hardware specification for a supported backend."""

    provider: str
    n_qubits: int
    basis_gates: tuple[str, ...]
    coupling_map: str  # human-readable topology description
    cost_per_shot: float  # USD
    median_cx_error: float  # two-qubit gate error rate (representative)
    median_readout_error: float
    t1_us: float  # median T1 in microseconds
    t2_us: float  # median T2 in microseconds

    @property
    def max_circuit_depth_heuristic(self) -> int:
        """Rough upper bound on useful depth before decoherence dominates.

        Based on T2 / (2-qubit gate time ~400 ns for superconducting,
        ~200 us for trapped-ion).  This is intentionally conservative.
        """
        gate_time_us = 200.0 if self.provider in ("ionq", "iqm_trapped_ion") else 0.4
        return max(1, int(self.t2_us / gate_time_us * 0.5))


BACKEND_CONFIGS: dict[str, BackendSpec] = {
    # ── IBM Heron (Eagle-r3 / Heron-r2) ─────────────────────────────
    "ibm_fez": BackendSpec(
        provider="ibm",
        n_qubits=156,
        basis_gates=("id", "rz", "sx", "x", "cx", "reset"),
        coupling_map="heavy-hex 156q (Heron r2)",
        cost_per_shot=0.00016,
        median_cx_error=0.005,
        median_readout_error=0.01,
        t1_us=300.0,
        t2_us=150.0,
    ),
    "ibm_torino": BackendSpec(
        provider="ibm",
        n_qubits=133,
        basis_gates=("id", "rz", "sx", "x", "cx", "reset"),
        coupling_map="heavy-hex 133q (Heron r1)",
        cost_per_shot=0.00014,
        median_cx_error=0.006,
        median_readout_error=0.012,
        t1_us=280.0,
        t2_us=130.0,
    ),
    "ibm_marrakesh": BackendSpec(
        provider="ibm",
        n_qubits=156,
        basis_gates=("id", "rz", "sx", "x", "cx", "reset"),
        coupling_map="heavy-hex 156q (Heron r2)",
        cost_per_shot=0.00016,
        median_cx_error=0.0055,
        median_readout_error=0.011,
        t1_us=290.0,
        t2_us=140.0,
    ),
    # ── Rigetti ──────────────────────────────────────────────────────
    "rigetti_ankaa": BackendSpec(
        provider="rigetti",
        n_qubits=84,
        basis_gates=("rx", "rz", "cz", "measure"),
        coupling_map="octagonal lattice 84q (Ankaa-3)",
        cost_per_shot=0.00035,
        median_cx_error=0.02,
        median_readout_error=0.03,
        t1_us=20.0,
        t2_us=10.0,
    ),
    # ── IonQ ─────────────────────────────────────────────────────────
    "ionq_aria": BackendSpec(
        provider="ionq",
        n_qubits=25,
        basis_gates=("gpi", "gpi2", "ms"),
        coupling_map="all-to-all 25q (Aria-2)",
        cost_per_shot=0.30,
        median_cx_error=0.004,
        median_readout_error=0.003,
        t1_us=1_000_000.0,  # effectively infinite for trapped-ion
        t2_us=500_000.0,
    ),
    "ionq_forte": BackendSpec(
        provider="ionq",
        n_qubits=36,
        basis_gates=("gpi", "gpi2", "ms"),
        coupling_map="all-to-all 36q (Forte-1)",
        cost_per_shot=0.30,
        median_cx_error=0.003,
        median_readout_error=0.003,
        t1_us=1_000_000.0,
        t2_us=500_000.0,
    ),
    # ── IQM ──────────────────────────────────────────────────────────
    "iqm_garnet": BackendSpec(
        provider="iqm",
        n_qubits=20,
        basis_gates=("prx", "cz", "measure"),
        coupling_map="square lattice 20q (Garnet)",
        cost_per_shot=0.00045,
        median_cx_error=0.015,
        median_readout_error=0.02,
        t1_us=30.0,
        t2_us=15.0,
    ),
    "iqm_emerald": BackendSpec(
        provider="iqm",
        n_qubits=5,
        basis_gates=("prx", "cz", "measure"),
        coupling_map="star topology 5q (Emerald)",
        cost_per_shot=0.00020,
        median_cx_error=0.008,
        median_readout_error=0.015,
        t1_us=40.0,
        t2_us=20.0,
    ),
    # ── Quantinuum ────────────────────────────────────────────────────
    "quantinuum_h2": BackendSpec(
        provider="quantinuum",
        n_qubits=32,
        basis_gates=("rz", "u1q", "zz"),
        coupling_map="all-to-all 32q (H2-1)",
        cost_per_shot=8.00,
        median_cx_error=0.001,
        median_readout_error=0.002,
        t1_us=10_000_000.0,  # effectively infinite for trapped-ion
        t2_us=1_000_000.0,
    ),
}


def get_backend_spec(backend: str) -> BackendSpec:
    """Return the :class:`BackendSpec` for *backend*, raising on unknown names."""
    try:
        return BACKEND_CONFIGS[backend]
    except KeyError:
        raise BackendNotSupportedError(backend, list(BACKEND_CONFIGS)) from None


# ── compiler configuration ───────────────────────────────────────────

_VALID_OPT_LEVELS = frozenset(range(4))


@dataclass
class CompilerConfig:
    """Full configuration for a :class:`~qb_compiler.compiler.QBCompiler` run.

    Parameters
    ----------
    backend:
        Target backend key (must be in :data:`BACKEND_CONFIGS` if not None).
    optimization_level:
        0 = no optimisation, 1 = light, 2 = standard, 3 = aggressive.
    target_basis_gates:
        Override basis gate set.  If *None*, inferred from *backend*.
    coupling_map:
        Override coupling map as adjacency list ``[(i, j), ...]``.
        If *None*, the compiler assumes all-to-all connectivity.
    calibration_max_age_hours:
        Maximum age of calibration data before it is considered stale.
    enable_calibration_aware:
        Use per-qubit/gate error rates from calibration data to guide routing.
    enable_noise_aware_scheduling:
        Reorder commuting gates to prefer lower-error time slots.
    seed:
        Reproducibility seed for stochastic passes (routing, layout search).
    """

    backend: str | None = None
    optimization_level: int = 2
    target_basis_gates: tuple[str, ...] | None = None
    coupling_map: list[tuple[int, int]] | None = None
    calibration_max_age_hours: float = 24.0
    enable_calibration_aware: bool = True
    enable_noise_aware_scheduling: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.optimization_level not in _VALID_OPT_LEVELS:
            raise ValueError(
                f"optimization_level must be in {sorted(_VALID_OPT_LEVELS)}, "
                f"got {self.optimization_level}"
            )
        if self.backend is not None and self.backend not in BACKEND_CONFIGS:
            raise BackendNotSupportedError(self.backend, list(BACKEND_CONFIGS))

    # convenience ----------------------------------------------------------

    @property
    def backend_spec(self) -> BackendSpec | None:
        """Resolved :class:`BackendSpec`, or *None* when no backend is set."""
        if self.backend is None:
            return None
        return BACKEND_CONFIGS[self.backend]

    @property
    def effective_basis_gates(self) -> tuple[str, ...] | None:
        """Basis gates to target — explicit override wins, else from backend."""
        if self.target_basis_gates is not None:
            return self.target_basis_gates
        spec = self.backend_spec
        return spec.basis_gates if spec else None

    def with_overrides(self, **kwargs: Any) -> CompilerConfig:
        """Return a shallow copy with selected fields replaced."""
        return replace(self, **kwargs)
