"""ML-powered noise model — uses QubitBoost V13 multi-vendor models.

This is a **proprietary integration stub**.  Full functionality requires
the ``qubitboost-sdk`` package (``pip install qb-compiler[qubitboost]``).

Uses QubitBoost V13 multi-vendor ML models trained on 8 backends
(IBM Fez, Torino, Marrakesh; IonQ Aria, Forte; IQM Garnet, Emerald;
Rigetti Ankaa) to predict noise characteristics beyond what calibration
data alone provides.  The ML model captures:

- Cross-talk correlations between neighbouring qubits
- Time-varying drift patterns in gate errors
- Non-Markovian noise signatures in repeated circuits
- Backend-specific decoherence profiles

This enables more accurate fidelity prediction and better-informed
routing decisions compared to the purely empirical
:class:`~qb_compiler.noise.empirical_model.EmpiricalNoiseModel`.
"""

from __future__ import annotations

from qb_compiler.noise.noise_model import NoiseModel

try:
    from qubitboost_sdk.noise_profiles import NoiseProfiler

    _HAS_QUBITBOOST = True
except ImportError:
    _HAS_QUBITBOOST = False


class MLNoiseModel(NoiseModel):
    """Noise model powered by QubitBoost V13 ML noise profiles.

    Wraps :class:`qubitboost_sdk.noise_profiles.NoiseProfiler` to
    provide the :class:`NoiseModel` interface.  The ML model is trained
    on calibration data from 8 quantum backends and can predict:

    - Per-qubit error rates with cross-talk corrections
    - Gate error rates accounting for time-of-day drift
    - Readout error with state-dependent asymmetry
    - Decoherence beyond simple T1/T2 exponential decay

    Parameters
    ----------
    backend:
        Target backend identifier (e.g. ``"ibm_fez"``).
    model_version:
        V13 model variant.  Default is ``"v13_latest"`` which
        auto-selects the best model for the backend.

    Raises
    ------
    ImportError
        If ``qubitboost-sdk`` is not installed.
    """

    def __init__(
        self,
        backend: str,
        *,
        model_version: str = "v13_latest",
    ) -> None:
        if not _HAS_QUBITBOOST:
            raise ImportError(
                "ML noise model requires qubitboost-sdk. "
                "Install with: pip install qb-compiler[qubitboost]"
            )

        self._backend = backend
        self._profiler = NoiseProfiler(
            backend=backend,
            model_version=model_version,
        )

    # ── NoiseModel interface ──────────────────────────────────────────

    def qubit_error(self, qubit: int) -> float:
        """ML-predicted combined qubit error with cross-talk corrections."""
        return self._profiler.predict_qubit_error(qubit)  # type: ignore[no-any-return]

    def gate_error(self, gate: str, qubits: tuple[int, ...]) -> float:
        """ML-predicted gate error accounting for time-varying drift."""
        return self._profiler.predict_gate_error(gate, qubits)  # type: ignore[no-any-return]

    def readout_error(self, qubit: int) -> float:
        """ML-predicted readout error with state-dependent asymmetry."""
        return self._profiler.predict_readout_error(qubit)  # type: ignore[no-any-return]

    def decoherence_factor(self, qubit: int, gate_time_ns: float) -> float:
        """ML-predicted decoherence beyond simple T1/T2 exponential."""
        return self._profiler.predict_decoherence(qubit, gate_time_ns)  # type: ignore[no-any-return]

    # ── Extended API ──────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Backend this model is configured for."""
        return self._backend

    @property
    def model_version(self) -> str:
        """Version string of the loaded ML model."""
        return self._profiler.model_version  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return (
            f"MLNoiseModel(backend={self._backend!r}, "
            f"version={self.model_version!r})"
        )
