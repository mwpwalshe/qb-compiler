"""Decoder-agnostic interface for surface-code memory experiments.

Provides a small protocol :class:`SurfaceCodeDecoder` with two
implementations:

* :class:`PyMatchingDecoder` — MWPM baseline built directly from the
  stim detector-error model.  Always available (``pymatching`` is a
  required runtime dep of :mod:`qb_compiler.ising`).
* :class:`IsingDecoderWrapper` — optional wrapper around NVIDIA's
  Ising-Decoder-SurfaceCode-1 pre-decoder chain.  Uses
  ``torch``/``safetensors`` + user-supplied weights (gated on
  HuggingFace under the NVIDIA Open Model License); graceful
  ``ImportError`` if torch is not installed.

The NVIDIA pre-decoder is NOT a standalone logical-error predictor —
it emits a 4-channel correction tensor that qb-compiler XORs onto the
input syndromes before feeding the residual to PyMatching.  Both
pipelines therefore return the same object type (an
:class:`np.ndarray` of shape ``(batch, num_observables)`` with bool
dtype) so they can be compared head-to-head by
:func:`evaluate_logical_error_rate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import stim

if TYPE_CHECKING:
    from qb_compiler.ising.patch_spec import SurfaceCodePatchSpec


class SurfaceCodeDecoder(Protocol):
    """Minimal decoder protocol used by qb-compiler's Ising integration."""

    spec: SurfaceCodePatchSpec

    def decode(self, detector_events: np.ndarray) -> np.ndarray:
        """Predict the observable flips for each shot.

        Parameters
        ----------
        detector_events:
            Bool/int array of shape ``(batch, num_detectors)``.

        Returns
        -------
        numpy.ndarray
            Bool array of shape ``(batch, num_observables)``.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
# PyMatching baseline
# ═══════════════════════════════════════════════════════════════════════


def _build_stim_circuit(spec: SurfaceCodePatchSpec) -> stim.Circuit:
    return stim.Circuit.generated(
        spec.stim_task_name,
        distance=spec.distance,
        rounds=spec.rounds,
        after_clifford_depolarization=spec.p_error,
        after_reset_flip_probability=spec.p_error,
        before_measure_flip_probability=spec.p_error,
        before_round_data_depolarization=spec.p_error,
    )


class PyMatchingDecoder:
    """MWPM decoder using ``pymatching`` and a stim DEM."""

    def __init__(self, spec: SurfaceCodePatchSpec) -> None:
        import pymatching

        self.spec = spec
        circuit = _build_stim_circuit(spec)
        dem = circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
        self._matching = pymatching.Matching.from_detector_error_model(dem)

    def decode(self, detector_events: np.ndarray) -> np.ndarray:
        if detector_events.ndim != 2:
            raise ValueError(
                f"detector_events must be 2-D, got shape {detector_events.shape}"
            )
        preds = self._matching.decode_batch(detector_events.astype(np.uint8))
        # pymatching returns (batch, num_observables) uint8 0/1
        return preds.astype(bool)


# ═══════════════════════════════════════════════════════════════════════
# NVIDIA Ising wrapper (optional — requires torch + weights)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class IsingDecoderConfig:
    """Configuration for the NVIDIA Ising pre-decoder wrapper.

    Parameters
    ----------
    weights_path:
        Filesystem path to a ``.pt`` or ``.safetensors`` checkpoint
        downloaded by the user from HuggingFace (gated under the NVIDIA
        Open Model License).  Recognised filenames:
        ``Ising-Decoder-SurfaceCode-1-Fast.pt``,
        ``Ising-Decoder-SurfaceCode-1-Accurate.pt``.
    device:
        Torch device string - ``"cpu"`` or ``"cuda"``.  Defaults to
        ``"cpu"`` because the decoder is small (900K-1.8M params) and
        gives correct results on CPU.
    build_model:
        Optional callable that returns the torch ``nn.Module`` given the
        :class:`SurfaceCodePatchSpec` — lets users plug in their own
        architecture if they are tracking the NVIDIA reference repo.
        If omitted, the wrapper raises :class:`NotImplementedError` with
        clear instructions on where to get the model definition.
    """

    weights_path: str
    device: str = "cpu"
    build_model: object = None


class IsingDecoderWrapper:
    """Wrap the NVIDIA Ising pre-decoder + PyMatching residual chain.

    The pre-decoder is consumed in pre-correction mode: its 4-channel
    output is binarised and XOR'd onto the input detector-event tensor
    (on the ``x_type`` / ``z_type`` channels), then the residual
    detector events are fed into the MWPM decoder built from the stim
    DEM.  This mirrors NVIDIA's own ablation harness in
    ``code/evaluation/logical_error_rate.py``.

    Users must supply:

    1. ``torch`` (install ``qb-compiler[ising-nvidia]`` or ``pip install torch``).
    2. A checkpoint file downloaded from HuggingFace.
    3. A callable ``build_model(spec)`` that instantiates the
       pre-decoder module (the Apache-2.0 model definition lives in
       NVIDIA's public repo; we do not vendor it).

    The wrapper emits informative errors if any of these are missing.
    """

    def __init__(
        self,
        spec: SurfaceCodePatchSpec,
        config: IsingDecoderConfig,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "IsingDecoderWrapper requires torch.  Install via "
                "`pip install qb-compiler[ising-nvidia]`."
            ) from exc

        if config.build_model is None:
            raise NotImplementedError(
                "IsingDecoderConfig.build_model is None.  qb-compiler does "
                "not vendor NVIDIA's model definition.  Supply a callable "
                "`build_model(spec) -> torch.nn.Module` sourced from "
                "https://github.com/NVIDIA/Ising-Decoding (Apache 2.0, "
                "model in code/model/predecoder.py)."
            )

        import torch

        self.spec = spec
        self.config = config
        self.device = torch.device(config.device)
        self._model = config.build_model(spec).to(self.device).eval()
        self._load_checkpoint(config.weights_path)
        self._baseline = PyMatchingDecoder(spec)

    def _load_checkpoint(self, path: str) -> None:
        import torch

        if path.endswith(".safetensors"):
            try:
                import safetensors.torch as st  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "safetensors is required to load .safetensors weights. "
                    "Install via `pip install qb-compiler[ising-nvidia]`."
                ) from exc
            state = st.load_file(path, device=self.config.device)
        else:
            state = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # Strip DDP 'module.' prefix if present
            if any(k.startswith("module.") for k in state):
                state = {k[len("module.") :]: v for k, v in state.items()}
        self._model.load_state_dict(state, strict=False)

    def _pre_decode(self, detector_events: np.ndarray) -> np.ndarray:
        """Run the pre-decoder and return XOR-corrected detector events."""
        import torch

        from qb_compiler.ising.stim_adapter import (
            build_ising_tensor,
            resolve_layout,
        )

        tensor = build_ising_tensor(self.spec, detector_events)
        t = torch.from_numpy(tensor).to(self.device)
        with torch.no_grad():
            out = self._model(t).cpu().numpy()
        # The pre-decoder emits 4-channel logits; the two syndrome-like
        # channels are at positions (2, 3) = (syn_x, syn_z) per the
        # NVIDIA ablation harness.  We threshold at zero to produce a
        # binary correction and XOR back onto the original x_type /
        # z_type channels.
        syn_x_corr = (out[:, 2] > 0).astype(np.uint8)
        syn_z_corr = (out[:, 3] > 0).astype(np.uint8)
        x_type = (tensor[:, 0].astype(np.uint8) ^ syn_x_corr).astype(np.uint8)
        z_type = (tensor[:, 1].astype(np.uint8) ^ syn_z_corr).astype(np.uint8)

        # Re-project (channel, round, row, col) back to detector order
        layout = resolve_layout(self.spec)
        num_dets = layout.detector_assignments.shape[0]
        batch = detector_events.shape[0]
        residual = np.zeros((batch, num_dets), dtype=np.uint8)
        for det, (ch, rnd, row, col) in enumerate(layout.detector_assignments):
            if ch == 0:
                residual[:, det] = x_type[:, rnd, row, col]
            elif ch == 1:
                residual[:, det] = z_type[:, rnd, row, col]
        return residual

    def decode(self, detector_events: np.ndarray) -> np.ndarray:
        residual = self._pre_decode(detector_events)
        return self._baseline.decode(residual)


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helper
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class LogicalErrorRate:
    """Summary of a decoder evaluation on a surface-code memory experiment."""

    distance: int
    rounds: int
    basis: str
    p_error: float
    shots: int
    logical_errors: int
    decoder_name: str

    @property
    def rate(self) -> float:
        return self.logical_errors / self.shots if self.shots else 0.0

    @property
    def standard_error(self) -> float:
        """1-sigma standard error on the logical-error-rate estimator."""
        if self.shots == 0:
            return 0.0
        p = self.rate
        return float(np.sqrt(p * (1.0 - p) / self.shots))

    def as_dict(self) -> dict:
        return {
            "distance": self.distance,
            "rounds": self.rounds,
            "basis": self.basis,
            "p_error": self.p_error,
            "shots": self.shots,
            "logical_errors": self.logical_errors,
            "rate": self.rate,
            "standard_error": self.standard_error,
            "decoder": self.decoder_name,
        }


def evaluate_logical_error_rate(
    spec: SurfaceCodePatchSpec,
    decoder: SurfaceCodeDecoder,
    shots: int,
    *,
    seed: int | None = None,
    batch: int = 16_384,
    decoder_name: str | None = None,
) -> LogicalErrorRate:
    """Estimate the decoder's logical error rate on *shots* samples.

    The sampler is batched to keep memory usage bounded for large
    distances.  ``decoder_name`` is a free-form string used in the
    returned :class:`LogicalErrorRate` record.
    """
    circuit = _build_stim_circuit(spec)
    sampler = circuit.compile_detector_sampler(seed=seed)
    remaining = shots
    errors = 0
    while remaining > 0:
        n = min(batch, remaining)
        det_events, obs_flips = sampler.sample(n, separate_observables=True)
        preds = decoder.decode(det_events)
        # Logical failure: any observable mismatch
        mismatched = np.any(preds != obs_flips, axis=1)
        errors += int(mismatched.sum())
        remaining -= n
    return LogicalErrorRate(
        distance=spec.distance,
        rounds=spec.rounds,
        basis=spec.basis,
        p_error=spec.p_error,
        shots=shots,
        logical_errors=errors,
        decoder_name=decoder_name or type(decoder).__name__,
    )
