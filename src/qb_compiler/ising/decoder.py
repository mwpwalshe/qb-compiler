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

v0.5.0 adds the :class:`IsingDecodeResult` telemetry surface designed
in ``docs/design/v050_telemetry_surface.md``: a richer, frozen result
record reachable via ``decode_full()`` on both decoders.  Everything
on it is a measurement; policy (thresholds, gating, accept/reject)
lives in downstream consumers, never here.

Design decisions (v0.6 implementation)
--------------------------------------
The design doc left four open questions.  They are resolved here as
follows, conservatively:

* **Q1, logits without doubling cost.**  Per-shot soft outputs
  (``pre_decoder_logits``) are captured ONLY when the underlying
  decode path already produces them in its forward pass.  The NVIDIA
  pre-decoder's raw output tensor now escapes ``_pre_decode`` instead
  of being discarded, so collecting it costs one numpy reference, not
  a second forward pass.  ``IsingDecoderWrapper.decode`` exposes
  ``collect_telemetry: bool = False``; the default path is unchanged
  and returns the legacy bool array.  When the decode path has no
  soft outputs (the bare PyMatching path, or a model that does not
  expose them), the field is ``None``; nothing is ever recomputed to
  fill it.
* **Q2, decoder_version provenance.**  When a checkpoint path was
  given, ``decoder_version = "sha256:<first 16 hex of the file>"``.
  When the user supplied ``build_model`` (and any state it carries)
  directly with no checkpoint path, ``decoder_version =
  "user-supplied"``.  No source hashing, no guessing.  The bare
  PyMatching path uses the qb-compiler package version per the
  design doc.
* **Q3, memory in the LER harness.**  Telemetry in
  :func:`evaluate_logical_error_rate` is opt-in
  (``collect_telemetry: bool = False``) and bounded: the harness
  stores aggregate counts plus a uniform reservoir sample of at most
  ``telemetry_max_shots`` (default 1024) per-shot records.  Full
  per-batch :class:`IsingDecodeResult` objects are never accumulated,
  so the harness cannot OOM regardless of shot count.
* **Q4, serialization vs boundary.**  :meth:`IsingDecodeResult.to_json_dict`
  emits aggregates and metadata ONLY by default (counts, rates,
  shapes, version, timing); per-shot arrays are included only with
  ``include_arrays=True``.  This is a data-volume control, not
  policy: there is no threshold, no score cutoff, no decision
  anywhere in the helper.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, overload

import numpy as np
import stim

if TYPE_CHECKING:
    from collections.abc import Callable

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
# Telemetry surface (pure measurement, no policy)
# ═══════════════════════════════════════════════════════════════════════


def _package_version() -> str:
    """qb-compiler package version, used as the bare-PyMatching provenance."""
    from qb_compiler import __version__

    return str(__version__)


def _weights_version(weights_path: str | None) -> str:
    """Provenance string for the Ising path (design decision Q2).

    ``"sha256:<first 16 hex>"`` of the checkpoint file when a path was
    given, ``"user-supplied"`` when the user supplied ``build_model``
    (and any state it carries) directly.
    """
    if not weights_path:
        return "user-supplied"
    digest = hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()
    return f"sha256:{digest[:16]}"


def _ising_decoder_name(weights_path: str | None) -> str:
    """Dispatch-key name for the Ising chain, derived from the checkpoint name."""
    if weights_path:
        filename = Path(weights_path).name.lower()
        if "fast" in filename:
            return "ising_fast"
        if "accurate" in filename:
            return "ising_accurate"
    return "ising"


@dataclass(frozen=True)
class IsingDecodeResult:
    """Telemetry surface from a decoder pass.  Pure measurement, no policy.

    Always-populated fields are cheap (at most 8 bytes per shot plus a
    handful of constant-size strings).  The opt-in fields can blow up
    memory at scale: ``pre_decoder_logits`` alone is ~52.7 GB at
    d=15, T=15, 1M shots, which is why both collection flags default
    to ``False`` on :meth:`PyMatchingDecoder.decode_full` and
    :meth:`IsingDecoderWrapper.decode_full`.

    Attributes
    ----------
    prediction:
        Bool array of shape ``(batch, num_observables)``, identical to
        the legacy ``decode()`` output.
    mwpm_weight:
        Float64 array of shape ``(batch,)``: the sum of MWPM edge
        weights per shot (PyMatching's ``return_weights=True`` output,
        computed during matching anyway).  For the Ising chain this is
        the residual MWPM weight after the pre-decoder pass.  Lower
        means a lighter matching.  Semantics depend on
        ``enable_correlations``: with correlated matching the weight
        reflects the modified edge weights of the two-pass algorithm,
        not the plain MWPM problem.
    spec:
        The :class:`SurfaceCodePatchSpec` the decoder was built from.
    layout_fingerprint:
        16-hex digest from :class:`SurfaceCodeTensorLayout`, for
        orientation-drift detection.
    decoder_name:
        Programmatic dispatch key: ``"pymatching"``,
        ``"pymatching+correlations"``, ``"ising_fast"``,
        ``"ising_accurate"`` or ``"ising"``.
    decoder_version:
        Build provenance.  qb-compiler package version on the bare
        PyMatching path; ``"sha256:<first16>"`` of the weights file or
        ``"user-supplied"`` on the Ising path (design decision Q2).
    decode_seconds:
        Wall-clock duration of this decode pass (``time.perf_counter``
        delta), a measurement like everything else here.
    residual_syndrome:
        Opt-in (``collect_residual=True``).  The detector events
        surviving the pre-decoder pass (Ising chain) or a copy of the
        raw input events (bare PyMatching path).  Bool array of shape
        ``(batch, num_detectors)``; ``None`` when not collected.
    pre_decoder_logits:
        Opt-in (``collect_logits=True``).  The NVIDIA pre-decoder's
        raw forward-pass output BEFORE any thresholding, float32 of
        shape ``(batch, 4, rounds, distance, distance)``.  Always
        ``None`` on the bare PyMatching path, which has no soft
        outputs (design decision Q1); never recomputed to fill it.
    """

    prediction: np.ndarray
    mwpm_weight: np.ndarray
    spec: SurfaceCodePatchSpec
    layout_fingerprint: str
    decoder_name: str
    decoder_version: str
    decode_seconds: float
    residual_syndrome: np.ndarray | None = None
    pre_decoder_logits: np.ndarray | None = None

    @property
    def num_shots(self) -> int:
        return int(self.prediction.shape[0])

    @property
    def num_observables(self) -> int:
        return int(self.prediction.shape[1])

    def to_json_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """JSON-serialisable view of this result (design decision Q4).

        By default emits aggregates and metadata only: counts, rates,
        shapes, version and timing.  The per-shot arrays are included
        under an ``"arrays"`` key only with ``include_arrays=True``.
        This is a data-volume control, not policy; no field here is a
        threshold or a decision.
        """
        shots = self.num_shots
        flipped_shots = int(np.any(self.prediction, axis=1).sum())
        weights = np.asarray(self.mwpm_weight, dtype=np.float64)
        out: dict[str, Any] = {
            "decoder_name": self.decoder_name,
            "decoder_version": self.decoder_version,
            "layout_fingerprint": self.layout_fingerprint,
            "spec": {
                "distance": self.spec.distance,
                "rounds": self.spec.rounds,
                "basis": self.spec.basis,
                "p_error": self.spec.p_error,
            },
            "num_shots": shots,
            "num_observables": self.num_observables,
            "predicted_flip_shots": flipped_shots,
            "predicted_flip_rate": flipped_shots / shots if shots else 0.0,
            "mwpm_weight_min": float(weights.min()) if shots else 0.0,
            "mwpm_weight_max": float(weights.max()) if shots else 0.0,
            "mwpm_weight_mean": float(weights.mean()) if shots else 0.0,
            "decode_seconds": self.decode_seconds,
            "shapes": {
                "prediction": list(self.prediction.shape),
                "mwpm_weight": list(self.mwpm_weight.shape),
                "residual_syndrome": (
                    None if self.residual_syndrome is None else list(self.residual_syndrome.shape)
                ),
                "pre_decoder_logits": (
                    None
                    if self.pre_decoder_logits is None
                    else list(self.pre_decoder_logits.shape)
                ),
            },
        }
        if include_arrays:
            out["arrays"] = {
                "prediction": self.prediction.astype(bool).tolist(),
                "mwpm_weight": weights.tolist(),
                "residual_syndrome": (
                    None
                    if self.residual_syndrome is None
                    else self.residual_syndrome.astype(bool).tolist()
                ),
                "pre_decoder_logits": (
                    None
                    if self.pre_decoder_logits is None
                    else self.pre_decoder_logits.astype(float).tolist()
                ),
            }
        return out


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
    """MWPM decoder using ``pymatching`` and a stim DEM.

    Parameters
    ----------
    spec:
        The surface-code memory experiment to build the matching graph
        for.
    enable_correlations:
        Opt-in pass-through to PyMatching's correlated-matching mode
        (Fowler 2013, arXiv:1310.0863), requires ``pymatching >= 2.3``.
        Default ``False``: flipping it changes the semantics of the
        ``mwpm_weight`` telemetry field, because the weight then
        reflects the modified edge weights of the two-pass correlated
        algorithm rather than the plain MWPM problem.  Downstream
        telemetry consumers must opt in knowingly.
    """

    def __init__(
        self,
        spec: SurfaceCodePatchSpec,
        *,
        enable_correlations: bool = False,
    ) -> None:
        import pymatching

        self.spec = spec
        self.enable_correlations = enable_correlations
        circuit = _build_stim_circuit(spec)
        dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        if enable_correlations:
            self._matching = pymatching.Matching.from_detector_error_model(
                dem, enable_correlations=True
            )
        else:
            self._matching = pymatching.Matching.from_detector_error_model(dem)

    def decode(self, detector_events: np.ndarray) -> np.ndarray:
        if detector_events.ndim != 2:
            raise ValueError(f"detector_events must be 2-D, got shape {detector_events.shape}")
        events = detector_events.astype(np.uint8)
        if self.enable_correlations:
            preds = self._matching.decode_batch(events, enable_correlations=True)
        else:
            preds = self._matching.decode_batch(events)
        # pymatching returns (batch, num_observables) uint8 0/1
        return cast("np.ndarray", preds.astype(bool))

    def decode_full(
        self,
        detector_events: np.ndarray,
        *,
        collect_residual: bool = False,
        collect_logits: bool = False,
    ) -> IsingDecodeResult:
        """Decode and return the full telemetry record.

        Memory warning: the opt-in fields scale per shot
        (``residual_syndrome`` as O(d^2 T) bits).  ``collect_logits``
        is accepted for signature symmetry with
        :meth:`IsingDecoderWrapper.decode_full` but this path has no
        soft outputs, so ``pre_decoder_logits`` is always ``None``
        (design decision Q1); nothing is recomputed to fill it.
        """
        del collect_logits  # no soft outputs on this path, field stays None
        from qb_compiler.ising.stim_adapter import resolve_layout

        if detector_events.ndim != 2:
            raise ValueError(f"detector_events must be 2-D, got shape {detector_events.shape}")
        events = detector_events.astype(np.uint8)
        start = time.perf_counter()
        if self.enable_correlations:
            preds, weights = self._matching.decode_batch(
                events, return_weights=True, enable_correlations=True
            )
        else:
            preds, weights = self._matching.decode_batch(events, return_weights=True)
        elapsed = time.perf_counter() - start
        return IsingDecodeResult(
            prediction=preds.astype(bool),
            mwpm_weight=np.asarray(weights, dtype=np.float64).reshape(-1),
            spec=self.spec,
            layout_fingerprint=resolve_layout(self.spec).orientation_fingerprint,
            decoder_name="pymatching+correlations" if self.enable_correlations else "pymatching",
            decoder_version=_package_version(),
            decode_seconds=elapsed,
            residual_syndrome=events.astype(bool) if collect_residual else None,
            pre_decoder_logits=None,
        )


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
        May be ``None`` when ``build_model`` returns a module that
        already carries its state; in that case no checkpoint is
        loaded and telemetry provenance reports
        ``decoder_version="user-supplied"`` (design decision Q2).
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

    weights_path: str | None = None
    device: str = "cpu"
    build_model: Callable[..., Any] | None = None


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
        # Provenance is fixed at construction time (design decision Q2):
        # checkpoint path given -> file hash; build_model state only ->
        # "user-supplied".
        self._decoder_version = _weights_version(config.weights_path)
        self._decoder_name = _ising_decoder_name(config.weights_path)
        if config.weights_path:
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

    def _pre_decode(self, detector_events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the pre-decoder; return (residual detector events, raw logits).

        The raw forward-pass output escapes here instead of being
        discarded (design decision Q1), so ``decode_full`` can collect
        it without a second forward pass.
        """
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
        return residual, np.asarray(out, dtype=np.float32)

    @overload
    def decode(
        self,
        detector_events: np.ndarray,
        *,
        collect_telemetry: Literal[False] = ...,
    ) -> np.ndarray: ...

    @overload
    def decode(
        self,
        detector_events: np.ndarray,
        *,
        collect_telemetry: Literal[True],
    ) -> IsingDecodeResult: ...

    def decode(
        self,
        detector_events: np.ndarray,
        *,
        collect_telemetry: bool = False,
    ) -> np.ndarray | IsingDecodeResult:
        """Decode a batch of detector events.

        The default path is unchanged from v0.4.x and returns the bool
        prediction array.  With ``collect_telemetry=True`` (design
        decision Q1) the same single pass returns an
        :class:`IsingDecodeResult` carrying the soft outputs the
        forward pass already produced; nothing is recomputed.
        """
        if collect_telemetry:
            return self.decode_full(detector_events, collect_residual=True, collect_logits=True)
        residual, _ = self._pre_decode(detector_events)
        return self._baseline.decode(residual)

    def decode_full(
        self,
        detector_events: np.ndarray,
        *,
        collect_residual: bool = False,
        collect_logits: bool = False,
    ) -> IsingDecodeResult:
        """Decode and return the full telemetry record.

        Memory warning: ``collect_logits=True`` keeps the pre-decoder's
        raw float32 output, ``(batch, 4, rounds, d, d)``, ~52.7 GB at
        d=15, T=15, 1M shots.  Both collection flags default off; the
        logits are taken from the forward pass that runs anyway, never
        from a second pass (design decision Q1).
        """
        start = time.perf_counter()
        residual, logits = self._pre_decode(detector_events)
        base = self._baseline.decode_full(residual)
        elapsed = time.perf_counter() - start
        return IsingDecodeResult(
            prediction=base.prediction,
            mwpm_weight=base.mwpm_weight,
            spec=self.spec,
            layout_fingerprint=base.layout_fingerprint,
            decoder_name=self._decoder_name,
            decoder_version=self._decoder_version,
            decode_seconds=elapsed,
            residual_syndrome=residual.astype(bool) if collect_residual else None,
            pre_decoder_logits=logits if collect_logits else None,
        )


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helper
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EvaluationTelemetry:
    """Bounded telemetry from an LER evaluation run.  Pure measurement.

    Built by :func:`evaluate_logical_error_rate` when
    ``collect_telemetry=True`` (design decision Q3): aggregate counts
    over every shot, plus a uniform reservoir sample of at most
    ``telemetry_max_shots`` per-shot records.  Memory is bounded by the
    reservoir capacity regardless of how many shots were evaluated.

    Attributes
    ----------
    decoder_name / decoder_version / layout_fingerprint:
        Provenance copied from the per-batch :class:`IsingDecodeResult`.
    shots_seen:
        Total shots that flowed through the harness.
    predicted_flip_shots:
        Shots where the decoder predicted at least one observable flip.
    mismatched_shots:
        Shots where the prediction disagreed with the sampled
        observables (same count as ``LogicalErrorRate.logical_errors``).
    mwpm_weight_sum / mwpm_weight_min / mwpm_weight_max:
        Aggregates over every shot's matching weight.
    sample_prediction:
        Bool array ``(k, num_observables)``, ``k <= telemetry_max_shots``.
    sample_mwpm_weight:
        Float64 array ``(k,)``; per-shot matching weights for the
        sampled shots.
    sample_mismatch:
        Bool array ``(k,)``; whether each sampled shot was a logical
        mismatch.
    """

    decoder_name: str
    decoder_version: str
    layout_fingerprint: str
    shots_seen: int
    predicted_flip_shots: int
    mismatched_shots: int
    mwpm_weight_sum: float
    mwpm_weight_min: float
    mwpm_weight_max: float
    sample_prediction: np.ndarray
    sample_mwpm_weight: np.ndarray
    sample_mismatch: np.ndarray

    @property
    def sample_size(self) -> int:
        return int(self.sample_mwpm_weight.shape[0])

    @property
    def mwpm_weight_mean(self) -> float:
        return self.mwpm_weight_sum / self.shots_seen if self.shots_seen else 0.0


def _reservoir_update(
    reservoir: list[tuple[np.ndarray, float, bool]],
    seen_before: int,
    capacity: int,
    rng: np.random.Generator,
    predictions: np.ndarray,
    weights: np.ndarray,
    mismatched: np.ndarray,
) -> None:
    """Algorithm-R reservoir update over one batch of per-shot records."""
    for i in range(predictions.shape[0]):
        record = (predictions[i].copy(), float(weights[i]), bool(mismatched[i]))
        if len(reservoir) < capacity:
            reservoir.append(record)
        else:
            slot = int(rng.integers(0, seen_before + i + 1))
            if slot < capacity:
                reservoir[slot] = record


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
    telemetry: EvaluationTelemetry | None = None

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
    collect_telemetry: bool = False,
    telemetry_max_shots: int = 1024,
) -> LogicalErrorRate:
    """Estimate the decoder's logical error rate on *shots* samples.

    The sampler is batched to keep memory usage bounded for large
    distances.  ``decoder_name`` is a free-form string used in the
    returned :class:`LogicalErrorRate` record.

    With ``collect_telemetry=True`` (design decision Q3) the decoder
    must expose ``decode_full`` and the returned record carries a
    bounded :class:`EvaluationTelemetry`: aggregate counts over every
    shot plus a uniform reservoir sample of at most
    ``telemetry_max_shots`` per-shot records.  Per-batch results are
    never accumulated, so memory stays bounded at any shot count.
    Default ``False``: existing call sites see no change.
    """
    if telemetry_max_shots < 0:
        raise ValueError(f"telemetry_max_shots must be >= 0, got {telemetry_max_shots}")
    decode_full = getattr(decoder, "decode_full", None)
    if collect_telemetry and not callable(decode_full):
        raise TypeError(
            f"collect_telemetry=True requires the decoder to implement decode_full(); "
            f"{type(decoder).__name__} does not."
        )

    circuit = _build_stim_circuit(spec)
    sampler = circuit.compile_detector_sampler(seed=seed)
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[np.ndarray, float, bool]] = []
    # (decoder_name, decoder_version, layout_fingerprint, num_observables)
    provenance: tuple[str, str, str, int] | None = None
    seen = 0
    predicted_flip_shots = 0
    weight_sum = 0.0
    weight_min = np.inf
    weight_max = -np.inf

    remaining = shots
    errors = 0
    while remaining > 0:
        n = min(batch, remaining)
        det_events, obs_flips = sampler.sample(n, separate_observables=True)
        if collect_telemetry and decode_full is not None:
            result: IsingDecodeResult = decode_full(det_events)
            preds = result.prediction
            mismatched = np.any(preds != obs_flips, axis=1)
            weights = result.mwpm_weight
            if provenance is None:
                provenance = (
                    result.decoder_name,
                    result.decoder_version,
                    result.layout_fingerprint,
                    result.num_observables,
                )
            predicted_flip_shots += int(np.any(preds, axis=1).sum())
            weight_sum += float(weights.sum())
            if n:
                weight_min = min(weight_min, float(weights.min()))
                weight_max = max(weight_max, float(weights.max()))
            _reservoir_update(
                reservoir, seen, telemetry_max_shots, rng, preds, weights, mismatched
            )
            seen += n
        else:
            preds = decoder.decode(det_events)
            # Logical failure: any observable mismatch
            mismatched = np.any(preds != obs_flips, axis=1)
        errors += int(mismatched.sum())
        remaining -= n

    telemetry: EvaluationTelemetry | None = None
    if collect_telemetry and provenance is not None:
        prov_name, prov_version, prov_fingerprint, num_obs = provenance
        telemetry = EvaluationTelemetry(
            decoder_name=prov_name,
            decoder_version=prov_version,
            layout_fingerprint=prov_fingerprint,
            shots_seen=seen,
            predicted_flip_shots=predicted_flip_shots,
            mismatched_shots=errors,
            mwpm_weight_sum=weight_sum,
            mwpm_weight_min=float(weight_min) if seen else 0.0,
            mwpm_weight_max=float(weight_max) if seen else 0.0,
            sample_prediction=(
                np.array([r[0] for r in reservoir], dtype=bool)
                if reservoir
                else np.empty((0, num_obs), dtype=bool)
            ),
            sample_mwpm_weight=np.array([r[1] for r in reservoir], dtype=np.float64),
            sample_mismatch=np.array([r[2] for r in reservoir], dtype=bool),
        )

    return LogicalErrorRate(
        distance=spec.distance,
        rounds=spec.rounds,
        basis=spec.basis,
        p_error=spec.p_error,
        shots=shots,
        logical_errors=errors,
        decoder_name=decoder_name or type(decoder).__name__,
        telemetry=telemetry,
    )
