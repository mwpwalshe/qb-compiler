"""Tests for the v0.5.0 IsingDecodeResult telemetry surface.

Covers the design decisions documented in
``qb_compiler.ising.decoder`` (Q1-Q4): result dataclass fields,
decoder_version provenance in both modes, the bounded reservoir in the
LER harness, ``to_json_dict`` array exclusion by default, and
backward compatibility of the legacy ``decode()`` path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import stim

from qb_compiler.ising import (
    EvaluationTelemetry,
    IsingDecoderConfig,
    IsingDecodeResult,
    IsingDecoderWrapper,
    PyMatchingDecoder,
    SurfaceCodePatchSpec,
    evaluate_logical_error_rate,
)
from qb_compiler.ising.decoder import _weights_version

if TYPE_CHECKING:
    from pathlib import Path


def _sample_events(spec: SurfaceCodePatchSpec, shots: int, seed: int = 11) -> np.ndarray:
    circuit = stim.Circuit.generated(
        spec.stim_task_name,
        distance=spec.distance,
        rounds=spec.rounds,
        after_clifford_depolarization=spec.p_error,
        after_reset_flip_probability=spec.p_error,
        before_measure_flip_probability=spec.p_error,
        before_round_data_depolarization=spec.p_error,
    )
    events, _ = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return np.asarray(events)


@pytest.fixture(scope="module")
def spec() -> SurfaceCodePatchSpec:
    return SurfaceCodePatchSpec(distance=3, rounds=3, basis="X", p_error=0.003)


@pytest.fixture(scope="module")
def decoder(spec: SurfaceCodePatchSpec) -> PyMatchingDecoder:
    return PyMatchingDecoder(spec)


class TestIsingDecodeResultFields:
    def test_always_populated_fields(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        events = _sample_events(spec, 16)
        result = decoder.decode_full(events)
        assert isinstance(result, IsingDecodeResult)
        assert result.prediction.shape == (16, 1)
        assert result.prediction.dtype == bool
        assert result.mwpm_weight.shape == (16,)
        assert result.mwpm_weight.dtype == np.float64
        assert result.spec is spec
        assert len(result.layout_fingerprint) == 16
        assert result.decoder_name == "pymatching"
        assert result.decode_seconds >= 0.0
        assert result.num_shots == 16
        assert result.num_observables == 1

    def test_opt_in_fields_default_none(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        result = decoder.decode_full(_sample_events(spec, 4))
        assert result.residual_syndrome is None
        assert result.pre_decoder_logits is None

    def test_collect_residual_copies_input(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        events = _sample_events(spec, 4)
        result = decoder.decode_full(events, collect_residual=True)
        assert result.residual_syndrome is not None
        assert result.residual_syndrome.dtype == bool
        np.testing.assert_array_equal(result.residual_syndrome, events.astype(bool))

    def test_logits_stay_none_on_pymatching_path(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        # Design decision Q1: no soft outputs on this path, never recomputed.
        result = decoder.decode_full(_sample_events(spec, 4), collect_logits=True)
        assert result.pre_decoder_logits is None

    def test_result_is_frozen(self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder) -> None:
        result = decoder.decode_full(_sample_events(spec, 2))
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.decoder_name = "tampered"  # type: ignore[misc]

    def test_prediction_matches_legacy_decode(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        events = _sample_events(spec, 32)
        np.testing.assert_array_equal(
            decoder.decode_full(events).prediction, decoder.decode(events)
        )

    def test_pymatching_version_is_package_version(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        import qb_compiler

        result = decoder.decode_full(_sample_events(spec, 2))
        assert result.decoder_version == str(qb_compiler.__version__)


class TestEnableCorrelations:
    def test_decoder_name_reflects_mode(self, spec: SurfaceCodePatchSpec) -> None:
        corr = PyMatchingDecoder(spec, enable_correlations=True)
        events = _sample_events(spec, 8)
        result = corr.decode_full(events)
        assert result.decoder_name == "pymatching+correlations"
        assert result.prediction.shape == (8, 1)
        assert corr.decode(events).shape == (8, 1)


class TestDecoderVersionProvenance:
    """Design decision Q2: checkpoint hash vs user-supplied."""

    def test_checkpoint_path_hashes_file(self, tmp_path: Path) -> None:
        payload = b"not real weights, just bytes to hash"
        target = tmp_path / "Ising-Decoder-SurfaceCode-1-Fast.pt"
        target.write_bytes(payload)
        expected = "sha256:" + hashlib.sha256(payload).hexdigest()[:16]
        assert _weights_version(str(target)) == expected

    def test_no_checkpoint_is_user_supplied(self) -> None:
        assert _weights_version(None) == "user-supplied"
        assert _weights_version("") == "user-supplied"


class TestIsingWrapperTelemetry:
    """Wrapper-level coverage of Q1 and Q2; needs torch."""

    @pytest.fixture()
    def torch(self) -> Any:
        return pytest.importorskip("torch")

    def _build_model(self, torch: Any) -> Any:
        class _Identity(torch.nn.Module):  # type: ignore[misc, name-defined]
            def forward(self, x: Any) -> Any:
                return x

        return lambda spec: _Identity()

    def test_user_supplied_version(self, torch: Any, spec: SurfaceCodePatchSpec) -> None:
        wrapper = IsingDecoderWrapper(
            spec, IsingDecoderConfig(build_model=self._build_model(torch))
        )
        events = _sample_events(spec, 4)
        result = wrapper.decode_full(events)
        assert result.decoder_version == "user-supplied"
        assert result.decoder_name == "ising"

    def test_checkpoint_version_hashes_file(
        self, torch: Any, spec: SurfaceCodePatchSpec, tmp_path: Path
    ) -> None:
        ckpt = tmp_path / "Ising-Decoder-SurfaceCode-1-Accurate.pt"
        torch.save({}, str(ckpt))
        expected = "sha256:" + hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]
        wrapper = IsingDecoderWrapper(
            spec,
            IsingDecoderConfig(weights_path=str(ckpt), build_model=self._build_model(torch)),
        )
        result = wrapper.decode_full(_sample_events(spec, 2))
        assert result.decoder_version == expected
        assert result.decoder_name == "ising_accurate"

    def test_logits_escape_forward_pass(self, torch: Any, spec: SurfaceCodePatchSpec) -> None:
        wrapper = IsingDecoderWrapper(
            spec, IsingDecoderConfig(build_model=self._build_model(torch))
        )
        events = _sample_events(spec, 4)
        result = wrapper.decode_full(events, collect_logits=True, collect_residual=True)
        assert result.pre_decoder_logits is not None
        assert result.pre_decoder_logits.shape == (4, 4, spec.rounds, spec.distance, spec.distance)
        assert result.pre_decoder_logits.dtype == np.float32
        assert result.residual_syndrome is not None
        assert result.residual_syndrome.shape == events.shape

    def test_decode_collect_telemetry_flag(self, torch: Any, spec: SurfaceCodePatchSpec) -> None:
        wrapper = IsingDecoderWrapper(
            spec, IsingDecoderConfig(build_model=self._build_model(torch))
        )
        events = _sample_events(spec, 4)
        # Default path: legacy bool array, unchanged.
        legacy = wrapper.decode(events)
        assert isinstance(legacy, np.ndarray)
        assert legacy.dtype == bool
        # Opt-in path: full telemetry from the same single forward pass.
        full = wrapper.decode(events, collect_telemetry=True)
        assert isinstance(full, IsingDecodeResult)
        assert full.pre_decoder_logits is not None
        np.testing.assert_array_equal(full.prediction, legacy)


class TestToJsonDict:
    """Design decision Q4: aggregates and metadata only by default."""

    def test_default_excludes_arrays(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        result = decoder.decode_full(_sample_events(spec, 8), collect_residual=True)
        payload = result.to_json_dict()
        assert "arrays" not in payload
        assert payload["num_shots"] == 8
        assert payload["decoder_name"] == "pymatching"
        assert payload["shapes"]["prediction"] == [8, 1]
        assert payload["shapes"]["residual_syndrome"] is not None
        assert payload["shapes"]["pre_decoder_logits"] is None
        assert payload["decode_seconds"] >= 0.0
        w_min, w_mean = payload["mwpm_weight_min"], payload["mwpm_weight_mean"]
        assert w_min <= w_mean <= payload["mwpm_weight_max"]
        # Must be plain-JSON serialisable.
        json.dumps(payload)

    def test_include_arrays_opt_in(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        result = decoder.decode_full(_sample_events(spec, 5), collect_residual=True)
        payload = result.to_json_dict(include_arrays=True)
        assert len(payload["arrays"]["prediction"]) == 5
        assert len(payload["arrays"]["mwpm_weight"]) == 5
        assert len(payload["arrays"]["residual_syndrome"]) == 5
        assert payload["arrays"]["pre_decoder_logits"] is None
        json.dumps(payload)

    def test_no_policy_fields(self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder) -> None:
        # Boundary rule: signals only, nothing threshold-shaped.
        payload = decoder.decode_full(_sample_events(spec, 2)).to_json_dict()
        forbidden = ("threshold", "gate", "accept", "abort", "confidence")
        for key in payload:
            assert not any(word in key.lower() for word in forbidden)


class TestHarnessTelemetry:
    """Design decision Q3: opt-in, bounded reservoir."""

    def test_reservoir_cap_honored(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        r = evaluate_logical_error_rate(
            spec,
            decoder,
            shots=300,
            seed=3,
            batch=64,
            collect_telemetry=True,
            telemetry_max_shots=50,
        )
        assert isinstance(r.telemetry, EvaluationTelemetry)
        assert r.telemetry.shots_seen == 300
        assert r.telemetry.sample_size == 50
        assert r.telemetry.sample_prediction.shape == (50, 1)
        assert r.telemetry.sample_mwpm_weight.shape == (50,)
        assert r.telemetry.sample_mismatch.shape == (50,)
        assert r.telemetry.mismatched_shots == r.logical_errors

    def test_reservoir_smaller_than_cap_keeps_all(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        r = evaluate_logical_error_rate(
            spec, decoder, shots=30, seed=3, collect_telemetry=True, telemetry_max_shots=1024
        )
        assert r.telemetry is not None
        assert r.telemetry.sample_size == 30

    def test_aggregates_consistent(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        r = evaluate_logical_error_rate(spec, decoder, shots=200, seed=5, collect_telemetry=True)
        t = r.telemetry
        assert t is not None
        assert 0 <= t.predicted_flip_shots <= t.shots_seen
        assert t.mwpm_weight_min <= t.mwpm_weight_mean <= t.mwpm_weight_max
        assert t.decoder_name == "pymatching"
        assert len(t.layout_fingerprint) == 16

    def test_negative_cap_rejected(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        with pytest.raises(ValueError, match="telemetry_max_shots"):
            evaluate_logical_error_rate(
                spec, decoder, shots=10, collect_telemetry=True, telemetry_max_shots=-1
            )

    def test_decoder_without_decode_full_rejected(self, spec: SurfaceCodePatchSpec) -> None:
        class _LegacyDecoder:
            def __init__(self, inner: PyMatchingDecoder) -> None:
                self.spec = inner.spec
                self._inner = inner

            def decode(self, detector_events: np.ndarray) -> np.ndarray:
                return self._inner.decode(detector_events)

        with pytest.raises(TypeError, match="decode_full"):
            evaluate_logical_error_rate(
                spec, _LegacyDecoder(PyMatchingDecoder(spec)), shots=10, collect_telemetry=True
            )


class TestBackwardCompat:
    def test_decode_default_path_unchanged(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        events = _sample_events(spec, 16)
        preds = decoder.decode(events)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (16, 1)
        assert preds.dtype == bool

    def test_harness_default_has_no_telemetry(
        self, spec: SurfaceCodePatchSpec, decoder: PyMatchingDecoder
    ) -> None:
        r = evaluate_logical_error_rate(spec, decoder, shots=100, seed=1)
        assert r.telemetry is None
        # as_dict keys unchanged from v0.4.x
        assert set(r.as_dict()) == {
            "distance",
            "rounds",
            "basis",
            "p_error",
            "shots",
            "logical_errors",
            "rate",
            "standard_error",
            "decoder",
        }
