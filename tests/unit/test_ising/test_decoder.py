"""Tests for the decoder wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from qb_compiler.ising import (
    IsingDecoderConfig,
    IsingDecoderWrapper,
    PyMatchingDecoder,
    SurfaceCodePatchSpec,
    evaluate_logical_error_rate,
)


class TestPyMatchingBaseline:
    def test_decoder_instantiates(self) -> None:
        spec = SurfaceCodePatchSpec(distance=3, rounds=3, basis="X")
        decoder = PyMatchingDecoder(spec)
        assert decoder.spec is spec

    def test_decode_shape(self) -> None:
        spec = SurfaceCodePatchSpec(distance=5, rounds=5, basis="Z")
        decoder = PyMatchingDecoder(spec)
        # Build a synthetic zero-syndrome batch — should decode to no flip
        import stim

        circuit = stim.Circuit.generated(
            spec.stim_task_name,
            distance=spec.distance,
            rounds=spec.rounds,
            after_clifford_depolarization=spec.p_error,
            after_reset_flip_probability=spec.p_error,
            before_measure_flip_probability=spec.p_error,
            before_round_data_depolarization=spec.p_error,
        )
        n_det = circuit.num_detectors
        zero_events = np.zeros((3, n_det), dtype=np.uint8)
        preds = decoder.decode(zero_events)
        assert preds.shape == (3, 1)
        assert preds.dtype == bool
        # No syndrome → no logical flip
        assert not preds.any()

    @pytest.mark.parametrize("d", [3, 5])
    def test_ler_decreases_with_distance_at_low_p(self, d: int) -> None:
        """Classic error-suppression check — deeper code → lower LER."""
        p = 0.002
        results = {}
        for dist in (d, d + 2):
            spec = SurfaceCodePatchSpec(distance=dist, rounds=dist, basis="X", p_error=p)
            decoder = PyMatchingDecoder(spec)
            r = evaluate_logical_error_rate(spec, decoder, shots=20_000, seed=7)
            results[dist] = r.rate
        # Deeper distance gives lower LER at p below threshold (~1%)
        assert results[d + 2] <= results[d] + 1e-3

    def test_evaluate_returns_record(self) -> None:
        spec = SurfaceCodePatchSpec(distance=3, rounds=3, basis="X")
        decoder = PyMatchingDecoder(spec)
        r = evaluate_logical_error_rate(
            spec, decoder, shots=1_000, seed=0, decoder_name="test"
        )
        assert r.shots == 1_000
        assert r.decoder_name == "test"
        assert 0.0 <= r.rate <= 1.0
        assert r.standard_error >= 0.0
        assert "rate" in r.as_dict()


class TestIsingDecoderWrapperRequirements:
    def test_missing_build_model_raises(self) -> None:
        spec = SurfaceCodePatchSpec(distance=3, rounds=3, basis="X")
        # torch may or may not be importable on this system; if torch is
        # missing the wrapper raises ImportError before reaching the
        # build_model check.  Either error type is acceptable — both
        # convey "you haven't supplied enough to run NVIDIA's model".
        try:
            IsingDecoderWrapper(
                spec,
                IsingDecoderConfig(weights_path="doesnt-exist.pt"),
            )
        except (ImportError, NotImplementedError) as exc:
            msg = str(exc)
            assert ("torch" in msg) or ("build_model" in msg)
        else:
            pytest.fail("expected ImportError or NotImplementedError")
