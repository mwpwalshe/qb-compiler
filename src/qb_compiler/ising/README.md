# qb-compiler → NVIDIA Ising integration

> **Status: beta (`v0.4.0b1`).** Stim only for now, no hw shots thru it yet.
> NVIDIA dropped the decoder on 2026-04-14 and this onramp went out a week
> later so expect the API to wobble a bit as people hit corners, pin the
> version if you're using it. Hw run on IBM Heron surface patches is up next.

This package is the first Qiskit-side onramp to NVIDIA's
`Ising-Decoder-SurfaceCode-1` model family (released 2026-04-14
under Apache 2.0 code + NVIDIA Open Model License weights).

## What's in the box

| Component | Purpose | Runtime deps |
|---|---|---|
| `SurfaceCodePatchSpec` | Minimal dataclass describing a rotated-surface-code memory experiment (`distance`, `rounds`, `basis`, `p_error`) |, |
| `build_ising_tensor()` | Stim detector events → `(B, 4, T, D, D)` float32 tensor with the channel layout `[x_type, z_type, x_present, z_present]` that the pretrained NVIDIA models consume | `stim`, `numpy` |
| `PyMatchingDecoder` | MWPM baseline built directly from `stim.Circuit.detector_error_model` | `stim`, `pymatching` |
| `IsingDecoderWrapper` | Optional pre-decoder + MWPM residual chain. Users supply their own gated-HF weights and a `build_model` callable sourced from NVIDIA's public repo, qb-compiler does not vendor NVIDIA code or weights | `torch`, `safetensors`, user-provided model def |
| `evaluate_logical_error_rate()` | Head-to-head decoder comparison harness | `stim`, `numpy` |
| `qiskit_bridge.stim_circuit_for()` / `qiskit_circuit_for()` | Reference circuit generators for users coming from either framework | `qiskit` (for the Qiskit one only) |

## Licensing

* **This integration module**: Apache 2.0 (same as the rest of qb-compiler).
* **NVIDIA Ising code**: Apache 2.0 (`github.com/NVIDIA/Ising-Decoding`).
* **NVIDIA Ising model weights**: NVIDIA Open Model License, gated download
  from HuggingFace (`nvidia/Ising-Decoder-SurfaceCode-1-Fast` and `-Accurate`).
  qb-compiler does not redistribute the weights.

## Quick start

```python
from qb_compiler.ising import (
    SurfaceCodePatchSpec,
    PyMatchingDecoder,
    evaluate_logical_error_rate,
)

spec = SurfaceCodePatchSpec(distance=7, rounds=7, basis="X", p_error=0.003)
decoder = PyMatchingDecoder(spec)
result = evaluate_logical_error_rate(spec, decoder, shots=50_000, seed=42)
print(result.as_dict())
# {'distance': 7, 'rounds': 7, 'basis': 'X', 'p_error': 0.003,
#  'shots': 50000, 'logical_errors': 126, 'rate': 0.00252, ...}
```

Run NVIDIA's pre-decoder against the same spec (requires
`pip install torch safetensors` + a gated-HF weights download + the
model definition from NVIDIA's repo):

```python
from qb_compiler.ising import IsingDecoderConfig, IsingDecoderWrapper

# build_model comes from NVIDIA's Apache-2.0 code at
# https://github.com/NVIDIA/Ising-Decoding (code/model/predecoder.py)
def build_model(spec):
    from model.predecoder import PreDecoderModelMemory_v1
    ...
    return PreDecoderModelMemory_v1(cfg)

config = IsingDecoderConfig(
    weights_path="Ising-Decoder-SurfaceCode-1-Fast.pt",
    device="cpu",
    build_model=build_model,
)
nvidia = IsingDecoderWrapper(spec, config)
nvidia_result = evaluate_logical_error_rate(spec, nvidia, shots=50_000, seed=42)
```

## Telemetry surface (v0.5.0)

Both decoders now expose `decode_full()`, which returns a frozen
`IsingDecodeResult` instead of the bare bool array. Design doc:
`docs/design/v050_telemetry_surface.md`. Everything on it is a
measurement (prediction, per-shot `mwpm_weight`, layout fingerprint,
decoder name/version provenance, decode timing). No thresholds, no
gating, no policy of any kind lives in this package; if you want to
make decisions on these signals, that happens in your own code.

```python
result = decoder.decode_full(detector_events)
result.mwpm_weight        # (B,) float64, free via pymatching return_weights
result.decoder_version    # package version, or "sha256:<first16>" of the
                          # weights file on the NVIDIA path ("user-supplied"
                          # when no checkpoint path was given)
result.to_json_dict()     # aggregates + metadata only; pass
                          # include_arrays=True to embed per-shot arrays
```

Opt-in per-shot fields (default off, they get big fast, see the memory
table in the design doc): `decode_full(collect_residual=True)` keeps the
residual detector events, `collect_logits=True` keeps the NVIDIA
pre-decoder's raw forward-pass output (always `None` on the bare
PyMatching path, nothing is recomputed to fill it). On the wrapper,
`decode(events, collect_telemetry=True)` returns the same record from a
single forward pass; the default `decode()` call is unchanged from
v0.4.x.

`PyMatchingDecoder(spec, enable_correlations=True)` opts into
PyMatching's correlated matching (needs `pymatching >= 2.3`). Note this
changes what `mwpm_weight` means (two-pass modified edge weights), which
is exactly why it stays off by default.

`evaluate_logical_error_rate(..., collect_telemetry=True)` attaches a
bounded `EvaluationTelemetry` to the returned record: aggregate counts
over every shot plus a uniform reservoir sample of at most
`telemetry_max_shots` (default 1024) per-shot records, so the harness
stays at fixed memory no matter the shot count.

## Hardware-data status (as of release)

NVIDIA Ising models target **rotated surface codes**.  Existing hardware
data inside the QubitBoost ecosystem comes from **repetition-code**
experiments (Trust Engine, 50K Fez shots).  Rep codes are NOT surface
codes and cannot be decoded by `Ising-Decoder-SurfaceCode-1` weights.

This release validates the integration on stim-simulated
circuit-level noise only; real-hardware surface-code validation will
follow once a dedicated Fez surface-code memory experiment has been
run.  The Stim benchmark harness in
`benchmarks/ising/run_pymatching_sweep.py` gives the baseline
PyMatching numbers that the NVIDIA pre-decoder must beat.

## Orientation caveat (pretrained weights)

qb-compiler's tensor is SHAPE-compatible with NVIDIA's models, but
the exact `(row, col)` convention for ancilla→grid mapping depends on
stim's rotated-surface-code layout, which may not match NVIDIA's
training-time `code_rotation='XV'` default.  Each
`SurfaceCodeTensorLayout` carries an `orientation_fingerprint` -
compare two layouts' fingerprints to detect orientation drift before
feeding qb-compiler tensors into pretrained NVIDIA weights.  An
alignment step (permute rows/cols of the output tensor) may be
required; this is an open enhancement tracked in the qb-compiler
issue tracker.
