# v0.5.0. IsingDecodeResult telemetry surface

**Status:** design, awaiting impl.
**Owner:** mwpwalshe
**Reference impl:** `proto/v050-decode-result` branch (do not merge as-is, real impl lands against this doc, not ahead of it).

---

## Problem

Today `IsingDecoderWrapper.decode()` and `PyMatchingDecoder.decode()` both
return a single `np.ndarray[bool]` of predictions. That's enough for a basic
LER measurement, but its the floor, anyone wanting to do anything beyond
"what fraction of shots flipped" has to either rebuild the decoder pipeline
themselves or rerun the whole thing twice. Concrete consumers who hit this
wall already:

- **Decoder confidence calibration researchers**, need pre-argmax logits to
  measure how well the pre-decoders predicted probabilities track empirical
  flip rates. Boolean output drops the entire signal.
- **People building their own gating layers**, need per-shot scores
  (matching weight, residual syndrome) to make their own accept/reject
  decisions. Currently they have to fork qb-compiler's decoder code.
- **NVIDIA-style ablation work**, want the pre-decoder output BEFORE the
  residual MWPM stage cleans it up. Current API gives only the post-residual
  prediction.
- **Downstream provenance** (Trust Passport / SafetyGate in the QubitBoost
  SDK, but also any third-party governance layer), needs decoder identity
  + version + a per-shot signal to bind to a passport field.

v0.5.0 adds a richer return type, `IsingDecodeResult`, that carries the
telemetry these consumers need. The current `decode()` signature stays
untouched for backward-compat, the new behaviour goes thru a separate
`decode_full()` entry point.

---

## What this is NOT (non-goals)

These are explicit and load-bearing, see the boundary-rule section at the
bottom for why each one is out.

- **No policy.** Nothing in `IsingDecodeResult` is a decision. It's all
  measurements.
- **No thresholding helpers.** No `should_accept_shot()`, no
  `confidence_threshold` parameter, no `min_score` knob, no callback like
  `on_low_confidence=...`. Anything that maps "signal → bool action" lives
  outside qb-compiler.
- **No gating logic.** Allow-rate gating, syndrome-consistency gating,
  Trust Passport field issuance, all that is downstream consumer territory
  (the QubitBoost SDK, in our case; any third-party gate, in the general
  case).
- **No streaming composition.** `IsingDecodeResult` fields are all `(B, ...)`
  shaped. If a consumer calls `decode_full` in a B=1 streaming loop and
  wants to concat results, thats their problem to solve. No `.concat()`
  classmethod, no `IsingDecodeResultBatch` accumulator. Streaming
  composition is not qb-compiler's job.
- **No calibration transforms.** Raw logits in, raw logits out. No
  temperature scaling, no Platt scaling, no `pre_decoder_calibrated_probs`
  field. If someone wants those they apply them to the raw tensor
  themselves.
- **No derived activation fields.** `pre_decoder_probs` was in an early
  sketch and has been removed, see "rejected during design" below. Apply
  `scipy.special.expit` yourself.

---

## The schema

```python
from dataclasses import dataclass

import numpy as np

from qb_compiler.ising import SurfaceCodePatchSpec


@dataclass(frozen=True)
class IsingDecodeResult:
    """Telemetry surface from a decoder pass. Pure measurement, no policy.

    Always-populated fields are cheap (≤ 8 B/shot). The opt-in fields can
    blow up memory at scale, see the per-shot table below before turning
    them on.
    """

    # -- always populated -----------------------------------------------
    # what the decoder predicted, identical to the legacy decode() output
    prediction: np.ndarray              # (B, n_observables) bool

    # sum of MWPM edge weights per shot. PyMatching always returns this when
    # asked (return_weights=True); for Ising chain its the residual MWPM
    # weight after the pre-decoder pass. Lower = more confident matching.
    # Semantics depend on enable_correlations, see below.
    mwpm_weight: np.ndarray             # (B,) float64

    # provenance
    spec: SurfaceCodePatchSpec
    layout_fingerprint: str             # 16-hex from SurfaceCodeTensorLayout
    decoder_name: str                   # "pymatching" | "pymatching+correlations" | "ising_fast" | "ising_accurate"
    decoder_version: str                # qb-compiler package version OR weights file SHA-256, depending on path

    # -- opt-in: collect_residual=True ----------------------------------
    # the detector events surviving the pre-decoder pass (for the Ising
    # chain) or the raw input events (for the bare PyMatching path).
    # Scales as O(d² · T) per shot.
    residual_syndrome: np.ndarray | None = None       # (B, n_detectors) bool

    # -- opt-in: collect_logits=True ------------------------------------
    # NVIDIA Ising pre-decoder raw output, BEFORE any sigmoid/argmax.
    # Same shape as the input ising tensor (B, 4, T, D, D).
    # Memory hungry, see table below. Always None for the PyMatching path.
    pre_decoder_logits: np.ndarray | None = None      # (B, 4, T, D, D) float32
```

### Why `frozen=True`

So nobody mutates a result object in place and ships it to a consumer that
expected immutability. Telemetry should be a snapshot. If you need to
transform it, build a new object.

### Why no `extras: dict` escape hatch

Considered and rejected. Untyped dicts on public dataclasses are where API
drift starts, someone shoves `extras["confidence_score"] = ...` in a PR,
it ships, and now its an undocumented public field that cant be removed
without breaking consumers. If a decoder needs to return something the
schema doesnt cover, its a schema change with a doc update, not a dict key.

### Why `decoder_version` is a separate field

`decoder_name` is a programmatic-dispatch key (which decoder ran). Six
months from now when somebody reports "this LER number doesn't match what
I got in May," the question is going to be *which build of the decoder
produced it*. Trust Passport provenance needs to distinguish the same
decoder across qb-compiler releases (and across NVIDIA weights versions
when the gated-HF download bumps). Two flavours:

- **Bare PyMatching path:** `decoder_version = qb_compiler.__version__`
  (e.g. `"0.5.0"`)
- **NVIDIA Ising path:** `decoder_version = sha256(weights_file)[:16]`
 , captures both qb-compiler's wrapper version AND which exact weights
  the user supplied. Weight-hash-only is fine since the wrapper code is
  pinned by qb-compiler version which is also visible to the consumer.

---

## API surface

Two new public methods on each decoder class. Existing `decode()` is
**untouched**.

```python
class PyMatchingDecoder:
    def __init__(
        self,
        spec: SurfaceCodePatchSpec,
        *,
        enable_correlations: bool = False,    # NEW, pass-through to PyMatching
    ) -> None: ...

    # unchanged from v0.4.x
    def decode(self, detector_events: np.ndarray) -> np.ndarray: ...

    # NEW
    def decode_full(
        self,
        detector_events: np.ndarray,
        *,
        collect_residual: bool = False,
        collect_logits: bool = False,         # always False on this path
    ) -> IsingDecodeResult: ...


class IsingDecoderWrapper:
    # unchanged constructor + decode()
    def decode(self, detector_events: np.ndarray) -> np.ndarray: ...

    # NEW
    def decode_full(
        self,
        detector_events: np.ndarray,
        *,
        collect_residual: bool = False,
        collect_logits: bool = False,
    ) -> IsingDecodeResult: ...
```

### `enable_correlations` default

**Default is `False`.** PyMatching's correlated-matching mode (Fowler 2013,
arXiv:1310.0863) is a quality lift, but flipping it changes the semantics
of `mwpm_weight`, the weight now reflects a different matching problem
(modified edge weights from the two-pass correlated algorithm). Changing
the default silently would change what every downstream telemetry consumer
sees from one release to the next.

Ship it as opt-in for v0.5.0. Document the weight-semantics delta in the
docstring. Reconsider the default in v0.6+ once we've seen how it affects
real LER sweeps and whether downstream consumers (esp the SDK) want both
weight families exposed side-by-side.

### `evaluate_logical_error_rate` change

Add a `collect_telemetry: bool = False` flag. When `True`, attach the
per-batch `IsingDecodeResult` to the returned `LogicalErrorRate` via a new
`telemetry: list[IsingDecodeResult] | None` field.

```python
result = evaluate_logical_error_rate(
    spec, decoder, shots=10_000, seed=42,
    collect_telemetry=True,        # NEW
)
# result.telemetry is now a list of IsingDecodeResult, one per batch
```

Default `False` for backward-compat, existing `LogicalErrorRate`
consumers see no change.

---

## Memory and performance footprint

Numbers from the prototype run (real PyMatching, architecture-faithful
NVIDIA stub). The opt-in fields are the ones to watch.

### Per-shot bytes

| Field | Per-shot bytes (d=5) | Per-shot bytes (d=15) | Default? |
|---|---|---|---|
| `prediction` | 1 | 1 | always |
| `mwpm_weight` | 8 | 8 | always |
| `layout_fingerprint` + `decoder_name` + `decoder_version` | ~50 (constant per result, not per shot) | ~50 | always |
| `residual_syndrome` | 120 | ~3,400 | opt-in |
| `pre_decoder_logits` | 1,950 | 54,000 | opt-in |

### Total memory at scale (`pre_decoder_logits` only)

| spec | per shot | 100K shots | 1M shots |
|---|---|---|---|
| d=5, T=5 | 1.95 KB | 191 MB | 1.86 GB |
| d=7, T=7 | 5.36 KB | 535 MB | 5.36 GB |
| d=9, T=9 | 11.4 KB | 1.14 GB | 11.4 GB |
| d=11, T=11 | 20.8 KB | 2.08 GB | 20.8 GB |
| d=15, T=15 | 52.7 KB | 5.27 GB | **52.7 GB** |

**Implication.** `collect_logits=True` is going to OOM consumers who treat
it as a free toggle. Defaults are off; the docstring on `decode_full`
opens with the memory warning.

For consumers who genuinely need logits at scale (e.g. building a
calibration training set), the path is to call `decode_full` in batches
small enough to fit, not to crank `collect_logits=True` on a million-shot
sweep. We are **not** going to ship a streaming / mmap-backed result type
to make this easier, see non-goals.

### Compute cost of the new fields

- `mwpm_weight`: free. PyMatching computes it anyway, `return_weights=True`
  just doesn't throw it away.
- `residual_syndrome`: a `.copy()` of the input (or post-pre-decoder)
  array. Negligible.
- `pre_decoder_logits`: the NVIDIA forward pass already produces this -
  zero extra compute, just don't `.argmax()` and discard.

So the perf impact of `decode_full` over `decode` is **zero on the
PyMatching path** and **a `.cpu().numpy()` copy on the Ising path**. The
"do I OR don't I collect" decision is purely about memory, not throughput.

---

## Backward-compatibility guarantees

- `decode(events) → np.ndarray[bool]` signature and behaviour unchanged.
  Existing scripts continue to work bit-for-bit.
- `evaluate_logical_error_rate(...)` signature gains `collect_telemetry`
  with a default of `False`. Existing call sites see no change in behaviour
  and no change in the returned `LogicalErrorRate` shape (the new
  `telemetry` field is `None` by default).
- `IsingDecoderConfig`, `SurfaceCodePatchSpec`, `SurfaceCodeTensorLayout`,
  the tensor builders, the qiskit/stim bridge, all untouched.
- Constructor of `PyMatchingDecoder` gains a keyword-only
  `enable_correlations` arg with default `False`. Existing positional /
  keyword calls keep working.

In short: a v0.4.x consumer who upgrades to v0.5.0 and changes nothing
sees nothing change.

---

## Boundary rule, where this surface ends

This section is the load-bearing one. External contributors need to see
this in the design doc, not just in private project memory, so the line
is visible BEFORE someone proposes crossing it.

**Forcing question.** When reviewing any PR that touches
`qb_compiler.ising` or extends `IsingDecodeResult`, ask:

> *If a competitor forked qb-compiler tomorrow, does this code give them
> SafetyGate?*

If yes, the PR is in the wrong repo. Point them at whatever paid /
proprietary governance layer they want to build, but the qb-compiler
free/Apache-2.0 surface is for **measurement**, not **policy**.

**Reject triggers.** PRs that get a NACK on sight:

- Any `confidence_threshold`, `min_score`, `gate_at`, `allow_below`,
  `abort_when` parameter on a decoder method or on `IsingDecodeResult`.
- Any module under `qb_compiler/` that imports from `qubitboost_sdk`,
  duplicates SafetyGate logic, or reproduces Trust Passport / governance
  schemas.
- Any "convenience" helper that wraps telemetry into a bool decision -
  `should_accept_shot()`, `is_high_confidence()`, `passes_consistency()`,
  any sibling.
- Any callback parameter on a `qb_compiler.ising` class -
  `on_low_confidence=lambda shot: ...`, `on_abort=...`, `event_handler=...`.
  Callback-driven behaviour is gating dressed as extensibility.
- Any docstring or example that frames a primitive as "for gating" rather
  than "for measurement".
- Any derived field on `IsingDecodeResult` that isn't a pure deterministic
  function of the raw fields with zero free parameters
  (e.g. `pre_decoder_calibrated_probs` with a temperature parameter, no;
  computing your own sigmoid in the docstring example, fine).

**What IS welcome.** The surface deliberately stays open for:

- Pure measurement primitives that emit numbers.
- Reference implementations of decoders / baselines.
- Anything that helps a researcher do their own analysis.
- Bug fixes, perf improvements, doc clarifications.
- Extensions that add MORE measurements without adding policy.

The line is whether the code makes a decision on a signal or just exposes
the signal. Decisions go elsewhere; signals stay here.

---

## Reference implementation

`proto/v050-decode-result` branch carries a working PyMatching
telemetry path + an architecture-faithful NVIDIA stub (random init, real
arch from NVIDIA's Apache-2.0 `code/model/predecoder.py`). It's a
prototype to confirm shapes and memory math, **not** the v0.5.0
implementation. The real impl lands as a fresh PR against this doc.

What the prototype confirmed:

- `pre_decoder_logits` shape is `(B, 4, T, D, D) float32`, same as input
  tensor, single tensor return from `forward()`, no internal sigmoid /
  softmax / argmax.
- `mwpm_weight` is free via PyMatching's `return_weights=True`. Range at
  d=5 p=0.005 is roughly 0-64 with mean ~21 on raw events.
- `mwpm_alternatives` (second-best matching weight) is **not** available
  thru PyMatching's public API, would need a re-decode without the
  chosen edges (~2× cost) or a fork. Out of scope.
- `layout_fingerprint` survives round-trip (16-hex from
  `SurfaceCodeTensorLayout.orientation_fingerprint`).
- Memory math matches expectations, at d=15, T=15, 1M shots the logits
  alone are 52.7 GB. `collect_logits=False` is the only sane default.

---

## Decisions log (rejected during design)

- **`extras: dict` escape hatch**, rejected (above). Closed dataclass.
- **`pre_decoder_probs` derived field**, rejected. Doubles the opt-in
  memory footprint (105 GB vs 52.7 GB at d=15 / 1M shots) and "derived
  convenience field on raw output" drifts toward policy (today its a
  sigmoid; tomorrow its a calibrated sigmoid; the day after its a
  threshold). Consumers call `scipy.special.expit(result.pre_decoder_logits)`
  themselves.
- **`mwpm_alternatives` second-best weight**, rejected. PyMatching
  doesnt expose it; the alternatives (re-decode or fork) cost more than
  the field is worth at v0.5.0. Revisit if real users actually ask.
- **`enable_correlations=True` default**, rejected for v0.5.0. Silently
  changes `mwpm_weight` semantics for downstream consumers. Ship as
  opt-in, document the delta, revisit in v0.6+.
- **`.concat()` classmethod for streaming**, rejected. Streaming
  composition is not qb-compilers job (see non-goals).

---

## Open questions for impl

These are real questions, not bikesheds. Surface them in the tracking
issue and resolve before the PR.

1. **Which side does the NVIDIA pre-decoder forward pass live on for
   `decode_full(collect_logits=True)`?** Today the wrapper consumes its
   output internally and discards before XOR-ing into the residual. Need
   to refactor so the raw logits tensor escapes the method scope without
   doubling the forward pass cost.

2. **Where does `decoder_version` come from on the Ising path** when the
   user passes a `build_model` callable instead of a checkpoint? Two
   options: (a) hash the checkpoint regardless and use that, (b) hash
   `inspect.getsource(config.build_model)` as a tiebreaker. Pick one,
   document it.

3. **`evaluate_logical_error_rate(collect_telemetry=True)` keeps every
   batch's full result in memory until the harness returns.** At
   `collect_logits=True` + multi-batch sweeps thats a recipe for OOM. Do
   we (a) document and walk away, (b) stream batches to disk, (c) require
   `collect_telemetry` and `collect_logits` to be mutually exclusive in
   the harness call? Lean toward (a), but flag for review.

4. **`pickle`/`json` round-trip support for `IsingDecodeResult`.** The
   numpy fields pickle fine. JSON needs a `.to_json_dict()` helper that
   downsamples the big tensors (e.g. summary stats only). This is genuinely
   useful for Trust Passport provenance, but its also exactly the kind of
   convenience helper that drifts toward policy. Decide explicitly.

---

## Tracking

Issue: TBD (open after this doc lands on master).
Prototype branch: `proto/v050-decode-result` (local, not pushed -
reference only, do not merge).
