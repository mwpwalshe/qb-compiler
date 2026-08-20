# Receipts: what ran, and who says so

A receipt records what a check saw. It is a plain JSON object, it is emitted by several commands,
and from 0.12.0 it can be signed with a key that outlives the call and verified by somebody who has
nothing but the receipt and a public key.

Two corrections landed in 0.12.0 and both are worth reading before you rely on an older receipt.

## The receipt describes the layout that ran

`selection_receipt()` used to describe the mapper's recommendation, whatever the caller executed. A
campaign here ran layout `[6, 5, 4, 3, 2]` and got a receipt naming `{144, 143, 136, 123, 124}`. The
receipt was internally consistent and described a run that never happened.

Pass `executed_layout=` when you run something other than the recommendation:

```python
import functools
from qb_compiler.passes.mapping import CalibrationMapper, selection_receipt

mapper = CalibrationMapper(backend)
result = mapper.run(circuit, {})

receipt = selection_receipt(
    result,
    calibration=backend,
    executed_layout={0: 6, 1: 5, 2: 4, 3: 3, 4: 2},
    scorer=functools.partial(mapper.score_layout, circuit=circuit),
)
```

What you get back:

| field | meaning |
|---|---|
| `selected_layout` | the layout that ran |
| `selected_score` | its calibration score, computed by the `scorer` when you did not supply one |
| `describes_executed_layout` | always present, so nothing has to be inferred |
| `recommended_layout`, `recommended_score` | what the pass would have picked |
| `executed_layout_matches_recommendation` | whether those two are the same |
| `score_penalty_vs_recommended` | what the override cost, on the mapper's own scale |
| `divergence_note` | a sentence saying which of the two happened |

Without `executed_layout` the receipt is exactly what it always was, plus
`describes_executed_layout: true`. Nothing that reads an older receipt breaks.

The score is a sum of weighted penalties on one calibration snapshot. It compares layouts against
each other on that snapshot. It is not a fidelity and does not convert to one.

## Signing, and what a signature is worth

Before 0.12.0, `sign=True` generated a fresh keypair for every call and put the public half into the
receipt it had just signed. Every such receipt verified, against the key travelling inside it, and
none of them said anything about where they came from. Do not treat a receipt carrying a
`public_key` field as evidence of origin. The verifier reports those as `LEGACY_SELF_SIGNED` and
refuses them.

Now:

```python
receipt = selection_receipt(result, calibration=backend, sign=True)
```

The key is created once at `~/.qb-compiler/signing_key` with mode 0600, or wherever
`QBC_SIGNING_KEY` points, and reused afterwards. The receipt carries the key's **fingerprint**, not
the key. `sign=True` raises rather than quietly handing back an unsigned receipt.

Publish the public half so people can check your receipts:

```python
from qb_compiler.signing import export_public_key
export_public_key("qbc-public-key.txt")
```

## Verifying, which is free and offline

```bash
qbc verify-receipt receipt.json --key qbc-public-key.txt
```

```python
from qb_compiler.signing import verify_receipt
result = verify_receipt(receipt, public_key=their_public_key)
print(result.status, result.ok)
```

| verdict | means |
|---|---|
| `VERIFIED` | the bytes were signed by the holder of that key |
| `UNSIGNED` | no signature. The contents can still be read; nothing attests to their origin |
| `NO_KEY` | signed, but no key was supplied, so it cannot be checked |
| `INVALID_SIGNATURE` | altered after signing, or signed by a key you do not have |
| `LEGACY_SELF_SIGNED` | the pre-0.12.0 shape described above |
| `MALFORMED` | not a receipt this verifier can read |

`ok` is true only for `VERIFIED`. "I could not check it" is never a pass.

The signature covers a canonical JSON serialisation of the whole receipt except the signature field
itself, so changing any other field, including the fingerprint, breaks it.

Verification needs no key of your own, no account, and no network. It falls back to a pure Python
Ed25519 implementation when a compiled one is not installed, so `pip install qb-compiler` is enough
to check anybody's receipt.

A signature says these bytes came from that key holder. It does not say the layout was good, the run
was worth doing, or the numbers are right. `verify_receipt` prints that distinction rather than
leaving a reader to assume the stronger one.

## Where receipts come from

| command | schema |
|---|---|
| `selection_receipt()` | `qb.selection_receipt.v1` |
| `qbc chem-audit --json` | `qb.chem_audit.v1` |
| `qbc measure-plan --json` | `qb.measure_plan.v1` |
| `qbc dem-audit --json` | `observablegate.receipt/1` |
| `qbc when --json` | `qb.cross_vendor_advice.v1` |
| `qbc compile --receipt` | compilation passport, see `qb_compiler.receipts` |

Any of them can be signed with `qb_compiler.signing.sign_receipt(receipt)` and checked with
`qbc verify-receipt`.

## Calibration age rides on every selection receipt

A layout is chosen from calibration data, and that data has an age nobody surfaces. Every selection
receipt carries it:

```json
"calibration_freshness": {
  "calibration_timestamp": "2026-03-14T08:41:35.899941+00:00",
  "age_minutes": 59.99,
  "tolerance_minutes": 30.0,
  "tolerance_basis": "builtin_default_not_measured_for_this_device",
  "exceeds_default_tolerance": true,
  "timestamp_status": "measured",
  "note": "Age is measured, tolerance is a fixed default. ..."
}
```

The age is measured. The tolerance is one blunt default for every device, and `tolerance_basis` says
so on every receipt, because a default presented as a measurement is worse than no number. A real
tolerance is a property of a specific device and its recalibration schedule, and measuring one means
comparing what was visible at a moment against what the vendor later reports was true at that
moment, over weeks. That measurement is not in this package.

`timestamp_status` is `measured`, `absent`, `unreadable`, `synthetic` (static specs, which have no
age) or `clock_skew` (a timestamp in the future, reported rather than treated as fresh).
