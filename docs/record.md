# Records: how they are built, and what a decoder leaves on them

A decoder result is a statement about a record. If the record was assembled wrong, the result is
wrong by a factor and nothing downstream can tell: the detectors build, the decoder runs, the
logical error rate comes out, and the plot looks like every other plot.

`qb_compiler.record` holds eight checks for the ways that happens, loaders that get two real
records right, the `.npz` contract for everything else, and one measurement.

```bash
pip install 'qb-compiler[record]'
```

## What "validated" means, and what it does not

A record that passes these checks is **correctly built**. That is the whole claim.

It is not complete. A correctly built record can still carry structure that the decoder reading
it does not use, and saying the construction is sound says nothing either way about that.

The QuEra d5 Z release is the worked example. It passes every check that can run on it: event
rate 0.133, first round at 0.45 times the steady median, all 48 deterministic stabilizer means
outside the band, labels reconstructing exactly from the final readout. And the confidence
value published alongside it is still worth
**0.02 bits per shot above a matching decoder's own summary on the published model, on the
released d5 Z record, all shots, 5-fold cross-validation**. Correctly built, and not exhausted.

## Quickstart

```python
import dataclasses

from qb_compiler.record import residual, validate
from qb_compiler.record.dem import build_repetition_dem, decode_records
from qb_compiler.record.loaders import ibm_fez_repetition

loaded = ibm_fez_repetition.load("~/records/ibm_fez", d=11, r=11)
spec = dataclasses.replace(loaded, dem=build_repetition_dem(11, 11, p_data=0.01, p_meas=0.01))
print(validate(spec))                                    # every check, run or skipped with a reason

served = decode_records(spec.dem, spec.detector_matrix())
print(residual(spec, {"prediction": served}, my_features))
```

## Two traps, with the numbers each one leaves behind

### Rounds stored last round first

**Where:** IBM Fez repetition-code memory runs. The `c_syndrome_<chain>` register stores its
rounds in reverse, so reshaping it to `(shots, rounds, sites)` and using it as it comes runs the
experiment backwards in time.

**Symptom:** the stored round 0 detector rate comes out at **0.19 to 0.28** across d5 r5 to
d11 r11, where the corrected orientation gives **0.05 to 0.08**. A first round differences
against a reset, so it should fire at roughly half the steady rate, not twice it. The final data
parity also mismatches the round the loader calls last 2.5 times more often than the one it calls
first, which is backwards: the final readout happens at the end.

**Cost of missing it:** logical error rate **3.83 percent instead of 1.62** at d5 r5, and **9.69
instead of 1.43** at d11 r11. Nothing errors.

**Fix:** reverse the round axis on load. `ibm_fez_repetition.load` does it by default;
`reverse_rounds=False` is there to measure the fault, not to decode in.

**Caught by:** `round_profile`, `endpoint_agreement`, `state_profile`.

### Detectors built by differencing raw readouts

**Where:** the public QuEra surface-code release
([Zenodo 15685795](https://doi.org/10.5281/zenodo.15685795)). The archive stores
`measurement_events`: raw ternary ancilla readouts, with `2` marking an atom lost before it could
be read. Raw readouts are not detectors, and differencing consecutive ones is the obvious thing
to do with them.

**Symptom:** detector event rate **0.448**, against **0.133** from the publisher's own
construction. The deterministic stabilizer means land between **0.487 and 0.633**, where the
correct construction puts them between **0.023 and 0.947**. The readouts were never referenced to
a frame, so they are close to coin flips.

**Cost of missing it:** a decoder still runs on them and still reports a logical error rate. It
means nothing.

**Fix:** build detectors through the decoding framework the publisher ships with the dataset.
`quera_surface.load` calls it, and refuses `detectors="naive"` rather than offering it.

**Caught by:** `event_density`, `type_consistency`.

Cite the release as the publisher asks:

> QuEra Computing, surface code dataset, Zenodo record 15685795. Cite the record and the paper it
> accompanies, both named on the record page.

Nothing is redistributed by this package. `qbc corpus verify quera-surface-code <your copy>`
checks the archive against a pinned digest first.

## The eight checks

| check | computes | passes when | catches |
|---|---|---|---|
| `round_profile` | event rate per round | the record declares a quiet first layer, and the first round is at most 0.9x the median of the steady rounds with the last at most 1.5x | mirrored rounds, a final round differenced against the wrong round |
| `endpoint_agreement` | final parity against the first and last stored rounds | the round treated as last does not mismatch more than twice as often as the one treated as first | rounds stored last round first |
| `event_density` | overall event rate | strictly inside (0.005, 0.35) | unframed readouts, dead detectors |
| `type_consistency` | per-(round, site) means of the deterministic type | every mean outside [0.3, 0.7] | unframed readouts |
| `time_mirror_control` | error rate decoded as given against decoded with the rounds reversed, model unchanged | the paired per-shot difference is at most 3 of its own standard errors above zero, with at least 30 failures in each arm | a record the reversed decode materially beats |
| `state_profile` | stored readout rate per round, split by prepared state | on excited-state chains the rate never falls by more than 0.005 round on round | mirrored storage, seen per state |
| `dem_fingerprint` | mechanism count, boundary count, degree and mechanism-size histograms | exact match against a supplied expectation | the wrong error model |
| `label_reconstruction` | GF(2) solve of the observable on shots with no loss | at least 0.999 agreement | the wrong observable |

`event_density` is always critical, `round_profile` is critical on a record that declares the
first layer premise, and `dem_fingerprint` is critical when an expected fingerprint is supplied.
`report.passed` is every critical check passing; a critical check that could not run does not
pass, because a record that was not checked has not been validated. Every other check is
advisory: it is reported, and it does not vote.

`round_profile` assumes the platform's first layer is quiet, which holds where the first round
differences against a prepared value and not where preparation and final readout are noisier than
a mid-run reading, so it runs on a record that declares that premise in
`meta["first_layer_premise"]` and abstains on one that does not, reporting the layer profile it
measured either way. The `ibm_fez_repetition` and `quera_surface` loaders declare it for their
platforms with the limits above; anyone else declares it for theirs:

```python
from qb_compiler.record import FIRST_LAYER_PREMISE, RecordSpec

spec = RecordSpec(detectors=dets, labels=labels, meta={FIRST_LAYER_PREMISE: {
    "source": "measured on our own memory runs: first layer 0.5 times the steady median",
}})
```

Pass `first_round_max_ratio` and `last_round_max_ratio` in that mapping to declare the ratios your
platform was measured at, or `True` to take the shipped ones.

`endpoint_agreement` and `time_mirror_control` are advisory on purpose, and each is one-sided.
They call a fault only when the record carries its signature, and they pass a record that gives
them nothing to read. A check with no power on a record should not vote against it, and both of
these have records on which they have none.

### Where the thresholds come from, and how to move them

Each default is in `DEFAULT_THRESHOLDS` with its measured reason in the source. Three are worth
stating here because they were set against real records rather than picked:

* `type_consistency` uses **[0.3, 0.7]**, not [0.2, 0.8]. The deterministic type drifts toward
  even as rounds accumulate: on the QuEra d5 Z release the 48 deterministic means run 0.023 to
  0.947 at round 0 but 0.228 to 0.901 at round 3, and nine of them sit inside [0.2, 0.8] on a
  record that is correctly built. None sit inside [0.3, 0.7]. The unframed construction on the
  same record gives 0.487 to 0.633, every one inside the band.
* `state_profile` allows a round-to-round fall of **0.005**. On corrected IBM Fez records the
  smallest step up is +0.008 at d11 r11, +0.017 at d9 r9 and +0.049 at d5 r5; read in stored
  order the same records step down by 0.086 to 0.098.
* `endpoint_agreement` fails only above **2.0x**, and reports no power between 0.8x and 1.25x. An
  earlier rule also required the last round to mismatch at most half as often as the first, which
  fails a correct record whose chains are all prepared in the ground state: those accumulate no
  asymmetry between the endpoints, and measure 0.094 against 0.110. The pooled IBM Fez record
  passed that rule only because its prepared-one chains dominate it.

Override one field of one check and the rest keep their defaults:

```python
report = validate(spec, thresholds={"event_density": {"high": 0.4}})
report.check("event_density").threshold["high"]   # 0.4, and it is in the report
```

```bash
qbc record validate record.npz --thresholds '{"event_density": {"high": 0.4}}'
```

### What the time mirror control does and does not catch

It decodes the record, then decodes it with the detector axis reversed and the error model
unchanged, and compares the two per shot. It catches a record that the reversed decode materially
beats.

**It does not catch a syndrome storage order fault**, which is what its name might suggest.
Detectors rebuilt from reversed stored rounds are not the reverse of the detectors built from the
correct ones, so the two operations are different. Reversing the detector axis of a corrected
IBM Fez d11 r11 record moves the logical error rate from 1.434 percent to 1.477, well inside
noise, and the control passes on the corrected record, on the stored-order record, and on one
deliberately reversed. Storage order is caught by `round_profile`, `endpoint_agreement` and
`state_profile`, which is why three checks look at it from three directions.

The rule is deliberately conservative, for a reason worth stating. The check compares the paired
per-shot difference against 3 of its own standard errors, and it refuses to run below 30 failures
in an arm. An earlier form of it passed when the mirrored rate was at least the real rate minus
one standard error, and that is a one-sigma one-sided test: under the null of no difference, which
is where every repetition memory sits, it fails a correct record about one time in six at any
sample size. It did exactly that on a synthetic d5 r5 record of 4,000 shots, reporting 13 failures
against 9 and calling it a fault. Hence the paired comparison, the wider limit, the failure floor,
and the check being advisory rather than critical.

## The residual metric

```python
report = residual(spec, decoder_output, features, holdout="group", C=0.1, n_nulls=20, seed=0)
```

**Definition.** Take the decoder's own per-shot output, everything soft it produced, plus the
coarsest counts of the record. Fit a model predicting whether the decoder failed on that shot and
measure its held-out log loss. Add the feature block, refit, measure again. The drop, in bits per
shot, is the residual.

**Estimator.** Standardize, then logistic regression at the given `C`. Hold out by `groups` when
the record carries them, otherwise stratified folds over shots, with a warning in the report
either way: shot-level folds put neighbouring shots on both sides of the split, so anything that
drifts within a run leaks across it.

**Baseline.** Every entry of `decoder_output`, which must include `prediction` and should include
every soft value the decoder produced, plus total events, per-round events, and total loss when
the record carries a loss array. The report lists the columns it used.

**Null floors.** A held-out difference of zero is not what an uninformative block scores, so the
same measurement runs on records with their structure destroyed and their counts preserved:

* **geometry nulls** shuffle detector events within each round and recompute the features through
  your own feature function. Pass `features` as a callable to get these; a bare matrix cannot be
  recomputed on a shuffled record and the report says the null did not run.
* **shot-permutation nulls** reorder the feature rows against the shots.

`null_floor_p95` is the 95th percentile over whichever nulls ran, and `above_floor` is
`residual_bits > null_floor_p95`.

**What a number means.** That a feature block anticipates some of this decoder's failures that
its own output did not. That is all it means. It is a comparison against one decoder's summary on
one record:

* it is **not a decoder benchmark**, and does not rank decoders;
* a residual above the floor does **not** say a better decoder exists, or that one could be built;
* a residual at the floor does **not** say the record holds nothing, only that these features on
  this record with this estimator did not separate from the null.

The report states the number, the floor, both AUCs, the columns, the fold kind, the seed and the
warnings. It does not read anything into any of them.

## The `.npz` contract

One file, named keys, declared dtypes. Required: `dets` `(n, rounds, sites)` uint8 and `labels`
`(n,)` uint8. Optional: `site_coords`, `groups`, `loss`, `det_index`, `H`, `L`, `weights`,
`edge_qubits`, `raw`, `final_parity`, `final_data`, `state`, `data0`, `det_sites`, and `meta` as a
JSON string. `det_index` gives each cell its error-model detector number, with `-1` where a cell
holds no detector, which is how a code with detectors on one stabilizer type in the first and last
rounds is written down; leaving it out means every cell is a detector, numbered row major.

```python
from qb_compiler.record.loaders import read_npz, write_npz

write_npz("record.npz", spec)
spec = read_npz("record.npz")
```

The reader refuses a wrong dtype or a wrong shape rather than coercing it. A record quietly
widened from uint8, or reshaped from three axes to two, still decodes, and the number it produces
is wrong in a way nothing later in the pipeline can see.

## Command line

```bash
qbc record validate record.npz
qbc record validate record.npz --thresholds thresholds.json
qbc record residual record.npz --decoder-output out.npz --features features.npz
qbc record load-fez ~/records/ibm_fez -d 11 -r 11 --out d11r11.npz
```

Human summary on stderr, JSON on stdout. Exit 0 every critical check passed, 2 a critical check
failed, 3 could not run, 1 error.

## Running the tests against a real record

The tests that read a real record are skipped unless one is pointed at. Nothing downloads and
nothing is redistributed.

| variable | points at |
|---|---|
| `QB_RECORD_DATA` | a directory holding `ibm_fez/`, which holds the `d<d>_r<r>` regime directories |
| `QB_QUERA_ZIP` | the QuEra archive as the publisher serves it. Defaults to `$QB_RECORD_DATA/quera_surface_code.zip` |
| `QB_QUERA_VENDOR` | the decoding framework published with that dataset, the directory holding `memory.py` and `noise_model.py` |
| `QB_CAT_DATA` | the published AWS cat qubit repetition-code deposit. Defaults to `~/qec_public_data/aws_cat_2025/data_upload` |

```bash
QB_RECORD_DATA=~/records QB_QUERA_VENDOR=~/ML_decoder pytest tests/integration/test_record_data.py
```

## Beyond validation

The QubitBoost SDK consumes these reports for record-sufficiency certification and governed
decoder adaptation. See [qubitboost.io/compiler](https://qubitboost.io/compiler).
