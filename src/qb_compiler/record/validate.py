# SPDX-License-Identifier: Apache-2.0
"""Eight checks on how a record was built.

None of these ask whether a decoder is any good. They ask whether the thing the decoder was handed
is the thing the experiment produced: rounds in the order they happened, readouts referenced to a
frame, detectors differenced against the right neighbour, the error model that belongs to the
run, and the observable the labels actually describe.

Every one of them was written because a real record failed it. A record that fails silently
produces a decoder benchmark that looks ordinary and is wrong by a factor, and nothing downstream
can tell.

============ ====================================================== ==========================
check        catches                                                needs
============ ====================================================== ==========================
V1 round_profile        mirrored rounds, final round differenced     detectors, a declared
                        against the wrong round                      first layer premise
V2 endpoint_agreement   last-round-first storage                     raw readouts, final parity
V3 event_density        unframed readouts, dead detectors            detectors
V4 type_consistency     unframed readouts                            raw readouts, site types
V5 time_mirror_control  a record the reversed decode beats           error model, labels
V6 state_profile        mirrored storage, seen per prepared state    raw readouts, states
V7 dem_fingerprint      the wrong error model                        error model, expectation
V8 label_reconstruction the wrong observable                         final readout, labels
============ ====================================================== ==========================

V1 and V3 are critical, and so is V7 when an expected fingerprint is supplied; ``passed`` is every
critical check passing. The rest are advisory and are reported. A check that cannot run says which
input it wanted rather than assuming a default.

V2 and V5 are deliberately advisory. Each is one-sided: V2 only calls a fault when the record
carries the mirrored signature, and V5 only when decoding the reversed record is materially
better. Both pass a record that simply gives them nothing to read, because a check with no power
on a record should not vote against it.

V1 has power only where the platform's first round is quiet, which is a property of the platform
and not of every record. The record declares that premise through
:data:`~qb_compiler.record.types.FIRST_LAYER_PREMISE` in its ``meta``, which the loaders that know
their platform set. On a record that declares none, V1 reports the layer profile it measured and
abstains: SKIP, not a fault, and it does not vote.

Thresholds
----------
:data:`DEFAULT_THRESHOLDS` ships the defaults below. Pass ``thresholds={"event_density":
{"high": 0.4}}`` to override one field of one check; everything not named keeps its default, and
the values actually used come back in each :class:`~qb_compiler.record.types.Check`.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from qb_compiler.record import controls
from qb_compiler.record.dem.fingerprint import fingerprint, fingerprint_matches
from qb_compiler.record.types import (
    FAIL,
    FIRST_LAYER_PREMISE,
    PASS,
    SKIP,
    Check,
    RecordSpec,
    ValidationReport,
)

SUPPORTED_DECODERS = ("mwpm",)

DEFAULT_THRESHOLDS: dict[str, dict[str, Any]] = {
    # V1. The first round is fresh: it differences against a reset rather than against a previous
    # round, so it fires at roughly half the steady rate. The last differences the final data
    # parity against the last stabilizer round, which is quieter than a stabilizer round, not
    # louder. Measured on corrected IBM Fez repetition-code memory runs, d5 r5 to d11 r11:
    # first-round rate 0.052 to 0.083 against steady medians 0.099 to 0.145, ratios 0.52 to 0.57;
    # the same records read in stored order give first-round 0.192 to 0.277, ratios 1.94 to 1.91.
    #
    # Both limits apply only to a record that declares the quiet first round as a premise, because
    # the premise is false on some platforms. On the published AWS cat qubit repetition-code
    # deposit (2025), where preparation and final readout are noisier than a mid-run ancilla
    # reading, a correctly built record is loud at both ends and flat between: the phase-flip
    # section at nbar 2.0 over 13 cycles measures a first layer of 0.2409 against a steady median
    # of 0.1824, a ratio of 1.32, on 13,503 shots. These numbers are unchanged from the records
    # they were measured on and nothing here is tuned to that deposit; the check abstains there
    # instead.
    "round_profile": {"first_round_max_ratio": 0.9, "last_round_max_ratio": 1.5},
    # V2. Asymmetric, and only the mirrored signature is a finding. Measured on corrected IBM Fez
    # repetition-code memory runs: the round the corrected loader treats as last mismatches 0.077
    # to 0.111 of the time against 0.202 to 0.283 for the round it treats as first, a ratio of
    # 0.38 to 0.39; reading them in stored order inverts it to 2.5 to 2.6. The earlier rule, which
    # also required the last round to mismatch at most half as often as the first, fails a correct
    # record whose chains are all prepared in the ground state: such chains accumulate no endpoint
    # asymmetry and give two near equal rates, measured at 0.094 against 0.110. The pooled Fez
    # record only passed that rule because its prepared-one chains dominate. So the band 0.8 to
    # 1.25 is reported as carrying no power rather than treated as a fault.
    "endpoint_agreement": {"max_ratio": 2.0, "no_power_low": 0.8, "no_power_high": 1.25},
    # V3. Below the floor the detectors are dead or the record is all one value; above the ceiling
    # they are coin flips, which is what unframed readouts look like. Measured: 0.133 on the QuEra
    # d5 Z release built with the vendor construction, 0.448 built naively.
    "event_density": {"low": 0.005, "high": 0.35},
    # V4. A deterministic stabilizer reads near 0 or near 1; an unframed one reads near 0.5. The
    # band is [0.3, 0.7] rather than the [0.2, 0.8] this check was first written with, because the
    # deterministic type drifts toward 0.5 as rounds accumulate and [0.2, 0.8] fails a correctly
    # built record: on the QuEra d5 Z release the 48 deterministic (round, stabilizer) means run
    # 0.023 to 0.947 at round 0 but 0.228 to 0.901 at round 3, nine of them inside [0.2, 0.8] and
    # none inside [0.3, 0.7]. The 48 unframed means on the same record run 0.487 to 0.633, every
    # one inside the band. Loss symbols are excluded from each mean.
    "type_consistency": {"low": 0.3, "high": 0.7, "min_shots": 100},
    # V5. Advisory, never critical, and a two-part rule. The earlier form passed when the mirrored
    # rate was at least the real rate minus one standard error, which is a one-sigma one-sided test
    # and so fails a time-symmetric record about one time in six at ANY sample size. Every
    # repetition memory is time symmetric to within noise, so that form failed correct records: on
    # a synthetic d5 r5 run of 4,000 shots it reported 13 failures against 9 and called FAIL. The
    # rule now compares the PAIRED per-shot difference against 3 of its own standard errors, and
    # refuses to run at all below 30 failures in an arm, where no one-sided comparison is possible.
    "time_mirror_control": {"n_sigma": 3.0, "min_failures": 30, "max_shots": 20000, "seed": 0},
    # V6. Relaxation accumulates, so on chains prepared in the excited state the stored readout
    # rate climbs round on round. Measured on corrected IBM Fez repetition-code memory runs: the
    # smallest round-to-round step is +0.008 (d11 r11), +0.017 (d9 r9), +0.049 (d5 r5); read in
    # stored order the same records step down by 0.086 to 0.098. The tolerance sits between.
    "state_profile": {"min_step": -0.005},
    # V7. No default expectation: pass one in as {"dem_fingerprint": {"expected": {...}}} and the
    # check becomes critical, leave it out and the fingerprint is recorded.
    "dem_fingerprint": {"expected": None},
    # V8. The observable is a fixed parity of the final data readout, so it reconstructs exactly on
    # shots with no loss. Measured: 1.000 on the 205 loss-free shots of the QuEra d5 Z release.
    "label_reconstruction": {"min_agreement": 0.999},
}


def _merge_thresholds(overrides: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    merged = copy.deepcopy(DEFAULT_THRESHOLDS)
    if not overrides:
        return merged
    unknown = set(overrides) - set(merged)
    if unknown:
        raise ValueError(
            f"unknown check name(s) in thresholds: {sorted(unknown)}. Known: {sorted(merged)}"
        )
    for check_name, values in overrides.items():
        merged[check_name].update(values)
    return merged


def _skip(
    name: str, reason: str, *, critical: bool = False, used: dict[str, Any] | None = None
) -> Check:
    return Check(name=name, status=SKIP, critical=critical, detail=reason, threshold=used or {})


def _flat_loss(spec: RecordSpec) -> np.ndarray | None:
    if spec.loss is None:
        return None
    loss = np.asarray(spec.loss)
    return loss.reshape(loss.shape[0], -1)


def _round_rates(spec: RecordSpec) -> np.ndarray:
    dets = np.asarray(spec.detectors, dtype=np.float64)
    valid = spec.site_valid
    rates = np.zeros(spec.n_rounds, dtype=np.float64)
    for t in range(spec.n_rounds):
        cols = np.flatnonzero(valid[t])
        rates[t] = float(dets[:, t, cols].mean()) if cols.size else np.nan
    return rates


def _declared_premise(spec: RecordSpec) -> tuple[dict[str, Any] | None, str]:
    """The record's first layer premise, and a phrase naming what stood in its place.

    ``True`` means the shipped limits. A mapping may carry ``first_round_max_ratio``,
    ``last_round_max_ratio`` and a ``source``. Anything else is not a declaration, and the phrase
    says what was there so a caller can see why the check abstained.
    """
    declared = spec.meta.get(FIRST_LAYER_PREMISE)
    if declared is True:
        return {}, ""
    if isinstance(declared, dict):
        return dict(declared), ""
    if declared is None:
        return None, ""
    return None, f", and meta[{FIRST_LAYER_PREMISE!r}] holds {declared!r}, which declares nothing"


# ---------------------------------------------------------------- V1
def check_round_profile(spec: RecordSpec, threshold: dict[str, Any]) -> Check:
    """Detector event rate per round, against the steady middle.

    Runs only on a record that declares :data:`~qb_compiler.record.types.FIRST_LAYER_PREMISE`, the
    premise that this platform's first round is quiet. The rule separates a fault from a sound
    record only where that holds; on a platform whose preparation and final readout are noisier
    than a mid-run reading, both ends are loud on a record that is built correctly. So the premise
    is declared rather than assumed, and a record that declares none has its layer profile
    reported and nothing decided on it.
    """
    name = "round_profile"
    if spec.n_rounds < 3:
        return _skip(
            name,
            f"needs at least 3 rounds to have a steady middle to compare against, got "
            f"{spec.n_rounds}",
            critical=True,
            used=threshold,
        )
    rates = _round_rates(spec)
    middle = rates[1:-1]
    if not np.isfinite(middle).any():
        return _skip(
            name,
            "no round between the first and the last holds a detector",
            critical=True,
            used=threshold,
        )
    median = float(np.nanmedian(middle))
    measured = {
        "rate_per_round": [float(r) for r in rates],
        "steady_median": median,
        "first_round_rate": float(rates[0]),
        "last_round_rate": float(rates[-1]),
    }
    if median <= 0.0:
        return _skip(
            name,
            f"the steady rounds have a median event rate of {median:.6f}, so there is no scale to "
            "compare the endpoints against",
            critical=True,
            used=threshold,
        )
    first_ratio = float(rates[0]) / median
    last_ratio = float(rates[-1]) / median
    measured["first_round_ratio"] = first_ratio
    measured["last_round_ratio"] = last_ratio

    premise, instead = _declared_premise(spec)
    if premise is None:
        detail = (
            "no declared first layer premise for this platform; supply one to enable the check"
            f"{instead}. Measured and not judged: first round {rates[0]:.4f} "
            f"({first_ratio:.2f}x the steady median {median:.4f}), last round {rates[-1]:.4f} "
            f"({last_ratio:.2f}x). The rule reads a first round that differences against a "
            "prepared value and so fires below the steady rate. Where preparation and final "
            "readout are noisier than a mid-run reading, both ends of a correctly built record "
            f"are loud, so the record says whether the premise holds: set meta"
            f"[{FIRST_LAYER_PREMISE!r}]"
        )
        return Check(name, SKIP, False, detail, measured, threshold)
    measured["premise"] = premise

    used = dict(threshold)
    for field in ("first_round_max_ratio", "last_round_max_ratio"):
        if field in premise:
            used[field] = float(premise[field])
    first_limit = float(used["first_round_max_ratio"])
    last_limit = float(used["last_round_max_ratio"])
    first_ok = first_ratio <= first_limit
    last_ok = last_ratio <= last_limit
    if first_ok and last_ok:
        detail = (
            f"first round {rates[0]:.4f} ({first_ratio:.2f}x the steady median {median:.4f}), "
            f"last round {rates[-1]:.4f} ({last_ratio:.2f}x)"
        )
        return Check(name, PASS, True, detail, measured, used)
    problems = []
    if not first_ok:
        problems.append(
            f"the first round fires at {rates[0]:.4f}, {first_ratio:.2f}x the steady median "
            f"{median:.4f}, above the {first_limit}x limit this record declares. A first round "
            "differences against a reset and should fire at roughly half the steady rate; firing "
            "at or above it is what a record stored last round first looks like"
        )
    if not last_ok:
        problems.append(
            f"the last round fires at {rates[-1]:.4f}, {last_ratio:.2f}x the steady median "
            f"{median:.4f}, above the {last_limit}x limit this record declares. The last round "
            "differences the final data parity against the last stabilizer round, so it should be "
            "quieter than a stabilizer round, not louder"
        )
    return Check(name, FAIL, True, "; ".join(problems), measured, used)


# ---------------------------------------------------------------- V2
def check_endpoint_agreement(spec: RecordSpec, threshold: dict[str, Any]) -> Check:
    """Does the final data parity disagree with the round the loader calls last?

    Asymmetric, and advisory. Only the mirrored signature is a finding: the round treated as last
    mismatching the final parity much more often than the round treated as first. Equal rates are
    not evidence of a fault, because a record whose chains are all prepared in the ground state
    accumulates no asymmetry between its endpoints and so gives two equal rates when it is
    correctly built.
    """
    name = "endpoint_agreement"
    if spec.raw_syndromes is None or spec.final_parity is None:
        missing = [
            field
            for field, value in (
                ("raw_syndromes", spec.raw_syndromes),
                ("final_parity", spec.final_parity),
            )
            if value is None
        ]
        return _skip(name, f"needs {' and '.join(missing)}; not in this record", used=threshold)
    raw = np.asarray(spec.raw_syndromes, dtype=np.uint8)
    final = np.asarray(spec.final_parity, dtype=np.uint8)
    if raw.ndim != 3 or raw.shape[1] < 2:
        return _skip(
            name,
            f"needs raw_syndromes of shape (n_shots, n_rounds >= 2, n_sites), got {raw.shape}",
            used=threshold,
        )
    if final.shape != raw.shape[0::2]:
        return _skip(
            name,
            f"final_parity {final.shape} does not match raw_syndromes {raw.shape} on shots and "
            "sites",
            used=threshold,
        )
    mismatch_last = float((final != raw[:, -1]).mean())
    mismatch_first = float((final != raw[:, 0]).mean())
    measured = {"mismatch_last_round": mismatch_last, "mismatch_first_round": mismatch_first}
    if mismatch_first <= 0.0:
        return _skip(
            name,
            "the round the loader treats as first agrees with the final parity everywhere, so "
            "there is nothing to compare against",
            used=threshold,
        )
    ratio = mismatch_last / mismatch_first
    measured["ratio"] = ratio
    low, high = float(threshold["no_power_low"]), float(threshold["no_power_high"])
    no_power = low <= ratio <= high
    measured["no_power"] = no_power
    if ratio <= float(threshold["max_ratio"]):
        detail = (
            f"the final parity mismatches the round stored last {mismatch_last:.4f} of the time "
            f"against {mismatch_first:.4f} for the round stored first ({ratio:.2f}x)"
        )
        if no_power:
            detail += (
                "; no power on this record, because the two rates are within "
                f"[{low}, {high}] of each other and a correct record whose chains are all prepared "
                "in the ground state looks exactly like this"
            )
        return Check(name, PASS, False, detail, measured, threshold)
    detail = (
        f"the final parity mismatches the round this loader treats as LAST {mismatch_last:.4f} of "
        f"the time, against {mismatch_first:.4f} for the round it treats as FIRST ({ratio:.2f}x, "
        f"limit {threshold['max_ratio']}x). The final data readout happens at the end of the run, "
        "so it should not disagree with the round nearest the end far more than with the far one. "
        "That is what rounds stored last round first look like: try loading with the round order "
        "reversed"
    )
    return Check(name, FAIL, False, detail, measured, threshold)


# ---------------------------------------------------------------- V3
def check_event_density(spec: RecordSpec, threshold: dict[str, Any]) -> Check:
    """Overall detector event rate."""
    name = "event_density"
    dets = np.asarray(spec.detectors, dtype=np.float64)
    valid = spec.site_valid
    total = float(dets[:, valid].sum())
    count = float(spec.n_shots * spec.n_detectors)
    if count <= 0:
        return _skip(name, "the record holds no detectors", critical=True, used=threshold)
    rate = total / count
    measured = {"event_rate": rate, "n_detectors": spec.n_detectors, "n_shots": spec.n_shots}
    low, high = float(threshold["low"]), float(threshold["high"])
    if low < rate < high:
        return Check(
            name,
            PASS,
            True,
            f"detector event rate {rate:.4f}, inside ({low}, {high})",
            measured,
            threshold,
        )
    if rate <= low:
        detail = (
            f"detector event rate {rate:.6f} is at or below {low}. Either the detectors are dead, "
            "or the record holds a constant, or the differencing produced nothing"
        )
    else:
        detail = (
            f"detector event rate {rate:.4f} is at or above {high}. Detectors this busy are close "
            "to coin flips, which is what readouts that were never referenced to a frame look "
            "like; a construction that differences consecutive raw readouts produces exactly this"
        )
    return Check(name, FAIL, True, detail, measured, threshold)


# ---------------------------------------------------------------- V4
def check_type_consistency(spec: RecordSpec, threshold: dict[str, Any]) -> Check:
    """Per-(round, site) means of the deterministic stabilizer type."""
    name = "type_consistency"
    if spec.raw_syndromes is None:
        return _skip(name, "needs raw_syndromes; not in this record", used=threshold)
    if spec.deterministic_sites is None:
        return _skip(
            name,
            "needs deterministic_sites naming which stabilizers are deterministic in this basis; "
            "not in this record",
            used=threshold,
        )
    raw = np.asarray(spec.raw_syndromes)
    sites = np.asarray(spec.deterministic_sites, dtype=np.int64).ravel()
    if sites.size == 0:
        return _skip(name, "deterministic_sites is empty", used=threshold)
    if raw.shape[0] < int(threshold["min_shots"]):
        return _skip(
            name,
            f"needs at least {threshold['min_shots']} shots for a per-(round, site) mean to mean "
            f"anything, got {raw.shape[0]}",
            used=threshold,
        )
    low, high = float(threshold["low"]), float(threshold["high"])
    means: list[float] = []
    inside: list[list[int]] = []
    for t in range(raw.shape[1]):
        for site in sites:
            column = raw[:, t, int(site)]
            usable = column[column <= 1]
            if usable.size == 0:
                continue
            mean = float(usable.mean())
            means.append(mean)
            if low <= mean <= high:
                inside.append([t, int(site)])
    if not means:
        return _skip(name, "every deterministic reading is a loss symbol", used=threshold)
    measured = {
        "n_means": len(means),
        "min_mean": float(np.min(means)),
        "max_mean": float(np.max(means)),
        "n_inside_band": len(inside),
        "inside_band": inside[:16],
    }
    if not inside:
        detail = (
            f"all {len(means)} deterministic (round, site) means sit outside [{low}, {high}] "
            f"(range {np.min(means):.4f} to {np.max(means):.4f})"
        )
        return Check(name, PASS, False, detail, measured, threshold)
    detail = (
        f"{len(inside)} of {len(means)} deterministic (round, site) means sit inside "
        f"[{low}, {high}], closest to even at {inside[0]}. A deterministic stabilizer reads near 0 "
        "or near 1; one that reads near 0.5 was never referenced to a frame"
    )
    return Check(name, FAIL, False, detail, measured, threshold)


# ---------------------------------------------------------------- V5
def _subsample_by_label(labels: np.ndarray, max_shots: int, seed: int) -> np.ndarray:
    n = labels.shape[0]
    if n <= max_shots:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for value in np.unique(labels):
        index = np.flatnonzero(labels == value)
        take = max(1, round(max_shots * index.size / n))
        take = min(take, index.size)
        keep.append(rng.choice(index, size=take, replace=False))
    return np.sort(np.concatenate(keep))


def check_time_mirror_control(spec: RecordSpec, threshold: dict[str, Any]) -> Check:
    """Decode the record, then decode it with the rounds reversed and the model unchanged.

    Advisory, never critical. It catches a record where decoding the reversed record is
    materially better. It does not catch a syndrome storage order fault: the detectors rebuilt
    from reversed stored rounds are not the reverse of the detectors built from the correct ones,
    and on a repetition memory the two decode within noise of each other. Those faults are caught
    by the round profile, the endpoint agreement and the state profile.
    """
    name = "time_mirror_control"
    if spec.dem is None or spec.labels is None:
        missing = [
            field for field, value in (("dem", spec.dem), ("labels", spec.labels)) if value is None
        ]
        return _skip(name, f"needs {' and '.join(missing)}; not in this record", used=threshold)
    if not controls.is_round_symmetric(spec.site_valid):
        return _skip(
            name,
            "the set of cells holding detectors changes when the round axis is reversed, so a "
            "mirrored record would put events where the model has no detector",
            used=threshold,
        )
    from qb_compiler.record.dem import decode_records

    labels = np.asarray(spec.labels, dtype=np.uint8)
    index = _subsample_by_label(labels, int(threshold["max_shots"]), int(threshold["seed"]))
    dets = np.asarray(spec.detectors, dtype=np.uint8)[index]
    used_labels = labels[index]
    real_pred = decode_records(spec.dem, spec.detector_matrix(dets))
    mirror_pred = decode_records(spec.dem, spec.detector_matrix(controls.time_mirror(dets)))
    real_fail = (real_pred != used_labels).astype(np.float64)
    mirror_fail = (mirror_pred != used_labels).astype(np.float64)
    real_ler = float(real_fail.mean())
    mirror_ler = float(mirror_fail.mean())
    n_used = int(index.size)
    n_real_failures = int(real_fail.sum())
    n_mirror_failures = int(mirror_fail.sum())
    paired = real_fail - mirror_fail
    difference = float(paired.mean())
    paired_stderr = float(paired.std(ddof=1) / np.sqrt(n_used)) if n_used > 1 else float("nan")
    measured = {
        "logical_error_rate": real_ler,
        "mirrored_logical_error_rate": mirror_ler,
        "difference": difference,
        "paired_standard_error": paired_stderr,
        "n_used": n_used,
        "n_available": int(labels.shape[0]),
        "n_failures": n_real_failures,
        "n_mirrored_failures": n_mirror_failures,
    }
    minimum = int(threshold["min_failures"])
    if min(n_real_failures, n_mirror_failures) < minimum:
        return _skip(
            name,
            f"fewer than {minimum} failures in an arm, no one-sided comparison is possible "
            f"({n_real_failures} decoded and {n_mirror_failures} reversed, on {n_used} shots)",
            used=threshold,
        )
    if not np.isfinite(paired_stderr) or paired_stderr <= 0.0:
        return _skip(
            name,
            "the two decodes agree on every shot, so the paired difference has no spread to "
            "measure against",
            used=threshold,
        )
    sigma = float(threshold["n_sigma"])
    limit = sigma * paired_stderr
    measured["limit"] = limit
    if difference <= limit:
        note = (
            f"decodes at {real_ler:.4f} and at {mirror_ler:.4f} with the rounds reversed, on "
            f"{n_used} shots. The paired difference is {difference:+.5f} against a standard error "
            f"of {paired_stderr:.5f}, inside the {sigma:g} sigma limit, so reversing does not "
            "materially beat the record as given"
        )
        return Check(name, PASS, False, note, measured, threshold)
    detail = (
        f"decodes at {real_ler:.4f} but at {mirror_ler:.4f} with the rounds REVERSED and the model "
        f"unchanged, on {n_used} shots. The paired difference is {difference:+.5f}, "
        f"{difference / paired_stderr:.1f} standard errors above zero and past the {sigma:g} sigma "
        "limit, so decoding the reversed record is materially better than decoding it as given"
    )
    return Check(name, FAIL, False, detail, measured, threshold)


# ---------------------------------------------------------------- V6
def check_state_profile(spec: RecordSpec, threshold: dict[str, Any]) -> Check:
    """Stored readout rate per round, split by the prepared logical state. Advisory."""
    name = "state_profile"
    if spec.raw_syndromes is None or spec.logical_state is None:
        missing = [
            field
            for field, value in (
                ("raw_syndromes", spec.raw_syndromes),
                ("logical_state", spec.logical_state),
            )
            if value is None
        ]
        return _skip(name, f"needs {' and '.join(missing)}; not in this record", used=threshold)
    raw = np.asarray(spec.raw_syndromes, dtype=np.float64)
    states = np.asarray(spec.logical_state).ravel()
    if raw.shape[1] < 2:
        return _skip(name, f"needs at least 2 stored rounds, got {raw.shape[1]}", used=threshold)
    per_state: dict[str, list[float]] = {}
    for value in np.unique(states):
        selected = raw[states == value]
        per_state[str(int(value))] = [float(selected[:, t].mean()) for t in range(raw.shape[1])]
    excited = per_state.get("1")
    measured: dict[str, Any] = {"rate_per_round_by_state": per_state}
    if excited is None:
        detail = (
            "no shots prepared in the excited state. Chains prepared in the ground state carry no "
            "relaxation signal, so this check has nothing to read on this record"
        )
        return Check(name, SKIP, False, detail, measured, threshold)
    steps = np.diff(np.asarray(excited))
    min_step = float(steps.min())
    measured["excited_steps"] = [float(s) for s in steps]
    measured["min_step"] = min_step
    measured["n_excited"] = int((states == 1).sum())
    if min_step >= float(threshold["min_step"]):
        detail = (
            f"on chains prepared in the excited state the stored readout rate climbs every round, "
            f"smallest step {min_step:+.5f} (rate {excited[0]:.4f} to {excited[-1]:.4f})"
        )
        return Check(name, PASS, False, detail, measured, threshold)
    detail = (
        f"on chains prepared in the excited state the stored readout rate falls by {-min_step:.5f} "
        f"at round {int(np.argmin(steps)) + 1} (rate {excited[0]:.4f} to {excited[-1]:.4f}, limit "
        f"{threshold['min_step']:+g}). Relaxation accumulates, so this rate should climb; a record "
        "whose stored rounds run backwards shows it falling"
    )
    return Check(name, FAIL, False, detail, measured, threshold)


# ---------------------------------------------------------------- V7
def check_dem_fingerprint(spec: RecordSpec, threshold: dict[str, Any]) -> Check:
    """Structural fingerprint of the error model, against an expectation if one was given."""
    name = "dem_fingerprint"
    if spec.dem is None:
        return _skip(name, "needs dem; not in this record", used=threshold)
    measured = fingerprint(spec.dem)
    expected = threshold.get("expected")
    if not expected:
        detail = (
            f"{measured['n_mechanisms']} mechanisms, {measured['n_boundary']} of them boundary; "
            "recorded, with no expectation to check it against"
        )
        return Check(name, PASS, False, detail, measured, threshold)
    ok, note = fingerprint_matches(measured, dict(expected))
    if ok:
        detail = f"{measured['n_mechanisms']} mechanisms, {measured['n_boundary']} boundary: {note}"
        return Check(name, PASS, True, detail, measured, threshold)
    detail = (
        f"the error model does not match the expected one ({note}). A decoder run against the "
        "wrong model produces numbers that look ordinary"
    )
    return Check(name, FAIL, True, detail, measured, threshold)


# ---------------------------------------------------------------- V8
def _gf2_solve(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve ``matrix @ x = target`` over GF(2), by Gaussian reduction."""
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rhs = np.asarray(target, dtype=np.uint8).copy() % 2
    n_rows, n_cols = work.shape
    pivots: list[int] = []
    row = 0
    for col in range(n_cols):
        candidates = np.flatnonzero(work[row:, col])
        if candidates.size == 0:
            continue
        pivot = row + int(candidates[0])
        work[[row, pivot]] = work[[pivot, row]]
        rhs[[row, pivot]] = rhs[[pivot, row]]
        for other in range(n_rows):
            if other != row and work[other, col]:
                work[other] ^= work[row]
                rhs[other] ^= rhs[row]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break
    solution = np.zeros(n_cols, dtype=np.uint8)
    for position, col in enumerate(pivots):
        solution[col] = rhs[position]
    return solution


def check_label_reconstruction(spec: RecordSpec, threshold: dict[str, Any]) -> Check:
    """Can the labels be rebuilt as a fixed parity of the final data readout?"""
    name = "label_reconstruction"
    if spec.final_data is None or spec.labels is None:
        missing = [
            field
            for field, value in (("final_data", spec.final_data), ("labels", spec.labels))
            if value is None
        ]
        return _skip(name, f"needs {' and '.join(missing)}; not in this record", used=threshold)
    final = np.asarray(spec.final_data, dtype=np.uint8)
    labels = np.asarray(spec.labels, dtype=np.uint8)
    loss = _flat_loss(spec)
    clean = np.ones(final.shape[0], dtype=bool) if loss is None else ~loss.astype(bool).any(axis=1)
    if clean.sum() < 2:
        return _skip(
            name, f"needs at least 2 shots with no loss, got {int(clean.sum())}", used=threshold
        )
    observable = _gf2_solve(final[clean], labels[clean])
    agreement = float((((final[clean] @ observable) % 2) == labels[clean]).mean())
    measured = {
        "agreement": agreement,
        "n_shots_used": int(clean.sum()),
        "observable_support": np.flatnonzero(observable).tolist(),
    }
    if agreement >= float(threshold["min_agreement"]):
        detail = (
            f"the labels are a fixed parity of {int(observable.sum())} data qubits on "
            f"{agreement:.4f} of the {int(clean.sum())} shots with no loss"
        )
        return Check(name, PASS, False, detail, measured, threshold)
    detail = (
        f"no fixed parity of the final data readout reproduces the labels: best agreement "
        f"{agreement:.4f} on {int(clean.sum())} shots with no loss, below "
        f"{threshold['min_agreement']}. Either the labels describe a different observable than the "
        "one the final readout carries, or the readout and the labels are not aligned shot for shot"
    )
    return Check(name, FAIL, False, detail, measured, threshold)


CHECKS = (
    ("round_profile", check_round_profile),
    ("endpoint_agreement", check_endpoint_agreement),
    ("event_density", check_event_density),
    ("type_consistency", check_type_consistency),
    ("time_mirror_control", check_time_mirror_control),
    ("state_profile", check_state_profile),
    ("dem_fingerprint", check_dem_fingerprint),
    ("label_reconstruction", check_label_reconstruction),
)


def validate(
    spec: RecordSpec,
    decoder: str = "mwpm",
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> ValidationReport:
    """Run every check that can run on ``spec`` and report each one.

    Parameters
    ----------
    spec :
        The record. Checks that need a field the record does not carry are skipped with the field
        named, not assumed away.
    decoder :
        Which decoder the checks that decode should use. ``"mwpm"`` is the only one implemented.
    thresholds :
        Per-check overrides merged over :data:`DEFAULT_THRESHOLDS`.

    Returns
    -------
    A :class:`~qb_compiler.record.types.ValidationReport`. ``passed`` is every critical check
    passing: ``event_density`` always, ``round_profile`` when the record declares the first layer
    premise, and ``dem_fingerprint`` when an expected fingerprint was supplied. Every other check
    is advisory and is reported without voting.

    Examples
    --------
    >>> from qb_compiler.record import validate                      # doctest: +SKIP
    >>> report = validate(spec)                                      # doctest: +SKIP
    >>> report.passed                                                # doctest: +SKIP
    True
    """
    if decoder not in SUPPORTED_DECODERS:
        raise ValueError(f"decoder must be one of {SUPPORTED_DECODERS}, got {decoder!r}")
    merged = _merge_thresholds(thresholds)
    results = tuple(function(spec, merged[name]) for name, function in CHECKS)
    passed = all(c.status == PASS for c in results if c.critical)
    report = ValidationReport(
        passed=passed,
        checks=results,
        n_shots=spec.n_shots,
        n_rounds=spec.n_rounds,
        n_sites=spec.n_sites,
        decoder=decoder,
        meta={
            "n_detectors": spec.n_detectors,
            "record_meta": dict(spec.meta),
            "n_run": int(sum(1 for c in results if c.status != SKIP)),
            "n_skipped": int(sum(1 for c in results if c.status == SKIP)),
        },
    )
    from qb_compiler.record import _sdk_hooks

    _sdk_hooks.on_validation_report(report)
    return report


__all__ = [
    "CHECKS",
    "DEFAULT_THRESHOLDS",
    "SUPPORTED_DECODERS",
    "check_dem_fingerprint",
    "check_endpoint_agreement",
    "check_event_density",
    "check_label_reconstruction",
    "check_round_profile",
    "check_state_profile",
    "check_time_mirror_control",
    "check_type_consistency",
    "validate",
]
