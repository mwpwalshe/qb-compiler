#!/usr/bin/env python3
"""Example 19: What a feature block adds over what the decoder already reported.

A decoder produces more than an answer: a weight, often a gap, sometimes a posterior. Given all of
that plus the coarsest counts of the record, how much more of the decoder's own failures can be
anticipated from some other quantity computed from the same record? The answer comes back in bits
per shot, against a floor built from records whose structure has been destroyed and whose counts
have not.

It is a comparison with one decoder's own summary on one record. It is not a decoder benchmark.

Needs the record extra: pip install 'qb-compiler[record]'
"""

from __future__ import annotations

import sys

import numpy as np

try:
    from qb_compiler.record import RecordSpec, residual
    from qb_compiler.record.dem import build_repetition_dem, decode_records
except ImportError as exc:
    print(f"skipped: needs the record extra (pip install 'qb-compiler[record]'): {exc}")
    sys.exit(0)

D, ROUNDS, SHOTS, P = 3, 5, 8000, 0.06

rng = np.random.default_rng(3)
state = np.zeros((SHOTS, D), dtype=np.uint8)
stored = np.zeros((SHOTS, ROUNDS, D - 1), dtype=np.uint8)
for t in range(ROUNDS):
    state ^= (rng.random((SHOTS, D)) < P).astype(np.uint8)
    parity = state[:, :-1] ^ state[:, 1:]
    stored[:, t] = parity ^ (rng.random((SHOTS, D - 1)) < P).astype(np.uint8)
state ^= (rng.random((SHOTS, D)) < P).astype(np.uint8)
final_parity = (state[:, :-1] ^ state[:, 1:]).astype(np.uint8)

detectors = np.empty((SHOTS, ROUNDS + 1, D - 1), dtype=np.uint8)
detectors[:, 0] = stored[:, 0]
detectors[:, 1:ROUNDS] = stored[:, 1:] ^ stored[:, :-1]
detectors[:, ROUNDS] = final_parity ^ stored[:, ROUNDS - 1]
labels = state[:, 0].astype(np.uint8)

dem = build_repetition_dem(D, ROUNDS, p_data=P, p_meas=P)
plain = RecordSpec(detectors=detectors, labels=labels, dem=dem)
first_pass = decode_records(dem, plain.detector_matrix())

print("=== The record and the decoder ===")
print(f"shots              {SHOTS}")
print(f"logical error rate {(first_pass != labels).mean():.4f}")
print(f"failures           {int((first_pass != labels).sum())}")
print()

# One site fires more often on the shots this decoder gets wrong. The decoder reads that site like
# any other, because its model says the site is ordinary.
tell = detectors.copy()
lit = (first_pass != labels) & (rng.random(SHOTS) < 0.5)
tell[lit, 2, 1] ^= 1

spec = RecordSpec(detectors=tell, labels=labels, dem=dem)
served = decode_records(dem, spec.detector_matrix())
print("=== With an association the model does not describe ===")
print(f"logical error rate {(served != labels).mean():.4f}")
print(f"failures           {int((served != labels).sum())}")
print()


def site_feature(record):
    """Events at one site in one round. A function of the record, so both nulls can run."""
    events = np.asarray(record.detectors, dtype=np.float64)
    return events[:, 2, 1][:, None], ["events_at_site_1_round_2"]


def unrelated(record):
    """Noise with no connection to the record at all."""
    generator = np.random.default_rng(7)
    return generator.normal(size=(record.n_shots, 1)), ["unrelated_noise"]


def show(title, report):
    print(f"=== {title} ===")
    print(f"residual          {report.residual_bits:+.5f} bits per shot")
    print(
        f"geometry floor    {report.nulls['geometry']['p95']:+.5f} "
        f"over {report.nulls['geometry']['n']} draws"
    )
    print(
        f"permutation floor {report.nulls['shot_permutation']['p95']:+.5f} "
        f"over {report.nulls['shot_permutation']['n']} draws"
    )
    print(f"pooled floor      {report.null_floor_p95:+.5f}")
    print(f"above floor       {report.above_floor}")
    print(f"area under curve  {report.auc_baseline:.4f} to {report.auc_augmented:.4f}")
    print()


carries = residual(spec, {"prediction": served}, site_feature, holdout="shot", n_nulls=20, seed=0)
nothing = residual(spec, {"prediction": served}, unrelated, holdout="shot", n_nulls=20, seed=0)
show("The site that fires on failures", carries)
show("Noise", nothing)

print("=== What it was measured against ===")
print(f"failures {carries.n_failures} of {carries.n_shots} shots")
print("baseline columns:")
for name in carries.baseline_names:
    print("  ", name)
print(f"hold out: {carries.holdout}, {carries.n_folds} folds, seed {carries.seed}")
for note in carries.warnings:
    print("note:", note)
print()

print("A residual above the floor means a feature block anticipates some of this decoder's")
print("failures that its own output did not. It does not say a better decoder exists. A residual")
print("at the floor does not say the record holds nothing.")
