#!/usr/bin/env python3
"""Example 18: Checking how a QEC record was built.

A decoder result is a statement about a record. If the record was assembled wrong the result is
wrong by a factor and nothing downstream can tell, because the detectors still build and the
decoder still runs. This script makes a repetition code memory record in software, runs the eight
construction checks on it, then reads the same shots with their rounds backwards and watches the
checks find it.

Needs the record extra: pip install 'qb-compiler[record]'
"""

from __future__ import annotations

import sys

import numpy as np

try:
    from qb_compiler.record import FIRST_LAYER_PREMISE, RecordSpec, validate
    from qb_compiler.record.dem import build_repetition_dem, decode_records, fingerprint
    from qb_compiler.record.dem.repetition import expected_repetition_fingerprint
except ImportError as exc:
    print(f"skipped: needs the record extra (pip install 'qb-compiler[record]'): {exc}")
    sys.exit(0)


def simulate(d=5, rounds=5, shots=8000, p_data=0.04, p_meas=0.04, seed=0):
    """Walk a repetition code memory forward in time and keep what the hardware would store."""
    rng = np.random.default_rng(seed)
    state = np.zeros((shots, d), dtype=np.uint8)
    stored = np.zeros((shots, rounds, d - 1), dtype=np.uint8)
    for t in range(rounds):
        state ^= (rng.random((shots, d)) < p_data).astype(np.uint8)
        parity = state[:, :-1] ^ state[:, 1:]
        stored[:, t] = parity ^ (rng.random((shots, d - 1)) < p_meas).astype(np.uint8)
    state ^= (rng.random((shots, d)) < p_data).astype(np.uint8)
    return {
        "stored": stored,
        "final_parity": (state[:, :-1] ^ state[:, 1:]).astype(np.uint8),
        "labels": state[:, 0].astype(np.uint8),
        "prepared": (rng.random(shots) < 0.5).astype(np.uint8),
        "d": d,
        "rounds": rounds,
    }


def build(run, reverse_rounds):
    """Rebuild detectors from the stored blocks, the way a loader does."""
    stored = run["stored"][:, ::-1, :] if reverse_rounds else run["stored"]
    stored = np.ascontiguousarray(stored)
    shots, rounds, sites = stored.shape
    detectors = np.empty((shots, rounds + 1, sites), dtype=np.uint8)
    detectors[:, 0] = stored[:, 0]
    detectors[:, 1:rounds] = stored[:, 1:] ^ stored[:, :-1]
    detectors[:, rounds] = run["final_parity"] ^ stored[:, rounds - 1]
    return RecordSpec(
        detectors=detectors,
        labels=run["labels"],
        raw_syndromes=stored,
        final_parity=run["final_parity"],
        logical_state=run["prepared"],
        dem=build_repetition_dem(run["d"], run["rounds"], p_data=0.04, p_meas=0.04),
        # The first round of this run differences against the prepared state, so it is quiet. The
        # round profile check reads that premise off the record rather than assuming it, because
        # it is false on platforms whose preparation and final readout are the noisy part.
        meta={FIRST_LAYER_PREMISE: {"source": "the generator above, first round against reset"}},
    )


def show(title, report):
    print(f"=== {title} ===")
    print(f"passed: {report.passed}")
    for item in report.checks:
        marker = " " if item.status == "PASS" else "*"
        role = "critical" if item.critical else "advisory"
        print(f" {marker} {item.status:<5} {item.name:<22} {role}")
    print()


run = simulate()
spec = build(run, reverse_rounds=False)
served = decode_records(spec.dem, spec.detector_matrix())

print("=== The record ===")
print(f"shots        {spec.n_shots}")
print(f"rounds       {spec.n_rounds}")
print(f"sites        {spec.n_sites}")
print(f"detectors    {spec.n_detectors}")
print(f"mechanisms   {spec.dem.n_mechanisms}")
print(f"logical error rate {(served != np.asarray(spec.labels)).mean():.4f}")
print()

report = validate(spec)
show("As recorded", report)

profile = report.check("round_profile")
print("round profile")
print("  rate per round:", [round(float(x), 4) for x in profile.measured["rate_per_round"]])
print(f"  first round {profile.measured['first_round_ratio']:.2f} times the steady median")
print(f"  limit {profile.threshold['first_round_max_ratio']}")
print()

mirrored = validate(build(run, reverse_rounds=True))
show("With the rounds stored backwards", mirrored)
for name in ("round_profile", "endpoint_agreement", "state_profile"):
    print(name)
    print(" ", mirrored.check(name).detail)
    print()

print("=== Model fingerprint against the closed form ===")
for d, rounds in ((3, 3), (5, 5), (7, 4), (11, 11)):
    measured = fingerprint(build_repetition_dem(d, rounds))
    closed = expected_repetition_fingerprint(d, rounds)
    print(
        f"d{d} r{rounds}: {measured['n_mechanisms']} mechanisms, "
        f"{measured['n_boundary']} boundary, matches {measured == closed}"
    )
print()

print("A record that passes these checks is correctly built. That is the whole claim. It is not")
print("complete, and none of these checks asks whether a decoder is any good.")
print()
print("The event density decides the verdict, the round profile too on a record that declares a")
print("quiet first layer, and the model fingerprint when an expected one is supplied. The rest")
print("report without voting, because several of them have records on which they have no power.")
