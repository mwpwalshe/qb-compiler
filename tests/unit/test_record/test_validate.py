# SPDX-License-Identifier: Apache-2.0
"""The eight construction checks, on synthetic records built right and built wrong.

Every failing case here is a record that decodes without error and reports a plausible looking
number, which is the whole reason the checks exist.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from qb_compiler.record import validate
from qb_compiler.record.dem import build_repetition_dem
from qb_compiler.record.dem.repetition import expected_repetition_fingerprint
from qb_compiler.record.types import FAIL, FIRST_LAYER_PREMISE, PASS, SKIP, RecordDem, RecordSpec
from qb_compiler.record.validate import (
    DEFAULT_THRESHOLDS,
    check_endpoint_agreement,
    check_event_density,
    check_label_reconstruction,
    check_round_profile,
    check_time_mirror_control,
    check_type_consistency,
)

from .conftest import make_spec, simulate_repetition

pytest.importorskip("pymatching")


class TestRoundProfile:
    def test_a_correctly_ordered_record_passes(self, good_spec):
        result = validate(good_spec).check("round_profile")
        assert result.status == PASS
        assert result.measured["first_round_ratio"] < 0.9

    def test_a_mirrored_record_is_caught(self, mirrored_spec):
        result = validate(mirrored_spec).check("round_profile")
        assert result.status == FAIL
        assert result.measured["first_round_ratio"] > 1.0
        assert "reset" in result.detail

    def test_round_profile_is_critical_either_way(self, good_spec, mirrored_spec):
        assert validate(good_spec).check("round_profile").critical
        assert validate(mirrored_spec).check("round_profile").critical

    def test_two_rounds_cannot_be_profiled(self):
        spec = RecordSpec(detectors=np.zeros((10, 2, 3), dtype=np.uint8))
        result = check_round_profile(spec, DEFAULT_THRESHOLDS["round_profile"])
        assert result.status == SKIP
        assert result.critical

    def test_a_dead_middle_leaves_nothing_to_compare_against(self):
        dets = np.zeros((50, 4, 3), dtype=np.uint8)
        dets[:, 0] = 1
        spec = RecordSpec(detectors=dets)
        result = check_round_profile(spec, DEFAULT_THRESHOLDS["round_profile"])
        assert result.status == SKIP
        assert result.critical

    def test_a_loud_last_round_is_caught(self):
        rng = np.random.default_rng(3)
        dets = (rng.random((400, 5, 4)) < 0.1).astype(np.uint8)
        dets[:, 0] = (rng.random((400, 4)) < 0.05).astype(np.uint8)
        dets[:, -1] = (rng.random((400, 4)) < 0.4).astype(np.uint8)
        spec = RecordSpec(detectors=dets, meta={FIRST_LAYER_PREMISE: True})
        result = check_round_profile(spec, DEFAULT_THRESHOLDS["round_profile"])
        assert result.status == FAIL
        assert result.measured["last_round_ratio"] > 1.5


class TestFirstLayerPremise:
    """The rule has power only where the first round is quiet, so the record declares that."""

    def _loud_ends(self, seed=5):
        """A record loud at both ends and flat between, built correctly on such a platform."""
        rng = np.random.default_rng(seed)
        dets = (rng.random((4000, 14, 2)) < 0.18).astype(np.uint8)
        dets[:, 0] = (rng.random((4000, 2)) < 0.24).astype(np.uint8)
        dets[:, -1] = (rng.random((4000, 2)) < 0.25).astype(np.uint8)
        return dets

    def test_a_record_that_declares_nothing_is_not_judged(self):
        result = check_round_profile(
            RecordSpec(detectors=self._loud_ends()), DEFAULT_THRESHOLDS["round_profile"]
        )
        assert result.status == SKIP
        assert not result.critical
        assert "no declared first layer premise" in result.detail

    def test_the_layer_profile_is_still_reported(self):
        result = check_round_profile(
            RecordSpec(detectors=self._loud_ends()), DEFAULT_THRESHOLDS["round_profile"]
        )
        assert len(result.measured["rate_per_round"]) == 14
        assert result.measured["first_round_ratio"] > 1.2
        assert result.measured["steady_median"] > 0.0

    def test_it_drops_out_of_the_critical_set_for_that_record(self, synthetic_run):
        """A record with no premise is not failed on this check, and does not fail the report."""
        spec = make_spec(synthetic_run, declare_premise=False)
        report = validate(spec)
        assert report.check("round_profile").status == SKIP
        assert not report.check("round_profile").critical
        assert report.passed

    def test_a_declared_premise_behaves_as_before(self, synthetic_run):
        declared = validate(make_spec(synthetic_run)).check("round_profile")
        assert declared.status == PASS
        assert declared.critical
        assert declared.threshold["first_round_max_ratio"] == 0.9

    def test_true_is_shorthand_for_the_shipped_limits(self, synthetic_run):
        detectors = make_spec(synthetic_run).detectors
        result = check_round_profile(
            RecordSpec(detectors=detectors, meta={FIRST_LAYER_PREMISE: True}),
            DEFAULT_THRESHOLDS["round_profile"],
        )
        assert result.status == PASS
        assert result.threshold["first_round_max_ratio"] == 0.9

    def test_a_caller_can_declare_its_own_ratios(self):
        """A platform measured by the caller, with the limits that platform was measured at."""
        spec = RecordSpec(
            detectors=self._loud_ends(),
            meta={
                FIRST_LAYER_PREMISE: {
                    "first_round_max_ratio": 1.4,
                    "last_round_max_ratio": 1.6,
                    "source": "measured on this caller's own platform",
                }
            },
        )
        result = check_round_profile(spec, DEFAULT_THRESHOLDS["round_profile"])
        assert result.status == PASS
        assert result.critical
        assert result.threshold["first_round_max_ratio"] == 1.4
        assert result.measured["premise"]["source"] == "measured on this caller's own platform"

    def test_a_declared_ratio_the_record_misses_still_fails(self):
        spec = RecordSpec(
            detectors=self._loud_ends(),
            meta={FIRST_LAYER_PREMISE: {"first_round_max_ratio": 1.1}},
        )
        result = check_round_profile(spec, DEFAULT_THRESHOLDS["round_profile"])
        assert result.status == FAIL
        assert result.critical
        assert "1.1x limit this record declares" in result.detail

    def test_something_that_is_not_a_premise_is_named(self):
        spec = RecordSpec(detectors=self._loud_ends(), meta={FIRST_LAYER_PREMISE: "yes"})
        result = check_round_profile(spec, DEFAULT_THRESHOLDS["round_profile"])
        assert result.status == SKIP
        assert not result.critical
        assert "declares nothing" in result.detail

    def test_the_shipped_limits_did_not_move(self):
        assert DEFAULT_THRESHOLDS["round_profile"] == {
            "first_round_max_ratio": 0.9,
            "last_round_max_ratio": 1.5,
        }


class TestEndpointAgreement:
    def _spec(self, mismatch_last, mismatch_first, n_shots=4000, sites=4, seed=0):
        """A record whose two endpoint mismatch rates are set by construction."""
        rng = np.random.default_rng(seed)
        final = (rng.random((n_shots, sites)) < 0.5).astype(np.uint8)
        stored = np.zeros((n_shots, 3, sites), dtype=np.uint8)
        stored[:, 0] = final ^ (rng.random((n_shots, sites)) < mismatch_first).astype(np.uint8)
        stored[:, 1] = final
        stored[:, 2] = final ^ (rng.random((n_shots, sites)) < mismatch_last).astype(np.uint8)
        return RecordSpec(
            detectors=np.zeros((n_shots, 4, sites), dtype=np.uint8),
            raw_syndromes=stored,
            final_parity=final,
        )

    def test_a_correctly_ordered_record_passes(self, good_spec):
        result = validate(good_spec).check("endpoint_agreement")
        assert result.status == PASS
        assert result.measured["ratio"] < 1.0

    def test_last_round_first_storage_is_caught(self, mirrored_spec):
        result = validate(mirrored_spec).check("endpoint_agreement")
        assert result.status == FAIL
        assert result.measured["ratio"] > 2.0
        assert "reversed" in result.detail

    def test_it_is_advisory(self, good_spec, mirrored_spec):
        assert not validate(good_spec).check("endpoint_agreement").critical
        assert not validate(mirrored_spec).check("endpoint_agreement").critical

    def test_v2_passes_equal_rates_and_fails_doubled(self):
        """Equal rates are a correct ground-state record, not a fault. Doubled is the fault."""
        equal = check_endpoint_agreement(
            self._spec(0.10, 0.10), DEFAULT_THRESHOLDS["endpoint_agreement"]
        )
        assert equal.status == PASS
        assert equal.measured["no_power"] is True
        assert "no power on this record" in equal.detail
        assert 0.8 <= equal.measured["ratio"] <= 1.25

        doubled = check_endpoint_agreement(
            self._spec(0.30, 0.10), DEFAULT_THRESHOLDS["endpoint_agreement"]
        )
        assert doubled.status == FAIL
        assert doubled.measured["ratio"] > 2.0
        assert doubled.measured["no_power"] is False

    def test_a_last_round_that_agrees_far_better_still_passes(self):
        """The check is one sided: only the mirrored signature is a finding."""
        result = check_endpoint_agreement(
            self._spec(0.02, 0.20), DEFAULT_THRESHOLDS["endpoint_agreement"]
        )
        assert result.status == PASS
        assert result.measured["ratio"] < 0.5
        assert "no power on this record" not in result.detail

    def test_it_says_which_input_it_wanted(self):
        spec = RecordSpec(detectors=np.zeros((10, 4, 3), dtype=np.uint8))
        result = validate(spec).check("endpoint_agreement")
        assert result.status == SKIP
        assert "raw_syndromes" in result.detail and "final_parity" in result.detail


class TestEventDensity:
    def test_a_real_looking_rate_passes(self, good_spec):
        result = validate(good_spec).check("event_density")
        assert result.status == PASS
        assert 0.005 < result.measured["event_rate"] < 0.35

    def test_coin_flip_detectors_are_caught(self):
        rng = np.random.default_rng(1)
        dets = (rng.random((500, 5, 6)) < 0.5).astype(np.uint8)
        spec = RecordSpec(detectors=dets)
        result = check_event_density(spec, DEFAULT_THRESHOLDS["event_density"])
        assert result.status == FAIL
        assert result.measured["event_rate"] > 0.35
        assert "frame" in result.detail

    def test_dead_detectors_are_caught(self):
        dets = np.zeros((500, 5, 6), dtype=np.uint8)
        spec = RecordSpec(detectors=dets)
        result = check_event_density(spec, DEFAULT_THRESHOLDS["event_density"])
        assert result.status == FAIL
        assert result.measured["event_rate"] == 0.0

    def test_the_band_can_be_moved(self):
        rng = np.random.default_rng(2)
        dets = (rng.random((500, 5, 6)) < 0.45).astype(np.uint8)
        spec = RecordSpec(detectors=dets)
        assert validate(spec).check("event_density").status == FAIL
        loosened = validate(spec, thresholds={"event_density": {"high": 0.6}})
        assert loosened.check("event_density").status == PASS
        assert loosened.check("event_density").threshold["high"] == 0.6

    def test_an_unknown_check_name_is_refused(self, good_spec):
        with pytest.raises(ValueError, match="unknown check name"):
            validate(good_spec, thresholds={"no_such_check": {"low": 1}})


class TestTypeConsistency:
    def _spec(self, means, n_shots=400, seed=0):
        rng = np.random.default_rng(seed)
        raw = np.zeros((n_shots, 3, len(means)), dtype=np.uint8)
        for site, mean in enumerate(means):
            raw[:, :, site] = (rng.random((n_shots, 3)) < mean).astype(np.uint8)
        return RecordSpec(
            detectors=np.zeros((n_shots, 4, len(means)), dtype=np.uint8),
            raw_syndromes=raw,
            deterministic_sites=np.arange(len(means)),
        )

    def test_deterministic_readouts_pass(self):
        spec = self._spec([0.02, 0.97, 0.05, 0.93])
        result = check_type_consistency(spec, DEFAULT_THRESHOLDS["type_consistency"])
        assert result.status == PASS
        assert result.measured["n_inside_band"] == 0

    def test_unframed_readouts_are_caught(self):
        spec = self._spec([0.5, 0.51, 0.49, 0.5])
        result = check_type_consistency(spec, DEFAULT_THRESHOLDS["type_consistency"])
        assert result.status == FAIL
        assert result.measured["n_inside_band"] == 12
        assert "frame" in result.detail

    def test_loss_symbols_are_left_out_of_the_mean(self):
        spec = self._spec([0.02, 0.97, 0.05, 0.93])
        raw = np.asarray(spec.raw_syndromes).copy()
        raw[:200, :, 0] = 2
        with_loss = RecordSpec(
            detectors=spec.detectors,
            raw_syndromes=raw,
            deterministic_sites=spec.deterministic_sites,
        )
        result = check_type_consistency(with_loss, DEFAULT_THRESHOLDS["type_consistency"])
        assert result.status == PASS

    def test_it_says_which_input_it_wanted(self, good_spec):
        result = validate(good_spec).check("type_consistency")
        assert result.status == SKIP
        assert "deterministic_sites" in result.detail


class TestTimeMirrorControl:
    def test_it_runs_quickly_on_five_thousand_shots(self):
        run = simulate_repetition(n_shots=5000, p_data=0.04, p_meas=0.04, seed=7)
        spec = make_spec(run, p_data=0.04, p_meas=0.04)
        start = time.perf_counter()
        result = check_time_mirror_control(spec, DEFAULT_THRESHOLDS["time_mirror_control"])
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"took {elapsed:.2f}s"
        assert result.status in (PASS, FAIL)
        assert result.measured["n_used"] == 5000

    def test_v5_advisory_and_skips_below_30_failures(self):
        """Never critical, and it declines to run where it cannot discriminate."""
        run = simulate_repetition(n_shots=4000, p_data=0.02, p_meas=0.02, seed=0)
        spec = make_spec(run)
        result = check_time_mirror_control(spec, DEFAULT_THRESHOLDS["time_mirror_control"])
        assert result.status == SKIP
        assert not result.critical
        assert "fewer than 30 failures in an arm" in result.detail
        assert result.measured == {}

        busy = simulate_repetition(n_shots=8000, p_data=0.04, p_meas=0.04, seed=0)
        ran = check_time_mirror_control(
            make_spec(busy, p_data=0.04, p_meas=0.04),
            DEFAULT_THRESHOLDS["time_mirror_control"],
        )
        assert ran.status == PASS
        assert not ran.critical
        assert ran.measured["n_failures"] >= 30
        assert ran.measured["n_mirrored_failures"] >= 30

    def test_v5_passes_a_time_symmetric_record_on_every_seed(self):
        """The regression this rule exists for.

        A repetition memory is time symmetric to within noise, so the old one-sigma one-sided
        form called a fault on a correct record about one seed in six. At this regime, which is
        the one that produced 13 failures against 9 in the notebook, the check must never report
        a fault; here it declines to run at all, for want of failures, which is the right answer.
        """
        verdicts = []
        for seed in range(10):
            run = simulate_repetition(n_shots=4000, p_data=0.02, p_meas=0.02, seed=seed)
            result = check_time_mirror_control(
                make_spec(run), DEFAULT_THRESHOLDS["time_mirror_control"]
            )
            verdicts.append(result.status)
        assert FAIL not in verdicts, f"reported a fault on a correct record: {verdicts}"
        assert verdicts.count(SKIP) == 10

    def test_it_compares_the_paired_difference_against_its_own_standard_error(self):
        run = simulate_repetition(n_shots=8000, p_data=0.04, p_meas=0.04, seed=3)
        result = check_time_mirror_control(
            make_spec(run, p_data=0.04, p_meas=0.04),
            DEFAULT_THRESHOLDS["time_mirror_control"],
        )
        measured = result.measured
        assert measured["difference"] == pytest.approx(
            measured["logical_error_rate"] - measured["mirrored_logical_error_rate"], abs=1e-12
        )
        assert measured["paired_standard_error"] > 0
        assert measured["limit"] == pytest.approx(3.0 * measured["paired_standard_error"])
        assert (result.status == PASS) == (measured["difference"] <= measured["limit"])

    def test_a_record_the_reversed_decode_beats_is_caught(self):
        """A run whose noise grows, against a model that knows it, handed over reversed."""
        from qb_compiler.record.controls import time_mirror

        d, rounds, shots = 3, 6, 12000
        rng = np.random.default_rng(1)
        p_data = 0.03
        p_meas = np.linspace(0.02, 0.30, rounds)
        state = np.zeros((shots, d), dtype=np.uint8)
        stored = np.zeros((shots, rounds, d - 1), dtype=np.uint8)
        for t_index in range(rounds):
            state ^= (rng.random((shots, d)) < p_data).astype(np.uint8)
            parity = state[:, :-1] ^ state[:, 1:]
            stored[:, t_index] = parity ^ (rng.random((shots, d - 1)) < p_meas[t_index]).astype(
                np.uint8
            )
        state ^= (rng.random((shots, d)) < p_data).astype(np.uint8)
        final_parity = (state[:, :-1] ^ state[:, 1:]).astype(np.uint8)
        dets = np.empty((shots, rounds + 1, d - 1), dtype=np.uint8)
        dets[:, 0] = stored[:, 0]
        dets[:, 1:rounds] = stored[:, 1:] ^ stored[:, :-1]
        dets[:, rounds] = final_parity ^ stored[:, rounds - 1]
        labels = state[:, 0].astype(np.uint8)

        flat = build_repetition_dem(d, rounds, p_data=p_data, p_meas=float(p_meas[0]))
        weights = np.asarray(flat.weights).copy()
        position = d * (rounds + 1)
        for t_index in range(rounds):
            for _site in range(d - 1):
                weights[position] = -np.log(p_meas[t_index] / (1 - p_meas[t_index]))
                position += 1
        matched = RecordDem(
            check_matrix=flat.check_matrix, observable=flat.observable, weights=weights
        )

        as_given = check_time_mirror_control(
            RecordSpec(detectors=dets, labels=labels, dem=matched),
            DEFAULT_THRESHOLDS["time_mirror_control"],
        )
        reversed_axis = check_time_mirror_control(
            RecordSpec(detectors=time_mirror(dets), labels=labels, dem=matched),
            DEFAULT_THRESHOLDS["time_mirror_control"],
        )
        assert as_given.status == PASS
        assert reversed_axis.status == FAIL
        assert "materially better" in reversed_axis.detail
        assert reversed_axis.measured["difference"] > reversed_axis.measured["limit"]

    def test_it_subsamples_and_says_so(self):
        run = simulate_repetition(n_shots=8000, p_data=0.04, p_meas=0.04, seed=8)
        spec = make_spec(run, p_data=0.04, p_meas=0.04)
        result = check_time_mirror_control(
            spec, {**DEFAULT_THRESHOLDS["time_mirror_control"], "max_shots": 4000}
        )
        assert result.measured["n_used"] <= 4020
        assert result.measured["n_available"] == 8000

    def test_the_subsample_is_seeded(self):
        run = simulate_repetition(n_shots=8000, p_data=0.04, p_meas=0.04, seed=9)
        spec = make_spec(run, p_data=0.04, p_meas=0.04)
        thresholds = {**DEFAULT_THRESHOLDS["time_mirror_control"], "max_shots": 4000}
        first = check_time_mirror_control(spec, thresholds)
        second = check_time_mirror_control(spec, thresholds)
        assert first.measured == second.measured

    def test_it_is_never_critical(self, good_spec, mirrored_spec):
        assert not validate(good_spec).check("time_mirror_control").critical
        assert not validate(mirrored_spec).check("time_mirror_control").critical

    def test_without_a_model_it_is_skipped_and_not_critical(self, synthetic_run):
        spec = make_spec(synthetic_run, with_dem=False)
        result = validate(spec).check("time_mirror_control")
        assert result.status == SKIP
        assert not result.critical
        assert "dem" in result.detail

    def test_a_ragged_validity_pattern_is_refused(self):
        index = np.array([[0, -1, -1], [1, 2, 3], [4, 5, 6]], dtype=np.int32)
        spec = RecordSpec(
            detectors=np.zeros((20, 3, 3), dtype=np.uint8),
            detector_index=index,
            labels=np.zeros(20, dtype=np.uint8),
            dem=build_repetition_dem(3, 2),
        )
        result = check_time_mirror_control(spec, DEFAULT_THRESHOLDS["time_mirror_control"])
        assert result.status == SKIP
        assert "reversed" in result.detail


class TestStateProfile:
    def test_a_prepared_one_record_climbs(self, good_spec):
        result = validate(good_spec).check("state_profile")
        assert result.status == PASS
        assert result.measured["min_step"] > 0
        assert result.measured["n_excited"] > 0

    def test_a_mirrored_record_falls(self, mirrored_spec):
        result = validate(mirrored_spec).check("state_profile")
        assert result.status == FAIL
        assert result.measured["min_step"] < -0.005
        assert "Relaxation accumulates" in result.detail

    def test_it_is_advisory(self, good_spec, mirrored_spec):
        assert not validate(good_spec).check("state_profile").critical
        assert not validate(mirrored_spec).check("state_profile").critical

    def test_ground_state_only_carries_no_signal(self, synthetic_run):
        spec = make_spec(synthetic_run)
        ground_only = RecordSpec(
            detectors=spec.detectors,
            raw_syndromes=spec.raw_syndromes,
            logical_state=np.zeros(spec.n_shots, dtype=np.uint8),
        )
        result = validate(ground_only).check("state_profile")
        assert result.status == SKIP
        assert "no relaxation signal" in result.detail or "carry no relaxation" in result.detail
        assert "0" in result.measured["rate_per_round_by_state"]


class TestDemFingerprint:
    @pytest.mark.parametrize(("d", "rounds"), [(3, 3), (5, 5), (7, 4), (11, 11)])
    def test_the_counts_are_the_arithmetic(self, d, rounds):
        from qb_compiler.record.dem import fingerprint

        measured = fingerprint(build_repetition_dem(d, rounds))
        assert measured["n_mechanisms"] == d * (rounds + 1) + (d - 1) * rounds
        assert measured["n_boundary"] == 2 * (rounds + 1)
        assert measured == expected_repetition_fingerprint(d, rounds)

    def test_without_an_expectation_it_records_and_is_not_critical(self, good_spec):
        result = validate(good_spec).check("dem_fingerprint")
        assert result.status == PASS
        assert not result.critical
        assert result.measured["n_mechanisms"] == 50

    def test_with_an_expectation_it_is_critical(self, good_spec, synthetic_run):
        expected = expected_repetition_fingerprint(synthetic_run.d, synthetic_run.rounds)
        result = validate(good_spec, thresholds={"dem_fingerprint": {"expected": expected}}).check(
            "dem_fingerprint"
        )
        assert result.status == PASS
        assert result.critical

    def test_the_wrong_model_is_caught(self, good_spec):
        wrong = expected_repetition_fingerprint(7, 7)
        report = validate(good_spec, thresholds={"dem_fingerprint": {"expected": wrong}})
        result = report.check("dem_fingerprint")
        assert result.status == FAIL
        assert result.critical
        assert not report.passed
        assert "n_mechanisms" in result.detail

    def test_a_partial_expectation_compares_only_what_it_names(self, good_spec):
        result = validate(
            good_spec, thresholds={"dem_fingerprint": {"expected": {"n_mechanisms": 50}}}
        ).check("dem_fingerprint")
        assert result.status == PASS


class TestLabelReconstruction:
    def test_a_parity_of_the_final_readout_is_found(self):
        rng = np.random.default_rng(4)
        final = (rng.random((300, 6)) < 0.5).astype(np.uint8)
        labels = (final[:, 0] ^ final[:, 3]).astype(np.uint8)
        spec = RecordSpec(
            detectors=np.zeros((300, 4, 3), dtype=np.uint8), labels=labels, final_data=final
        )
        result = check_label_reconstruction(spec, DEFAULT_THRESHOLDS["label_reconstruction"])
        assert result.status == PASS
        assert result.measured["agreement"] == 1.0

    def test_labels_from_a_different_observable_are_caught(self):
        rng = np.random.default_rng(5)
        final = (rng.random((300, 6)) < 0.5).astype(np.uint8)
        labels = (rng.random(300) < 0.5).astype(np.uint8)
        spec = RecordSpec(
            detectors=np.zeros((300, 4, 3), dtype=np.uint8), labels=labels, final_data=final
        )
        result = check_label_reconstruction(spec, DEFAULT_THRESHOLDS["label_reconstruction"])
        assert result.status == FAIL
        assert result.measured["agreement"] < 0.999
        assert "observable" in result.detail

    def test_shots_with_loss_are_left_out_of_the_solve(self):
        rng = np.random.default_rng(6)
        final = (rng.random((300, 6)) < 0.5).astype(np.uint8)
        labels = (final[:, 1] ^ final[:, 4]).astype(np.uint8)
        loss = np.zeros((300, 6), dtype=np.uint8)
        loss[150:] = 1
        labels[150:] = rng.integers(0, 2, 150).astype(np.uint8)
        spec = RecordSpec(
            detectors=np.zeros((300, 4, 3), dtype=np.uint8),
            labels=labels,
            final_data=final,
            loss=loss,
        )
        result = check_label_reconstruction(spec, DEFAULT_THRESHOLDS["label_reconstruction"])
        assert result.status == PASS
        assert result.measured["n_shots_used"] == 150


class TestReport:
    def test_every_check_is_run_or_skipped_with_a_reason(self, good_spec):
        report = validate(good_spec)
        assert len(report.checks) == 8
        for item in report.checks:
            assert item.status in (PASS, FAIL, SKIP)
            assert item.detail
        assert report.meta["n_run"] + report.meta["n_skipped"] == 8

    def test_passed_is_the_critical_checks(self, good_spec, mirrored_spec):
        good = validate(good_spec)
        assert good.passed
        assert all(c.status == PASS for c in good.checks if c.critical)
        bad = validate(mirrored_spec)
        assert not bad.passed
        assert any(c.status == FAIL and c.critical for c in bad.checks)

    def test_an_unknown_decoder_is_refused(self, good_spec):
        with pytest.raises(ValueError, match="decoder must be one of"):
            validate(good_spec, decoder="belief")

    def test_a_named_check_that_did_not_run_raises(self, good_spec):
        with pytest.raises(KeyError):
            validate(good_spec).check("no_such_check")

    def test_the_report_reads_as_text(self, good_spec):
        text = str(validate(good_spec))
        assert "record validation: PASS" in text
        assert "round_profile" in text
