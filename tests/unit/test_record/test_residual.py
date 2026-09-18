# SPDX-License-Identifier: Apache-2.0
"""The residual metric, its null floor, and what it recovers when the answer is known.

Two properties matter and both are tested here. Features that carry nothing must land at or below
the floor, because a held-out difference of zero is not what noise scores. Features that carry a
known amount must recover roughly that amount, because a metric that cannot find a planted signal
cannot be trusted when it reports one.
"""

from __future__ import annotations

import numpy as np
import pytest

from qb_compiler.record.residual import residual
from qb_compiler.record.types import RecordSpec

pytest.importorskip("sklearn")


def _binary_entropy_bits(p: np.ndarray | float) -> np.ndarray:
    prob = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(prob * np.log2(prob) + (1 - prob) * np.log2(1 - prob))


def _planted_record(
    n_shots: int = 4000, strength: float = 1.6, seed: int = 0
) -> tuple[RecordSpec, dict[str, np.ndarray], np.ndarray, float]:
    """A record whose failures depend on one hidden column by a known number of bits."""
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=n_shots)
    probability = 1.0 / (1.0 + np.exp(-(strength * hidden - 1.0)))
    failed = (rng.random(n_shots) < probability).astype(np.uint8)
    # The decoder's answer and the truth are each coin flips; only their disagreement carries the
    # planted signal, so nothing in the baseline can read it off either one alone.
    labels = (rng.random(n_shots) < 0.5).astype(np.uint8)
    spec = RecordSpec(
        detectors=(rng.random((n_shots, 4, 5)) < 0.08).astype(np.uint8),
        labels=labels,
        meta={"loader": "synthetic-planted"},
    )
    decoder_output = {
        "prediction": (labels ^ failed).astype(np.uint8),
        "weight": rng.normal(size=n_shots),
    }
    planted_bits = float(
        _binary_entropy_bits(failed.mean()).item() - _binary_entropy_bits(probability).mean()
    )
    return spec, decoder_output, hidden[:, None], planted_bits


def _noise_record(
    n_shots: int = 700, seed: int = 0, n_features: int = 2
) -> tuple[RecordSpec, dict[str, np.ndarray], np.ndarray]:
    """A record whose failures depend on nothing the features can see."""
    rng = np.random.default_rng(seed)
    failed = (rng.random(n_shots) < 0.2).astype(np.uint8)
    labels = (rng.random(n_shots) < 0.5).astype(np.uint8)
    spec = RecordSpec(
        detectors=(rng.random((n_shots, 4, 4)) < 0.08).astype(np.uint8),
        labels=labels,
    )
    decoder_output = {
        "prediction": (labels ^ failed).astype(np.uint8),
        "weight": rng.normal(size=n_shots),
    }
    return spec, decoder_output, rng.normal(size=(n_shots, n_features))


class TestPlantedSignal:
    def test_a_planted_signal_is_recovered(self):
        spec, output, features, planted = _planted_record()
        report = residual(spec, output, (features, ["hidden"]), holdout="shot", n_nulls=5)
        assert planted > 0.05
        assert abs(report.residual_bits - planted) / planted < 0.20, (
            f"recovered {report.residual_bits:.4f} against a planted {planted:.4f}"
        )
        assert report.above_floor
        assert report.auc_augmented > report.auc_baseline

    def test_the_report_carries_its_own_settings(self):
        spec, output, features, _ = _planted_record(n_shots=800)
        report = residual(spec, output, features, holdout="shot", n_nulls=3, seed=11, C=0.5)
        assert report.seed == 11
        assert report.regularization_c == 0.5
        assert report.n_shots == 800
        assert report.holdout == "shot"
        assert report.n_folds == 5
        assert report.feature_names == ("feature_0",)
        assert "total_events" in report.baseline_names
        assert "prediction" in report.baseline_names
        assert "weight" in report.baseline_names
        assert sum(1 for n in report.baseline_names if n.startswith("events_round_")) == 4


class TestNullFloor:
    def test_noise_features_land_at_the_floor_on_nineteen_of_twenty_seeds(self):
        # 20 null draws, so the 95th percentile is estimated from 20 values rather than from a
        # handful where it collapses onto the maximum and the floor reads high.
        within = 0
        for seed in range(20):
            spec, output, features = _noise_record(n_shots=500, seed=seed)
            report = residual(spec, output, features, holdout="shot", n_nulls=20, seed=seed)
            within += int(not report.above_floor)
        assert within >= 19, f"only {within} of 20 seeds landed at the floor"

    def test_a_matrix_cannot_produce_the_geometry_null(self):
        spec, output, features = _noise_record(seed=3)
        report = residual(spec, output, features, holdout="shot", n_nulls=4)
        assert report.nulls["geometry"]["available"] is False
        assert "callable" in report.nulls["geometry"]["reason"]
        assert report.nulls["shot_permutation"]["n"] == 4
        assert any("geometry null did not run" in w for w in report.warnings)

    def test_a_callable_produces_the_geometry_null(self):
        spec, output, _ = _noise_record(seed=4)

        def counts(record: RecordSpec) -> tuple[np.ndarray, list[str]]:
            dets = np.asarray(record.detectors, dtype=np.float64)
            block = np.column_stack([dets[:, :, 0].sum(axis=1), dets[:, 0, :].sum(axis=1)])
            return block, ["site0_over_rounds", "round0_over_sites"]

        report = residual(spec, output, counts, holdout="shot", n_nulls=4)
        assert report.nulls["geometry"]["available"] is True
        assert report.nulls["geometry"]["n"] == 4
        assert report.nulls["shot_permutation"]["n"] == 4
        assert report.feature_names == ("site0_over_rounds", "round0_over_sites")

    def test_the_floor_pools_every_null_that_ran(self):
        spec, output, features = _noise_record(seed=5)
        report = residual(spec, output, features, holdout="shot", n_nulls=6)
        values = report.nulls["shot_permutation"]["values"]
        assert report.null_floor_mean == pytest.approx(float(np.mean(values)))
        assert report.null_floor_p95 == pytest.approx(float(np.percentile(values, 95)))

    def test_no_nulls_means_no_floor(self):
        spec, output, features = _noise_record(seed=6)
        report = residual(spec, output, features, holdout="shot", n_nulls=0)
        assert not report.above_floor
        assert np.isnan(report.null_floor_p95)
        assert any("no null ran" in w for w in report.warnings)


class TestHoldout:
    def test_groups_are_used_when_the_record_carries_them(self):
        spec, output, features, _ = _planted_record(n_shots=900, seed=2)
        rng = np.random.default_rng(0)
        grouped = RecordSpec(
            detectors=spec.detectors,
            labels=spec.labels,
            groups=rng.integers(0, 8, spec.n_shots),
        )
        report = residual(grouped, output, features, n_nulls=2)
        assert report.holdout == "group"
        assert report.meta["n_groups"] == 8
        assert not any("no groups" in w for w in report.warnings)

    def test_without_groups_it_falls_back_and_warns(self):
        spec, output, features, _ = _planted_record(n_shots=800, seed=3)
        report = residual(spec, output, features, holdout="group", n_nulls=2)
        assert report.holdout == "shot"
        assert any("carries no groups" in w for w in report.warnings)
        assert any("leak" in w for w in report.warnings)

    def test_shot_folds_always_warn_about_drift(self):
        spec, output, features, _ = _planted_record(n_shots=800, seed=4)
        report = residual(spec, output, features, holdout="shot", n_nulls=2)
        assert any("drifts within a run" in w for w in report.warnings)

    def test_one_group_cannot_be_split(self):
        spec, output, features, _ = _planted_record(n_shots=600, seed=5)
        single = RecordSpec(
            detectors=spec.detectors,
            labels=spec.labels,
            groups=np.zeros(spec.n_shots, dtype=int),
        )
        report = residual(single, output, features, n_nulls=1)
        assert report.holdout == "shot"
        assert any("cannot be split" in w for w in report.warnings)

    def test_an_unknown_holdout_is_refused(self):
        spec, output, features, _ = _planted_record(n_shots=400)
        with pytest.raises(ValueError, match="holdout must be"):
            residual(spec, output, features, holdout="fold", n_nulls=0)


class TestRefusals:
    def test_a_record_with_no_labels_is_refused(self):
        spec, output, features, _ = _planted_record(n_shots=400)
        unlabelled = RecordSpec(detectors=spec.detectors)
        with pytest.raises(ValueError, match="no labels"):
            residual(unlabelled, output, features, n_nulls=0)

    def test_decoder_output_without_a_prediction_is_refused(self):
        spec, output, features, _ = _planted_record(n_shots=400)
        with pytest.raises(ValueError, match="must carry a 'prediction'"):
            residual(spec, {"weight": output["weight"]}, features, n_nulls=0)

    def test_one_class_is_refused(self):
        rng = np.random.default_rng(0)
        spec = RecordSpec(
            detectors=(rng.random((300, 3, 3)) < 0.1).astype(np.uint8),
            labels=np.zeros(300, dtype=np.uint8),
        )
        output = {"prediction": np.zeros(300, dtype=np.uint8)}
        with pytest.raises(ValueError, match="one class"):
            residual(spec, output, rng.normal(size=(300, 2)), n_nulls=0)

    def test_too_few_failures_to_hold_out_is_refused(self):
        rng = np.random.default_rng(0)
        prediction = np.zeros(300, dtype=np.uint8)
        labels = np.zeros(300, dtype=np.uint8)
        labels[:1] = 1
        spec = RecordSpec(detectors=(rng.random((300, 3, 3)) < 0.1).astype(np.uint8), labels=labels)
        with pytest.raises(ValueError, match="too few"):
            residual(spec, {"prediction": prediction}, rng.normal(size=(300, 2)), n_nulls=0)

    def test_a_mismatched_feature_block_is_refused(self):
        spec, output, _, _ = _planted_record(n_shots=400)
        with pytest.raises(ValueError, match="must have 400 rows"):
            residual(spec, output, np.zeros((17, 2)), n_nulls=0)

    def test_mismatched_feature_names_are_refused(self):
        spec, output, features, _ = _planted_record(n_shots=400)
        with pytest.raises(ValueError, match="name"):
            residual(spec, output, (features, ["a", "b"]), n_nulls=0)

    def test_a_negative_null_count_is_refused(self):
        spec, output, features, _ = _planted_record(n_shots=400)
        with pytest.raises(ValueError, match="n_nulls"):
            residual(spec, output, features, n_nulls=-1)

    def test_a_three_dimensional_feature_block_is_refused(self):
        spec, output, _, _ = _planted_record(n_shots=400)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            residual(spec, output, np.zeros((400, 2, 2)), n_nulls=0)


class TestLossInTheBaseline:
    def test_total_loss_joins_the_baseline_when_the_record_has_a_loss_array(self):
        spec, output, features, _ = _planted_record(n_shots=600, seed=7)
        rng = np.random.default_rng(1)
        with_loss = RecordSpec(
            detectors=spec.detectors,
            labels=spec.labels,
            loss=(rng.random((600, 9)) < 0.1).astype(np.uint8),
        )
        report = residual(with_loss, output, features, holdout="shot", n_nulls=1)
        assert "total_loss" in report.baseline_names

    def test_no_loss_array_means_no_loss_column(self):
        spec, output, features, _ = _planted_record(n_shots=600, seed=8)
        report = residual(spec, output, features, holdout="shot", n_nulls=1)
        assert "total_loss" not in report.baseline_names
