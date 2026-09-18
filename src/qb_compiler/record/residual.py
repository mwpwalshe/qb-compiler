# SPDX-License-Identifier: Apache-2.0
"""The residual metric: what a feature block adds over what the decoder already reported.

The question it answers is narrow and worth stating precisely. Take the decoder's own per-shot
output, everything soft it produced, plus the coarsest counts of the record. Fit a model that
predicts whether the decoder failed on that shot, and measure its held-out log loss. Now add a
block of features computed from the record, refit, and measure again. The drop, in bits per shot,
is the residual.

It is a comparison against the decoder's own summary, not a decoder benchmark. A large residual
does not say a better decoder exists, and a small one does not say the record holds nothing. It
says how much of the decoder's failures a logistic model can anticipate from these features that
it could not anticipate from the decoder's own report.

The floor
---------
A held-out difference of zero is not what an uninformative feature block scores. Extra columns
move the number around, folds are finite, and the direction is not symmetric. So the same
measurement runs on records whose structure has been destroyed and whose counts have not:

* **geometry nulls** shuffle detector events within each round, preserving per-round counts
  exactly, and recompute the features through the caller's own feature function. This needs the
  feature function; when only a matrix is supplied there is nothing to recompute and the geometry
  null is reported as unavailable.
* **shot-permutation nulls** reorder the feature rows against the shots, preserving both marginal
  distributions and destroying the pairing.

``null_floor_p95`` is the 95th percentile over whichever nulls ran, and ``above_floor`` is the one
comparison this module makes. The report carries no reading of what any number means.

Usage::

    from qb_compiler.record import residual

    report = residual(
        spec,
        decoder_output={"prediction": pred, "weight": w, "gap": gap},
        features=lambda s: (per_round_counts(s), ["events_round_0", "events_round_1"]),
    )
    print(report.residual_bits, report.null_floor_p95, report.above_floor)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from qb_compiler.record import controls
from qb_compiler.record.types import RecordSpec, ResidualReport

N_FOLDS = 5
MAX_ITER = 5000
PROB_CLIP = 1e-15

FeatureBlock = np.ndarray | tuple[np.ndarray, Sequence[str]]
FeatureSource = FeatureBlock | Callable[[RecordSpec], FeatureBlock]


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the residual metric needs scikit-learn. Install with: "
            "pip install 'qb-compiler[record]'"
        ) from None


def _split_block(block: FeatureBlock, fallback_prefix: str) -> tuple[np.ndarray, list[str]]:
    if isinstance(block, tuple):
        matrix, names = block
        matrix = np.asarray(matrix, dtype=np.float64)
        names = list(names)
    else:
        matrix = np.asarray(block, dtype=np.float64)
        names = []
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(
            f"a feature block must be 1-D or 2-D, got {matrix.ndim}-D with shape {matrix.shape}"
        )
    if not names:
        names = [f"{fallback_prefix}_{i}" for i in range(matrix.shape[1])]
    if len(names) != matrix.shape[1]:
        raise ValueError(
            f"feature block has {matrix.shape[1]} column(s) but {len(names)} name(s) were given"
        )
    return matrix, names


def _baseline_block(
    spec: RecordSpec, decoder_output: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    """Everything the decoder reported, plus the coarsest counts of the record."""
    if "prediction" not in decoder_output:
        raise ValueError(
            "decoder_output must carry a 'prediction' entry: the residual is measured against "
            "whether the decoder failed, which needs its answer. Known keys: "
            f"{sorted(decoder_output)}"
        )
    columns: list[np.ndarray] = []
    names: list[str] = []
    for key in sorted(decoder_output):
        column = np.asarray(decoder_output[key], dtype=np.float64)
        if column.ndim == 1:
            column = column[:, None]
        if column.ndim != 2 or column.shape[0] != spec.n_shots:
            raise ValueError(
                f"decoder_output[{key!r}] must have {spec.n_shots} rows, got shape {column.shape}"
            )
        columns.append(column)
        names.extend(
            [key] if column.shape[1] == 1 else [f"{key}_{i}" for i in range(column.shape[1])]
        )

    dets = np.asarray(spec.detectors, dtype=np.float64)
    valid = spec.site_valid
    columns.append(dets[:, valid].sum(axis=1)[:, None])
    names.append("total_events")
    per_round = np.zeros((spec.n_shots, spec.n_rounds), dtype=np.float64)
    for t in range(spec.n_rounds):
        cols = np.flatnonzero(valid[t])
        per_round[:, t] = dets[:, t, cols].sum(axis=1) if cols.size else 0.0
    columns.append(per_round)
    names.extend([f"events_round_{t}" for t in range(spec.n_rounds)])

    if spec.loss is not None:
        loss = np.asarray(spec.loss)
        columns.append(loss.reshape(loss.shape[0], -1).astype(np.float64).sum(axis=1)[:, None])
        names.append("total_loss")

    return np.column_stack(columns), names


def _folds(
    target: np.ndarray, groups: np.ndarray | None, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    dummy = np.zeros((target.shape[0], 1))
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(dummy, target))
    grouped = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(grouped.split(dummy, target, groups=groups))


def _out_of_fold(
    design: np.ndarray,
    target: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    regularization_c: float,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = np.zeros(target.shape[0], dtype=np.float64)
    for train, test in folds:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=regularization_c, max_iter=MAX_ITER),
        )
        model.fit(design[train], target[train])
        out[test] = model.predict_proba(design[test])[:, 1]
    return out


def _bits(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, PROB_CLIP, 1.0 - PROB_CLIP)
    truth = target.astype(np.float64)
    return np.asarray(-(truth * np.log2(clipped) + (1.0 - truth) * np.log2(1.0 - clipped)))


def _auc(target: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(target, probability))


def residual(
    spec: RecordSpec,
    decoder_output: dict[str, np.ndarray],
    features: FeatureSource,
    *,
    holdout: str = "group",
    C: float = 0.1,  # noqa: N803 - the name scikit-learn gives this parameter
    n_nulls: int = 20,
    seed: int = 0,
) -> ResidualReport:
    """Held-out bits per shot that ``features`` adds over the decoder's own output.

    Parameters
    ----------
    spec :
        The record. ``labels`` is required: the target is whether the decoder's prediction differs
        from them.
    decoder_output :
        Per-shot output from the decoder that ran, keyed by name. ``prediction`` is required;
        everything else soft the decoder produced (solution weight, gap, posterior) belongs here
        too, because the residual is measured *against* it. Every entry lands in the baseline.
    features :
        The block being measured: an ``(n_shots, k)`` array, a ``(array, names)`` pair, or a
        callable taking a :class:`~qb_compiler.record.types.RecordSpec` and returning either. Pass
        the callable to get geometry nulls, which need the features recomputed on a shuffled
        record; a bare matrix cannot be recomputed and the geometry null is reported unavailable.
    holdout :
        ``"group"`` uses ``spec.groups`` when the record carries them, falling back to stratified
        folds over shots with a warning in the report. ``"shot"`` always uses stratified folds over
        shots. Shot-level folds put neighbouring shots on both sides of the split, so anything that
        drifts within a run can leak across it.
    C :
        Inverse regularization strength of the logistic model, after standardization.
    n_nulls :
        How many draws of each null to run.
    seed :
        Fixes the folds and every null draw.

    Returns
    -------
    A :class:`~qb_compiler.record.types.ResidualReport`.

    Raises
    ------
    ValueError
        When the record has no labels, when ``decoder_output`` has no prediction, or when the
        decoder failed on no shot or on every shot, which leaves nothing to model.
    """
    _require_sklearn()
    if spec.labels is None:
        raise ValueError(
            "the record has no labels, so whether the decoder failed on a shot is not defined"
        )
    if holdout not in ("group", "shot"):
        raise ValueError(f"holdout must be 'group' or 'shot', got {holdout!r}")
    if n_nulls < 0:
        raise ValueError(f"n_nulls must be non-negative, got {n_nulls}")

    if "prediction" not in decoder_output:
        raise ValueError(
            "decoder_output must carry a 'prediction' entry: the residual is measured against "
            "whether the decoder failed, which needs its answer. Known keys: "
            f"{sorted(decoder_output)}"
        )
    labels = np.asarray(spec.labels, dtype=np.uint8)
    prediction = np.asarray(decoder_output["prediction"], dtype=np.uint8).ravel()
    if prediction.shape[0] != spec.n_shots:
        raise ValueError(
            f"decoder_output['prediction'] must have {spec.n_shots} rows, got {prediction.shape[0]}"
        )
    target = (prediction != labels).astype(np.int64)
    n_failures = int(target.sum())
    if n_failures == 0 or n_failures == spec.n_shots:
        raise ValueError(
            f"the decoder failed on {n_failures} of {spec.n_shots} shots. With one class there is "
            "nothing to model, so no residual is defined on this record"
        )

    warnings: list[str] = []
    baseline, baseline_names = _baseline_block(spec, decoder_output)

    callable_features = callable(features)
    block: FeatureBlock = (
        features(spec) if callable_features else features  # type: ignore[operator,assignment]
    )
    feature_matrix, feature_names = _split_block(block, "feature")
    if feature_matrix.shape[0] != spec.n_shots:
        raise ValueError(
            f"the feature block must have {spec.n_shots} rows, got {feature_matrix.shape[0]}"
        )

    groups = None
    holdout_used = "shot"
    if holdout == "group":
        if spec.groups is None:
            warnings.append(
                "holdout='group' was asked for but the record carries no groups, so folds are "
                "stratified over shots. Shots next to each other in a run end up on opposite "
                "sides of the split, so anything drifting within the run can leak across it"
            )
        else:
            groups = np.asarray(spec.groups).ravel()
            holdout_used = "group"
    if holdout_used == "shot":
        warnings.append(
            "folds are stratified over shots, not over groups. Anything that drifts within a run "
            "is present on both sides of every split"
        )

    n_folds = N_FOLDS
    if groups is not None:
        n_unique = int(np.unique(groups).size)
        if n_unique < 2:
            groups = None
            holdout_used = "shot"
            warnings.append(
                f"the record carries {n_unique} distinct group(s), which cannot be split, so "
                "folds are stratified over shots instead"
            )
        else:
            n_folds = min(N_FOLDS, n_unique)
    n_folds = min(n_folds, n_failures, spec.n_shots - n_failures)
    if n_folds < 2:
        raise ValueError(
            f"only {n_failures} failure(s) in {spec.n_shots} shots, which is too few to hold any "
            "of them out. No residual is defined on this record"
        )
    if n_folds != N_FOLDS:
        warnings.append(f"{n_folds} folds rather than {N_FOLDS}, limited by the data")

    folds = _folds(target, groups, n_folds, seed)
    proba_baseline = _out_of_fold(baseline, target, folds, C)
    proba_augmented = _out_of_fold(np.column_stack([baseline, feature_matrix]), target, folds, C)
    bits_baseline = float(_bits(target, proba_baseline).mean())
    bits_augmented = float(_bits(target, proba_augmented).mean())
    residual_bits = bits_baseline - bits_augmented

    null_values: list[float] = []
    nulls: dict[str, Any] = {}

    permutation_values: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(n_nulls):
        order = controls.shot_permutation(spec.n_shots, rng)
        proba = _out_of_fold(np.column_stack([baseline, feature_matrix[order]]), target, folds, C)
        permutation_values.append(bits_baseline - float(_bits(target, proba).mean()))
    nulls["shot_permutation"] = {
        "available": n_nulls > 0,
        "n": len(permutation_values),
        "mean": float(np.mean(permutation_values)) if permutation_values else float("nan"),
        "p95": float(np.percentile(permutation_values, 95)) if permutation_values else float("nan"),
        "values": permutation_values,
        "reason": None if n_nulls > 0 else "n_nulls is 0",
    }
    null_values.extend(permutation_values)

    if not callable_features:
        nulls["geometry"] = {
            "available": False,
            "n": 0,
            "mean": float("nan"),
            "p95": float("nan"),
            "values": [],
            "reason": (
                "features were supplied as a matrix, so they cannot be recomputed on a shuffled "
                "record. Pass a callable to get this null"
            ),
        }
        warnings.append(
            "the geometry null did not run: it needs the features recomputed on a record whose "
            "within-round geometry has been shuffled, which needs a callable rather than a matrix"
        )
    else:
        geometry_values: list[float] = []
        geometry_rng = np.random.default_rng(seed + 1)
        valid = spec.site_valid
        for _ in range(n_nulls):
            shuffled = spec.replace_detectors(
                controls.geometry_shuffle(spec.detectors, geometry_rng, site_valid=valid)
            )
            null_block, _ = _split_block(
                features(shuffled),  # type: ignore[operator]
                "feature",
            )
            proba = _out_of_fold(np.column_stack([baseline, null_block]), target, folds, C)
            geometry_values.append(bits_baseline - float(_bits(target, proba).mean()))
        nulls["geometry"] = {
            "available": n_nulls > 0,
            "n": len(geometry_values),
            "mean": float(np.mean(geometry_values)) if geometry_values else float("nan"),
            "p95": float(np.percentile(geometry_values, 95)) if geometry_values else float("nan"),
            "values": geometry_values,
            "reason": None if n_nulls > 0 else "n_nulls is 0",
        }
        null_values.extend(geometry_values)

    if null_values:
        floor_mean = float(np.mean(null_values))
        floor_p95 = float(np.percentile(null_values, 95))
        above_floor = bool(residual_bits > floor_p95)
    else:
        floor_mean = float("nan")
        floor_p95 = float("nan")
        above_floor = False
        warnings.append("no null ran, so there is no floor to compare the residual against")

    report = ResidualReport(
        residual_bits=residual_bits,
        null_floor_mean=floor_mean,
        null_floor_p95=floor_p95,
        above_floor=above_floor,
        auc_baseline=_auc(target, proba_baseline),
        auc_augmented=_auc(target, proba_augmented),
        loss_baseline_bits=bits_baseline,
        loss_augmented_bits=bits_augmented,
        feature_names=tuple(feature_names),
        baseline_names=tuple(baseline_names),
        holdout=holdout_used,
        n_folds=n_folds,
        n_shots=spec.n_shots,
        n_failures=n_failures,
        seed=seed,
        regularization_c=float(C),
        nulls=nulls,
        warnings=tuple(warnings),
        meta={
            "holdout_requested": holdout,
            "n_nulls_requested": int(n_nulls),
            "n_groups": int(np.unique(groups).size) if groups is not None else 0,
            "record_meta": dict(spec.meta),
        },
    )
    from qb_compiler.record import _sdk_hooks

    _sdk_hooks.on_residual_report(report)
    return report


__all__ = ["MAX_ITER", "N_FOLDS", "PROB_CLIP", "residual"]
