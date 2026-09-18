# SPDX-License-Identifier: Apache-2.0
"""Types for record construction checks and the residual metric.

A *record* is what a QEC experiment leaves behind: detector events laid out in rounds and sites,
the labels a decoder is asked to predict, and whatever else the run happened to store. This module
holds the container (:class:`RecordSpec`), the error model a decoder runs on (:class:`RecordDem`),
and the two report types the checks and the metric return.

Detector layout
---------------
``RecordSpec.detectors`` is ``(n_shots, n_rounds, n_sites)``. Not every (round, site) cell is a
detector in every code: a surface-code memory in one basis has detectors on one stabilizer type in
the first and last rounds and on both in between. ``detector_index`` carries that, mapping each
cell to its position in the error model's detector numbering and holding ``-1`` where the cell is
not a detector. When it is ``None`` every cell is a detector, numbered row major, which is the
repetition-code case.

``detector_matrix()`` is the flat ``(n_shots, n_detectors)`` view in error-model order, which is
what a matching decoder wants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

VALIDATION_SCHEMA = "qb.record_validation.v1"
RESIDUAL_SCHEMA = "qb.record_residual.v1"

#: ``RecordSpec.meta`` key declaring that this platform's first round is quiet: that it
#: differences against a prepared value rather than against a previous round, and so fires below
#: the steady rate. It is a property of the platform, not of every record, so the round profile
#: check reads it and abstains on a record that does not declare it. The value is a mapping,
#: optionally carrying ``first_round_max_ratio``, ``last_round_max_ratio`` and a ``source`` saying
#: where those came from, or ``True`` to take the shipped limits. See
#: :func:`qb_compiler.record.validate.check_round_profile`.
FIRST_LAYER_PREMISE = "first_layer_premise"

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass(frozen=True)
class RecordDem:
    """An error model as a decoder consumes it: one column per error mechanism.

    Attributes
    ----------
    check_matrix :
        ``(n_detectors, n_mechanisms)`` uint8. Column ``j`` is the detector signature of
        mechanism ``j``. A column of weight one is a boundary mechanism.
    observable :
        ``(n_mechanisms,)`` uint8, 1 where the mechanism flips the logical observable.
    weights :
        ``(n_mechanisms,)`` float matching weights.
    edge_qubits :
        Optional ``(n_mechanisms,)`` int naming the physical qubit or measurement each mechanism
        belongs to. Used by the loss-map check; ``-1`` where a mechanism has no single owner.
    meta :
        Free-form provenance.
    """

    check_matrix: np.ndarray
    observable: np.ndarray
    weights: np.ndarray
    edge_qubits: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check = np.asarray(self.check_matrix)
        if check.ndim != 2:
            raise ValueError(
                f"check_matrix must be 2-D (n_detectors, n_mechanisms), got {check.ndim}-D"
            )
        n_mech = check.shape[1]
        for name, arr in (("observable", self.observable), ("weights", self.weights)):
            got = np.asarray(arr)
            if got.shape != (n_mech,):
                raise ValueError(
                    f"{name} must have shape ({n_mech},) to match check_matrix, got {got.shape}"
                )
        if self.edge_qubits is not None and np.asarray(self.edge_qubits).shape != (n_mech,):
            raise ValueError(
                f"edge_qubits must have shape ({n_mech},), got {np.asarray(self.edge_qubits).shape}"
            )

    @property
    def n_detectors(self) -> int:
        return int(np.asarray(self.check_matrix).shape[0])

    @property
    def n_mechanisms(self) -> int:
        return int(np.asarray(self.check_matrix).shape[1])


@dataclass(frozen=True)
class RecordSpec:
    """One experimental record, plus whatever of it the run stored.

    Only ``detectors`` is required. Every other field is what some checks need and others do not;
    a check that cannot run says so rather than guessing.

    Attributes
    ----------
    detectors :
        ``(n_shots, n_rounds, n_sites)`` uint8 detector events.
    detector_index :
        Optional ``(n_rounds, n_sites)`` int32 mapping each cell to its error-model detector
        number, ``-1`` where the cell holds no detector. ``None`` means every cell is a detector,
        numbered row major.
    labels :
        Optional ``(n_shots,)`` uint8 observable-flip truth. A decoder fails on a shot when its
        prediction differs from this.
    groups :
        Optional ``(n_shots,)`` grouping used for held-out splits, e.g. one entry per chain or per
        submitted circuit.
    loss :
        Optional loss symbols, ``(n_shots, n_rounds, n_sites)`` or ``(n_shots, n_measurements)``.
    raw_syndromes :
        Optional ``(n_shots, n_raw_rounds, n_sites)`` stored readouts *in loader order*, before
        differencing. Stored in the order the loader believes is chronological, so a loader that
        reverses the stored rounds returns them reversed here too.
    final_parity :
        Optional ``(n_shots, n_sites)`` parity of the final data readout.
    final_data :
        Optional ``(n_shots, n_data_qubits)`` raw final data readout.
    logical_state :
        Optional ``(n_shots,)`` prepared logical state.
    data0 :
        Optional ``(n_shots,)`` first data qubit of the final readout.
    deterministic_sites :
        Optional indices into the site axis whose raw readout is deterministic given the frame.
    site_coords :
        Optional ``(n_sites, k)`` site coordinates.
    dem :
        Optional error model.
    meta :
        Free-form provenance: loader, version, options. One key is read by a check rather than
        just reported: :data:`FIRST_LAYER_PREMISE`, which a loader that knows its platform sets.
    """

    detectors: np.ndarray
    detector_index: np.ndarray | None = None
    labels: np.ndarray | None = None
    groups: np.ndarray | None = None
    loss: np.ndarray | None = None
    raw_syndromes: np.ndarray | None = None
    final_parity: np.ndarray | None = None
    final_data: np.ndarray | None = None
    logical_state: np.ndarray | None = None
    data0: np.ndarray | None = None
    deterministic_sites: np.ndarray | None = None
    site_coords: np.ndarray | None = None
    dem: RecordDem | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dets = np.asarray(self.detectors)
        if dets.ndim != 3:
            raise ValueError(
                f"detectors must be 3-D (n_shots, n_rounds, n_sites), got {dets.ndim}-D "
                f"with shape {dets.shape}"
            )
        n_shots, n_rounds, n_sites = dets.shape
        if self.detector_index is not None:
            index = np.asarray(self.detector_index)
            if index.shape != (n_rounds, n_sites):
                raise ValueError(
                    f"detector_index must have shape ({n_rounds}, {n_sites}) to match detectors, "
                    f"got {index.shape}"
                )
        for name in ("labels", "groups", "logical_state", "data0"):
            arr = getattr(self, name)
            if arr is not None and np.asarray(arr).shape[0] != n_shots:
                raise ValueError(
                    f"{name} must have {n_shots} rows to match detectors, "
                    f"got {np.asarray(arr).shape[0]}"
                )

    @property
    def n_shots(self) -> int:
        return int(np.asarray(self.detectors).shape[0])

    @property
    def n_rounds(self) -> int:
        return int(np.asarray(self.detectors).shape[1])

    @property
    def n_sites(self) -> int:
        return int(np.asarray(self.detectors).shape[2])

    @property
    def site_valid(self) -> np.ndarray:
        """``(n_rounds, n_sites)`` bool: which cells hold a detector."""
        if self.detector_index is None:
            return np.ones((self.n_rounds, self.n_sites), dtype=bool)
        return np.asarray(self.detector_index) >= 0

    @property
    def n_detectors(self) -> int:
        return int(self.site_valid.sum())

    def detector_matrix(self, detectors: np.ndarray | None = None) -> np.ndarray:
        """``(n_shots, n_detectors)`` uint8 in error-model detector order.

        ``detectors`` flattens a different block through this record's layout, which is how a
        subset of shots or a rearranged copy gets to a decoder without building a record around
        it. It must have this record's rounds and sites; the shot count is free.
        """
        dets = np.asarray(self.detectors if detectors is None else detectors, dtype=np.uint8)
        if dets.ndim != 3 or dets.shape[1:] != (self.n_rounds, self.n_sites):
            raise ValueError(
                f"detectors must be (n_shots, {self.n_rounds}, {self.n_sites}), got {dets.shape}"
            )
        flat = dets.reshape(dets.shape[0], self.n_rounds * self.n_sites)
        if self.detector_index is None:
            return flat
        index = np.asarray(self.detector_index).ravel()
        cells = np.flatnonzero(index >= 0)
        order = cells[np.argsort(index[cells], kind="stable")]
        return np.ascontiguousarray(flat[:, order])

    def round_of_detector(self) -> np.ndarray:
        """``(n_detectors,)`` int: the round each error-model detector sits in."""
        rounds = np.repeat(np.arange(self.n_rounds), self.n_sites)
        if self.detector_index is None:
            return rounds
        index = np.asarray(self.detector_index).ravel()
        cells = np.flatnonzero(index >= 0)
        order = cells[np.argsort(index[cells], kind="stable")]
        return rounds[order]

    def replace_detectors(self, detectors: np.ndarray) -> RecordSpec:
        """A copy carrying different detector events and everything else unchanged."""
        return RecordSpec(
            detectors=np.asarray(detectors, dtype=np.uint8),
            detector_index=self.detector_index,
            labels=self.labels,
            groups=self.groups,
            loss=self.loss,
            raw_syndromes=self.raw_syndromes,
            final_parity=self.final_parity,
            final_data=self.final_data,
            logical_state=self.logical_state,
            data0=self.data0,
            deterministic_sites=self.deterministic_sites,
            site_coords=self.site_coords,
            dem=self.dem,
            meta=dict(self.meta),
        )

    def as_dict(self) -> dict[str, Any]:
        """Shapes and provenance, not the arrays.

        A record does not fit in JSON and should not be moved through it. Use
        :func:`qb_compiler.record.loaders.generic_npz.write_npz` to serialise the arrays.
        """
        present = [
            name
            for name in (
                "labels",
                "groups",
                "loss",
                "raw_syndromes",
                "final_parity",
                "final_data",
                "logical_state",
                "data0",
                "deterministic_sites",
                "site_coords",
                "dem",
            )
            if getattr(self, name) is not None
        ]
        return {
            "schema": "qb.record_spec.v1",
            "n_shots": self.n_shots,
            "n_rounds": self.n_rounds,
            "n_sites": self.n_sites,
            "n_detectors": self.n_detectors,
            "present": present,
            "meta": dict(self.meta),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """The :meth:`as_dict` summary as JSON. The arrays are not in it, by design."""
        return json.dumps(self.as_dict(), indent=indent, default=str)


@dataclass(frozen=True)
class Check:
    """One construction check, run or skipped.

    ``measured`` holds what was computed, ``threshold`` what it was compared against, so a reader
    can see why the status is what it is without re-running anything.
    """

    name: str
    status: str
    critical: bool
    detail: str
    measured: dict[str, Any] = field(default_factory=dict)
    threshold: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def skipped(self) -> bool:
        return self.status == SKIP

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "critical": self.critical,
            "detail": self.detail,
            "measured": self.measured,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Check:
        return cls(
            name=data["name"],
            status=data["status"],
            critical=bool(data["critical"]),
            detail=data["detail"],
            measured=dict(data.get("measured", {})),
            threshold=dict(data.get("threshold", {})),
        )

    def __str__(self) -> str:
        return f"{self.status:<4} {self.name}: {self.detail}"


@dataclass(frozen=True)
class ValidationReport:
    """The outcome of :func:`qb_compiler.record.validate.validate`.

    ``passed`` is every critical check passing. A critical check that could not run does not pass:
    a record whose round profile cannot be computed has not been checked, whatever else came back.
    """

    passed: bool
    checks: tuple[Check, ...]
    n_shots: int
    n_rounds: int
    n_sites: int
    decoder: str
    meta: dict[str, Any] = field(default_factory=dict)
    schema: str = VALIDATION_SCHEMA

    def check(self, name: str) -> Check:
        """One check by name. Raises ``KeyError`` when the name is not in the report."""
        for item in self.checks:
            if item.name == name:
                return item
        known = ", ".join(c.name for c in self.checks)
        raise KeyError(f"no check named {name!r}. Ran: {known}")

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == FAIL)

    @property
    def skipped(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == SKIP)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "passed": self.passed,
            "n_shots": self.n_shots,
            "n_rounds": self.n_rounds,
            "n_sites": self.n_sites,
            "decoder": self.decoder,
            "checks": [c.as_dict() for c in self.checks],
            "meta": self.meta,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=float)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationReport:
        return cls(
            passed=bool(data["passed"]),
            checks=tuple(Check.from_dict(c) for c in data["checks"]),
            n_shots=int(data["n_shots"]),
            n_rounds=int(data["n_rounds"]),
            n_sites=int(data["n_sites"]),
            decoder=data["decoder"],
            meta=dict(data.get("meta", {})),
            schema=data.get("schema", VALIDATION_SCHEMA),
        )

    @classmethod
    def from_json(cls, text: str) -> ValidationReport:
        parsed: dict[str, Any] = json.loads(text)
        return cls.from_dict(parsed)

    def __str__(self) -> str:
        head = f"record validation: {'PASS' if self.passed else 'FAIL'} "
        head += f"({self.n_shots} shots, {self.n_rounds} rounds, {self.n_sites} sites)"
        return "\n".join([head, *(f"  {c}" for c in self.checks)])


@dataclass(frozen=True)
class ResidualReport:
    """The outcome of :func:`qb_compiler.record.residual.residual`.

    ``residual_bits`` is held-out log loss of the baseline minus held-out log loss of the baseline
    plus the supplied features, in bits per shot. ``null_floor_p95`` is the 95th percentile of the
    same quantity over the nulls. ``above_floor`` is the one comparison this report makes; it
    carries no reading of what a number means.
    """

    residual_bits: float
    null_floor_mean: float
    null_floor_p95: float
    above_floor: bool
    auc_baseline: float
    auc_augmented: float
    loss_baseline_bits: float
    loss_augmented_bits: float
    feature_names: tuple[str, ...]
    baseline_names: tuple[str, ...]
    holdout: str
    n_folds: int
    n_shots: int
    n_failures: int
    seed: int
    regularization_c: float
    nulls: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    schema: str = RESIDUAL_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "residual_bits": self.residual_bits,
            "null_floor_mean": self.null_floor_mean,
            "null_floor_p95": self.null_floor_p95,
            "above_floor": self.above_floor,
            "auc_baseline": self.auc_baseline,
            "auc_augmented": self.auc_augmented,
            "loss_baseline_bits": self.loss_baseline_bits,
            "loss_augmented_bits": self.loss_augmented_bits,
            "feature_names": list(self.feature_names),
            "baseline_names": list(self.baseline_names),
            "holdout": self.holdout,
            "n_folds": self.n_folds,
            "n_shots": self.n_shots,
            "n_failures": self.n_failures,
            "seed": self.seed,
            "regularization_c": self.regularization_c,
            "nulls": self.nulls,
            "warnings": list(self.warnings),
            "meta": self.meta,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=float)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualReport:
        return cls(
            residual_bits=float(data["residual_bits"]),
            null_floor_mean=float(data["null_floor_mean"]),
            null_floor_p95=float(data["null_floor_p95"]),
            above_floor=bool(data["above_floor"]),
            auc_baseline=float(data["auc_baseline"]),
            auc_augmented=float(data["auc_augmented"]),
            loss_baseline_bits=float(data["loss_baseline_bits"]),
            loss_augmented_bits=float(data["loss_augmented_bits"]),
            feature_names=tuple(data["feature_names"]),
            baseline_names=tuple(data["baseline_names"]),
            holdout=data["holdout"],
            n_folds=int(data["n_folds"]),
            n_shots=int(data["n_shots"]),
            n_failures=int(data["n_failures"]),
            seed=int(data["seed"]),
            regularization_c=float(data["regularization_c"]),
            nulls=dict(data.get("nulls", {})),
            warnings=tuple(data.get("warnings", ())),
            meta=dict(data.get("meta", {})),
            schema=data.get("schema", RESIDUAL_SCHEMA),
        )

    @classmethod
    def from_json(cls, text: str) -> ResidualReport:
        parsed: dict[str, Any] = json.loads(text)
        return cls.from_dict(parsed)

    def __str__(self) -> str:
        return (
            f"residual {self.residual_bits:+.5f} bits/shot over a "
            f"{len(self.baseline_names)}-column baseline; "
            f"null floor p95 {self.null_floor_p95:+.5f}; above floor: {self.above_floor} "
            f"({self.holdout} hold-out, {self.n_folds} folds, n={self.n_shots})"
        )


__all__ = [
    "FAIL",
    "FIRST_LAYER_PREMISE",
    "PASS",
    "RESIDUAL_SCHEMA",
    "SKIP",
    "VALIDATION_SCHEMA",
    "Check",
    "RecordDem",
    "RecordSpec",
    "ResidualReport",
    "ValidationReport",
]
