# SPDX-License-Identifier: Apache-2.0
"""The record checks and the residual metric against real records.

Skipped unless the records are on the machine. Nothing here is redistributed with this package
and nothing downloads.

===================== ==========================================================================
variable              what it points at
===================== ==========================================================================
``QB_RECORD_DATA``    a directory holding ``ibm_fez/``, which holds the ``d<d>_r<r>`` regime
                      directories of an IBM Fez repetition-code memory run
``QB_QUERA_ZIP``      the QuEra surface-code archive as the publisher serves it. Defaults to
                      ``$QB_RECORD_DATA/quera_surface_code.zip``
``QB_QUERA_VENDOR``   the decoding framework published with that dataset, the directory holding
                      ``memory.py`` and ``noise_model.py``
``QB_CAT_DATA``       the published AWS cat qubit repetition-code deposit, the directory holding
                      the per-section folders and the README. Defaults to
                      ``~/qec_public_data/aws_cat_2025/data_upload``
===================== ==========================================================================

The QuEra sections rebuild the publisher's own decoding setup here in the test, which is the only
place that logic lives in this repository: the package itself never vendors third-party code. The
cat qubit section builds its record here for the same reason: there is no loader for that deposit
in the package, and this test reads it through its README.
"""

from __future__ import annotations

import dataclasses
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from qb_compiler.record import residual, validate
from qb_compiler.record.controls import time_mirror
from qb_compiler.record.dem import (
    build_repetition_dem,
    decode_records,
    decode_with_weight,
    fingerprint,
)
from qb_compiler.record.dem.repetition import expected_repetition_fingerprint
from qb_compiler.record.loaders import ibm_fez_repetition, quera_surface
from qb_compiler.record.types import FIRST_LAYER_PREMISE, RecordDem, RecordSpec

pytestmark = pytest.mark.integration

RECORD_DATA = os.environ.get("QB_RECORD_DATA")
QUERA_VENDOR = os.environ.get("QB_QUERA_VENDOR")
QUERA_MEMBER = "Zenodo/SurfaceCodeData/Data/Distance5/d5_Z_memory.npz"

#: Measured on the released d5 Z record, ordinary matching on the published model, all
#: 5,834 shots: 631 failures, 10.8159 percent.
PLAIN_RATE = 10.82
RESIDUAL_LOW = 0.015
RESIDUAL_HIGH = 0.031
CAT_DATA = os.environ.get("QB_CAT_DATA") or "~/qec_public_data/aws_cat_2025/data_upload"


def _fez_root() -> Path | None:
    if not RECORD_DATA:
        return None
    root = Path(RECORD_DATA).expanduser() / "ibm_fez"
    return root if root.is_dir() else None


def _quera_zip() -> Path | None:
    explicit = os.environ.get("QB_QUERA_ZIP")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    if not RECORD_DATA:
        return None
    path = Path(RECORD_DATA).expanduser() / "quera_surface_code.zip"
    return path if path.is_file() else None


requires_record_data = pytest.mark.skipif(
    _fez_root() is None,
    reason="set QB_RECORD_DATA to a directory holding ibm_fez/",
)
requires_quera = pytest.mark.skipif(
    _quera_zip() is None or not QUERA_VENDOR or not Path(QUERA_VENDOR).expanduser().is_dir(),
    reason="set QB_QUERA_ZIP (or put the archive under QB_RECORD_DATA) and QB_QUERA_VENDOR",
)


def _cat_root() -> Path | None:
    root = Path(CAT_DATA).expanduser()
    return root if root.is_dir() else None


requires_cat_data = pytest.mark.skipif(
    _cat_root() is None,
    reason="set QB_CAT_DATA to the published AWS cat qubit repetition-code deposit",
)

pytest.importorskip("pymatching")
pytest.importorskip("stim")


def _fez_spec(d: int, rounds: int, *, stored_order: bool = False) -> RecordSpec:
    root = _fez_root()
    assert root is not None
    loaded = ibm_fez_repetition.load(
        root, d=d, r=rounds, basis="Z", reverse_rounds=not stored_order
    )
    return RecordSpec(
        detectors=loaded.detectors,
        labels=loaded.labels,
        groups=loaded.groups,
        raw_syndromes=loaded.raw_syndromes,
        final_parity=loaded.final_parity,
        logical_state=loaded.logical_state,
        data0=loaded.data0,
        dem=build_repetition_dem(d, rounds, p_data=0.01, p_meas=0.01),
        meta=loaded.meta,
    )


def _logical_error_rate(spec: RecordSpec) -> float:
    assert spec.dem is not None
    prediction = decode_records(spec.dem, spec.detector_matrix())
    return float((prediction != np.asarray(spec.labels)).mean())


# ---------------------------------------------------------------- IBM Fez


@requires_record_data
class TestFezOrientation:
    def test_the_corrected_orientation_passes(self):
        report = validate(_fez_spec(5, 5))
        assert report.passed
        assert report.check("round_profile").status == "PASS"
        assert report.check("endpoint_agreement").status == "PASS"
        assert report.check("state_profile").status == "PASS"

    def test_the_stored_orientation_is_caught(self):
        report = validate(_fez_spec(5, 5, stored_order=True))
        assert not report.passed
        assert report.check("round_profile").status == "FAIL"
        assert report.check("endpoint_agreement").status == "FAIL"
        assert report.check("state_profile").status == "FAIL"

    def test_the_first_round_rate_is_the_tell(self):
        corrected = validate(_fez_spec(5, 5)).check("round_profile").measured
        stored = validate(_fez_spec(5, 5, stored_order=True)).check("round_profile").measured
        assert 0.05 <= corrected["first_round_rate"] <= 0.06
        assert 0.19 <= stored["first_round_rate"] <= 0.20
        assert corrected["first_round_ratio"] < 0.9 < stored["first_round_ratio"]

    def test_the_endpoint_ratio_inverts(self):
        corrected = validate(_fez_spec(5, 5)).check("endpoint_agreement").measured
        stored = validate(_fez_spec(5, 5, stored_order=True)).check("endpoint_agreement").measured
        assert corrected["ratio"] < 0.5
        assert stored["ratio"] > 2.0
        assert corrected["mismatch_last_round"] == pytest.approx(
            stored["mismatch_first_round"], rel=1e-9
        )

    def test_the_time_mirror_control_has_no_power_on_this_record(self):
        """Measured, and worth stating: V5 does not catch the stored-round fault.

        Reversing the detector axis of a repetition memory costs almost nothing, because the
        record is nearly symmetric in time: a fresh first round and a final data round that fire
        at similar rates, with homogeneous rounds between them. So the control passes on the
        corrected record, on the stored-order record, and on a record whose detector axis has
        deliberately been reversed. The stored-round fault is caught by the round profile, the
        endpoint agreement and the state profile, not by this control. The check is advisory, so
        none of this decides the verdict either way.
        """
        corrected = _fez_spec(5, 5)
        stored = _fez_spec(5, 5, stored_order=True)
        flipped = RecordSpec(
            detectors=time_mirror(corrected.detectors),
            labels=corrected.labels,
            dem=corrected.dem,
        )
        for spec in (corrected, stored, flipped):
            result = validate(spec).check("time_mirror_control")
            assert result.status == "PASS"
            assert not result.critical
            assert result.measured["n_failures"] >= 30
            assert result.measured["difference"] <= result.measured["limit"]


@requires_record_data
class TestFezReferenceRates:
    @pytest.mark.parametrize(
        ("d", "rounds", "low", "high"),
        [(11, 11, 0.0135, 0.0155), (5, 5, 0.0155, 0.0170)],
    )
    def test_flat_prior_matching_lands_in_the_reference_band(self, d, rounds, low, high):
        """No monotonicity in distance is asserted, because the record does not show it."""
        assert low <= _logical_error_rate(_fez_spec(d, rounds)) <= high

    def test_the_stored_orientation_is_several_times_worse(self):
        corrected = _logical_error_rate(_fez_spec(11, 11))
        stored = _logical_error_rate(_fez_spec(11, 11, stored_order=True))
        assert stored > 5 * corrected

    def test_the_model_fingerprint_is_the_arithmetic(self):
        spec = _fez_spec(11, 11)
        assert spec.dem is not None
        assert fingerprint(spec.dem) == expected_repetition_fingerprint(11, 11)
        report = validate(
            spec,
            thresholds={"dem_fingerprint": {"expected": expected_repetition_fingerprint(11, 11)}},
        )
        assert report.check("dem_fingerprint").status == "PASS"
        assert report.check("dem_fingerprint").critical
        assert report.passed

    def test_the_wrong_model_is_caught_on_a_real_record(self):
        report = validate(
            _fez_spec(11, 11),
            thresholds={"dem_fingerprint": {"expected": expected_repetition_fingerprint(9, 9)}},
        )
        assert report.check("dem_fingerprint").status == "FAIL"
        assert not report.passed


def _augmented_gap(dem: RecordDem, detector_matrix: np.ndarray):
    """``(prediction, solution weight, gap)`` with the observable carried as an extra detector."""
    import pymatching

    n_shots, n_detectors = detector_matrix.shape
    augmented = np.zeros((n_detectors + 1, dem.n_mechanisms), dtype=np.uint8)
    augmented[:n_detectors] = np.asarray(dem.check_matrix, dtype=np.uint8)
    augmented[n_detectors] = np.asarray(dem.observable, dtype=np.uint8)
    matching = pymatching.Matching(augmented, weights=np.asarray(dem.weights, dtype=np.float64))
    prediction = np.zeros(n_shots, dtype=np.uint8)
    weight = np.zeros(n_shots)
    gap = np.zeros(n_shots)
    syndrome = np.zeros(n_detectors + 1, dtype=np.uint8)
    for shot in range(n_shots):
        syndrome[:n_detectors] = detector_matrix[shot]
        syndrome[n_detectors] = 0
        _, even = matching.decode(syndrome, return_weight=True)
        syndrome[n_detectors] = 1
        _, odd = matching.decode(syndrome, return_weight=True)
        if even <= odd:
            prediction[shot], weight[shot], gap[shot] = 0, even, odd - even
        else:
            prediction[shot], weight[shot], gap[shot] = 1, odd, even - odd
    return prediction, weight, gap


def _per_round_counts(record: RecordSpec):
    dets = np.asarray(record.detectors, dtype=np.float64)
    valid = record.site_valid
    columns = [dets[:, t, np.flatnonzero(valid[t])].sum(axis=1) for t in range(record.n_rounds)]
    return np.column_stack(columns), [f"count_round_{t}" for t in range(record.n_rounds)]


@requires_record_data
class TestFezResidual:
    @pytest.mark.parametrize(("d", "rounds"), [(11, 11), (9, 9)])
    def test_per_round_counts_add_nothing_over_the_decoder_summary(self, d, rounds):
        spec = _fez_spec(d, rounds)
        assert spec.dem is not None
        detectors = spec.detector_matrix()
        _prediction, weight, gap = _augmented_gap(spec.dem, detectors)
        served = decode_records(spec.dem, detectors)
        report = residual(
            spec,
            {"prediction": served, "weight": weight, "gap": gap},
            _per_round_counts,
            holdout="group",
            n_nulls=20,
            seed=0,
        )
        assert report.residual_bits < 0.002
        assert not report.above_floor
        assert report.holdout == "group"
        assert report.auc_baseline > 0.95

    def test_the_gap_sign_agrees_off_the_ties(self):
        """At flat priors the repetition graph has exact ties; they are reported, not hidden."""
        spec = _fez_spec(11, 11)
        assert spec.dem is not None
        detectors = spec.detector_matrix()
        prediction, _weight, gap = _augmented_gap(spec.dem, detectors)
        served = decode_records(spec.dem, detectors)
        ties = int((gap == 0).sum())
        untied = gap > 0
        assert ties == 240
        assert np.array_equal(prediction[untied], served[untied])
        assert int(untied.sum()) + ties == spec.n_shots


# ---------------------------------------------------------------- QuEra


def _quera_spec() -> RecordSpec:
    archive = _quera_zip()
    assert archive is not None and QUERA_VENDOR is not None
    return quera_surface.load(archive, QUERA_MEMBER, QUERA_VENDOR)


def _quera_model(spec: RecordSpec, observable_support: list[int]) -> RecordDem:
    """The publisher's own decoding setup, rebuilt here.

    Returns the error model the published circuit describes. This is the only copy of this logic
    in the repository and it lives in the test, not in the package.
    """
    import pymatching
    import stim

    archive = _quera_zip()
    assert archive is not None and QUERA_VENDOR is not None
    vendor_dir = str(Path(QUERA_VENDOR).expanduser().resolve())
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    from memory import MemorySimulator
    from noise_model import NoiseModel

    d = int(spec.meta["d"])
    n_data = d * d
    simulator = MemorySimulator(
        basis=spec.meta["basis"],
        noise_model=NoiseModel(NoiseModel.DEFAULT_NOISE_PARAMS),
        d=d,
        quadrant=None,
    )
    circuit = stim.Circuit(
        "\n".join(str(i) for i in simulator.lc.cleanse_custom_instrs().instructions)
    )
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-n_data + int(i)) for i in observable_support],
        0,
    )
    graph = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    )
    n_detectors = spec.n_detectors
    edges = [
        (u, n_detectors if v is None else v, a.get("weight", 1.0), bool(a.get("fault_ids")))
        for u, v, a in graph.edges()
    ]
    check = np.zeros((n_detectors, len(edges)), dtype=np.uint8)
    observable = np.zeros(len(edges), dtype=np.uint8)
    for i, (u, v, _, flips) in enumerate(edges):
        check[u, i] = 1
        if v != n_detectors:
            check[v, i] = 1
        observable[i] = 1 if flips else 0
    return RecordDem(
        check_matrix=check,
        observable=observable,
        weights=np.asarray([e[2] for e in edges], dtype=np.float64),
    )


@requires_quera
class TestQueraConstruction:
    def test_the_naive_construction_is_refused(self):
        archive = _quera_zip()
        with pytest.raises(ValueError, match="refused"):
            quera_surface.load(archive, QUERA_MEMBER, QUERA_VENDOR, detectors="naive")

    def test_the_published_construction_passes_the_density_and_profile_checks(self):
        report = validate(_quera_spec())
        density = report.check("event_density")
        profile = report.check("round_profile")
        assert density.status == "PASS"
        assert 0.13 <= density.measured["event_rate"] <= 0.135
        assert profile.status == "PASS"
        assert 0.062 <= profile.measured["first_round_rate"] <= 0.064

    def test_the_deterministic_stabilizers_are_not_coin_flips(self):
        result = validate(_quera_spec()).check("type_consistency")
        assert result.status == "PASS"
        assert result.measured["n_means"] == 48
        assert result.measured["n_inside_band"] == 0
        assert result.measured["min_mean"] < 0.03
        assert result.measured["max_mean"] > 0.94

    def test_the_labels_reconstruct_from_the_final_readout(self):
        result = validate(_quera_spec()).check("label_reconstruction")
        assert result.status == "PASS"
        assert result.measured["agreement"] == 1.0
        assert result.measured["n_shots_used"] == 205


@requires_quera
class TestQueraModel:
    def test_the_fingerprint_is_the_published_graph(self):
        spec = _quera_spec()
        support = validate(spec).check("label_reconstruction").measured["observable_support"]
        measured = fingerprint(_quera_model(spec, support))
        assert measured["n_mechanisms"] == 388
        assert measured["n_boundary"] == 66
        assert measured["mechanism_size_histogram"] == {"1": 66, "2": 322}

    def test_the_published_model_checks_out_against_the_record(self):
        spec = _quera_spec()
        support = validate(spec).check("label_reconstruction").measured["observable_support"]
        dem = _quera_model(spec, support)
        full = RecordSpec(
            detectors=spec.detectors,
            detector_index=spec.detector_index,
            labels=spec.labels,
            loss=spec.loss,
            raw_syndromes=spec.raw_syndromes,
            final_data=spec.final_data,
            deterministic_sites=spec.deterministic_sites,
            dem=dem,
            meta=spec.meta,
        )
        report = validate(full, thresholds={"dem_fingerprint": {"expected": {"n_mechanisms": 388}}})
        assert report.passed
        assert report.check("dem_fingerprint").status == "PASS"

    def test_the_matching_decoder_lands_where_it_did(self):
        """Ordinary matching on the published model and the published detectors."""
        spec = _quera_spec()
        support = validate(spec).check("label_reconstruction").measured["observable_support"]
        dem = _quera_model(spec, support)
        prediction = decode_records(dem, spec.detector_matrix())
        rate = 100.0 * float((prediction != np.asarray(spec.labels, dtype=np.uint8)).mean())
        assert abs(rate - PLAIN_RATE) <= 0.05, f"came out at {rate:.4f} percent"


@requires_quera
class TestQueraResidual:
    def _served(self, spec: RecordSpec, dem: RecordDem):
        """The ordinary matching decoder's own output: what it predicted and what it paid."""
        prediction, weight = decode_with_weight(dem, spec.detector_matrix())
        return {"prediction": prediction, "weight": weight}

    def _setup(self):
        spec = _quera_spec()
        support = validate(spec).check("label_reconstruction").measured["observable_support"]
        dem = _quera_model(spec, support)
        archive = _quera_zip()
        assert archive is not None
        with (
            zipfile.ZipFile(archive) as handle,
            np.load(io.BytesIO(handle.read(QUERA_MEMBER))) as payload,
        ):
            gaps = np.asarray(payload["gaps"], dtype=np.float64)
        full = RecordSpec(
            detectors=spec.detectors,
            detector_index=spec.detector_index,
            labels=spec.labels,
            loss=spec.loss,
            dem=dem,
            meta=spec.meta,
        )
        return full, self._served(spec, dem), gaps

    def test_the_published_confidence_is_above_the_floor(self):
        """The record is correctly built and the decoder still leaves something on it."""
        spec, output, gaps = self._setup()
        report = residual(
            spec, output, (gaps[:, None], ["published_confidence"]), n_nulls=20, seed=0
        )
        assert RESIDUAL_LOW <= report.residual_bits <= RESIDUAL_HIGH, f"{report.residual_bits:.5f}"
        assert report.above_floor
        assert report.null_floor_p95 < 0.001
        assert report.holdout == "shot"
        assert report.n_shots == 5834

    def test_per_round_counts_are_at_the_floor(self):
        spec, output, _ = self._setup()
        report = residual(spec, output, _per_round_counts, n_nulls=20, seed=0)
        assert report.residual_bits < 0.002
        assert not report.above_floor
        assert report.nulls["geometry"]["available"] is True
        assert report.nulls["geometry"]["n"] == 20


# ---------------------------------------------------------------- AWS cat qubit deposit

# section, code distance, and the shots left after the erased preparations are dropped.
CAT_CASES = [("second_d3_phase_flip", 3, 13503), ("d5_phase_flip", 5, 12959)]


def _cat_record(section: str, d: int, cycles: int = 13, nbar: str = "nbar_2.0") -> RecordSpec:
    """One phase-flip record of the published cat qubit deposit, built to its README.

    The README gives the symbol map and the layout: a storage state is 0 for even parity and 1 for
    odd, a syndrome reading is 0 for even, 1 for an erasure and 2 for odd, and ancilla ``A<j>``
    measures the parity of storages ``S<j>`` and ``S<j+1>``. So the first layer differences the
    first valid ancilla reading against the parity the storages were prepared in, each later layer
    differences a valid reading against the most recent earlier valid one, and the last layer
    differences the final storage parity against that. Shots whose preparation or final readout
    was itself erased are dropped, because their parity is not defined.

    There is no loader for this deposit in the package and this does not become one. It is here so
    the checks can be run against a record from a platform with no quiet first layer.
    """
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    root = _cat_root()
    assert root is not None
    frame = pandas.read_parquet(root / section / nbar / f"num_cycles_{cycles}")

    storages = sorted(
        {c.split("_")[0] for c in frame.columns if c.endswith("_initial_state")},
        key=lambda name: int(name[1:]),
    )
    ancillas = sorted(
        {c.split("_")[0] for c in frame.columns if "_syndrome_" in c},
        key=lambda name: int(name[1:]),
    )
    rounds = max(int(c.split("_")[-1]) for c in frame.columns if "_syndrome_" in c) + 1
    assert len(storages) == d and len(ancillas) == d - 1

    initial = frame[[s + "_initial_state" for s in storages]].values
    final = frame[[s + "_final_state" for s in storages]].values
    prepared = (initial < 2).all(axis=1) & (final < 2).all(axis=1)
    initial, final = initial[prepared], final[prepared]
    readings = np.stack(
        [
            np.stack([frame[f"{a}_syndrome_{t}"].values for a in ancillas], axis=1)
            for t in range(rounds)
        ],
        axis=1,
    )[prepared]

    n_shots = int(initial.shape[0])
    detectors = np.zeros((n_shots, rounds + 1, d - 1), dtype=np.uint8)
    reference = (initial[:, :-1] ^ initial[:, 1:]).astype(np.uint8)
    for t in range(rounds):
        odd = (readings[:, t] == 2).astype(np.uint8)
        valid = readings[:, t] != 1
        detectors[:, t] = np.where(valid, odd ^ reference, 0)
        reference = np.where(valid, odd, reference).astype(np.uint8)
    final_parity = (final[:, :-1] ^ final[:, 1:]).astype(np.uint8)
    detectors[:, rounds] = final_parity ^ reference

    labels = (np.bitwise_xor.reduce(initial, axis=1) ^ np.bitwise_xor.reduce(final, axis=1)).astype(
        np.uint8
    )
    return RecordSpec(
        detectors=detectors,
        labels=labels,
        meta={
            "source": "published AWS cat qubit repetition-code deposit, built in the test",
            "section": section,
            "nbar": nbar,
            "cycles": int(rounds),
            "d": int(d),
            "n_dropped_erased_preparation": int((~prepared).sum()),
        },
    )


@requires_cat_data
class TestCatQubitFirstLayerPremise:
    """A correctly built record from a platform whose first layer is not quiet.

    Preparation and final readout on this platform are noisier than a mid-run ancilla reading, so
    both ends of the layer profile are loud and the interior is flat. The round profile rule reads
    a quiet first layer, so on this record it has nothing to separate and abstains rather than
    calling a sound record a fault.
    """

    @pytest.mark.parametrize(("section", "d", "n_shots"), CAT_CASES)
    def test_the_round_profile_abstains_and_the_record_passes(self, section, d, n_shots):
        spec = _cat_record(section, d)
        assert spec.n_shots == n_shots
        report = validate(spec)
        profile = report.check("round_profile")
        assert profile.status == "SKIP"
        assert not profile.critical
        assert "no declared first layer premise" in profile.detail
        assert report.passed
        assert [c.name for c in report.checks if c.critical] == ["event_density"]
        assert report.check("event_density").status == "PASS"

    @pytest.mark.parametrize(("section", "d", "n_shots"), CAT_CASES)
    def test_both_ends_are_loud_and_the_interior_is_flat(self, section, d, n_shots):
        del n_shots
        measured = validate(_cat_record(section, d)).check("round_profile").measured
        assert 1.30 <= measured["first_round_ratio"] <= 1.34
        assert 1.28 <= measured["last_round_ratio"] <= 1.39
        interior = measured["rate_per_round"][1:-1]
        assert max(interior) - min(interior) < 0.02
        assert 0.18 <= measured["steady_median"] <= 0.19

    def test_the_measured_profile_is_the_one_the_deposit_gives(self):
        """The d3 phase-flip section at nbar 2.0 over 13 cycles, 13,503 shots after the drop."""
        measured = validate(_cat_record("second_d3_phase_flip", 3)).check("round_profile").measured
        assert measured["first_round_rate"] == pytest.approx(0.2409, abs=0.0005)
        assert measured["steady_median"] == pytest.approx(0.1824, abs=0.0005)
        assert measured["last_round_rate"] == pytest.approx(0.2511, abs=0.0005)

    def test_declaring_the_premise_on_this_record_fails_it(self):
        """The check is not broken here, it is out of scope: declare the premise and it fires."""
        spec = _cat_record("second_d3_phase_flip", 3)
        declared = dataclasses.replace(spec, meta={**spec.meta, FIRST_LAYER_PREMISE: True})
        report = validate(declared)
        assert report.check("round_profile").status == "FAIL"
        assert not report.passed
