# SPDX-License-Identifier: Apache-2.0
"""Loaders: the .npz contract's refusals, and the two refusals the real loaders make.

No data is needed here. The QuEra loader refuses the naive detector construction before it opens
anything, and the Fez loader refuses a layout that is not the layout it reads.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qb_compiler.record.dem import build_repetition_dem
from qb_compiler.record.loaders import generic_npz, ibm_fez_repetition, quera_surface
from qb_compiler.record.types import FIRST_LAYER_PREMISE, RecordSpec

from .conftest import make_spec


def _write(path, **payload):
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return path


class TestGenericNpzRoundTrip:
    def test_a_full_record_survives_the_round_trip(self, tmp_path, good_spec):
        target = tmp_path / "record.npz"
        generic_npz.write_npz(target, good_spec)
        restored = generic_npz.read_npz(target)
        assert np.array_equal(restored.detectors, good_spec.detectors)
        assert np.array_equal(restored.labels, good_spec.labels)
        assert np.array_equal(restored.raw_syndromes, good_spec.raw_syndromes)
        assert np.array_equal(restored.final_parity, good_spec.final_parity)
        assert np.array_equal(restored.logical_state, good_spec.logical_state)
        assert np.array_equal(restored.data0, good_spec.data0)
        assert np.array_equal(restored.groups, good_spec.groups)
        assert restored.meta["loader"] == "synthetic"
        assert restored.meta["schema"] == generic_npz.SCHEMA
        assert restored.dem is not None
        assert np.array_equal(restored.dem.check_matrix, good_spec.dem.check_matrix)
        assert np.array_equal(restored.dem.weights, good_spec.dem.weights)
        assert np.array_equal(restored.dem.edge_qubits, good_spec.dem.edge_qubits)

    def test_a_validation_report_is_the_same_either_side(self, tmp_path, good_spec):
        pytest.importorskip("pymatching")
        from qb_compiler.record import validate

        target = generic_npz.write_npz(tmp_path / "record.npz", good_spec)
        before = validate(good_spec)
        after = validate(generic_npz.read_npz(target))
        assert before.passed == after.passed
        assert [c.status for c in before.checks] == [c.status for c in after.checks]

    def test_an_index_and_a_loss_array_survive(self, tmp_path):
        index = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int32)
        spec = RecordSpec(
            detectors=np.ones((7, 2, 3), dtype=np.uint8),
            detector_index=index,
            labels=np.zeros(7, dtype=np.uint8),
            loss=np.zeros((7, 9), dtype=bool),
            site_coords=np.arange(6).reshape(3, 2),
        )
        restored = generic_npz.read_npz(generic_npz.write_npz(tmp_path / "r.npz", spec))
        assert np.array_equal(restored.detector_index, index)
        assert restored.loss.dtype == np.dtype("bool")
        assert restored.n_detectors == 5
        assert np.array_equal(restored.site_coords, spec.site_coords)

    def test_a_record_without_labels_cannot_be_written(self, tmp_path):
        spec = RecordSpec(detectors=np.zeros((5, 2, 2), dtype=np.uint8))
        with pytest.raises(ValueError, match="requires labels"):
            generic_npz.write_npz(tmp_path / "r.npz", spec)

    def test_the_first_layer_premise_survives_the_round_trip(self, tmp_path, good_spec):
        """It travels in meta, so a record written to the contract keeps it and stays checkable."""
        pytest.importorskip("pymatching")
        from qb_compiler.record import validate

        restored = generic_npz.read_npz(generic_npz.write_npz(tmp_path / "r.npz", good_spec))
        assert restored.meta[FIRST_LAYER_PREMISE] == good_spec.meta[FIRST_LAYER_PREMISE]
        assert validate(restored).check("round_profile").status == "PASS"


class TestGenericNpzRefusals:
    def test_a_missing_required_key_is_named(self, tmp_path):
        target = _write(tmp_path / "r.npz", dets=np.zeros((4, 2, 2), dtype=np.uint8))
        with pytest.raises(ValueError, match="'labels' is missing"):
            generic_npz.read_npz(target)

    def test_a_widened_dtype_is_refused_rather_than_coerced(self, tmp_path):
        target = _write(
            tmp_path / "r.npz",
            dets=np.zeros((4, 2, 2), dtype=np.int64),
            labels=np.zeros(4, dtype=np.uint8),
        )
        with pytest.raises(ValueError, match="does not coerce"):
            generic_npz.read_npz(target)

    def test_a_flattened_record_is_refused(self, tmp_path):
        target = _write(
            tmp_path / "r.npz",
            dets=np.zeros((4, 4), dtype=np.uint8),
            labels=np.zeros(4, dtype=np.uint8),
        )
        with pytest.raises(ValueError, match="expected 3"):
            generic_npz.read_npz(target)

    def test_a_label_count_that_does_not_match_is_refused(self, tmp_path):
        target = _write(
            tmp_path / "r.npz",
            dets=np.zeros((4, 2, 2), dtype=np.uint8),
            labels=np.zeros(5, dtype=np.uint8),
        )
        with pytest.raises(ValueError, match="against 4 shots"):
            generic_npz.read_npz(target)

    def test_a_model_without_its_observable_mask_is_refused(self, tmp_path):
        target = _write(
            tmp_path / "r.npz",
            dets=np.zeros((4, 2, 2), dtype=np.uint8),
            labels=np.zeros(4, dtype=np.uint8),
            H=np.zeros((4, 6), dtype=np.uint8),
        )
        with pytest.raises(ValueError, match="'L' is not"):
            generic_npz.read_npz(target)

    def test_a_model_whose_parts_disagree_is_refused(self, tmp_path):
        target = _write(
            tmp_path / "r.npz",
            dets=np.zeros((4, 2, 2), dtype=np.uint8),
            labels=np.zeros(4, dtype=np.uint8),
            H=np.zeros((4, 6), dtype=np.uint8),
            L=np.zeros(3, dtype=np.uint8),
        )
        with pytest.raises(ValueError, match="mechanisms in 'H'"):
            generic_npz.read_npz(target)

    def test_a_model_without_weights_gets_flat_ones(self, tmp_path):
        dem = build_repetition_dem(3, 2)
        target = _write(
            tmp_path / "r.npz",
            dets=np.zeros((4, 3, 2), dtype=np.uint8),
            labels=np.zeros(4, dtype=np.uint8),
            H=np.asarray(dem.check_matrix, dtype=np.uint8),
            L=np.asarray(dem.observable, dtype=np.uint8),
        )
        restored = generic_npz.read_npz(target)
        assert np.array_equal(restored.dem.weights, np.ones(dem.n_mechanisms))

    def test_a_loss_array_of_the_wrong_dtype_is_refused(self, tmp_path):
        target = _write(
            tmp_path / "r.npz",
            dets=np.zeros((4, 2, 2), dtype=np.uint8),
            labels=np.zeros(4, dtype=np.uint8),
            loss=np.zeros((4, 3), dtype=np.float64),
        )
        with pytest.raises(ValueError, match="expected uint8 or bool"):
            generic_npz.read_npz(target)

    def test_a_detector_index_of_the_wrong_shape_is_refused(self, tmp_path):
        target = _write(
            tmp_path / "r.npz",
            dets=np.zeros((4, 2, 2), dtype=np.uint8),
            labels=np.zeros(4, dtype=np.uint8),
            det_index=np.zeros((3, 2), dtype=np.int32),
        )
        with pytest.raises(ValueError, match="det_index"):
            generic_npz.read_npz(target)

    def test_meta_that_is_not_an_object_is_refused(self, tmp_path):
        target = _write(
            tmp_path / "r.npz",
            dets=np.zeros((4, 2, 2), dtype=np.uint8),
            labels=np.zeros(4, dtype=np.uint8),
            meta=np.array(json.dumps([1, 2, 3])),
        )
        with pytest.raises(ValueError, match="JSON object"):
            generic_npz.read_npz(target)


class TestQueraRefusals:
    def test_the_naive_construction_is_refused_before_anything_is_opened(self):
        with pytest.raises(ValueError, match="refused"):
            quera_surface.load("no-such.zip", "no-such-member.npz", "nowhere", detectors="naive")

    def test_the_refusal_says_why(self):
        assert "0.45" in quera_surface.NAIVE_REFUSED
        assert "frame" in quera_surface.NAIVE_REFUSED
        assert "get_detectors" in quera_surface.NAIVE_REFUSED

    def test_an_unknown_construction_is_refused(self):
        with pytest.raises(ValueError, match="must be 'vendor' or 'naive'"):
            quera_surface.load("x.zip", "m.npz", "nowhere", detectors="guess")

    def test_a_member_it_cannot_read_a_distance_from_is_refused(self):
        with pytest.raises(ValueError, match="cannot read the distance"):
            quera_surface.load("x.zip", "not_a_memory_file.npz", "nowhere")

    def test_a_missing_vendor_directory_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not bundled"):
            quera_surface.load(
                tmp_path / "x.zip",
                "Data/Distance5/d5_Z_memory.npz",
                tmp_path / "nowhere",
            )

    def test_the_citation_names_the_record(self):
        assert "15685795" in quera_surface.CITATION


class TestFezRefusals:
    def test_impossible_parameters_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="d must be at least 2"):
            ibm_fez_repetition.load(tmp_path, d=1, r=3)
        with pytest.raises(ValueError, match="r must be at least 1"):
            ibm_fez_repetition.load(tmp_path, d=5, r=0)

    def test_a_missing_regime_directory_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="d5_r5"):
            ibm_fez_repetition.load(tmp_path, d=5, r=5)

    def test_a_regime_with_no_jobs_is_named(self, tmp_path):
        (tmp_path / "d5_r5").mkdir()
        with pytest.raises(FileNotFoundError, match="no job_"):
            ibm_fez_repetition.load(tmp_path, d=5, r=5)

    def test_a_basis_that_is_not_there_lists_the_ones_that_are(self, tmp_path):
        job = tmp_path / "d5_r5" / "job_1"
        job.mkdir(parents=True)
        (job / "info.json").write_text(json.dumps({"basis": "Z", "shots": 10}))
        with pytest.raises(FileNotFoundError, match=r"no X-basis job"):
            ibm_fez_repetition.load(tmp_path, d=5, r=5, basis="X")

    def test_a_syndrome_register_of_the_wrong_size_is_refused(self, tmp_path):
        job = tmp_path / "d3_r2" / "job_1"
        job.mkdir(parents=True)
        (job / "info.json").write_text(json.dumps({"basis": "Z", "shots": 4}))
        (job / "bitstrings.json").write_text(
            json.dumps(
                [
                    {
                        "metadata": {"logical_state": 0},
                        "per_shot_cregs": {
                            "c_data_a": [[0, 0, 0]] * 4,
                            "c_syndrome_a": [[0] * 3] * 4,
                        },
                    }
                ]
            )
        )
        with pytest.raises(ValueError, match="expected 16"):
            ibm_fez_repetition.load(tmp_path, d=3, r=2)

    def test_a_data_register_without_its_syndrome_register_is_refused(self, tmp_path):
        job = tmp_path / "d3_r2" / "job_1"
        job.mkdir(parents=True)
        (job / "info.json").write_text(json.dumps({"basis": "Z", "shots": 4}))
        (job / "bitstrings.json").write_text(
            json.dumps(
                [
                    {
                        "metadata": {"logical_state": 0},
                        "per_shot_cregs": {"c_data_a": [[0, 0, 0]] * 4},
                    }
                ]
            )
        )
        with pytest.raises(ValueError, match="no matching c_syndrome_a"):
            ibm_fez_repetition.load(tmp_path, d=3, r=2)

    def test_the_stored_order_is_the_default_that_is_corrected(self, tmp_path, synthetic_run):
        """The loader's default matches what make_spec calls the right way round."""
        assert make_spec(synthetic_run, mirrored=False).meta["mirrored"] is False
        del tmp_path
