# SPDX-License-Identifier: Apache-2.0
"""Hamiltonian integrity checks and the measurement bill.

The audit's job is to catch a file that contradicts itself, so most of these tests take a
well-formed record and damage exactly one thing in memory. A check nobody has watched fail is not
evidence that the check works.
"""

from __future__ import annotations

import copy
import json
from typing import ClassVar

import pytest

from qb_compiler.chem import (
    HamiltonianFormatError,
    audit_hamiltonian,
    audit_hamiltonian_file,
    load_hamiltonian,
    measurement_plan,
    parse_hamiltonian,
    qwc_groups,
)
from qb_compiler.chem.hamiltonian import PauliTerm

GOOD = {
    "metadata": {
        "id": "h2_2e2o",
        "name": "Hydrogen",
        "basis": "sto-3g",
        "active_space": {"n_electrons": 2, "n_orbitals": 2},
        "n_qubits": 4,
        "mapping": "jordan_wigner",
        "provenance": "RDKit -> PySCF -> OpenFermion -> JW -> sparse_FCI",
        "synthetic": False,
    },
    "pauli_groups": [
        {"pauli_strings": ["I"], "coefficients": [-0.81], "measurable": False},
        {"pauli_strings": ["Z0", "Z1"], "coefficients": [0.17, 0.17], "measurable": True},
        {"pauli_strings": ["X0 X1 Y2 Y3"], "coefficients": [0.045], "measurable": True},
    ],
    "reference_energy_type": "FCI_sparse",
    "reference_rhf_energy": -1.116,
    "reference_exact_energy": -1.137,
}


def _damaged(**changes):
    """A copy of GOOD with one defect injected, never written to disk."""
    record = copy.deepcopy(GOOD)
    for path, value in changes.items():
        keys = path.split(".")
        target = record
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return record


class TestAudit:
    def test_a_well_formed_file_is_accepted(self):
        verdict = audit_hamiltonian(GOOD)
        assert verdict.verdict == "ACCEPT"
        assert verdict.accepted
        assert verdict.failures() == []
        assert len(verdict.checks) == 5

    def test_mean_field_reference_is_refused(self):
        verdict = audit_hamiltonian(_damaged(reference_energy_type="RHF"))
        assert verdict.verdict == "REFUSE"
        assert "reference method is correlated" in verdict.failures()

    def test_exact_equal_to_rhf_is_refused(self):
        """Correlation energy of exactly zero is the signature of a placeholder record."""
        verdict = audit_hamiltonian(_damaged(reference_exact_energy=GOOD["reference_rhf_energy"]))
        assert verdict.verdict == "REFUSE"
        assert "exact energy sits below RHF" in verdict.failures()

    def test_exact_above_rhf_is_refused(self):
        verdict = audit_hamiltonian(_damaged(reference_exact_energy=-1.0))
        assert "exact energy sits below RHF" in verdict.failures()

    def test_truncated_provenance_is_refused(self):
        verdict = audit_hamiltonian(
            _damaged(**{"metadata.provenance": "RDKit -> PySCF -> OpenFermion"})
        )
        assert "provenance chain reaches a correlated solve" in verdict.failures()

    def test_synthetic_flag_is_refused(self):
        verdict = audit_hamiltonian(_damaged(**{"metadata.synthetic": True}))
        assert "not flagged synthetic" in verdict.failures()

    def test_the_qubit_count_bug(self):
        """n_qubits = n_orbitals instead of 2 x n_orbitals: the defect the check exists for."""
        record = copy.deepcopy(GOOD)
        record["metadata"]["n_qubits"] = 2
        # Keep the operator inside the declared width, so the failure is the active space
        # arithmetic rather than a term reaching past the end.
        record["pauli_groups"] = [
            {"pauli_strings": ["I"], "coefficients": [-0.81], "measurable": False},
            {"pauli_strings": ["Z0 X1"], "coefficients": [0.17], "measurable": True},
        ]
        verdict = audit_hamiltonian(record)
        assert verdict.verdict == "REFUSE"
        assert "qubit count matches the active space" in verdict.failures()
        detail = next(c.detail for c in verdict.checks if c.status == "FAIL")
        assert "half the space" in detail

    def test_a_declared_width_narrower_than_the_operator_is_refused(self):
        verdict = audit_hamiltonian(_damaged(**{"metadata.n_qubits": 2}))
        assert verdict.verdict == "REFUSE"
        detail = next(c.detail for c in verdict.checks if c.status == "FAIL")
        assert "past the declared n_qubits" in detail

    def test_operator_wider_than_the_declared_qubit_count_is_refused(self):
        record = copy.deepcopy(GOOD)
        record["metadata"]["n_qubits"] = 4
        record["metadata"]["active_space"] = {"n_electrons": 2, "n_orbitals": 2}
        record["pauli_groups"].append(
            {"pauli_strings": ["Z9"], "coefficients": [0.01], "measurable": True}
        )
        verdict = audit_hamiltonian(record)
        assert "qubit count matches the active space" in verdict.failures()

    def test_a_sparse_file_is_incomplete_rather_than_refused(self):
        """Somebody else's file declares less than ours. That is not the same as a bad file."""
        verdict = audit_hamiltonian({"n_qubits": 2, "terms": [{"pauli": "Z0", "coefficient": 1.0}]})
        assert verdict.verdict == "INCOMPLETE"
        assert verdict.failures() == []
        assert len(verdict.undeclared()) == 5

    def test_strict_turns_undeclared_into_a_refusal(self):
        verdict = audit_hamiltonian(
            {"n_qubits": 2, "terms": [{"pauli": "Z0", "coefficient": 1.0}]}, strict=True
        )
        assert verdict.verdict == "REFUSE"

    def test_unknown_mapping_is_not_judged(self):
        verdict = audit_hamiltonian(_damaged(**{"metadata.mapping": "some_new_encoding"}))
        assert verdict.verdict == "INCOMPLETE"
        assert "qubit count matches the active space" in verdict.undeclared()

    def test_verdict_serialises(self):
        payload = audit_hamiltonian(GOOD).as_dict()
        assert payload["schema"] == "qb.chem_audit.v1"
        assert payload["verdict"] == "ACCEPT"
        assert len(payload["checks"]) == 5
        assert payload["signature"] is None
        json.dumps(payload)

    def test_verdict_prints_every_check(self):
        text = str(audit_hamiltonian(GOOD))
        for name in ("reference method", "provenance chain", "qubit count"):
            assert name in text

    def test_file_roundtrip(self, tmp_path):
        path = tmp_path / "h2.json"
        path.write_text(json.dumps(GOOD))
        verdict = audit_hamiltonian_file(path)
        assert verdict.accepted
        assert str(path) in verdict.source

    def test_unreadable_file_raises_a_clear_error(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{oh dear")
        with pytest.raises(HamiltonianFormatError, match="not valid JSON"):
            load_hamiltonian(path)


class TestFormats:
    def test_flat_term_list(self):
        h = parse_hamiltonian(
            {"n_qubits": 3, "terms": [{"pauli": "Z0 Z1", "coefficient": 0.5}, ["X2", 0.25]]}
        )
        assert h.n_qubits == 3
        assert len(h.terms) == 2
        assert h.terms[0].paulis == {0: "Z", 1: "Z"}

    def test_mapping_of_label_to_coefficient(self):
        h = parse_hamiltonian({"n_qubits": 2, "terms": {"Z0": 1.0, "X1": 0.5}})
        assert len(h.terms) == 2

    def test_dense_pauli_string(self):
        h = parse_hamiltonian({"n_qubits": 4, "terms": [["ZIIX", 1.0]]})
        assert h.terms[0].paulis == {0: "Z", 3: "X"}

    def test_dense_string_of_the_wrong_width_is_rejected(self):
        with pytest.raises(HamiltonianFormatError, match="characters"):
            parse_hamiltonian({"n_qubits": 4, "terms": [["ZIX", 1.0]]})

    def test_identity_forms(self):
        h = parse_hamiltonian({"n_qubits": 2, "terms": [["I", 2.0], ["", 1.0]]})
        assert all(term.is_identity for term in h.terms)
        assert h.identity_coefficient == 3.0

    def test_openfermion_style_terms(self):
        h = parse_hamiltonian({"n_qubits": 2, "terms": {((0, "Z"), (1, "X")): 0.5}})
        assert h.terms[0].paulis == {0: "Z", 1: "X"}

    def test_openfermion_qubit_operator_duck_type(self):
        class FakeQubitOperator:
            terms: ClassVar[dict] = {((0, "Z"),): 1.0, (): -0.5}

        h = parse_hamiltonian(FakeQubitOperator())
        assert h.n_qubits == 1
        assert h.identity_coefficient == -0.5
        assert h.n_qubits_declared is False

    def test_qiskit_sparse_pauli_op(self):
        qiskit_quantum_info = pytest.importorskip("qiskit.quantum_info")
        op = qiskit_quantum_info.SparsePauliOp.from_list([("XZ", 1.0), ("II", 0.5)])
        h = parse_hamiltonian(op, metadata={"n_qubits": 2})
        assert h.n_qubits == 2
        # Qiskit labels are little endian: "XZ" is Z on qubit 0, X on qubit 1.
        assert h.terms[0].paulis == {0: "Z", 1: "X"}
        assert h.identity_coefficient == 0.5

    def test_a_file_with_no_terms_is_rejected(self):
        with pytest.raises(HamiltonianFormatError, match="no Pauli terms"):
            parse_hamiltonian({"metadata": {"n_qubits": 2}})

    def test_mismatched_group_lengths_are_rejected(self):
        with pytest.raises(HamiltonianFormatError, match="coefficients"):
            parse_hamiltonian(
                {"pauli_groups": [{"pauli_strings": ["Z0", "Z1"], "coefficients": [1.0]}]}
            )

    def test_a_non_dict_is_rejected(self):
        with pytest.raises(HamiltonianFormatError, match="cannot read"):
            parse_hamiltonian(42)


class TestMeasurementPlan:
    def test_counts_and_shots(self):
        plan = measurement_plan(GOOD, shots_per_setting=1000)
        assert plan.n_qubits == 4
        assert plan.n_terms == 4
        assert plan.n_measurable_terms == 3
        assert plan.n_settings_qwc == 2  # Z0 and Z1 share a setting; XXYY needs its own
        assert plan.largest_group == 2
        assert plan.grouping_factor == pytest.approx(1.5)
        assert plan.shots_qwc == 2000
        assert plan.shots_term_by_term == 3000
        assert plan.identity_coefficient == pytest.approx(-0.81)

    def test_plan_declares_it_is_structural_only(self):
        plan = measurement_plan(GOOD)
        assert "structural" in plan.notes[0]
        assert "variance" in plan.notes[0]
        assert "precision claim" in plan.notes[0]

    def test_plan_serialises_and_prints(self):
        plan = measurement_plan(GOOD)
        payload = plan.as_dict()
        assert payload["schema"] == "qb.measure_plan.v1"
        json.dumps(payload)
        assert "QWC settings" in str(plan)

    def test_identity_only_operator_is_refused(self):
        with pytest.raises(ValueError, match="no measurable terms"):
            measurement_plan({"n_qubits": 2, "terms": [["I", 1.0]]})

    def test_shots_must_be_positive(self):
        with pytest.raises(ValueError, match="shots_per_setting"):
            measurement_plan(GOOD, shots_per_setting=0)

    def test_grouping_respects_qubitwise_commutation(self):
        terms = [
            PauliTerm({0: "X"}, 1.0),
            PauliTerm({0: "X", 1: "Y"}, 1.0),
            PauliTerm({0: "Z"}, 1.0),
        ]
        groups = qwc_groups(terms)
        assert groups == [[0, 1], [2]]

    def test_every_term_lands_in_exactly_one_group(self):
        terms = [
            PauliTerm({0: "X", 1: "Y"}, 1.0),
            PauliTerm({0: "Y"}, 1.0),
            PauliTerm({1: "Y"}, 1.0),
            PauliTerm({0: "Z", 1: "Z"}, 1.0),
        ]
        groups = qwc_groups(terms)
        placed = sorted(i for group in groups for i in group)
        assert placed == list(range(len(terms)))

    def test_terms_in_a_group_pairwise_commute(self):
        terms = [
            PauliTerm({0: "X"}, 1.0),
            PauliTerm({1: "Z"}, 1.0),
            PauliTerm({0: "X", 1: "Z"}, 1.0),
            PauliTerm({0: "Y"}, 1.0),
        ]
        for group in qwc_groups(terms):
            for i in group:
                for j in group:
                    assert terms[i].commutes_qubitwise(terms[j])
