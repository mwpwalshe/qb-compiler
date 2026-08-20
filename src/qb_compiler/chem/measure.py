# SPDX-License-Identifier: Apache-2.0
"""What measuring a Hamiltonian costs, counted before anything is submitted.

Given an operator and a shots-per-setting figure, this reports the number of measurable terms, the
number of qubit-wise commuting settings they collapse into, the grouping factor, the largest group,
and the resulting shot count. That is the bill for the measurement, in the same units the vendor
invoices in.

What this is not
----------------
It is a **structural** count and nothing else. It does not weight a setting by the variance of the
terms in it, does not allocate shots between settings, and makes no claim about the error on the
resulting energy estimate. Two operators with identical setting counts can need very different
shot budgets to reach the same precision, because precision is set by coefficients and by state
dependent variance, neither of which is visible here.

That distinction is worth keeping straight: the number below is what you must pay to observe every
term at least once at your chosen rate, and it is a floor on the structural side of the problem,
not an estimate of what a chemistry result will cost to converge.

Grouping quality is not the point either. The grouping is textbook greedy qubit-wise commuting,
chosen because it is the one everybody already knows and can reproduce. PennyLane, Qiskit Nature
and the shot-frugal VQE literature all do this better; the contribution here is that the number is
printed before the job is submitted rather than inferred afterwards from a bill.

Measured on a 14 molecule corpus, the spread is the finding worth carrying: at 12 qubits the
setting count varies by about 1.79x across molecules, at 16 qubits by about 1.59x, and qubit count
does not predict where in that range a given molecule lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qb_compiler.chem.hamiltonian import Hamiltonian, PauliTerm, parse_hamiltonian

SCHEMA = "qb.measure_plan.v1"

STRUCTURAL_ONLY_NOTE = (
    "structural count only: settings and shots follow from which terms share a measurement "
    "basis, not from term variance. No shot allocation and no precision claim is implied."
)


@dataclass(frozen=True)
class MeasurementPlan:
    """The measurement bill for one operator.

    Attributes
    ----------
    n_qubits :
        Width of the operator.
    n_terms :
        Every term in the file, identity included.
    n_measurable_terms :
        Terms needing a measurement. The identity carries the nuclear and frozen core energy: it
        is a constant, so it is excluded here rather than paid for.
    n_settings_term_by_term :
        Settings if every term were measured on its own. The naive baseline.
    n_settings_qwc :
        Settings after qubit-wise commuting grouping.
    grouping_factor :
        ``n_measurable_terms / n_settings_qwc``. How much the grouping saved.
    largest_group :
        Term count in the biggest group.
    shots_per_setting :
        The rate the caller chose.
    shots_term_by_term, shots_qwc :
        Total shots at that rate, before and after grouping.
    identity_coefficient :
        The constant offset, reported so it is not mistaken for something that needs measuring.
    """

    n_qubits: int
    n_terms: int
    n_measurable_terms: int
    n_settings_term_by_term: int
    n_settings_qwc: int
    grouping_factor: float
    largest_group: int
    shots_per_setting: int
    shots_term_by_term: int
    shots_qwc: int
    identity_coefficient: float
    source: str = "in memory"
    notes: list[str] = field(default_factory=lambda: [STRUCTURAL_ONLY_NOTE])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "source": self.source,
            "n_qubits": self.n_qubits,
            "n_terms": self.n_terms,
            "n_measurable_terms": self.n_measurable_terms,
            "n_settings_term_by_term": self.n_settings_term_by_term,
            "n_settings_qwc": self.n_settings_qwc,
            "grouping_factor": self.grouping_factor,
            "largest_group": self.largest_group,
            "shots_per_setting": self.shots_per_setting,
            "shots_term_by_term": self.shots_term_by_term,
            "shots_qwc": self.shots_qwc,
            "identity_coefficient": self.identity_coefficient,
            "notes": list(self.notes),
            "signature": None,
            "signing": "unsigned",
        }

    def __str__(self) -> str:
        saved = self.shots_term_by_term - self.shots_qwc
        return "\n".join(
            [
                f"Measurement plan: {self.source}",
                f"  qubits              : {self.n_qubits}",
                f"  measurable terms    : {self.n_measurable_terms} of {self.n_terms}",
                f"  QWC settings        : {self.n_settings_qwc} "
                f"(grouping factor {self.grouping_factor:.2f}x)",
                f"  largest group       : {self.largest_group} terms",
                f"  shots per setting   : {self.shots_per_setting}",
                f"  shots, grouped      : {self.shots_qwc:,}",
                f"  shots, term by term : {self.shots_term_by_term:,} "
                f"({saved:,} more than grouped)",
                f"  identity offset     : {self.identity_coefficient:.6f}",
                f"  note                : {self.notes[0]}",
            ]
        )


def qwc_groups(terms: list[PauliTerm]) -> list[list[int]]:
    """Group *terms* into qubit-wise commuting sets, returning indices into *terms*.

    Greedy first-fit: each term joins the first group whose accumulated basis it does not
    contradict, otherwise it opens a new one. Two terms share a setting when, on every qubit,
    either they ask for the same basis or one of them asks for nothing.

    Greedy is not optimal, and minimum clique cover is NP-hard, so a better grouper finds fewer
    settings on the same operator. Any such improvement moves the number down, which means the
    count here is an upper bound on the settings a good grouper needs and a fair basis for a
    budget.
    """
    masks: list[dict[int, str]] = []
    members: list[list[int]] = []
    for index, term in enumerate(terms):
        placed = False
        for group_index, mask in enumerate(masks):
            if all(mask.get(q, basis) == basis for q, basis in term.paulis.items()):
                mask.update(term.paulis)
                members[group_index].append(index)
                placed = True
                break
        if not placed:
            masks.append(dict(term.paulis))
            members.append([index])
    return members


def measurement_plan(
    source: Any,
    *,
    shots_per_setting: int = 4096,
) -> MeasurementPlan:
    """Price the measurement of *source* at *shots_per_setting*.

    *source* is a :class:`~qb_compiler.chem.hamiltonian.Hamiltonian` or anything
    :func:`~qb_compiler.chem.hamiltonian.parse_hamiltonian` accepts.
    """
    if shots_per_setting < 1:
        raise ValueError(f"shots_per_setting must be >= 1, got {shots_per_setting}")

    hamiltonian = source if isinstance(source, Hamiltonian) else parse_hamiltonian(source)
    measurable = hamiltonian.measurable_terms
    if not measurable:
        raise ValueError(
            "operator has no measurable terms: every term is the identity, so there is "
            "nothing to measure and nothing to price"
        )

    groups = qwc_groups(measurable)
    n_measurable = len(measurable)
    n_groups = len(groups)

    return MeasurementPlan(
        n_qubits=hamiltonian.n_qubits,
        n_terms=len(hamiltonian.terms),
        n_measurable_terms=n_measurable,
        n_settings_term_by_term=n_measurable,
        n_settings_qwc=n_groups,
        grouping_factor=round(n_measurable / n_groups, 3),
        largest_group=max(len(g) for g in groups),
        shots_per_setting=shots_per_setting,
        shots_term_by_term=n_measurable * shots_per_setting,
        shots_qwc=n_groups * shots_per_setting,
        identity_coefficient=float(hamiltonian.identity_coefficient.real),
        source=hamiltonian.source,
    )
