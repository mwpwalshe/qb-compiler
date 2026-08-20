# SPDX-License-Identifier: Apache-2.0
"""Pre-submit checks on a chemistry input, before a shot is bought.

Two things live here, and both stay on the near side of a line worth stating.

:mod:`~qb_compiler.chem.audit` reads a Hamiltonian file and refuses it when its own declared
metadata contradicts itself. :mod:`~qb_compiler.chem.measure` counts what measuring that operator
costs: terms, qubit-wise commuting settings, and shots at a rate you choose.

Neither computes a chemistry result, ranks anything, or says whether a run is worth believing.
They tell you whether the input is well formed and what the run costs. That is the whole scope,
and it is deliberate.
"""

from __future__ import annotations

from qb_compiler.chem.audit import (
    CHECK_NAMES,
    IntegrityCheck,
    IntegrityVerdict,
    audit_hamiltonian,
    audit_hamiltonian_file,
)
from qb_compiler.chem.hamiltonian import (
    Hamiltonian,
    HamiltonianFormatError,
    PauliTerm,
    load_hamiltonian,
    parse_hamiltonian,
)
from qb_compiler.chem.measure import (
    MeasurementPlan,
    measurement_plan,
    qwc_groups,
)

__all__ = [
    "CHECK_NAMES",
    "Hamiltonian",
    "HamiltonianFormatError",
    "IntegrityCheck",
    "IntegrityVerdict",
    "MeasurementPlan",
    "PauliTerm",
    "audit_hamiltonian",
    "audit_hamiltonian_file",
    "load_hamiltonian",
    "measurement_plan",
    "parse_hamiltonian",
    "qwc_groups",
]
