# SPDX-License-Identifier: Apache-2.0
"""Five integrity checks on a qubit Hamiltonian, run before anything is submitted.

Every check is arithmetic on the file's own declared metadata. Nothing here computes an energy,
ranks a method, or says whether a run is worth doing. It says whether the file contradicts itself.

Why the fifth check exists
--------------------------
A generator wrote ``n_qubits = n_active_orbitals`` where it should have written
``2 * n_active_orbitals``, so a file describing an 8 electron, 8 orbital active space carried a
16 orbital operator's worth of chemistry in an 8 qubit label. Nothing downstream noticed, because
nothing downstream checks, and the file carried a confident name the whole time. That class of
defect is invisible to every tool in the stack and it invalidates whatever was run on it.

Verdicts
--------
``ACCEPT`` every check passed. ``REFUSE`` at least one check failed. ``INCOMPLETE`` nothing failed
but the file did not declare enough to check.

``INCOMPLETE`` is a real third answer and not a soft pass. A file from another group's pipeline
usually declares less than ours does, and calling that a failure would make this a tool for our
own files, while calling it a pass would be an assurance nobody measured. Pass ``strict=True``,
which is what the CLI's ``--strict`` and the CI action do, to treat an undeclared field as a
failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qb_compiler.chem.hamiltonian import Hamiltonian, load_hamiltonian, parse_hamiltonian

PASS = "PASS"
FAIL = "FAIL"
NOT_DECLARED = "NOT_DECLARED"

ACCEPT = "ACCEPT"
REFUSE = "REFUSE"
INCOMPLETE = "INCOMPLETE"

SCHEMA = "qb.chem_audit.v1"

CHECK_NAMES = (
    "reference method is correlated",
    "exact energy sits below RHF",
    "provenance chain reaches a correlated solve",
    "not flagged synthetic",
    "qubit count matches the active space",
)

#: Post-Hartree-Fock methods, i.e. those that recover correlation energy. A reference energy from
#: anything outside this list is a mean-field number, and comparing a variational result against
#: it measures the basis set rather than the algorithm.
CORRELATED_METHODS = ("FCI", "CASCI", "CASSCF", "CCSD", "CCSD(T)", "SHCI", "DMRG", "MRCI", "SCI")

#: Markers that a provenance chain names a qubit mapping. Without one, a fermionic operator never
#: became a qubit operator, whatever the file says it holds.
_MAPPING_MARKERS = (
    "JW",
    "JORDAN_WIGNER",
    "BK",
    "BRAVYI_KITAEV",
    "PARITY",
)

#: Mappings under which one spin orbital costs one qubit, so the active space fixes the width.
_ONE_QUBIT_PER_SPIN_ORBITAL = ("JORDAN_WIGNER", "JW", "BRAVYI_KITAEV", "BK", "PARITY")


@dataclass(frozen=True)
class IntegrityCheck:
    """One check and what it found.

    ``status`` is ``PASS``, ``FAIL``, or ``NOT_DECLARED``. ``detail`` always states the values the
    verdict was reached on, so a refusal can be argued with.
    """

    name: str
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class IntegrityVerdict:
    """The outcome of auditing one Hamiltonian file."""

    verdict: str
    source: str
    checks: list[IntegrityCheck] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    strict: bool = False

    @property
    def accepted(self) -> bool:
        return self.verdict == ACCEPT

    def failures(self) -> list[str]:
        return [c.name for c in self.checks if c.status == FAIL]

    def undeclared(self) -> list[str]:
        return [c.name for c in self.checks if c.status == NOT_DECLARED]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "verdict": self.verdict,
            "source": self.source,
            "strict": self.strict,
            "checks": [c.as_dict() for c in self.checks],
            "facts": dict(self.facts),
            "signature": None,
            "signing": "unsigned",
        }

    def __str__(self) -> str:
        symbol = {PASS: "pass", FAIL: "FAIL", NOT_DECLARED: "not declared"}
        lines = [f"{self.verdict}: {self.source}"]
        for check in self.checks:
            lines.append(f"  [{symbol[check.status]:>12}] {check.name}")
            lines.append(f"                 {check.detail}")
        if self.verdict == REFUSE:
            lines.append(f"  refused on: {', '.join(self.failures())}")
        elif self.verdict == INCOMPLETE:
            lines.append(f"  not declared: {', '.join(self.undeclared())}")
        return "\n".join(lines)


def _check_reference_method(meta: dict[str, Any]) -> IntegrityCheck:
    kind = meta.get("reference_energy_type")
    if kind is None:
        return IntegrityCheck(
            CHECK_NAMES[0], NOT_DECLARED, "no reference_energy_type field in the file"
        )
    text = str(kind).upper()
    matched = [m for m in CORRELATED_METHODS if m in text]
    if matched:
        return IntegrityCheck(
            CHECK_NAMES[0], PASS, f"reference_energy_type = {kind!r}, names {matched[0]}"
        )
    return IntegrityCheck(
        CHECK_NAMES[0],
        FAIL,
        f"reference_energy_type = {kind!r} is not a correlated method; expected one of "
        f"{', '.join(CORRELATED_METHODS)}",
    )


def _check_exact_below_rhf(meta: dict[str, Any]) -> IntegrityCheck:
    exact = meta.get("reference_exact_energy")
    rhf = meta.get("reference_rhf_energy")
    if exact is None or rhf is None:
        missing = [
            name
            for name, value in (
                ("reference_exact_energy", exact),
                ("reference_rhf_energy", rhf),
            )
            if value is None
        ]
        return IntegrityCheck(CHECK_NAMES[1], NOT_DECLARED, f"missing {' and '.join(missing)}")
    correlation = float(exact) - float(rhf)
    detail = f"exact {exact} vs RHF {rhf}, correlation {correlation:+.6f} Ha"
    if correlation < 0:
        return IntegrityCheck(CHECK_NAMES[1], PASS, detail)
    if correlation == 0:
        return IntegrityCheck(
            CHECK_NAMES[1],
            FAIL,
            f"{detail}; a reference energy identical to the mean field one means no correlation "
            "energy is present, which is the signature of a synthetic or placeholder record",
        )
    return IntegrityCheck(
        CHECK_NAMES[1],
        FAIL,
        f"{detail}; a variational reference cannot sit above the mean field energy",
    )


def _check_provenance(meta: dict[str, Any]) -> IntegrityCheck:
    raw = meta.get("provenance")
    if raw is None:
        return IntegrityCheck(CHECK_NAMES[2], NOT_DECLARED, "no provenance field in the file")
    text = " -> ".join(str(stage) for stage in raw) if isinstance(raw, list | tuple) else str(raw)
    upper = text.upper().replace("→", "->")
    has_solve = [m for m in CORRELATED_METHODS if m in upper]
    has_mapping = [m for m in _MAPPING_MARKERS if m in upper]
    if has_solve and has_mapping:
        return IntegrityCheck(
            CHECK_NAMES[2],
            PASS,
            f"provenance {text!r} names {has_solve[0]} and the {has_mapping[0]} mapping",
        )
    missing = []
    if not has_solve:
        missing.append("a correlated solve")
    if not has_mapping:
        missing.append("a qubit mapping")
    return IntegrityCheck(
        CHECK_NAMES[2],
        FAIL,
        f"provenance {text!r} does not name {' or '.join(missing)}",
    )


def _check_not_synthetic(meta: dict[str, Any]) -> IntegrityCheck:
    flag = meta.get("synthetic")
    if flag is None:
        return IntegrityCheck(
            CHECK_NAMES[3],
            NOT_DECLARED,
            "no synthetic flag in the file; absence is not evidence either way",
        )
    if bool(flag) is False:
        return IntegrityCheck(CHECK_NAMES[3], PASS, "metadata.synthetic = False")
    return IntegrityCheck(
        CHECK_NAMES[3],
        FAIL,
        f"metadata.synthetic = {flag!r}: the file says so itself",
    )


def _check_qubit_count(hamiltonian: Hamiltonian) -> IntegrityCheck:
    meta = hamiltonian.metadata
    space = meta.get("active_space") or {}
    n_orbitals = space.get("n_orbitals") if isinstance(space, dict) else None
    if n_orbitals is None:
        n_orbitals = meta.get("n_orbitals")
    declared = meta.get("n_qubits") if hamiltonian.n_qubits_declared else None

    widest = -1
    for term in hamiltonian.terms:
        for qubit in term.paulis:
            widest = max(widest, qubit)
    operator_width = widest + 1

    if declared is not None and operator_width > int(declared):
        return IntegrityCheck(
            CHECK_NAMES[4],
            FAIL,
            f"the operator acts on qubit {widest}, past the declared n_qubits {declared}",
        )

    if n_orbitals is None:
        return IntegrityCheck(
            CHECK_NAMES[4],
            NOT_DECLARED,
            f"no active space declared; the operator itself is {operator_width} qubits wide",
        )
    if declared is None:
        return IntegrityCheck(
            CHECK_NAMES[4],
            NOT_DECLARED,
            f"no n_qubits declared; active space is {n_orbitals} orbitals and the operator is "
            f"{operator_width} qubits wide",
        )

    mapping = str(meta.get("mapping") or "").upper().replace("-", "_")
    expected = 2 * int(n_orbitals)
    if mapping and not any(marker in mapping for marker in _ONE_QUBIT_PER_SPIN_ORBITAL):
        return IntegrityCheck(
            CHECK_NAMES[4],
            NOT_DECLARED,
            f"mapping {mapping!r} is not one this check knows the qubit cost of; declared "
            f"n_qubits {declared}, active space {n_orbitals} orbitals",
        )
    if int(declared) == expected:
        return IntegrityCheck(
            CHECK_NAMES[4],
            PASS,
            f"n_qubits {declared} = 2 x {n_orbitals} orbitals",
        )
    return IntegrityCheck(
        CHECK_NAMES[4],
        FAIL,
        f"n_qubits {declared} but the active space needs {expected} (2 x {n_orbitals} orbitals). "
        "A spin orbital costs one qubit under this mapping, so a file declaring one qubit per "
        "spatial orbital is describing half the space it claims",
    )


def audit_hamiltonian(
    source: Any,
    *,
    metadata: dict[str, Any] | None = None,
    strict: bool = False,
    label: str | None = None,
) -> IntegrityVerdict:
    """Run the five checks over *source*.

    Parameters
    ----------
    source :
        A :class:`~qb_compiler.chem.hamiltonian.Hamiltonian`, or anything
        :func:`~qb_compiler.chem.hamiltonian.parse_hamiltonian` accepts.
    metadata :
        Extra declarations to audit alongside the file's own, for a format that carries the
        operator and its provenance separately.
    strict :
        Treat ``NOT_DECLARED`` as a failure. What CI should use.
    label :
        Name for the audited object in the verdict, defaulting to its source path.
    """
    hamiltonian = (
        source
        if isinstance(source, Hamiltonian)
        else parse_hamiltonian(source, metadata=metadata, source=label or "in memory")
    )
    if metadata and isinstance(source, Hamiltonian):
        merged = dict(hamiltonian.metadata)
        merged.update(metadata)
        hamiltonian = Hamiltonian(
            n_qubits=hamiltonian.n_qubits,
            terms=hamiltonian.terms,
            metadata=merged,
            source=hamiltonian.source,
            n_qubits_declared=hamiltonian.n_qubits_declared,
        )

    meta = hamiltonian.metadata
    checks = [
        _check_reference_method(meta),
        _check_exact_below_rhf(meta),
        _check_provenance(meta),
        _check_not_synthetic(meta),
        _check_qubit_count(hamiltonian),
    ]

    failed = any(c.status == FAIL for c in checks)
    undeclared = any(c.status == NOT_DECLARED for c in checks)
    if failed or (strict and undeclared):
        verdict = REFUSE
    elif undeclared:
        verdict = INCOMPLETE
    else:
        verdict = ACCEPT

    facts: dict[str, Any] = {
        "name": meta.get("name") or meta.get("id"),
        "n_qubits": hamiltonian.n_qubits,
        "n_qubits_declared": hamiltonian.n_qubits_declared,
        "n_terms": len(hamiltonian.terms),
        "n_measurable_terms": len(hamiltonian.measurable_terms),
        "basis": meta.get("basis"),
        "mapping": meta.get("mapping"),
        "active_space": meta.get("active_space"),
        "reference_energy_type": meta.get("reference_energy_type"),
        "reference_rhf_energy": meta.get("reference_rhf_energy"),
        "reference_exact_energy": meta.get("reference_exact_energy"),
    }

    return IntegrityVerdict(
        verdict=verdict,
        source=label or hamiltonian.source,
        checks=checks,
        facts=facts,
        strict=strict,
    )


def audit_hamiltonian_file(path: str | Path, *, strict: bool = False) -> IntegrityVerdict:
    """Load a Hamiltonian from *path* and audit it."""
    hamiltonian = load_hamiltonian(path)
    return audit_hamiltonian(hamiltonian, strict=strict, label=str(Path(path).expanduser()))
