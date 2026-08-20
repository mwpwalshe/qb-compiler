# SPDX-License-Identifier: Apache-2.0
"""Reading a qubit Hamiltonian out of whatever shape it arrived in.

A chemistry group's operator is on disk in one of several layouts and none of them is a standard.
This module normalises the ones that show up in practice into one small record, so the audit and
the measurement plan never have to know where the file came from.

Accepted inputs
---------------
1. **Grouped JSON**, the layout used by the QubitBoost chemistry corpus::

     {
       "metadata": {
         "n_qubits": 16,
         "active_space": {"n_electrons": 8, "n_orbitals": 8},
         "provenance": "RDKit -> PySCF -> OpenFermion -> JW -> sparse_FCI",
         "synthetic": false,
         "mapping": "jordan_wigner"
       },
       "pauli_groups": [
         {"pauli_strings": ["I"],  "coefficients": [-503.72], "measurable": false},
         {"pauli_strings": ["Z0"], "coefficients": [0.2109],  "measurable": true}
       ],
       "reference_energy_type": "FCI_sparse",
       "reference_rhf_energy": -505.8517767880335,
       "reference_exact_energy": -505.9147257008191
     }

2. **Flat term list**, the shape most tools export::

     {"n_qubits": 4, "terms": [{"pauli": "Z0 Z1", "coefficient": 0.17}, ...]}

   ``terms`` also accepts ``[["Z0 Z1", 0.17], ...]`` and ``{"Z0 Z1": 0.17, ...}``.

3. **OpenFermion**: a ``QubitOperator``, or a ``MolecularData``-style record whose terms live
   under ``terms`` keyed by ``((index, letter), ...)`` tuples.

4. **Qiskit**: a ``SparsePauliOp``, optionally with a metadata block passed alongside it.

Pauli strings are read in both common spellings: sparse (``"Z0 X3"``) and dense (``"ZIIX"``,
left to right from qubit 0). Identity is ``"I"`` or an empty string.

Nothing here evaluates, simulates, or transforms the operator. It reads it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PAULIS = ("I", "X", "Y", "Z")
_SPARSE_TOKEN = re.compile(r"^([XYZI])(\d+)$", re.IGNORECASE)


class HamiltonianFormatError(ValueError):
    """Raised when an input cannot be read as a qubit Hamiltonian."""


@dataclass(frozen=True)
class PauliTerm:
    """One Pauli term: which qubit carries which single-qubit Pauli, and its coefficient.

    ``paulis`` maps qubit index to one of ``X``, ``Y``, ``Z``. Identity factors are absent rather
    than stored as ``I``, so an empty mapping is the identity term.
    """

    paulis: dict[int, str]
    coefficient: complex

    @property
    def is_identity(self) -> bool:
        return not self.paulis

    @property
    def weight(self) -> int:
        """Number of qubits this term acts on non-trivially."""
        return len(self.paulis)

    def commutes_qubitwise(self, other: PauliTerm) -> bool:
        """True when both terms can be read from the same measurement setting."""
        for qubit, basis in self.paulis.items():
            theirs = other.paulis.get(qubit)
            if theirs is not None and theirs != basis:
                return False
        return True


@dataclass(frozen=True)
class Hamiltonian:
    """A qubit Hamiltonian plus whatever provenance its file declared.

    Attributes
    ----------
    n_qubits :
        Qubit count. Taken from the file when declared, otherwise inferred from the highest
        qubit index that appears, and ``n_qubits_declared`` records which of the two happened.
    terms :
        Every term, identity included.
    metadata :
        The file's own metadata block, unaltered. The audit reads its declarations; nothing here
        rewrites or completes them.
    source :
        Where it was read from, for the receipt.
    n_qubits_declared :
        ``True`` when the file stated its own qubit count. The audit's qubit-count check is only
        meaningful when it did.
    """

    n_qubits: int
    terms: list[PauliTerm]
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "in memory"
    n_qubits_declared: bool = True

    @property
    def measurable_terms(self) -> list[PauliTerm]:
        """Every term except the identity, which is a constant offset and needs no shots."""
        return [t for t in self.terms if not t.is_identity]

    @property
    def identity_coefficient(self) -> complex:
        return sum((t.coefficient for t in self.terms if t.is_identity), 0j)


def _parse_pauli_string(text: str, n_qubits: int | None) -> dict[int, str]:
    """Read one Pauli string in either the sparse or the dense spelling."""
    cleaned = text.strip()
    if not cleaned or cleaned.upper() == "I":
        return {}

    tokens = cleaned.split()
    if all(_SPARSE_TOKEN.match(tok) for tok in tokens):
        out: dict[int, str] = {}
        for tok in tokens:
            match = _SPARSE_TOKEN.match(tok)
            if match is None:  # pragma: no cover - guarded by the all() above
                raise HamiltonianFormatError(f"unreadable Pauli token {tok!r}")
            letter, index = match.group(1).upper(), int(match.group(2))
            if letter != "I":
                out[index] = letter
        return out

    dense = cleaned.upper()
    if len(tokens) == 1 and all(ch in _PAULIS for ch in dense):
        if n_qubits is not None and len(dense) != n_qubits:
            raise HamiltonianFormatError(
                f"dense Pauli string {text!r} has {len(dense)} characters on a "
                f"{n_qubits} qubit operator"
            )
        return {i: ch for i, ch in enumerate(dense) if ch != "I"}

    raise HamiltonianFormatError(f"unreadable Pauli string {text!r}")


def _terms_from_pairs(pairs: Any, n_qubits: int | None) -> list[PauliTerm]:
    terms: list[PauliTerm] = []
    for entry in pairs:
        if isinstance(entry, dict):
            label = entry.get("pauli", entry.get("label", entry.get("string")))
            coeff = entry.get("coefficient", entry.get("coeff", entry.get("value")))
        elif isinstance(entry, list | tuple) and len(entry) == 2:
            label, coeff = entry
        else:
            raise HamiltonianFormatError(f"unreadable term entry {entry!r}")
        if label is None or coeff is None:
            raise HamiltonianFormatError(f"term entry {entry!r} is missing a Pauli or coefficient")
        terms.append(
            PauliTerm(paulis=_parse_pauli_string(str(label), n_qubits), coefficient=complex(coeff))
        )
    return terms


def _terms_from_openfermion_keys(mapping: dict[Any, Any]) -> list[PauliTerm]:
    """Read ``{((0, 'Z'), (1, 'X')): coeff}``, the OpenFermion ``QubitOperator.terms`` shape."""
    terms: list[PauliTerm] = []
    for key, coeff in mapping.items():
        paulis: dict[int, str] = {}
        for factor in key:
            index, letter = factor
            letter = str(letter).upper()
            if letter not in _PAULIS:
                raise HamiltonianFormatError(f"unreadable Pauli factor {factor!r}")
            if letter != "I":
                paulis[int(index)] = letter
        terms.append(PauliTerm(paulis=paulis, coefficient=complex(coeff)))
    return terms


def _n_qubits_from_terms(terms: list[PauliTerm]) -> int:
    highest = -1
    for term in terms:
        for qubit in term.paulis:
            highest = max(highest, qubit)
    return highest + 1


def parse_hamiltonian(
    data: Any,
    *,
    metadata: dict[str, Any] | None = None,
    source: str = "in memory",
) -> Hamiltonian:
    """Read *data* as a :class:`Hamiltonian`. See the module docstring for accepted shapes."""
    # Qiskit SparsePauliOp, duck-typed so qiskit is not imported for a file that is not one.
    if hasattr(data, "paulis") and hasattr(data, "coeffs"):
        labels = [str(p) for p in data.paulis]
        coeffs = list(data.coeffs)
        n_qubits = int(getattr(data, "num_qubits", len(labels[0]) if labels else 0))
        terms = [
            PauliTerm(paulis=_parse_pauli_string(label[::-1], n_qubits), coefficient=complex(c))
            for label, c in zip(labels, coeffs, strict=True)
        ]
        return Hamiltonian(
            n_qubits=n_qubits,
            terms=terms,
            metadata=dict(metadata or {}),
            source=source,
        )

    # OpenFermion QubitOperator
    if hasattr(data, "terms") and not isinstance(data, dict):
        terms = _terms_from_openfermion_keys(dict(data.terms))
        return Hamiltonian(
            n_qubits=_n_qubits_from_terms(terms),
            terms=terms,
            metadata=dict(metadata or {}),
            source=source,
            n_qubits_declared=False,
        )

    if not isinstance(data, dict):
        raise HamiltonianFormatError(f"cannot read a Hamiltonian from {type(data).__name__}")

    meta = dict(data.get("metadata") or {})
    if metadata:
        meta.update(metadata)

    declared = meta.get("n_qubits", data.get("n_qubits"))
    declared_qubits: int | None = int(declared) if declared is not None else None

    if "pauli_groups" in data:
        terms = []
        for group in data["pauli_groups"]:
            strings = group.get("pauli_strings") or []
            coeffs = group.get("coefficients") or []
            if len(strings) != len(coeffs):
                raise HamiltonianFormatError(
                    f"group {group.get('id')} has {len(strings)} Pauli strings and "
                    f"{len(coeffs)} coefficients"
                )
            for label, coeff in zip(strings, coeffs, strict=True):
                terms.append(
                    PauliTerm(
                        paulis=_parse_pauli_string(str(label), declared_qubits),
                        coefficient=complex(coeff),
                    )
                )
    elif "terms" in data:
        raw = data["terms"]
        if isinstance(raw, dict):
            keys = list(raw)
            if keys and isinstance(keys[0], tuple):
                terms = _terms_from_openfermion_keys(raw)
            else:
                terms = _terms_from_pairs(list(raw.items()), declared_qubits)
        else:
            terms = _terms_from_pairs(raw, declared_qubits)
    elif "paulis" in data and "coefficients" in data:
        terms = _terms_from_pairs(
            list(zip(data["paulis"], data["coefficients"], strict=True)), declared_qubits
        )
    else:
        raise HamiltonianFormatError(
            "no Pauli terms found: expected one of 'pauli_groups', 'terms', or "
            "'paulis' plus 'coefficients'"
        )

    # Reference energies sit at the top level in the grouped layout; keep them with the metadata
    # so the audit reads one record rather than two.
    for key in (
        "reference_energy_type",
        "reference_rhf_energy",
        "reference_exact_energy",
        "correlation_energy_mHa",
    ):
        if key in data and key not in meta:
            meta[key] = data[key]

    return Hamiltonian(
        n_qubits=declared_qubits if declared_qubits is not None else _n_qubits_from_terms(terms),
        terms=terms,
        metadata=meta,
        source=source,
        n_qubits_declared=declared_qubits is not None,
    )


def load_hamiltonian(path: str | Path) -> Hamiltonian:
    """Read a Hamiltonian from a JSON file on disk."""
    file_path = Path(path).expanduser()
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HamiltonianFormatError(f"cannot read {file_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HamiltonianFormatError(f"{file_path} is not valid JSON: {exc}") from exc
    return parse_hamiltonian(data, source=str(file_path))
