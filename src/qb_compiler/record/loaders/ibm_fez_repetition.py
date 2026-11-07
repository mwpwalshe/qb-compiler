# SPDX-License-Identifier: Apache-2.0
"""Loader for the IBM repetition-code memory records of Samuel Stein's public release.

The records are a public dataset: Samuel Stein, "Calibration-Conditioned FiLM Decoders for
Low-Latency Decoding of Quantum Error Correction Evaluated on IBM Repetition-Code Experiments -
Datasets", Zenodo, 2026, doi:10.5281/zenodo.20768087, licensed CC BY 4.0. The ``ibm_fez`` records
come in ``ibm_fez.tar.gz`` of that record, and the ibm_kingston and ibm_pittsburgh archives share
the layout. Nothing from the release is redistributed here: this reads the layout of a copy you
fetched. Cite the record if you use it.

Layout this reads: one directory per regime, named ``d{d}_r{rounds}``, holding one or more
``job_*`` directories. Each job has an ``info.json`` recording ``d``, ``rounds``, ``basis``,
``logical_states``, ``shots`` and ``n_chains``, and a ``bitstrings.json`` holding one entry per
submitted circuit with ``metadata.logical_state`` and ``per_shot_cregs``. The registers come in
pairs: one ``c_data_<chain>`` holding the final data readout and one ``c_syndrome_<chain>``
holding every stabilizer round, for each chain packed into the circuit.

The bit order
-------------
The circuits fill ``c_syndrome_<chain>`` in index order, round after round, and
``per_shot_cregs`` lists each register's bits **highest index first**, which is the order Qiskit
prints a classical register in. Read as listed, the rounds come **last round first**. That is
Qiskit's convention, not a fault in the release. Reshaping the register to
``(shots, rounds, d - 1)`` and using it as it comes puts the run backwards in time. Nothing errors.
The detectors build, the decoder runs, and the logical error rate comes out roughly two to seven
times too high: measured on these records, 3.83 percent against 1.62 at d5 r5, and 9.69 against
1.43 at d11 r11.

``reverse_rounds=True``, the default, reverses the round axis and is what these records need. The
data register is listed the same way, so the sites of every round and the data qubits are both
read from the far end of the chain, and the record stays consistent.
``reverse_rounds=False`` is kept so the mirrored orientation can be measured side by side, which
is what the checks in :mod:`qb_compiler.record.validate` were calibrated against. It is not a
mode anyone should decode in.

What comes back is a :class:`~qb_compiler.record.types.RecordSpec` with the detectors, the labels,
one group per chain, and the raw stabilizer block and final parity that the endpoint and state
checks need. The raw block is returned in loader order, so it is reversed too when
``reverse_rounds`` is on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from qb_compiler.record.types import FIRST_LAYER_PREMISE, RecordSpec

LOADER_VERSION = "1.1"


def _job_info(job: Path) -> dict[str, Any]:
    info: dict[str, Any] = json.loads((job / "info.json").read_text(encoding="utf-8"))
    return info


def _pick_job(regime: Path, basis: str) -> Path:
    if not regime.is_dir():
        raise FileNotFoundError(
            f"{regime} is not a directory. Expected a regime directory named d<d>_r<rounds> "
            "holding one or more job_* directories"
        )
    jobs = sorted(regime.glob("job_*"))
    if not jobs:
        raise FileNotFoundError(f"no job_* directory under {regime}")
    matching = [job for job in jobs if _job_info(job).get("basis") == basis]
    if not matching:
        available = sorted({str(_job_info(job).get("basis")) for job in jobs})
        raise FileNotFoundError(
            f"no {basis}-basis job under {regime}; the jobs there are in basis {available}"
        )
    return max(matching, key=lambda job: int(_job_info(job).get("shots", 0)))


def load(
    root: str | Path,
    d: int,
    r: int,
    basis: str = "Z",
    reverse_rounds: bool = True,
) -> RecordSpec:
    """Load one regime of an IBM repetition-code memory record in the Stein release layout.

    Parameters
    ----------
    root :
        Directory holding the ``d{d}_r{r}`` regime directories.
    d, r :
        Chain length and number of stabilizer rounds.
    basis :
        Preparation basis, matched against ``info.json``.
    reverse_rounds :
        Reverse the listed round order. True is correct for these records; see the module
        docstring.

    Returns
    -------
    A :class:`~qb_compiler.record.types.RecordSpec` with ``detectors`` ``(n, r + 1, d - 1)``,
    ``labels`` the observable-flip truth, ``groups`` the chain each shot came from,
    ``raw_syndromes`` and ``final_parity`` in loader order, ``logical_state``, ``data0``, and a
    ``meta`` recording the loader version and the orientation used.
    """
    if d < 2:
        raise ValueError(f"d must be at least 2, got {d}")
    if r < 1:
        raise ValueError(f"r must be at least 1, got {r}")
    regime = Path(root).expanduser() / f"d{d}_r{r}"
    job = _pick_job(regime, basis)
    circuits = json.loads((job / "bitstrings.json").read_text(encoding="utf-8"))

    detector_blocks: list[np.ndarray] = []
    syndrome_blocks: list[np.ndarray] = []
    parity_blocks: list[np.ndarray] = []
    first_data: list[np.ndarray] = []
    states: list[int] = []
    groups: list[str] = []

    for circuit in circuits:
        state = int(circuit["metadata"]["logical_state"])
        registers = circuit["per_shot_cregs"]
        for data_key in [key for key in registers if key.startswith("c_data_")]:
            chain = data_key[len("c_data_") :]
            syndrome_key = "c_syndrome_" + chain
            if syndrome_key not in registers:
                raise ValueError(
                    f"{job}: register {data_key} has no matching {syndrome_key}. Each chain needs "
                    "one data register and one syndrome register"
                )
            data = np.asarray(registers[data_key], dtype=np.uint8)
            if data.ndim != 2 or data.shape[1] != d:
                raise ValueError(f"{job}: {data_key} has shape {data.shape}, expected (shots, {d})")
            syndromes = np.asarray(registers[syndrome_key], dtype=np.uint8)
            if syndromes.size != data.shape[0] * r * (d - 1):
                raise ValueError(
                    f"{job}: {syndrome_key} holds {syndromes.size} values, expected "
                    f"{data.shape[0] * r * (d - 1)} for {data.shape[0]} shots of {r} rounds on "
                    f"{d - 1} sites"
                )
            syndromes = syndromes.reshape(-1, r, d - 1)
            if reverse_rounds:
                syndromes = syndromes[:, ::-1, :]
            syndromes = np.ascontiguousarray(syndromes)

            n_shots = data.shape[0]
            parity = (data[:, :-1] ^ data[:, 1:]).astype(np.uint8)
            detectors = np.empty((n_shots, r + 1, d - 1), dtype=np.uint8)
            detectors[:, 0] = syndromes[:, 0]
            detectors[:, 1:r] = syndromes[:, 1:] ^ syndromes[:, :-1]
            detectors[:, r] = parity ^ syndromes[:, r - 1]

            detector_blocks.append(detectors)
            syndrome_blocks.append(syndromes)
            parity_blocks.append(parity)
            first_data.append(data[:, 0])
            states.extend([state] * n_shots)
            groups.extend([chain] * n_shots)

    if not detector_blocks:
        raise ValueError(f"{job}: no c_data_* register in bitstrings.json")

    data0 = np.concatenate(first_data).astype(np.uint8)
    logical_state = np.asarray(states, dtype=np.uint8)
    spec = RecordSpec(
        detectors=np.concatenate(detector_blocks),
        labels=(data0 ^ logical_state).astype(np.uint8),
        groups=np.asarray(groups),
        raw_syndromes=np.concatenate(syndrome_blocks),
        final_parity=np.concatenate(parity_blocks),
        logical_state=logical_state,
        data0=data0,
        meta={
            "loader": "ibm_fez_repetition",
            "loader_version": LOADER_VERSION,
            "source": (
                "layout of the IBM repetition-code memory records in Samuel Stein's public "
                "release, Zenodo doi:10.5281/zenodo.20768087, CC BY 4.0 (OpenQASM 3 circuits, one "
                "c_data_<chain> and one c_syndrome_<chain> register per chain, info.json with d, "
                "rounds, basis, logical_states, shots, n_chains)"
            ),
            "d": int(d),
            "rounds": int(r),
            "basis": basis,
            "reverse_rounds": bool(reverse_rounds),
            "observable": "data qubit 0",
            "n_chains": len(set(groups)),
            # The release's circuits reset the ancillas before the first stabilizer round, so that
            # round differences against a prepared value and fires below the steady rate: measured
            # 0.52 to 0.57 times the steady median across d5 r5 to d11 r11 of its ibm_fez records
            # in the corrected orientation. The round profile check runs on this platform because
            # of that.
            FIRST_LAYER_PREMISE: {
                "source": (
                    "corrected ibm_fez records of the Stein release (Zenodo "
                    "doi:10.5281/zenodo.20768087), d5 r5 to d11 r11: first-round rate 0.052 to "
                    "0.083 against steady medians 0.099 to 0.145, ratios 0.52 to 0.57"
                )
            },
        },
    )
    return spec


__all__ = ["LOADER_VERSION", "load"]
