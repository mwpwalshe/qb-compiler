---
title: "qb-compiler: preflight, cost and provenance checks for quantum circuit execution"
tags:
  - Python
  - quantum computing
  - compilers
  - Qiskit
  - quantum error correction
  - reproducibility
authors:
  - name: Michael William Perry Walshe
    affiliation: 1
affiliations:
  - name: QubitBoost
    index: 1
date: 15 August 2026
bibliography: paper.bib
---

# Summary

`qb-compiler` is a preflight layer for quantum circuit execution. Before a job is submitted it
answers four questions that a user otherwise answers after the fact, or not at all: is this circuit
viable on this device, what will it cost, which physical qubits were chosen and how old was the
calibration behind that choice, and is the input file well formed.

It is a Python package with a command line interface, registered as a Qiskit [@Javadi-Abhari2024]
transpiler layout plugin and a member of the Qiskit Ecosystem. Transpilation and routing are
Qiskit's; the package adds the layer around them. Every check emits a machine-readable receipt, and
receipts can be signed with a persistent key and verified offline by a third party who has only the
receipt and a public key.

The package covers three input classes. For circuits: a viability estimate with a stated error band,
a cross-vendor cost ranking, and a calibration-aware layout search whose ranked candidates and
scores are exposed rather than hidden behind a single answer. For quantum error correction: an audit
of a Stim [@Gidney2021] detector error model that detects error mechanisms which are identical in
their detectors but carry conflicting logical observable masks, a case where merging by detector
signature alone discards logical information, plus a sizing projection for a surface code memory
experiment and hash-verified loaders for public datasets. For chemistry: five integrity checks on a
qubit Hamiltonian file and a structural measurement bill in settings and shots.

# Statement of need

Quantum processor time is metered, noisy, and slow to obtain, so the expensive mistakes are the ones
made before submission: a circuit too deep to produce signal, a run priced on the wrong device, a
layout chosen from calibration data hours out of date, or an input file whose declared metadata
contradicts itself.

Tooling for the compilation step itself is mature. Qiskit, tket [@Sivarajah2020] and BQSKit
[@Younis2021] optimise circuits, and calibration-aware layout selection is available through
`VF2PostLayout` in Qiskit and through mapomatic [@Nation2023]. What is thinner is the layer around
that step: what a run will cost across vendors before it is submitted, whether the inputs are
self-consistent, and what evidence survives the run afterwards.

The last of those is the one that motivated this package. A compiler that silently used hour-old
calibration data and one that used fresh data produce indistinguishable output, and neither reports
which it was. Reproducing a published quantum result then requires information that was never
recorded. `qb-compiler` records it: the layout that ran, the calibration fingerprint and its
measured age, the tool versions, and, where the user chooses, a signature over all of it.

Input validation is the same argument applied earlier in the pipeline. The Hamiltonian audit exists
because a qubit count that disagrees with its own declared active space is invisible to every
downstream tool and invalidates whatever is run on it. The detector error model audit exists because
a canonicalisation that merges mechanisms by detector signature alone can discard logical
information and shift a measured logical error rate. Neither check is novel mathematics. Both are
arithmetic that nothing in the usual pipeline performs.

The package is deliberate about the limits of what it reports. A shot count from the measurement
plan is a structural count and carries no variance weighting or precision claim. A calibration
staleness tolerance is a fixed default and every receipt says so rather than presenting it as a
measurement of the device. A projected logical error rate is decoded with matching and the result
states that whether matching is faithful on that error model was not assessed. Refusing to state a
number a tool did not measure is the design rule the package is built to, and it is the reason its
output is usable as evidence.

# Software description

Installation is `pip install qb-compiler`. The CLI exposes `preflight`, `analyze`, `diff`, `when`,
`compile`, `verify`, `doctor`, `backends`, `info`, `dem-audit`, `dem-canonicalize`, `chem-audit`,
`measure-plan`, `verify-receipt`, `corpus` and `calibration show`; every command that produces a
verdict has documented exit codes for use in continuous integration, and a composite GitHub Action
wraps the input checks. The Python API mirrors the CLI. Calibration data can come from a bundled
snapshot or from a vendor API, with providers for IBM, Amazon Braket, Azure Quantum and Quantinuum
behind optional extras. Signing and verification use Ed25519 and fall back to a pure Python
implementation when a compiled one is unavailable, so verification never requires more than the base
install.

The software is Apache 2.0 licensed, tested against Python 3.10 to 3.12 and Qiskit 1.4 and 2.3, and
documented at the project site with executable notebooks covering each capability.

# Acknowledgements

The package builds on Qiskit for transpilation and routing, on Stim and PyMatching
[@Higgott2025] for the error correction paths, and on rustworkx [@Treinish2022] for graph search.

# References
