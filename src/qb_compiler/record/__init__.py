# SPDX-License-Identifier: Apache-2.0
"""Record construction checks, correct loaders, and the residual metric.

Before a decoder result means anything, the record it decoded has to be the record the experiment
produced. Rounds in the order they happened, readouts referenced to a frame, detectors differenced
against the right neighbour, the error model that belongs to the run, the observable the labels
describe. None of that announces itself when it is wrong: a record stored last round first decodes
without error and reports a logical error rate two to seven times too high, and a record built by
differencing unframed readouts reports one that is pure noise. Both look ordinary in every plot.

This package holds eight checks for those conditions, loaders that get two real records right, the
``.npz`` contract for everything else, and one measurement.

What a passing record is, and what it is not
--------------------------------------------
A record that passes these checks is **correctly built**. That is all it is. It is not complete,
and nothing here says it is: a correctly built record can still carry structure that the decoder
reading it does not use. The QuEra d5 Z release passes every check that can run on it, and 0.02
bits per shot of what a matching decoder on the published model gets wrong is still anticipated by
a summary the decoder did not consult.

The residual metric measures exactly that quantity and reports it against a floor built from
records whose structure has been destroyed and whose counts have not. It is a comparison against
one decoder's own output. It is not a decoder benchmark, it does not say a better decoder exists,
and a number at the floor does not say the record holds nothing. The report states the number, the
floor, and whether one exceeds the other; it does not read anything into either.

Usage::

    from qb_compiler.record import validate, residual
    from qb_compiler.record.loaders import ibm_fez_repetition

    spec = ibm_fez_repetition.load(root, d=11, r=11)
    report = validate(spec)
    print(report)

Needs the ``record`` extra for the parts that decode or fit: ``pip install 'qb-compiler[record]'``.
"""

from __future__ import annotations

from qb_compiler.record.residual import residual
from qb_compiler.record.types import (
    FAIL,
    FIRST_LAYER_PREMISE,
    PASS,
    SKIP,
    Check,
    RecordDem,
    RecordSpec,
    ResidualReport,
    ValidationReport,
)
from qb_compiler.record.validate import DEFAULT_THRESHOLDS, validate

__all__ = [
    "DEFAULT_THRESHOLDS",
    "FAIL",
    "FIRST_LAYER_PREMISE",
    "PASS",
    "SKIP",
    "Check",
    "RecordDem",
    "RecordSpec",
    "ResidualReport",
    "ValidationReport",
    "residual",
    "validate",
]
