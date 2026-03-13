"""Backend target definitions and vendor-specific support."""

from __future__ import annotations

from qb_compiler.backends.base import BackendTarget
from qb_compiler.backends.ibm import (
    IBM_EAGLE_BASIS,
    IBM_HERON_BASIS,
    IBM_TYPICAL_GATE_TIMES_NS,
    load_ibm_calibration,
    parse_ibm_calibration,
)
from qb_compiler.backends.ibm.adapter import ibm_backend_target
from qb_compiler.backends.ionq import (
    IONQ_ARIA_BASIS,
    IONQ_TYPICAL_GATE_TIMES_NS,
    load_ionq_calibration,
    parse_ionq_calibration,
)
from qb_compiler.backends.iqm import (
    IQM_GARNET_BASIS,
    IQM_TYPICAL_GATE_TIMES_NS,
    load_iqm_calibration,
    parse_iqm_calibration,
)
from qb_compiler.backends.rigetti import (
    RIGETTI_ANKAA_BASIS,
    RIGETTI_TYPICAL_GATE_TIMES_NS,
    load_rigetti_calibration,
    parse_rigetti_calibration,
)

__all__ = [
    "IBM_EAGLE_BASIS",
    "IBM_HERON_BASIS",
    "IBM_TYPICAL_GATE_TIMES_NS",
    "IONQ_ARIA_BASIS",
    "IONQ_TYPICAL_GATE_TIMES_NS",
    "IQM_GARNET_BASIS",
    "IQM_TYPICAL_GATE_TIMES_NS",
    "RIGETTI_ANKAA_BASIS",
    "RIGETTI_TYPICAL_GATE_TIMES_NS",
    "BackendTarget",
    "ibm_backend_target",
    "load_ibm_calibration",
    "load_ionq_calibration",
    "load_iqm_calibration",
    "load_rigetti_calibration",
    "parse_ibm_calibration",
    "parse_ionq_calibration",
    "parse_iqm_calibration",
    "parse_rigetti_calibration",
]
