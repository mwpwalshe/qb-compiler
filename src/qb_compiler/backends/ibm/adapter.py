"""IBM backend adapter wrapping BackendTarget with IBM-specific defaults."""

from __future__ import annotations

from qb_compiler.backends.base import BackendTarget
from qb_compiler.backends.ibm.native_gates import (
    IBM_EAGLE_BASIS,
    IBM_HERON_BASIS,
    IBM_TYPICAL_1Q_ERROR,
    IBM_TYPICAL_2Q_ERROR,
    IBM_TYPICAL_READOUT_ERROR,
)

# Well-known IBM backend names and their processor families
_IBM_HERON_BACKENDS = frozenset({
    "ibm_fez",
    "ibm_torino",
    "ibm_marrakesh",
    "ibm_brisbane",
    "ibm_kyoto",
})

_IBM_EAGLE_BACKENDS = frozenset({
    "ibm_sherbrooke",
    "ibm_nazca",
    "ibm_cusco",
})


def ibm_backend_target(
    name: str,
    n_qubits: int,
    coupling_map: list[tuple[int, int]] | None = None,
    *,
    processor_family: str | None = None,
) -> BackendTarget:
    """Create a :class:`BackendTarget` with IBM-specific defaults.

    Parameters
    ----------
    name:
        Backend identifier (e.g. ``"ibm_fez"``).
    n_qubits:
        Number of physical qubits.
    coupling_map:
        Directed coupling pairs.  If *None*, an empty list is used
        (all-to-all, which is incorrect for IBM but allows basic usage).
    processor_family:
        ``"heron"`` or ``"eagle"``.  If *None*, inferred from *name*.

    Returns
    -------
    BackendTarget
        Configured for IBM hardware.
    """
    if processor_family is None:
        processor_family = _infer_processor_family(name)

    basis = IBM_HERON_BASIS if processor_family == "heron" else IBM_EAGLE_BASIS

    return BackendTarget(
        n_qubits=n_qubits,
        basis_gates=basis,
        coupling_map=coupling_map or [],
        name=name,
    )


def _infer_processor_family(name: str) -> str:
    """Infer IBM processor family from backend name."""
    lower = name.lower()
    if lower in _IBM_EAGLE_BACKENDS:
        return "eagle"
    # Default to Heron for known Heron backends and unknown IBM backends
    return "heron"


# Convenience: pre-built error budget defaults for IBM backends
IBM_DEFAULT_ERROR_BUDGET = {
    "typical_1q_error": IBM_TYPICAL_1Q_ERROR,
    "typical_2q_error": IBM_TYPICAL_2Q_ERROR,
    "typical_readout_error": IBM_TYPICAL_READOUT_ERROR,
}
