"""Backend auto-discovery and PUB-aware preflight.

Walks a runtime service object (duck-typed against
``qiskit_ibm_runtime.QiskitRuntimeService``), records what each
backend reports about itself, and runs the standard viability check
across every discovered backend that can hold the circuit.

Everything here is observational: the module collects and ranks
signals, it never decides whether a job should run.

Usage::

    from qb_compiler.discovery import discover_backends, rank_discovered

    backends = discover_backends(service)
    ranked = rank_discovered(circuit, service, top=3)
    for db, result in ranked:
        print(db.name, result.estimated_fidelity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qb_compiler.viability import ViabilityResult, check_viability

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveredBackend:
    """Snapshot of a backend as reported by its runtime service.

    Attributes
    ----------
    name :
        Backend name (e.g. ``"ibm_fez"``).
    num_qubits :
        Number of qubits the backend reports.
    operational :
        ``True`` if the backend's status call reported it operational.
        ``False`` when the status call failed or reported otherwise.
    pending_jobs :
        Queue depth from the status call (0 if unavailable).
    has_target :
        ``True`` if the backend exposes a Qiskit ``Target``.
    basis_gates :
        Operation names from the backend target, empty if unavailable.
    """

    name: str
    num_qubits: int
    operational: bool
    pending_jobs: int
    has_target: bool
    basis_gates: tuple[str, ...] = ()


def _backend_name(backend: Any) -> str:
    """Resolve a backend's name (attribute on V2, method on V1)."""
    name = getattr(backend, "name", None)
    if callable(name):
        name = name()
    return str(name) if name else "unknown"


def _discover_one(backend: Any) -> DiscoveredBackend:
    """Build a :class:`DiscoveredBackend` snapshot from one backend object."""
    name = _backend_name(backend)
    num_qubits = int(getattr(backend, "num_qubits", 0) or 0)

    operational = False
    pending_jobs = 0
    try:
        status = backend.status()
        operational = bool(getattr(status, "operational", False))
        pending_jobs = int(getattr(status, "pending_jobs", 0) or 0)
    except Exception:  # status endpoints fail in many ways, record and move on
        logger.debug("status() failed for backend %s, recording non-operational", name)

    target = getattr(backend, "target", None)
    basis_gates: tuple[str, ...] = ()
    if target is not None:
        op_names = getattr(target, "operation_names", None)
        if op_names is not None:
            basis_gates = tuple(str(op) for op in op_names)

    return DiscoveredBackend(
        name=name,
        num_qubits=num_qubits,
        operational=operational,
        pending_jobs=pending_jobs,
        has_target=target is not None,
        basis_gates=basis_gates,
    )


def discover_backends(service: Any) -> list[DiscoveredBackend]:
    """Discover backends from a runtime *service*.

    Duck-typed against ``qiskit_ibm_runtime.QiskitRuntimeService``: any
    object with a ``backends()`` method returning backend-like objects
    works, which keeps tests offline with simple stubs.

    Parameters
    ----------
    service :
        Object exposing ``backends() -> iterable``.  Each backend may
        expose ``name``, ``num_qubits``, ``status()``, and ``target``;
        missing pieces are recorded with safe defaults.

    Returns
    -------
    list[DiscoveredBackend]
        One snapshot per backend, in service order.
    """
    return [_discover_one(b) for b in service.backends()]


def rank_discovered(
    circuit: Any,
    service: Any,
    *,
    n_seeds: int = 2,
    top: int = 5,
) -> list[tuple[DiscoveredBackend, ViabilityResult]]:
    """Run viability checks across discovered backends and rank them.

    Every operational discovered backend that exposes a target and has
    enough qubits for *circuit* gets a :func:`check_viability` run
    against its live target.  Backends that fail the check (transpile
    errors, malformed targets) are skipped, not fatal.

    Parameters
    ----------
    circuit :
        Qiskit ``QuantumCircuit`` to assess.
    service :
        Runtime service, see :func:`discover_backends`.
    n_seeds :
        Transpiler seeds per backend (kept low: this runs once per
        backend).
    top :
        Maximum number of ranked entries to return.

    Returns
    -------
    list[tuple[DiscoveredBackend, ViabilityResult]]
        Pairs sorted by ``estimated_fidelity`` descending, at most
        *top* entries.
    """
    ranked: list[tuple[DiscoveredBackend, ViabilityResult]] = []
    skipped: list[str] = []

    for backend in service.backends():
        db = _discover_one(backend)
        target = getattr(backend, "target", None)
        if not db.operational or target is None or db.num_qubits < circuit.num_qubits:
            skipped.append(db.name)
            continue
        try:
            result = check_viability(
                circuit,
                backend=db.name,
                qiskit_target=target,
                n_seeds=n_seeds,
            )
        except Exception:  # one bad backend must not kill the scan
            skipped.append(db.name)
            continue
        ranked.append((db, result))

    if skipped:
        logger.debug("rank_discovered skipped backends: %s", ", ".join(skipped))

    ranked.sort(key=lambda pair: pair[1].estimated_fidelity, reverse=True)
    return ranked[:top]


def check_viability_pub(
    pub: tuple,
    *,
    backend: str | None = None,
    **kwargs: Any,
) -> ViabilityResult:
    """Viability check for a V2-primitives-style PUB tuple.

    Accepts ``(circuit,)``, ``(circuit, observables)``, or
    ``(circuit, observables, shots)`` and forwards the circuit and
    shot count to :func:`check_viability`.  Observables are accepted
    for ergonomics (so a PUB built for ``EstimatorV2`` can be passed
    as-is) but are currently unused by the estimate.

    Parameters
    ----------
    pub :
        PUB tuple.  ``pub[0]`` must be a Qiskit ``QuantumCircuit``;
        ``pub[2]``, when present and not ``None``, is the shot count
        (default 4096).
    backend :
        Target backend name, forwarded to :func:`check_viability`.
    **kwargs :
        Extra keyword arguments forwarded to :func:`check_viability`.

    Returns
    -------
    ViabilityResult
        Full viability assessment for the PUB's circuit.
    """
    if not pub:
        raise ValueError("PUB tuple is empty: expected (circuit, observables, shots)")

    circuit = pub[0]
    shots = 4096
    if len(pub) > 2 and pub[2] is not None:
        shots = int(pub[2])

    return check_viability(circuit, backend=backend, shots=shots, **kwargs)
