"""Verify mode: mirror-circuit fidelity check plus a local accuracy record.

Compares the fidelity that :func:`qb_compiler.viability.check_viability`
predicts against a measured mirror-circuit success rate, and keeps a local
JSONL log of predicted-vs-actual pairs so the prediction accuracy can be
summarised over time.

The mirror construction is deliberately simple: strip final measurements,
append the inverse of the stripped circuit, measure all qubits.  The
all-zeros return probability of that mirror is a SUCCESS PROXY related to
(but not equal to) the squared circuit fidelity.  See :func:`build_mirror`
for the assumptions.  This module reports numbers only; it makes no
pass/fail judgement and applies no threshold.

Usage::

    from qb_compiler.verify import verify_viability

    result = verify_viability(circuit, "aer", backend="ibm_fez")
    print(result)
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qb_compiler._store import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

_RECORDS_FILE = "verify_records.jsonl"


@dataclass(frozen=True, slots=True)
class MirrorResult:
    """Outcome of a mirror-circuit run.

    Attributes
    ----------
    shots :
        Total shots observed in the returned counts.
    zeros_count :
        Number of shots that returned the all-zeros bitstring.
    mirror_success :
        ``zeros_count / shots``: the all-zeros return probability.
    notes :
        Free-form notes about how the run was executed (runner type,
        shot mismatches, and similar).
    """

    shots: int
    zeros_count: int
    mirror_success: float
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Mirror success: {self.mirror_success:.4f} "
            f"({self.zeros_count}/{self.shots} all-zeros shots)"
        )


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Predicted-vs-measured comparison for one circuit.

    No pass/fail judgement and no threshold: this is a signal, not a
    policy decision.  ``discrepancy`` is signed
    (``mirror_success - predicted_squared``), so positive values mean
    the measurement came out ABOVE the squared prediction.
    """

    timestamp: str
    backend: str | None
    n_qubits: int
    predicted_fidelity: float
    predicted_squared: float
    mirror_success: float
    shots: int
    discrepancy: float

    def __str__(self) -> str:
        backend = self.backend or "unknown"
        return (
            f"Verify ({backend}, {self.n_qubits}q, {self.shots} shots): "
            f"predicted^2 {self.predicted_squared:.4f} vs "
            f"measured {self.mirror_success:.4f} "
            f"(discrepancy {self.discrepancy:+.4f})"
        )


def build_mirror(circuit: Any) -> Any:
    """Build the mirror of *circuit*: stripped circuit, then its inverse.

    Final measurements are removed (``remove_final_measurements``), the
    inverse of the stripped circuit is appended, and all qubits are
    measured at the end.  Under noiseless execution the mirror returns
    the all-zeros bitstring with probability 1 (assuming the input state
    is all zeros).

    Assumptions, stated plainly:

    * The all-zeros return probability of the mirror is a SUCCESS PROXY
      related to (not equal to) the squared fidelity of the original
      circuit: the state traverses the circuit roughly twice, so errors
      accumulate over ~2x the gate count.
    * The proxy assumes errors do not systematically cancel between the
      circuit and its inverse.  Coherent errors CAN echo out on the
      mirrored path, making the proxy optimistic; transpiler-level
      simplification across the circuit/inverse seam can do the same.
    * This is NOT randomized benchmarking and is NOT a randomized mirror
      circuit protocol (no Pauli frame randomization); no formal claim
      about the relation to average gate fidelity is made.

    Parameters
    ----------
    circuit :
        A Qiskit ``QuantumCircuit``.

    Returns
    -------
    Any
        A new ``QuantumCircuit``: stripped circuit + inverse + final
        measurement of all qubits.
    """
    stripped = circuit.remove_final_measurements(inplace=False)
    mirror = stripped.compose(stripped.inverse())
    mirror.name = f"{getattr(circuit, 'name', 'circuit')}_mirror"
    mirror.measure_all()
    return mirror


RunnerCallable = Callable[[Any, int], dict[str, int]]


def _is_all_zeros(bitstring: str) -> bool:
    """``True`` when a counts key contains only zeros (spaces ignored)."""
    stripped = bitstring.replace(" ", "")
    return bool(stripped) and set(stripped) == {"0"}


def _counts_via_aer(mirror: Any, shots: int) -> dict[str, int]:
    """Run *mirror* on a lazily imported AerSimulator."""
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "runner='aer' requires qiskit-aer. Install it with: pip install qiskit-aer"
        ) from exc

    from qiskit import transpile

    simulator = AerSimulator()
    transpiled = transpile(mirror, simulator)
    job = simulator.run(transpiled, shots=shots)
    counts = job.result().get_counts()
    return dict(counts)


def _counts_via_sampler(mirror: Any, runner: Any, shots: int) -> dict[str, int]:
    """Extract counts from a SamplerV2-style ``runner.run(...)`` result."""
    try:
        job = runner.run([mirror], shots=shots)
        pub_result = job.result()[0]
        data = pub_result.data
        # Common register name from measure_all is "meas"; fall back to
        # the first field that exposes get_counts().
        for name in ("meas", "c"):
            reg = getattr(data, name, None)
            if reg is not None and hasattr(reg, "get_counts"):
                return dict(reg.get_counts())
        for name in dir(data):
            if name.startswith("_"):
                continue
            reg = getattr(data, name, None)
            if reg is not None and hasattr(reg, "get_counts"):
                return dict(reg.get_counts())
        raise TypeError("no field with get_counts() in pub result data")
    except Exception as exc:
        raise TypeError(f"runner.run(...) did not yield SamplerV2-style counts: {exc}") from exc


def run_mirror(circuit: Any, runner: Any, *, shots: int = 256) -> MirrorResult:
    """Build the mirror of *circuit*, execute it, and return the result.

    Parameters
    ----------
    circuit :
        A Qiskit ``QuantumCircuit`` (final measurements are stripped
        before mirroring).
    runner :
        How to execute the mirror.  Duck-typed, one of:

        * a callable ``(qiskit_circuit, shots) -> counts`` returning a
          dict of bitstring to count,
        * an object with ``.run(...)`` following the Qiskit SamplerV2
          interface (pub result exposing ``get_counts()``),
        * the string ``"aer"``, which lazily imports
          ``qiskit_aer.AerSimulator`` (the mirror is transpiled to the
          simulator first; raises ``ImportError`` with an install hint
          when qiskit-aer is absent).
    shots :
        Number of shots requested from the runner.

    Returns
    -------
    MirrorResult
        Shots observed, all-zeros count, and the success proxy.
    """
    mirror = build_mirror(circuit)
    notes: list[str] = []

    if isinstance(runner, str):
        if runner != "aer":
            raise ValueError(f"unknown runner string {runner!r}; expected 'aer'")
        counts = _counts_via_aer(mirror, shots)
        notes.append("runner: aer simulator")
    elif callable(runner):
        counts = dict(runner(mirror, shots))
        notes.append("runner: callable")
    elif hasattr(runner, "run"):
        counts = _counts_via_sampler(mirror, runner, shots)
        notes.append("runner: SamplerV2-style object")
    else:
        raise TypeError(
            "runner must be a callable (circuit, shots) -> counts, an object "
            "with .run(...) (SamplerV2 interface), or the string 'aer'"
        )

    total = int(sum(counts.values()))
    if total <= 0:
        raise ValueError("runner returned empty counts; cannot compute mirror success")
    if total != shots:
        notes.append(f"observed {total} shots, requested {shots}")

    zeros = int(sum(v for k, v in counts.items() if _is_all_zeros(str(k))))
    return MirrorResult(
        shots=total,
        zeros_count=zeros,
        mirror_success=zeros / total,
        notes=notes,
    )


def verify_viability(
    circuit: Any,
    runner: Any,
    *,
    backend: str | None = None,
    shots: int = 256,
    record: bool = True,
) -> VerifyResult:
    """Compare the predicted fidelity of *circuit* against a measured mirror run.

    Runs :func:`qb_compiler.viability.check_viability` to get the predicted
    fidelity, executes the mirror of *circuit* via *runner*, and compares the
    mirror success proxy against ``predicted ** 2`` (the state traverses the
    circuit roughly twice in the mirror).  When *record* is ``True`` the
    predicted-vs-actual pair is appended to a local JSONL log
    (``verify_records.jsonl`` under ``QBC_DATA_DIR``) so prediction accuracy
    can be summarised later with :func:`accuracy_summary`.

    Reports numbers only: no pass/fail judgement, no threshold.

    Parameters
    ----------
    circuit :
        A Qiskit ``QuantumCircuit``.
    runner :
        Execution backend for the mirror; see :func:`run_mirror`.
    backend :
        Backend name passed to ``check_viability`` (e.g. ``"ibm_fez"``).
    shots :
        Shots for the mirror run.
    record :
        Append the comparison to the local verify record log.

    Returns
    -------
    VerifyResult
        Timestamped predicted-vs-measured comparison.
    """
    from qb_compiler.viability import check_viability

    viability = check_viability(circuit, backend=backend)
    predicted = float(viability.estimated_fidelity)
    predicted_squared = predicted**2

    mirror = run_mirror(circuit, runner, shots=shots)
    discrepancy = mirror.mirror_success - predicted_squared

    result = VerifyResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        backend=backend,
        n_qubits=int(circuit.num_qubits),
        predicted_fidelity=predicted,
        predicted_squared=predicted_squared,
        mirror_success=mirror.mirror_success,
        shots=mirror.shots,
        discrepancy=discrepancy,
    )

    if record:
        path = append_jsonl(
            _RECORDS_FILE,
            {
                "timestamp": result.timestamp,
                "backend": result.backend,
                "n_qubits": result.n_qubits,
                "predicted_fidelity": result.predicted_fidelity,
                "predicted_squared": result.predicted_squared,
                "mirror_success": result.mirror_success,
                "shots": result.shots,
                "discrepancy": result.discrepancy,
            },
        )
        logger.debug("verify record appended to %s", path)

    return result


def accuracy_summary() -> dict[str, Any]:
    """Summarise the local predicted-vs-actual verify records.

    Reads ``verify_records.jsonl`` from ``QBC_DATA_DIR`` and returns
    aggregate accuracy numbers for the viability fidelity model on this
    machine.  Signals only: how far off the predictions were, not whether
    that is acceptable.

    Returns
    -------
    dict
        ``n`` (record count), ``median_abs_discrepancy``,
        ``mean_signed_discrepancy``, and ``per_backend_counts``
        (backend name to record count).  When the log is empty,
        ``n`` is 0 and the discrepancy fields are ``None``.
    """
    records = read_jsonl(_RECORDS_FILE)
    discrepancies = [float(r["discrepancy"]) for r in records if r.get("discrepancy") is not None]
    if not discrepancies:
        return {
            "n": 0,
            "median_abs_discrepancy": None,
            "mean_signed_discrepancy": None,
            "per_backend_counts": {},
        }

    backends = Counter(str(r.get("backend") or "unknown") for r in records)
    return {
        "n": len(discrepancies),
        "median_abs_discrepancy": statistics.median(abs(d) for d in discrepancies),
        "mean_signed_discrepancy": statistics.mean(discrepancies),
        "per_backend_counts": dict(backends),
    }
