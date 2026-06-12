"""Compilation receipts (passport) and regression watch.

Every compilation can leave a small local receipt: what was compiled,
for which backend, what the compiler predicted, and which tool versions
produced the prediction. Receipts accumulate in a plain JSONL file under
``QBC_DATA_DIR`` (see :mod:`qb_compiler._store`), so the same workload
can be compared across sessions: "last week this circuit compiled to
41 two-qubit gates with predicted fidelity 0.62, today it is 0.48".

Signals only. This module reports and suggests; it never blocks,
aborts, or gates a submission. What to do with a regression signal is
entirely the caller's decision.

Usage::

    from qb_compiler.receipts import make_receipt, record_receipt, regression_check

    receipt = make_receipt(circuit, viability_result, backend="ibm_fez")
    report = regression_check(receipt)
    print(report.message)
    record_receipt(receipt)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from qb_compiler._store import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

_RECEIPTS_FILE = "receipts.jsonl"


def structural_hash(circuit: Any) -> str:
    """Coarse structural fingerprint of an UNtranspiled circuit.

    Returns the first 16 hex chars of a sha256 over a canonical text of
    the circuit's structure: sorted op-name counts, qubit count, and
    depth of the untranspiled circuit.

    Deliberately coarse: the goal is to identify "the same workload"
    across sessions so receipts can be compared, not to fingerprint the
    exact circuit. Parameter values are excluded on purpose, so two VQE
    iterations of the same ansatz hash identically.
    """
    op_counts = circuit.count_ops()
    parts = [f"{name}={count}" for name, count in sorted(op_counts.items())]
    canonical = ";".join(parts) + f"|n_qubits={circuit.num_qubits}|depth={circuit.depth()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class CompilationReceipt:
    """A single compilation passport entry.

    Attributes
    ----------
    timestamp :
        ISO-8601 UTC time the receipt was created.
    backend :
        Target backend name.
    structural_hash :
        Coarse workload fingerprint (see :func:`structural_hash`).
    circuit_name :
        Name or description of the circuit.
    n_qubits :
        Qubit count of the untranspiled circuit.
    depth_in :
        Depth of the untranspiled circuit.
    post_2q_count :
        Two-qubit gate count after transpilation.
    post_depth :
        Depth after transpilation.
    predicted_fidelity :
        Compiler's fidelity estimate for the transpiled circuit.
    fidelity_typical_abs_error :
        Typical absolute error of the fidelity estimate, when known.
    error_budget :
        Per-source fidelity-loss breakdown, when known.
    calibration_age_days :
        Age of the calibration snapshot behind the estimate, when known.
    qiskit_version :
        Qiskit version that produced the transpilation.
    qb_compiler_version :
        qb-compiler version that produced the estimate.
    seed :
        Transpiler seed, when recorded.
    layout :
        Final physical-qubit layout, when recorded.
    notes :
        Free-form annotations.
    """

    timestamp: str
    backend: str
    structural_hash: str
    circuit_name: str
    n_qubits: int
    depth_in: int
    post_2q_count: int
    post_depth: int
    predicted_fidelity: float
    fidelity_typical_abs_error: float | None = None
    error_budget: dict[str, float] | None = None
    calibration_age_days: float | None = None
    qiskit_version: str = "unknown"
    qb_compiler_version: str = "dev"
    seed: int | None = None
    layout: list[int] | None = None
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """Plain JSON-serialisable dict (round-trips via ``CompilationReceipt(**d)``)."""
        return asdict(self)

    def __str__(self) -> str:
        band = (
            f" (+-{self.fidelity_typical_abs_error:.2f} typical)"
            if self.fidelity_typical_abs_error is not None
            else ""
        )
        return (
            f"Receipt[{self.structural_hash}] {self.circuit_name} on {self.backend} "
            f"@ {self.timestamp}: {self.n_qubits}q depth {self.depth_in} -> "
            f"{self.post_2q_count} 2q gates, depth {self.post_depth}, "
            f"predicted fidelity {self.predicted_fidelity:.4f}{band} "
            f"(qiskit {self.qiskit_version}, qb-compiler {self.qb_compiler_version})"
        )


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """Outcome of comparing a receipt against its most recent baseline.

    Attributes
    ----------
    status :
        ``"NO_BASELINE"``, ``"REGRESSION"``, ``"IMPROVEMENT"``, or
        ``"STABLE"``.
    current :
        The current receipt as a dict.
    baseline :
        The most recent prior receipt for the same workload and
        backend, or ``None`` when no baseline exists.
    fidelity_delta :
        ``current - baseline`` predicted fidelity, when a baseline exists.
    two_q_delta :
        ``current - baseline`` post-transpilation two-qubit gate count,
        when a baseline exists.
    band_used :
        The combined typical-abs-error band the deltas were judged
        against, when a baseline exists.
    message :
        One human sentence summarising the comparison.
    """

    status: str
    current: dict[str, Any]
    baseline: dict[str, Any] | None
    fidelity_delta: float | None
    two_q_delta: int | None
    band_used: float | None
    message: str


def _versions() -> tuple[str, str]:
    """Best-effort (qiskit, qb-compiler) version strings."""
    try:
        import qiskit

        qiskit_version = str(qiskit.__version__)
    except Exception:
        qiskit_version = "unknown"
    try:
        from importlib.metadata import version

        qbc_version = version("qb-compiler")
    except Exception:
        qbc_version = "dev"
    return qiskit_version, qbc_version


def make_receipt(
    circuit: Any,
    viability_result: Any,
    *,
    backend: str,
    seed: int | None = None,
    layout: list[int] | None = None,
) -> CompilationReceipt:
    """Assemble a :class:`CompilationReceipt` for *circuit* on *backend*.

    *viability_result* is duck-typed: anything exposing
    ``estimated_fidelity``, ``two_qubit_gate_count``, and ``depth``
    works (``error_budget``, ``fidelity_typical_abs_error``, and
    ``calibration_age_days`` are read when present).
    """
    qiskit_version, qbc_version = _versions()
    n_qubits = int(circuit.num_qubits)
    circuit_name = getattr(circuit, "name", None) or f"{n_qubits}q circuit"
    return CompilationReceipt(
        timestamp=datetime.now(timezone.utc).isoformat(),
        backend=backend,
        structural_hash=structural_hash(circuit),
        circuit_name=str(circuit_name),
        n_qubits=n_qubits,
        depth_in=int(circuit.depth()),
        post_2q_count=int(viability_result.two_qubit_gate_count),
        post_depth=int(viability_result.depth),
        predicted_fidelity=float(viability_result.estimated_fidelity),
        fidelity_typical_abs_error=getattr(viability_result, "fidelity_typical_abs_error", None),
        error_budget=getattr(viability_result, "error_budget", None),
        calibration_age_days=getattr(viability_result, "calibration_age_days", None),
        qiskit_version=qiskit_version,
        qb_compiler_version=qbc_version,
        seed=seed,
        layout=list(layout) if layout is not None else None,
    )


def record_receipt(receipt: CompilationReceipt) -> None:
    """Append *receipt* to the local receipts log (``receipts.jsonl``)."""
    path = append_jsonl(_RECEIPTS_FILE, receipt.to_json_dict())
    logger.debug("Recorded compilation receipt %s to %s", receipt.structural_hash, path)


def receipt_history(
    structural_hash: str | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """Read recorded receipts, optionally filtered by hash and/or backend."""
    records = read_jsonl(_RECEIPTS_FILE)
    if structural_hash is not None:
        records = [r for r in records if r.get("structural_hash") == structural_hash]
    if backend is not None:
        records = [r for r in records if r.get("backend") == backend]
    return records


def regression_check(receipt: CompilationReceipt) -> RegressionReport:
    """Compare *receipt* against the most recent prior receipt for the
    same workload (structural hash) and backend.

    The fidelity delta is judged against the COMBINED typical-abs-error
    bands of the current and baseline estimates, so ordinary estimate
    noise does not cry wolf: only a drop larger than both bands together
    is flagged ``"REGRESSION"`` (and only a rise larger than that is
    ``"IMPROVEMENT"``). This is a signal-significance heuristic, not
    policy: the report never blocks or aborts anything, it just tells
    the caller what changed.
    """
    current = receipt.to_json_dict()
    priors = [r for r in receipt_history(receipt.structural_hash, receipt.backend) if r != current]
    priors.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)

    if not priors:
        return RegressionReport(
            status="NO_BASELINE",
            current=current,
            baseline=None,
            fidelity_delta=None,
            two_q_delta=None,
            band_used=None,
            message=(
                f"No prior receipt for workload {receipt.structural_hash} on "
                f"{receipt.backend}; recorded as the new baseline."
            ),
        )

    baseline = priors[0]
    baseline_fid = float(baseline.get("predicted_fidelity", 0.0))
    baseline_2q = int(baseline.get("post_2q_count", 0))
    baseline_ts = str(baseline.get("timestamp", "unknown time"))

    band_missing = (
        receipt.fidelity_typical_abs_error is None
        and (baseline.get("fidelity_typical_abs_error") if baseline else None) is None
    )
    band = (receipt.fidelity_typical_abs_error or 0.0) + (
        float(baseline.get("fidelity_typical_abs_error") or 0.0)
    )
    fidelity_delta = receipt.predicted_fidelity - baseline_fid
    two_q_delta = receipt.post_2q_count - baseline_2q

    if fidelity_delta < -band:
        status = "NO_BAND" if band_missing else "REGRESSION"
        verdict = "dropped beyond the combined typical-error band"
    elif fidelity_delta > band:
        status = "NO_BAND" if band_missing else "IMPROVEMENT"
        verdict = "rose beyond the combined typical-error band"
    else:
        status = "STABLE"
        verdict = "is within the combined typical-error band"

    message = (
        f"Predicted fidelity {receipt.predicted_fidelity:.4f} vs "
        f"{baseline_fid:.4f} at baseline ({baseline_ts}): delta "
        f"{fidelity_delta:+.4f} {verdict} (+-{band:.4f}); two-qubit gate "
        f"count changed by {two_q_delta:+d}."
    )
    return RegressionReport(
        status=status,
        current=current,
        baseline=baseline,
        fidelity_delta=round(fidelity_delta, 6),
        two_q_delta=two_q_delta,
        band_used=round(band, 6),
        message=message,
    )
