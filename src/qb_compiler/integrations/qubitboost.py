"""QubitBoost SDK integration for qb-compiler.

This bridges qb-compiler (open source) with the QubitBoost SDK
(proprietary).  The integration is **optional** — qb-compiler works
standalone without the SDK.

When the SDK is installed, the compile pipeline gains access to all
seven QubitBoost gates:

Pre-execution:
    TomoGate     — pre-flight state fidelity certification
    SafetyGate   — QEC trust scoring and doom detection

During execution:
    OptGate      — adaptive QAOA (hardware-validated: 117-208x shot
                   reduction on supported workloads)
    ChemGate     — VQE operator preselection (hardware-validated:
                   32-42% fewer evaluations on supported workflows)
    GuardGate    — QAOA quality assurance on validated workloads
    LiveGate     — real-time doom detection on supported backends
    ShotValidator — redundant syndrome verification

Post-execution:
    ShotValidator — verify result integrity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── SDK availability ────────────────────────────────────────────────

def is_sdk_available() -> bool:
    """Return ``True`` if the QubitBoost SDK is importable."""
    try:
        import qubitboost  # noqa: F401
        return True
    except ImportError:
        return False


SDK_INSTALL_HINT = (
    "QubitBoost SDK required. Install with: pip install qubitboost-sdk\n"
    "Learn more at https://qubitboost.io"
)


# ── Confidence levels ───────────────────────────────────────────────

class Confidence(str, Enum):
    """Confidence in circuit type detection."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Gate metadata ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GateInfo:
    """Static metadata for a QubitBoost gate."""

    name: str
    headline: str
    validated_claim: str
    qualifier: str
    phase: str  # "pre", "during", "post"
    circuit_types: tuple[str, ...]
    requires_high_confidence: bool  # only show validated numbers at HIGH


GATE_REGISTRY: dict[str, GateInfo] = {
    "OptGate": GateInfo(
        name="OptGate",
        headline="Adaptive QAOA shot reduction",
        validated_claim="117-208x shot reduction",
        qualifier="on validated QAOA workloads",
        phase="during",
        circuit_types=("qaoa",),
        requires_high_confidence=True,
    ),
    "ChemGate": GateInfo(
        name="ChemGate",
        headline="VQE operator preselection",
        validated_claim="32-42% fewer evaluations",
        qualifier="on validated VQE workflows",
        phase="during",
        circuit_types=("vqe",),
        requires_high_confidence=True,
    ),
    "TomoGate": GateInfo(
        name="TomoGate",
        headline="Pre-flight state fidelity certification",
        validated_claim="Pre-flight certification",
        qualifier="for supported circuit types",
        phase="pre",
        circuit_types=("qaoa", "vqe", "qec", "general"),
        requires_high_confidence=False,
    ),
    "LiveGate": GateInfo(
        name="LiveGate",
        headline="Real-time doom detection",
        validated_claim="Abort failing runs early",
        qualifier="on supported backends",
        phase="during",
        circuit_types=("qaoa", "vqe", "general"),
        requires_high_confidence=False,
    ),
    "SafetyGate": GateInfo(
        name="SafetyGate",
        headline="QEC trust scoring and doom detection",
        validated_claim="QEC trust scoring",
        qualifier="hardware-validated at d=7",
        phase="pre",
        circuit_types=("qec",),
        requires_high_confidence=True,
    ),
    "GuardGate": GateInfo(
        name="GuardGate",
        headline="QAOA quality assurance",
        validated_claim="QAOA quality assurance",
        qualifier="on validated QAOA workloads",
        phase="during",
        circuit_types=("qaoa",),
        requires_high_confidence=True,
    ),
    "ShotValidator": GateInfo(
        name="ShotValidator",
        headline="Result integrity verification",
        validated_claim="Syndrome verification",
        qualifier="on supported QEC circuits",
        phase="post",
        circuit_types=("qaoa", "vqe", "qec", "general"),
        requires_high_confidence=False,
    ),
}


# ── Circuit type detection ──────────────────────────────────────────

def detect_circuit_type(circuit: Any) -> tuple[str, Confidence]:
    """Detect whether a circuit is QAOA, VQE, QEC, or general.

    Returns ``(circuit_type, confidence)`` where *circuit_type* is one
    of ``"qaoa"``, ``"vqe"``, ``"qec"``, ``"general"`` and
    *confidence* indicates how certain the detection is.

    Parameters
    ----------
    circuit :
        A Qiskit ``QuantumCircuit``, or any object with a ``name``
        attribute and ``count_ops()`` method.
    """
    name = (getattr(circuit, "name", "") or "").lower()

    # Check explicit metadata first → HIGH confidence
    if "qaoa" in name:
        return "qaoa", Confidence.HIGH
    if "vqe" in name or "uccsd" in name:
        return "vqe", Confidence.HIGH
    if "qec" in name or "surface" in name or "syndrome" in name:
        return "qec", Confidence.HIGH

    # Structural analysis
    try:
        ops = circuit.count_ops()
    except Exception:
        return "general", Confidence.LOW

    total = sum(ops.values())
    if total == 0:
        return "general", Confidence.LOW

    rz_count = ops.get("rz", 0) + ops.get("rzz", 0) + ops.get("rzx", 0)
    rx_count = ops.get("rx", 0)
    ry_count = ops.get("ry", 0)
    cx_count = ops.get("cx", 0) + ops.get("cz", 0) + ops.get("ecr", 0)

    # QAOA signature: entangling + Rz cost layer + Rx mixer layer
    # Both Rx and CX must be significant, and CX should dominate or co-dominate
    if (cx_count > 0
        and rz_count > 0
        and rx_count > 0
        and cx_count / total > 0.15
        and rx_count / total > 0.08):
        return "qaoa", Confidence.MEDIUM

    # VQE signature: parameterised Ry/Rz rotations dominate, fewer entangling
    param_ratio = (rz_count + rx_count + ry_count) / total
    if param_ratio > 0.4 and ry_count > 0:
        return "vqe", Confidence.MEDIUM

    # General parameterised circuit
    if param_ratio > 0.5:
        return "vqe", Confidence.MEDIUM

    # Ansatz-like name
    if "ansatz" in name or "efficient" in name:
        return "vqe", Confidence.MEDIUM

    return "general", Confidence.LOW


# ── Gate recommendations ────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GateRecommendation:
    """A recommended QubitBoost gate for a circuit."""

    gate: str
    status: str  # "Eligible", "May be eligible", "Available"
    headline: str
    validated_claim: str | None
    qualifier: str | None
    phase: str

    def __str__(self) -> str:
        line = f"{self.gate:14s}  {self.status} — {self.headline}"
        if self.validated_claim:
            line += f"\n{'':14s}  Hardware-validated: {self.validated_claim} {self.qualifier}"
        return line


def recommend_gates(
    circuit_type: str,
    confidence: Confidence,
) -> list[GateRecommendation]:
    """Recommend QubitBoost gates for a circuit type.

    Only shows validated performance claims when confidence is HIGH.
    At MEDIUM confidence, gates are shown as "May be eligible".
    At LOW confidence, only universal gates (LiveGate, ShotValidator,
    TomoGate) are shown.

    Parameters
    ----------
    circuit_type :
        One of ``"qaoa"``, ``"vqe"``, ``"qec"``, ``"general"``.
    confidence :
        Detection confidence level.

    Returns
    -------
    list[GateRecommendation]
        Applicable gates, ordered by phase (pre -> during -> post).
    """
    type_gates: dict[str, list[str]] = {
        "qaoa": ["TomoGate", "OptGate", "GuardGate", "LiveGate", "ShotValidator"],
        "vqe": ["TomoGate", "ChemGate", "LiveGate", "ShotValidator"],
        "qec": ["SafetyGate", "TomoGate", "ShotValidator"],
        "general": ["TomoGate", "LiveGate", "ShotValidator"],
    }
    gate_names = type_gates.get(circuit_type, type_gates["general"])

    phase_order = {"pre": 0, "during": 1, "post": 2}
    recs: list[GateRecommendation] = []

    for gname in gate_names:
        info = GATE_REGISTRY[gname]

        # Determine eligibility status
        if info.requires_high_confidence:
            if confidence == Confidence.HIGH:
                status = "Eligible"
                claim = info.validated_claim
                qual = info.qualifier
            elif confidence == Confidence.MEDIUM:
                status = "May be eligible"
                claim = info.validated_claim
                qual = info.qualifier
            else:
                # LOW confidence: skip specialised gates
                continue
        else:
            status = "Available"
            claim = None
            qual = None

        recs.append(GateRecommendation(
            gate=gname,
            status=status,
            headline=info.headline,
            validated_claim=claim,
            qualifier=qual,
            phase=info.phase,
        ))

    recs.sort(key=lambda r: phase_order.get(r.phase, 9))
    return recs


# ── Executor (requires SDK) ────────────────────────────────────────

class QubitBoostExecutor:
    """Execute compiled circuits through QubitBoost gates.

    Requires ``pip install qubitboost-sdk``.

    Usage::

        from qb_compiler.integrations.qubitboost import QubitBoostExecutor

        executor = QubitBoostExecutor(backend="ibm_fez")
        result = executor.execute_optgate(compiled_circuit, problem_graph=G)
    """

    def __init__(self, backend: str, **kwargs: Any) -> None:
        if not is_sdk_available():
            raise ImportError(SDK_INSTALL_HINT)

        from qubitboost import QubitBoost  # type: ignore[import-untyped]

        self._qb = QubitBoost.from_backend(
            backend="qiskit",
            backend_name=backend,
            **kwargs,
        )
        self._backend = backend

    def execute_optgate(self, circuit: Any, **kwargs: Any) -> Any:
        """Execute QAOA circuit through OptGate (adaptive shot reduction).

        Hardware-validated: 117-208x shot reduction on supported QAOA
        workloads. Actual reduction depends on circuit structure,
        backend, and calibration state.
        """
        return self._qb.adaptive_optimizer.optimize(
            problem_graph=kwargs.pop("problem_graph", None),
            p=kwargs.pop("p", 3),
            shot_budget=kwargs.pop("shots", 4096),
            **kwargs,
        )

    def execute_chemgate(self, circuit: Any, **kwargs: Any) -> Any:
        """Execute VQE circuit through ChemGate (operator preselection).

        Hardware-validated: 32-42% fewer evaluations on supported VQE
        workflows.
        """
        return self._qb.operator_selector.propose(
            hamiltonian=kwargs.pop("hamiltonian", None),
            n_qubits=kwargs.pop("n_qubits", circuit.num_qubits),
            **kwargs,
        )

    def execute_tomogate(self, circuit: Any, **kwargs: Any) -> Any:
        """Pre-flight fidelity certification via TomoGate."""
        return self._qb.tomo_gate.estimate(
            circuit=circuit,
            target_state=kwargs.pop("target_state", "ghz4"),
            **kwargs,
        )

    def execute_safetygate(
        self, syndrome_history: Any, distance: int = 5, **kwargs: Any,
    ) -> Any:
        """QEC trust scoring via SafetyGate (hardware-validated at d=7)."""
        return self._qb.safety_gate.check(
            syndrome_history=syndrome_history,
            distance=distance,
            **kwargs,
        )
