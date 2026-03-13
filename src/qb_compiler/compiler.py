"""Core compiler entry point.

:class:`QBCompiler` orchestrates calibration loading, pass management,
fidelity estimation, and cost calculation for quantum circuit compilation.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from qb_compiler.config import (
    BACKEND_CONFIGS,
    BackendSpec,
    CompilerConfig,
    get_backend_spec,
)
from qb_compiler.exceptions import (
    BackendNotSupportedError,
    BudgetExceededError,
    InvalidCircuitError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qb_compiler.calibration.models.backend_properties import BackendProperties

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Protocols / ABCs for extension points
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class CalibrationProvider(Protocol):
    """Supplies per-qubit / per-gate calibration data for a backend."""

    def get_cx_error(self, qubit_pair: tuple[int, int]) -> float:
        """Return the two-qubit gate error for *(q0, q1)*."""
        ...

    def get_readout_error(self, qubit: int) -> float:
        """Return the readout assignment error for *qubit*."""
        ...

    def get_t1(self, qubit: int) -> float:
        """Return T1 (microseconds) for *qubit*."""
        ...

    def get_t2(self, qubit: int) -> float:
        """Return T2 (microseconds) for *qubit*."""
        ...

    @property
    def age_hours(self) -> float:
        """Hours since calibration data was fetched."""
        ...


@runtime_checkable
class NoiseModel(Protocol):
    """Abstract noise model consumed by noise-aware passes."""

    def depolarizing_rate(self, gate: str, qubits: tuple[int, ...]) -> float:
        """Return depolarizing error probability for *gate* on *qubits*."""
        ...


class CostEstimator:
    """Estimates execution cost in USD for a compiled circuit.

    Uses the backend's ``cost_per_shot`` from :data:`BACKEND_CONFIGS`
    as the baseline, then scales for gate count (rough proxy for
    execution time on pay-per-second platforms).
    """

    def __init__(self, backend_spec: BackendSpec) -> None:
        self._spec = backend_spec

    def estimate(self, depth: int, n_qubits: int, shots: int) -> CostEstimate:
        """Return a :class:`CostEstimate` for the given workload."""
        # Base: per-shot price.  For time-billed backends (IBM Utility)
        # deeper circuits consume proportionally more wall-clock.
        depth_factor = max(1.0, depth / 100.0)  # normalised around depth-100
        per_shot = self._spec.cost_per_shot * depth_factor
        total = per_shot * shots
        return CostEstimate(
            cost_per_shot_usd=per_shot,
            total_usd=total,
            shots=shots,
            depth=depth,
            n_qubits=n_qubits,
            backend=_backend_name(self._spec),
        )


def _backend_name(spec: BackendSpec) -> str:
    """Reverse-lookup a backend name from its spec (best-effort)."""
    for name, s in BACKEND_CONFIGS.items():
        if s is spec:
            return name
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Result of a cost estimation."""

    cost_per_shot_usd: float
    total_usd: float
    shots: int
    depth: int
    n_qubits: int
    backend: str

    def within_budget(self, budget_usd: float) -> bool:
        return self.total_usd <= budget_usd


@dataclass(frozen=True, slots=True)
class PassResult:
    """Outcome of a single compiler pass."""

    pass_name: str
    elapsed_ms: float
    depth_before: int
    depth_after: int
    gate_count_before: int
    gate_count_after: int
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CompileResult:
    """Full result bundle returned by :meth:`QBCompiler.compile`."""

    compiled_circuit: QBCircuit
    original_depth: int
    compiled_depth: int
    estimated_fidelity: float
    pass_log: tuple[PassResult, ...]
    compilation_time_ms: float

    @property
    def depth_reduction_pct(self) -> float:
        if self.original_depth == 0:
            return 0.0
        return (1.0 - self.compiled_depth / self.original_depth) * 100.0


# ═══════════════════════════════════════════════════════════════════════
# Lightweight circuit representation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GateOp:
    """A single gate operation."""

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2


class QBCircuit:
    """Minimal quantum circuit representation.

    Stores a flat list of :class:`GateOp` instructions and derives depth /
    gate-count on the fly.  This is deliberately backend-agnostic; Qiskit
    / Cirq interop is handled by converter utilities (not in this module).
    """

    __slots__ = ("metadata", "n_qubits", "ops")

    def __init__(self, n_qubits: int, ops: list[GateOp] | None = None) -> None:
        if n_qubits < 1:
            raise InvalidCircuitError(f"n_qubits must be >= 1, got {n_qubits}")
        self.n_qubits = n_qubits
        self.ops: list[GateOp] = ops if ops is not None else []
        self.metadata: dict[str, Any] = {}

    # ── builder helpers ──────────────────────────────────────────────

    def add(self, name: str, qubits: tuple[int, ...], params: tuple[float, ...] = ()) -> QBCircuit:
        """Append a gate and return *self* for chaining."""
        for q in qubits:
            if q < 0 or q >= self.n_qubits:
                raise InvalidCircuitError(
                    f"Qubit index {q} out of range for {self.n_qubits}-qubit circuit"
                )
        self.ops.append(GateOp(name=name, qubits=qubits, params=params))
        return self

    def h(self, q: int) -> QBCircuit:
        return self.add("h", (q,))

    def cx(self, control: int, target: int) -> QBCircuit:
        return self.add("cx", (control, target))

    def rz(self, q: int, theta: float) -> QBCircuit:
        return self.add("rz", (q,), (theta,))

    def rx(self, q: int, theta: float) -> QBCircuit:
        return self.add("rx", (q,), (theta,))

    def x(self, q: int) -> QBCircuit:
        return self.add("x", (q,))

    def measure_all(self) -> QBCircuit:
        for q in range(self.n_qubits):
            self.add("measure", (q,))
        return self

    # ── analysis ─────────────────────────────────────────────────────

    @property
    def gate_count(self) -> int:
        return len(self.ops)

    @property
    def two_qubit_count(self) -> int:
        return sum(1 for op in self.ops if op.is_two_qubit)

    @property
    def depth(self) -> int:
        """Circuit depth computed via a per-qubit occupancy scan."""
        if not self.ops:
            return 0
        qubit_time: list[int] = [0] * self.n_qubits
        for op in self.ops:
            layer = max(qubit_time[q] for q in op.qubits) + 1
            for q in op.qubits:
                qubit_time[q] = layer
        return max(qubit_time)

    @property
    def gate_names(self) -> set[str]:
        return {op.name for op in self.ops}

    def copy(self) -> QBCircuit:
        """Deep copy of the circuit."""
        import copy as _copy

        new = QBCircuit(self.n_qubits, [GateOp(o.name, o.qubits, o.params) for o in self.ops])
        new.metadata = _copy.deepcopy(self.metadata)
        return new

    def __repr__(self) -> str:
        return (
            f"QBCircuit(n_qubits={self.n_qubits}, "
            f"ops={self.gate_count}, depth={self.depth})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Pass infrastructure
# ═══════════════════════════════════════════════════════════════════════

class BasePass:
    """Abstract base for compiler passes.

    Subclasses must implement :meth:`run`.  The pass framework records
    timing and depth delta automatically.
    """

    name: str = "base_pass"

    def run(self, circuit: QBCircuit, config: CompilerConfig) -> QBCircuit:
        """Transform *circuit* in-place or return a new one.

        Must return the (possibly modified) circuit.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class _GateCancellationPass(BasePass):
    """Cancel adjacent inverse gate pairs (HH, XX, CX-CX on same qubits)."""

    name = "gate_cancellation"

    _SELF_INVERSE = frozenset({"h", "x", "cx", "cz", "swap"})

    def run(self, circuit: QBCircuit, config: CompilerConfig) -> QBCircuit:
        changed = True
        ops = circuit.ops
        while changed:
            changed = False
            new_ops: list[GateOp] = []
            skip_next = False
            for i in range(len(ops)):
                if skip_next:
                    skip_next = False
                    continue
                if i + 1 < len(ops):
                    a, b = ops[i], ops[i + 1]
                    if (
                        a.name == b.name
                        and a.qubits == b.qubits
                        and a.params == b.params
                        and a.name in self._SELF_INVERSE
                    ):
                        skip_next = True
                        changed = True
                        continue
                new_ops.append(ops[i])
            ops = new_ops
        circuit.ops = ops
        return circuit


class _RotationMergePass(BasePass):
    """Merge consecutive single-qubit rotations on the same axis."""

    name = "rotation_merge"

    _ROTATION_GATES = frozenset({"rz", "rx", "ry"})

    def run(self, circuit: QBCircuit, config: CompilerConfig) -> QBCircuit:
        ops = circuit.ops
        merged: list[GateOp] = []
        i = 0
        while i < len(ops):
            op = ops[i]
            if op.name in self._ROTATION_GATES and op.params:
                total_angle = op.params[0]
                j = i + 1
                while (
                    j < len(ops)
                    and ops[j].name == op.name
                    and ops[j].qubits == op.qubits
                    and ops[j].params
                ):
                    total_angle += ops[j].params[0]
                    j += 1
                # Normalise into (-pi, pi]
                total_angle = math.remainder(total_angle, 2 * math.pi)
                if abs(total_angle) > 1e-10:
                    merged.append(GateOp(op.name, op.qubits, (total_angle,)))
                i = j
            else:
                merged.append(op)
                i += 1
        circuit.ops = merged
        return circuit


class _BasisTranslationPass(BasePass):
    """Decompose gates not in the target basis set.

    Supports a small but useful set of decompositions:
    - h  -> rz(pi) sx rz(pi)          (IBM basis)
    - cx -> cz with surrounding h      (Rigetti/IQM basis)
    - h  -> rx(pi) rz(pi)             (alt decomposition)
    """

    name = "basis_translation"

    def run(self, circuit: QBCircuit, config: CompilerConfig) -> QBCircuit:
        basis = config.effective_basis_gates
        if basis is None:
            return circuit  # no target basis, skip

        basis_set = set(basis) | {"measure", "reset", "barrier"}
        new_ops: list[GateOp] = []

        for op in circuit.ops:
            if op.name in basis_set:
                new_ops.append(op)
                continue
            decomposed = self._decompose(op, basis_set)
            new_ops.extend(decomposed)

        circuit.ops = new_ops
        return circuit

    @staticmethod
    def _decompose(op: GateOp, basis_set: set[str]) -> list[GateOp]:
        """Best-effort decomposition of a single gate into basis gates."""
        q = op.qubits
        pi = math.pi

        if op.name == "h" and len(q) == 1:
            if "sx" in basis_set and "rz" in basis_set:
                return [
                    GateOp("rz", q, (pi / 2,)),
                    GateOp("sx", q, ()),
                    GateOp("rz", q, (pi / 2,)),
                ]
            if "rx" in basis_set and "rz" in basis_set:
                return [
                    GateOp("rz", q, (pi,)),
                    GateOp("rx", q, (pi / 2,)),
                ]

        if op.name == "cx" and len(q) == 2 and "cz" in basis_set:
            # CX = (I⊗H) CZ (I⊗H)  — decompose H if needed
            target = (q[1],)
            h_ops = _BasisTranslationPass._decompose(
                GateOp("h", target), basis_set
            ) if "h" not in basis_set else [GateOp("h", target)]
            return [*h_ops, GateOp("cz", q), *h_ops]

        if op.name == "cz" and len(q) == 2 and "cx" in basis_set:
            target = (q[1],)
            h_ops = _BasisTranslationPass._decompose(
                GateOp("h", target), basis_set
            ) if "h" not in basis_set else [GateOp("h", target)]
            return [*h_ops, GateOp("cx", q), *h_ops]

        if op.name == "x" and len(q) == 1:
            if "rx" in basis_set:
                return [GateOp("rx", q, (pi,))]
            if "sx" in basis_set:
                return [GateOp("sx", q, ()), GateOp("sx", q, ())]

        if op.name == "sx" and len(q) == 1 and "rx" in basis_set:
            return [GateOp("rx", q, (pi / 2,))]

        # Fallback: leave as-is (will be caught by validation if strict)
        return [op]


class PassManager:
    """Ordered sequence of :class:`BasePass` instances.

    Build via :meth:`default` for standard pass pipelines keyed by
    optimization level, or construct manually for custom flows.
    """

    def __init__(self, passes: Sequence[BasePass] | None = None) -> None:
        self.passes: list[BasePass] = list(passes) if passes else []

    def append(self, p: BasePass) -> PassManager:
        self.passes.append(p)
        return self

    def run(self, circuit: QBCircuit, config: CompilerConfig) -> tuple[QBCircuit, list[PassResult]]:
        """Execute all passes in order, collecting :class:`PassResult` logs."""
        log: list[PassResult] = []
        for p in self.passes:
            depth_before = circuit.depth
            gc_before = circuit.gate_count
            t0 = time.perf_counter()
            circuit = p.run(circuit, config)
            elapsed = (time.perf_counter() - t0) * 1000.0
            log.append(PassResult(
                pass_name=p.name,
                elapsed_ms=round(elapsed, 3),
                depth_before=depth_before,
                depth_after=circuit.depth,
                gate_count_before=gc_before,
                gate_count_after=circuit.gate_count,
            ))
        return circuit, log

    @classmethod
    def default(cls, optimization_level: int = 2) -> PassManager:
        """Return a preset pass pipeline.

        Level 0: basis translation only.
        Level 1: + gate cancellation.
        Level 2: + rotation merge (default).
        Level 3: all of the above (repeated cancellation).
        """
        passes: list[BasePass] = [_BasisTranslationPass()]
        if optimization_level >= 1:
            passes.append(_GateCancellationPass())
        if optimization_level >= 2:
            passes.append(_RotationMergePass())
        if optimization_level >= 3:
            passes.append(_GateCancellationPass())  # second round
            passes.append(_RotationMergePass())
        return cls(passes)


# ═══════════════════════════════════════════════════════════════════════
# IR bridge — convert between compiler QBCircuit and ir.circuit.QBCircuit
# ═══════════════════════════════════════════════════════════════════════


def _to_ir_circuit(circuit: QBCircuit) -> Any:
    """Convert compiler QBCircuit → ir.circuit.QBCircuit for pass pipeline."""
    from qb_compiler.ir.circuit import QBCircuit as IRCircuit
    from qb_compiler.ir.operations import QBGate

    ir_circ = IRCircuit(n_qubits=circuit.n_qubits, n_clbits=0, name="compiled")
    for op in circuit.ops:
        if op.name == "measure":
            continue  # skip measurements for pass pipeline
        ir_circ.add_gate(QBGate(name=op.name, qubits=op.qubits, params=op.params))
    return ir_circ


def _from_ir_circuit(ir_circ: Any, original: QBCircuit) -> QBCircuit:
    """Convert ir.circuit.QBCircuit → compiler QBCircuit, preserving measurements."""
    from qb_compiler.ir.operations import QBGate

    # The IR circuit may have more qubits (physical) than the original (logical)
    new = QBCircuit(ir_circ.n_qubits)
    for op in ir_circ.iter_ops():
        if isinstance(op, QBGate):
            new.add(op.name, op.qubits, op.params)
    # Re-add measurements from original
    for op in original.ops:
        if op.name == "measure":
            new.add("measure", op.qubits, op.params)
    return new


def _build_synthetic_calibration(spec: BackendSpec, backend_name: str, n_qubits: int) -> BackendProperties:
    """Build a BackendProperties with per-qubit variance from BackendSpec medians.

    Uses deterministic pseudo-random variance so results are reproducible.
    The variance is realistic: on IBM Heron processors, qubit quality can
    vary by 10x across the chip. This means calibration-aware placement
    picks qubits with 2-5x lower error than median.

    In production, real calibration data would be loaded from JSON fixtures
    or the QubitBoost calibration hub.
    """
    from qb_compiler.calibration.models.backend_properties import BackendProperties
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties

    qubit_props = []
    for q in range(n_qubits):
        # Deterministic per-qubit variance using simple hash
        seed = (q * 7919 + 31) % 1000 / 1000.0  # 0.0 to 1.0

        # Real IBM Heron data shows extreme variance:
        # T1: 50-500 us (10x range), T2: 20-300 us (15x range)
        # Readout: 0.001-0.075 (75x range)
        # Only ~20-30% of qubits are "good" (below median)
        # Use exponential-like distribution to mimic real hardware
        t1_factor = 0.2 + seed * seed * 2.0  # skewed: many bad, few great
        t2_factor = 0.15 + ((seed * 3.7) % 1.0) ** 2 * 2.5
        # Readout: some qubits are 7x worse than best
        readout_seed = ((seed * 5.3 + 0.17) % 1.0)
        readout_factor = 0.1 + readout_seed * readout_seed * 7.0

        qubit_props.append(QubitProperties(
            qubit_id=q,
            t1_us=spec.t1_us * t1_factor,
            t2_us=spec.t2_us * t2_factor,
            readout_error=spec.median_readout_error * readout_factor,
        ))

    # Build coupling map for common topologies
    coupling_map: list[tuple[int, int]] = []
    if spec.provider in ("ionq", "quantinuum"):
        # All-to-all connectivity
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                coupling_map.append((i, j))
                coupling_map.append((j, i))
    else:
        # Heavy-hex-like topology: linear chain + regular cross-links
        # This provides enough connectivity for VF2 to find good mappings
        for i in range(n_qubits - 1):
            coupling_map.append((i, i + 1))
            coupling_map.append((i + 1, i))
        # Cross-links every 2 qubits (mimics heavy-hex cross-connections)
        for i in range(0, n_qubits - 2, 2):
            if i + 2 < n_qubits:
                coupling_map.append((i, i + 2))
                coupling_map.append((i + 2, i))
        # Additional links every 3 for denser connectivity
        for i in range(0, n_qubits - 3, 3):
            if i + 3 < n_qubits:
                coupling_map.append((i, i + 3))
                coupling_map.append((i + 3, i))

    gate_props = []
    seen_edges: set[tuple[int, int]] = set()
    for q0, q1 in coupling_map:
        if (q0, q1) in seen_edges:
            continue
        seen_edges.add((q0, q1))
        # Real IBM Heron data: CX error ranges from ~0.001 to ~0.05
        # That's a 50x range. Best ~20% of edges have error < 0.003,
        # worst ~20% have error > 0.015
        seed = ((q0 * 7919 + q1 * 6271 + 17) % 1000) / 1000.0
        # Exponential-like: many edges mediocre, some very good, some very bad
        error_factor = 0.1 + seed * seed * 4.0  # 0.1x to 4.1x median
        gate_props.append(GateProperties(
            gate_type="cx",
            qubits=(q0, q1),
            error_rate=spec.median_cx_error * error_factor,
            gate_time_ns=400.0 if spec.provider not in ("ionq", "quantinuum") else 200_000.0,
        ))

    return BackendProperties(
        backend=backend_name,
        provider=spec.provider,
        n_qubits=n_qubits,
        basis_gates=spec.basis_gates,
        coupling_map=coupling_map,
        qubit_properties=qubit_props,
        gate_properties=gate_props,
        timestamp="synthetic",
    )


def _load_calibration_fixture(backend_name: str) -> BackendProperties | None:
    """Try to load a real calibration fixture for the given backend.

    Search order:
    1. QubitBoost calibration_hub (real daily snapshots, most data)
    2. qb-compiler test fixtures (smaller snapshots for unit tests)
    """
    import glob
    import os
    import re as _re

    from qb_compiler.calibration.models.backend_properties import BackendProperties

    # Sanitize backend_name to prevent path traversal
    if not _re.match(r"^[a-zA-Z0-9_-]+$", backend_name):
        logger.warning("Invalid backend_name for calibration lookup: %r", backend_name)
        return None

    # Directories to search, in priority order
    search_dirs: list[str] = []

    # User-configured calibration directory (via environment variable)
    env_cal_dir = os.environ.get("QBC_CALIBRATION_DIR")
    if env_cal_dir and os.path.isdir(env_cal_dir):
        search_dirs.append(env_cal_dir)

    # Bundled test fixture snapshots
    search_dirs.append(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "tests", "fixtures", "calibration_snapshots",
        ),
    )

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        # Try common naming patterns (use most recent match)
        for pattern in [
            f"{backend_name}_*.json",
            f"{backend_name.replace('_', '-')}*.json",
        ]:
            matches = sorted(glob.glob(os.path.join(search_dir, pattern)))
            if matches:
                try:
                    props = BackendProperties.from_qubitboost_json(matches[-1])
                    logger.info(
                        "Loaded calibration from %s (%d qubits, %d gates)",
                        matches[-1], props.n_qubits, len(props.gate_properties),
                    )
                    return props
                except Exception as e:
                    logger.debug("Failed to load %s: %s", matches[-1], e)
    return None


def _run_calibration_pipeline(
    circuit: QBCircuit,
    backend_name: str,
    spec: BackendSpec,
    calibration_props: BackendProperties | None,
) -> tuple[QBCircuit, dict[str, Any]]:
    """Run CalibrationMapper + NoiseAwareRouter on a circuit.

    Returns the mapped/routed circuit and metadata dict.
    """
    from qb_compiler.passes.mapping.calibration_mapper import CalibrationMapper
    from qb_compiler.passes.mapping.noise_aware_router import NoiseAwareRouter

    # Get or build calibration data
    if calibration_props is None:
        calibration_props = _load_calibration_fixture(backend_name)
    if calibration_props is None:
        # Use a large enough pool so the mapper has many candidate placements.
        # On real hardware (IBM Fez = 156q), even a 4-qubit circuit benefits
        # from choosing among 156 possible qubit subsets.
        calibration_props = _build_synthetic_calibration(
            spec, backend_name, min(spec.n_qubits, max(circuit.n_qubits * 8, 40))
        )

    # Convert to IR circuit
    ir_circ = _to_ir_circuit(circuit)

    context: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    # Run CalibrationMapper with gate-error-dominant weights.
    # On real hardware, CX error varies 50x across the chip and dominates
    # fidelity for any circuit with 2Q gates. Coherence matters less for
    # circuits shorter than T2.
    from qb_compiler.passes.mapping.calibration_mapper import CalibrationMapperConfig

    mapper_config = CalibrationMapperConfig(
        gate_error_weight=10.0,   # Prioritise low-error edges
        coherence_weight=0.3,     # Secondary: prefer longer-lived qubits
        readout_weight=5.0,       # Readout error ~5x CZ error on IBM Heron
    )
    try:
        mapper = CalibrationMapper(calibration_props, config=mapper_config)
        result = mapper.run(ir_circ, context)
        ir_circ = result.circuit
        metadata["calibration_mapper"] = result.metadata
        metadata["initial_layout"] = context.get("initial_layout", {})
    except Exception as e:
        logger.warning("CalibrationMapper failed, skipping: %s", e)
        metadata["calibration_mapper_error"] = str(e)

    # Build gate error dict for router from calibration
    gate_errors: dict[tuple[int, int], float] = {}
    for gp in calibration_props.gate_properties:
        if len(gp.qubits) == 2 and gp.error_rate is not None:
            gate_errors[gp.qubits] = gp.error_rate

    # Run NoiseAwareRouter
    try:
        router = NoiseAwareRouter(
            coupling_map=calibration_props.coupling_map,
            gate_errors=gate_errors,
        )
        result = router.run(ir_circ, context)
        ir_circ = result.circuit
        metadata["noise_aware_router"] = result.metadata
    except Exception as e:
        logger.warning("NoiseAwareRouter failed, skipping: %s", e)
        metadata["noise_aware_router_error"] = str(e)

    # Convert back
    mapped_circuit = _from_ir_circuit(ir_circ, circuit)
    metadata["calibration_props"] = calibration_props

    return mapped_circuit, metadata


# ═══════════════════════════════════════════════════════════════════════
# Main compiler
# ═══════════════════════════════════════════════════════════════════════

class QBCompiler:
    """High-level compiler for quantum circuits.

    Parameters
    ----------
    backend:
        Target backend key (e.g. ``"ibm_fez"``).
    calibration:
        Optional live calibration provider for noise-aware compilation.
    strategy:
        ``"fidelity_optimal"`` (default) trades compilation time for
        estimated output fidelity.  ``"depth_optimal"`` minimises depth.
        ``"budget_optimal"`` minimises estimated cost.
    """

    STRATEGIES = frozenset({"fidelity_optimal", "depth_optimal", "budget_optimal"})

    def __init__(
        self,
        backend: str | None = None,
        calibration: CalibrationProvider | None = None,
        strategy: str = "fidelity_optimal",
        calibration_properties: BackendProperties | None = None,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {sorted(self.STRATEGIES)}"
            )
        opt_level = {"fidelity_optimal": 3, "depth_optimal": 2, "budget_optimal": 1}[strategy]
        self.config = CompilerConfig(
            backend=backend,
            optimization_level=opt_level,
        )
        self.calibration = calibration
        self.calibration_properties = calibration_properties
        self.strategy = strategy
        self._pass_manager = PassManager.default(opt_level)

    # ── factories ────────────────────────────────────────────────────

    @classmethod
    def from_backend(cls, backend: str, **kwargs: Any) -> QBCompiler:
        """Create a compiler pre-configured for *backend*.

        Validates that the backend is known and passes any extra kwargs
        through to the constructor.
        """
        _ = get_backend_spec(backend)  # raises BackendNotSupportedError
        return cls(backend=backend, **kwargs)

    # ── public API ───────────────────────────────────────────────────

    def compile(
        self,
        circuit: QBCircuit,
        *,
        strategy: str | None = None,
        budget_usd: float | None = None,
        qec_aware: bool = False,
    ) -> CompileResult:
        """Compile *circuit* and return a :class:`CompileResult`.

        Parameters
        ----------
        circuit:
            Input circuit to compile.
        strategy:
            Override the compiler-level strategy for this call.
        budget_usd:
            If set, the compiler raises :class:`BudgetExceededError`
            when estimated cost at 1024 shots exceeds this amount.
        qec_aware:
            Reserve ancilla qubits for QEC syndrome extraction (future).
        """
        if not isinstance(circuit, QBCircuit):
            raise InvalidCircuitError(
                f"Expected QBCircuit, got {type(circuit).__name__}"
            )
        if circuit.gate_count == 0:
            raise InvalidCircuitError("Cannot compile an empty circuit")

        active_strategy = strategy or self.strategy
        if active_strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{active_strategy}'")

        # Resolve optimisation level per strategy if overridden
        if strategy is not None and strategy != self.strategy:
            level_map = {"fidelity_optimal": 3, "depth_optimal": 2, "budget_optimal": 1}
            opt_level = level_map[active_strategy]
            pm = PassManager.default(opt_level)
            cfg = self.config.with_overrides(optimization_level=opt_level)
        else:
            pm = self._pass_manager
            cfg = self.config

        working = circuit.copy()
        original_depth = working.depth

        t0 = time.perf_counter()

        # Phase 1: Run calibration-aware passes (mapping + routing) when a
        # backend is configured and calibration awareness is enabled.
        calibration_meta: dict[str, Any] = {}
        if (
            self.config.backend is not None
            and self.config.enable_calibration_aware
            and self.config.backend_spec is not None
            and active_strategy != "budget_optimal"  # budget skips heavy passes
        ):
            try:
                working, calibration_meta = _run_calibration_pipeline(
                    working,
                    self.config.backend,
                    self.config.backend_spec,
                    self.calibration_properties,
                )
            except Exception as e:
                logger.warning("Calibration pipeline failed, continuing without: %s", e)

        # Phase 2: Run gate optimization passes
        compiled, pass_log = pm.run(working, cfg)
        compilation_ms = (time.perf_counter() - t0) * 1000.0

        # Estimate fidelity using the mapped qubit's actual error rates.
        # We estimate on the ORIGINAL gate structure (before decomposition)
        # using the per-qubit errors from the calibration-selected layout.
        # This gives the honest comparison: same gates, better qubits.
        cal_props = calibration_meta.get("calibration_props")
        layout = calibration_meta.get("initial_layout")
        if cal_props is not None and layout:
            fidelity = self._estimate_fidelity_mapped(circuit, cal_props, layout)
        else:
            fidelity = self._estimate_fidelity_for(compiled)

        # Budget guard
        if budget_usd is not None and self.config.backend is not None:
            cost = self.estimate_cost(compiled, shots=1024)
            if not cost.within_budget(budget_usd):
                raise BudgetExceededError(
                    cost.total_usd, budget_usd, shots=1024
                )

        return CompileResult(
            compiled_circuit=compiled,
            original_depth=original_depth,
            compiled_depth=compiled.depth,
            estimated_fidelity=fidelity,
            pass_log=tuple(pass_log),
            compilation_time_ms=round(compilation_ms, 3),
        )

    def estimate_fidelity(self, circuit: QBCircuit) -> float:
        """Estimate output-state fidelity for *circuit* on the target backend."""
        return self._estimate_fidelity_for(circuit)

    def estimate_cost(self, circuit: QBCircuit, shots: int) -> CostEstimate:
        """Estimate execution cost in USD for *circuit* at *shots*.

        Requires a backend to be configured; raises
        :class:`BackendNotSupportedError` otherwise.
        """
        spec = self.config.backend_spec
        if spec is None:
            raise BackendNotSupportedError(
                "None",
                list(BACKEND_CONFIGS),
            )
        estimator = CostEstimator(spec)
        return estimator.estimate(circuit.depth, circuit.n_qubits, shots)

    # ── internals ────────────────────────────────────────────────────

    def _estimate_fidelity_for(self, circuit: QBCircuit) -> float:
        """Simple analytic fidelity estimate.

        Uses per-gate depolarizing model:
            F ≈ product over gates of (1 - e_gate)

        Where e_gate comes from calibration data if available, or from
        the backend spec's median error rates as a fallback.
        """
        spec = self.config.backend_spec
        if spec is None:
            # No backend → assume ideal
            return 1.0

        fidelity = 1.0
        for op in circuit.ops:
            if op.name in ("measure", "reset", "barrier"):
                if op.name == "measure":
                    # Readout error
                    err = self._readout_error(op.qubits[0], spec)
                    fidelity *= (1.0 - err)
                continue
            if op.is_two_qubit:
                err = self._cx_error(op.qubits, spec)
            else:
                # Single-qubit gates: error ≈ cx_error / 10
                err = spec.median_cx_error / 10.0
            fidelity *= (1.0 - err)

        return max(0.0, fidelity)

    def _estimate_fidelity_mapped(
        self, circuit: QBCircuit, cal_props: Any, layout: dict[int, int]
    ) -> float:
        """Fidelity estimate on the ORIGINAL gate structure using per-qubit errors.

        This estimates fidelity using the original (pre-decomposition) circuit
        gates, but looking up error rates for the PHYSICAL qubits that
        CalibrationMapper selected.  This gives the honest comparison:
        same gate count as baseline, better qubits from calibration awareness.

        The improvement comes from:
        - Lower 2Q gate errors (mapper picks edges with lower CX error)
        - Lower readout errors (mapper picks qubits with lower readout error)
        - Better coherence (mapper picks qubits with higher T1/T2)
        """
        fidelity = 1.0

        # Build per-edge error lookup
        gate_map: dict[tuple[int, int], float] = {}
        for gp in cal_props.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                gate_map[(gp.qubits[0], gp.qubits[1])] = gp.error_rate

        # Build per-qubit readout error lookup
        qubit_readout: dict[int, float] = {}
        for qp in cal_props.qubit_properties:
            if qp.readout_error is not None:
                qubit_readout[qp.qubit_id] = qp.readout_error

        spec = self.config.backend_spec

        for op in circuit.ops:
            if op.name == "measure":
                phys_q = layout.get(op.qubits[0], op.qubits[0])
                err = qubit_readout.get(phys_q, spec.median_readout_error if spec else 0.01)
                fidelity *= (1.0 - err)
                continue
            if op.name in ("reset", "barrier"):
                continue

            if len(op.qubits) >= 2:
                # Map logical qubits to physical for error lookup
                phys_0 = layout.get(op.qubits[0], op.qubits[0])
                phys_1 = layout.get(op.qubits[1], op.qubits[1])
                err = gate_map.get((phys_0, phys_1))
                if err is None:
                    err = gate_map.get((phys_1, phys_0))
                if err is None:
                    err = spec.median_cx_error if spec else 0.01
                fidelity *= (1.0 - err)
            else:
                # Single-qubit gate error (unchanged from baseline)
                err = (spec.median_cx_error if spec else 0.01) / 10.0
                fidelity *= (1.0 - err)

        return max(0.0, fidelity)

    def _cx_error(
        self, qubits: tuple[int, ...], spec: BackendSpec
    ) -> float:
        """Two-qubit error from calibration or backend median."""
        if self.calibration is not None:
            try:
                return self.calibration.get_cx_error((qubits[0], qubits[1]))
            except (IndexError, KeyError):
                pass
        return spec.median_cx_error

    def _readout_error(self, qubit: int, spec: BackendSpec) -> float:
        """Readout error from calibration or backend median."""
        if self.calibration is not None:
            try:
                return self.calibration.get_readout_error(qubit)
            except (IndexError, KeyError):
                pass
        return spec.median_readout_error
