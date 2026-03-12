"""Core compiler entry point.

:class:`QBCompiler` orchestrates calibration loading, pass management,
fidelity estimation, and cost calculation for quantum circuit compilation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
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
    CompilationError,
    InvalidCircuitError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


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

    __slots__ = ("n_qubits", "ops", "metadata")

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

        if op.name == "cx" and len(q) == 2:
            if "cz" in basis_set:
                # CX = (I⊗H) CZ (I⊗H)  — decompose H if needed
                target = (q[1],)
                h_ops = _BasisTranslationPass._decompose(
                    GateOp("h", target), basis_set
                ) if "h" not in basis_set else [GateOp("h", target)]
                return h_ops + [GateOp("cz", q)] + h_ops

        if op.name == "cz" and len(q) == 2:
            if "cx" in basis_set:
                target = (q[1],)
                h_ops = _BasisTranslationPass._decompose(
                    GateOp("h", target), basis_set
                ) if "h" not in basis_set else [GateOp("h", target)]
                return h_ops + [GateOp("cx", q)] + h_ops

        if op.name == "x" and len(q) == 1:
            if "rx" in basis_set:
                return [GateOp("rx", q, (pi,))]
            if "sx" in basis_set:
                return [GateOp("sx", q, ()), GateOp("sx", q, ())]

        if op.name == "sx" and len(q) == 1:
            if "rx" in basis_set:
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
            opt_level = {"fidelity_optimal": 3, "depth_optimal": 2, "budget_optimal": 1}[active_strategy]
            pm = PassManager.default(opt_level)
            cfg = self.config.with_overrides(optimization_level=opt_level)
        else:
            pm = self._pass_manager
            cfg = self.config

        working = circuit.copy()
        original_depth = working.depth

        t0 = time.perf_counter()
        compiled, pass_log = pm.run(working, cfg)
        compilation_ms = (time.perf_counter() - t0) * 1000.0

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
