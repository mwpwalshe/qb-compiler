"""Circuit viability checking.

Determines whether a quantum circuit is likely to produce meaningful
results on a given backend, **before** spending QPU time.

A circuit is "not viable" when its estimated fidelity is so low that
the output distribution is indistinguishable from noise.  This module
saves real money: every job submitted below the viability threshold
is wasted QPU time.

Usage::

    from qb_compiler.viability import check_viability

    result = check_viability(circuit, backend="ibm_fez")
    if not result.viable:
        print(result.reason)
        print(result.suggestions)
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ViabilityResult:
    """Outcome of a viability check.

    Attributes
    ----------
    viable :
        ``True`` if the circuit is likely to produce meaningful results.
    status :
        Short human-readable status: ``"VIABLE"``, ``"MARGINAL"``, or
        ``"NOT VIABLE"``.
    circuit_name :
        Name or description of the circuit.
    backend :
        Target backend name.
    estimated_fidelity :
        Multiplicative fidelity estimate from calibration data.
    noise_floor :
        Probability of the correct answer from random guessing
        (``1 / 2**n_qubits`` for most circuits).
    signal_to_noise :
        ``estimated_fidelity / noise_floor``.  Values below 2.0 mean
        the signal is buried in noise.
    two_qubit_gate_count :
        Number of two-qubit gates after transpilation.
    depth :
        Circuit depth after transpilation.
    viable_depth :
        Estimated maximum useful depth for this backend at current
        error rates.
    reason :
        Human-readable explanation of why the circuit is/isn't viable.
    suggestions :
        Actionable suggestions to improve viability.
    cost_estimate_usd :
        Estimated cost for a standard 4096-shot run.
    """

    viable: bool
    status: str
    circuit_name: str
    backend: str
    estimated_fidelity: float
    noise_floor: float
    signal_to_noise: float
    two_qubit_gate_count: int
    depth: int
    viable_depth: int
    reason: str
    suggestions: list[str] = field(default_factory=list)
    cost_estimate_usd: float | None = None

    def __str__(self) -> str:
        lines = [
            f"Circuit: {self.circuit_name}",
            f"Backend: {self.backend}",
            f"Estimated fidelity: {self.estimated_fidelity:.4f}",
            f"Noise floor: {self.noise_floor:.4f}",
            f"Signal/noise ratio: {self.signal_to_noise:.1f}x",
            f"Status: {self.status}",
            f"Reason: {self.reason}",
        ]
        if self.cost_estimate_usd is not None:
            lines.append(f"Cost (4096 shots): ${self.cost_estimate_usd:.4f}")
        if self.suggestions:
            lines.append("Suggestions:")
            for s in self.suggestions:
                lines.append(f"  - {s}")
        return "\n".join(lines)


def _estimate_viable_depth(
    median_2q_error: float,
    median_readout_error: float,
    n_qubits: int,
) -> int:
    """Estimate the maximum circuit depth that produces useful results.

    A circuit is viable when its fidelity is well above the noise floor.
    We require fidelity > 2 / 2^n (i.e., signal-to-noise > 2).

    For a circuit of depth *d* with ~d two-qubit gates and n measurements:
        F ≈ (1 - e_2q)^d x (1 - e_ro)^n

    Solving for d where F = 2 / 2^n:
        d = (ln(2) - n·ln(1-e_ro) + n·ln(2)) / (-ln(1-e_2q))
    """
    if median_2q_error <= 0 or median_2q_error >= 1:
        return 1000  # can't estimate, be optimistic

    noise_floor = 1.0 / (2 ** n_qubits)
    target_fidelity = max(2.0 * noise_floor, 0.01)  # at least 1%

    readout_fidelity = (1.0 - median_readout_error) ** n_qubits
    if readout_fidelity <= target_fidelity:
        return 0  # readout alone kills viability

    # Remaining fidelity budget after readout
    gate_fidelity_budget = target_fidelity / readout_fidelity
    # (1 - e_2q)^d = gate_fidelity_budget
    # d = ln(gate_fidelity_budget) / ln(1 - e_2q)
    viable_d = math.log(gate_fidelity_budget) / math.log(1.0 - median_2q_error)
    return max(1, int(viable_d))


def _count_2q(tc: Any) -> int:
    """Count two-qubit gates in a Qiskit QuantumCircuit."""
    return sum(
        1 for inst in tc.data
        if len(inst.qubits) == 2
        and inst.operation.name not in ("barrier", "measure", "reset")
    )


def _estimate_routed_fidelity(
    tc: Any,
    backend_props: Any,
    median_2q_error: float,
    median_readout_error: float,
) -> float:
    """Estimate fidelity of a routed circuit using per-edge calibration."""
    if backend_props is not None:
        gate_map: dict[frozenset[int], float] = {}
        for gp in backend_props.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                gate_map[frozenset(gp.qubits)] = gp.error_rate

        qubit_ro: dict[int, float] = {}
        for qp in backend_props.qubit_properties:
            if qp.readout_error is not None:
                qubit_ro[qp.qubit_id] = qp.readout_error

        fidelity = 1.0
        for inst in tc.data:
            if inst.operation.name in ("barrier", "reset", "delay"):
                continue
            if inst.operation.name == "measure":
                q = tc.find_bit(inst.qubits[0]).index
                fidelity *= (1.0 - qubit_ro.get(q, median_readout_error))
            elif len(inst.qubits) == 2 and inst.operation.name not in (
                "barrier", "measure", "reset",
            ):
                q0 = tc.find_bit(inst.qubits[0]).index
                q1 = tc.find_bit(inst.qubits[1]).index
                err = gate_map.get(frozenset({q0, q1}), median_2q_error)
                fidelity *= (1.0 - err)
        return max(0.0, fidelity)

    # Fallback: use median errors
    fidelity = 1.0
    for inst in tc.data:
        if inst.operation.name == "measure":
            fidelity *= (1.0 - median_readout_error)
        elif (
            len(inst.qubits) == 2
            and inst.operation.name not in ("barrier", "measure", "reset")
        ):
            fidelity *= (1.0 - median_2q_error)
    return max(0.0, fidelity)


def check_viability(
    circuit: Any,
    *,
    backend: str | None = None,
    backend_props: Any | None = None,
    qiskit_target: Any | None = None,
    n_seeds: int = 10,
    viability_threshold: float = 2.0,
    shots: int = 4096,
) -> ViabilityResult:
    """Check whether *circuit* will produce meaningful results on *backend*.

    Parameters
    ----------
    circuit :
        A Qiskit ``QuantumCircuit`` to check.
    backend :
        Target backend name (e.g. ``"ibm_fez"``).  Used for pricing
        and median error rates if *backend_props* is not given.
    backend_props :
        Calibration data (:class:`BackendProperties`).  When provided,
        per-qubit error rates are used for a more accurate estimate.
    qiskit_target :
        Qiskit ``Target`` for transpilation.  If ``None``, one is
        built from *backend_props* or backend defaults.
    n_seeds :
        Number of Qiskit transpiler seeds to try.
    viability_threshold :
        Minimum signal-to-noise ratio to be considered viable.
        Default 2.0 means estimated fidelity must be at least 2x
        the noise floor.
    shots :
        Number of shots for cost estimation.

    Returns
    -------
    ViabilityResult
        Full viability assessment with actionable suggestions.
    """
    from qiskit import transpile

    from qb_compiler.config import get_backend_spec
    from qb_compiler.cost.pricing import get_pricing

    # Resolve backend spec
    spec = None
    if backend is not None:
        with contextlib.suppress(Exception):
            spec = get_backend_spec(backend)

    median_2q_error = spec.median_cx_error if spec else 0.005
    median_ro_error = spec.median_readout_error if spec else 0.01

    # Resolve target for transpilation
    if qiskit_target is None and backend_props is not None:
        qiskit_target = _build_target_from_props(backend_props)

    if qiskit_target is None and spec is not None:
        # Load calibration fixture if available
        from qb_compiler.compiler import _load_calibration_fixture
        loaded_props = _load_calibration_fixture(backend or "")
        if loaded_props is not None:
            backend_props = loaded_props
            qiskit_target = _build_target_from_props(loaded_props)

    if qiskit_target is None:
        # Can't transpile — estimate from circuit structure alone
        return _estimate_without_transpile(
            circuit, backend or "unknown", median_2q_error, median_ro_error,
            viability_threshold, shots,
        )

    # Transpile with N seeds, pick best
    best_tc = None
    best_2q: int | float = float("inf")
    for seed in range(n_seeds):
        tc = transpile(
            circuit, target=qiskit_target,
            optimization_level=3, seed_transpiler=seed,
        )
        c2q = _count_2q(tc)
        if c2q < best_2q:
            best_2q = c2q
            best_tc = tc

    if best_tc is None:
        raise ValueError("Transpilation failed: no valid result from any seed")

    n_qubits = circuit.num_qubits
    depth = best_tc.depth()

    # Estimate fidelity
    fidelity = _estimate_routed_fidelity(
        best_tc, backend_props, median_2q_error, median_ro_error,
    )

    # Noise floor and viable depth
    noise_floor = 1.0 / (2 ** n_qubits)
    snr = fidelity / noise_floor if noise_floor > 0 else float("inf")
    viable_depth = _estimate_viable_depth(
        median_2q_error, median_ro_error, n_qubits,
    )

    # Cost
    cost = None
    pricing = get_pricing(backend) if backend else None
    if pricing:
        cost = pricing.cost_per_shot_usd * shots

    # Determine status
    if snr >= viability_threshold and fidelity >= 0.01:
        if fidelity >= 0.3:
            status = "VIABLE"
            viable = True
            reason = (
                f"Estimated fidelity {fidelity:.3f} is well above noise floor "
                f"({noise_floor:.4f}). Circuit should produce meaningful results."
            )
        else:
            status = "MARGINAL"
            viable = True
            reason = (
                f"Estimated fidelity {fidelity:.3f} is above noise floor but low. "
                f"Results will be noisy — consider error mitigation."
            )
    else:
        status = "NOT VIABLE"
        viable = False
        reason = (
            f"Estimated fidelity {fidelity:.4f} is too close to noise floor "
            f"({noise_floor:.4f}). Output is indistinguishable from random noise."
        )

    # Build suggestions
    suggestions = _build_suggestions(
        fidelity, snr, depth, viable_depth, int(best_2q), n_qubits, status,
    )

    circuit_name = getattr(circuit, "name", None) or f"{n_qubits}q circuit"

    return ViabilityResult(
        viable=viable,
        status=status,
        circuit_name=circuit_name,
        backend=backend or "unknown",
        estimated_fidelity=round(fidelity, 6),
        noise_floor=round(noise_floor, 6),
        signal_to_noise=round(snr, 2),
        two_qubit_gate_count=int(best_2q),
        depth=depth,
        viable_depth=viable_depth,
        reason=reason,
        suggestions=suggestions,
        cost_estimate_usd=round(cost, 4) if cost else None,
    )


def _estimate_without_transpile(
    circuit: Any,
    backend: str,
    median_2q_error: float,
    median_ro_error: float,
    viability_threshold: float,
    shots: int,
) -> ViabilityResult:
    """Rough viability estimate when we can't transpile."""
    n_qubits = circuit.num_qubits
    ops = circuit.count_ops()

    # Count 2Q gates from the unrouted circuit (underestimate — routing adds SWAPs)
    cx_count = sum(v for k, v in ops.items() if k in ("cx", "cz", "cp", "swap", "ecr"))
    # Routing typically adds 30-100% more 2Q gates
    estimated_2q = int(cx_count * 1.5)

    fidelity = ((1.0 - median_2q_error) ** estimated_2q
                * (1.0 - median_ro_error) ** n_qubits)
    fidelity = max(0.0, fidelity)

    noise_floor = 1.0 / (2 ** n_qubits)
    snr = fidelity / noise_floor if noise_floor > 0 else float("inf")
    viable_depth = _estimate_viable_depth(median_2q_error, median_ro_error, n_qubits)

    from qb_compiler.cost.pricing import get_pricing
    pricing = get_pricing(backend)
    cost = pricing.cost_per_shot_usd * shots if pricing else None

    if snr >= viability_threshold and fidelity >= 0.01:
        status = "VIABLE" if fidelity >= 0.3 else "MARGINAL"
        viable = True
        reason = (
            f"Rough estimate (no transpilation): fidelity ~{fidelity:.3f}. "
            f"Actual fidelity may be lower due to routing overhead."
        )
    else:
        status = "NOT VIABLE"
        viable = False
        reason = (
            f"Rough estimate (no transpilation): fidelity ~{fidelity:.4f}. "
            f"Even before routing, circuit is likely below noise floor."
        )

    circuit_name = getattr(circuit, "name", None) or f"{n_qubits}q circuit"
    return ViabilityResult(
        viable=viable,
        status=status,
        circuit_name=circuit_name,
        backend=backend,
        estimated_fidelity=round(fidelity, 6),
        noise_floor=round(noise_floor, 6),
        signal_to_noise=round(snr, 2),
        two_qubit_gate_count=estimated_2q,
        depth=circuit.depth(),
        viable_depth=viable_depth,
        reason=reason,
        suggestions=["Provide a Qiskit Target or calibration data for more accurate analysis."],
        cost_estimate_usd=round(cost, 4) if cost else None,
    )


def _build_suggestions(
    fidelity: float,
    snr: float,
    depth: int,
    viable_depth: int,
    two_q_count: int,
    n_qubits: int,
    status: str,
) -> list[str]:
    """Build actionable suggestions based on viability analysis."""
    suggestions: list[str] = []

    if depth > viable_depth:
        suggestions.append(
            f"Reduce circuit depth from {depth} to <{viable_depth} "
            f"(current depth exceeds viable limit for this backend)."
        )

    if status == "NOT VIABLE":
        suggestions.append(
            "Circuit output will be indistinguishable from noise. "
            "Do not submit — you will waste QPU time and money."
        )
        if two_q_count > 20:
            suggestions.append(
                f"Circuit has {two_q_count} two-qubit gates. Consider "
                f"circuit cutting or reducing problem size."
            )
        if n_qubits > 10:
            suggestions.append(
                "For >10 qubits at this depth, consider a simulator instead."
            )

    if status == "MARGINAL":
        suggestions.append(
            "Consider ZNE or PEC error mitigation (2-5x fidelity improvement possible)."
        )
        if fidelity < 0.1:
            suggestions.append(
                "Fidelity below 10% — error mitigation may not be sufficient. "
                "Consider reducing circuit depth."
            )

    if status == "VIABLE" and fidelity < 0.5:
        suggestions.append(
            "Good candidate for error mitigation to further improve results."
        )

    if not suggestions:
        suggestions.append("Circuit looks good — proceed with execution.")

    return suggestions


def _build_target_from_props(props: Any) -> Any:
    """Build a Qiskit Target from BackendProperties for transpilation."""
    from qiskit.circuit import Delay, Measure, Parameter
    from qiskit.circuit.library import (
        CXGate,
        CZGate,
        HGate,
        RZGate,
        SXGate,
        XGate,
    )
    from qiskit.transpiler import InstructionProperties, Target

    n_q = props.n_qubits
    target = Target(num_qubits=n_q, dt=2.2222222222222221e-10)

    # 2Q gates
    gate_types = {
        gp.gate_type for gp in props.gate_properties if len(gp.qubits) == 2
    }
    gate_cls = CZGate if "cz" in gate_types else CXGate

    twoq_props = {}
    for gp in props.gate_properties:
        if len(gp.qubits) == 2:
            dur = gp.gate_time_ns * 1e-9 if gp.gate_time_ns else 68e-9
            twoq_props[gp.qubits] = InstructionProperties(
                error=gp.error_rate, duration=dur,
            )
    target.add_instruction(gate_cls(), twoq_props)

    # 1Q gates
    sq_dur = 25e-9
    sq = {(q,): InstructionProperties(duration=sq_dur) for q in range(n_q)}
    theta = Parameter("theta")
    target.add_instruction(
        RZGate(theta),
        {(q,): InstructionProperties(duration=0) for q in range(n_q)},
    )
    target.add_instruction(SXGate(), sq)
    target.add_instruction(XGate(), sq)
    target.add_instruction(HGate(), sq)

    # Measure + delay (required for DD)
    meas = {(q,): InstructionProperties(duration=1.6e-6) for q in range(n_q)}
    target.add_instruction(Measure(), meas)
    delay_p = Parameter("t")
    target.add_instruction(
        Delay(delay_p), {(q,): None for q in range(n_q)}, name="delay",
    )

    return target
