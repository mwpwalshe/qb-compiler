"""Circuit fidelity estimation from noise models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qb_compiler.noise.noise_model import NoiseModel

# ── lightweight circuit representation ───────────────────────────────
# The full QBCircuit IR is being built in a separate module.  For now,
# FidelityEstimator operates on the minimal contract below so it can be
# used as soon as the IR is ready.


@dataclass(frozen=True, slots=True)
class QBCircuit:
    """Minimal circuit descriptor for fidelity estimation.

    Parameters
    ----------
    gates:
        Ordered sequence of gate operations.  Each element is a
        ``(gate_name, qubit_tuple)`` pair.
    n_qubits:
        Total number of qubits in the circuit.
    measurements:
        Set of qubit indices that are measured at the end.
    """

    gates: list[tuple[str, tuple[int, ...]]]
    n_qubits: int
    measurements: frozenset[int] = frozenset()


class FidelityEstimator:
    """Estimates expected output fidelity of a circuit under a noise model.

    The estimation works in two stages:

    1. **Gate fidelity product** — multiply ``(1 - error)`` for every gate
       in the circuit.
    2. **Decoherence penalty** — for each qubit, compute the total time it
       is active (sum of gate durations on its critical path) and apply
       the T1/T2 decoherence factor.
    3. **Readout penalty** — multiply by ``(1 - readout_error)`` for every
       measured qubit.
    """

    def __init__(self, *, default_gate_time_ns: float = 40.0) -> None:
        self._default_gate_time_ns = default_gate_time_ns

    def estimate(self, circuit: QBCircuit, noise_model: NoiseModel) -> float:
        """Return estimated output fidelity in [0, 1].

        Parameters
        ----------
        circuit:
            Circuit to analyse.
        noise_model:
            Noise model providing per-gate and per-qubit error rates.

        Returns
        -------
        float
            Estimated probability that the circuit produces the correct
            output bitstring on a single shot.  Values close to 1.0 are
            good; values below ~0.5 suggest the circuit is too deep for
            the hardware.
        """
        if not circuit.gates:
            return 1.0

        fidelity = 1.0

        # ── 1. Gate fidelity product ─────────────────────────────────
        # Track per-qubit accumulated gate time for decoherence
        qubit_time_ns: dict[int, float] = defaultdict(float)

        for gate_name, qubits in circuit.gates:
            err = noise_model.gate_error(gate_name, qubits)
            fidelity *= 1.0 - err

            # Accumulate gate time for decoherence calculation
            gate_time = self._gate_time(gate_name, qubits, noise_model)
            for q in qubits:
                qubit_time_ns[q] += gate_time

        # ── 2. Decoherence penalty ───────────────────────────────────
        # For each qubit, apply decoherence based on total active time.
        # We use the critical-path time (the qubit with the most
        # accumulated gate time determines depth, but each qubit's
        # decoherence is individual).
        for qubit in range(circuit.n_qubits):
            active_time = qubit_time_ns.get(qubit, 0.0)
            if active_time > 0.0:
                dec_err = noise_model.decoherence_factor(qubit, active_time)
                fidelity *= 1.0 - dec_err

        # ── 3. Readout penalty ───────────────────────────────────────
        measured = (
            circuit.measurements if circuit.measurements else frozenset(range(circuit.n_qubits))
        )
        for qubit in measured:
            ro_err = noise_model.readout_error(qubit)
            fidelity *= 1.0 - ro_err

        return max(0.0, fidelity)

    def _gate_time(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        noise_model: NoiseModel,
    ) -> float:
        """Try to get gate time from noise model, else use default."""
        # EmpiricalNoiseModel exposes gate_time_ns; duck-type check
        if hasattr(noise_model, "gate_time_ns"):
            return float(noise_model.gate_time_ns(gate_name, qubits))  # type: ignore[attr-defined]
        return self._default_gate_time_ns

    def estimate_depth_limited_fidelity(
        self,
        n_two_qubit_gates: int,
        avg_two_qubit_error: float,
        n_one_qubit_gates: int = 0,
        avg_one_qubit_error: float = 0.001,
    ) -> float:
        """Quick estimate without a full circuit — useful for planning.

        Parameters
        ----------
        n_two_qubit_gates:
            Number of 2-qubit gates.
        avg_two_qubit_error:
            Average 2-qubit gate error rate.
        n_one_qubit_gates:
            Number of 1-qubit gates.
        avg_one_qubit_error:
            Average 1-qubit gate error rate.

        Returns
        -------
        float
            Estimated fidelity (gate errors only, no decoherence/readout).
        """
        f2 = (1.0 - avg_two_qubit_error) ** n_two_qubit_gates
        f1 = (1.0 - avg_one_qubit_error) ** n_one_qubit_gates
        return f1 * f2
