"""Example 04: Error budget estimation.

Uses ErrorBudgetEstimator to predict circuit fidelity from calibration
data. Compares a circuit placed on good qubits vs bad qubits to show
how calibration-aware placement impacts expected output quality.
"""

from pathlib import Path

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate, QBMeasure
from qb_compiler.passes.analysis.error_budget_estimator import ErrorBudgetEstimator

CAL_PATH = Path(__file__).resolve().parent.parent / (
    "tests/fixtures/calibration_snapshots/ibm_fez_2026_02_15.json"
)


def build_bell_on(q0: int, q1: int, n_qubits: int) -> QBCircuit:
    """Build a Bell state + measurement on specific physical qubits."""
    circ = QBCircuit(n_qubits=n_qubits, n_clbits=2, name=f"bell_q{q0}_q{q1}")
    circ.add_gate(QBGate(name="h", qubits=(q0,)))
    circ.add_gate(QBGate(name="cx", qubits=(q0, q1)))
    circ.add_measurement(q0, 0)
    circ.add_measurement(q1, 1)
    return circ


def main() -> None:
    props = BackendProperties.from_qubitboost_json(CAL_PATH)

    # Rank qubits by T1 (higher is better)
    ranked = sorted(
        props.qubit_properties,
        key=lambda qp: qp.t1_us or 0,
        reverse=True,
    )
    best_pair = (ranked[0].qubit_id, ranked[1].qubit_id)
    worst_pair = (ranked[-1].qubit_id, ranked[-2].qubit_id)

    # Build gate error lookup
    gate_errors: dict[str, float] = {"h": 0.0003, "cx": 0.005}
    for gp in props.gate_properties:
        if gp.error_rate is not None:
            gate_errors.setdefault(gp.gate_type, gp.error_rate)

    n_phys = props.n_qubits

    for label, (q0, q1) in [("BEST", best_pair), ("WORST", worst_pair)]:
        circ = build_bell_on(q0, q1, n_phys)
        estimator = ErrorBudgetEstimator(
            qubit_properties=props.qubit_properties,
            gate_error_rates=gate_errors,
        )
        ctx: dict = {}
        estimator.analyze(circ, ctx)

        fidelity = ctx["estimated_fidelity"]
        budget = ctx["error_budget"]
        qp0 = props.qubit(q0)
        qp1 = props.qubit(q1)

        print(f"=== {label} Qubits: q{q0}, q{q1} ===")
        print(f"  T1:  {qp0.t1_us if qp0 else 'N/A'}, {qp1.t1_us if qp1 else 'N/A'} us")
        print(f"  Est. fidelity:       {fidelity:.4f}")
        print(f"  Gate error budget:   {budget['gate']:.6f}")
        print(f"  Decoherence budget:  {budget['decoherence']:.6f}")
        print(f"  Readout budget:      {budget['readout']:.6f}")
        print()

    print("Takeaway: Calibration-aware placement can significantly")
    print("impact circuit fidelity, even for identical logical circuits.")


if __name__ == "__main__":
    main()
