"""Example 06: Full compilation pipeline end-to-end.

Demonstrates the complete flow: load calibration -> map logical qubits
to physical qubits -> route through lowest-error paths -> estimate
fidelity. Uses the IR-level passes directly for maximum control.
"""

from pathlib import Path

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.static_provider import StaticCalibrationProvider
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate, QBMeasure
from qb_compiler.passes.analysis.error_budget_estimator import ErrorBudgetEstimator
from qb_compiler.passes.mapping.calibration_mapper import CalibrationMapper
from qb_compiler.passes.mapping.noise_aware_router import NoiseAwareRouter

CAL_PATH = Path(__file__).resolve().parent.parent / (
    # Download sample calibration: https://github.com/mwpwalshe/qb-compiler/tree/main/tests/fixtures/calibration_snapshots
    "tests/fixtures/calibration_snapshots/ibm_fez_2026_02_15.json"  # Or use your own: qb calibrate fetch ibm_fez
)


def main() -> None:
    # Step 1: Load calibration
    provider = StaticCalibrationProvider.from_json(CAL_PATH)
    props = provider.properties
    print(f"[1] Loaded calibration: {props.backend}, {props.n_qubits} qubits")
    print(f"    Timestamp: {props.timestamp}")

    # Step 2: Build a 4-qubit circuit (linear entanglement)
    circ = QBCircuit(n_qubits=4, n_clbits=4, name="linear-entangle-4")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    for i in range(3):
        circ.add_gate(QBGate(name="cx", qubits=(i, i + 1)))
    for i in range(4):
        circ.add_measurement(i, i)

    print(f"\n[2] Original circuit: depth={circ.depth}, gates={circ.gate_count}")

    # Step 3: Calibration-aware mapping
    mapper = CalibrationMapper(props)
    ctx: dict = {}
    map_result = mapper.run(circ, ctx)
    mapped = map_result.circuit
    layout = ctx["initial_layout"]

    print(f"\n[3] Mapped: {dict(sorted(layout.items()))}")
    print(f"    Score: {ctx['calibration_score']:.6f}")
    print(f"    Physical qubits: {sorted(mapped.qubits_used())}")

    # Step 4: Noise-aware routing
    gate_errors: dict[tuple[int, int], float] = {}
    for gp in props.gate_properties:
        if len(gp.qubits) == 2 and gp.error_rate is not None:
            gate_errors[gp.qubits] = gp.error_rate

    router = NoiseAwareRouter(
        coupling_map=props.coupling_map,
        gate_errors=gate_errors,
    )
    route_result = router.run(mapped, ctx)
    routed = route_result.circuit

    print(f"\n[4] Routed: depth={routed.depth}, gates={routed.gate_count}")
    print(f"    SWAPs inserted: {ctx.get('swaps_inserted', 0)}")

    # Step 5: Fidelity estimation
    gate_err_rates = {"h": 0.0003, "cx": 0.005, "sx": 0.0003, "rz": 0.0}
    estimator = ErrorBudgetEstimator(
        qubit_properties=props.qubit_properties,
        gate_error_rates=gate_err_rates,
    )
    estimator.analyze(routed, ctx)

    fidelity = ctx["estimated_fidelity"]
    budget = ctx["error_budget"]

    print(f"\n[5] Fidelity estimation:")
    print(f"    Estimated fidelity: {fidelity:.4f}")
    print(f"    Error breakdown:")
    print(f"      Gate errors:   {budget['gate']:.6f}")
    print(f"      Decoherence:   {budget['decoherence']:.6f}")
    print(f"      Readout:       {budget['readout']:.6f}")

    print(f"\n{'='*50}")
    print(f"Pipeline summary: {circ.gate_count} gates -> {routed.gate_count} gates, "
          f"fidelity={fidelity:.4f}")


if __name__ == "__main__":
    main()
