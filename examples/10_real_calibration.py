"""Example 10: Working with real IBM Fez calibration data.

Loads a real calibration snapshot from IBM Fez (156 qubits) and
demonstrates how calibration data drives compilation decisions.
"""

from pathlib import Path

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.static_provider import StaticCalibrationProvider
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.calibration_mapper import (
    CalibrationMapper,
    CalibrationMapperConfig,
)

# Look for calibration data
CAL_DIR = Path(__file__).resolve().parent.parent / "tests/fixtures/calibration_snapshots"


def find_calibration_file(backend: str) -> Path | None:
    """Find a calibration snapshot for the given backend."""
    for f in CAL_DIR.glob("*.json"):
        if backend.replace("_", "") in f.stem.replace("_", ""):
            return f
    return None


def explore_calibration(props: BackendProperties) -> None:
    """Print key statistics from calibration data."""
    print(f"\nBackend: {props.backend}")
    print(f"Qubits:  {props.n_qubits}")
    print(f"Time:    {props.timestamp}")
    print(f"Basis:   {props.basis_gates}")

    # Qubit quality distribution
    readout_errors = [qp.readout_error for qp in props.qubit_properties if qp.readout_error]
    t1_values = [qp.t1_us for qp in props.qubit_properties if qp.t1_us]

    if readout_errors:
        print(f"\nReadout error:")
        print(f"  Best:   {min(readout_errors):.4f}")
        print(f"  Worst:  {max(readout_errors):.4f}")
        print(f"  Median: {sorted(readout_errors)[len(readout_errors)//2]:.4f}")
        print(f"  Range:  {max(readout_errors)/max(min(readout_errors), 1e-9):.0f}x")

    if t1_values:
        print(f"\nT1 (μs):")
        print(f"  Best:   {max(t1_values):.1f}")
        print(f"  Worst:  {min(t1_values):.1f}")
        print(f"  Median: {sorted(t1_values)[len(t1_values)//2]:.1f}")

    # Gate error distribution
    two_q_errors = [
        gp.error_rate
        for gp in props.gate_properties
        if len(gp.qubits) == 2 and gp.error_rate is not None
    ]
    if two_q_errors:
        print(f"\n2Q gate error:")
        print(f"  Best:   {min(two_q_errors):.6f}")
        print(f"  Worst:  {max(two_q_errors):.6f}")
        print(f"  Median: {sorted(two_q_errors)[len(two_q_errors)//2]:.6f}")
        print(f"  Range:  {max(two_q_errors)/max(min(two_q_errors), 1e-12):.0f}x")

    # Top 10 best qubits
    ranked = sorted(props.qubit_properties, key=lambda q: q.readout_error or 1.0)
    print(f"\nTop 10 qubits (lowest readout error):")
    for qp in ranked[:10]:
        print(
            f"  Q{qp.qubit_id:3d}: readout={qp.readout_error:.4f}  "
            f"T1={qp.t1_us:6.1f}μs  T2={qp.t2_us:6.1f}μs"
        )


def compare_layouts(props: BackendProperties) -> None:
    """Compare different CalibrationMapper configurations on a GHZ-5 circuit."""
    circ = QBCircuit(n_qubits=5, n_clbits=5, name="ghz-5")
    circ.add_gate(QBGate("h", (0,)))
    for i in range(4):
        circ.add_gate(QBGate("cx", (i, i + 1)))
    for i in range(5):
        circ.add_measurement(i, i)

    configs = {
        "balanced": CalibrationMapperConfig(
            gate_error_weight=10.0, coherence_weight=0.3, readout_weight=5.0,
        ),
        "gate-error focused": CalibrationMapperConfig(
            gate_error_weight=10.0, coherence_weight=0.1, readout_weight=1.0,
        ),
        "readout focused": CalibrationMapperConfig(
            gate_error_weight=1.0, coherence_weight=0.1, readout_weight=10.0,
        ),
    }

    print(f"\nGHZ-5 layout comparison:")
    print(f"{'Config':25s}  {'Layout':35s}  {'Score':>10s}")
    print("-" * 75)

    for name, config in configs.items():
        mapper = CalibrationMapper(calibration=props, config=config)
        ctx: dict = {}
        mapper.run(circ, ctx)
        layout = ctx.get("initial_layout", {})
        score = ctx.get("calibration_score", 0)
        layout_str = " -> ".join(f"Q{layout[i]}" for i in range(5))
        print(f"{name:25s}  {layout_str:35s}  {score:10.6f}")


def main() -> None:
    cal_file = find_calibration_file("ibm_fez")
    if cal_file is None:
        print(f"No calibration snapshot found in {CAL_DIR}")
        print("Available files:")
        for f in CAL_DIR.glob("*.json"):
            print(f"  {f.name}")
        return

    print(f"Loading: {cal_file.name}")
    provider = StaticCalibrationProvider.from_json(cal_file)
    props = provider.properties

    explore_calibration(props)
    compare_layouts(props)


if __name__ == "__main__":
    main()
