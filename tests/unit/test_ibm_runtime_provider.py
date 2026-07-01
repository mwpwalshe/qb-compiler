"""Offline tests for the IBM live-calibration provider.

No IBM creds or qiskit-ibm-runtime: we exercise the pure parser and the offline ``from_target``
path with a hand-built qiskit ``Target`` (qiskit is a core dependency).
"""

from __future__ import annotations

import pytest

from qb_compiler.calibration.ibm_runtime_provider import (
    IBMRuntimeCalibrationProvider,
    parse_ibm_target,
)


def _build_target():
    from qiskit.circuit.library import CXGate, Measure, XGate
    from qiskit.providers.backend import QubitProperties
    from qiskit.transpiler import InstructionProperties, Target

    target = Target(
        num_qubits=2,
        qubit_properties=[
            QubitProperties(t1=100e-6, t2=80e-6, frequency=5e9),
            QubitProperties(t1=90e-6, t2=70e-6, frequency=5.1e9),
        ],
    )
    target.add_instruction(
        XGate(),
        {
            (0,): InstructionProperties(duration=35e-9, error=2e-4),
            (1,): InstructionProperties(duration=35e-9, error=3e-4),
        },
    )
    target.add_instruction(CXGate(), {(0, 1): InstructionProperties(duration=3e-7, error=6e-3)})
    target.add_instruction(
        Measure(),
        {
            (0,): InstructionProperties(duration=1e-6, error=1e-2),
            (1,): InstructionProperties(duration=1e-6, error=1.5e-2),
        },
    )
    return target


def test_parse_units_and_structure():
    props = parse_ibm_target(_build_target(), backend="ibm_test")
    assert props.n_qubits == 2
    assert props.provider == "ibm"
    assert "x" in props.basis_gates and "cx" in props.basis_gates
    assert "measure" not in props.basis_gates  # non-gate op filtered
    assert sorted(props.coupling_map) == [(0, 1)]
    q0 = props.qubit_properties[0]
    assert q0.t1_us == pytest.approx(100.0)  # 100e-6 s -> us
    assert q0.t2_us == pytest.approx(80.0)
    assert q0.frequency_ghz == pytest.approx(5.0)  # 5e9 Hz -> GHz
    assert q0.readout_error == pytest.approx(1e-2)  # from measure instruction


def test_parse_gate_error_and_duration():
    props = parse_ibm_target(_build_target(), backend="ibm_test")
    cx = next(g for g in props.gate_properties if g.gate_type == "cx" and g.qubits == (0, 1))
    assert cx.error_rate == pytest.approx(6e-3)
    assert cx.gate_time_ns == pytest.approx(300.0)  # 3e-7 s -> ns
    x0 = next(g for g in props.gate_properties if g.gate_type == "x" and g.qubits == (0,))
    assert x0.error_rate == pytest.approx(2e-4)


def test_from_target_lookups():
    prov = IBMRuntimeCalibrationProvider.from_target(_build_target(), backend="ibm_test")
    assert prov.backend_name == "ibm_test"
    assert prov.backend_properties.n_qubits == 2
    q1 = prov.get_qubit_properties(1)
    assert q1 is not None and q1.t1_us == pytest.approx(90.0)
    g = prov.get_gate_properties("cx", (0, 1))
    assert g is not None and g.error_rate == pytest.approx(6e-3)
    assert len(prov.get_all_qubit_properties()) == 2


def test_missing_calibration_is_graceful():
    from qiskit.transpiler import Target

    # a Target with no qubit_properties and no instructions -> no crash, empty calibration
    props = parse_ibm_target(Target(num_qubits=3), backend="bare")
    assert props.n_qubits == 3
    assert all(q.t1_us is None for q in props.qubit_properties)
    assert props.gate_properties == []
