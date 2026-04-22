"""Fuzz calibration data parsing.

Targets ``BackendProperties.from_qubitboost_dict`` with:
- Missing fields, negative T1/T2, NaN/Inf error rates
- Coupling map with invalid indices
- Extremely large qubit counts
- Duplicate entries
"""

from __future__ import annotations

import math
import sys

import atheris

with atheris.instrument_imports():
    from qb_compiler.calibration.models.backend_properties import BackendProperties
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties


_SPECIAL_FLOATS = [0.0, -1.0, 1e-10, 1e30, math.inf, -math.inf, math.nan, -0.0]


def _make_qubit_dict(fdp: atheris.FuzzedDataProvider) -> dict:
    """Build a fuzzed qubit_properties entry."""
    d: dict = {}
    if fdp.ConsumeBool():
        d["qubit"] = fdp.ConsumeIntInRange(-100, 10000)
    if fdp.ConsumeBool():
        d["T1"] = (
            fdp.PickValueInList(_SPECIAL_FLOATS) if fdp.ConsumeBool() else fdp.ConsumeRegularFloat()
        )
    if fdp.ConsumeBool():
        d["T2"] = (
            fdp.PickValueInList(_SPECIAL_FLOATS) if fdp.ConsumeBool() else fdp.ConsumeRegularFloat()
        )
    if fdp.ConsumeBool():
        d["frequency"] = fdp.ConsumeRegularFloat()
    if fdp.ConsumeBool():
        d["readout_error_0to1"] = (
            fdp.PickValueInList(_SPECIAL_FLOATS) if fdp.ConsumeBool() else fdp.ConsumeRegularFloat()
        )
    if fdp.ConsumeBool():
        d["readout_error_1to0"] = (
            fdp.PickValueInList(_SPECIAL_FLOATS) if fdp.ConsumeBool() else fdp.ConsumeRegularFloat()
        )
    return d


def _make_gate_dict(fdp: atheris.FuzzedDataProvider) -> dict:
    """Build a fuzzed gate_properties entry."""
    d: dict = {}
    if fdp.ConsumeBool():
        d["gate"] = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 20))
    if fdp.ConsumeBool():
        n_qubits = fdp.ConsumeIntInRange(0, 5)
        d["qubits"] = [fdp.ConsumeIntInRange(-10, 1000) for _ in range(n_qubits)]
    if fdp.ConsumeBool():
        params: dict = {}
        if fdp.ConsumeBool():
            params["gate_error"] = (
                fdp.PickValueInList(_SPECIAL_FLOATS)
                if fdp.ConsumeBool()
                else fdp.ConsumeRegularFloat()
            )
        if fdp.ConsumeBool():
            params["gate_length"] = (
                fdp.PickValueInList(_SPECIAL_FLOATS)
                if fdp.ConsumeBool()
                else fdp.ConsumeRegularFloat()
            )
        d["parameters"] = params
    return d


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    choice = fdp.ConsumeIntInRange(0, 3)

    try:
        if choice == 0:
            # Full BackendProperties.from_qubitboost_dict with fuzzed data
            n_qubits_entries = fdp.ConsumeIntInRange(0, 20)
            n_gate_entries = fdp.ConsumeIntInRange(0, 20)
            n_coupling = fdp.ConsumeIntInRange(0, 30)

            cal_dict: dict = {}
            if fdp.ConsumeBool():
                cal_dict["backend_name"] = fdp.ConsumeUnicodeNoSurrogates(
                    fdp.ConsumeIntInRange(0, 50)
                )
            if fdp.ConsumeBool():
                cal_dict["n_qubits"] = fdp.ConsumeIntInRange(-10, 10000)
            if fdp.ConsumeBool():
                cal_dict["timestamp"] = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 30))
            if fdp.ConsumeBool():
                cal_dict["basis_gates"] = [
                    fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 10))
                    for _ in range(fdp.ConsumeIntInRange(0, 10))
                ]
            if fdp.ConsumeBool():
                cal_dict["provider"] = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 20))

            # Qubit properties — may have missing "qubit" key
            qprops = []
            for _ in range(n_qubits_entries):
                qprops.append(_make_qubit_dict(fdp))
            cal_dict["qubit_properties"] = qprops

            # Gate properties — may have missing "gate" key
            gprops = []
            for _ in range(n_gate_entries):
                gprops.append(_make_gate_dict(fdp))
            cal_dict["gate_properties"] = gprops

            # Coupling map with potentially invalid indices
            coupling = []
            for _ in range(n_coupling):
                coupling.append(
                    [fdp.ConsumeIntInRange(-100, 10000), fdp.ConsumeIntInRange(-100, 10000)]
                )
            cal_dict["coupling_map"] = coupling

            BackendProperties.from_qubitboost_dict(cal_dict)

        elif choice == 1:
            # Individual QubitProperties.from_qubitboost_dict
            d = _make_qubit_dict(fdp)
            if "qubit" not in d:
                d["qubit"] = fdp.ConsumeIntInRange(0, 100)
            QubitProperties.from_qubitboost_dict(d)

        elif choice == 2:
            # Individual GateProperties.from_qubitboost_dict
            d = _make_gate_dict(fdp)
            if "gate" not in d:
                d["gate"] = "cx"
            if "qubits" not in d:
                d["qubits"] = [0, 1]
            GateProperties.from_qubitboost_dict(d)

        else:
            # Duplicate entries: same qubit_id multiple times
            qubit_id = fdp.ConsumeIntInRange(0, 10)
            n_dupes = fdp.ConsumeIntInRange(2, 10)
            qprops = []
            for _ in range(n_dupes):
                qprops.append(
                    {
                        "qubit": qubit_id,
                        "T1": fdp.ConsumeRegularFloat(),
                        "T2": fdp.ConsumeRegularFloat(),
                    }
                )
            cal_dict = {
                "backend_name": "test",
                "n_qubits": 20,
                "qubit_properties": qprops,
                "coupling_map": [],
                "gate_properties": [],
            }
            bp = BackendProperties.from_qubitboost_dict(cal_dict)
            bp.qubit(qubit_id)

    except (ValueError, IndexError, KeyError, TypeError, OverflowError):
        pass


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
