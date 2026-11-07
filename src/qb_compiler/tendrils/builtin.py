# SPDX-License-Identifier: Apache-2.0
"""The tendrils qb-compiler ships with: one per hardware vendor it has a spec for, one per format.

These are declarations over code that already exists. Nothing here changes how compiling,
pricing or ranking behaves. The hardware labels come from the same registry that ``qbc when``
reads, so the two cannot disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qb_compiler.tendrils.conformance import Check
from qb_compiler.tendrils.model import PriceRow, Tendril, ValidationRecord

if TYPE_CHECKING:
    from collections.abc import Callable

MAINTAINER = "qb-compiler maintainers"
CONTACT = "https://github.com/mwpwalshe/qb-compiler/issues"

#: Registry status to label, the same mapping ``qbc when`` uses.
LABEL_FOR_STATUS = {
    "live": "validated",
    "live-unvalidated": "UNVALIDATED",
    "static": "fixture-only",
    "none": "no-adapter",
}
_CALIBRATION_FOR_STATUS = {
    "live": "live",
    "live-unvalidated": "live",
    "static": "snapshot",
    "none": "none",
}

#: The real job the IBM label rests on. The record is committed in this repository.
IBM_VALIDATION = ValidationRecord(
    date="2026-03-14",
    backend="ibm_fez",
    job_id="d6qknvfr88ds73dch6r0",
    raw_sha256="ab898699c3fc00151c8e34dd8c2216f0595f19c25c81116d3e7b3c382581cf84",
    source=(
        "results/hardware_validation_dd.json in the qb-compiler repository, committed 2026-03-14"
    ),
)

_VENDOR = {
    "ibm": "IBM",
    "ionq": "IonQ",
    "iqm": "IQM",
    "rigetti": "Rigetti",
    "quantinuum": "Quantinuum",
}


def _version() -> str:
    from qb_compiler import __version__

    return str(__version__)


def _price_rows(backends: list[str]) -> tuple[PriceRow, ...]:
    from qb_compiler.cost.pricing import PRICING_AS_OF, get_pricing

    rows: list[PriceRow] = []
    for backend in backends:
        p = get_pricing(backend)
        if p is None:
            continue
        source = p.notes.strip() or "qb-compiler price table"
        rows.append(
            PriceRow(backend, "per_shot", p.cost_per_shot_usd, p.currency, PRICING_AS_OF, source)
        )
        if p.cost_per_task_usd > 0:
            rows.append(
                PriceRow(
                    backend,
                    "per_task",
                    p.cost_per_task_usd,
                    p.currency,
                    PRICING_AS_OF,
                    source,
                    note="charged once per task on top of the shots",
                )
            )
    return tuple(rows)


def hardware_tendrils() -> list[Tendril]:
    """One tendril per vendor in the backend table."""
    from qb_compiler.calibration.registry import all_backend_statuses

    by_vendor: dict[str, list[Any]] = {}
    for status in all_backend_statuses():
        by_vendor.setdefault(status.provider, []).append(status)
    out = []
    for provider, statuses in sorted(by_vendor.items()):
        backends = [s.backend for s in statuses]
        status = statuses[0].live_status.value
        label = LABEL_FOR_STATUS.get(status, "no-adapter")
        prices = _price_rows(backends)
        calibration = _CALIBRATION_FOR_STATUS.get(status, "none")
        out.append(
            Tendril(
                name=provider,
                kind="hardware",
                vendor=_VENDOR.get(provider, provider),
                maintainer=MAINTAINER,
                contact=CONTACT,
                adapter_version=_version(),
                capabilities={
                    "circuits": True,
                    "qec_results": False,
                    "price": bool(prices),
                    "calibration": calibration != "none",
                    "execute": False,
                },
                label=label,
                backends=tuple(backends),
                calibration=calibration,
                price_rows=prices,
                no_public_price=not prices,
                validation=IBM_VALIDATION if label == "validated" and provider == "ibm" else None,
                origin="builtin",
            )
        )
    return out


# ── formats ─────────────────────────────────────────────────────────


def _ghz3() -> Any:
    from qb_compiler.ir.circuit import QBCircuit
    from qb_compiler.ir.operations import QBGate

    c = QBCircuit(3, 3, name="ghz3")
    c.add_gate(QBGate("h", (0,)))
    c.add_gate(QBGate("cx", (0, 1)))
    c.add_gate(QBGate("cx", (1, 2)))
    for q in range(3):
        c.add_measurement(q, q)
    return c


def _shape(c: Any) -> tuple[Any, ...]:
    gates = tuple((g.name, tuple(g.qubits), tuple(round(p, 9) for p in g.params)) for g in c.gates)
    return c.n_qubits, gates, len(c.measurements)


#: name, format label, required module, to-native, from-native (converter names in ir.converters)
_FORMATS: tuple[tuple[str, str, str | None, str, str], ...] = (
    ("openqasm2", "OpenQASM 2", None, "to_qasm", "from_qasm"),
    ("openqasm3", "OpenQASM 3", "qiskit_qasm3_import", "to_qasm3", "from_qasm3"),
    ("qiskit", "Qiskit QuantumCircuit", "qiskit", "to_qiskit", "from_qiskit"),
    ("cirq", "Cirq Circuit", "cirq", "to_cirq", "from_cirq"),
    ("pennylane", "PennyLane tape", "pennylane", "to_pennylane", "from_pennylane"),
)


def _has(module: str | None) -> bool:
    if module is None:
        return True
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def round_trip(name: str) -> Check:
    """Send a three-qubit GHZ circuit through a format and back, and compare its shape."""
    spec = next((f for f in _FORMATS if f[0] == name), None)
    if spec is None:
        return Check("round_trip", None, "no round trip defined for this tendril")
    _, _, module, to_name, from_name = spec
    if not _has(module):
        return Check("round_trip", None, f"skipped: {module} is not installed")
    from qb_compiler.ir import converters

    to_native: Callable[[Any], Any] = getattr(converters, to_name)
    from_native: Callable[[Any], Any] = getattr(converters, from_name)
    before = _ghz3()
    try:
        after = from_native(to_native(before))
    except Exception as exc:  # a converter fault is the finding, not a crash
        return Check("round_trip", False, f"{type(exc).__name__}: {exc}")
    same = _shape(after) == _shape(before)
    return Check(
        "round_trip",
        same,
        "GHZ-3 comes back unchanged" if same else "GHZ-3 comes back changed",
    )


def format_tendrils() -> list[Tendril]:
    return [
        Tendril(
            name=name,
            kind="format",
            vendor="open format" if name.startswith("openqasm") else name.capitalize(),
            maintainer=MAINTAINER,
            contact=CONTACT,
            adapter_version=_version(),
            capabilities={
                "circuits": True,
                "qec_results": False,
                "price": False,
                "calibration": False,
                "execute": False,
            },
            label="fixture-only",
            formats=(label,),
            origin="builtin",
        )
        for name, label, _, _, _ in _FORMATS
    ]


def builtin_tendrils() -> list[Tendril]:
    return hardware_tendrils() + format_tendrils()
