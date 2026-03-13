#!/usr/bin/env python3
"""Generate a calibration snapshot JSON file for testing.

Since we cannot call IBM/Rigetti/IonQ APIs without credentials, this script
generates realistic mock calibration data in the BackendProperties format.

Usage:
    python scripts/fetch_calibration_snapshot.py --backend ibm_fez --output tests/fixtures/calibration_snapshots/
    python scripts/fetch_calibration_snapshot.py --backend rigetti_ankaa --qubits 20
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

# Realistic ranges per vendor
VENDOR_PROFILES = {
    "ibm": {
        "t1_range": (80.0, 300.0),
        "t2_range": (40.0, 200.0),
        "readout_error_range": (0.005, 0.04),
        "single_gate_error": (0.0001, 0.001),
        "two_gate_error": (0.003, 0.02),
        "single_gate_time_ns": 40.0,
        "two_gate_time_ns": 500.0,
        "basis_gates": ["ecr", "rz", "sx", "x", "id"],
        "two_qubit_gate": "ecr",
    },
    "rigetti": {
        "t1_range": (10.0, 50.0),
        "t2_range": (5.0, 30.0),
        "readout_error_range": (0.01, 0.08),
        "single_gate_error": (0.001, 0.005),
        "two_gate_error": (0.01, 0.05),
        "single_gate_time_ns": 40.0,
        "two_gate_time_ns": 200.0,
        "basis_gates": ["cz", "rx", "rz", "id"],
        "two_qubit_gate": "cz",
    },
    "ionq": {
        "t1_range": (10000.0, 100000.0),
        "t2_range": (1000.0, 10000.0),
        "readout_error_range": (0.001, 0.01),
        "single_gate_error": (0.0001, 0.001),
        "two_gate_error": (0.001, 0.01),
        "single_gate_time_ns": 10000.0,
        "two_gate_time_ns": 600000.0,
        "basis_gates": ["gpi", "gpi2", "ms", "id"],
        "two_qubit_gate": "ms",
    },
    "iqm": {
        "t1_range": (20.0, 80.0),
        "t2_range": (10.0, 50.0),
        "readout_error_range": (0.01, 0.06),
        "single_gate_error": (0.0005, 0.003),
        "two_gate_error": (0.005, 0.03),
        "single_gate_time_ns": 32.0,
        "two_gate_time_ns": 60.0,
        "basis_gates": ["cz", "prx", "id"],
        "two_qubit_gate": "cz",
    },
}

BACKEND_VENDORS = {
    "ibm_fez": "ibm",
    "ibm_torino": "ibm",
    "ibm_marrakesh": "ibm",
    "rigetti_ankaa": "rigetti",
    "ionq_aria": "ionq",
    "ionq_forte": "ionq",
    "iqm_garnet": "iqm",
    "iqm_emerald": "iqm",
}


def _linear_coupling(n: int) -> list[list[int]]:
    """Generate a linear chain coupling map."""
    edges = []
    for i in range(n - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return edges


def generate_snapshot(backend: str, n_qubits: int, seed: int = 42) -> dict:
    """Generate a realistic calibration snapshot."""
    rng = random.Random(seed)
    vendor = BACKEND_VENDORS.get(backend, "ibm")
    profile = VENDOR_PROFILES[vendor]

    qubits = []
    for i in range(n_qubits):
        t1 = rng.uniform(*profile["t1_range"])
        t2 = min(rng.uniform(*profile["t2_range"]), 2 * t1)
        qubits.append({
            "qubit_id": i,
            "t1_us": round(t1, 2),
            "t2_us": round(t2, 2),
            "frequency_ghz": round(rng.uniform(4.5, 5.5), 4),
            "readout_error": round(rng.uniform(*profile["readout_error_range"]), 5),
            "readout_length_ns": round(rng.uniform(500, 1200), 1),
        })

    coupling_map = _linear_coupling(n_qubits)
    gates = []
    for edge in coupling_map:
        gates.append({
            "gate_type": profile["two_qubit_gate"],
            "qubits": edge,
            "error_rate": round(rng.uniform(*profile["two_gate_error"]), 5),
            "gate_length_ns": profile["two_gate_time_ns"],
        })

    for i in range(n_qubits):
        for g in profile["basis_gates"]:
            if g in (profile["two_qubit_gate"], "id"):
                continue
            gates.append({
                "gate_type": g,
                "qubits": [i],
                "error_rate": round(rng.uniform(*profile["single_gate_error"]), 6),
                "gate_length_ns": profile["single_gate_time_ns"],
            })

    return {
        "backend": backend,
        "vendor": vendor,
        "n_qubits": n_qubits,
        "basis_gates": profile["basis_gates"],
        "coupling_map": coupling_map,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "qubit_properties": qubits,
        "gate_properties": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calibration snapshot")
    parser.add_argument("--backend", "-b", default="ibm_fez", help="Backend name")
    parser.add_argument("--qubits", "-q", type=int, default=10, help="Number of qubits")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    snapshot = generate_snapshot(args.backend, args.qubits, args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y_%m_%d")
    out_path = out_dir / f"{args.backend}_{date_str}.json"

    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Wrote {out_path} ({args.qubits} qubits, {len(snapshot['gate_properties'])} gates)")


if __name__ == "__main__":
    main()
