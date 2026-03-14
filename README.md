# qb-compiler

[![PyPI](https://img.shields.io/pypi/v/qb-compiler.svg)](https://pypi.org/project/qb-compiler/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://github.com/mwpwalshe/qb-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/mwpwalshe/qb-compiler/actions)

**Quantum Execution Intelligence. Know before you run.**

---

## What is qb-compiler?

qb-compiler helps quantum developers make better execution decisions. Know which backend to use, whether your circuit is viable, what fidelity to expect, and what it will cost — before you spend QPU time.

Built on top of Qiskit's transpiler.

```bash
pip install qb-compiler
```

---

## Quick Start

```python
from qb_compiler import QBCompiler, check_viability

# Is my circuit worth running?
result = check_viability(circuit, backend="ibm_fez")
print(result)
# → Status: VIABLE
# → Est. fidelity: 0.847
# → Cost (4096 shots): $0.6554
# → Suggestions:
# →   - Circuit looks good — proceed with execution.

# Compile with automatic optimizations
compiler = QBCompiler.from_backend("ibm_fez")
compiled = compiler.compile(circuit)
```

---

## CLI

### `qbc preflight` — Should I run this?

```
$ qbc preflight circuit.qasm --backend ibm_fez

  Circuit: GHZ-8
  Backend: ibm_fez (156q)

  Status: VIABLE
  Estimated fidelity: 0.8519
  Depth: 12  (viable limit: 188)
  2Q gates: 7
  Cost (4096 shots): $0.6554
```

### `qbc analyze` — Detailed analysis with suggestions

```
$ qbc analyze circuit.qasm --backend ibm_fez

  Circuit Analysis: QAOA-MaxCut
  Qubits: 6  Gates: 84  Depth: 47
  Gate breakdown: cx:24, rz:18, rx:12, h:6, measure:6

  Backend: ibm_fez (156q)
  Status: MARGINAL
  Estimated fidelity: 0.1823
  Signal/noise ratio: 11.7x
  Depth: 47  (viable limit: 188)
  2Q gates after transpilation: 24
  Cost (4096 shots): $0.6554

  Suggestions:
    - Consider ZNE or PEC error mitigation (2-5x improvement possible).
    - Good candidate for error mitigation to further improve results.
```

### `qbc diff` — Compare two backends

```
$ qbc diff circuit.qasm --backend ibm_fez --vs ibm_torino

  Circuit: GHZ-5

                              ibm_fez        ibm_torino
                           ----------------   ----------------
  Status                          VIABLE           VIABLE
  Est. fidelity                  0.9430           0.9285 <
  2Q gates                            4                4
  Depth                               5                5
  Cost/4096 shots              $0.6554          $0.5734 <

  Recommendation: ibm_fez (+0.0145 fidelity)
```

### `qbc doctor` — Environment health check

```
$ qbc doctor

qbc doctor

✔  qb-compiler 0.2.0
✔  Python 3.11.14
✔  Qiskit 1.4.5
✔  IBM credentials configured (2 account(s))
✔  9 backends configured
✔  5 calibration snapshot(s) available
✔  numpy 2.3.5
✔  rustworkx 0.17.1

Environment looks good!
```

### `qbc compile` — Compile with receipt

```
$ qbc compile circuit.qasm --backend ibm_fez --receipt

Compiled: depth 12 -> 8 (33.3% reduction)
Estimated fidelity: 0.8519
Compilation time: 142.3 ms
Receipt saved to circuit.receipt.json
```

---

## Feature Comparison

| Feature | Qiskit | qb-compiler |
|---------|--------|-------------|
| Transpilation | Excellent | Uses Qiskit internally |
| Circuit viability check | No | `qbc preflight` |
| Pre-execution fidelity estimate | No | Yes |
| Backend recommendation | No | Yes (single), Pro (multi) |
| Selective dynamical decoupling | No | Yes |
| Cost estimation | No | Yes |
| Budget enforcement | No | Yes |
| Compilation receipts | No | `--receipt` |
| Multi-vendor backend specs | IBM only | IBM, Rigetti, IonQ, IQM, Quantinuum |
| Environment health check | No | `qbc doctor` |
| Circuit analysis with suggestions | No | `qbc analyze` |
| Backend comparison | No | `qbc diff` |

---

## Hardware Validation

Validated on IBM Fez (156 qubits, March 2026):

**Layout selection:** qb-compiler's CalibrationMapper matches or slightly exceeds
Qiskit opt_level=3 on GHZ circuits. On GHZ-5 with different layout selections,
qb-compiler achieved +1.0% measured fidelity improvement. On GHZ-8 and GHZ-10
where both converged to the same optimal region, results were statistically
equivalent.

**Dynamical decoupling:** Selective DD improved fidelity on circuits with
significant idle time (QFT-6: +27% relative improvement). DD is automatically
skipped for dense circuits where it adds noise without benefit (QAOA-6: DD
correctly identified as harmful, -6.6%).

All transpilation uses Qiskit's routing engine internally. qb-compiler focuses on
execution intelligence — helping you decide where to run and what to expect —
rather than replacing Qiskit's transpiler.

---

## How It Works

```
Your Circuit
  │
  ├─→ Viability Check         Is it worth running?
  │
  ├─→ Backend Selection        Where should it run?
  │
  ├─→ Qiskit Transpilation     Best of N seeds, opt_level=3
  │
  ├─→ Selective DD             Protect idle qubits (skip dense circuits)
  │
  ├─→ Fidelity Estimation      What to expect
  │
  ├─→ Cost Estimation          What it will cost
  │
  └─→ Compilation Receipt      Full audit trail (JSON)
```

---

## Supported Backends

| Vendor     | Backends                       | Qubits  | Native Basis  |
|------------|--------------------------------|---------|---------------|
| IBM        | Fez, Torino, Marrakesh (Heron) | 133-156 | ECR, RZ, SX   |
| Rigetti    | Ankaa-3                        | 84      | CZ, RZ, RX    |
| IonQ       | Aria, Forte                    | 25-36   | MS, GPI, GPI2 |
| IQM        | Garnet, Emerald                | 5-20    | CZ, PRX       |
| Quantinuum | H2                             | 32      | RZ, U1Q, ZZ   |

Calibration data can be loaded from local JSON files or fetched from vendor APIs.

---

## Open Source vs QubitBoost Pro

| Feature | Open Source | QubitBoost Pro |
|---------|------------|----------------|
| Single-backend preflight | Yes | Yes |
| Multi-backend ranking | — | Yes |
| Cached calibration | Yes | Live calibration |
| Local receipts | Yes | Cloud dashboard |
| Cost estimation | Single vendor | Cross-vendor |
| Drift alerts | — | Yes |
| Circuit watchlist | — | Yes |

qb-compiler is free and fully functional standalone. QubitBoost Pro adds live
multi-vendor calibration, cloud execution history, and advanced optimizations.
Learn more at [qubitboost.io](https://qubitboost.io).

---

## Installation

```bash
# Core (IBM backends via Qiskit)
pip install qb-compiler

# With ML acceleration (optional)
pip install "qb-compiler[ml]"

# With GNN layout predictor (optional)
pip install "qb-compiler[gnn]"

# Development
pip install "qb-compiler[dev]"
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `qbc preflight <circuit> -b <backend>` | Quick viability check: VIABLE / CAUTION / DO NOT RUN |
| `qbc analyze <circuit> -b <backend>` | Detailed analysis with suggestions |
| `qbc diff <circuit> -b <backend> --vs <backend>` | Side-by-side backend comparison |
| `qbc doctor` | Environment health check |
| `qbc compile <circuit> -b <backend> --receipt` | Compile with audit trail |
| `qbc info` | Show version and available backends |
| `qbc calibration show <backend>` | Show calibration summary |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
pip install "qb-compiler[dev]"
pytest
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full text.

Copyright 2026 [QubitBoost](https://www.qubitboost.io).
