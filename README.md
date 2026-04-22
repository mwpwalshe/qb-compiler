# qb-compiler

[![Qiskit Ecosystem](https://img.shields.io/badge/Qiskit%20Ecosystem-Member-6929C4?logo=Qiskit)](https://www.ibm.com/quantum/ecosystem)
[![PyPI](https://img.shields.io/pypi/v/qb-compiler.svg)](https://pypi.org/project/qb-compiler/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://github.com/mwpwalshe/qb-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/mwpwalshe/qb-compiler/actions)
[![Coverage](https://img.shields.io/badge/coverage-60%25-yellow)]()
[![Docs](https://img.shields.io/badge/docs-README-blue)](https://github.com/mwpwalshe/qb-compiler#readme)

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

✔  qb-compiler 0.4.0b1
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
| Backend recommendation | No | Yes |
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

Validated on IBM Fez (156 qubits, March 2026). All results are measured fidelity from real hardware, 4096 shots per circuit.

### Layout Selection — GHZ Circuits

qb-compiler's CalibrationMapper (post-routing scoring, multi-region search) vs Qiskit `transpile` `optimization_level=3`. Both use Qiskit's SabreSwap for routing — the only difference is initial qubit placement.

| Circuit | Qiskit | qb-compiler | Delta | Notes |
|---------|--------|-------------|-------|-------|
| GHZ-3 | 96.5% | 96.7% | +0.2% | Both find optimal region |
| GHZ-5 | 92.5% | 93.2% | +0.7% | Different regions selected |
| GHZ-8 | 82.1% | 87.5% | **+5.3%** | Best result — region 120-143 |
| GHZ-10 | 78.8% | 79.8% | +1.0% | Region 120-147 |

Fidelity = P(000...0) + P(111...1) over 4096 shots.

Results vary by calibration window. In runs where both mappers converge on the same optimal region (identical qubit selection), results are statistically equivalent. Improvement is largest when qb-compiler discovers a better region than Qiskit's default search.

### Dynamical Decoupling

Selective DD applied after Qiskit routing. DD is automatically skipped for dense circuits where it adds noise without benefit.

| Circuit | Without DD | With DD | Delta | Notes |
|---------|-----------|---------|-------|-------|
| GHZ-8 | 83.5% | 83.6% | +0.1% | Minimal idle time |
| QFT-6 | 2.1% | 2.6% | +27% rel. | Long idle periods — DD helps |
| QAOA-6 | 5.9% | 5.5% | -6.6% rel. | Dense circuit — DD skipped in v0.2.1 |

QFT-6 and QAOA-6 base fidelities are in the noise floor (circuit depth exceeds viable limit). `qbc preflight` would flag these as DO NOT RUN, saving QPU time.

### Journey to These Results

These results followed an iterative hardware validation process:

1. Initial mapper lost to Qiskit by up to 10.6% (pre-routing scoring flaw)
2. Post-routing scoring fix closed the gap
3. Multi-region search + routed fidelity tiebreaker achieved positive results
4. Qiskit seed injection ensures qb-compiler never selects a worse layout than Qiskit's own best

All raw validation data is in the `results/` directory. Reproduce: `python scripts/hardware_validation.py --dry-run`

Full walkthrough: [docs/tutorials/hardware_validation_walkthrough.ipynb](docs/tutorials/hardware_validation_walkthrough.ipynb)

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

All transpilation uses Qiskit's routing engine internally. qb-compiler's value is in execution intelligence (preflight, viability, cost estimation) and calibration-aware layout selection that matches or exceeds Qiskit's default on hardware-validated benchmarks.

---

## NVIDIA Ising Decoder Integration (v0.4.0b1, beta)

First Qiskit-side onramp to NVIDIA's `Ising-Decoder-SurfaceCode-1`
(dropped 2026-04-14). Takes a rotated surface-code memory experiment,
spits out the 4-channel `(B, 4, T, D, D)` tensor the pretrained decoder
eats. A PyMatching MWPM baseline ships in the box so youve somethign to
beat. Plug the NVIDIA pre-decoder in when youve got the gated-HF weights
— qb-compiler doesnt vendor them.

Stim only for now, no hw shots thru it yet. Install:

```bash
pip install --pre qb-compiler[ising]          # stim + pymatching
pip install --pre qb-compiler[ising-nvidia]   # adds torch + safetensors
```

```python
from qb_compiler.ising import (
    SurfaceCodePatchSpec, PyMatchingDecoder, evaluate_logical_error_rate,
)

spec = SurfaceCodePatchSpec(distance=7, rounds=7, basis="X", p_error=0.003)
result = evaluate_logical_error_rate(
    spec, PyMatchingDecoder(spec), shots=50_000, seed=42,
)
print(result.as_dict())
```

See [`src/qb_compiler/ising/README.md`](src/qb_compiler/ising/README.md)
for the full API, licensing breakdown (Apache 2.0 integration code;
NVIDIA Open Model License weights distributed separately by NVIDIA),
and the [Notebook 17 walkthrough](notebooks/17_nvidia_ising_integration.ipynb).

## Optional QubitBoost SDK Integration

qb-compiler works fully standalone. For supported workloads, it can optionally integrate with the QubitBoost SDK to surface compatible execution paths:

- OptGate — adaptive shot reduction for supported QAOA workloads
- ChemGate — evaluation reduction for supported VQE workflows
- LiveGate / ShotValidator — optional runtime checks

Gate recommendations appear in `qbc preflight` and `qbc analyze` when circuit type is detected with high confidence. Performance figures are hardware-validated, workload-dependent, and documented separately at [qubitboost.io](https://qubitboost.io).

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

## Installation

**Compatibility:** Qiskit 1.0-1.4 | Python 3.10-3.12 | Tested on IBM Fez, Torino, Marrakesh, Rigetti Ankaa-3

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

Copyright 2026 QubitBoost.
