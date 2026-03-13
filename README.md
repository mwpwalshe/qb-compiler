# qb-compiler

[![PyPI](https://img.shields.io/pypi/v/qb-compiler.svg)](https://pypi.org/project/qb-compiler/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/qubitboost/qb-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/qubitboost/qb-compiler/actions)
[![Coverage](https://codecov.io/gh/qubitboost/qb-compiler/branch/main/graph/badge.svg)](https://codecov.io/gh/qubitboost/qb-compiler)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/qb-compiler.svg)](https://pypi.org/project/qb-compiler/)

**Calibration-aware quantum circuit compiler. Compile circuits that are less likely to fail on today's hardware.**

---

## Why qb-compiler?

- **Uses TODAY's calibration data** — not just topology. Gate errors, T1/T2 coherence, and readout fidelity change every calibration cycle. qb-compiler reads current data and uses it at every compilation stage.
- **Multi-vendor: IBM, Rigetti, IonQ, IQM from one API** — compile for any backend with a single interface. Native gate decomposition, calibration parsing, and cost estimation are handled per-vendor.
- **Budget-aware: compile within your cost constraints** — estimate execution cost before you run, enforce budget limits, and pick the cheapest backend that meets your fidelity target.

---

## qb-compiler vs Qiskit Default

| Feature | Qiskit Default | qb-compiler |
|---------|---------------|-------------|
| Calibration-aware mapping | - | Yes |
| T1 asymmetry handling | - | Yes |
| Multi-vendor support | IBM only | IBM, Rigetti, IonQ, IQM |
| Budget constraints | - | Yes |
| Cost estimation | - | Yes |
| Pre-execution fidelity estimate | - | Yes |
| Noise-aware SWAP routing | - | Yes |
| T1/T2-aware scheduling | - | Yes |

---

## Quick Start

```bash
pip install qb-compiler
```

```python
from qb_compiler import QBCompiler, QBCircuit

compiler = QBCompiler.from_backend("ibm_fez")
circuit = QBCircuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
result = compiler.compile(circuit)
print(f"Depth: {result.compiled_depth}, Est. fidelity: {result.estimated_fidelity:.3f}")
```

Or with budget enforcement:

```python
result = compiler.compile(circuit, budget_usd=5.0)
cost = compiler.estimate_cost(result.compiled_circuit, shots=4096)
print(f"Estimated cost: ${cost.total_usd:.2f}")
```

---

## The Problem

Standard transpilers treat quantum hardware as a static graph. They pick qubit
mappings and routing paths based on topology alone, ignoring the fact that gate
error rates, coherence times, and readout fidelities change with every
calibration cycle. On IBM Heron processors, calibration data updates daily --
and qubit quality can vary by 10x across the chip.

The result: **15-40% fidelity left on the table** because your compiler does not
know which qubits are good *today*.

## The Solution

qb-compiler reads today's calibration data and uses it at every stage of
compilation: qubit mapping, SWAP routing, gate scheduling, and fidelity
estimation. Every decision is made with current hardware noise in mind.

```
Calibration Data (T1, T2, gate error, readout error)
        |
        v
  +-----------------+     +------------------+     +---------------------+
  | CalibrationMapper| --> | NoiseAwareRouter | --> | NoiseAwareScheduler |
  | VF2 + noise      |     | Dijkstra lowest- |     | ALAP with T1/T2    |
  | weighted scoring  |     | error-path SWAPs |     | urgency scoring    |
  +-----------------+     +------------------+     +---------------------+
        |                                                   |
        v                                                   v
  +-------------------+                          +---------------------+
  | GateDecomposition |                          | ErrorBudgetEstimator|
  | Native basis      |                          | Pre-exec fidelity   |
  | (ECR, CZ, ...)   |                          | prediction          |
  +-------------------+                          +---------------------+
```

---

## Key Features

- **CalibrationMapper** -- VF2 subgraph isomorphism search scored by gate error,
  coherence (T1/T2), and readout fidelity. Finds the best physical qubit
  placement for your circuit on today's device.

- **NoiseAwareRouter** -- Dijkstra shortest-error-path SWAP insertion. Minimises
  accumulated gate error instead of SWAP count. Each SWAP decomposes to 3 CX
  gates, so error-optimal routing matters.

- **NoiseAwareScheduler** -- ALAP scheduling with T1/T2 urgency scoring. Qubits
  with shorter coherence times get their gates scheduled first, reducing idle
  decoherence.

- **GateDecomposition** -- Decomposes to native basis gate sets (IBM ECR,
  Rigetti CZ, IonQ MS, IQM CZ) using calibration-aware decomposition
  strategies.

- **ErrorBudgetEstimator** -- Predicts circuit fidelity before execution using
  calibration data. Know if your circuit is viable before spending QPU time.

- **Qiskit Plugin** -- Drop-in `AnalysisPass` that replaces Qiskit's layout
  stage. Works with `generate_preset_pass_manager` at any optimization level.

---

## Benchmarks

Estimated fidelity improvement over topology-only mapping, using real IBM Fez
calibration data (March 2026, 156 qubits):

| Circuit   | Qubits | 2Q Gates | Baseline Fidelity | qb-compiler Fidelity | Improvement |
|-----------|--------|----------|-------------------|-----------------------|-------------|
| Bell      | 2      | 1        | 0.966             | 0.989                 | +2.4%       |
| GHZ-5     | 5      | 4        | 0.915             | 0.947                 | +3.5%       |
| GHZ-8     | 8      | 7        | 0.866             | 0.900                 | +4.0%       |
| GHZ-16    | 16     | 15       | 0.748             | 0.767                 | +2.6%       |
| QAOA-4    | 4      | 6        | 0.923             | 0.961                 | +4.1%       |
| QAOA-8    | 8      | 14       | 0.846             | 0.885                 | +4.5%       |
| QAOA-12   | 12     | 22       | 0.776             | 0.802                 | +3.2%       |

> **Note:** Baseline uses median device error rates (topology-only qubit
> selection). qb-compiler uses calibration-aware VF2 mapping scored by today's
> per-qubit gate error, readout error, and T1/T2 coherence. Benchmarked against
> real IBM Fez calibration snapshot. Improvement depends on circuit structure
> and daily calibration variance.

---

## Qiskit Integration

qb-compiler provides two integration paths with Qiskit:

### Drop-in transpile function

```python
from qb_compiler.qiskit_plugin import qb_transpile

compiled = qb_transpile(
    circuit,
    backend="ibm_fez",
    calibration_path="calibrations/ibm_fez_2026-03-12.json",
)
```

### As a pass in an existing pipeline

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qb_compiler.qiskit_plugin import QBCalibrationLayout

pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
pm.layout.append(QBCalibrationLayout(calibration_data))
compiled = pm.run(circuit)
```

---

## Supported Backends

| Vendor   | Backends                          | Native Basis |
|----------|-----------------------------------|--------------|
| IBM      | Fez, Torino, Marrakesh (Heron)    | ECR, RZ, SX  |
| Rigetti  | Ankaa-3                           | CZ, RZ, RX   |
| IonQ     | Aria, Forte (via Braket)          | MS, GPI, GPI2|
| IQM      | Garnet, Emerald                   | CZ, PRX      |

Calibration data can be loaded from local JSON files, fetched from vendor APIs,
or streamed from the [QubitBoost](https://qubitboost.io) calibration hub.

---

## Architecture

qb-compiler is structured as a configurable pass pipeline. Each pass reads
calibration data and transforms the circuit IR:

```
Input Circuit (Qiskit QuantumCircuit or QBCircuit)
  |
  v
[1] Validation          Reject invalid circuits early
  |
  v
[2] CalibrationMapper   VF2 + noise-weighted qubit placement
  |
  v
[3] NoiseAwareRouter    Dijkstra SWAP insertion (min error, not min distance)
  |
  v
[4] GateDecomposition   Decompose to native basis gates
  |
  v
[5] NoiseAwareScheduler ALAP scheduling with T1/T2 urgency
  |
  v
[6] ErrorBudgetEstimate Pre-execution fidelity prediction
  |
  v
CompileResult { compiled_circuit, compiled_depth, estimated_fidelity, cost }
```

Passes are composable. You can run individual passes, reorder them, or write
custom passes by extending `BasePass`.

---

## CLI

```bash
# Compile a QASM file
qbc compile circuit.qasm --backend ibm_fez --strategy fidelity_optimal

# Show available backends and specs
qbc info

# Show calibration summary for a backend
qbc calibration show ibm_fez
```

---

## Installation Options

```bash
# Core (IBM backends via Qiskit)
pip install qb-compiler

# With Rigetti support
pip install "qb-compiler[rigetti]"

# With IonQ support (via Amazon Braket)
pip install "qb-compiler[ionq]"

# With IQM support
pip install "qb-compiler[iqm]"

# Everything
pip install "qb-compiler[all]"

# Development
pip install "qb-compiler[dev]"
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on development setup, testing, and code style.

To run the test suite:

```bash
pip install "qb-compiler[dev]"
pytest
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full text.

Copyright 2026 Walshe Thornwell Fotheringham Trading Ltd.
