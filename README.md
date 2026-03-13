# qb-compiler

[![PyPI](https://img.shields.io/pypi/v/qb-compiler.svg)](https://pypi.org/project/qb-compiler/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/mwpwalshe/qb-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/mwpwalshe/qb-compiler/actions)
[![Coverage](https://codecov.io/gh/mwpwalshe/qb-compiler/branch/master/graph/badge.svg)](https://codecov.io/gh/mwpwalshe/qb-compiler)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/qb-compiler.svg)](https://pypi.org/project/qb-compiler/)

**Calibration-aware quantum circuit compiler by [QubitBoost](https://qubitboost.io). Compile circuits that are less likely to fail on today's hardware.**

---

## Why qb-compiler?

- **Uses TODAY's calibration data** — not just topology. Gate errors, T1/T2 coherence, and readout fidelity change every calibration cycle. qb-compiler reads current data and uses it at every compilation stage.
- **Multi-vendor: IBM, Rigetti, IonQ, IQM from one API** — compile for any backend with a single interface. Native gate decomposition, calibration parsing, and cost estimation are handled per-vendor.
- **Budget-aware: compile within your cost constraints** — estimate execution cost before you run, enforce budget limits, and pick the cheapest backend that meets your fidelity target.

---

## Feature Comparison

| Feature | Standard Transpiler | qb-compiler |
|---------|---------------|-------------|
| Calibration-aware mapping | - | Yes |
| ML-accelerated layout | - | Yes (optional) |
| T1 asymmetry handling | - | Yes |
| Temporal correlation detection | - | Yes |
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

Topology-only compilation treats quantum hardware as a static graph — picking
qubit mappings and routing paths based on connectivity alone. But gate error
rates, coherence times, and readout fidelities change with every calibration
cycle. On IBM Heron processors, calibration data updates daily and qubit
quality can vary by 10x across the chip.

The result: **significant fidelity left on the table** because compilation
decisions aren't informed by which qubits are good *today*.

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

- **ML-Accelerated Layout** -- XGBoost model predicts the best physical qubits
  for your circuit, narrowing VF2 search from 156 qubits to ~20 candidates.
  **Up to 5% better fidelity** and **23x faster** than standard VF2 on 8-qubit
  circuits. Install with `pip install "qb-compiler[ml]"`.

- **CalibrationMapper** -- VF2 subgraph isomorphism search scored by gate error,
  coherence (T1/T2), readout fidelity, **T1 asymmetry**, and **temporal
  correlation**. Finds the best physical qubit placement for your circuit on
  today's device.

- **T1 Asymmetry Awareness** -- On IBM Heron, the probability of reading `0`
  when a qubit is in `|1⟩` (P(0|1)) can be **up to 24x higher** than P(1|0).
  qb-compiler goes beyond symmetric readout error and uses the raw asymmetric
  readout data to penalise high-asymmetry qubits for circuits that hold
  qubits in `|1⟩`.

- **Temporal Correlation Detection** -- When multiple calibration snapshots are
  available, qb-compiler detects qubit pairs whose error rates co-vary over
  time. Correlated errors break QEC's independent-error assumption. The mapper
  penalises correlated edges during layout selection.

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

- **Qiskit Plugin** -- Drop-in `AnalysisPass` that adds calibration-aware
  layout to any Qiskit pipeline. Works with `generate_preset_pass_manager`
  at any optimization level.

---

## Benchmarks

All tables use the same baseline and fidelity estimator. Reproduce with
`python scripts/benchmark_readme.py`.

### Core Compiler (no ML dependencies)

Calibration-aware VF2 mapping vs naive greedy placement, on real IBM Fez
calibration data (156 qubits, March 2026):

| Circuit | Qubits | 2Q Gates | Greedy | VF2 | Improvement |
|---------|-------:|---------:|-------:|----:|------------:|
| Bell    |      2 |        1 | 0.9868 | 0.9868 | +0.0% |
| GHZ-5   |      5 |        4 | 0.8923 | 0.9471 | **+6.1%** |
| GHZ-8   |      8 |        7 | 0.8372 | 0.8519 | **+1.8%** |
| QAOA-4  |      4 |        6 | 0.9018 | 0.9608 | **+6.5%** |
| QAOA-8  |      8 |       14 | 0.8204 | 0.8342 | **+1.7%** |

*Greedy = edge-ranked placement. VF2 = calibration-weighted subgraph
isomorphism (max_candidates=500, call_limit=50k).*

### With ML Acceleration (`pip install "qb-compiler[ml,gnn]"`)

XGBoost narrows VF2 search to ~20 candidates. GNN captures device topology
structure that flat features miss. Both shine on larger circuits:

| Circuit | Qubits | Greedy | VF2 | ML+VF2 | GNN+VF2 | Best vs Greedy |
|---------|-------:|-------:|----:|-------:|--------:|---------------:|
| Bell    |      2 | 0.9868 | 0.9868 | 0.9868 | 0.9868 | +0.0% |
| GHZ-5   |      5 | 0.8923 | 0.9471 | 0.9463 | 0.9440 | **+6.1%** |
| GHZ-8   |      8 | 0.8372 | 0.8519 | 0.8975 | 0.9062 | **+8.2%** |
| QAOA-4  |      4 | 0.9018 | 0.9608 | 0.9552 | 0.9608 | **+6.5%** |
| QAOA-8  |      8 | 0.8204 | 0.8342 | 0.8771 | 0.8880 | **+8.2%** |

*ML+VF2: XGBoost, AUC=0.94, 454 KB. GNN+VF2: dual-graph GCN, 8,833 params,
42 KB. Both trained on IBM Fez calibration data.*

### T1 Asymmetry: Beyond Symmetric Readout Error

Symmetric readout error models average P(0|1) and P(1|0), hiding T1-driven
asymmetry. qb-compiler models the raw asymmetric readout, revealing hidden
fidelity loss:

| Circuit | Qubits | Symmetric | Asymmetric | Overestimate |
|---------|-------:|----------:|-----------:|-------------:|
| Bell    |      2 | 0.9868    | 0.9851     | 0.17%        |
| GHZ-5   |      5 | 0.9471    | 0.9437     | 0.36%        |
| GHZ-8   |      8 | 0.8519    | 0.8481     | 0.45%        |
| QAOA-4  |      4 | 0.9608    | 0.9569     | 0.40%        |
| QAOA-8  |      8 | 0.8342    | 0.8304     | 0.45%        |

> The symmetric model overestimates fidelity because it hides T1-driven
> `|1⟩` decay. On IBM Fez, qubit asymmetry ratios range from 0.2x to 24x.
> Standard symmetric readout models miss this effect.

> **Footnote:** Greedy = edge-ranked qubit placement (no search). All fidelity
> estimates use per-qubit calibration data from IBM Fez (March 2026, 156 qubits).
> All tables use the same baseline and estimator.
> Reproduce: `python scripts/benchmark_readme.py`

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

# With ML-accelerated layout (XGBoost)
pip install "qb-compiler[ml]"

# With GNN layout predictor (PyTorch)
pip install "qb-compiler[gnn]"

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

Copyright 2026 [QubitBoost](https://www.qubitboost.io).
