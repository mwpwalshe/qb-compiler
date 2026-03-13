# qb-compiler Benchmark Suite

Benchmark circuits and tooling for measuring qb-compiler performance against baseline transpilers.

## Circuit Library

The `circuits/` directory contains OpenQASM 2.0 circuits organised by algorithm family:

### QAOA (Quantum Approximate Optimization Algorithm)

| File | Qubits | Description |
|------|--------|-------------|
| `qaoa/maxcut_4q.qasm` | 4 | MaxCut on a 4-node ring graph, p=1 |
| `qaoa/maxcut_8q.qasm` | 8 | MaxCut on an 8-node ring graph, p=2 |

QAOA circuits contain alternating problem (ZZ interactions via CX-RZ-CX) and mixer (RX) layers. These stress-test two-qubit gate routing and calibration-aware mapping since CNOT fidelity dominates overall circuit fidelity.

### VQE (Variational Quantum Eigensolver)

| File | Qubits | Description |
|------|--------|-------------|
| `vqe/h2_4q.qasm` | 4 | H2 molecule, minimal UCCSD ansatz |
| `vqe/h2_8q.qasm` | 8 | H2 molecule, UCCSD-inspired ansatz |

VQE circuits use Hartree-Fock initial states with parameterised rotation layers and CNOT ladders. They test the compiler's ability to handle linear entanglement patterns and preserve variational structure.

### QFT (Quantum Fourier Transform)

| File | Qubits | Description |
|------|--------|-------------|
| `qft/qft_8q.qasm` | 8 | Standard 8-qubit QFT |
| `qft/qft_16q.qasm` | 16 | Standard 16-qubit QFT |

QFT circuits use Hadamard gates, controlled-phase (cp) gates, and final swap operations. They produce dense interaction graphs that challenge qubit mapping and routing passes, especially at 16 qubits.

### Random Circuits

| File | Qubits | Depth | Description |
|------|--------|-------|-------------|
| `random/random_5q_depth20.qasm` | 5 | ~20 | Mixed gate set, moderate depth |
| `random/random_10q_depth30.qasm` | 10 | ~30 | Mixed gate set, high depth |

Random circuits use a mix of H, RZ, RY, RX, X, and CX gates with varied connectivity patterns. They provide a general-purpose stress test that does not favour any particular optimisation strategy.

## Running Benchmarks

### Quick run (all backends)

```bash
python benchmarks/run_benchmarks.py
```

This compiles a set of built-in circuits against IBM, Rigetti, and IonQ backends and prints a Rich-formatted table with depth reduction, gate count, estimated fidelity, and compilation time.

### Using pytest-benchmark

```bash
# Run benchmark tests with timing
pytest tests/benchmarks/ --benchmark-json benchmarks/results/latest.json

# Compare against a saved baseline
pytest tests/benchmarks/ --benchmark-compare=benchmarks/results/baseline.json
```

### Specific circuit families

```bash
# Run only QAOA benchmarks
pytest tests/benchmarks/bench_compilation_time.py -k "qaoa"

# Run only fidelity improvement benchmarks
pytest tests/benchmarks/bench_fidelity_improvement.py
```

## Interpreting Results

The benchmark suite tracks these key metrics:

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Depth reduction** | `(original - compiled) / original` | > 10% vs Qiskit default |
| **CNOT reduction** | Reduction in two-qubit gates | > 5% vs Qiskit default |
| **Estimated fidelity** | Pre-execution fidelity estimate from noise model | Higher is better |
| **Compilation time** | Wall-clock time for full pipeline | < 5s for 100-qubit circuits |
| **Fidelity improvement** | Fidelity gain vs Qiskit default transpile | > 0% (our key selling point) |

### What to look for

- **Calibration-aware mapping** should show measurable fidelity improvement over topology-only mapping, especially on backends with non-uniform error rates.
- **Depth reduction** should be comparable to or better than Qiskit at the same optimisation level.
- **Compilation time** should remain practical. A 10x fidelity improvement that takes 10 minutes to compile is not useful for iterative VQE/QAOA workflows.

## Results Directory

The `results/` directory stores benchmark output JSON files. These are git-tracked to enable regression detection across commits. The CI pipeline (`.github/workflows/ci.yml`) automatically runs benchmarks and compares against the previous baseline on each PR.

## Adding New Benchmark Circuits

1. Write the circuit as valid OpenQASM 2.0 in the appropriate `circuits/<family>/` directory.
2. Use descriptive filenames: `<algorithm>_<qubits>q.qasm` or `<algorithm>_<qubits>q_<variant>.qasm`.
3. Include a comment header describing the circuit, its qubit count, and any fixed parameters.
4. Add the circuit to the relevant benchmark test in `tests/benchmarks/`.
