# Contributing to qb-compiler

Thank you for your interest in contributing to qb-compiler. This guide covers
everything you need to get started.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/qb-compiler.git
   cd qb-compiler
   ```
3. Set up your development environment:
   ```bash
   make dev
   ```
   This installs the package in editable mode with all development dependencies.

## Development Setup

**Requirements:**
- Python 3.10 or later
- A virtual environment is strongly recommended

**Manual setup (alternative to `make dev`):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Pre-commit hooks:**
```bash
pre-commit install
```

This enables automatic linting and formatting checks before each commit.

## Code Style

- **Linter/formatter:** ruff (line-length 100).
- **Type checking:** mypy in strict mode.
- **Future annotations:** Every module must begin with `from __future__ import annotations`.
- **Docstrings:** Use Google-style docstrings for all public classes and functions.
- **Imports:** Sort with ruff's isort rules. Prefer absolute imports.

Run checks locally:
```bash
make lint      # ruff check + ruff format --check
make typecheck # mypy
```

## Testing

Run the full test suite:
```bash
make test
```

**Pytest markers:**
| Marker        | Purpose                                  |
|---------------|------------------------------------------|
| `unit`        | Fast, isolated unit tests (default)      |
| `integration` | Tests that exercise multiple components   |
| `benchmark`   | Performance benchmarks (excluded by default) |

Run a specific marker:
```bash
pytest -m unit
pytest -m integration
pytest -m benchmark
```

**Test naming conventions:**
- Test files: `test_<module>.py`
- Test functions: `test_<behaviour_under_test>`
- Place tests in `tests/` mirroring the `src/` directory structure.

**Coverage:** Aim for at least 80% line coverage on new code. Check with:
```bash
pytest --cov=qb_compiler --cov-report=term-missing
```

## Architecture Overview

qb-compiler uses a linear pass pipeline that transforms circuits through
well-defined stages:

```
Input (Qiskit / OpenQASM 2.0)
  -> IR conversion (QBCircuit / QBDag)
  -> Mapping (CalibrationMapper — VF2 placement)
  -> Routing (NoiseAwareRouter — SWAP insertion)
  -> Scheduling (NoiseAwareScheduler — ALAP with T1/T2 urgency)
  -> Decomposition (GateDecomposition — native basis gates)
  -> Analysis (depth, gate count, error budget estimation)
  -> Output (QBCircuit / Qiskit QuantumCircuit)
```

Each pass inherits from `BasePass` and implements a `run(dag: QBDag) -> QBDag`
method. Passes are stateless: all configuration is supplied at construction time.

## Adding a New Pass

1. **Create the pass** in `src/qb_compiler/passes/`:
   ```python
   from __future__ import annotations
   from qb_compiler.passes.base import BasePass
   from qb_compiler.ir.dag import QBDag

   class MyNewPass(BasePass):
       """One-line description of what this pass does."""

       def run(self, dag: QBDag) -> QBDag:
           # Transform the DAG and return it.
           return dag
   ```
2. **Export** the pass from `src/qb_compiler/passes/__init__.py`.
3. **Write tests** in `tests/passes/test_my_new_pass.py`. Include at least:
   - A trivial circuit (1-2 qubits) verifying correctness.
   - An edge case (empty circuit, single gate, barriers).
4. **Integrate** the pass into the pipeline in the appropriate position.
5. **Document** the pass with a docstring and, if user-facing, update the README.

## Adding a Backend

Backend configurations live in `src/qb_compiler/backends/`. To add support for
a new hardware backend:

1. **Create a configuration module** (e.g., `my_backend.py`) that defines:
   - Native gate set and gate durations.
   - Connectivity map (coupling graph).
   - Default noise parameters (T1, T2, gate fidelities).
2. **Implement a calibration provider** if the backend exposes live calibration
   data, extending `BaseCalibrationProvider`.
3. **Add decomposition rules** in `src/qb_compiler/passes/decomposition.py` for
   the backend's native gate set.
4. **Write tests** covering mapping, routing, and decomposition for the new
   backend's topology.
5. **Add a configuration entry** so the CLI and transpiler plugin can reference
   the backend by name.

## ML Models

ML models live in `src/qb_compiler/ml/`. The pipeline has four phases:

| Phase | Module | Architecture | Install |
|-------|--------|-------------|---------|
| 1 | `data_generator.py` | Training data pipeline | Core |
| 2 | `layout_predictor.py` | XGBoost qubit scorer | `[ml]` |
| 3 | `gnn_router.py` | Dual-graph GCN | `[gnn]` |
| 4 | `rl_router.py` | PPO SWAP routing | `[gnn]` |

**Model weights** are stored in `src/qb_compiler/ml/_weights/` with
`*.meta.json` metadata files. To retrain:

```bash
# Phase 2: XGBoost
python -m qb_compiler.ml.train

# Phase 3: GNN
python -c "from qb_compiler.ml.gnn_router import train_gnn_model; train_gnn_model()"
```

**Adding a new ML model:**
1. Create the model in `src/qb_compiler/ml/`.
2. Follow the `predict_candidate_qubits(circuit, backend) -> list[int]` interface
   so it's plug-compatible with `CalibrationMapper`.
3. Save weights in `_weights/` with a `.meta.json` metadata file.
4. Write tests in `tests/unit/test_ml/`.
5. Guard imports behind optional dependency checks (see `ml/__init__.py`).

## Pull Request Process

**Branch naming:**
- `feat/<short-description>` for new features
- `fix/<short-description>` for bug fixes
- `docs/<short-description>` for documentation changes
- `refactor/<short-description>` for refactoring

**Commit messages:** Use conventional commit style:
```
feat: add Quantinuum H2 backend configuration
fix: correct SWAP count in noise-aware routing
docs: update calibration provider examples
```

**Before opening a PR:**
1. Ensure all checks pass: `make lint && make typecheck && make test`
2. Add or update tests for any changed behaviour.
3. Update documentation if the public API changed.

**PR expectations:**
- Keep PRs focused on a single concern.
- Provide a clear description of what changed and why.
- Link any related issues.
- Respond to review feedback promptly.

## Code of Conduct

This project follows the
[Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating, you agree to uphold a welcoming and inclusive environment.

Please report unacceptable behaviour to the maintainers.

## License

By contributing to qb-compiler, you agree that your contributions will be
licensed under the [Apache License 2.0](LICENSE), the same license that covers
the project.
