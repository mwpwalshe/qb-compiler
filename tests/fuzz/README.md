# Fuzz Tests for qb-compiler

This directory contains [atheris](https://github.com/google/atheris)-based fuzz tests
for the qb-compiler project.

## Prerequisites

```bash
pip install atheris
```

## Fuzz Targets

| File | Target | What it fuzzes |
|------|--------|----------------|
| `fuzz_qasm_parser.py` | `from_qasm()` | OpenQASM 2.0 parsing with random/corrupted QASM strings |
| `fuzz_circuit_ir.py` | `QBCircuit`, `QBDag` | Circuit IR with invalid gates, NaN params, DAG round-trips |
| `fuzz_calibration_parser.py` | `BackendProperties.from_qubitboost_dict()` | Calibration parsing with missing/invalid fields |
| `fuzz_compiler_config.py` | `CompilerConfig` | Config validation with path traversal, invalid levels |
| `fuzz_cli_input.py` | `cli` (Click) | CLI with random backend names, strategies, file paths |

## Running

Each fuzzer runs as a standalone script. Use `-max_total_time=N` to limit
the run duration in seconds:

```bash
# Run a single fuzzer for 60 seconds
python tests/fuzz/fuzz_qasm_parser.py -max_total_time=60

# Run all fuzzers for 30 seconds each
for f in tests/fuzz/fuzz_*.py; do
    echo "=== $f ==="
    timeout 35 python "$f" -max_total_time=30
done
```

Crash-reproducing inputs are saved to the current directory by libFuzzer.
To reproduce a crash:

```bash
python tests/fuzz/fuzz_qasm_parser.py /path/to/crash-input
```

## Adding a New Fuzzer

Follow this pattern:

```python
import atheris
import sys

with atheris.instrument_imports():
    from qb_compiler.xxx import yyy

def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    try:
        # feed fuzzed data to the target
        ...
    except (ExpectedExceptionType,):
        pass

def main():
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()

if __name__ == "__main__":
    main()
```

Only catch exception types that represent expected input validation failures.
Unexpected crashes (segfaults, assertion errors, uncaught exceptions) should
be investigated and fixed in the source code.
