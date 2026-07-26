# Adversarial Hardening Report: qb-compiler

Offensive/adversarial pass over every CLI command, the ObservableGate API, and the
QASM/DEM parsers. Goal: hostile inputs must degrade gracefully (clean error + nonzero
exit for CLI, typed exception for API), never a raw traceback, hang, or unbounded
resource use.

Tests live in `tests/adversarial/`. Source fixes are minimal input-validation /
`try-except` guards in `src/qb_compiler/cli/main.py` and
`src/qb_compiler/observable_gate.py`. No existing behavior was weakened.

## Real defects found and fixed

| Endpoint | Attack | Before behavior | Fix / locked |
|---|---|---|---|
| `compile` | non-UTF8 / binary / `.png` file as `.qasm` | **Uncaught `UnicodeDecodeError`** (raw traceback) | `path.read_text` wrapped → clean "could not read … as UTF-8" + exit 1. FIXED |
| `compile` | `qreg q[999999999]` (pathological qubit count) | **HANG / OOM** (timed out >30s building the circuit) | `_MAX_QUBITS=100_000` guard after parse → clean error + exit 1 in <0.01s. FIXED |
| `compile -o <dir>` | output path is an existing directory | **Uncaught `IsADirectoryError`** | output write wrapped → "could not write output to …" + exit 1. FIXED |
| `compile -o <nodir>/x` | output parent dir missing | **Uncaught `FileNotFoundError`** | same wrapper. FIXED |
| `compile` (compile stage) | inputs that make the compiler raise | unwrapped, would traceback | `compiler.compile` wrapped → clean error + exit 1. FIXED (defensive) |
| `compile --receipt` | receipt dir not writable | unwrapped `write_text` | wrapped → clean error + exit 1. FIXED (defensive) |
| `dem-audit` / `dem-canonicalize` | malformed DEM text | **Uncaught `IndexError: Unrecognized instruction`** | `_load_dem` wraps `from_file` → "could not parse … as a stim DEM" + exit 2. FIXED |
| `dem-audit` / `dem-canonicalize` | a stim **circuit** passed as a `.dem` | **Uncaught `IndexError`** | same `_load_dem` guard + hint. FIXED |
| `dem-audit` / `dem-canonicalize` | binary / non-UTF8 `.dem` | **Uncaught `IndexError`** | same. FIXED |
| `dem-audit -o` / `dem-canonicalize -o` | output is a directory / parent missing | **Uncaught `ValueError: Failed to open`** | `_write_dem` wraps `to_file` → "could not write DEM to …" + exit 1. FIXED |
| `audit_matrices` (API) | `obs_matrix` / `priors` column count ≠ `check_matrix` | **Bare `IndexError`** deep in loop | shape check → `ValueError("column-count mismatch …")`. FIXED |
| `audit_matrices` (API) | 1-D / flat / empty arrays | **`IndexError: tuple index out of range`** | ndim check → `ValueError("must be 2-D …")`. FIXED |
| `audit_matrices` (API) | non-numeric (string) dtype | cryptic numpy cast `ValueError` | wrapped → `ValueError("must be numeric arrays …")`. FIXED |
| `audit_matrices` (API) | `priors` 2-D | would mis-broadcast silently | ndim check → `ValueError("priors must be 1-D")`. FIXED |

## Already robust (locked in with tests, no code change needed)

| Endpoint | Attack | Behavior (confirmed safe) |
|---|---|---|
| all CLI | nonexistent file / directory-as-file arg | `click.Path(exists=True, dir_okay=False)` → clean usage error, exit 2 |
| all CLI | missing required arg / unknown command / unknown subcommand | click usage error, exit 2 |
| `compile` | empty file | "could not determine qubit count", exit 1 |
| `compile` | deeply nested `(((…)))` param expression | parser is regex-based (no eval, no recursion) → no hang, exit cleanly |
| `preflight` / `analyze` | bad / emoji / unicode backend name | graceful fallback, no traceback |
| `calibration show` | unknown backend | "Error: …", exit 1 |
| `dem-audit` | empty `.dem` file | PASS, exit 0 |
| `dem-audit` | witness FAIL (detector-identical, logical-distinct) | FAIL, exit 2 (CI-gate semantics intact) |
| `audit_dem` / `preflight_dem_gate` | empty DEM, DEM with no detectors/observables | PASS / typed result, no crash |
| `audit_dem` | 30k-mechanism DEM, 2k mixed collapse-risk pairs | completes <2s, no OOM; mixed pairs correctly FAIL |
| `audit_matrices` | NaN / negative priors, zero mechanisms | typed result, no crash |
| `canonicalize_dem` | detector-identical / logical-distinct mechanisms | masks preserved (stays 2 mechanisms, never merged) |

## Verification

- `ruff check src tests` → All checks passed.
- `python -m pytest tests/adversarial tests/unit tests/security -q` → **709 passed**.
- Adversarial suite alone: 46 tests, ~10s (oversized inputs capped to keep it fast).
