# qb-compiler Security Audit

Scope: hardening the `qb-compiler` library for public (pip-install) release.
Focus on genuine, exploitable risk for code that third parties install and run.
Audit performed against `master` @ working tree, 2026-06-27.

Tooling versions used: `ruff 0.15.6`, `pip-audit` (latest), `torch 2.9.1`,
`httpx 0.28.1`, `stim 1.15.0`, `pymatching 2.3.1`, Python 3.11.

## Findings

| # | Target | Severity | Status | Regression test |
|---|--------|----------|--------|-----------------|
| 1 | `ising/decoder.py` `IsingDecoderWrapper._load_checkpoint` — `torch.load(..., weights_only=False)` on a third-party `.pt` checkpoint | **CRITICAL** | **FIXED** | `test_model_loading_rce.py` |
| 2 | `ml/gnn_router.py` / `ml/rl_router.py` model loading (`torch.load`) | CRITICAL (potential) | Safe already (used `weights_only=True`) + locked by static guard | `test_no_unsafe_deserialization.py` |
| 3 | `calibration/live_provider.py` `create_hardened_http_client` (TLS/timeouts/redirects/SSRF/JSON-bomb) | LOW | Safe already; `verify=True` pinned explicitly (defense-in-depth) | `test_http_client_hardening.py` |
| 4 | `cli/main.py` `__import__(import_name)` (~line 357) | — | Safe already (fixed allowlist) | n/a (see notes) |
| 5 | File-output paths (`-o` in compile / dem-audit / dem-canonicalize / receipt) | — | Safe already (local CLI, `click.Path`, user owns target) | n/a |
| 6 | Parser DoS — OpenQASM parser + DEM parsing | — | Safe already (no-eval AST parser, graceful degradation; DEM parsed by trusted `stim`) | n/a |
| 7 | Dependency CVEs + hardcoded secrets | — | Safe already (`pip-audit`: no known vulns; no secrets found) | n/a |

## Detail

### 1. CRITICAL — Arbitrary code execution via `torch.load` (pickle RCE) — FIXED

`IsingDecoderWrapper._load_checkpoint` loaded a non-`.safetensors` checkpoint with:

```python
loaded = torch.load(path, map_location=self.device, weights_only=False)
```

The checkpoint is a NVIDIA Ising pre-decoder `.pt` file the user downloads from
HuggingFace — i.e. **untrusted third-party input**. `torch.load` with
`weights_only=False` unpickles arbitrary Python objects, so a malicious
checkpoint runs attacker code at load time via a pickle `__reduce__` gadget
(no decoding even needs to happen). This is the classic ML-supply-chain RCE.

**Fix** (`src/qb_compiler/ising/decoder.py`): load with `weights_only=True`,
which restricts unpickling to tensors + a small allowlist of safe primitives,
and raise a clear `ValueError` if a checkpoint carries unsafe pickled objects:

```python
try:
    loaded = torch.load(path, map_location=self.device, weights_only=True)
except Exception as exc:
    raise ValueError("Failed to safely load checkpoint ... weights_only=True ...") from exc
```

`weights_only=True` is fully compatible with the existing handling: plain
state-dicts and `{"model_state_dict": ...}` / `{"state_dict": ...}` wrappers
(dicts of tensors + str/int) load unchanged. The `.safetensors` path was
already safe (the safetensors format cannot carry executable pickle).

**Regression test** `test_model_loading_rce.py`:
- builds a real pickle gadget (`__reduce__` writes a marker file),
- a control test proves the gadget fires under `weights_only=False`,
- asserts the gadget does **not** execute when loaded through the fixed
  decoder, and that the loader raises instead,
- asserts legitimate `state_dict` and `{"model_state_dict": ...}` checkpoints
  still load (no API/behaviour regression).

Verified the test **fails on the pre-fix code** (gadget executes) and **passes
after the fix**.

### 2. CRITICAL (potential) — GNN/RL model loading — SAFE ALREADY

`ml/gnn_router.py:407` and `ml/rl_router.py:433` already use
`torch.load(..., weights_only=True)`. No change needed. To prevent a silent
regression here (or anywhere), `test_no_unsafe_deserialization.py` statically
scans `src/` and fails if any `weights_only=False`, any `torch.load(` without
`weights_only=True`, any `pickle.load`, or unsafe `yaml.load` appears. This
guard runs without torch installed.

### 3. LOW — Hardened HTTP client — SAFE ALREADY (+ pinned `verify=True`)

`create_hardened_http_client` already sets explicit timeouts
(connect/read/write/pool), bounded redirects (`max_redirects=5`), and
connection limits; httpx verifies TLS by default. No exploitable path exists in
this repo: the helper is the only HTTP construct, nothing in `src/` fetches or
parses untrusted JSON through it (live calibration fetching is delegated to the
proprietary `qubitboost-sdk`, out of scope). There is therefore no SSRF sink
and no JSON-bomb sink to fix here.

As defense-in-depth for downstream consumers, `"verify": True` is now pinned
explicitly in the defaults so the hardened baseline never silently relies on
the library default and any caller disabling verification must do so visibly via
an override. `test_http_client_hardening.py` locks down verify-on, bounded
timeouts, and bounded redirects.

> Recommendation for whoever wires the actual remote fetch later: enforce
> `https` scheme + a host allowlist, and stream responses with a byte cap
> before `response.json()` to close SSRF / JSON-bomb at the real call site.

### 4. `__import__(import_name)` in CLI `doctor` — SAFE ALREADY

`import_name` iterates a hardcoded literal list
`[("numpy","numpy"), ("rustworkx","rustworkx"), ("rich","rich")]`. It is never
user-controlled, so there is no arbitrary-import risk.

### 5. File-output paths — SAFE ALREADY

`-o` outputs go through `click.Path` and `Path(output).write_text(...)` /
`stim ... .to_file(...)`. For a local CLI the user owns the target path;
overwriting an explicitly named output is intended behaviour, and there is no
path-injection vector (no shell, no template expansion).

### 6. Parser DoS — SAFE ALREADY

`ir/converters/openqasm_converter.py` evaluates QASM parameter expressions with
a tokenizer + restricted-AST evaluator (`_safe_eval`): a strict character
allowlist (`[\d.\+\-\*/eE\s]`, no parentheses → no deep nesting), only
`Add/Sub/Mult/Div` binary ops and unary `+/-`, and no `eval`. Pathological input
degrades gracefully — any parse/recursion failure is caught and the parameter
falls back to `0.0`. DEM files are parsed by `stim.DetectorErrorModel.from_file`
(trusted compiled dependency), not by in-repo code. No fix required.

### 7. Dependencies + secrets — SAFE ALREADY

- `pip-audit` over the declared core dependencies: **no known vulnerabilities**.
- Grep for hardcoded credentials/tokens/API keys across `src/`: **none found**.
  `QUBITBOOST_API_KEY` is read from the environment; no secrets are embedded.

## Verification

- `ruff check src tests` — passes.
- `python -m pytest tests/unit -q` — 624 passed (unchanged from baseline).
- `python -m pytest tests/security -q` — 12 passed.
