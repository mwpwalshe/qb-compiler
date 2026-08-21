# qb-compiler 0.8.0: ObservableGate

The correctness-preflight release. ObservableGate, a QEC decoder-input correctness audit, plus a
hardened security-reviewed core.

## NEW

```
- ObservableGate      QEC decoder-input correctness preflight. a stim
                      Detector Error Model mechanism carries the detectors it
                      flips, the logical-observable masks it flips, and a
                      probability; a decoder predicts the logical frame from
                      the detector symptom. if a DEM-to-matrix step merges
                      mechanisms by detector signature alone, two mechanisms
                      that are detector-identical but logical-distinct
                      collapse into one and the logical mask is lost or
                      arbitrarily chosen. that is not a semantics-preserving
                      operation and it can inflate the logical error rate:
                      measured at ~60% relative LER inflation on a
                      color_code:memory_xyz distance-3 DEM under naive
                      detector-only merging. ObservableGate detects the
                      condition before decoding and offers an
                      observable-preserving fix.
- qbc dem-audit       PASS / WARN / FAIL on a DEM. CI-safe exit codes: 0
                      PASS, 1 WARN with --strict, 2 FAIL. --json emits a
                      machine-readable community-tier receipt.
- qbc dem-canonicalize
                      writes the observable-preserving canonical DEM. keeps
                      detector-identical / logical-distinct mechanisms
                      separate, merges only exact duplicates.
- qec_preflight       attaches an observable_audit receipt to its result
                      automatically.
```

```bash
# Audit a DEM. CI-safe exit codes: 0 = PASS, 1 = WARN (with --strict), 2 = FAIL.
qbc dem-audit model.dem
qbc dem-audit model.dem --strict
qbc dem-audit model.dem --json        # machine-readable community-tier receipt

# Write an observable-preserving canonical DEM (keeps detector-identical / logical-distinct
# mechanisms separate; merges only exact duplicates).
qbc dem-canonicalize model.dem -o safe.dem
```

```yaml
# Use it as a CI gate before decoder benchmarking or hardware submission:
- run: qbc dem-audit path/to/model.dem --strict
```

### Python API

```python
from qb_compiler.observable_gate import audit_dem, canonicalize_dem, preflight_dem_gate

result = audit_dem(dem)              # -> ObservableAuditResult (PASS / WARN / FAIL receipt)
print(result.status, result.mixed_groups, result.mixed_mass)
safe = canonicalize_dem(dem)          # observable-preserving canonical form
```

## FIXED (hardening pass)

```
- pickle-RCE          in the optional ML-decoder checkpoint loader.
                      torch.load now uses weights_only=True, and a CI guard
                      fails the build on any unsafe deserialization in src/.
- resource exhaustion a hang in qbc compile on pathological qubit counts. now
                      a bounded, clean error.
- CLI input handling  hardened across endpoints. malformed, binary and
                      wrong-type files and bad output paths produce clear
                      errors and correct exit codes instead of tracebacks.
- HTTP calibration client
                      confirmed hardened: TLS verification, timeouts,
                      bounded redirects.
- gates               709 tests, unit plus security plus adversarial. ruff,
                      ruff format and mypy clean.
```

## SCOPE (honest)

Standard production paths audit **PASS** and are unaffected: `surface_code`, `repetition_code`, and
the full **bivariate-bicycle / Gross family**, `[[72,12,6]]` ... `[[144,12,12]]` (the Gross code)
... `[[288,12,18]]`, in both X and Z basis, 20 configs swept, all PASS. Decomposed DEMs are
XOR-benign and handled correctly by standard converters. The hazard is real and measured on
**graphlike DEMs with genuine detector-identical / logical-distinct mechanisms**. ObservableGate
detects that condition and offers the fix. It does **not** claim any specific decoder is broken.

ObservableGate and its receipts are free and open source. Team and enterprise receipt workflows,
signed receipts, batch reports, shared dashboards and CI policy bundles are coming under QubitBoost
Pro (see [`docs/open-core.md`](../open-core.md)). Pro is additive and never blocks the open-source
path.

Docs: [`docs/observablegate.md`](../observablegate.md).

## UPGRADING

```bash
pip install -U qb-compiler            # core (IBM backends via Qiskit)
pip install -U "qb-compiler[ising]"   # + QEC extra (stim, pymatching) for ObservableGate / qec_preflight
```

Full changelog: [CHANGELOG.md](../../CHANGELOG.md). Demo notebook:
`notebooks/21_observablegate_qec_preflight.ipynb`.
