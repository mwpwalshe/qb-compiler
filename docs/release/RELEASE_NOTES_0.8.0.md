# qb-compiler 0.8.0 — ObservableGate

**The correctness-preflight release.** qb-compiler gains **ObservableGate**, a QEC decoder-input
correctness audit, and ships a hardened, security-reviewed core.

`pip install -U qb-compiler` · `pip install -U "qb-compiler[ising]"` (adds the QEC extra: stim + pymatching)

---

## ✨ New: ObservableGate — QEC decoder-input correctness preflight

A quantum-error-correction **Detector Error Model (DEM)** error mechanism carries three things: the
detectors it flips, the **logical-observable masks** it flips, and a probability. A decoder predicts the
logical *frame* from the detector *symptom*.

The hazard: if a DEM-to-matrix step merges mechanisms by **detector signature alone**, two mechanisms
that are *detector-identical but logical-distinct* collapse into one — and the logical mask is lost or
arbitrarily chosen. That is not a semantics-preserving operation, and it can inflate the logical error
rate. (Measured: ~60% relative LER inflation on a `color_code:memory_xyz` distance-3 DEM under naive
detector-only merging.)

ObservableGate detects this condition **before decoding** and offers an observable-preserving fix.

### New commands

```bash
# Audit a DEM. CI-safe exit codes: 0 = PASS, 1 = WARN (with --strict), 2 = FAIL.
qbc dem-audit model.dem
qbc dem-audit model.dem --strict
qbc dem-audit model.dem --json        # machine-readable community-tier receipt

# Write an observable-preserving canonical DEM (keeps detector-identical / logical-distinct
# mechanisms separate; merges only exact duplicates).
qbc dem-canonicalize model.dem -o safe.dem
```

> ObservableGate and its receipts are **free and open source**. Team/enterprise receipt workflows —
> signed receipts, batch reports, shared dashboards, and CI policy bundles — are coming under
> **QubitBoost Pro** (see [`docs/open-core.md`](../open-core.md)). Adopt the free CLI now; Pro is
> additive and never blocks the open-source path.

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

`qec_preflight()` now attaches an `observable_audit` receipt to its result automatically.

### Honest scope

Standard production paths audit **PASS** and are unaffected: `surface_code`, `repetition_code`, and the
full **bivariate-bicycle / Gross family** — `[[72,12,6]]` … `[[144,12,12]]` (the Gross code) …
`[[288,12,18]]`, in both X and Z basis (20 configs swept, all PASS). Decomposed DEMs are XOR-benign and
handled correctly by standard converters. The hazard is real and measured on **graphlike DEMs with
genuine detector-identical / logical-distinct mechanisms**. ObservableGate's job is to *detect* that
condition and offer the fix — **not** to claim any specific decoder is broken.

Docs: [`docs/observablegate.md`](../observablegate.md).

---

## 🛡 Hardened core

This release also includes a security + adversarial hardening pass:

- **Fixed a pickle-RCE** in the optional ML-decoder checkpoint loader (`torch.load` now uses
  `weights_only=True`); added a CI guard that fails the build on any unsafe deserialization in `src/`.
- **Fixed a resource-exhaustion hang** in `qbc compile` on pathological qubit counts (now a bounded,
  clean error).
- Hardened CLI input handling across endpoints: malformed / binary / wrong-type files and bad output
  paths now produce clear errors and correct exit codes instead of tracebacks.
- HTTP calibration client confirmed hardened (TLS verification, timeouts, bounded redirects).

**709 tests** pass (unit + security + adversarial); `ruff`, `ruff format`, and `mypy` clean.

---

## Install / upgrade

```bash
pip install -U qb-compiler            # core (IBM backends via Qiskit)
pip install -U "qb-compiler[ising]"   # + QEC extra (stim, pymatching) for ObservableGate / qec_preflight
```

## Full changelog

See [CHANGELOG.md](../../CHANGELOG.md). Highlights: ObservableGate (`dem-audit`, `dem-canonicalize`),
`qec_preflight` observable-audit receipt, security + adversarial hardening, new tested demo notebook
(`notebooks/21_observablegate_qec_preflight.ipynb`).

---

*qb-compiler is the open-core preflight layer for quantum execution: circuit viability, backend
recommendation, cost estimation, calibration-aware compilation, and now QEC decoder-input correctness.*
