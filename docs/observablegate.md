# ObservableGate — QEC decoder-input correctness preflight

ObservableGate audits a quantum-error-correction **Detector Error Model (DEM)** for *observable-mask
collapse* before it is handed to a decoder, and can rewrite it into an observable-preserving canonical
form. It is a correctness check, not a performance optimization.

## The problem

A stim DEM error mechanism carries three things: the **detectors** it flips (`H`), the **logical
observables** it flips (`L`), and a **probability** (`p`). A decoder's job is to predict the logical
*frame* (`L`) from the detector *symptom* (`H`).

Two mechanisms can be **detector-identical but logical-distinct**:

```
error(0.01) D0
error(0.01) D0 L0
```

Both flip detector `D0`; only one flips logical `L0`. If a DEM-to-matrix canonicalization merges columns
by detector signature `H` **alone**, these collapse into a single column and the logical mask is lost or
arbitrarily chosen. That is **not** a semantics-preserving operation, and it can inflate the logical
error rate.

**Measured harm:** on `color_code:memory_xyz` distance 3, decoding the same real syndromes with BP+OSD,
merging detector-identical mechanisms while keeping an arbitrary logical mask raises the logical error
rate from **8.2% to 13.1%** (~60% relative). Preserving the distinct `(H, L)` columns (or keeping the
dominant mask) removes the harm.

## The invariant

Group DEM columns by detector signature `H`. The DEM is unsafe for detector-only merging when

```
unique(H)  <  unique(H, L)
```

i.e. some detector signature carries more than one logical mask. ObservableGate reports the number of
such **mixed groups**, the **probability mass** inside them, and the **worst mask ratio** (the
second-largest vs largest competing-mask probability; `1.0` means maximally ambiguous).

Status:

| Status | Meaning |
|--------|---------|
| `PASS` | no mixed groups — detector-only canonicalization is observable-safe |
| `WARN` | mixed groups present but the DEM is decomposed (likely XOR-benign; review) |
| `FAIL` | genuine detector-identical / logical-distinct mechanisms — do **not** merge by detector alone |

## CLI

```bash
# Audit a DEM. Exit code: 0 = PASS, 1 = WARN (with --strict), 2 = FAIL. CI-safe.
qbc dem-audit model.dem
qbc dem-audit model.dem --strict        # also fail on WARN

# Write an observable-preserving canonical DEM.
qbc dem-canonicalize model.dem -o safe.dem
```

Use it as a CI gate before decoder benchmarking or hardware submission:

```yaml
- name: ObservableGate DEM audit
  run: qbc dem-audit path/to/model.dem --strict
```

## Python API

```python
from qb_compiler.observable_gate import (
    audit_dem,            # audit a stim DEM  -> ObservableAuditResult
    audit_matrices,       # audit (check_matrix, obs_matrix, priors) numpy arrays (no stim)
    canonicalize_dem,     # observable-preserving canonical DEM
    preflight_dem_gate,   # raises ObservableMaskCollapseError on FAIL
)

result = audit_dem(dem)
print(result.status, result.mixed_groups, result.mixed_mass)
if not result.ok:                       # FAIL
    safe = canonicalize_dem(dem)        # distinct (H, L) preserved, never erased
```

`qec_preflight()` attaches an `observable_audit` receipt to its result automatically.

## The fix

Canonicalize by `(H, L)` — keep distinct `(detectors, observables)` pairs as separate columns, merging
only **exact** `(H, L)` duplicates (XOR-combining their probabilities). The detector-identical /
logical-distinct ambiguity is *intrinsic* and is preserved, never erased; the actionable signal is "do
not feed this DEM to a detector-signature-only merge path." If you must merge by `H`, carry the logical
mixture `P(L | H)` rather than a single mask.

## Scope (honest)

Standard production paths audit **PASS**:

- `surface_code` and `repetition_code` (raw DEMs have no mixed groups);
- the full **bivariate-bicycle / Gross family** — `[[72,12,6]]`, `[[90,8,10]]`, `[[108,8,10]]`,
  `[[144,12,12]]` (the Gross code), `[[288,12,18]]`, in both X and Z basis (20 configs swept, all PASS);
- decomposed DEMs show mixed groups but are XOR-benign and are handled correctly by standard converters.

The hazard is real and measured on **graphlike DEMs with genuine detector-identical / logical-distinct
mechanisms** (e.g. `color_code:memory_xyz`, hand-built witnesses). ObservableGate's job is to detect that
condition and offer the observable-preserving fix — not to claim any specific decoder is broken.
