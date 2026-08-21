# Changelog

All notable changes to [qb-compiler](https://qubitboost.io/compiler), the open-source quantum circuit compiler by [QubitBoost](https://qubitboost.io), will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.12.0 - 2026-08-20

Full notes: `docs/release/RELEASE_NOTES_0.12.0.md`.

```
FIXED
- selection_receipt   now records the executed layout. previously recorded the
                      recommendation regardless of what ran. pass
                      executed_layout=, get the layout that ran, the
                      recommendation, and score_penalty_vs_recommended.
                      schema stays qb.selection_receipt.v1, additions additive;
                      score_breakdown comes back empty with a note when
                      executed differs from recommended, because the breakdown
                      belongs to the recommendation.
- signing             one persistent ed25519 key at ~/.qb-compiler/signing_key
                      (0600, or QBC_SIGNING_KEY), fingerprint in the receipt.
                      previously sign=True minted a keypair per call and
                      embedded its public half in the receipt it signed, which
                      attests to nothing; treat no 0.9.0 to 0.11.0 signature as
                      evidence of origin, re-issue anything load bearing.
                      sign=True raises when it cannot sign.
- calibration age     calibration_freshness reads ISO strings, datetimes and
                      epochs. previously datetimes only, so most snapshots
                      reported no age. timestamp_status: measured / absent /
                      unreadable / synthetic / clock-skewed.
- qec_preflight       matching_faithfulness field, always "not_assessed".
                      nothing here checks whether matching is faithful on the
                      model it was handed. the gap is named, not closed.
- badges              coverage names the enforced floor and links to it,
                      previously a hardcoded 60 percent behind an empty link.
                      docs badge points at the docs site. unsigned receipt
                      string says "unsigned", the SDK pitch is gone.

ADDED (all free)
- qbc verify-receipt  offline signature check for any receipt this package
                      emits. VERIFIED / UNSIGNED / NO_KEY / INVALID_SIGNATURE /
                      LEGACY_SELF_SIGNED / MALFORMED, only VERIFIED passes.
                      pure python ed25519 fallback, works on a base install.
- qbc chem-audit      five integrity checks on a qubit hamiltonian file.
                      ACCEPT / INCOMPLETE / REFUSE, --strict, --json. reads
                      grouped JSON, flat terms, OpenFermion, SparsePauliOp.
- qbc measure-plan    measurable terms, QWC settings, grouping factor, largest
                      group, shots at your rate. structural counts only.
                      corpus it was built against: setting count spans 1.79x
                      at 12 qubits, 1.59x at 16; qubit count does not predict
                      where a molecule lands.
- qbc corpus          list / show / verify. pinned sha256, publisher URL, DOI
                      and citation for public QEC datasets, hard refusal when
                      your copy does not match. nothing redistributed.
- rank_layouts        CalibrationMapper.rank_layouts() and score_layout(),
                      public API. max_overlap drops the near-duplicate that is
                      the same hardware with the labels permuted.
- calibration_freshness on every selection receipt, with tolerance_basis
                      stating the 30 minute tolerance is a builtin default,
                      not a measurement of your device.
- qbc doctor          names the qiskit 1.x + qiskit-ibm-runtime >= 0.40
                      pairing that raises TranspilerError on a plain
                      transpile, prints the workaround.
- github action       action.yml runs the free checks in CI.
                      docs/github-action.md.
- citation            CITATION.cff, paper.md, paper.bib. DOI not minted yet.
- docs                receipts, chemistry, corpora, github-action pages, new
                      commands in the CLI reference.
```


## 0.11.0 - 2026-07-27

Minor rather than patch: two CLI options and a new public field on `BackendValue`.

```
CHANGED
- qbc when            neutral cross-vendor advisor. ranks backends by
                      predicted fidelity per dollar across every configured
                      vendor rather than one, which is the comparison a
                      vendor SDK cannot give you. two new options:
                      --backend/-b, repeatable, names the set to rank, and
                      --json for a machine-readable advice receipt
                      (qb.cross_vendor_advice.v1), unsigned and
                      community-tier like the other --json receipts.
- BackendValue        new validation field, shown as a Data column, so every
                      ranked row states the provenance of its own numbers:
                      validated where the fidelity model is validated against
                      that backend's real hardware, UNVALIDATED where an
                      adapter exists but has not been proven on the device,
                      fixture-only, or no-adapter. unvalidated rows also
                      carry a note saying so in words. only IBM is validated
                      today, so a cross-vendor table is mostly UNVALIDATED,
                      and it says so rather than letting a model estimate sit
                      next to a measurement as though they were the same kind
                      of thing. the labels derive from the calibration
                      registry's live status, so they cannot drift from what
                      qbc backends reports. tests pin both the derivation and
                      the fact that no vendor is currently labelled validated
                      except IBM.
```

## 0.10.0 - 2026-07-27

First release published to PyPI since 0.7.0, so installing it brings everything in 0.8.0
(ObservableGate, the QEC decoder-input correctness audit, and its `qbc dem-audit` /
`qbc dem-canonicalize` commands) and 0.9.0 (selection receipts for calibration-aware layout) as
well as the changes below. See those entries further down for detail.

Minor rather than patch: this adds public API and changes one returned value. `BudgetOptimizer`
now recommends `budget_optimal` / `depth_optimal` where it previously returned `speed_optimal` /
`cost_optimal`. Anything reading that string sees different values; anything passing it to
`QBCompiler`, which is what it is for, now works where it previously raised.

```
FIXED
- entry points        check_viability() and QBCompiler.compile() accept the
                      same circuit. previously check_viability took a Qiskit
                      QuantumCircuit and compile took a
                      qb_compiler.QBCircuit, so running both on one object,
                      which is what the tutorials describe, raised either an
                      AttributeError thrown from inside Qiskit's transpiler
                      or an InvalidCircuitError naming a type the caller
                      never chose. both now accept a Qiskit circuit, the
                      public QBCircuit, or the IR circuit, via the new
                      any_to_qiskit and any_to_compiler_circuit converters.
                      unconvertible input still raises, with a message naming
                      what is accepted.
- BudgetOptimizer     its recommendations can be passed to the compiler.
                      previously it returned strategy names from the
                      qb_compiler.strategies registry (speed_optimal,
                      cost_optimal) that QBCompiler does not accept, so the
                      documented "find the cheapest backend, then compile
                      with the recommended strategy" flow raised ValueError
                      for any backend under $0.10 per shot, which is every
                      IBM device. recommendations now use the compiler's own
                      vocabulary, mapping the three cost tiers onto its three
                      optimization levels.
- ML layout predictor used only on the hardware it was trained on. only
                      ibm_heron weights ship, but the predictor was loaded
                      for every backend. because it narrows the candidate
                      qubit set before scoring, patterns learned on a fixed
                      coupling superconducting device were filtering layouts
                      on all to all trapped ion hardware. everything else now
                      uses the full calibration-driven search.
- layout selection    _get_two_qubit_error rescanned the whole gate map on
                      every call, tens of thousands of times per compile. it
                      is now indexed once by qubit pair, which cut layout
                      selection from roughly 61s to 3s for a small circuit on
                      an all to all backend. results are unchanged: the
                      indexed lookup was checked against the previous one
                      across every qubit pair on four backends.
- qbc backends        works on installs without the optional vendor SDKs. the
                      availability probes for Quantinuum and Azure passed a
                      dotted module path to importlib.util.find_spec, which
                      imports the parent package in order to locate a
                      submodule and therefore raises ModuleNotFoundError when
                      that parent is absent, rather than returning None.
                      every vendor SDK is an optional extra, so a default
                      pip install qb-compiler has none of them, and qbc
                      backends exited 1 with a traceback instead of listing
                      the backends. the probes now report an absent SDK as
                      unavailable, so the backend is still listed with
                      deps: no. covered by a test that simulates each vendor
                      SDK being missing, so the result no longer depends on
                      what the test machine has installed.

CHANGED
- package root        exports 64 names instead of 50. since 0.7.0 the
                      headline additions (ObservableGate, selection receipts,
                      the multi-vendor calibration registry) were reachable
                      only by full module path, so dir(qb_compiler) showed
                      none of them. added: audit_dem, audit_matrices,
                      canonicalize_dem, preflight_dem_gate,
                      ObservableAuditResult, CalibrationMapper,
                      CalibrationMapperConfig, selection_receipt,
                      calibration_fingerprint, get_calibration_provider,
                      all_backend_statuses, get_backend_status,
                      any_to_qiskit and any_to_compiler_circuit.
- import cost         names resolve on first use (PEP 562) rather than at
                      import, so the larger surface costs nothing. import
                      qb_compiler went from about 490ms to about 70ms and no
                      longer pulls numpy or package metadata. a type-checking
                      block re-states every export so editors and mypy still
                      resolve them, and the test suite asserts that the
                      export map and __all__ agree and that every entry
                      resolves.
- FidelityCircuit     qb_compiler.noise.QBCircuit renamed. three unrelated
                      classes shared the name QBCircuit, and this one was
                      exported, so from qb_compiler.noise import QBCircuit
                      returned a fidelity descriptor with none of the public
                      circuit's methods.
```

## 0.9.0 - 2026-07-27

```
ADDED
- selection_receipt   qb_compiler.passes.mapping.selection_receipt. a
                      signed-passport-ready record of the layout
                      CalibrationMapper chose and why: its layout, score, and
                      per-signal breakdown (gate error, coherence, readout,
                      T1 asymmetry, temporal correlation) plus a stable
                      calibration fingerprint. derived purely from the
                      mapper's PassResult, so it never re-implements the
                      layout objective. unsigned by default; sign=True uses
                      the QubitBoost SDK's Ed25519 signer, soft-imported only
                      if present, and degrades to an unsigned receipt with a
                      pointer otherwise. Apache-2.0, zero paid-SDK
                      dependency.
- exports             CalibrationMapper and CalibrationMapperConfig from
                      qb_compiler.passes.mapping. previously importable only
                      by full module path.

FIXED
- native gate test    test_qb_transpile_reduces_to_native_gates asserted cx
                      as the IBM Fez native 2Q gate. Fez is a Heron r2 device
                      whose native 2Q gate is cz. the test now derives the
                      allowed set from IBM_HERON_BASIS, the codebase's own
                      basis definition, so it cannot drift from the
                      transpiler target.
```

## 0.8.0 - 2026-06-27

The correctness-preflight release. qb-compiler gains ObservableGate, a QEC decoder-input
correctness audit. A stim DEM error mechanism carries detectors, logical-observable masks, and a
probability. A decoder-input canonicalization that merges by detector signature alone can collapse
detector-identical but logical-distinct mechanisms and erase the logical frame, which can inflate
the logical error rate. ObservableGate detects that condition before decoding and offers an
observable-preserving canonical form.

```
ADDED
- observable_gate     audit_matrices (pure-numpy invariant, no stim),
                      audit_dem, canonicalize_dem, preflight_dem_gate, and
                      ObservableAuditResult, a PASS/WARN/FAIL receipt with
                      mixed-group count, mixed probability mass and
                      worst-mask ratio.
- qbc dem-audit       model.dem, CI-safe exit codes: 0 PASS, 1 WARN with
                      --strict, 2 FAIL. --json emits a machine-readable
                      community-tier receipt.
- qbc dem-canonicalize
                      model.dem -o safe.dem, the observable-preserving
                      canonical form. merges only exact
                      (detectors, observables) duplicates, keeps
                      detector-identical / logical-distinct mechanisms
                      separate.
- qec_preflight       attaches an observable_audit receipt to its result and
                      prints its status.
- tests               the pure-numpy core runs in base CI without the
                      [ising] extra.

SCOPE (honest)
- standard paths      surface/repetition and the full bivariate-bicycle /
                      Gross family ([[72,12,6]] ... [[144,12,12]] ...
                      [[288,12,18]], X and Z basis) audit PASS. decomposed
                      DEMs are XOR-benign.
- the hazard          real and measured: color_code:memory_xyz d3, ~60%
                      relative LER inflation under naive detector-only
                      merging, on graphlike DEMs with genuine
                      detector-identical / logical-distinct mechanisms.
- licensing           ObservableGate and its receipts are free and open
                      source. signed receipts, batch reports, shared
                      dashboards and CI policy bundles are QubitBoost Pro.
                      see docs/open-core.md.

HARDENED
- ML checkpoint       unsafe-deserialization path in the optional ML-decoder
                      loader fixed with torch.load(weights_only=True), plus a
                      CI guard against future regressions.
- qbc compile         a resource-exhaustion case is now bounded.
- CLI                 clean errors and exit codes for malformed or
                      wrong-type inputs and bad output paths.
- gates               709 tests, unit plus security plus adversarial. ruff
                      and mypy clean.
```

## 0.7.0 - 2026-06-12

Folds the planned 0.5.3, 0.6 and 0.7 roadmap buckets into one release, plus the QEC experiment
preflight pulled forward from 0.8.

The trust-layer release: every number qb-compiler prints now carries its uncertainty, its
provenance, or both, and the package gets a memory.

```
ADDED
- error budget        viability results break fidelity loss down by source,
                      two-qubit gates against readout, with pct-of-loss
                      rendering.
- fidelity band       estimates print with a typical-abs-error band derived
                      from the committed Fez hardware validation pairs (n=6,
                      GHZ family, model runs optimistic; provenance comment
                      in code).
- calibration age     on preflight, with a suggestion when the snapshot is
                      stale.
- verify mode         build_mirror / run_mirror / verify_viability compare
                      the prediction against a mirror-circuit measurement, a
                      success proxy, honestly disclaimed. qbc verify runs it
                      on aer. records grow a local predicted-vs-actual
                      accuracy log; ideal-sim runs are tagged and excluded
                      from accuracy_summary by default.
- compilation receipts
                      a passport per compile: predicted fidelity plus band,
                      error budget, calibration age, versions, seed, layout,
                      in a local jsonl store. regression_check flags a drop
                      only beyond the combined error bands and never blocks
                      anything.
- best-of-n           qb_transpile n_seeds sweeps the transpiler and returns
                      the candidate with the best calibration-aware fidelity
                      score. return_candidates exposes the per-seed evidence;
                      the fallback path returns a tagged single candidate.
- rank_value          fidelity-per-dollar ranking, exposed as qbc when, with
                      a naive calibration trend per backend. explicitly no
                      forecasting.
- shot budgets        shots_for_expectation, shots_for_rate.
- backend discovery   from the user's own runtime service, plus pub-aware
                      preflight.
- qec preflight       memory-experiment preflight on stim + pymatching:
                      projected ler band, detector fraction,
                      shots-for-confidence.
- ising telemetry     IsingDecodeResult, bounded opt-in harness telemetry,
                      provenance hashes. closes the v0.5.0 design doc.
- bundled snapshots   a small calibration snapshot set ships in the wheel, so
                      qbc when and fixture-based preflight work from a pip
                      install. point QBC_CALIBRATION_DIR at your own
                      snapshots for fresh data.
- py.typed            the typing claim in pyproject is now true.

FIXED
- heron basis gates   corrected to cz in BACKEND_CONFIGS (fez, torino,
                      marrakesh). the 0.5.2 entry below says heron r2 uses
                      ecr. that was wrong; heron's native two-qubit gate is
                      cz.
- gate registry       stale marketing claims removed. the old optgate
                      multiplier was retracted in april and should not have
                      still been shipping. safetygate qualifier neutralised.
- ionq prices         aria 0.03, forte 0.08 usd. 0.30 is braket's per-task
                      fee, now modelled separately. the price table gains a
                      last-reviewed stamp and a staleness warning.
- local store         corrupt lines no longer brick reads. they are skipped
                      with a logged count.

CHANGED
- ci                  tests qiskit 1.4 and 2.3.
- ising extra         requires pymatching >= 2.3 (enable_correlations).
```



## [0.5.2] - 2026-04-30

```
CHANGED
- qb_transpile        accepts a Qiskit backend object directly, not only a
                      string name. qb_transpile(circuit, backend=...) used to
                      require a string in BACKEND_CONFIGS; it now also takes
                      a Qiskit BackendV1 / BackendV2 instance, in which case
                      basis_gates and coupling_map are pulled from
                      .configuration() / .target / .basis_gates at runtime.
                      this closes the class of bug where the registry's
                      hardcoded basis_gates drifts from the real device:
                      BACKEND_CONFIGS["ibm_fez"] shipped cx as the native 2q
                      gate, IBM Heron r2 (Fez, Marrakesh) uses ecr, so the
                      routed circuit emitted cx and IBM Runtime rejected it.
                      the same trap waits for any future Heron-family
                      gate-set update. querying the live backend avoids it
                      permanently.
- backward compat     the string path is unchanged. callers passing
                      backend="ibm_fez" keep the legacy registry behaviour.
                      the registry entries for ibm_fez, ibm_marrakesh and
                      ibm_torino were left as-is on purpose; the object path
                      makes them advisory rather than load-bearing for
                      transpilation.
- tests               3 new integration tests: the object path, the legacy
                      string path unchanged, and the error case where a
                      backend object exposes none of the inspected
                      attributes.
```

```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
backend = service.backend("ibm_fez")

# v0.5.1 and earlier (still works, still emits stale cx for Heron):
compiled = qb_transpile(circuit, backend="ibm_fez", ...)

# v0.5.2 (recommended): pulls live ecr basis from the backend itself
compiled = qb_transpile(circuit, backend=backend, ...)
```

## [0.5.1] - 2026-04-27

Connectivity-aware chain selection. Closes the v0.5.0 UCCSD/HEA regression.

v0.5.0 had three latent bugs that combined to make the live-calibration path underperform the v0.4
static-fixture path on dense-1q workloads (UCCSD, hardware-efficient ansatzes). v0.5.1 fixes all
three. Headline benchmark result on IBM Fez (n=30 random seeds, paired Wilcoxon signed-rank,
Bonferroni-adjusted, classical noise-aware fidelity scoring against a fresh live calibration
snapshot):

| Comparison                       | v0.5.0 (broken) | v0.5.1 (fixed)        |
|----------------------------------|-----------------|------------------------|
| v0.5 vs v0.4 fixture path        | 2W / 3L / 3T    | **0W / 0L / 8T**       |
| v0.5 vs Qiskit `optimization_level=3` | 3W / 5L / 0T  | **5W / 2L / 1T**       |
| UCCSD-H4 vs Qiskit (median delta) | -3.9 % (loss)  | **+12.3 %** (p<0.0001) |
| QAOA-8 ring p=2 vs Qiskit         | +13.9 %        | +8.8 % (p<0.0001)      |
| HEA-8 d=4 vs Qiskit               | -5.5 %         | +0.3 %                 |

Full circuit suite + raw data:
`QubitBoost-internal/experiments/qb_compiler_v0_5_benchmarks/`.

### Hardware companion (re-run, n=16 supersedes initial n=4)

The initial release (2026-04-27 17:24 UTC) reported a hardware companion result of "v0.5.1 lands
6.4 mHa closer to E_RHF than Qiskit opt=3" on the H2O 4e4o HF state on IBM Fez at n=4 reps per
arm. **A 90-minute follow-up at n=16 reps reversed the verdict**: same circuit, same layouts, but
Qiskit opt=3 came in at |delta E_RHF| = 7.21 mHa against v0.5.1's 12.33 mHa. The 5.12 mHa gap at
n=16 is below 1 sigma of the combined SEMs (~12.5 mHa), so the honest verdict is **statistically
equivalent on this single circuit at p=0.05**. The initial n=4 win was a tail event; the n>=5,
ideally n=8, hardware-claim rule applies and n=4 was below threshold for any defensible
single-circuit verdict.

The classical n=30-seed benchmark above is unaffected: different sample-size regime, paired
comparison rather than absolute, drift-isolated by scoring against a single fresh snapshot for
both arms. **The +12.3% UCCSD-H4 estimated-fidelity result vs Qiskit stands.**

A larger statistical-power hardware run (n>=32, multi-window) is scheduled for v0.5.2 to make a
hardware-validated absolute claim defensible.

```
FIXED
- chain selection     the load-bearing fix. QBCalibrationLayout previously
                      picked the N best-scoring physical qubits regardless of
                      whether they formed a connected subgraph on the device
                      coupling map. on dense-2q circuits that picked qubits
                      scattered across the chip, forcing the downstream
                      router to insert many SWAPs and crashing post-routing
                      fidelity. v0.5.1 adds _vf2_calibration_aware(), which
                      uses rustworkx.vf2_mapping to enumerate subgraph
                      isomorphisms of the circuit's 2q interaction graph onto
                      the device coupling map and scores each candidate by
                      sum(per-qubit scores) + sum(per-edge gate errors x
                      interaction count). falls back to the v0.5.0
                      topology-blind path if VF2 finds no mapping, for
                      instance when the circuit has no 2q interactions.
- gate-error pooling  v0.5.0's _build_qubit_scores pooled single-qubit and
                      two-qubit gate errors into a single arithmetic mean.
                      with the v0.5 live calibration's full coverage the
                      small 1q errors (~1e-4) diluted the larger 2q errors
                      (~5e-3) by ~5x, distorting score ordering. v0.5.1
                      tracks gate_error_1q and gate_error_2q on separate
                      score keys with weights w_2q=0.40, w_1q=0.00. the 1q
                      signal is captured but weighted at zero, because the
                      connectivity-aware scorer above does not usefully
                      consume it without per-edge 1q modelling; that is
                      scheduled for v0.6.
- _provider_to_dict   LiveCalibrationProvider unwrapping. called with a
                      LiveCalibrationProvider, the materializer's
                      getattr(provider, "_props") returned None, because the
                      BackendProperties lives at provider._snapshot._props,
                      one level deeper. the materialised calibration dict was
                      missing coupling_map, n_qubits and basis_gates, and
                      without coupling_map the new VF2 path was a silent
                      no-op. fixed by drilling into _snapshot._props when
                      _props is absent at the top level.

DEPRECATED
- nothing             v0.5.1 is wire-compatible with v0.5.0.
```

### Notes on prior v0.5.0 release notes

The v0.5.0 entry below carries a workload-dependent regression disclosure that is now obsolete.
The regression is closed in v0.5.1. Treat v0.5.1 as the authoritative version; v0.5.0 is retained
below for historical context.

## [0.5.0] - 2026-04-26

Live calibration end-to-end against real backends. The `LiveCalibrationProvider` path stops being
a stub: it delegates to a working `qubitboost_sdk.calibration.CalibrationHub`.

### Architectural scope of v0.5: the "in-process tier"

```
SHIPS
- CalibrationHub      a Python class that runs inside whatever process
                      imports qubitboost_sdk.calibration: qb-compiler, a
                      notebook, a script. on-demand fetch from IBM Quantum
                      via qiskit-ibm-runtime. per-user disk JSON cache at
                      ~/.cache/qubitboost/calibration/ with 30-min TTL. each
                      consuming process holds its own hub instance and its
                      own view of the disk cache.

DOES NOT SHIP
- background daemon   no polling daemon.
- shared cache        no Redis or other shared cache, no cross-process
                      synchronisation.
- HTTP surface        no FastAPI endpoint, no Hardware Observatory page
                      consumer.
- rate-limit budget   not shared across consumers. today two parallel
                      processes can both fetch the same backend within
                      seconds of each other. for the demo that is fine.
```

The PM2-managed daemon, Redis cache, FastAPI endpoint and Observatory page are scoped at
QubitBoost-internal `sales/CALIBRATION_HUB_DESIGN.md` and are scheduled for a follow-on release.
They are not implied or required by v0.5.

### Authentication

`LiveCalibrationProvider` (and `CalibrationHub`) authenticate to IBM via a saved-credential
profile resolved by `QiskitRuntimeService(name=...)`. The default profile name is
`"qubitboost_cloud"`. External users must save their own IBM credentials before invoking the live
provider::

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        name="my_account", channel="ibm_quantum", token="...",
    )

then pass `account="my_account"` to either constructor. No tokens are embedded in the package. No
environment variable is read in v0.5. Profile resolution is identical to plain
`QiskitRuntimeService` use.

### Freshness contract (locked)

```
- cache age 0-30 min  default cache_ttl_minutes=30. get_latest serves the
                      cached snapshot. no IBM contact.
- cache age > 30 min  get_latest attempts a fresh fetch. on success the new
                      snapshot replaces the cache and is returned. on failure
                      (IBM unreachable, 5xx, network error) the previous
                      stale cached snapshot is returned with a UserWarning of
                      the form "CalibrationHub: fresh fetch for {backend}
                      failed (...); serving stale cache (age N min)".
- fetch()             always bypasses cache, always performs a fresh vendor
                      call. on failure it raises, with no stale fallback. use
                      it for explicit pre-experiment freshness guarantees.
- stale-fallback age  no upper bound inside the hub. callers needing a hard
                      floor, a demo harness that refuses to launch on
                      >24h-old calibration for instance, must enforce it
                      themselves by reading provider.timestamp and comparing
                      to datetime.now(timezone.utc).
```

The 30-min TTL is a small fraction of IBM Heron-class devices' ~12h calibration cadence on
Fez/Torino/Marrakesh/Kingston: long enough to amortise IBM API traffic across multiple
compilations of the same circuit, short enough to catch event-driven recalibrations within ~30 min
of IBM publishing them.

### Coverage improvement vs v0.4 fixture path

`LiveCalibrationProvider` returns snapshots with the full property surface IBM exposes. For
ibm_fez specifically:

```
- v0.4 fixture        hand-fetched 2026-03-14. 156 qubit_properties (T1, T2,
                      readout_error_0to1, readout_error_1to0,
                      frequency=None) plus 352 gate_properties, 2-qubit ECR
                      errors per coupling only.
- v0.5 live fetch     the same 156 qubit_properties plus IBM-specific fields
                      (prob_meas0_prep1, prob_meas1_prep0, readout_length)
                      plus 1796 gate_properties, ~5x, adding per-basis-gate
                      single-qubit error rates for id, sx, rz, x.
```

The added 1444 entries are per-basis-gate single-qubit error rates that v0.4's fixture-based path
did not capture. The practical effect on chain selection is small, single-qubit errors are
typically much smaller than 2-qubit, but the data is now complete.

### Field-format reconciliation

```
- disk cache JSON     dual-format. legacy readout_error_0to1 /
                      readout_error_1to0 aliases and modern
                      prob_meas1_prep0 / prob_meas0_prep1 fields are both
                      present per qubit. gate parameters are written both
                      flat (gate_error: ...) and nested under
                      parameters: {...}.
- _provider_to_dict() the snapshot it materialises for QBCalibrationLayout
                      uses the legacy field-name convention internally,
                      because qb-compiler's BackendProperties dataclass keeps
                      only the legacy fields after parsing.
                      QBCalibrationLayout consumes via the nested
                      parameters.gate_error path, which _provider_to_dict()
                      emits. verified working.
```

### Benchmarks (re-run 2026-04-26)

v0.5's layout-selection algorithm was benchmarked against the v0.4 static-fixture path and against
Qiskit `optimization_level=3` on a fixed circuit set on IBM Fez calibration data. n=30 random
seeds per circuit, paired Wilcoxon signed-rank with Bonferroni correction across 3 comparisons per
circuit, classical noise-aware fidelity scoring, no QPU execution. The QPU companion is deferred
to v0.5.1.

Circuit set: GHZ-{4,8,12}, QAOA-8 ring p={1,2}, UCCSD-H4 4e4o, HEA-{8,12} d=4.

**Headline finding: v0.5 is workload-dependent, not uniformly better.**

| Workload class | v0.5 vs v0.4 fixture | v0.5 vs Qiskit opt=3 |
|---|---|---|
| Ring QAOA (sparse 1q, dense 2q) | **+4 to +4 % median fid (p<0.05)** | **+5 to +14 % median fid (p<0.05)** |
| GHZ (mostly 2q) | tied | mixed (1 win at 12q, 2 losses at 4q/8q) |
| UCCSD / HEA (dense 1q + 2q) | **-5 to -7 % median fid (p<0.0001)** | **-3 to -6 % median fid (p<0.0001)** |

**Tally for v0.5 vs Qiskit opt=3:** 3 wins, 5 losses, 0 ties (8 circuits).
**Tally for v0.5 vs v0.4:** 2 wins, 3 losses, 3 ties (8 circuits).

**Interpretation:** the added single-qubit gate-error data in v0.5 appears to distort
qb-compiler's chain-scoring on dense-single-qubit workloads (UCCSD, HEA), and that distortion
costs more than the QAOA-side gains on most circuit classes. An algorithm-level retune of the
chain-scoring weights between single-qubit and 2-qubit error contributions is scheduled for
v0.5.1; the goal is uniformly equal-or-better-than-v0.4 across all circuit classes.

**Practical guidance for v0.5 callers:**
- QAOA-style workloads: use `LiveCalibrationProvider`, the v0.5 default.
- Dense-single-qubit workloads (UCCSD, HEA, generic VQE ansatz): use `calibration_path=...` with
  the v0.4-style fixture format until v0.5.1 ships the algorithm fix. The live data path is
  correct; the chain-scoring algorithm using it is not yet tuned for this workload class.

Full results: `QubitBoost-internal experiments/qb_compiler_v0_5_benchmarks/results/`.

### README claim correction (line 222)

The wording "calibration-aware layout selection that matches or exceeds Qiskit's default on
hardware-validated benchmarks" is workload-dependent under v0.5: 5 of 8 circuits regress vs Qiskit
opt=3 in the benchmark above. Recommended replacement wording on the README:

> "calibration-aware layout selection that matches or exceeds Qiskit's
> default on QAOA-style hardware workloads. Performance is workload-
> dependent; see CHANGELOG v0.5.0 for the full benchmark table including
> circuit classes where qb-compiler currently underperforms (UCCSD-style
> chemistry ansatzes and dense hardware-efficient ansatzes)."

```
ADDED
- calibration_provider qb_transpile(..., calibration_provider=...) accepts any
                      CalibrationProvider instance directly, including the
                      live one. previously only calibration_path and
                      calibration_data were supported.
- _provider_to_dict   materialises a provider's snapshot into the calibration
                      dict QBCalibrationLayout consumes, so the live provider
                      can drive the layout pass without bespoke wiring.
- account=            LiveCalibrationProvider(..., account="...") so external
                      users can pass their saved-credential profile name.
                      defaults to "qubitboost_cloud" for backward
                      compatibility.

CHANGED
- refresh()           LiveCalibrationProvider.refresh() calls hub.fetch()
                      directly to bypass the cache TTL. previously it called
                      hub.get_latest(), which silently served cached data
                      inside the TTL window, contradicting the docstring's
                      "force re-fetch" promise.
- cache_ttl_minutes   propagated down to the hub LiveCalibrationProvider
                      constructs, so both layers agree on freshness. v0.5 dev
                      versions briefly defaulted LiveCalibrationProvider to
                      30 min while the underlying hub defaulted to 60. fixed.
- ImportError         LiveCalibrationProvider no longer raises it when
                      qubitboost-sdk>=2.6 is installed. earlier versions
                      pointed at qubitboost_sdk.calibration.CalibrationHub,
                      which did not exist; v2.6 of the SDK ships that module.

HONEST DISCLOSURE ON PRIOR VERSIONS
- v0.1-v0.4           the "calibration-aware" claims operated on real IBM
                      backend properties (audit at QubitBoost-internal
                      sales/QB_COMPILER_FIXTURE_PROVENANCE.md), but the
                      shipped fixture files carried 2-qubit gate errors only
                      and dropped per-basis-gate single-qubit error data.
                      from v0.5 the live fetch via CalibrationHub provides
                      the full property surface. for chain selection the
                      practical impact of the prior partial coverage was
                      small, single-qubit errors are typically much smaller
                      than 2-qubit errors, but v0.5 closes the gap.
- README claim        "Calibration data can be loaded from local JSON files
                      or fetched from vendor APIs" was incorrect on v0.4 and
                      earlier: the vendor-API path raised ImportError. it is
                      correct on v0.5 with pip install
                      qb-compiler[qubitboost].
```

## [0.4.0b1] - 2026-04-22

Beta release. Stim-validated only, no hardware runs yet. API may shift.

```
ADDED
- qb_compiler.ising   first Qiskit-side integration for NVIDIA's
                      Ising-Decoder-SurfaceCode-1 model family, released
                      2026-04-14. converts rotated-surface-code memory
                      experiments (Qiskit or stim) into the 4-channel
                      (B, 4, T, D, D) float32 tensor the pretrained decoder
                      consumes. public API: SurfaceCodePatchSpec,
                      build_ising_tensor, PyMatchingDecoder (MWPM baseline),
                      IsingDecoderWrapper (pre-decoder plus residual-MWPM
                      chain; users bring their own gated-HF weights and
                      NVIDIA's Apache-2.0 model definition, qb-compiler does
                      not vendor NVIDIA code or weights),
                      evaluate_logical_error_rate harness. install
                      qb-compiler[ising] for the PyMatching baseline,
                      qb-compiler[ising-nvidia] to add torch + safetensors
                      for the NVIDIA pre-decoder. see
                      src/qb_compiler/ising/README.md.
- benchmark harness   benchmarks/ising/run_pymatching_sweep.py sweeps
                      (distance, rounds, p_error, basis) to establish the
                      baseline any pre-decoder must beat.
- extras              ising, ising-nvidia.
```

## [0.3.0] - 2026-04-16

```
ADDED
- qiskit 2.x          compatibility. the qiskit dependency widens to
                      >=1.0,<3.0, and CI runs the suite against both Qiskit
                      1.4 and Qiskit 2.0 in matrix.
- QBCalibrationLayoutPlugin
                      a proper qiskit.transpiler.layout stage plugin. invoke
                      via generate_preset_pass_manager(
                      layout_method="qb_calibration") with the
                      QB_CALIBRATION_PATH env var set. discoverable through
                      Qiskit's entry-point system.

CHANGED
- qb_transpile()      injects QBCalibrationLayout into the pass manager's
                      pre_layout stage instead of layout. on Qiskit 2.x the
                      previous approach triggered an ApplyLayout KeyError and
                      silently fell back to stock qiskit.transpile, bypassing
                      calibration-aware layout. the custom pipeline is now
                      the primary code path on both Qiskit versions.
- QBTranspilerPlugin  entry-point group corrected from the non-existent
                      qiskit.transpiler.stage to qiskit.transpiler.layout,
                      now pointing at QBCalibrationLayoutPlugin. the plugin
                      was previously undiscoverable via Qiskit's loader.

DEPRECATED
- get_pass_manager()  QBTranspilerPlugin.get_pass_manager(
                      calibration_data=...) emits a DeprecationWarning and
                      will be removed in 0.4.0. migrate to
                      generate_preset_pass_manager(
                      layout_method="qb_calibration") with
                      QB_CALIBRATION_PATH set, or call qb_transpile()
                      directly.

FIXED
- ci.yml              triggers on master as well as main. the repo's default
                      branch is master, so the workflow had been dormant.
- CI install          the phantom [qiskit] optional-dependency extra removed
                      from the install commands. it did not exist in
                      pyproject.toml and was silently ignored.
```

## [0.1.0] - 2026-03-13

```
ADDED
- core IR             QBCircuit, QBDag, QBGate, QBMeasure, QBBarrier.
- converters          Qiskit and OpenQASM 2.0.
- CalibrationMapper   VF2-based calibration-weighted qubit placement.
- NoiseAwareRouter    Dijkstra shortest-error-path SWAP routing.
- NoiseAwareScheduler ALAP scheduling with T1/T2 urgency scoring.
- GateDecomposition   native basis decomposition. IBM ECR/RZ/SX/X, Rigetti
                      CZ/RX/RZ, IonQ MS/GPI/GPI2, IQM CZ/PRX.
- ErrorBudgetEstimator
                      pre-execution fidelity prediction.
- T1 asymmetry        readout-scaled penalty for qubits with high |1> decay.
- temporal correlation
                      Pearson correlation across calibration snapshots.
- calibration         StaticCalibrationProvider, CachedCalibrationProvider,
                      BackendProperties.
- noise modelling     EmpiricalNoiseModel, FidelityEstimator.
- backends            IBM Heron, Rigetti Ankaa, IonQ Aria/Forte, IQM
                      Garnet/Emerald.
- cost estimation     with vendor pricing.
- qiskit plugin       QBCalibrationLayout, qb_transpile(), QBPassManager.
- CLI                 qbc compile, qbc info, qbc calibration show.
- optimisation passes gate cancellation and commutation analysis.
- analysis passes     depth and gate count.
- ML phase 2          XGBoost layout predictor. AUC=0.94, 454KB, +5.4%
                      fidelity on GHZ-8.
- ML phase 3          GNN layout predictor. dual-graph GCN, 42KB, +6.5%
                      fidelity on QAOA-8.
- ML phase 4          RL SWAP router. PPO actor-critic, 190KB,
                      calibration-aware routing.
- ML infrastructure   data generator, feature extraction, model training
                      scripts.
- tests               461, covering all passes, IR, calibration, backends and
                      the ML pipeline.
- CI/CD               GitHub Actions for lint, typecheck and the test matrix,
                      Python 3.10-3.12.
- examples            10 scripts demonstrating key features.
- benchmarks          comparing all ML phases.
```
