# qb-compiler 0.8.0 — announcement copy

Honest, bounded positioning. Do NOT say "qLDPC decoders are broken." Say: some DEM workflows can be
unsafe if detector-identical but observable-distinct mechanisms are merged; ObservableGate audits this.

---

## GitHub Release — short summary (the box at the top)

> **ObservableGate: QEC decoder-input correctness auditing.** A stim DEM error mechanism carries
> detectors *and* logical-observable masks; merging by detector signature alone can collapse
> detector-identical but observable-distinct mechanisms and erase the logical frame. New `qbc dem-audit`
> (CI-safe exit codes, `--json` receipt) and `qbc dem-canonicalize` (observable-preserving form). Plus a
> security + adversarial hardening pass (pickle-RCE fix, OOM-hang fix, 709 tests). Standard surface /
> BB-Gross codes audit PASS — bounded scope. Free and open source; team/enterprise receipt workflows
> coming under QubitBoost Pro.

---

## Qiskit ecosystem / Slack #ecosystem (technical, ~70 words)

> New in **qb-compiler 0.8.0**: **ObservableGate**, a QEC decoder-input correctness preflight. A stim
> Detector Error Model mechanism carries detectors *and* logical-observable masks — a DEM→matrix step
> that merges by detector signature alone can drop the logical mask and inflate the logical error rate.
> `qbc dem-audit model.dem` produces a CI-safe receipt (exit 0/1/2); `qbc dem-canonicalize` writes an
> observable-preserving form. Standard surface / bivariate-bicycle / Gross codes audit PASS.
> `pip install -U "qb-compiler[ising]"` · docs + repo: <link>

---

## LinkedIn (founder voice, ~150 words)

> We just shipped **qb-compiler 0.8.0**, and it adds something I think the QEC community will find useful:
> **ObservableGate** — a correctness check for quantum-error-correction decoder inputs.
>
> Here's the issue it catches. A Detector Error Model carries, per error mechanism, both the *detectors*
> it flips and the *logical observables* it flips. If a step that turns a DEM into decoder matrices merges
> mechanisms by detector pattern alone, two mechanisms that look identical to the detectors but differ in
> their logical effect get collapsed — and the logical information is silently lost. In the cases where it
> bites, that inflates the logical error rate.
>
> ObservableGate detects this *before* you decode, with a CI-safe `qbc dem-audit` and an
> observable-preserving `qbc dem-canonicalize`. It's bounded and honest: standard surface and
> bivariate-bicycle / Gross codes pass cleanly.
>
> This is the pattern we're building QubitBoost on — every check becomes a receipt you can trust.
>
> `pip install -U "qb-compiler[ising]"` · <link>

---

## QubitBoost site / news (1–2 sentences)

> **qb-compiler 0.8.0** adds **ObservableGate**, a QEC decoder-input correctness preflight that detects
> when a Detector Error Model canonicalization could erase logical-observable information before decoding,
> with a CI-safe audit (`qbc dem-audit`) and an observable-preserving canonical form
> (`qbc dem-canonicalize`).

---

## Reddit r/QuantumComputing (careful, technical, no hype, ~120 words)

> **[Tool] qb-compiler 0.8.0 adds a DEM observable-mask correctness check**
>
> Sharing a small open-source addition that might be useful if you work with stim Detector Error Models
> and decoders. A DEM error mechanism carries both detector targets (D#) and logical-observable targets
> (L#). If a DEM→matrix construction merges mechanisms by detector signature alone, detector-identical but
> logical-distinct mechanisms can collapse and lose the logical frame, which can inflate the logical error
> rate.
>
> `qbc dem-audit model.dem` flags this (PASS/WARN/FAIL, CI-safe exit codes); `qbc dem-canonicalize` writes
> an observable-preserving form. Scope is bounded and honest — standard surface and bivariate-bicycle /
> Gross codes audit PASS; the issue shows up on graphlike DEMs with genuine detector-identical /
> logical-distinct mechanisms. Feedback welcome. <link>

---

## Posting order (suggested)

1. Tag `v0.8.0`, publish PyPI, publish GitHub release (with the notes).
2. Qiskit Slack #ecosystem (discovery + credibility).
3. LinkedIn (narrative).
4. QubitBoost site/news.
5. Reddit (optional, only if comfortable fielding technical questions).

Watch: PyPI `without_mirrors` / `last_week`, GitHub stars/clones — together, not `with_mirrors` alone.
