# Build outline: `22_willow_complementary_gap_validation.ipynb`

Goal: a FULLY OPEN, self-reproducing notebook (public Willow data + published method) that regenerates
every number on the /research page. No proprietary SDK code, complementary-gap/Gidney gap-decoding is
published, so implement it clean-room here. Openness is deliberate: this artifact has no IP to protect
and maximum credibility to gain. Deps: stim, pymatching, numpy, matplotlib, requests (all OSS).

Cell-by-cell:

### 1. [markdown] Title + honest-scope banner
Title; one-paragraph abstract; the "what this is / is not" box (validation of a PUBLISHED method on
real hardware, not a new decoder; cite Gidney gap-decoding + Google Willow Nature 2024).

### 2. [markdown] Background
Postselection, the complementary gap, why it ranks shots by trustworthiness. Cite the gap-decoding
paper and the Willow dataset DOI.

### 3. [code] Environment
Imports + printed versions (stim, pymatching, numpy). Fixed RNG seed for any sampling. Assert versions
so a reader's run is byte-comparable.

### 4. [code] Fetch public Willow data
Download the public Google Willow surface-code memory dataset (Zenodo/GitHub DOI) into ./willow_data/
if absent. Parse per (distance d in {3,5,7}, rounds): the DEM (or stim circuit -> DEM), real
detection_events (.b8), real obs_flips_actual (.b8). Print shot counts per config. (Cache so re-runs
are instant.)

### 5. [markdown] Method
Explain complementary-gap decoding in 4 lines: decode normally -> committed class + weight w0; force
opposite logical class -> weight w1; gap = w1 - w0; large gap = confident. Note it's correct-by-
construction (committed decode == plain MWPM).

### 6. [code] Decoder from DEM (pymatching)
`pymatching.Matching.from_detector_error_model(dem)`. Baseline: decode all shots, compare predicted
observable to real obs_flips_actual -> base LER per config. This reproduces the standard MWPM number
(sanity check vs Google's published LER).

### 7. [code] Complementary gap per shot
For each shot: decode -> prediction + solution weight; re-decode with the logical observable pinned to
the opposite value (add observable as a hard constraint / augmented matching) -> weight; gap = |w1-w0|.
Return per-shot (predicted_correct: bool, gap: float). Vectorize with decode_batch where possible.

### 8. [code] Postselection sweep
Sort shots by gap ascending; for discard fraction f in [0..0.5], drop the lowest-gap f, recompute LER
on survivors. Produce LER-vs-discard curve per d. Record LER cut at f=0.20. Compute ranking AUC of gap
vs (predicted_correct) per config.

### 9. [code] Baselines (strength-matched)
Same sweep using (a) matching-weight-only confidence (w0), (b) RANDOM gap. Shows complementary-gap ~2x
matching-weight, ~50x random. This is the honesty control, the gain must beat the naive baseline.

### 10. [code] Results table + figures
Table: d | base LER | LER cut @20% | AUC (for gap / weight / random). Plots: (i) LER vs discard
fraction per d; (ii) AUC vs d; (iii) the low-LER operating point (d7 r30) with AUC 0.945 / +92%.
Save figures to ./figures/ for the /research page + PDF.

### 11. [markdown] Simulation-reproduces-on-hardware
Show the sim-predicted point (+87% / 0.95) beside the real-hardware point (+92% / 0.945). This is the
credibility crux: the effect is physics, not a sim artifact.

### 12. [markdown] Honest caveats + how to use
- Adopted published SOTA, not invented here (cite).
- "Grows with distance" is PRIMARILY base-LER-driven; modest pure-distance at matched LER, do not overclaim.
- Costs ~20% throughput; only where you can afford to discard shots.
- Excluded dead-logical long-flow runs (44-50% randomized LER) + near-trivial single-round.
- Practical use: adopt for memory / shot-budget protocols; in the QubitBoost stack it ships in SafetyGate
  with governance + a signed receipt recording the per-run payoff.

### Repro footer
Print the exact package versions + data DOI + a one-line "every figure above regenerated from public
data" statement. Link back to /research.

NOTES:
- Keep it clean-room open (no qec/commercial/sdk imports). If a reader wants the governed/receipted
  version, that's the paid SDK, the notebook proves the mechanism; the product is the governance wrapper.
- Slots into the existing notebook suite after 20_proof_not_promises / 21_observablegate.
