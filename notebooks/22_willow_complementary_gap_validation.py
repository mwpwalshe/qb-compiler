# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---
# Convert to a notebook with:  jupytext --to notebook 22_willow_complementary_gap_validation.py

# %% [markdown]
# # Complementary-gap postselection on real Google Willow hardware
#
# **A reproducible validation of a published method, not a new decoder.**
#
# We run complementary-gap / *gap decoding* (Gidney, 2024) on Google's real Willow
# surface-code memory data (Nature, 2024; Zenodo record 13273331) and measure how much
# it cuts the logical error rate when we discard the least-confident shots.
#
# - **What this is:** an independent validation of a published, best-in-class postselection method on
#   real hardware, fully reproducible from public inputs.
# - **What this is not:** a new decoder. The method is published; we cite it and reimplement it
#   clean-room here (no proprietary code). Openness is deliberate: there is no IP to protect and
#   maximum credibility to gain.
#
# Requires only open packages: `stim`, `pymatching`, `numpy`, `matplotlib`.

# %%
import io, os, re, zipfile, tempfile, urllib.request
from collections import defaultdict
import numpy as np
import stim
import pymatching
print("stim", stim.__version__, "| pymatching", pymatching.__version__, "| numpy", np.__version__)
RNG_SEED = 0

# %% [markdown]
# ## 1. Get the public Willow dataset
# Google's 105-qubit Willow surface-code data (d=3/5/7), Zenodo record 13273331. Each experiment
# folder inside the zip carries the stim circuit, the *real* detection events, and the *real*
# recorded logical flips (`obs_flips_actual`). We decode the real syndromes and score against the
# real outcomes, never simulation.

# %%
ZENODO_URL = "https://zenodo.org/records/13273331/files/google_105Q_surface_code_d3_d5_d7.zip"
# Use a local copy if present, else download the public zip.
CANDIDATES = [
    os.path.expanduser("~/willow_data/google_105Q_surface_code_d3_d5_d7.zip"),
    "google_105Q_surface_code_d3_d5_d7.zip",
]
ZIP = next((p for p in CANDIDATES if os.path.exists(p)), None)
if ZIP is None:
    ZIP = "google_105Q_surface_code_d3_d5_d7.zip"
    print("downloading public Willow dataset (~1 GB) ...")
    urllib.request.urlretrieve(ZENODO_URL, ZIP)
print("using dataset:", ZIP)

# %% [markdown]
# ## 2. Load one experiment (circuit -> DEM, real detection events, real logical flips)

# %%
def _read_b8(raw, n_det, n_obs):
    with tempfile.NamedTemporaryFile(suffix=".b8", delete=False) as f:
        f.write(raw); tmp = f.name
    try:
        arr = stim.read_shot_data_file(path=tmp, format="b8",
                                       num_detectors=n_det, num_observables=n_obs)
    finally:
        os.unlink(tmp)
    return np.asarray(arr, dtype=np.uint8)

def list_experiments(zip_path):
    z = zipfile.ZipFile(zip_path)
    prefixes = sorted({n.rsplit("/detection_events.b8", 1)[0]
                       for n in z.namelist() if n.endswith("detection_events.b8")})
    by_d = defaultdict(list)
    for p in prefixes:
        m = re.match(r"d(\d+)", p.split("/")[1])
        if m:
            by_d[int(m.group(1))].append(p)
    return z, by_d

def load_from_zip(z, prefix, max_shots=None):
    circuit = stim.Circuit(z.read(f"{prefix}/circuit_noisy_si1000.stim").decode())
    n_det, n_obs = circuit.num_detectors, circuit.num_observables
    dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True)
    dets = _read_b8(z.read(f"{prefix}/detection_events.b8"), n_det, 0)
    obs = _read_b8(z.read(f"{prefix}/obs_flips_actual.b8"), 0, n_obs)[:, 0]
    if max_shots:
        dets, obs = dets[:max_shots], obs[:max_shots]
    return dets.astype(np.uint8), obs.astype(np.uint8), dem

z, by_d = list_experiments(ZIP)
print("distances found:", {d: len(v) for d, v in sorted(by_d.items())})

# %% [markdown]
# ## 3. The method: complementary gap (clean-room, from the published construction)
#
# For a shot, decode into each logical class and compare the minimum matching weights. We compute
# the opposite-class weight *exactly* by promoting the logical observable to an auxiliary detector
# node: an observable-flipping boundary edge becomes an edge to the aux node. Forcing the aux
# detector to 0 vs 1 gives the two class weights `w_even, w_odd`. The **complementary gap** is
# `|w_even - w_odd|`; a small gap = the two logical outcomes were nearly tied = an unreliable shot.
# The committed prediction (argmin) equals a plain MWPM decode, correct by construction.

# %%
def build_augmented(matching, obs_id=0):
    aux = matching.num_nodes
    aug = pymatching.Matching()
    for u, v, attr in matching.edges():
        w = float(attr.get("weight", 1.0))
        flips = obs_id in (attr.get("fault_ids") or set())
        if v is None:                          # boundary edge
            if flips:
                aug.add_edge(u, aux, weight=w, merge_strategy="smallest-weight")
            else:
                aug.add_boundary_edge(u, weight=w, merge_strategy="smallest-weight")
        else:
            if flips:                          # internal observable edge -> not exactly gap-decodable
                raise ValueError("internal observable edge; exact gap not supported for this graph")
            aug.add_edge(u, v, weight=w, merge_strategy="smallest-weight")
    return aug, aux

def decode_with_gap(dem, dets):
    matching = pymatching.Matching.from_detector_error_model(dem)
    aug, aux = build_augmented(matching, obs_id=0)
    n_aug = aug.num_detectors
    n = dets.shape[0]
    even = np.zeros((n, n_aug), np.uint8); even[:, :dets.shape[1]] = dets
    odd = even.copy(); odd[:, aux] = 1
    _, w_even = aug.decode_batch(even, return_weights=True)
    _, w_odd = aug.decode_batch(odd, return_weights=True)
    gaps = np.abs(w_even - w_odd)
    preds = (w_odd < w_even).astype(np.uint8)   # argmin class == plain MWPM prediction
    return preds, gaps

# %% [markdown]
# ## 4. Postselection primitives + a dependency-free AUC (Mann-Whitney)

# %%
def auc(risk, labels):
    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(risk, kind="mergesort")
    ranks = np.empty(len(risk)); ranks[order] = np.arange(1, len(risk) + 1)
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def postselect(risk, labels, discard, seed=RNG_SEED):
    n = len(risk); base = labels.mean()
    k = int(round(discard * n))
    if k <= 0:
        return base, 0.0
    jitter = np.random.RandomState(seed).uniform(0, 1e-9, n)
    keep = np.argsort(-(risk + jitter))[k:]           # drop highest-risk
    kept = labels[keep].mean()
    return float(kept), float((base - kept) / max(base, 1e-12) * 100.0)

# %% [markdown]
# ## 5. Run the sweep over d = 3, 5, 7
# Honesty guards (mirroring the study): ground truth is the real `obs_flips_actual`; we exclude the
# dead-logical long runs (LER >= 40%, randomized outcome) and near-trivial single-round data.

# %%
def run_distance(z, prefixes, max_shots=50000):
    labels, gaps, weights = [], [], []
    for p in prefixes:
        dets, obs, dem = load_from_zip(z, p, max_shots)
        try:
            preds, g = decode_with_gap(dem, dets)
        except ValueError:
            continue                                   # graph not exactly gap-decodable
        lab = (preds != obs).astype(int)
        if lab.mean() >= 0.40:                         # dead-logical, uninformative
            continue
        labels.append(lab); gaps.append(g)
    if not labels:
        return None
    return np.concatenate(labels), np.concatenate(gaps)

ROUND_FILTER = "r30"     # a low-LER operating point; set to "" to include all rounds
rows = []
for d in sorted(by_d):
    prefixes = [p for p in by_d[d] if (ROUND_FILTER in p or ROUND_FILTER == "")]
    out = run_distance(z, prefixes)
    if out is None:
        continue
    labels, gaps = out
    base = labels.mean()
    gap_risk = -gaps                                    # error correlates with SMALL gap
    rand_risk = np.random.RandomState(RNG_SEED).uniform(size=len(labels))
    g20, gred = postselect(gap_risk, labels, 0.20)
    r20, rred = postselect(rand_risk, labels, 0.20)
    rows.append(dict(d=d, n=len(labels), base=base,
                     auc_gap=auc(gap_risk, labels), auc_rand=auc(rand_risk, labels),
                     ler_cut_20=gred, rand_cut_20=rred, ler_at_20=g20))
    print(f"d={d}: n={len(labels):6d} baseLER={base:.3%} "
          f"AUC(gap)={auc(gap_risk,labels):.3f} | LER cut @20% discard = {gred:+.1f}% "
          f"(random {rred:+.1f}%)")

# %% [markdown]
# ## 6. Results table

# %%
print(f"{'d':>3} {'n':>7} {'baseLER':>9} {'AUC(gap)':>9} {'LER cut@20%':>12} {'random@20%':>11}")
for r in rows:
    print(f"{r['d']:>3} {r['n']:>7} {r['base']:>8.2%} {r['auc_gap']:>9.3f} "
          f"{r['ler_cut_20']:>11.1f}% {r['rand_cut_20']:>10.1f}%")

# %% [markdown]
# ## 7. Honest scope
# - **Adopted, not invented.** Complementary-gap / gap decoding is Gidney (2024); this notebook is a
#   clean-room reimplementation for an independent check.
# - **"Grows with distance" is primarily base-LER-driven** (a lower error rate sharpens the gap),
#   with only a modest pure-distance effect at matched error rate. Do not overclaim pure distance.
# - **Cost:** postselection discards ~20% of shots, it buys fidelity with throughput, and only
#   makes sense where you can afford to drop shots (memory experiments, protocols with a retry budget).
# - **Reproducibility:** every number above regenerates on your machine from the public Zenodo dataset.
#
# In the QubitBoost stack this method ships inside **SafetyGate** with governance around it (spend
# caps, drift gating, and a signed receipt recording the per-run payoff). The method is public; the
# product is the governed, auditable wrapper. The whole point: **every number reproduces.**
