#!/usr/bin/env python3
"""Paired diff of two Ising-integration sweep JSONs.

Matches records from a baseline sweep (e.g. ``pymatching_sweep.json``)
against a candidate sweep (e.g. ``ising_sweep.json``) on the
``(distance, rounds, basis, p_error)`` key and reports, per config, the
candidate's logical-error-rate delta versus the baseline plus a
two-proportion significance verdict.  Writes the full table and a
summary to ``ising_vs_pymatching.json``.

Statistics note.  Both sweeps seed stim identically
(``compile_detector_sampler(seed=...)``), so if they ran on the same
seed and grid each decoder saw the SAME syndromes per config and the
comparison is paired at the shot level.  This helper only has the
aggregate counts, so it uses the standard pooled two-proportion z-test,
which treats the two runs as independent.  That is CONSERVATIVE on paired
data (it overstates the variance of the difference), so a 95% verdict
here is a floor, not a ceiling.  For the tightest test, run both
decoders in one loop on identical syndromes and apply McNemar on the
per-shot disagreement counts.

Run::

    python benchmarks/ising/diff_sweeps.py \
        benchmarks/ising/pymatching_sweep.json \
        benchmarks/ising/ising_sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

Z_95 = 1.959963984540054


def _key(rec: dict) -> tuple:
    return (rec["distance"], rec["rounds"], rec["basis"], rec["p_error"])


def _load(path: Path) -> tuple[dict, dict]:
    blob = json.loads(path.read_text())
    results = blob["results"] if isinstance(blob, dict) and "results" in blob else blob
    metadata = blob.get("metadata", {}) if isinstance(blob, dict) else {}
    return metadata, {_key(r): r for r in results}


def _two_proportion_z(e_base: int, n_base: int, e_cand: int, n_cand: int) -> float | None:
    """Pooled two-proportion z for (p_base - p_cand).  None if undefined.

    Positive z means the candidate has the LOWER error rate (the win).
    """
    if n_base == 0 or n_cand == 0:
        return None
    p_base = e_base / n_base
    p_cand = e_cand / n_cand
    p_pool = (e_base + e_cand) / (n_base + n_cand)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_base + 1.0 / n_cand))
    if se == 0.0:
        return None
    return (p_base - p_cand) / se


def diff(baseline: dict, candidate: dict) -> list[dict]:
    rows: list[dict] = []
    for key in sorted(set(baseline) | set(candidate)):
        b = baseline.get(key)
        c = candidate.get(key)
        d, t, basis, p = key
        if b is None or c is None:
            rows.append(
                {
                    "distance": d,
                    "rounds": t,
                    "basis": basis,
                    "p_error": p,
                    "status": "unmatched",
                    "in_baseline": b is not None,
                    "in_candidate": c is not None,
                }
            )
            continue

        p_base, p_cand = b["rate"], c["rate"]
        delta = p_base - p_cand  # positive = candidate better (lower LER)
        rel = (delta / p_base * 100.0) if p_base > 0 else None
        z = _two_proportion_z(b["logical_errors"], b["shots"], c["logical_errors"], c["shots"])
        if z is None:
            verdict = "tie"
        elif z >= Z_95:
            verdict = "candidate_better"
        elif z <= -Z_95:
            verdict = "candidate_worse"
        else:
            verdict = "tie"

        rows.append(
            {
                "distance": d,
                "rounds": t,
                "basis": basis,
                "p_error": p,
                "status": "matched",
                "baseline_ler": p_base,
                "candidate_ler": p_cand,
                "delta_ler": delta,
                "relative_reduction_pct": rel,
                "z": z,
                "verdict": verdict,
                "baseline_shots": b["shots"],
                "candidate_shots": c["shots"],
            }
        )
    return rows


def _fmt(x: float | None, spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'d':>2} {'T':>2} {'bas':>3} {'p':>7} "
        f"{'base_LER':>10} {'cand_LER':>10} {'delta':>11} "
        f"{'rel%':>7} {'z':>7}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["status"] == "unmatched":
            where = "baseline only" if r["in_baseline"] else "candidate only"
            print(
                f"{r['distance']:>2} {r['rounds']:>2} {r['basis']:>3} "
                f"{r['p_error']:>7.4f} {'':>10} {'':>10} {'':>11} "
                f"{'':>7} {'':>7}  UNMATCHED ({where})"
            )
            continue
        print(
            f"{r['distance']:>2} {r['rounds']:>2} {r['basis']:>3} "
            f"{r['p_error']:>7.4f} "
            f"{r['baseline_ler']:>10.4e} {r['candidate_ler']:>10.4e} "
            f"{r['delta_ler']:>+11.3e} "
            f"{_fmt(r['relative_reduction_pct'], '>7.1f')} "
            f"{_fmt(r['z'], '>7.2f')}  {r['verdict']}"
        )


def summarize(rows: list[dict]) -> dict:
    matched = [r for r in rows if r["status"] == "matched"]
    better = [r for r in matched if r["verdict"] == "candidate_better"]
    worse = [r for r in matched if r["verdict"] == "candidate_worse"]
    tie = [r for r in matched if r["verdict"] == "tie"]
    unmatched = [r for r in rows if r["status"] == "unmatched"]
    return {
        "configs_matched": len(matched),
        "configs_unmatched": len(unmatched),
        "candidate_significantly_better": len(better),
        "candidate_significantly_worse": len(worse),
        "tie": len(tie),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path, help="Baseline sweep JSON (pymatching_sweep.json)")
    parser.add_argument("candidate", type=Path, help="Candidate sweep JSON (ising_sweep.json)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the diff JSON (default: alongside the candidate "
        "as ising_vs_pymatching.json).",
    )
    args = parser.parse_args()

    base_meta, baseline = _load(args.baseline)
    cand_meta, candidate = _load(args.candidate)

    # Pairing guard: warn loudly if the two runs are not comparable.
    b_seed, c_seed = base_meta.get("seed"), cand_meta.get("seed")
    b_shots, c_shots = base_meta.get("shots"), cand_meta.get("shots")
    if b_seed is not None and c_seed is not None and b_seed != c_seed:
        print(
            f"WARNING: seeds differ (baseline={b_seed}, candidate={c_seed}). "
            f"The two runs did not see identical syndromes, so this is an "
            f"UNPAIRED comparison and the z-verdicts lose power.",
            flush=True,
        )
    if b_shots is not None and c_shots is not None and b_shots != c_shots:
        print(
            f"NOTE: shot counts differ (baseline={b_shots}, candidate={c_shots}). "
            f"Rates are still comparable; the z-test accounts for both Ns.",
            flush=True,
        )

    rows = diff(baseline, candidate)
    print_table(rows)
    summary = summarize(rows)

    print()
    print(
        f"summary: {summary['configs_matched']} matched | "
        f"candidate better {summary['candidate_significantly_better']} | "
        f"worse {summary['candidate_significantly_worse']} | "
        f"tie {summary['tie']} | "
        f"unmatched {summary['configs_unmatched']}  (95% pooled two-proportion z)"
    )

    out = args.output or (args.candidate.parent / "ising_vs_pymatching.json")
    out.write_text(
        json.dumps(
            {
                "baseline_file": str(args.baseline),
                "candidate_file": str(args.candidate),
                "baseline_metadata": base_meta,
                "candidate_metadata": cand_meta,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"Wrote diff to {out}")


if __name__ == "__main__":
    main()
