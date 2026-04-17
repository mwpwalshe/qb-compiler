#!/usr/bin/env python3
"""PyMatching baseline sweep for qb-compiler's NVIDIA-Ising integration.

Runs :class:`~qb_compiler.ising.PyMatchingDecoder` on rotated
surface-code memory experiments across a grid of ``(distance,
rounds, p_error, basis)`` and writes the results to
``benchmarks/ising/pymatching_sweep.json``.

This script is the reference the NVIDIA Ising pre-decoder must beat.
Users with the NVIDIA weights can run
``benchmarks/ising/run_ising_sweep.py`` (not included — requires
gated HuggingFace weights) on the same grid and diff the two
JSON files to quantify the pre-decoder's marginal win.

Run::

    python benchmarks/ising/run_pymatching_sweep.py \
        --shots 50000 --output benchmarks/ising/pymatching_sweep.json

On a modern laptop this completes in a few minutes for the default
sweep grid.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qb_compiler.ising import (
    PyMatchingDecoder,
    SurfaceCodePatchSpec,
    evaluate_logical_error_rate,
)


def run_sweep(
    shots: int,
    distances: list[int],
    rounds_mode: str,
    p_errors: list[float],
    bases: list[str],
    seed: int,
) -> list[dict]:
    results: list[dict] = []
    for d in distances:
        rounds = d if rounds_mode == "distance" else int(rounds_mode)
        for basis in bases:
            for p in p_errors:
                spec = SurfaceCodePatchSpec(
                    distance=d, rounds=rounds, basis=basis, p_error=p
                )
                t0 = time.time()
                decoder = PyMatchingDecoder(spec)
                record = evaluate_logical_error_rate(
                    spec,
                    decoder,
                    shots=shots,
                    seed=seed,
                    decoder_name="pymatching",
                )
                elapsed = time.time() - t0
                as_dict = record.as_dict()
                as_dict["elapsed_seconds"] = round(elapsed, 3)
                print(
                    f"d={d:2d} T={rounds:2d} basis={basis} p={p:.4f} "
                    f"→ LER = {record.rate:.4e} ± {record.standard_error:.2e} "
                    f"(errors={record.logical_errors}/{shots}, {elapsed:.2f}s)",
                    flush=True,
                )
                results.append(as_dict)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument(
        "--distances", type=int, nargs="+", default=[3, 5, 7, 9]
    )
    parser.add_argument(
        "--rounds-mode",
        default="distance",
        help="'distance' (T=d) or an integer literal.",
    )
    parser.add_argument(
        "--p-errors",
        type=float,
        nargs="+",
        default=[0.001, 0.002, 0.003, 0.005, 0.008, 0.012],
    )
    parser.add_argument("--bases", nargs="+", default=["X", "Z"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "pymatching_sweep.json",
    )
    args = parser.parse_args()

    print(
        f"# qb-compiler Ising integration — PyMatching baseline sweep\n"
        f"# shots={args.shots}, distances={args.distances}, "
        f"p_errors={args.p_errors}, bases={args.bases}, seed={args.seed}"
    )

    results = run_sweep(
        shots=args.shots,
        distances=args.distances,
        rounds_mode=args.rounds_mode,
        p_errors=args.p_errors,
        bases=args.bases,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(
            {
                "metadata": {
                    "shots": args.shots,
                    "seed": args.seed,
                    "rounds_mode": args.rounds_mode,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
