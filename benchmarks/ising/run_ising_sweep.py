#!/usr/bin/env python3
"""NVIDIA Ising pre-decoder sweep for qb-compiler's Ising integration.

The counterpart to ``run_pymatching_sweep.py``.  It runs
:class:`~qb_compiler.ising.IsingDecoderWrapper` (NVIDIA's
Ising-Decoder-SurfaceCode-1 pre-decoder chained into the PyMatching
residual) across the SAME ``(distance, rounds, p_error, basis)`` grid
and writes the results to ``benchmarks/ising/ising_sweep.json`` in the
same record shape as the baseline.  Feed both JSONs to
``diff_sweeps.py`` to quantify the pre-decoder's marginal win.

Three things qb-compiler deliberately does not vendor, so you supply them:

1. The gated weights.  Accept the NVIDIA Open Model License on the HF
   repo page, ``huggingface-cli login``, then download e.g.
   ``Ising-Decoder-SurfaceCode-1-Fast.pt`` and pass ``--weights``.
2. The model definition.  Clone ``github.com/NVIDIA/Ising-Decoding``
   (Apache 2.0, model in ``code/model/predecoder.py``) and point
   ``--nvidia-repo`` at the ``code/`` dir so ``from model.predecoder
   import ...`` resolves.
3. The ``build_model`` glue below.  The cfg construction depends on
   NVIDIA's repo; reconcile it with their ``predecoder.py`` and flip
   ``_CFG_VERIFIED`` to ``True``.  Until then the script refuses to run
   so you never benchmark a guessed config (a wrong cfg silently makes
   the pre-decoder lose, which is exactly the fake result to avoid).

Pairing note: this sweep and the PyMatching baseline both seed stim via
``compile_detector_sampler(seed=...)``, so with a matching seed and grid
both decoders see IDENTICAL syndromes per config.  Keep ``--seed`` and
the grid args equal to the baseline run or the diff is not paired.

Run::

    python benchmarks/ising/run_ising_sweep.py \
        --weights ~/weights/Ising-Decoder-SurfaceCode-1-Fast.pt \
        --nvidia-repo ~/src/Ising-Decoding/code \
        --shots 50000 \
        --output benchmarks/ising/ising_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from qb_compiler.ising import (
    IsingDecoderConfig,
    IsingDecoderWrapper,
    SurfaceCodePatchSpec,
    evaluate_logical_error_rate,
)
from qb_compiler.ising.stim_adapter import resolve_layout

# Flip to True only after you have reconciled the cfg in build_model()
# below with NVIDIA's code/model/predecoder.py.  Left False so a guessed
# config can never produce a benchmarked number.
_CFG_VERIFIED = False


def build_model(spec: SurfaceCodePatchSpec):
    """Instantiate NVIDIA's pre-decoder module for *spec*.

    qb-compiler feeds the model a float32 tensor of shape
    ``(batch, 4, rounds, distance, distance)`` with the channel layout
    ``[x_type, z_type, x_present, z_present]`` (see
    ``stim_adapter.build_ising_tensor``).  Whatever you return must
    consume that shape and emit the 4-channel correction tensor the
    wrapper expects.

    The class name and cfg fields come from NVIDIA's repo, not from
    here.  The block below is the shape you almost certainly need; the
    exact field names are theirs to define.
    """
    if not _CFG_VERIFIED:
        raise NotImplementedError(
            "build_model() is not wired up yet.  Open NVIDIA's "
            "code/model/predecoder.py, reconcile the cfg fields below with "
            "their constructor, then set _CFG_VERIFIED = True at the top of "
            "this file.  Do not benchmark a guessed config."
        )

    # The model definition lives in NVIDIA's Apache-2.0 repo, pulled in
    # via --nvidia-repo (prepended to sys.path in main()).
    from model.predecoder import PreDecoderModelMemory_v1  # type: ignore[import-not-found]

    # Note: these keys mirror NVIDIA's cfg dataclass / namespace, so they must
    # track upstream. The values are derived from the patch spec; the KEYS are
    # theirs, and will need updating if upstream renames them.
    cfg = dict(
        distance=spec.distance,
        rounds=spec.rounds,
        in_channels=4,
    )
    return PreDecoderModelMemory_v1(cfg)


def run_sweep(
    weights_path: str,
    device: str,
    shots: int,
    distances: list[int],
    rounds_mode: str,
    p_errors: list[float],
    bases: list[str],
    seed: int,
    expect_fingerprint: str | None,
) -> list[dict]:
    results: list[dict] = []
    for d in distances:
        rounds = d if rounds_mode == "distance" else int(rounds_mode)
        for basis in bases:
            for p in p_errors:
                spec = SurfaceCodePatchSpec(distance=d, rounds=rounds, basis=basis, p_error=p)

                fingerprint = getattr(resolve_layout(spec), "orientation_fingerprint", None)
                if expect_fingerprint and fingerprint != expect_fingerprint:
                    raise SystemExit(
                        f"orientation fingerprint mismatch at d={d} basis={basis}: "
                        f"got {fingerprint}, expected {expect_fingerprint}.  Feeding a "
                        f"mismatched layout into pretrained NVIDIA weights produces "
                        f"garbage corrections (a spurious NVIDIA loss).  Resolve the "
                        f"layout convention before benchmarking."
                    )

                t0 = time.time()
                config = IsingDecoderConfig(
                    weights_path=weights_path,
                    device=device,
                    build_model=build_model,
                )
                decoder = IsingDecoderWrapper(spec, config)
                record = evaluate_logical_error_rate(
                    spec,
                    decoder,
                    shots=shots,
                    seed=seed,
                    decoder_name="ising_nvidia",
                )
                elapsed = time.time() - t0
                as_dict = record.as_dict()
                as_dict["elapsed_seconds"] = round(elapsed, 3)
                as_dict["orientation_fingerprint"] = fingerprint
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
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the gated NVIDIA checkpoint (.pt or .safetensors).",
    )
    parser.add_argument(
        "--nvidia-repo",
        type=Path,
        default=None,
        help="Path to NVIDIA Ising-Decoding 'code/' dir (added to sys.path "
        "so 'from model.predecoder import ...' resolves).",
    )
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'.")
    parser.add_argument(
        "--expect-fingerprint",
        default=None,
        help="If set, every spec's orientation_fingerprint must equal this "
        "or the run aborts.  Use once you know NVIDIA's training layout.",
    )
    # Grid args mirror run_pymatching_sweep.py so the two sweeps line up.
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
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
        default=Path(__file__).parent / "ising_sweep.json",
    )
    args = parser.parse_args()

    if args.nvidia_repo is not None:
        sys.path.insert(0, str(args.nvidia_repo))

    print(
        f"# qb-compiler Ising integration: NVIDIA pre-decoder sweep\n"
        f"# weights={args.weights}, device={args.device}, shots={args.shots}, "
        f"distances={args.distances}, p_errors={args.p_errors}, "
        f"bases={args.bases}, seed={args.seed}"
    )

    results = run_sweep(
        weights_path=args.weights,
        device=args.device,
        shots=args.shots,
        distances=args.distances,
        rounds_mode=args.rounds_mode,
        p_errors=args.p_errors,
        bases=args.bases,
        seed=args.seed,
        expect_fingerprint=args.expect_fingerprint,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(
            {
                "metadata": {
                    "shots": args.shots,
                    "seed": args.seed,
                    "rounds_mode": args.rounds_mode,
                    "weights": str(args.weights),
                    "device": args.device,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {len(results)} records to {args.output}")
    print(
        f"Diff against the baseline with:\n"
        f"  python benchmarks/ising/diff_sweeps.py "
        f"benchmarks/ising/pymatching_sweep.json {args.output}"
    )


if __name__ == "__main__":
    main()
