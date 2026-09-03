"""The eruption-v2 numbers the Andrews numbers are read against.

Same checkpoint, same statistics, so the cross-benchmark comparison is not
against figures measured on a different arm. Emits the headline NLL against
climatology, the per-group median share ratio, and climatology per split slice
-- the last because the recorded tailed-tier number is val-scored while the
Andrews one is test-scored, and a margin over the null only means something
when both sides are on the same slice.

Reads the benchmark on its default basis (the visual phenocryst estimate), so
these are the figures everything before 2026-08-06 was measured against.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import numpy as np

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean, mixture_conditional_mean
from cnmineral.eval.real_rock import (
    climatology, load_benchmark, predict, real_rock_nll, split_mask,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_CHECKPOINT = "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt"
TERMS = ("bernoulli", "dirichlet", "joint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder.to(device), benchmark, device)
    pred = predict(ckpt.build_mode_head().to(device), latents, benchmark)

    clean = benchmark.clean & benchmark.has_magnitude
    nll = real_rock_nll(pred["p7"], pred["alpha7"], benchmark.presence,
                        benchmark.magnitude, benchmark.numeric)
    null = climatology(benchmark)
    null_nll = real_rock_nll(null["p7"], null["alpha7"], benchmark.presence,
                             benchmark.magnitude, benchmark.numeric)

    print(f"eruption-v2, {int(clean.sum())} least-altered rocks")
    for term in TERMS:
        m, c = nll[term][clean].mean(), null_nll[term][clean].mean()
        print(f"  {term:10s} model {m:7.3f}   climatology {c:7.3f}   "
              f"delta {m - c:+.3f}")

    print("\nclimatology joint by slice (the Andrews margin needs a matched one):")
    by_slice = {}
    for name in ("val", "test"):
        rows = clean & split_mask(benchmark, name)
        by_slice[name] = float(null_nll["joint"][rows].mean())
        print(f"  {name:5s} {int(rows.sum()):4d} rocks   {by_slice[name]:7.3f}")
    by_slice["all_clean"] = float(null_nll["joint"][clean].mean())
    print(f"  {'all':5s} {int(clean.sum()):4d} rocks   {by_slice['all_clean']:7.3f}")

    numeric = benchmark.numeric.astype(np.float64)
    share = (conditional_mean(pred["alpha7"], numeric) if log_w is None
             else mixture_conditional_mean(pred["alpha7"], log_w, numeric))
    print("\nmedian predicted/measured share ratio, this checkpoint:")
    ratios = {}
    for i, group in enumerate(GROUPS):
        rows = clean & benchmark.numeric[:, i] & (benchmark.magnitude[:, i] > 0)
        if rows.sum() == 0:
            print(f"  {group:4s} n=0")
            continue
        ratios[group] = float(np.median(share[rows, i]
                                        / benchmark.magnitude[rows, i]))
        print(f"  {group:4s} n={int(rows.sum()):4d}  median x{ratios[group]:.2f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "dubray_reference.json"
    out.write_text(json.dumps({
        "benchmark": "eruption-v2 (visual basis, the default)",
        "n_clean": int(clean.sum()),
        "model": {t: float(nll[t][clean].mean()) for t in TERMS},
        "climatology": {t: float(null_nll[t][clean].mean()) for t in TERMS},
        "climatology_joint_by_slice": by_slice,
        "median_share_ratio": ratios,
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
