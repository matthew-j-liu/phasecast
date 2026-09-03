"""
Real-rock headline figures: what the model does on real rocks, and what it costs.

Five canvases from a joint-heads checkpoint, scored zero-shot on the
least-altered rocks of the eruption benchmark:

    cohort_card.png         inputs, which minerals the model calls, how much
                            of each on the measured and on its own assemblage
    magnitude_ratio.png     per-rock magnitude error, paired
    nll_distribution.png    the headline split by term, against climatology
    presence_brier.png      presence Brier and its Murphy decomposition
    coverage_sharpness.png  80% interval coverage and the width it costs

    python scripts/view/real_rock_card.py \\
        --checkpoint checkpoints/joint_heads_supportmag_seed0/heads_best.pt

The figures land in `results/<checkpoint-dir-name>/` unless `--out-dir` says
otherwise. The fuller diagnostic sweep stays in `runs/real_rock/<tag>/`,
written by the eval itself.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import (
    climatology,
    load_benchmark,
    plot_cohort_card,
    plot_coverage_sharpness,
    plot_magnitude_ratio,
    plot_nll_distribution,
    plot_presence_brier,
    predict,
    real_rock_nll,
    share_interval,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to a joint-heads checkpoint (.pt)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output dir (default results/<checkpoint dir>)")
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    benchmark = load_benchmark()
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)
    pred = predict(ckpt.build_mode_head().to(device), latents, benchmark)

    clean = benchmark.clean & benchmark.has_magnitude
    p7, alpha7 = pred["p7"], pred["alpha7"]
    forced_share = conditional_mean(alpha7, benchmark.numeric.astype(np.float64))
    model_share = conditional_mean(
        pred["alpha7_model_support"], pred["model_support7"].astype(np.float64)
    )
    nll = real_rock_nll(p7, alpha7, benchmark.presence, benchmark.magnitude,
                        benchmark.numeric)
    null = climatology(benchmark)
    null_nll = real_rock_nll(null["p7"], null["alpha7"], benchmark.presence,
                             benchmark.magnitude, benchmark.numeric)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("results") / args.checkpoint.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    subtitle = f"{int(clean.sum())} least-altered rocks"

    plot_cohort_card(
        p7[clean], forced_share[clean], model_share[clean],
        pred["model_support7"][clean], benchmark.subset(clean),
        out_dir / "cohort_card.png",
    )
    plot_magnitude_ratio(
        forced_share[clean], benchmark.magnitude[clean],
        benchmark.numeric[clean], out_dir / "magnitude_ratio.png", subtitle,
    )
    plot_nll_distribution(
        {k: v[clean] for k, v in nll.items()},
        {k: v[clean] for k, v in null_nll.items()},
        out_dir / "nll_distribution.png", subtitle,
    )
    plot_presence_brier(
        p7[clean], null["p7"][clean], benchmark.presence[clean],
        benchmark.unit[clean], out_dir / "presence_brier.png", subtitle,
    )
    mask = benchmark.numeric.astype(np.float64)
    lo, hi = share_interval(alpha7, mask)
    null_lo, null_hi = share_interval(null["alpha7"], mask)
    plot_coverage_sharpness(
        {"model": (lo[clean], hi[clean]),
         "climatology": (null_lo[clean], null_hi[clean])},
        benchmark.magnitude[clean], benchmark.numeric[clean],
        benchmark.unit[clean], out_dir / "coverage_sharpness.png", subtitle,
    )


if __name__ == "__main__":
    main()
