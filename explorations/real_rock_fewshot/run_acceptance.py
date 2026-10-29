"""The recalibrator acceptance read on the production stack.

Val levels for every fit tier against zero-shot, the paired unit-bootstrap
vector-vs-matrix contrast (is the 7x7 coupling worth its 98 parameters), and
per-group 80% coverage with median interval width reported alongside so
vacuous widening is visible.  Fit on the committed train roster, reported on
val; the test slice is not touched.  The climatology comparison printed at the
end is internal-only and never travels to the paper.

Until 2026-09-03 this was the tailed-vs-matrix read; the tailed tier's margin
was traced to the amphibole pin and the tier deleted (journal 2026-09-03).

    conda run -n mineral python explorations/real_rock_fewshot/run_acceptance.py \
        --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import (
    climatology,
    load_benchmark,
    plot_coverage_sharpness,
    real_rock_nll,
    share_interval,
    split_mask,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    fit_recalibrator,
    paired_bootstrap,
    score_recalibrator,
)

RESULTS = Path(__file__).resolve().parent / "results"

# the values selected on val for run_curve; see run_curve.WEIGHT_DECAY
WEIGHT_DECAY = {"vector": 0.01, "matrix": 1.0}

FIT_TIERS = ("vector", "matrix")


def _corrected_alpha(recalibrator, gp) -> np.ndarray:
    if recalibrator is None:
        return gp.alpha7.numpy()
    with torch.no_grad():
        _, alpha = recalibrator(gp.logits7, gp.alpha7)
    return alpha.numpy()


def calibration_rows(recalibrator, gp) -> list[dict]:
    """Per-group cov80, median width, and the silica-residual correlation.

    Same eligibility rule as the residual decomposition: rocks with one
    quantified group are excluded, their marginal always covers.
    """
    numeric = gp.numeric.numpy()
    numeric_b = numeric.astype(bool)
    magnitude = gp.magnitude.numpy()
    sio2 = gp.sio2.numpy()
    eligible = numeric_b & (numeric_b.sum(1, keepdims=True) >= 2)
    alpha = _corrected_alpha(recalibrator, gp)
    lo, hi = share_interval(alpha, numeric)
    cond = conditional_mean(alpha, numeric)

    rows = []
    for g, group in enumerate(GROUPS):
        m = eligible[:, g]
        n = int(m.sum())
        if n < 3:
            rows.append({"group": group, "n": n})
            continue
        x = magnitude[m, g]
        resid = x - cond[m, g]
        rows.append({
            "group": group,
            "n": n,
            "cov80": float(((x >= lo[m, g]) & (x <= hi[m, g])).mean()),
            "median_width": float(np.median(hi[m, g] - lo[m, g])),
            "spearman_resid_sio2": float(spearmanr(resid, sio2[m]).statistic),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder.to(device), benchmark, device)
    gp = build_group_predictions(ckpt.build_mode_head().to(device),
                                 benchmark, latents)

    scored = benchmark.clean & benchmark.has_magnitude
    train_mask = split_mask(benchmark, "train") & scored
    val_mask = split_mask(benchmark, "val") & scored
    train, val = gp.subset(train_mask), gp.subset(val_mask)
    print(f"train {len(train)} rocks / {len(np.unique(train.unit))} units   "
          f"val {len(val)} rocks / {len(np.unique(val.unit))} units")

    fits = {tier: fit_recalibrator(train, tier=tier,
                                   weight_decay=WEIGHT_DECAY[tier])
            for tier in FIT_TIERS}
    scores = {"zero_shot": score_recalibrator(None, val)}
    scores.update({tier: score_recalibrator(fits[tier], val)
                   for tier in FIT_TIERS})

    print("\nval levels (joint / dirichlet / bernoulli):")
    for name, s in scores.items():
        print(f"  {name:9s} {s['joint'].mean():7.3f} "
              f"{s['dirichlet'].mean():7.3f} {s['bernoulli'].mean():7.3f}")

    contrasts = {}
    for term in ("joint", "dirichlet", "bernoulli"):
        boot = paired_bootstrap(scores["vector"][term], scores["matrix"][term],
                                val.unit)
        contrasts[f"vector_minus_matrix_{term}"] = boot
        print(f"vector minus matrix, {term:9s}: {boot['mean']:+.3f} "
              f"[{boot['lo']:+.3f}, {boot['hi']:+.3f}]")

    calibration = {name: calibration_rows(
        None if name == "zero_shot" else fits[name], val)
        for name in ("zero_shot", "vector", "matrix")}
    print("\nval cov80 (median width) per group:")
    for g, group in enumerate(GROUPS):
        line = f"  {group:4s}"
        for name in ("zero_shot", "vector", "matrix"):
            r = calibration[name][g]
            line += (f"   {r['cov80']:.2f} ({r['median_width']:.2f})"
                     if "cov80" in r else "   --")
        print(line)

    # internal-only reference (owner-locked 2026-07-30: never in the paper,
    # its figures, or claims.md)
    null = climatology(benchmark)
    null_nll = real_rock_nll(null["p7"], null["alpha7"], benchmark.presence,
                             benchmark.magnitude, benchmark.numeric)
    null_joint = float(null_nll["joint"][val_mask].mean())
    print(f"\ninternal: climatology val joint {null_joint:.3f} vs "
          f"matrix {scores['matrix']['joint'].mean():.3f}")

    tag = args.checkpoint.parent.name
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    val_rows = np.flatnonzero(val_mask)
    numeric = val.numeric.numpy()
    arms = {
        "zero-shot": share_interval(_corrected_alpha(None, val), numeric),
        "matrix": share_interval(_corrected_alpha(fits["matrix"], val), numeric),
        "climatology": tuple(
            a[val_rows] for a in share_interval(
                null["alpha7"], benchmark.numeric.astype(np.float64))
        ),
    }
    plot_coverage_sharpness(
        arms, val.magnitude.numpy(), numeric, val.unit,
        out_dir / "acceptance_coverage_sharpness.png",
        f"{len(val)} val rocks, recalibrators fit on {len(train)} train rocks",
    )

    out = {
        "checkpoint": str(args.checkpoint),
        "benchmark": "eruption-v2",
        "fit": f"full train roster ({len(train)} rocks)",
        "weight_decay": dict(WEIGHT_DECAY),
        "val_levels": {
            name: {term: float(s[term].mean())
                   for term in ("joint", "dirichlet", "bernoulli")}
            for name, s in scores.items()
        },
        "paired_contrasts": contrasts,
        "calibration": calibration,
        "internal_climatology_val_joint": null_joint,
    }
    out_path = out_dir / "acceptance.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
