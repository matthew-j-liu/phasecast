"""The tailed-vs-matrix acceptance read on the production stack.

The tailed tier is matrix plus the precision contamination the residual
decomposition measured a need for (the residual is heavier-tailed than any
Beta at the fitted mean).  This scores whether it delivers, on the criteria
fixed before the code was written:

    (a) paired unit-bootstrap val joint NLL, tailed vs matrix -- since the
        tiers differ by the contamination alone, this contrast is also its
        attribution
    (b) per-group 80% coverage, with median interval width reported alongside
        so vacuous widening is visible
    (c) the presence term flat (the upgrade is magnitude-side only)

A silica location term was part of the original pre-registered design; it
measured +0.007 [-0.004, +0.016] nats and was excised (journal 2026-08-06).
The per-group Spearman(residual, SiO2) stays in the JSON as a diagnostic.

Fit on the committed train roster, reported on val; the test slice (spent
2026-08-03) is not touched.  The climatology comparison printed at the end is
internal-only and never travels to the paper.

    conda run -n mineral python explorations/real_rock_fewshot/run_tailed_acceptance.py \
        --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean, mixture_conditional_mean
from cnmineral.eval.real_rock import (
    climatology,
    load_benchmark,
    mixture_share_interval,
    plot_coverage_sharpness,
    real_rock_nll,
    share_interval,
    split_mask,
)
from cnmineral.models.real_rock_recalibrator import CONTAM_LOGIT_BIAS
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    fit_recalibrator,
    paired_bootstrap,
    score_recalibrator,
    select_weight_decay,
)

RESULTS = Path(__file__).resolve().parent / "results"

# the values selected on val for run_curve; see run_curve.WEIGHT_DECAY
WEIGHT_DECAY = {"vector": 0.1, "matrix": 1.0, "tailed": 1.0}

FIT_TIERS = ("vector", "matrix", "tailed")


def _intervals(recalibrator, gp):
    """(lo, hi) of the stated 80% interval under one recalibrator (None = zero-shot)."""
    numeric = gp.numeric.numpy()
    if recalibrator is None:
        return share_interval(gp.alpha7.numpy(), numeric)
    with torch.no_grad():
        _, alpha, log_w = recalibrator(gp.logits7, gp.alpha7, gp.sio2)
    if log_w is None:
        return share_interval(alpha.numpy(), numeric)
    return mixture_share_interval(alpha.numpy(), log_w.numpy(), numeric)


def _cond_mean(recalibrator, gp):
    numeric = gp.numeric.numpy()
    if recalibrator is None:
        return conditional_mean(gp.alpha7.numpy(), numeric)
    with torch.no_grad():
        _, alpha, log_w = recalibrator(gp.logits7, gp.alpha7, gp.sio2)
    if log_w is None:
        return conditional_mean(alpha.numpy(), numeric)
    return mixture_conditional_mean(alpha.numpy(), log_w.numpy(), numeric)


def calibration_rows(recalibrator, gp) -> list[dict]:
    """Per-group cov80, median width, and the silica-residual correlation.

    Same eligibility rule as the residual decomposition: rocks with one
    quantified group are excluded, their marginal always covers.
    """
    numeric_b = gp.numeric.numpy().astype(bool)
    magnitude = gp.magnitude.numpy()
    sio2 = gp.sio2.numpy()
    eligible = numeric_b & (numeric_b.sum(1, keepdims=True) >= 2)
    lo, hi = _intervals(recalibrator, gp)
    cond = _cond_mean(recalibrator, gp)

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


def tailed_params(recalibrator) -> dict:
    with torch.no_grad():
        return {
            "eps": float(torch.sigmoid(recalibrator.contam_logit
                                       + CONTAM_LOGIT_BIAS)),
            "k": float(recalibrator.contam_logk.exp()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--select-wd", action="store_true",
                        help="select the tailed tier's weight decay on val "
                             "before fitting (vector/matrix keep their "
                             "recorded constants)")
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

    weight_decay = dict(WEIGHT_DECAY)
    wd_records = None
    if args.select_wd:
        best, wd_records = select_weight_decay(
            train, val, fit=partial(fit_recalibrator, tier="tailed"),
        )
        weight_decay["tailed"] = best
        print(f"selected weight_decay[tailed] = {best:g}")

    fits = {tier: fit_recalibrator(train, tier=tier,
                                   weight_decay=weight_decay[tier])
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
        boot = paired_bootstrap(scores["matrix"][term], scores["tailed"][term],
                                val.unit)
        contrasts[f"matrix_minus_tailed_{term}"] = boot
        print(f"matrix minus tailed, {term:9s}: {boot['mean']:+.3f} "
              f"[{boot['lo']:+.3f}, {boot['hi']:+.3f}]")

    calibration = {name: calibration_rows(
        None if name == "zero_shot" else fits[name], val)
        for name in ("zero_shot", "matrix", "tailed")}
    print("\nval cov80 (median width) per group:")
    for g, group in enumerate(GROUPS):
        line = f"  {group:4s}"
        for name in ("zero_shot", "matrix", "tailed"):
            r = calibration[name][g]
            line += (f"   {r['cov80']:.2f} ({r['median_width']:.2f})"
                     if "cov80" in r else "   --")
        print(line)
    print(f"\ntailed fit: {tailed_params(fits['tailed'])}")

    # internal-only reference (owner-locked 2026-07-30: never in the paper,
    # its figures, or claims.md)
    null = climatology(benchmark)
    null_nll = real_rock_nll(null["p7"], null["alpha7"], benchmark.presence,
                             benchmark.magnitude, benchmark.numeric)
    null_joint = float(null_nll["joint"][val_mask].mean())
    print(f"\ninternal: climatology val joint {null_joint:.3f} vs "
          f"tailed {scores['tailed']['joint'].mean():.3f}")

    tag = args.checkpoint.parent.name
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    val_rows = np.flatnonzero(val_mask)
    arms = {
        "zero-shot": _intervals(None, val),
        "matrix": _intervals(fits["matrix"], val),
        "tailed": _intervals(fits["tailed"], val),
        "climatology": tuple(
            a[val_rows] for a in share_interval(
                null["alpha7"], benchmark.numeric.astype(np.float64))
        ),
    }
    plot_coverage_sharpness(
        arms, val.magnitude.numpy(), val.numeric.numpy(), val.unit,
        out_dir / "tailed_coverage_sharpness.png",
        f"{len(val)} val rocks, recalibrators fit on {len(train)} train rocks",
    )

    out = {
        "checkpoint": str(args.checkpoint),
        "benchmark": "eruption-v2",
        "fit": f"full train roster ({len(train)} rocks)",
        "weight_decay": {t: weight_decay[t] for t in FIT_TIERS},
        "weight_decay_selection": wd_records,
        "val_levels": {
            name: {term: float(s[term].mean())
                   for term in ("joint", "dirichlet", "bernoulli")}
            for name, s in scores.items()
        },
        "paired_contrasts": contrasts,
        "calibration": calibration,
        "tailed_params": tailed_params(fits["tailed"]),
        "internal_climatology_val_joint": null_joint,
    }
    out_path = out_dir / "tailed_acceptance.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
