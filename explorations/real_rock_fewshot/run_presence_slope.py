"""Do the q/amp presence silica slopes earn their two parameters?

The slopes entered the recalibrator's parameterization at design time from a
residual-vs-silica read (journal 2026-07-24) and were never contrasted against
their own absence.  The magnitude location term was: it measured +0.007
[-0.004, +0.016] nats and was excised (journal 2026-08-06).  This is the
missing contrast for the presence side.

The canonical tailed tier is fit twice on the same rocks, once with
`presence_sio2` free and once held at its zero initialization, and scored
paired on val with the unit bootstrap.  Positive means the slopes pay.  Both
arms take the tailed tier's recorded shrinkage constant; `--select-wd` also
re-selects one for the pinned arm, since the shipped protocol chooses a
constant per correction form and the pinned arm is a different form.

Fit on the committed train roster, reported on val; the test slice (spent
2026-08-03) is not touched.

    conda run -n alignn_fix python explorations/real_rock_fewshot/run_presence_slope.py \
        --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, split_mask
from cnmineral.models.real_rock_recalibrator import SIO2_PRESENCE_GROUPS, SIO2_SCALE
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

TIER = "tailed"

# the value selected on val for the tailed tier; see run_curve.WEIGHT_DECAY
WEIGHT_DECAY = 1.0

TERMS = ("joint", "bernoulli", "dirichlet")


def slope_contrast(train, val, *, weight_decay: float = WEIGHT_DECAY,
                   pinned_weight_decay: float | None = None) -> dict:
    """Tailed tier with the presence slopes free vs pinned at zero, paired on val.

    The tiers are deterministic given their zero/one initialization, so the two
    arms differ by the slopes alone and the contrast is their attribution.
    """
    pinned_wd = weight_decay if pinned_weight_decay is None else pinned_weight_decay
    fits = {
        "free": fit_recalibrator(train, tier=TIER, weight_decay=weight_decay),
        "pinned": fit_recalibrator(train, tier=TIER, weight_decay=pinned_wd,
                                   fit_presence_sio2=False),
    }
    scores = {name: score_recalibrator(fit, val) for name, fit in fits.items()}
    with torch.no_grad():
        slopes = dict(zip(SIO2_PRESENCE_GROUPS,
                          fits["free"].presence_sio2.tolist()))
    return {
        "weight_decay": {"free": weight_decay, "pinned": pinned_wd},
        "val_levels": {
            name: {term: float(s[term].mean()) for term in TERMS}
            for name, s in scores.items()
        },
        "val_share_mae": {name: float(s["share_mae_mean"])
                          for name, s in scores.items()},
        "pinned_minus_free": {
            term: paired_bootstrap(scores["pinned"][term], scores["free"][term],
                                   val.unit)
            for term in TERMS
        },
        # x enters as (SiO2 - 65) / 10, so the fitted value is the logit shift
        # across 10 wt% SiO2
        "fitted_logit_shift_per_10wt_pct_sio2": slopes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--select-wd", action="store_true",
                        help="re-select the pinned arm's weight decay on val "
                             "instead of sharing the tailed tier's constant")
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder.to(device), benchmark, device)
    gp = build_group_predictions(ckpt.build_mode_head().to(device),
                                 benchmark, latents)

    scored = benchmark.clean & benchmark.has_magnitude
    train = gp.subset(split_mask(benchmark, "train") & scored)
    val = gp.subset(split_mask(benchmark, "val") & scored)
    print(f"train {len(train)} rocks / {len(np.unique(train.unit))} units   "
          f"val {len(val)} rocks / {len(np.unique(val.unit))} units")

    pinned_wd, wd_records = None, None
    if args.select_wd:
        pinned_wd, wd_records = select_weight_decay(
            train, val,
            fit=partial(fit_recalibrator, tier=TIER, fit_presence_sio2=False),
        )
        print(f"selected weight_decay[pinned] = {pinned_wd:g}")

    out = slope_contrast(train, val, pinned_weight_decay=pinned_wd)
    out["weight_decay_selection"] = wd_records

    print("\nval levels (joint / bernoulli / dirichlet):")
    for name, levels in out["val_levels"].items():
        print(f"  {name:6s} " + " ".join(f"{levels[t]:7.3f}" for t in TERMS)
              + f"   share mae {out['val_share_mae'][name]:.4f}")
    print("\npinned minus free (positive = the slopes pay):")
    for term, boot in out["pinned_minus_free"].items():
        print(f"  {term:9s} {boot['mean']:+.3f} "
              f"[{boot['lo']:+.3f}, {boot['hi']:+.3f}] nats")
    print(f"\nfitted logit shift per {SIO2_SCALE:g} wt% SiO2: "
          + "  ".join(f"{g} {v:+.3f}" for g, v
                      in out["fitted_logit_shift_per_10wt_pct_sio2"].items()))

    out["checkpoint"] = str(args.checkpoint)
    out["benchmark"] = "eruption-v2"
    out["fit"] = f"full train roster ({len(train)} rocks)"
    out_dir = RESULTS / args.checkpoint.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "presence_slope.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
