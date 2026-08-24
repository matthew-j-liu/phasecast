"""The root results/ headline figures, redrawn after the recalibrator fit.

Mirrors the four canvases of scripts/view/real_rock_card.py -- cohort card,
magnitude ratio, NLL distribution, presence Brier -- for one checkpoint, with
the recalibrator fit on the full train roster and every figure drawn on the
val slice only. That is the residual-decomposition protocol, so running this
for all three checkpoints puts them on identical post-correction footing.
The test slice (spent 2026-08-03) is not touched here.

    conda run -n mineral python explorations/real_rock_fewshot/run_adapted_figures.py \
        --checkpoint checkpoints/joint_heads_regen_seed0/heads_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import (
    climatology,
    coarsen_alpha,
    coarsen_support,
    load_benchmark,
    plot_cohort_card,
    plot_magnitude_ratio,
    plot_nll_distribution,
    plot_presence_brier,
    real_rock_nll,
    split_mask,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    fit_recalibrator,
    lift_group_presence_to_phases,
    score_recalibrator,
)

RESULTS = Path(__file__).resolve().parent / "results"

# the values selected on val for run_curve; see run_curve.WEIGHT_DECAY
WEIGHT_DECAY = {"vector": 0.1, "matrix": 1.0, "mlp": 3.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    # single-Dirichlet tiers only: the interval and mean panels here read the
    # Beta marginal directly, which the tailed tier's mixture output breaks
    parser.add_argument("--tier", choices=("vector", "matrix", "mlp"),
                        default="matrix")
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder.to(device), benchmark, device)
    mode_head = ckpt.build_mode_head().to(device)
    gp = build_group_predictions(mode_head, benchmark, latents)

    scored = benchmark.clean & benchmark.has_magnitude
    train_mask = split_mask(benchmark, "train") & scored
    val_mask = split_mask(benchmark, "val") & scored

    recalibrator = fit_recalibrator(gp.subset(train_mask), tier=args.tier,
                                    weight_decay=WEIGHT_DECAY[args.tier])

    val = gp.subset(val_mask)
    before, after = score_recalibrator(None, val), score_recalibrator(recalibrator, val)
    print(f"val slice: joint {before['joint'].mean():.3f} -> "
          f"{after['joint'].mean():.3f}   "
          f"dirichlet {before['dirichlet'].mean():.3f} -> "
          f"{after['dirichlet'].mean():.3f}")

    with torch.no_grad():
        adj_logits, adj_alpha, _ = recalibrator(val.logits7, val.alpha7, val.sio2)
    p7 = torch.sigmoid(adj_logits).numpy()
    numeric = benchmark.numeric[val_mask].astype(np.float64)
    forced_share = conditional_mean(adj_alpha.numpy(), numeric)

    # the model-own-assemblage column: corrected group presence lifted back
    # onto 21 phases, the head rerun on that support, and the resulting alphas
    # sent through the same magnitude correction
    with torch.no_grad():
        p21 = torch.sigmoid(mode_head.presence_logits(latents)).cpu().numpy()
    p21_adj = lift_group_presence_to_phases(recalibrator, gp.logits7, p21, gp.sio2)
    support21 = (p21_adj > 0.5).astype(np.float64)
    with torch.no_grad():
        alpha21 = mode_head.magnitude_alpha(
            latents, torch.as_tensor(support21, dtype=latents.dtype,
                                     device=latents.device),
        ).cpu().numpy()
    alpha7_model, _ = coarsen_alpha(alpha21, support21)
    model_support7 = coarsen_support(support21)
    with torch.no_grad():
        _, adj_alpha_model, _ = recalibrator(
            gp.logits7, torch.as_tensor(alpha7_model, dtype=gp.alpha7.dtype),
            gp.sio2,
        )
    model_share = conditional_mean(adj_alpha_model.numpy()[val_mask],
                                   model_support7[val_mask])

    null = climatology(benchmark)
    null_nll = real_rock_nll(null["p7"], null["alpha7"], benchmark.presence,
                             benchmark.magnitude, benchmark.numeric)

    out_dir = RESULTS / args.checkpoint.parent.name / f"figures_{args.tier}_fulltrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train, n_val = int(train_mask.sum()), int(val_mask.sum())
    subtitle = (f"{args.tier} correction fit on {n_train} train rocks, "
                f"drawn on {n_val} val rocks; test not used")

    plot_cohort_card(
        p7, forced_share, model_share, model_support7[val_mask],
        benchmark.subset(val_mask), out_dir / "cohort_card.png",
    )
    plot_magnitude_ratio(
        forced_share, benchmark.magnitude[val_mask],
        benchmark.numeric[val_mask], out_dir / "magnitude_ratio.png", subtitle,
    )
    plot_nll_distribution(
        {k: after[k] for k in ("bernoulli", "dirichlet", "joint")},
        {k: np.asarray(v)[val_mask] for k, v in null_nll.items()},
        out_dir / "nll_distribution.png", subtitle,
    )
    plot_presence_brier(
        p7, null["p7"][val_mask], benchmark.presence[val_mask],
        benchmark.unit[val_mask], out_dir / "presence_brier.png", subtitle,
    )
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
