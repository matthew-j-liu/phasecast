"""Score a joint checkpoint on the real-rock benchmark, zero-shot or adapted.

  python scripts/eval/real_rock.py --checkpoint checkpoints/joint_heads_supportmag_seed0/heads_best.pt

Runs the frozen encoder + trained ModeHead over the eruption benchmark
(`data/real_mineralogy/`) with nothing fit: the 21-phase vocabulary is
coarsened onto the 7 groups real modal data resolves, and the same hurdle NLL
the corpus eval reports is assembled against measured phenocryst modes.
Figures + `metrics.json` land in `runs/real_rock/<checkpoint dir name>/`.

`--fit-recalibrator` instead fits the sim-to-real recalibrator on the committed train
roster and reports it on the val slice, paired against the zero-shot baseline
on the same rocks. Encoder and heads stay frozen either way. The test slice
was spent 2026-08-03 on the sealed three-way scoring (journal), so it is not
an available reporting slice here -- any further test read is a disclosed
second look, made deliberately, not via this default path.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

import numpy as np

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, run_real_rock_eval, split_mask
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    fit_recalibrator,
    paired_bootstrap,
    score_recalibrator,
)


def report_recalibrator(ckpt, benchmark, latents, device, weight_decay: float) -> None:
    """Fit on the train roster, report on val, paired on the same rocks.

    Absolute levels on a ~160-rock slice carry roughly +/- 2 nats, so the
    number that means something is the paired difference, bootstrapped over
    units rather than rocks.
    """
    gp = build_group_predictions(ckpt.build_mode_head().to(device),
                                 benchmark, latents)
    scored = benchmark.clean & benchmark.has_magnitude
    train = gp.subset(split_mask(benchmark, "train") & scored)
    val = gp.subset(split_mask(benchmark, "val") & scored)
    print(f"train {len(train)} rocks / {len(np.unique(train.unit))} units   "
          f"val {len(val)} rocks / {len(np.unique(val.unit))} units")

    recalibrator = fit_recalibrator(train, tier="matrix", weight_decay=weight_decay)
    base, got = score_recalibrator(None, val), score_recalibrator(recalibrator, val)
    print(f"\nmean composition size {base['mean_composition_size']:.3f} "
          "(unchanged by the recalibrator -- same rocks, same truth)")
    for term in ("dirichlet", "joint", "bernoulli"):
        boot = paired_bootstrap(base[term], got[term], val.unit)
        print(f"{term:10s} {base[term].mean():7.3f} -> {got[term].mean():7.3f}   "
              f"recovered {boot['mean']:+.3f} "
              f"[{boot['lo']:+.3f}, {boot['hi']:+.3f}] nats")
    print(f"share MAE  {base['share_mae_mean']:7.4f} -> "
          f"{got['share_mae_mean']:7.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="JointHeadCheckpoint produced by scripts/training/joint_heads.py",
    )
    parser.add_argument(
        "--fit-recalibrator", action="store_true",
        help="fit the matrix recalibrator on the train roster and report on val (test is spent)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1.0,
        help="shrinkage; the value selected on val for the matrix tier",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    ckpt = JointHeadCheckpoint.load(checkpoint)
    device = pick_device()
    print(f"device:  {device}")
    print(f"heads:   {checkpoint}")
    print(f"encoder: {ckpt.encoder_checkpoint}")

    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    benchmark = load_benchmark()
    print(f"rocks:   {len(benchmark)} with chemistry + modes "
          f"({int(benchmark.clean.sum())} least-altered)")
    latents = bulk_latents(encoder, benchmark, device)
    if args.fit_recalibrator:
        report_recalibrator(ckpt, benchmark, latents, device, args.weight_decay)
        return
    run_real_rock_eval(ckpt, benchmark, latents, device=device,
                       tag=checkpoint.parent.name)


if __name__ == "__main__":
    main()
