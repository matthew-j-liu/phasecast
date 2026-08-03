"""Paired three-way checkpoint contrast under the full-train matrix correction.

The residual decomposition and the adapted figures give each checkpoint a
post-correction level on the val slice, but levels on a ~110-rock slice carry
+/- nats of unit noise -- only paired, unit-clustered contrasts on identical
rocks are decision-grade. This scores every checkpoint under the identical
protocol (matrix tier fit on the full train roster), pairs them rock-for-rock,
and bootstraps over stratigraphic units.

Scores the val slice by default. The sealed test scoring was taken once,
2026-08-03 (0.478 / 0.769 / 1.371 joint NLL for supportmag / regen / tuned);
`--score-test` re-reads that slice, which is a disclosed second look -- pass
it deliberately and say so wherever the number lands.

    conda run -n alignn_fix python explorations/real_rock_fewshot/run_checkpoint_contrast.py \
        checkpoints/joint_heads_supportmag_seed0/heads_best.pt \
        checkpoints/joint_heads_regen_seed0/heads_best.pt \
        checkpoints/joint_heads_regen_tuned_seed0/heads_best.pt
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, split_mask
from cnmineral.models.real_rock_recalibrator import TIERS
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
WEIGHT_DECAY = {"vector": 0.1, "matrix": 1.0, "mlp": 3.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--tier", choices=TIERS, default="matrix")
    parser.add_argument("--score-test", action="store_true",
                        help="also score the test slice (spent 2026-08-03; "
                             "a re-read is a disclosed second look)")
    args = parser.parse_args()

    benchmark = load_benchmark()
    device = pick_device()
    scored = benchmark.clean & benchmark.has_magnitude
    train_mask = split_mask(benchmark, "train") & scored
    slices = {"val": split_mask(benchmark, "val") & scored}
    if args.score_test:
        slices["test"] = split_mask(benchmark, "test") & scored

    per_model: dict[str, dict] = {}
    for path in args.checkpoints:
        tag = path.parent.name
        ckpt = JointHeadCheckpoint.load(path)
        encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
        latents = bulk_latents(encoder.to(device), benchmark, device)
        gp = build_group_predictions(ckpt.build_mode_head().to(device),
                                     benchmark, latents)
        recalibrator = fit_recalibrator(gp.subset(train_mask), tier=args.tier,
                                        weight_decay=WEIGHT_DECAY[args.tier])
        per_model[tag] = {
            name: {
                "zero_shot": score_recalibrator(None, gp.subset(mask)),
                "adapted": score_recalibrator(recalibrator, gp.subset(mask)),
                "unit": benchmark.unit[mask],
            }
            for name, mask in slices.items()
        }
        for name in slices:
            got = per_model[tag][name]
            print(f"{tag:34s} {name:4s} joint "
                  f"{got['zero_shot']['joint'].mean():7.3f} -> "
                  f"{got['adapted']['joint'].mean():6.3f}")

    contrasts: dict[str, dict] = {}
    for name in slices:
        contrasts[name] = {}
        for a, b in itertools.combinations(per_model, 2):
            boot = paired_bootstrap(
                per_model[a][name]["adapted"]["joint"],
                per_model[b][name]["adapted"]["joint"],
                per_model[a][name]["unit"],
            )
            contrasts[name][f"{a} - {b}"] = boot
            print(f"{name:4s} adapted joint, {a} minus {b}: "
                  f"{boot['mean']:+.3f} [{boot['lo']:+.3f}, {boot['hi']:+.3f}]")

    out = {
        "benchmark": "eruption-v2",
        "tier": args.tier,
        "weight_decay": WEIGHT_DECAY[args.tier],
        "fit": f"full train roster ({int(train_mask.sum())} rocks)",
        "levels": {
            tag: {
                name: {
                    "zero_shot_joint": float(v["zero_shot"]["joint"].mean()),
                    "adapted_joint": float(v["adapted"]["joint"].mean()),
                    "adapted_dirichlet": float(v["adapted"]["dirichlet"].mean()),
                    "adapted_bernoulli": float(v["adapted"]["bernoulli"].mean()),
                    "n_rocks": int(len(v["unit"])),
                }
                for name, v in by_slice.items()
            }
            for tag, by_slice in per_model.items()
        },
        "paired_contrasts_adapted_joint": contrasts,
    }
    out_path = RESULTS / f"checkpoint_contrast_{args.tier}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
