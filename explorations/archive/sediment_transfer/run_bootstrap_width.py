"""How much the per-rock bootstrap understates, measured on eruption-v2.

Andrews has no locality metadata, so its intervals can only be the per-rock
kind. This prices that substitution on the one benchmark where both are
available: same statistic, same rocks, real stratigraphic units against one
unit per rock. The ratio is printed on the two Andrews figures that draw bars.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import numpy as np

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, predict, unit_bootstrap
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_CHECKPOINT = "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt"


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
    y = (benchmark.presence[clean] > 0).astype(float)
    unit = benchmark.unit[clean]
    squared = (pred["p7"][clean] - y) ** 2

    def stat(rows):
        return np.append(squared[rows].mean(0), squared[rows].mean(0).mean())

    clustered = unit_bootstrap(stat, unit)
    per_rock = unit_bootstrap(stat, np.arange(len(y)).astype(str))
    width_clustered = clustered[1] - clustered[0]
    width_per_rock = per_rock[1] - per_rock[0]
    ratio = width_clustered / width_per_rock

    print(f"eruption-v2: {len(y)} rocks in {len(np.unique(unit))} units")
    print("presence Brier interval width, clustered vs per-rock:")
    for label, c, p, r in zip([*GROUPS, "mean"], width_clustered,
                              width_per_rock, ratio):
        print(f"  {label:5s} clustered {c:.4f}  per-rock {p:.4f}  x{r:.2f}")
    print(f"\noverall: the per-rock bootstrap is {ratio.mean():.2f}x too narrow "
          f"(median {np.median(ratio):.2f}x)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "bootstrap_width.json"
    out.write_text(json.dumps({
        "benchmark": "eruption-v2",
        "n_rocks": int(len(y)),
        "n_units": int(len(np.unique(unit))),
        "statistic": "presence Brier, per group and mean",
        "width_ratio_clustered_over_per_rock": dict(
            zip([*GROUPS, "mean"], [float(v) for v in ratio])
        ),
        "summary_ratio_mean": float(ratio.mean()),
        "summary_ratio_median": float(np.median(ratio)),
    }, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
