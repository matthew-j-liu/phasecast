"""The honest 25-rock number: median over independent draws, both tiers.

The card run showed one lucky draw; this prices the draw. Fits matrix and
tailed corrections on 10 independent 25-rock draws from the train slice
(per-rock draws -- no unit labels exist), scores each on the never-spent
test slice, and reports the median beside the zero-shot, climatology, and
full-train references.

    conda run -n alignn_fix python explorations/sediment_transfer/run_fewshot_budget.py
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmineral.eval import real_rock
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import climatology, real_rock_nll
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    draw_units,
    fit_recalibrator,
    score_recalibrator,
)

from load_andrews import load_andrews, write_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_CHECKPOINT = "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt"
TERMS = ("bernoulli", "dirichlet", "joint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=REPO / DEFAULT_CHECKPOINT)
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--n-draws", type=int, default=10)
    args = parser.parse_args()

    benchmark, _info = load_andrews()
    split_path = write_split(benchmark, HERE / "split.csv")
    real_rock.SPLIT_PATH = split_path
    subset = pd.read_csv(split_path)["subset"].to_numpy()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)
    gp = build_group_predictions(ckpt.build_mode_head().to(device), benchmark, latents)
    train, test = (gp.subset(subset == n) for n in ("train", "test"))

    zero = score_recalibrator(None, test)
    null = climatology(benchmark)
    null_nll = real_rock_nll(null["p7"], null["alpha7"], benchmark.presence,
                             benchmark.magnitude, benchmark.numeric)
    clim = {t: float(null_nll[t][subset == "test"].mean()) for t in TERMS}
    wd = {t: v["weight_decay"]
          for t, v in json.loads((HERE / "results" / "recalibrate.json")
                                 .read_text())["tiers"].items()}

    out = {"budget": args.budget, "n_draws": args.n_draws,
           "draws": "per-rock (no unit labels exist)",
           "zero_shot": {t: float(zero[t].mean()) for t in TERMS},
           "climatology": clim, "tiers": {}}
    for tier in ("matrix", "tailed"):
        full = score_recalibrator(
            fit_recalibrator(train, tier=tier, weight_decay=wd[tier]), test)
        joints = []
        for seed in range(args.n_draws):
            rng = np.random.default_rng(seed)
            drawn = draw_units(train, args.budget, rng)
            fitted = fit_recalibrator(train.subset(drawn), tier=tier,
                                      weight_decay=wd[tier])
            scored = score_recalibrator(fitted, test)
            joints.append(float(scored["joint"].mean()))
        med = statistics.median(joints)
        full_joint = float(full["joint"].mean())
        frac = (out["zero_shot"]["joint"] - med) / (out["zero_shot"]["joint"] - full_joint)
        out["tiers"][tier] = {
            "weight_decay": wd[tier],
            "full_train": {t: float(full[t].mean()) for t in TERMS},
            "per_draw_joint": joints,
            "median_joint": med,
            "joint_range": [min(joints), max(joints)],
            "fraction_of_full_recovery": frac,
        }
        print(f"{tier:7s} zero-shot {out['zero_shot']['joint']:7.3f}  "
              f"median@{args.budget} {med:7.3f}  full {full_joint:7.3f}  "
              f"range [{min(joints):.3f}, {max(joints):.3f}]  "
              f"fraction {frac:.0%}")

    path = HERE / "results" / "fewshot_budget.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
