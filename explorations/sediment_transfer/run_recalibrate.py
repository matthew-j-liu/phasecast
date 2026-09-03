"""How much of the sediment failure is recoverable by rescaling?

The du Bray protocol on a benchmark whose test slice has never been spent:
weight decay chosen once on val, the layer fit on train, scored on test.  Both
tiers.  (A third, `tailed`, was scored here while it existed: its contamination
parameters were the part #63 suspected of encoding du Bray's visual-estimate
precision, and Andrews is XRD Rietveld; the tier was deleted 2026-09-03 once its
margin was traced to the amphibole pin.)
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmineral.eval import real_rock
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import climatology, real_rock_nll, unit_bootstrap
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions, fit_recalibrator, score_recalibrator,
    select_weight_decay,
)

from load_andrews import load_andrews, write_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_CHECKPOINT = "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt"
TIERS = ("vector", "matrix")
TERMS = ("bernoulli", "dirichlet", "joint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    benchmark, _info = load_andrews()
    split_path = write_split(benchmark, HERE / "split.csv")
    real_rock.SPLIT_PATH = split_path
    subset = pd.read_csv(split_path)["subset"].to_numpy()
    masks = {name: subset == name for name in ("train", "val", "test")}

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)
    data = build_group_predictions(
        ckpt.build_mode_head().to(device), benchmark, latents
    )

    train, val, test = (data.subset(masks[n]) for n in ("train", "val", "test"))
    print(f"andrews: train {len(train)}  val {len(val)}  test {len(test)}")

    null = climatology(benchmark)
    null_nll = real_rock_nll(null["p7"], null["alpha7"], benchmark.presence,
                             benchmark.magnitude, benchmark.numeric)
    zero = score_recalibrator(None, test)
    print(f"\ntest slice, {len(test)} rocks")
    print(f"  {'arm':22s} {'presence':>9s} {'comp':>9s} {'joint':>9s}")
    print(f"  {'zero-shot':22s} "
          + " ".join(f"{zero[t].mean():9.3f}" for t in TERMS))
    print(f"  {'climatology':22s} "
          + " ".join(f"{null_nll[t][masks['test']].mean():9.3f}" for t in TERMS))

    out = {"n": {k: int(v.sum()) for k, v in masks.items()},
           "zero_shot": {t: float(zero[t].mean()) for t in TERMS},
           "climatology": {t: float(null_nll[t][masks["test"]].mean())
                           for t in TERMS},
           "tiers": {}}

    for tier in TIERS:
        wd, _records = select_weight_decay(
            train, val, fit=partial(fit_recalibrator, tier=tier),
        )
        fitted = fit_recalibrator(train, tier=tier, weight_decay=wd)
        scored = score_recalibrator(fitted, test)
        delta = scored["joint"] - zero["joint"]

        def stat(rows, _d=delta):
            return np.array([_d[rows].mean()])

        lo, hi = unit_bootstrap(stat, np.arange(len(test)).astype(str))
        label = f"{tier} (wd={wd:g})"
        print(f"  {label:22s} "
              + " ".join(f"{scored[t].mean():9.3f}" for t in TERMS)
              + f"   delta {delta.mean():+7.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}]")
        out["tiers"][tier] = {
            "weight_decay": float(wd),
            **{t: float(scored[t].mean()) for t in TERMS},
            "joint_delta_vs_zero_shot": [float(delta.mean()),
                                         float(lo[0]), float(hi[0])],
        }

    best = min(TIERS, key=lambda t: out["tiers"][t]["joint"])
    gap = out["tiers"][best]["joint"] - out["climatology"]["joint"]
    side = "ahead of" if gap < 0 else "behind"
    print(f"\nbest tier {best}: joint {out['tiers'][best]['joint']:.3f} vs "
          f"climatology {out['climatology']['joint']:.3f} -> "
          f"{abs(gap):.3f} nats {side} it")
    out["best_tier"] = best
    # name matches the key build_numbers.py reads for sedAhead; renaming it
    # silently drops that macro into the TBD path on the next regeneration
    out["best_vs_climatology"] = float(gap)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "recalibrate.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
