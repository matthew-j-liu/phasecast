"""Does an encoder that saw sedimentary chemistry predict sediments any better?

The igneous-only encoder against the all-rock one, both through their matched
tuned heads, on the Andrews sediments. Paired per rock, since both arms score
the same 605. The climatology null is checkpoint-independent, so it is computed
once and quoted for scale.

This is an arm contrast, not an encoder-only ablation: each arm carries the head
dials its own HPO chose, which is the matched-protocol comparison cascade 2
committed to.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval import real_rock
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean, mixture_conditional_mean
from cnmineral.eval.real_rock import (
    _unit_bootstrap, climatology, predict, real_rock_nll,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint

from load_andrews import load_andrews, write_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARMS = {
    "igneous-tuned": "checkpoints/joint_heads_igneous_tuned_seed0/heads_best.pt",
    "allrock-tuned": "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt",
}
TERMS = ("bernoulli", "dirichlet", "joint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    benchmark, _info = load_andrews()
    real_rock.SPLIT_PATH = write_split(benchmark, HERE / "split.csv")
    device = pick_device()
    clean = benchmark.clean & benchmark.has_magnitude

    null = climatology(benchmark)
    null_nll = real_rock_nll(null["p7"], null["alpha7"], benchmark.presence,
                             benchmark.magnitude, benchmark.numeric)

    results = {}
    for name, path in ARMS.items():
        ckpt = JointHeadCheckpoint.load(REPO / path)
        encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
        latents = bulk_latents(encoder.to(device), benchmark, device)
        pred = predict(ckpt.build_mode_head().to(device), latents, benchmark)
        log_w = pred.get("log_w")
        nll = real_rock_nll(pred["p7"], pred["alpha7"], benchmark.presence,
                            benchmark.magnitude, benchmark.numeric, log_w=log_w)
        numeric = benchmark.numeric.astype(np.float64)
        share = (conditional_mean(pred["alpha7"], numeric) if log_w is None
                 else mixture_conditional_mean(pred["alpha7"], log_w, numeric))
        results[name] = {"nll": nll, "share": share,
                         "encoder": ckpt.encoder_checkpoint}

    print(f"Andrews, {int(clean.sum())} sedimentary rocks\n")
    print(f"{'arm':16s} {'encoder':34s} {'presence':>9s} {'comp':>9s} {'joint':>9s}")
    for name, r in results.items():
        enc = Path(r["encoder"]).parent.name
        print(f"{name:16s} {enc:34s} "
              + " ".join(f"{r['nll'][t][clean].mean():9.3f}" for t in TERMS))
    print(f"{'climatology':16s} {'-':34s} "
          + " ".join(f"{null_nll[t][clean].mean():9.3f}" for t in TERMS))

    a, b = results["igneous-tuned"], results["allrock-tuned"]
    print("\npaired per-rock difference, all-rock minus igneous "
          "(negative = all-rock better):")
    deltas = {}
    for term in TERMS:
        d = b["nll"][term][clean] - a["nll"][term][clean]

        def stat(rows, _d=d):
            return np.array([_d[rows].mean()])

        lo, hi = _unit_bootstrap(stat, np.arange(int(clean.sum())).astype(str))
        deltas[term] = [float(d.mean()), float(lo[0]), float(hi[0])]
        print(f"  {term:10s} {d.mean():+7.3f}  [{lo[0]:+.3f}, {hi[0]:+.3f}]")

    print("\nmedian predicted/measured share ratio per group:")
    print(f"  {'group':6s} {'igneous':>9s} {'allrock':>9s}")
    ratios = {}
    for i, group in enumerate(GROUPS):
        rows = clean & benchmark.numeric[:, i] & (benchmark.magnitude[:, i] > 0)
        if rows.sum() == 0:
            continue
        truth = benchmark.magnitude[rows, i]
        ratios[group] = {n: float(np.median(r["share"][rows, i] / truth))
                         for n, r in results.items()}
        print(f"  {group:6s} {ratios[group]['igneous-tuned']:9.2f} "
              f"{ratios[group]['allrock-tuned']:9.2f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "encoder_contrast.json"
    out.write_text(json.dumps({
        "benchmark": "andrews_xrd",
        "n_rocks": int(clean.sum()),
        "arms": {n: {"encoder": r["encoder"],
                     **{t: float(r["nll"][t][clean].mean()) for t in TERMS}}
                 for n, r in results.items()},
        "climatology": {t: float(null_nll[t][clean].mean()) for t in TERMS},
        "paired_delta_allrock_minus_igneous": deltas,
        "median_share_ratio": ratios,
        "bootstrap": "per-rock (no units available)",
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
