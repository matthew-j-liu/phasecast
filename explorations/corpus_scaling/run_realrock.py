"""Score the corpus-scaling ladder on the real-rock benchmark (#60).

  python explorations/corpus_scaling/run_realrock.py
  python explorations/corpus_scaling/run_realrock.py --tier tailed
  python explorations/corpus_scaling/run_realrock.py --tags joint_heads_allrock_tuned_seed0

Run from the repo root after `run_ladder.py` has trained the
`corpus_scaling_allrock_*` points. Each point is scored zero-shot on the
eruption-v2 headline slice (640 rocks) and recalibrated on the val slice,
with the recalibrator fit on the full train roster -- the identical
protocol for every point, so paired differences read as corpus-size effects.
The test slice was spent 2026-07-31 and is never scored here. Point-to-point
comparisons are paired unit-clustered bootstraps plus per-rock slopes of
adapted NLL against log2(bulks); levels alone on a ~110-rock slice carry
more unit noise than the effects being measured.

Writes `results/scaling_realrock.json` and `.png`; a non-default `--tier`
suffixes both filenames (`scaling_realrock_tailed.*`) so each instrument
keeps its own record -- zero-shot numbers are tier-independent and are
reused from `runs/real_rock/<tag>/metrics.json` when present. `--tags`
restricts to named checkpoints and prints levels without writing (the
production-only smoke run).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cnmineral.eval.joint_heads import run_joint_eval
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

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
CHECKPOINTS = REPO / "checkpoints"

# The 10000 point is the production train itself (identical config; #27
# validated that the roster-restricted path reproduces the unrestricted
# train), so the production checkpoint is scored directly, not retrained.
LADDER = {
    625: "corpus_scaling_allrock_n625",
    1250: "corpus_scaling_allrock_n1250",
    2500: "corpus_scaling_allrock_n2500",
    5000: "corpus_scaling_allrock_n5000",
    10000: "joint_heads_allrock_tuned_seed0",
}
SEED_REPEAT_N = 2500
SEED_REPEAT_TAG = "corpus_scaling_allrock_n2500_seed1"
TOP_OF_LADDER = (2500, 5000, 10000)

# values selected on val per tier; see run_curve.WEIGHT_DECAY
WEIGHT_DECAY = {"matrix": 1.0, "tailed": 1.0}


def assert_nested_rosters() -> None:
    rosters = {}
    for n, tag in LADDER.items():
        path = CHECKPOINTS / tag / "train_bulk_ids.csv"
        if path.exists():
            rosters[n] = set(pd.read_csv(path)["sample_id"])
            assert len(rosters[n]) == n, f"{tag}: roster has {len(rosters[n])} ids"
    for small, large in itertools.pairwise(sorted(rosters)):
        assert rosters[small] < rosters[large], \
            f"roster n{small} is not a subset of n{large}"
    repeat_path = CHECKPOINTS / SEED_REPEAT_TAG / "train_bulk_ids.csv"
    if repeat_path.exists() and SEED_REPEAT_N in rosters:
        repeat = set(pd.read_csv(repeat_path)["sample_id"])
        assert repeat == rosters[SEED_REPEAT_N], "seed repeat roster differs"


def corpus_val_metrics(tag: str, ckpt_path: Path) -> dict:
    metrics_path = REPO / "runs" / "mode_head" / tag / "metrics.json"
    if not metrics_path.exists():
        run_joint_eval(ckpt_path, heads=("mode",))
    metrics = json.loads(metrics_path.read_text())
    return {k: metrics[k] for k in ("joint_nll", "bernoulli_nll", "dirichlet_nll")}


def score_point(tag: str, benchmark, latents, masks, device, tier: str) -> dict:
    ckpt_path = CHECKPOINTS / tag / "heads_best.pt"
    ckpt = JointHeadCheckpoint.load(ckpt_path)
    zero_shot_path = REPO / "runs" / "real_rock" / tag / "metrics.json"
    if zero_shot_path.exists():
        metrics = json.loads(zero_shot_path.read_text())
    else:
        metrics = run_real_rock_eval(ckpt, benchmark, latents, device=device,
                                     tag=tag)

    gp = build_group_predictions(ckpt.build_mode_head().to(device),
                                 benchmark, latents)
    recalibrator = fit_recalibrator(gp.subset(masks["train"]), tier=tier,
                                    weight_decay=WEIGHT_DECAY[tier])
    val = gp.subset(masks["val"])
    zero_val = score_recalibrator(None, val)
    adapted = score_recalibrator(recalibrator, val)

    print(f"{tag:34s} zero-shot 640 {metrics['headline']['joint_nll']:7.3f}   "
          f"val {zero_val['joint'].mean():7.3f} -> adapted "
          f"{adapted['joint'].mean():6.3f}")
    return {
        "encoder_checkpoint": ckpt.encoder_checkpoint,
        "corpus_val": corpus_val_metrics(tag, ckpt_path),
        "zero_shot_640": metrics["headline"],
        "censored_prob": metrics["censored"]["mean_prob_below_bound"],
        "zero_shot_val_joint": float(zero_val["joint"].mean()),
        "adapted_val": {
            "joint": float(adapted["joint"].mean()),
            "bernoulli": float(adapted["bernoulli"].mean()),
            "dirichlet": float(adapted["dirichlet"].mean()),
            "share_mae_mean": adapted["share_mae_mean"],
        },
        "adapted_val_joint_per_rock": adapted["joint"],
    }


def clustered_slope(per_rock: dict[int, np.ndarray], unit: np.ndarray,
                    ns: tuple[int, ...]) -> dict:
    x = np.log2(np.array(ns, dtype=float))
    x = x - x.mean()
    stack = np.stack([per_rock[n] for n in ns], axis=1)
    slopes = stack @ x / (x ** 2).sum()
    boot = paired_bootstrap(slopes, np.zeros_like(slopes), unit)
    return {"n_bulks": list(ns), **boot}


def make_figure(points: list[dict], contrasts: dict, slopes: dict,
                repeat: dict | None, out_path: Path, tier: str) -> None:
    import matplotlib.pyplot as plt

    ladder = [p for p in points if p["seed"] == 0]
    ns = [p["n_bulks"] for p in ladder]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    panels = [
        ("corpus val joint NLL", [p["corpus_val"]["joint_nll"] for p in ladder]),
        ("zero-shot 640 joint NLL", [p["zero_shot_640"]["joint_nll"] for p in ladder]),
        (f"adapted val joint NLL ({tier})", [p["adapted_val"]["joint"] for p in ladder]),
    ]
    full_level = ladder[-1]["adapted_val"]["joint"]
    for ax, (label, values) in zip(axes, panels):
        ax.plot(ns, values, "o-", color="black")
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("train bulks")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    # error bars on the adapted panel: each point's paired contrast against
    # the full-N point, re-anchored at its own level (paired mean of deltas
    # equals the level difference exactly, same rocks)
    ax = axes[2]
    xlim = ax.get_xlim()
    boots = [contrasts[f"n{p['n_bulks']} - n10000"] for p in ladder[:-1]]
    ax.errorbar([p["n_bulks"] for p in ladder[:-1]],
                [full_level + b["mean"] for b in boots],
                yerr=[[b["mean"] - b["lo"] for b in boots],
                      [b["hi"] - b["mean"] for b in boots]],
                fmt="none", ecolor="0.5", capsize=3)
    ax.set_xlim(xlim)  # errorbar on a log axis wrecks the autoscale window
    if repeat is not None:
        ax.plot(SEED_REPEAT_N, repeat["adapted_val"]["joint"], "s",
                mfc="none", color="black", label="seed 1")
        ax.legend(fontsize=8)
    lines = [
        f"slope {'/'.join(str(n) for n in s['n_bulks'])}: "
        f"{s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] nats/doubling"
        for s in (slopes["full_ladder"], slopes["top_of_ladder"])
    ]
    ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=7,
            va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tags", default=None,
        help="comma-separated checkpoint tags to score alone (prints levels, "
             "writes nothing; the pre-ladder smoke run)",
    )
    parser.add_argument(
        "--tier", choices=sorted(WEIGHT_DECAY), default="matrix",
        help="recalibrator tier; non-default suffixes the output filenames",
    )
    args = parser.parse_args()

    smoke_tags = args.tags.split(",") if args.tags else None
    if smoke_tags is None:
        assert_nested_rosters()

    benchmark = load_benchmark()
    device = pick_device()
    scored = benchmark.clean & benchmark.has_magnitude
    masks = {"train": split_mask(benchmark, "train") & scored,
             "val": split_mask(benchmark, "val") & scored}
    val_unit = benchmark.unit[masks["val"]]

    todo = smoke_tags if smoke_tags is not None else (
        [tag for tag in LADDER.values()] + [SEED_REPEAT_TAG])
    encoder_stamps = {
        tag: JointHeadCheckpoint.load(CHECKPOINTS / tag / "heads_best.pt"
                                      ).encoder_checkpoint
        for tag in todo
    }
    assert len(set(encoder_stamps.values())) == 1, \
        f"points disagree on the encoder: {encoder_stamps}"
    encoder, _ = load_encoder_from_checkpoint(next(iter(encoder_stamps.values())))
    latents = bulk_latents(encoder.to(device), benchmark, device)

    results = {tag: score_point(tag, benchmark, latents, masks, device, args.tier)
               for tag in todo}
    if smoke_tags is not None:
        return

    n_by_tag = {tag: n for n, tag in LADDER.items()}
    per_rock = {n_by_tag[tag]: r["adapted_val_joint_per_rock"]
                for tag, r in results.items() if tag != SEED_REPEAT_TAG}

    contrasts = {}
    for a, b in itertools.combinations(sorted(per_rock), 2):
        boot = paired_bootstrap(per_rock[a], per_rock[b], val_unit)
        contrasts[f"n{a} - n{b}"] = boot
        print(f"adapted val joint, n{a} minus n{b}: "
              f"{boot['mean']:+.3f} [{boot['lo']:+.3f}, {boot['hi']:+.3f}]")

    slopes = {
        "full_ladder": clustered_slope(per_rock, val_unit, tuple(sorted(per_rock))),
        "top_of_ladder": clustered_slope(per_rock, val_unit, TOP_OF_LADDER),
    }
    for name, s in slopes.items():
        print(f"slope {name}: {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] "
              f"nats/doubling")

    repeat_boot = paired_bootstrap(
        per_rock[SEED_REPEAT_N],
        results[SEED_REPEAT_TAG]["adapted_val_joint_per_rock"], val_unit)
    print(f"seed repeat at n{SEED_REPEAT_N}, seed0 minus seed1: "
          f"{repeat_boot['mean']:+.3f} [{repeat_boot['lo']:+.3f}, "
          f"{repeat_boot['hi']:+.3f}]")

    points = []
    for tag, r in results.items():
        entry = {"tag": tag,
                 "n_bulks": SEED_REPEAT_N if tag == SEED_REPEAT_TAG else n_by_tag[tag],
                 "seed": 1 if tag == SEED_REPEAT_TAG else 0}
        entry.update({k: v for k, v in r.items()
                      if k not in ("adapted_val_joint_per_rock", "encoder_checkpoint")})
        points.append(entry)
    points.sort(key=lambda p: (p["n_bulks"], p["seed"]))

    out = {
        "instrument": {
            "benchmark": "eruption-v2",
            "zero_shot_slice": "headline 640 (clean & has_magnitude)",
            "recalibration": {
                "tier": args.tier,
                "weight_decay": WEIGHT_DECAY[args.tier],
                "fit": f"train roster ({int(masks['train'].sum())} rocks)",
            },
            "reporting_slice": f"val ({int(masks['val'].sum())} rocks / "
                               f"{len(np.unique(val_unit))} units); test spent "
                               "2026-07-31, never scored here",
            "stack": {
                "dials": "joint_heads_allrock_hpo trial 32, 120 epochs",
                "encoder": next(iter(encoder_stamps.values())),
                "corpus": "old broad-draw data/corpus/train, 24 draws/bulk, "
                          "rungs 50+30",
            },
            "corpus_side": "valtest128-broad val, teacher-forced hurdle "
                           "(runs/mode_head)",
            "date": date.today().isoformat(),
        },
        "points": points,
        "paired_contrasts_adapted_val_joint": contrasts,
        "slopes_adapted_val_joint": slopes,
        "seed_repeat": {"n_bulks": SEED_REPEAT_N,
                        "contrast_seed0_minus_seed1": repeat_boot},
    }
    RESULTS.mkdir(exist_ok=True)
    suffix = "" if args.tier == "matrix" else f"_{args.tier}"
    out_path = RESULTS / f"scaling_realrock{suffix}.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"saved {out_path}")

    fig_path = RESULTS / f"scaling_realrock{suffix}.png"
    make_figure(points, contrasts, slopes,
                results.get(SEED_REPEAT_TAG), fig_path, args.tier)
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
