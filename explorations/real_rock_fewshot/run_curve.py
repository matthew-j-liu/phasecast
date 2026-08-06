"""How many labelled real rocks close the sim-to-real gap? (#52)

Fits the frozen-head recalibrator on real modal labels drawn as whole stratigraphic
units, and scores it on a fixed held-out slice (--eval-split; val since the
test slice was spent 2026-08-03), over a sweep of labelled-rock budgets.  The deliverable is the curve's shape: a point estimate on a
~160-rock slice carries roughly +/- 2 nats, but the shape survives that, and
it is what says whether harvesting more eruptions is worth the transcription.

    conda run -n alignn_fix python explorations/real_rock_fewshot/run_curve.py \
        --checkpoint checkpoints/joint_heads_supportmag_seed0/heads_best.pt

Weight decay is chosen once per tier on the fixed val slice before the curve
runs (--select-wd), so every drawn rock on the x-axis is a fitted rock.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, split_mask
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    draw_units,
    fit_recalibrator,
    paired_bootstrap,
    per_group_offsets,
    score_recalibrator,
    select_weight_decay,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"

# Chosen once on the val slice by `select_weight_decay`; see README. Recorded
# rather than re-selected on every run so the curve is reproducible and the
# val slice's contribution stays one scalar per tier.
WEIGHT_DECAY: dict[str, float] = {
    "vector": 0.1, "matrix": 1.0, "mlp": 3.0, "tailed": 1.0,
}

# The curve compares vector, matrix and tailed; mlp is off the curve because
# its capacity never repaid itself at any labelled budget (#52).
CURVE_TIERS: tuple[str, ...] = ("vector", "matrix", "tailed")

BUDGETS: tuple[int, ...] = (25, 50, 100, 200, 400)
TIER_COLOR = {"vector": "tab:blue", "matrix": "tab:orange", "mlp": "tab:green",
              "tailed": "tab:red"}


def run_curve(train, test, *, budgets, n_draws, weight_decay, seed=0):
    """One fit per (tier, budget, draw), each scored on the fixed eval slice."""
    baseline = score_recalibrator(None, test)
    rows = []
    for tier in CURVE_TIERS:
        for budget in budgets:
            # reseeded per budget rather than carried across tiers, so all
            # three tiers see the identical draws and the tier comparison is
            # paired -- at 25 rocks which units you drew dominates everything.
            rng = np.random.default_rng(seed + budget)
            for draw in range(n_draws):
                mask = draw_units(train, budget, rng)
                if not mask.any():
                    continue
                drawn = train.subset(mask)
                recalibrator = fit_recalibrator(
                    drawn, tier=tier, weight_decay=weight_decay[tier], seed=draw,
                )
                got = score_recalibrator(recalibrator, test)
                boot = paired_bootstrap(
                    baseline["dirichlet"], got["dirichlet"], test.unit, seed=draw,
                )
                rows.append({
                    "tier": tier,
                    "budget": budget,
                    "draw": draw,
                    "n_rocks": int(len(drawn)),
                    "n_units": int(len(np.unique(drawn.unit))),
                    "dirichlet": float(got["dirichlet"].mean()),
                    "joint": float(got["joint"].mean()),
                    "bernoulli": float(got["bernoulli"].mean()),
                    "share_mae": got["share_mae_mean"],
                    "recovered_dirichlet": boot["mean"],
                    "recovered_lo": boot["lo"],
                    "recovered_hi": boot["hi"],
                    "per_group_share_mae": [
                        None if np.isnan(v) else float(v) for v in got["share_mae"]
                    ],
                })
            print(f"  {tier:9s} budget {budget:4d}: "
                  f"{len([r for r in rows if r['tier'] == tier and r['budget'] == budget])}"
                  f" draws done")
    return baseline, rows


def _band(ax, rows, tier, key, color):
    """Median with a 10-90 spread band, x = realized labelled rocks."""
    budgets = sorted({r["budget"] for r in rows if r["tier"] == tier})
    x, med, lo, hi = [], [], [], []
    for budget in budgets:
        vals = [r[key] for r in rows if r["tier"] == tier and r["budget"] == budget]
        n = [r["n_rocks"] for r in rows if r["tier"] == tier and r["budget"] == budget]
        if not vals:
            continue
        x.append(np.mean(n))
        med.append(np.median(vals))
        lo.append(np.quantile(vals, 0.1))
        hi.append(np.quantile(vals, 0.9))
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, med, "o-", color=color, ms=4, label=tier)
    return x, med


def _rock_ticks(ax, rows) -> None:
    """Tick at the realized budgets, since a log axis otherwise labels only 100."""
    ticks = sorted({int(round(np.mean(
        [r["n_rocks"] for r in rows if r["budget"] == b]
    ))) for b in {r["budget"] for r in rows}})
    ax.set_xticks(ticks, [str(t) for t in ticks], fontsize=8)
    ax.minorticks_off()


def plot_curve(baseline, rows, out_path: Path, subtitle: str,
               slice_label: str = "test") -> None:
    """The two headline panels: Dirichlet term and per-group share error."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for tier in CURVE_TIERS:
        _band(ax1, rows, tier, "dirichlet", TIER_COLOR[tier])
    ax1.axhline(baseline["dirichlet"].mean(), lw=1.0, color="0.35", ls="--")
    ax1.annotate("zero-shot", (0.02, baseline["dirichlet"].mean()),
                 xycoords=("axes fraction", "data"), va="bottom", fontsize=8,
                 color="0.35")
    ax1.set_xscale("log")
    ax1.set_xlabel("labelled real rocks (drawn as whole units)", fontsize=9)
    ax1.set_ylabel(f"Dirichlet NLL on the {slice_label} slice (nats)", fontsize=9)
    ax1.set_title("mode magnitude: the headline", fontsize=10)
    ax1.legend(fontsize=8)
    _rock_ticks(ax1, rows)

    for tier in CURVE_TIERS:
        _band(ax2, rows, tier, "share_mae", TIER_COLOR[tier])
    ax2.axhline(baseline["share_mae_mean"], lw=1.0, color="0.35", ls="--")
    ax2.set_xscale("log")
    ax2.set_xlabel("labelled real rocks (drawn as whole units)", fontsize=9)
    ax2.set_ylabel("mean per-group share MAE", fontsize=9)
    ax2.set_title("mode magnitude: absolute error", fontsize=10)
    _rock_ticks(ax2, rows)

    fig.suptitle(f"few-shot sim-to-real adaptation -- {subtitle}\n"
                 "band: 10-90 over random unit draws; dashed: nothing fit",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_units_axis(rows, out_path: Path, subtitle: str,
                    slice_label: str = "test") -> None:
    """The same budget expressed in eruptions, which is how rocks are harvested."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for tier in CURVE_TIERS:
        pts = [(r["n_units"], r["dirichlet"]) for r in rows if r["tier"] == tier]
        if pts:
            ax.plot(*zip(*pts), "o", ms=3.5, alpha=0.5,
                    color=TIER_COLOR[tier], label=tier)
    ax.set_xlabel("labelled stratigraphic units (eruptions)", fontsize=9)
    ax.set_ylabel(f"Dirichlet NLL on the {slice_label} slice (nats)", fontsize=9)
    ax.set_title(f"the same curve priced in eruptions -- {subtitle}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--select-wd", action="store_true",
                        help="re-select weight decay on val before the curve")
    parser.add_argument("--eval-split", choices=("test", "val"), default="val",
                        help="slice the curve is scored on; test was spent "
                             "2026-08-03, so scoring it is a disclosed "
                             "second look")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder, benchmark, device)
    gp = build_group_predictions(ckpt.build_mode_head().to(device),
                                 benchmark, latents)

    clean = benchmark.clean & benchmark.has_magnitude
    train = gp.subset(split_mask(benchmark, "train") & clean)
    val = gp.subset(split_mask(benchmark, "val") & clean)
    test = gp.subset(split_mask(benchmark, "test") & clean)
    print(f"train {len(train)} rocks / {len(np.unique(train.unit))} units   "
          f"val {len(val)}   test {len(test)} rocks / "
          f"{len(np.unique(test.unit))} units")

    weight_decay = dict(WEIGHT_DECAY)
    wd_records = {}
    if args.select_wd:
        for tier in CURVE_TIERS:
            best, records = select_weight_decay(train, val, tier=tier)
            weight_decay[tier] = best
            wd_records[tier] = records
            print(f"selected weight_decay[{tier}] = {best:g}")

    eval_data = {"test": test, "val": val}[args.eval_split]
    baseline, rows = run_curve(
        train, eval_data, budgets=BUDGETS, n_draws=args.draws,
        weight_decay=weight_decay,
    )

    tag = args.tag or args.checkpoint.parent.name
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.eval_split == "test" else f"_{args.eval_split}"
    subtitle = (f"{len(eval_data)} {args.eval_split} rocks, "
                f"{len(np.unique(eval_data.unit))} units")
    plot_curve(baseline, rows, out_dir / f"learning_curve{suffix}.png",
               subtitle, args.eval_split)
    plot_units_axis(rows, out_dir / f"curve_by_units{suffix}.png",
                    subtitle, args.eval_split)

    # the full-data fit, whose per-group constants are what a reader compares
    # against the measured residual shape
    full = {tier: per_group_offsets(
        fit_recalibrator(train, tier=tier, weight_decay=weight_decay[tier])
    ) for tier in CURVE_TIERS}

    payload = {
        "checkpoint": str(args.checkpoint),
        "benchmark": "eruption-v2",
        "eval_split": args.eval_split,
        "weight_decay": weight_decay,
        "weight_decay_selection": wd_records,
        "splits": {
            "train": {"rocks": len(train), "units": int(len(np.unique(train.unit)))},
            "val": {"rocks": len(val), "units": int(len(np.unique(val.unit)))},
            "test": {"rocks": len(test), "units": int(len(np.unique(test.unit)))},
        },
        "zero_shot": {
            "joint": float(baseline["joint"].mean()),
            "dirichlet": float(baseline["dirichlet"].mean()),
            "bernoulli": float(baseline["bernoulli"].mean()),
            "share_mae": baseline["share_mae_mean"],
            "mean_composition_size": baseline["mean_composition_size"],
        },
        "groups": list(GROUPS),
        "full_train_offsets": full,
        "curve": rows,
    }
    (out_dir / f"curve{suffix}.json").write_text(json.dumps(payload, indent=2))
    print(f"saved {out_dir / f'curve{suffix}.json'}")


if __name__ == "__main__":
    main()
