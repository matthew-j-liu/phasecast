"""How many labelled real rocks close the sim-to-real gap, and what were the
simulated labels worth? (#52, #53)

Six arms fit on real modal labels drawn as whole stratigraphic units and
scored on a fixed held-out slice (--eval-split; val since the test slice was
spent 2026-08-03), over a sweep of labelled-rock budgets.

Two correct the frozen simulator-supervised head (`vector` in `matrix`), and
answer how many eruptions are worth transcribing.  Three more train a
group-space hurdle head on measured labels
alone, never seeing a simulated label, and differ only in what they read about
the rock: the frozen encoder's latent (`measured`), the assay that latent was
computed from (`measured_raw`), or a constant (`measured_ablated`).  A sixth
(`adapted`) adapts the simulator-supervised head itself rather than its
outputs, which is what separates "the simulated labels were worth less" from
"the correction was too thin to carry them".

Differencing across those six prices four things separately -- what the
simulator supervision bought, what the encoder bought over the raw chemistry,
what the cohort's own label marginal bought without reading the rock at all,
and how much of the correction arms' shortfall was the correction being
external to the head rather than the supervision being simulated.

    conda run -n mineral python explorations/real_rock_fewshot/run_curve.py \
        --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt

Weight decay is chosen once per arm on the fixed val slice before the curve
runs (--select-wd), so every drawn rock on the x-axis is a fitted rock.
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

# `pythonpath = ["."]` in pyproject covers pytest, not a direct script run, and
# the sibling module has to import under the same package name it does in the
# tests or there are two `AdaptedModeHead` classes.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, split_mask
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.measured_mode_head import (
    ablate_latent,
    fit_measured_head,
    score_measured_head,
    use_raw_chemistry,
)
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    draw_units,
    fit_recalibrator,
    paired_bootstrap,
    per_group_offsets,
    score_recalibrator,
    select_weight_decay,
)

from explorations.real_rock_fewshot.adapted_mode_head import (
    fit_adapted_head,
    score_adapted_head,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"

# Chosen once on the val slice by `select_weight_decay`; see README. Recorded
# rather than re-selected on every run so the curve is reproducible and the
# val slice's contribution stays one scalar per arm.
WEIGHT_DECAY: dict[str, float] = {
    "vector": 0.01, "matrix": 1.0,
    "measured": 1.0, "measured_raw": 1.0, "measured_ablated": 10.0,
    "adapted": 1.0,
}

# The correction arms compared on the curve; mlp is off it because its capacity
# never repaid itself at any labelled budget (#52).
CORRECTION_ARMS: tuple[str, ...] = ("vector", "matrix")

# One head, three inputs.  `measured` reads the frozen encoder's latent;
# `measured_raw` reads the assay that latent was computed from; and
# `measured_ablated` reads a constant.  None of them ever sees a simulated
# label, so together they price what the simulator supervision, the encoder,
# and the cohort's own label marginal each contributed (#53).
MEASURED_ARMS: tuple[str, ...] = ("measured", "measured_raw", "measured_ablated")

# The correction arms fit at most 119 parameters on the frozen head's outputs,
# so their loss to the measured arm reads as "the head was never adapted, only
# rescaled".  This arm adapts it: low-rank updates inside its own linear layers,
# trained on measured labels, at a trainable budget matched to the measured
# head's.
ADAPTED_ARM: str = "adapted"

CURVE_ARMS: tuple[str, ...] = CORRECTION_ARMS + MEASURED_ARMS + (ADAPTED_ARM,)

# Everything the latent-reading measured arm is differenced against.
CONTRAST_ARMS: tuple[str, ...] = (
    CORRECTION_ARMS + (ADAPTED_ARM, "measured_raw", "measured_ablated")
)

# Every arm is measured; not every arm communicates.  The figures drop `vector`
# (dominated everywhere, and its ceiling is not the one being argued about), so
# the reader adjudicates between label sources rather than between two rungs of
# the same correction.  `matrix` is the canonical rung, so it is the one the
# figures name.
PLOT_ARMS: tuple[str, ...] = ("matrix",) + MEASURED_ARMS

BUDGETS: tuple[int, ...] = (25, 50, 100, 200, 400)
ARM_COLOR = {"vector": "tab:blue", "matrix": "tab:orange",
             "measured": "tab:purple",
             "measured_raw": "tab:brown", "measured_ablated": "tab:gray",
             "adapted": "tab:cyan"}


def arm_fitter(arm: str, head):
    """`(fit, score)` for one arm, both on `select_weight_decay`'s contract.

    Dispatch lives here rather than in `src/` so the three training modules stay
    independent; only the caller that runs every arm needs to know them all.
    `head` is the frozen ModeHead, which only the adapted arm reads -- every
    other arm works from the group-space outputs it already produced.
    """
    if arm == ADAPTED_ARM:
        return partial(fit_adapted_head, head=head), score_adapted_head
    if arm not in MEASURED_ARMS:
        return partial(fit_recalibrator, tier=arm), score_recalibrator
    # the input swap applies on the way into both the fit and the scoring, so
    # an arm never reads on one side what it was denied on the other
    view = {"measured": lambda d: d,
            "measured_raw": use_raw_chemistry,
            "measured_ablated": ablate_latent}[arm]
    return (lambda data, **kw: fit_measured_head(view(data), **kw),
            lambda fitted, data: score_measured_head(fitted, view(data)))


def run_curve(train, test, *, budgets, n_draws, weight_decay, head, seed=0):
    """One fit per (arm, budget, draw), each scored on the fixed eval slice.

    The draw is taken once per (budget, draw) and every arm fits that same set
    of rocks, so the arm comparison is paired by construction -- at 25 rocks
    which units you drew dominates everything, and two arms measured on
    different draws are two overlapping bands rather than a contrast.  Returns
    the per-arm rows and, separately, the per-draw contrasts against the
    measured arm.
    """
    baseline = score_recalibrator(None, test)
    rows, contrasts = [], []
    for budget in budgets:
        rng = np.random.default_rng(seed + budget)
        for draw in range(n_draws):
            mask = draw_units(train, budget, rng)
            if not mask.any():
                continue
            drawn = train.subset(mask)
            scored = {}
            for arm in CURVE_ARMS:
                fit, score = arm_fitter(arm, head)
                model = fit(drawn, weight_decay=weight_decay[arm], seed=draw)
                got = score(model, test)
                scored[arm] = got
                boot = paired_bootstrap(
                    baseline["dirichlet"], got["dirichlet"], test.unit, seed=draw,
                )
                rows.append({
                    "arm": arm,
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
            for arm in CONTRAST_ARMS:
                # positive = the measured-label head is ahead of that arm
                boot = paired_bootstrap(
                    scored[arm]["joint"], scored["measured"]["joint"],
                    test.unit, seed=draw,
                )
                contrasts.append({
                    "arm": arm, "budget": budget, "draw": draw,
                    "n_rocks": int(len(drawn)),
                    "minus_measured_joint": boot["mean"],
                    "lo": boot["lo"], "hi": boot["hi"],
                })
        done = len({r["draw"] for r in rows if r["budget"] == budget})
        print(f"  budget {budget:4d}: {done} draws x {len(CURVE_ARMS)} arms done")
    return baseline, rows, contrasts


def _band(ax, rows, arm, key, color):
    """Median with a 10-90 spread band, x = realized labelled rocks."""
    budgets = sorted({r["budget"] for r in rows if r["arm"] == arm})
    x, med, lo, hi = [], [], [], []
    for budget in budgets:
        vals = [r[key] for r in rows if r["arm"] == arm and r["budget"] == budget]
        n = [r["n_rocks"] for r in rows if r["arm"] == arm and r["budget"] == budget]
        if not vals:
            continue
        x.append(np.mean(n))
        med.append(np.median(vals))
        lo.append(np.quantile(vals, 0.1))
        hi.append(np.quantile(vals, 0.9))
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, med, "o-", color=color, ms=4, label=arm)
    return x, med


def _rock_ticks(ax, rows) -> None:
    """Tick at the realized budgets, since a log axis otherwise labels only 100."""
    ticks = sorted({int(round(np.mean(
        [r["n_rocks"] for r in rows if r["budget"] == b]
    ))) for b in {r["budget"] for r in rows}})
    ax.set_xticks(ticks, [str(t) for t in ticks], fontsize=8)
    ax.minorticks_off()


def plot_curve(zero_shot, rows, out_path: Path, subtitle: str,
               slice_label: str = "test") -> None:
    """The two headline panels: Dirichlet term and per-group share error."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for arm in CURVE_ARMS:
        _band(ax1, rows, arm, "dirichlet", ARM_COLOR[arm])
    ax1.axhline(zero_shot["dirichlet"], lw=1.0, color="0.35", ls="--")
    ax1.annotate("zero-shot", (0.02, zero_shot["dirichlet"]),
                 xycoords=("axes fraction", "data"), va="bottom", fontsize=8,
                 color="0.35")
    ax1.set_xscale("log")
    ax1.set_xlabel("labelled real rocks (drawn as whole units)", fontsize=9)
    ax1.set_ylabel(f"Dirichlet NLL on the {slice_label} slice (nats)", fontsize=9)
    ax1.set_title("mode magnitude: the headline", fontsize=10)
    ax1.legend(fontsize=8)
    _rock_ticks(ax1, rows)

    for arm in CURVE_ARMS:
        _band(ax2, rows, arm, "share_mae", ARM_COLOR[arm])
    ax2.axhline(zero_shot["share_mae"], lw=1.0, color="0.35", ls="--")
    ax2.set_xscale("log")
    ax2.set_xlabel("labelled real rocks (drawn as whole units)", fontsize=9)
    ax2.set_ylabel("mean per-group share MAE", fontsize=9)
    ax2.set_title("mode magnitude: absolute error", fontsize=10)
    _rock_ticks(ax2, rows)

    fig.suptitle(f"few-shot sim-to-real adaptation -- {subtitle}\n"
                 "band: 10-90 over random unit draws; dashed: nothing fit; "
                 "measured = head trained on measured labels only",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_units_axis(rows, out_path: Path, subtitle: str,
                    slice_label: str = "test") -> None:
    """The same budget expressed in eruptions, which is how rocks are harvested."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for arm in CURVE_ARMS:
        pts = [(r["n_units"], r["dirichlet"]) for r in rows if r["arm"] == arm]
        if pts:
            ax.plot(*zip(*pts), "o", ms=3.5, alpha=0.5,
                    color=ARM_COLOR[arm], label=arm)
    ax.set_xlabel("labelled stratigraphic units (eruptions)", fontsize=9)
    ax.set_ylabel(f"Dirichlet NLL on the {slice_label} slice (nats)", fontsize=9)
    ax.set_title(f"the same curve priced in eruptions -- {subtitle}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


ARM_LABEL = {
    "matrix": "corrected simulator head",
    "measured": "measured labels, reads encoder latent",
    "measured_raw": "measured labels, reads raw assay",
    "measured_ablated": "measured labels, reads nothing",
}


def plot_share_mae(zero_shot, rows, out_path: Path, subtitle: str,
                   slice_label: str = "val") -> None:
    """Absolute share error against budget, the metric a geologist reads first.

    Single panel and four arms, because the point is a three-way division of
    labour that a six-arm plot buries.  The zero-shot line is drawn rather than
    annotated: most arms sit above it at the low budgets, and "fitting a few
    measured rocks makes absolute accuracy worse before it makes it better" is
    only visible if the reader can see where that line is.
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for arm in PLOT_ARMS:
        _band(ax, rows, arm, "share_mae", ARM_COLOR[arm])
    ax.axhline(zero_shot["share_mae"], lw=1.2, color="0.35", ls="--")
    ax.annotate("zero-shot (nothing fit)", (0.98, zero_shot["share_mae"]),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("labelled real rocks (drawn as whole eruptions)", fontsize=9)
    ax.set_ylabel("mean absolute error in predicted mineral share", fontsize=9)
    ax.set_title(f"how close the predicted proportions land -- {subtitle}",
                 fontsize=10)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [ARM_LABEL[a] for a in PLOT_ARMS], fontsize=8)
    _rock_ticks(ax, rows)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def crossing_budget(rows, arm: str = "matrix") -> dict:
    """Where the measured arm's median overtakes `arm`'s, in labelled rocks.

    The exchange rate the two arms price: below this budget the simulated
    labels are worth more than the measured ones someone could have collected
    instead, above it they are not.  Interpolated on the log-budget axis
    between the bracketing grid points, since the grid is coarse; `None` means
    the arms never cross inside the swept range, which is itself the answer.
    """
    budgets = sorted({r["budget"] for r in rows})
    def median(a, b):
        return float(np.median([r["joint"] for r in rows
                                if r["arm"] == a and r["budget"] == b]))
    def rocks(b):
        return float(np.mean([r["n_rocks"] for r in rows if r["budget"] == b]))

    delta = [median("measured", b) - median(arm, b) for b in budgets]
    for i in range(len(budgets) - 1):
        if delta[i] > 0 >= delta[i + 1]:
            lo, hi = np.log(rocks(budgets[i])), np.log(rocks(budgets[i + 1]))
            f = delta[i] / (delta[i] - delta[i + 1])
            return {"against": arm, "rocks": float(np.exp(lo + f * (hi - lo))),
                    "brackets": [budgets[i], budgets[i + 1]]}
    return {"against": arm, "rocks": None,
            "brackets": None,
            "measured_ahead_throughout": bool(all(d < 0 for d in delta))}


def plot_contrast(contrasts, out_path: Path, subtitle: str) -> None:
    """Paired per-draw difference against the measured arm, the exchange rate."""
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for arm in CONTRAST_ARMS:
        budgets = sorted({c["budget"] for c in contrasts if c["arm"] == arm})
        x, med, lo, hi = [], [], [], []
        for budget in budgets:
            vals = [c["minus_measured_joint"] for c in contrasts
                    if c["arm"] == arm and c["budget"] == budget]
            n = [c["n_rocks"] for c in contrasts
                 if c["arm"] == arm and c["budget"] == budget]
            x.append(np.mean(n))
            med.append(np.median(vals))
            lo.append(np.quantile(vals, 0.1))
            hi.append(np.quantile(vals, 0.9))
        ax.fill_between(x, lo, hi, color=ARM_COLOR[arm], alpha=0.18, lw=0)
        ax.plot(x, med, "o-", color=ARM_COLOR[arm], ms=4, label=arm)
    ax.axhline(0.0, lw=1.0, color="0.35", ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("labelled real rocks (drawn as whole units)", fontsize=9)
    ax.set_ylabel("correction minus measured, joint NLL (nats)", fontsize=9)
    ax.set_title(f"what the simulated labels were worth -- {subtitle}",
                 fontsize=10)
    ax.annotate("above 0: the measured-label head wins", (0.03, 0.95),
                xycoords="axes fraction", va="top", fontsize=8, color="0.35")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def draw_figures(zero_shot, rows, contrasts, out_dir: Path, suffix: str,
                 subtitle: str, slice_label: str) -> None:
    """Every figure the sweep produces, from the recorded curve alone.

    Split out from `main` so `--replot` redraws from `curve*.json` rather than
    refitting: the sweep is minutes of compute and figure wording is not.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curve(zero_shot, rows, out_dir / f"learning_curve{suffix}.png",
               subtitle, slice_label)
    plot_share_mae(zero_shot, rows, out_dir / f"share_mae{suffix}.png",
                   subtitle, slice_label)
    plot_units_axis(rows, out_dir / f"curve_by_units{suffix}.png",
                    subtitle, slice_label)
    plot_contrast(contrasts, out_dir / f"arm_contrast{suffix}.png", subtitle)


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
    parser.add_argument("--replot", action="store_true",
                        help="redraw the figures from the recorded curve "
                             "without refitting anything")
    args = parser.parse_args()

    tag = args.tag or args.checkpoint.parent.name
    out_dir = RESULTS / tag
    suffix = "" if args.eval_split == "test" else f"_{args.eval_split}"
    if args.replot:
        recorded = json.loads((out_dir / f"curve{suffix}.json").read_text())
        sp = recorded["splits"][args.eval_split]
        draw_figures(recorded["zero_shot"], recorded["curve"],
                     recorded["arm_contrasts"], out_dir, suffix,
                     f"{sp['rocks']} {args.eval_split} rocks, {sp['units']} units",
                     args.eval_split)
        return

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder, benchmark, device)
    head = ckpt.build_mode_head().to(device)
    gp = build_group_predictions(head, benchmark, latents)

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
        for arm in CURVE_ARMS:
            fit, score = arm_fitter(arm, head)
            best, records = select_weight_decay(train, val, fit=fit, score=score)
            weight_decay[arm] = best
            wd_records[arm] = records
            print(f"selected weight_decay[{arm}] = {best:g}")

    eval_data = {"test": test, "val": val}[args.eval_split]
    baseline, rows, contrasts = run_curve(
        train, eval_data, budgets=BUDGETS, n_draws=args.draws,
        weight_decay=weight_decay, head=head,
    )

    zero_shot = {
        "joint": float(baseline["joint"].mean()),
        "dirichlet": float(baseline["dirichlet"].mean()),
        "bernoulli": float(baseline["bernoulli"].mean()),
        "share_mae": baseline["share_mae_mean"],
        "mean_composition_size": baseline["mean_composition_size"],
    }
    subtitle = (f"{len(eval_data)} {args.eval_split} rocks, "
                f"{len(np.unique(eval_data.unit))} units")
    draw_figures(zero_shot, rows, contrasts, out_dir, suffix, subtitle,
                 args.eval_split)

    crossings = [crossing_budget(rows, arm) for arm in CORRECTION_ARMS]
    for c in crossings:
        where = "never inside the swept range" if c["rocks"] is None \
            else f"{c['rocks']:.0f} rocks"
        print(f"  measured overtakes {c['against']}: {where}")

    # the full-data fit, whose per-group constants are what a reader compares
    # against the measured residual shape.  Correction arms only: the measured
    # arm has no per-group constants to read, since it is not a correction.
    full = {arm: per_group_offsets(
        fit_recalibrator(train, tier=arm, weight_decay=weight_decay[arm])
    ) for arm in CORRECTION_ARMS}

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
        "zero_shot": zero_shot,
        "groups": list(GROUPS),
        "full_train_offsets": full,
        "curve": rows,
        "arm_contrasts": contrasts,
        "crossings": crossings,
    }
    (out_dir / f"curve{suffix}.json").write_text(json.dumps(payload, indent=2))
    print(f"saved {out_dir / f'curve{suffix}.json'}")


if __name__ == "__main__":
    main()
