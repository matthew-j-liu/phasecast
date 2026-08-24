"""Fix the adapted arm's learning rate and epoch budget off the train loss alone.

The correction arms spend exactly one val-selected scalar each (weight decay).
An arm that also tuned `lr` and `epochs` on val would be carrying three, and its
margin would partly be the extra selection rather than the adaptation.  So this
reads the train loss and nothing else: the largest learning rate whose curve is
still monotone, and the epoch at which that curve has stopped moving.  The
shrinkage it runs at is fixed and inherited rather than selected -- see
`PROBE_WEIGHT_DECAY` for why it cannot be zero.

    conda run -n mineral python \
        explorations/real_rock_fewshot/run_plateau_probe.py \
        --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt

Writes its curves to a scratch directory, not to `results/` -- the output is two
numbers, which land in `adapted_mode_head.LR` and `EPOCHS`.
"""

import argparse
import json
import sys
from pathlib import Path

# `pythonpath = ["."]` in pyproject covers pytest, not a direct script run, and
# the sibling module has to import under the same package name it does in the
# tests or there are two `AdaptedModeHead` classes.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, split_mask
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    draw_units,
)

from explorations.real_rock_fewshot.adapted_mode_head import fit_adapted_head

LR_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)

# The ends of the curve's swept range.  A learning rate that converges on 368
# rocks can oscillate on 25, and the one constant has to serve both.
PROBE_BUDGETS: tuple[int, ...] = (25, 400)
SEEDS: tuple[int, ...] = (0, 1, 2)
EPOCH_CAP: int = 6000

# Probing at zero shrinkage measures nothing: 33k adapters against at most 368
# rocks drive the Dirichlet density down without bound, so the train loss keeps
# descending (still -0.4 nats per 500 epochs at epoch 2000) and there is no
# plateau to find.  Shrinkage is what makes the objective bounded, so the probe
# runs at one -- and at the value the measured arms and both surviving
# correction tiers already selected, so it is inherited rather than chosen here.
# This arm's own constant is still picked on val afterwards, with lr and epochs
# already fixed.
PROBE_WEIGHT_DECAY: float = 1.0

# An excursion above the run's own best, as a fraction of the descent that run
# achieved.  Scaled rather than absolute: a converged Adam fit wiggles by ~1e-3
# nats forever, so any fixed threshold near that is measuring the noise floor and
# calling every learning rate unstable.  A genuinely too-large rate oscillates by
# whole nats, which is percent of the descent, not per-mille.
RISE_FRACTION: float = 0.01

# Converged means the remaining descent is negligible against what the curve can
# resolve, so the threshold is set from that rather than from zero: arm contrasts
# run to whole nats and a slice carries roughly +/- 2, so 0.01 is already two
# orders below anything readable.  Tightening it to 1e-3 triples the epoch budget
# to buy 0.004 nats.
PLATEAU_TOLERANCE: float = 0.01

TAIL_FRACTION: float = 0.2


def is_stable(trace: list[float]) -> bool:
    """Whether the tail stays near the best loss the run reached.

    Read against the running minimum rather than the previous epoch: the
    quantity that matters is whether the optimizer is holding its optimum or
    bouncing off it, and per-epoch differences cannot tell those apart at this
    objective's noise floor.
    """
    values = np.asarray(trace)
    if not np.isfinite(values).all():
        return False
    tail = slice(-int(len(values) * TAIL_FRACTION), None)
    descent = values[0] - np.minimum.accumulate(values)[-1]
    excursion = (values[tail] - np.minimum.accumulate(values)[tail]).max()
    return bool(excursion <= RISE_FRACTION * descent)


def plateau_epoch(trace: list[float]) -> int:
    """The epoch by which the run has stopped making progress.

    Measured on the running minimum, which is monotone by construction, so the
    wiggle a converged Adam fit never stops making cannot push this to the
    epoch cap.
    """
    best = np.minimum.accumulate(np.asarray(trace))
    return int(np.argmax(best - best[-1] < PLATEAU_TOLERANCE)) + 1


def probe(train, head, *, budgets, lr_grid, seeds, epochs, weight_decay) -> list[dict]:
    """One fit per (budget, lr, seed), each recording its whole train curve."""
    rows = []
    for budget in budgets:
        rng = np.random.default_rng(budget)
        drawn = train.subset(draw_units(train, budget, rng))
        for lr in lr_grid:
            for seed in seeds:
                fitted = fit_adapted_head(
                    drawn, head=head, weight_decay=weight_decay,
                    epochs=epochs, lr=lr, seed=seed,
                )
                trace = fitted.train_loss
                rows.append({
                    "budget": budget,
                    "n_rocks": int(len(drawn)),
                    "lr": lr,
                    "seed": seed,
                    "final": trace[-1],
                    "best": min(trace),
                    "stable": is_stable(trace),
                    "plateau_epoch": plateau_epoch(trace),
                    "trace": trace,
                })
                print(f"  {len(drawn):3d} rocks  lr {lr:g}  seed {seed}  "
                      f"best {min(trace):.4f}  "
                      f"{'stable' if rows[-1]['stable'] else 'UNSTABLE'}  "
                      f"plateau @ {rows[-1]['plateau_epoch']}")
    return rows


def choose(rows: list[dict]) -> dict:
    """The largest stable learning rate, and the epoch every curve has settled by.

    Largest rather than best-final: the fastest stable descent is the one that
    reaches the plateau inside a budget the sweep can afford, and picking on the
    final value would be selecting on a fit rather than on convergence.  The
    epoch is a max over budgets and seeds so every point on the curve is
    converged, not the median one.
    """
    stable = [
        lr for lr in sorted({r["lr"] for r in rows})
        if all(r["stable"] for r in rows if r["lr"] == lr)
    ]
    if not stable:
        raise ValueError(
            "no learning rate on the grid was monotone at both budgets; widen "
            "LR_GRID downward before reading anything else here"
        )
    lr = max(stable)
    at_lr = [r for r in rows if r["lr"] == lr]
    return {
        "lr": lr,
        "epochs": max(r["plateau_epoch"] for r in at_lr),
        "stable_grid": stable,
        "per_budget_plateau": {
            str(b): max(r["plateau_epoch"] for r in at_lr if r["budget"] == b)
            for b in sorted({r["budget"] for r in at_lr})
        },
    }


def plot_traces(rows: list[dict], chosen: dict, out_path: Path) -> None:
    """Every train curve, one panel per budget, with the chosen epoch marked."""
    budgets = sorted({r["budget"] for r in rows})
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.6 * len(budgets), 4.2),
                             squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len({r["lr"] for r in rows})))
    lrs = sorted({r["lr"] for r in rows})
    for ax, budget in zip(axes.flat, budgets):
        for r in [r for r in rows if r["budget"] == budget]:
            ax.plot(r["trace"], lw=0.9, alpha=0.75,
                    color=colors[lrs.index(r["lr"])],
                    label=f"lr {r['lr']:g}" if r["seed"] == 0 else None)
        ax.axvline(chosen["epochs"], lw=1.0, color="0.35", ls="--")
        n = next(r["n_rocks"] for r in rows if r["budget"] == budget)
        ax.set_title(f"{n} labelled rocks", fontsize=10)
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_ylabel("train joint NLL (nats)", fontsize=9)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"adapted arm: where the train loss plateaus -- chosen lr "
        f"{chosen['lr']:g}, {chosen['epochs']} epochs\n"
        "no shrinkage, no val; dashed: the chosen epoch budget",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="scratch directory for the curves; not results/")
    parser.add_argument("--epochs", type=int, default=EPOCH_CAP)
    parser.add_argument("--weight-decay", type=float, default=PROBE_WEIGHT_DECAY,
                        help="re-check convergence at the shrinkage the val "
                             "selection actually landed on")
    args = parser.parse_args()

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
    print(f"train {len(train)} rocks / {len(np.unique(train.unit))} units")

    rows = probe(train, head, budgets=PROBE_BUDGETS, lr_grid=LR_GRID,
                 seeds=SEEDS, epochs=args.epochs,
                 weight_decay=args.weight_decay)

    # traces land before anything adjudicates them: this is twenty minutes of
    # compute, and a criterion that rejects the whole grid should cost a reread
    # rather than the run
    args.out.mkdir(parents=True, exist_ok=True)
    traces = args.out / "plateau_traces.json"
    traces.write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "epoch_cap": args.epochs,
         "weight_decay": args.weight_decay, "runs": rows}, indent=2,
    ))
    print(f"saved {traces}")

    chosen = choose(rows)
    print(f"\nchose lr {chosen['lr']:g}, {chosen['epochs']} epochs")
    print(f"  stable across the whole grid at: "
          f"{', '.join(f'{lr:g}' for lr in chosen['stable_grid'])}")
    print(f"  plateau by budget: {chosen['per_budget_plateau']}")

    plot_traces(rows, chosen, args.out / "plateau_probe.png")
    (args.out / "plateau_probe.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "epoch_cap": args.epochs,
         "weight_decay": args.weight_decay, "chosen": chosen,
         "runs": [{k: v for k, v in r.items() if k != "trace"} for r in rows]},
        indent=2,
    ))
    print(f"saved {args.out / 'plateau_probe.json'}")


if __name__ == "__main__":
    main()
