"""Collect the corpus-scaling ladder into one figure + results JSON.

  python explorations/corpus_scaling/aggregate.py

Run from the repo root after `run_ladder.py`. Reads every
`runs/mode_head/corpus_scaling_*/metrics.json` plus the matching checkpoint
monitor (per-head best selection NLLs), writes `results/scaling.json` and
`results/scaling.png` next to this file (plus `scaling_draws.png`, the same
panel set vs draws/bulk, when draws-axis points exist), and prints the
last-doubling deltas that anchor the data-limited-vs-not verdict. Verdict
prose lives in the README, not here.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
TAG_PREFIX = "corpus_scaling_"
DRAWS_PER_BULK = 48
N_RUNGS = 2
FULL_BULKS = 10000

HIGHLIGHTS = {"g": "garnet", "spl": "spinel", "amp": "amphibole", "ilm": "ilmenite"}
PHASE_METRICS = ("pf_r", "ece", "magnitude_mae", "magnitude_r",
                 "bernoulli_nll", "frequency")


def point_record(tag: str, metrics: dict, head_bests: dict) -> dict:
    """One ladder point: axis position + the metrics the figure/slopes use."""
    kind, value = tag.removeprefix(TAG_PREFIX)[0], int(tag.removeprefix(TAG_PREFIX)[1:])
    n_bulks = value if kind == "n" else FULL_BULKS
    draws = DRAWS_PER_BULK if kind == "n" else value
    return {
        "tag": tag,
        "axis": "bulks" if kind == "n" else "draws",
        "n_bulks": n_bulks,
        "draws_per_bulk": draws,
        "n_train_rows_nominal": n_bulks * draws * N_RUNGS,
        "joint_nll": metrics["joint_nll"],
        "bernoulli_nll": metrics["bernoulli_nll"],
        "dirichlet_nll": metrics["dirichlet_nll"],
        "head_bests": head_bests,
        "per_phase": {
            p["phase"]: {k: p[k] for k in PHASE_METRICS} for p in metrics["per_phase"]
        },
    }


def last_doubling_deltas(points: list[dict]) -> list[dict]:
    """Metric change over the final bulk-count doubling, the slope the NERSC
    sizing reads: still moving = data-limited, flat = not."""
    bulks = sorted((p for p in points if p["axis"] == "bulks"),
                   key=lambda p: p["n_bulks"])
    if len(bulks) < 2:
        return []
    prev, last = bulks[-2], bulks[-1]
    rows = [{
        "metric": "joint_nll", "phase": None,
        "at_prev": prev["joint_nll"], "at_last": last["joint_nll"],
        "delta": last["joint_nll"] - prev["joint_nll"],
    }]
    for phase in HIGHLIGHTS:
        for metric in ("pf_r", "ece", "magnitude_mae"):
            a, b = prev["per_phase"][phase][metric], last["per_phase"][phase][metric]
            rows.append({"metric": metric, "phase": phase,
                         "at_prev": a, "at_last": b, "delta": b - a})
    return rows


def collect_points() -> list[dict]:
    from cnmineral.training.joint_heads import JointHeadCheckpoint, selection_series

    points = []
    for metrics_path in sorted((REPO / "runs" / "mode_head").glob(f"{TAG_PREFIX}*/metrics.json")):
        tag = metrics_path.parent.name
        with open(metrics_path) as f:
            metrics = json.load(f)
        ckpt = JointHeadCheckpoint.load(REPO / "checkpoints" / tag / "heads_best.pt")
        head_bests = {
            name: min(vals) for name, vals in selection_series(ckpt.monitor).items()
        }
        points.append(point_record(tag, metrics, head_bests))
    return points


def draws_series(points: list[dict]) -> list[dict]:
    """Draws-axis points plus the full corpus as the 48-draw endpoint."""
    pts = [p for p in points if p["axis"] == "draws"]
    pts += [p for p in points if p["axis"] == "bulks" and p["n_bulks"] == FULL_BULKS]
    return sorted(pts, key=lambda p: p["draws_per_bulk"])


def _standard_panels(axes, pts: list[dict], x: list[int], xlabel: str) -> None:
    ax = axes[0, 0]
    ax.plot(x, [p["joint_nll"] for p in pts], "o-", color="k")
    ax.set_ylabel("joint hurdle NLL (nats/row)")
    ax.set_title("val joint NLL")

    for ax, metric, title in (
        (axes[0, 1], "pf_r", "presence p-vs-f correlation"),
        (axes[0, 2], "ece", "presence ECE"),
        (axes[1, 0], "magnitude_mae", "conditional magnitude MAE"),
    ):
        for phase in pts[0]["per_phase"]:
            series = [p["per_phase"][phase][metric] for p in pts]
            if phase in HIGHLIGHTS:
                ax.plot(x, series, "o-", label=HIGHLIGHTS[phase], zorder=3)
            else:
                ax.plot(x, series, "-", color="0.8", lw=0.8, zorder=1)
        ax.set_title(title)
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name in ("mode", "comp", "cond"):
        ax.plot(x, [p["head_bests"][name] for p in pts], "o-", label=name)
    ax.set_ylabel("best val selection NLL")
    ax.set_title("per-head selection bests")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        if ax.has_data():
            ax.grid(alpha=0.3)
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.set_xscale("log")
        ax.set_xticks(x, [str(v) for v in x])
        ax.minorticks_off()
        ax.set_xlabel(xlabel)


def plot(points: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    bulks = sorted((p for p in points if p["axis"] == "bulks"),
                   key=lambda p: p["n_bulks"])
    draws = sorted((p for p in points if p["axis"] == "draws"),
                   key=lambda p: p["draws_per_bulk"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _standard_panels(axes, bulks, [p["n_bulks"] for p in bulks], "training bulks")

    ax = axes[1, 2]
    if draws:
        ax.plot([p["n_train_rows_nominal"] for p in bulks],
                [p["joint_nll"] for p in bulks], "o-", label="fewer bulks, 48 draws")
        full = [p for p in bulks if p["n_bulks"] == FULL_BULKS]
        ax.plot([p["n_train_rows_nominal"] for p in draws + full],
                [p["joint_nll"] for p in draws + full],
                "s--", label="all bulks, fewer draws")
        ax.set_xscale("log")
        ax.set_xlabel("training rows (nominal)")
        ax.set_title("bulks vs draws at matched rows")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_draws(points: list[dict], out_path: Path) -> None:
    """Same panel set as the bulk figure, but vs draws/bulk at full bulk count."""
    import matplotlib.pyplot as plt

    pts = draws_series(points)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _standard_panels(axes, pts, [p["draws_per_bulk"] for p in pts],
                     "draws per bulk (all 10k bulks)")
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points = collect_points()
    if not any(p["axis"] == "bulks" for p in points):
        raise SystemExit("no bulk-ladder points under runs/mode_head/; "
                         "run run_ladder.py first")
    RESULTS.mkdir(exist_ok=True)
    deltas = last_doubling_deltas(points)
    with open(RESULTS / "scaling.json", "w") as f:
        json.dump({"points": points, "last_doubling_deltas": deltas}, f, indent=1)
    plot(points, RESULTS / "scaling.png")
    print(f"{len(points)} points -> {RESULTS / 'scaling.json'}")
    print(f"figure -> {RESULTS / 'scaling.png'}")
    if any(p["axis"] == "draws" for p in points):
        plot_draws(points, RESULTS / "scaling_draws.png")
        print(f"figure -> {RESULTS / 'scaling_draws.png'}")
    if deltas:
        bulks = sorted(p["n_bulks"] for p in points if p["axis"] == "bulks")[-2:]
        print(f"\nlast doubling ({bulks[0]} -> {bulks[1]} bulks):")
        for row in deltas:
            name = f"{row['phase']}.{row['metric']}" if row["phase"] else row["metric"]
            print(f"  {name}: {row['at_prev']:.4f} -> {row['at_last']:.4f} "
                  f"(delta {row['delta']:+.4f})")


if __name__ == "__main__":
    main()
