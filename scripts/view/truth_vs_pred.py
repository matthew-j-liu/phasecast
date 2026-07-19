"""
Truth-vs-prediction views of one rock through all three heads.

Renders three figures from a joint-heads checkpoint for a single bulk on
the valtest val roster: per-mineral share + presence-rate boxes (ModeHead),
per-oxide chemistry boxes for the rock's main minerals (CompHead), and
formation-condition histograms under teacher-forced supports (CondHead).
Truth is the corpus draws at one rung; predictions are posterior samples.

    python scripts/view/truth_vs_pred.py \\
        --checkpoint checkpoints/joint_heads_seed0/heads_best.pt --bulk 871826

Omitting `--bulk` uses the first val-roster bulk. Figures land in
`results/summary/<checkpoint-dir-name>/` unless `--out-dir` says otherwise.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from cnmineral.data.magemin_corpus import COMP_OXIDES
from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.data.phase_vocab import SOLID_PHASES
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.models.cond_head import COND_AXES
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.mode_head import _bulk_latents

TRUTH_COLOR, PRED_COLOR = "tab:blue", "tab:orange"
LEGEND = [
    Patch(facecolor=TRUTH_COLOR, alpha=0.7, label="truth"),
    Patch(facecolor=PRED_COLOR, alpha=0.7, hatch="//", label="prediction"),
]


def _pair_boxes(ax, vals_truth, vals_pred, x):
    for vals, xoff, color, hatch in (
        (vals_truth, x - 0.18, TRUTH_COLOR, None),
        (vals_pred, x + 0.18, PRED_COLOR, "//"),
    ):
        if len(vals) == 0:
            continue
        bp = ax.boxplot(
            vals, positions=[xoff], widths=0.3, whis=(10, 90), showfliers=False,
            patch_artist=True, medianprops={"color": "black"},
        )
        bp["boxes"][0].set(facecolor=color, alpha=0.7, hatch=hatch)


def sample_assemblages(p, alpha, n, rng):
    modes = np.zeros((n, len(p)))
    for s in range(n):
        for _ in range(100):
            z = rng.random(len(p)) < p
            if z.any():
                break
        modes[s, z] = rng.dirichlet(alpha[z])
    return modes


def mode_figure(truth, pred, cutoff, title, out_path):
    show = [
        k for k in range(len(SOLID_PHASES))
        if truth[:, k].mean() > 1e-3 or pred[:, k].mean() > 1e-3
    ]
    show.sort(key=lambda k: -truth[:, k].mean())
    fig, (ax_box, ax_rate) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, height_ratios=(2.2, 1)
    )
    for i, k in enumerate(show):
        _pair_boxes(
            ax_box, truth[:, k][truth[:, k] > cutoff],
            pred[:, k][pred[:, k] > cutoff], i,
        )
        ax_rate.bar(i - 0.18, (truth[:, k] > cutoff).mean(), 0.36,
                    color=TRUTH_COLOR, alpha=0.7)
        ax_rate.bar(i + 0.18, (pred[:, k] > cutoff).mean(), 0.36,
                    color=PRED_COLOR, alpha=0.7, hatch="//", edgecolor="white")
    ax_box.set_yscale("log")
    ax_box.set_ylabel("share of the solids, when present (log)")
    ax_box.set_title(
        f"ModeHead -- {title}\nbox = middle 50%, whiskers = p10-p90 of the "
        "mineral's share in draws containing it"
    )
    ax_rate.set_ylabel("fraction of draws\ncontaining the mineral")
    ax_rate.set_ylim(0, 1)
    ax_rate.set_xticks(range(len(show)))
    ax_rate.set_xticklabels([SOLID_PHASES[k] for k in show])
    ax_box.legend(handles=LEGEND, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def comp_figure(presence, comp_target, comp_alpha, n_samples, rng, title, out_path):
    counts = presence.sum(0)
    show = [k for k in np.argsort(-counts)[:4] if counts[k] > 0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ax, k in zip(axes.ravel(), show):
        truth_c = comp_target[presence[:, k] > 0, k, :]
        pred_c = rng.dirichlet(comp_alpha[k], size=n_samples)
        for i in range(len(COMP_OXIDES)):
            _pair_boxes(ax, truth_c[:, i], pred_c[:, i], i)
        ax.set_yscale("log")
        ax.set_ylim(5e-5, 1.5)
        ax.set_xticks(range(len(COMP_OXIDES)))
        ax.set_xticklabels(COMP_OXIDES, rotation=45, ha="right", fontsize=8)
        ax.set_title(
            f"{SOLID_PHASES[k]} (in {int(counts[k])}/{len(presence)} draws)",
            fontsize=10,
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("oxide fraction of the mineral (log)")
    axes[0, 0].legend(handles=LEGEND, loc="lower left", fontsize=9)
    fig.suptitle(
        f"CompHead -- {title}\nbox = middle 50%, whiskers = p10-p90; truth "
        "across draws containing the mineral"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def cond_figure(cond_target, log_w, mu, sigma, rng, title, out_path):
    w = np.exp(log_w)
    n_rows, _, n_comp = mu.shape
    per_row = 4
    pred = np.zeros((n_rows * per_row, len(COND_AXES)))
    for a in range(len(COND_AXES)):
        for r in range(n_rows):
            comp_idx = rng.choice(n_comp, size=per_row, p=w[r, a] / w[r, a].sum())
            pred[r * per_row : (r + 1) * per_row, a] = rng.normal(
                mu[r, a, comp_idx], sigma[r, a, comp_idx]
            )
    labels = {"T": "temperature (C)", "P_kbar": "pressure (kbar)", "H2O": "water (wt%)"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for a, (ax, axis_name) in enumerate(zip(axes, COND_AXES)):
        bins = np.histogram_bin_edges(
            np.concatenate([cond_target[:, a], pred[:, a]]), bins=30
        )
        ax.hist(cond_target[:, a], bins=bins, density=True, color=TRUTH_COLOR,
                alpha=0.7, label="truth")
        ax.hist(pred[:, a], bins=bins, density=True, color=PRED_COLOR,
                alpha=0.55, hatch="//", label="prediction")
        ax.set_xlabel(labels[axis_name])
        ax.set_yticks([])
    axes[0].set_ylabel("relative frequency")
    axes[0].legend(fontsize=9)
    fig.suptitle(
        f"CondHead -- {title}\ntruth = the corpus draws; prediction = samples "
        "from the per-draw mixtures, each given that draw's true mineral list"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to a joint-heads checkpoint (.pt)")
    parser.add_argument("--bulk", type=int, default=None,
                        help="val-roster bulk id (default: first on the roster)")
    parser.add_argument("--rung", type=int, default=50,
                        help="melt-fraction rung for the truth rows")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="posterior samples for the mode/comp views")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output dir (default results/summary/<checkpoint dir>)")
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    mode_head = ckpt.build_mode_head()
    comp_head = ckpt.build_comp_head()
    cond_head = ckpt.build_cond_head()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.eval()

    ds = JointHeadDataset(
        corpus_dir=ckpt.label_config["valtest_corpus_dir"],
        rungs=tuple(ckpt.label_config["rungs"]),
        presence_cutoff=ckpt.label_config["presence_cutoff"],
        comp_eps=ckpt.label_config["comp_eps"],
        bulk_ids=load_valtest_split("val"),
    )
    bulk = int(ds.bulk_ids[0]) if args.bulk is None else args.bulk
    hits = np.flatnonzero(ds.bulk_ids == bulk)
    if len(hits) == 0:
        raise SystemExit(f"bulk {bulk} is not on the val roster")
    bpos = int(hits[0])
    rows = np.flatnonzero(
        (ds.row_bulk_idx.numpy() == bpos) & (ds.rung.numpy() == args.rung)
    )
    if len(rows) == 0:
        raise SystemExit(f"bulk {bulk} has no rows at rung {args.rung}")

    latents = _bulk_latents(encoder, ds, torch.device("cpu"))
    rng = np.random.default_rng(0)
    with torch.no_grad():
        logits, mode_alpha = mode_head(latents[bpos : bpos + 1])
        comp_alpha = comp_head(latents[bpos : bpos + 1])[0].numpy()
        log_w, mu, sigma = cond_head(
            latents[bpos].expand(len(rows), -1), ds.presence[rows]
        )
    p = torch.sigmoid(logits)[0].numpy()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("results/summary") / args.checkpoint.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"{ds.bulk_qap_names[bpos]} (bulk {bulk}), {args.rung}%-melt rung -- "
        f"truth ({len(rows)} draws) vs prediction ({args.n_samples} samples)"
    )

    pred_modes = sample_assemblages(
        p, mode_alpha[0].numpy(), args.n_samples, rng
    )
    mode_figure(
        ds.magnitude[rows].numpy(), pred_modes, ds.presence_cutoff, title,
        out_dir / f"truth_vs_pred_mode_bulk{bulk}.png",
    )
    comp_figure(
        ds.presence[rows].numpy(), ds.comp_target[rows].numpy(), comp_alpha,
        args.n_samples, rng, title,
        out_dir / f"truth_vs_pred_comp_bulk{bulk}.png",
    )
    cond_figure(
        ds.cond_target[rows].numpy(), log_w.numpy(), mu.numpy(), sigma.numpy(),
        rng, title, out_dir / f"truth_vs_pred_cond_bulk{bulk}.png",
    )


if __name__ == "__main__":
    main()
