"""Headline qualitative figures for a joint checkpoint: inference card + scenario view.

Two of the four canonical per-version figures (results/README.md): the
inference card shows one rock end to end (bulk chemistry in, assemblage
posterior and condition posteriors out, truth overlaid), the scenario view
shows the top-2 predicted assemblages of an ambiguous rock with their
condition posteriors. Pure figure/statistic functions; the thin CLIs are
`scripts/view/inference_card.py` and `scripts/view/scenario_view.py`.
`sample_assemblages` is the shared posterior sampler, also used by
`scripts/view/truth_vs_pred.py`.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..data.magemin_corpus import COMP_OXIDES
from ..data.phase_vocab import SOLID_PHASES
from ..models.cond_head import COND_AXES

COND_LABELS = {"T": "temperature (C)", "P_kbar": "pressure (kbar)", "H2O": "water (wt%)"}


def sample_assemblages(p, alpha, n, rng):
    """Sample `n` assemblages: Bernoulli support from `p`, Dirichlet slice of `alpha`."""
    modes = np.zeros((n, len(p)))
    for s in range(n):
        for _ in range(100):
            z = rng.random(len(p)) < p
            if z.any():
                break
        modes[s, z] = rng.dirichlet(alpha[z])
    return modes


def pick_ambiguous_bulk(presence, row_bulk_idx, rung, bulk_ids, rung_sel):
    """Val bulk whose truth draws split most evenly over support patterns.

    Score = share of the second-most-common exact support pattern among the
    bulk's rows at `rung_sel` (0 if unimodal). Returns (bulk_id, score);
    ties go to the lower bulk id via the scan order.
    """
    best_bulk, best_score = int(bulk_ids[0]), -1.0
    for bpos in range(len(bulk_ids)):
        rows = np.flatnonzero((row_bulk_idx == bpos) & (rung == rung_sel))
        if len(rows) == 0:
            continue
        pat = presence[rows].astype(np.uint8)
        keys = np.ascontiguousarray(pat).view(
            np.dtype((np.void, pat.shape[1]))
        ).ravel()
        _, counts = np.unique(keys, return_counts=True)
        if len(counts) < 2:
            score = 0.0
        else:
            score = float(np.sort(counts)[-2] / len(rows))
        if score > best_score:
            best_bulk, best_score = int(bulk_ids[bpos]), score
    return best_bulk, best_score


def top_support_patterns(sampled_modes, k=2):
    """Top-`k` support patterns among sampled assemblages.

    Pattern = the set of phases with nonzero mode in a sample. Returns one
    dict per pattern, most frequent first: `pattern` (bool [P]), `share` of
    samples, `mean_modes` (mean over the pattern's samples, [P]).
    """
    pat = (sampled_modes > 0).astype(np.uint8)
    keys = np.ascontiguousarray(pat).view(np.dtype((np.void, pat.shape[1]))).ravel()
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    out = []
    for gid in np.argsort(-counts)[:k]:
        rows = np.flatnonzero(inverse == gid)
        out.append({
            "pattern": pat[rows[0]].astype(bool),
            "share": float(len(rows) / len(sampled_modes)),
            "mean_modes": sampled_modes[rows].mean(0),
        })
    return out


def _bar_with_whiskers(ax, y, values, color, hatch=None):
    """Horizontal bar at `y` = mean of `values`, whiskers = p10-p90.

    Whisker arms clamp at 0: a mostly-zero sample (rare phase) puts the
    mean outside [p10, p90], which matplotlib rejects as negative error.
    """
    mean = values.mean()
    lo, hi = np.quantile(values, [0.1, 0.9])
    ax.barh(y, mean, height=0.36, color=color, alpha=0.7, hatch=hatch,
            edgecolor="white" if hatch else None)
    ax.errorbar(mean, y, xerr=[[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]],
                fmt="none", ecolor="black", lw=0.9, capsize=2)


def inference_card(
    bulk_11_rows, truth_modes, pred_modes, cond_samples, cond_truth,
    presence_cutoff, title, out_path,
):
    """One rock end to end: chemistry in, posteriors out, truth overlaid.

    `bulk_11_rows` [R, 11] input oxides across the bulk's draws (H2O varies
    per draw, hence whiskers); `truth_modes` [R, P]; `pred_modes` [S, P]
    sampled assemblages; `cond_samples`/`cond_truth` [*, A] pooled posterior
    samples and the bulk's truth draws.
    """
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.3, 1.1], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    for i in range(len(COMP_OXIDES)):
        _bar_with_whiskers(ax, i, bulk_11_rows[:, i], "tab:gray")
    ax.set_yticks(range(len(COMP_OXIDES)), COMP_OXIDES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("mass fraction of the rock (log)")
    ax.set_title("input: bulk chemistry", fontsize=10)

    ax = fig.add_subplot(gs[1])
    show = [
        k for k in range(len(SOLID_PHASES))
        if truth_modes[:, k].mean() > 1e-3 or pred_modes[:, k].mean() > 1e-3
    ]
    show.sort(key=lambda k: -truth_modes[:, k].mean())
    for i, k in enumerate(show):
        _bar_with_whiskers(ax, i - 0.19, pred_modes[:, k], "tab:orange", hatch="//")
        _bar_with_whiskers(ax, i + 0.19, truth_modes[:, k], "tab:blue")
    ax.set_yticks(range(len(show)), [SOLID_PHASES[k] for k in show], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean share of the solids (whiskers p10-p90)")
    ax.set_title("output: assemblage posterior vs truth", fontsize=10)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor="tab:orange", alpha=0.7, hatch="//"),
        plt.Rectangle((0, 0), 1, 1, facecolor="tab:blue", alpha=0.7),
    ], labels=["prediction", "truth"], fontsize=8, loc="lower right")

    inner = gs[2].subgridspec(len(COND_AXES), 1, hspace=0.55)
    for a, axis_name in enumerate(COND_AXES):
        ax = fig.add_subplot(inner[a])
        bins = np.histogram_bin_edges(
            np.concatenate([cond_truth[:, a], cond_samples[:, a]]), bins=35
        )
        ax.hist(cond_truth[:, a], bins=bins, density=True, color="tab:blue",
                alpha=0.35, label="truth draws")
        ax.hist(cond_samples[:, a], bins=bins, density=True, color="tab:orange",
                alpha=0.6, histtype="step", lw=1.4, label="posterior")
        ax.set_yticks([])
        ax.set_title(COND_LABELS[axis_name], fontsize=8, pad=2)
        if a == 0:
            ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def scenario_figure(scenarios, tp_samples, title, out_path):
    """Top-K predicted assemblages of one rock, with per-scenario conditions.

    `scenarios` from `top_support_patterns`; `tp_samples` list of [S, 2]
    (T, P) sample arrays, one per scenario. T and P are sampled
    independently per scenario -- the per-axis mixtures model no cross-axis
    correlation, so the scatter shows each scenario's marginal ranges, not a
    joint density.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )
    cmap = plt.get_cmap("tab20")
    phase_color = {k: cmap(k % 20) for k in range(len(SOLID_PHASES))}
    for i, sc in enumerate(scenarios):
        left = 0.0
        for k in np.flatnonzero(sc["pattern"]):
            f = sc["mean_modes"][k]
            ax1.barh(i, f, left=left, color=phase_color[k], edgecolor="white")
            if f > 0.05:
                ax1.text(left + f / 2, i, SOLID_PHASES[k], ha="center",
                         va="center", fontsize=8)
            left += f
        ax1.text(1.02, i, f"p={sc['share']:.2f}", va="center", fontsize=9)
    ax1.set_yticks(range(len(scenarios)),
                   [f"scenario {i + 1}" for i in range(len(scenarios))])
    ax1.set_xlim(0, 1.18)
    ax1.invert_yaxis()
    ax1.set_xlabel("mean share of the solids")
    ax1.set_title("same rock, alternative predicted assemblages", fontsize=10)

    scen_colors = ["tab:green", "tab:orange", "tab:purple"]
    for i, (sc, tp) in enumerate(zip(scenarios, tp_samples)):
        ax2.scatter(tp[:, 0], tp[:, 1], s=5, alpha=0.35,
                    color=scen_colors[i % len(scen_colors)],
                    label=f"scenario {i + 1} (p={sc['share']:.2f})")
    ax2.set_xlabel(COND_LABELS["T"])
    ax2.set_ylabel(COND_LABELS["P_kbar"])
    ax2.set_title("conditions posterior per scenario", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
