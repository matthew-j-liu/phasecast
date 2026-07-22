"""Headline qualitative figures for a joint checkpoint: inference card + scenario view.

The two per-model headline figures (results/README.md): the inference card
shows one rock end to end (bulk chemistry in, assemblage posterior and
condition posteriors out, truth overlaid), the scenario view shows the
top-2 predicted assemblages of an ambiguous rock with their condition
posteriors. Pure figure/statistic functions; the thin CLIs are
`scripts/view/inference_card.py` and `scripts/view/scenario_view.py`.
`sample_assemblages` is the shared posterior sampler, also used by
`scripts/view/truth_vs_pred.py`.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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


def pick_bulk_by_nll(joint_nll_rows, row_bulk_idx, rung, rung_sel, pick):
    """Bulk position by per-bulk mean hurdle NLL at `rung_sel`.

    `pick` = "best" (lowest mean NLL) or "median" (middle of the sorted
    order) -- median is the representative public-card choice, best the
    internal showcase. Bulks with no rows at the rung are excluded.
    """
    n_bulks = int(row_bulk_idx.max()) + 1
    means = np.full(n_bulks, np.nan)
    for b in range(n_bulks):
        m = (row_bulk_idx == b) & (rung == rung_sel)
        if m.any():
            means[b] = joint_nll_rows[m].mean()
    order = np.argsort(means)
    order = order[~np.isnan(means[order])]
    if len(order) == 0:
        raise ValueError(f"no bulk has rows at rung {rung_sel}")
    if pick == "best":
        return int(order[0])
    if pick == "median":
        return int(order[len(order) // 2])
    raise ValueError(f"pick must be `median` or `best`, got {pick!r}")


def inference_card(
    bulk_11_rows, truth_modes, pred_modes, cond_samples, cond_truth,
    title, out_path,
):
    """One rock end to end, no teacher forcing anywhere.

    `bulk_11_rows` [R, 11] input oxides across the bulk's draws; `truth_modes`
    [R, P] hurdle magnitude targets; `pred_modes` [S, P] assemblages sampled
    from the model's own presence probabilities; `cond_samples` [S, A] from
    CondHead conditioned on the model's own sampled mineral lists;
    `cond_truth` [R, A]. The hurdle renders as two panels -- presence rate,
    then share-when-present -- so absences never dilute the share statistics.
    """
    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.85, 0.55, 1.1, 1.05], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    med = np.median(bulk_11_rows, axis=0)
    ax.barh(range(len(COMP_OXIDES)), med, color="tab:gray", alpha=0.8, height=0.6)
    h2o = COMP_OXIDES.index("H2O")
    lo, hi = np.quantile(bulk_11_rows[:, h2o], [0.1, 0.9])
    ax.errorbar(med[h2o], h2o, xerr=[[med[h2o] - lo], [hi - med[h2o]]],
                fmt="none", ecolor="black", lw=0.9, capsize=2)
    ax.set_yticks(range(len(COMP_OXIDES)), COMP_OXIDES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("mass fraction of the rock (log)", fontsize=9)
    ax.set_title("input: bulk chemistry\n(whiskers: H2O varies by draw)", fontsize=9)

    show = [
        k for k in range(len(SOLID_PHASES))
        if truth_modes[:, k].mean() > 1e-3 or pred_modes[:, k].mean() > 1e-3
    ]
    show.sort(key=lambda k: -truth_modes[:, k].mean())
    genuine = [k for k in show if (truth_modes[:, k] > 0).any()]

    ax_rate = fig.add_subplot(gs[1])
    for i, k in enumerate(show):
        false_positive = k not in genuine
        ax_rate.plot((pred_modes[:, k] > 0).mean(), i - 0.15, "o", ms=4,
                     color="0.55" if false_positive else "tab:orange",
                     mfc="white")
        ax_rate.plot((truth_modes[:, k] > 0).mean(), i + 0.15, "o", ms=4,
                     color="tab:blue")
    ax_rate.set_yticks(range(len(show)), [SOLID_PHASES[k] for k in show], fontsize=8)
    ax_rate.invert_yaxis()
    ax_rate.set_xlim(-0.05, 1.05)
    ax_rate.set_xticks([0, 0.5, 1])
    ax_rate.grid(axis="x", lw=0.3, alpha=0.5)
    ax_rate.set_xlabel("fraction of draws\ncontaining the mineral", fontsize=8)
    ax_rate.set_title("is it there?", fontsize=9)

    ax_share = fig.add_subplot(gs[2])
    for i, k in enumerate(genuine):
        for vals, off, color, filled in (
            (pred_modes[:, k], -0.19, "tab:orange", False),
            (truth_modes[:, k], +0.19, "tab:blue", True),
        ):
            present = vals[vals > 0]
            if len(present):
                m = present.mean()
                q25, q75 = np.quantile(present, [0.25, 0.75])
                ax_share.errorbar(
                    m, i + off, xerr=[[max(m - q25, 0)], [max(q75 - m, 0)]],
                    fmt="o", ms=5, color=color, mfc=color if filled else "white",
                    ecolor=color, lw=1.2, capsize=2.5,
                )
    ax_share.set_yticks(range(len(genuine)),
                        [SOLID_PHASES[k] for k in genuine], fontsize=8)
    ax_share.invert_yaxis()
    ax_share.set_xscale("log")
    ax_share.set_xlim(5e-4, 1.0)
    ax_share.grid(axis="x", lw=0.3, alpha=0.5)
    ax_share.set_xlabel("share of solids when present (log)\nwhiskers p25-p75",
                        fontsize=9)
    ax_share.set_title("how much, when there?", fontsize=9)
    ax_share.legend(handles=[
        Line2D([], [], marker="o", ls="", color="tab:orange", mfc="white",
               label="prediction"),
        Line2D([], [], marker="o", ls="", color="tab:blue", label="truth"),
        Line2D([], [], marker="o", ls="", color="0.55", mfc="white",
               label="predicted but never in truth\n(rate panel only)"),
    ], fontsize=7, loc="lower right")

    inner = gs[3].subgridspec(len(COND_AXES), 1, hspace=0.55)
    for a, axis_name in enumerate(COND_AXES):
        axc = fig.add_subplot(inner[a])
        bins = np.histogram_bin_edges(
            np.concatenate([cond_truth[:, a], cond_samples[:, a]]), bins=35
        )
        axc.hist(cond_truth[:, a], bins=bins, density=True, color="tab:blue",
                 alpha=0.35, label="truth draws")
        axc.hist(cond_samples[:, a], bins=bins, density=True, color="tab:orange",
                 alpha=0.6, histtype="step", lw=1.4,
                 label="posterior (own mineral lists)")
        axc.set_yticks([])
        axc.set_title(COND_LABELS[axis_name], fontsize=8, pad=2)
        if a == 0:
            axc.legend(fontsize=6)
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
