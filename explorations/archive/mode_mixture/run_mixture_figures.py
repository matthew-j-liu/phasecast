"""K=2 mockups of the canonical figures for joint_heads_allrock_mixture_seed0.

Round 1 of the mixture visualization pass, for iteration:
  card_k2_median.png / card_k2_engaged.png -- inference card with the share
      whiskers replaced by split violins of the sampled posterior
  card_k2_components.png -- per-phase component decomposition of the engaged
      bulk on its dominant support (analytic Beta marginals per component)
  beta_overlay_k2_amp.png -- amp K-curves on the most bimodal truth groups
  alpha_concentration_k2.png -- per-component alpha0 + the engagement map
  scenario_view_k2.png -- scenario bars split per magnitude component
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import beta as beta_dist, gaussian_kde

from cnmineral.data.magemin_corpus import COMP_OXIDES
from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.data.phase_vocab import SOLID_PHASES
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import (
    component_separation,
    gate_entropy,
    hurdle_nll_rows,
    magnitude_mixture_rows,
    support_alpha_fn,
)
from cnmineral.eval.summary_cards import (
    LEGEND_FS,
    SUPTITLE_FS,
    TICK_FS,
    TITLE_FS,
    XLABEL_FS,
    pick_bulk_by_nll,
)
from cnmineral.models.cond_head import COND_AXES
from cnmineral.training.joint_heads import JointHeadCheckpoint

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
WORK = BASE / "work"
CACHE = WORK / "mixture_mockup_cache.pkl"
CKPT = Path("checkpoints/joint_heads_allrock_mixture_seed0/heads_best.pt")

PRED, TRUE = "tab:orange", "tab:blue"
COMP_COLORS = ("tab:purple", "tab:green")
COMP_NAMES = ("component A", "component B")
RUNG = 50
N_SAMPLES = 512
SHARE_FLOOR = 5e-4
FLOOR_LOG = np.log10(SHARE_FLOOR)
AMP = SOLID_PHASES.index("amp")


def sample_assemblages_tracked(p, alpha_fn, n, rng):
    """`summary_cards.sample_assemblages`, K>1 path only, returning the drawn
    component per sample so mockups can color by magnitude regime."""
    supports = np.zeros((n, len(p)), dtype=bool)
    for s in range(n):
        for _ in range(100):
            z = rng.random(len(p)) < p
            if z.any():
                break
        supports[s] = z
    uniq, inverse = np.unique(supports, axis=0, return_inverse=True)
    alphas, log_w = alpha_fn(uniq.astype(np.float64))
    gate = np.exp(log_w)
    modes = np.zeros((n, len(p)))
    comp = np.zeros(n, dtype=int)
    for s in range(n):
        z, u = supports[s], inverse[s]
        k = rng.choice(gate.shape[1], p=gate[u] / gate[u].sum())
        comp[s] = k
        modes[s, z] = rng.dirichlet(alphas[u, k][z])
    return modes, comp


def build_cache():
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    ckpt = JointHeadCheckpoint.load(CKPT)
    mode_head = ckpt.build_mode_head()
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
    latents = bulk_latents(encoder, ds, torch.device("cpu"))
    row_bulk = ds.row_bulk_idx.numpy()
    rung = ds.rung.numpy()
    presence = ds.presence.numpy()
    magnitude = ds.magnitude.numpy()
    with torch.no_grad():
        logits_all = mode_head.presence_logits(latents)
    alpha_t, log_w_t = magnitude_mixture_rows(mode_head, latents, row_bulk,
                                              ds.presence)
    alpha, log_w = alpha_t.numpy(), log_w_t.numpy()
    nll = hurdle_nll_rows(logits_all[row_bulk], alpha_t, ds.presence,
                          ds.magnitude, log_w=log_w_t)["joint"]
    ent = gate_entropy(log_w)
    sep = component_separation(alpha, log_w, presence)

    # bulk picks: the representative median card, and the amp-bearing bulk
    # where the gate is most engaged (mean entropy x separation over its rows)
    med_pos = pick_bulk_by_nll(nll, row_bulk, rung, RUNG, "median")
    n_bulks = int(row_bulk.max()) + 1
    engage = np.full(n_bulks, -1.0)
    for b in range(n_bulks):
        m = (row_bulk == b) & (rung == RUNG)
        if m.sum() >= 20 and (presence[m, AMP] > 0).mean() > 0.3:
            engage[b] = float((ent[m] * sep[m]).mean())
    eng_pos = int(np.argmax(engage))

    per_bulk = {}
    rng_np = np.random.default_rng(0)
    for tag, bpos in (("median", med_pos), ("engaged", eng_pos)):
        rows = np.flatnonzero((row_bulk == bpos) & (rung == RUNG))
        p = torch.sigmoid(logits_all[bpos]).numpy()
        pred_modes, comp = sample_assemblages_tracked(
            p, support_alpha_fn(mode_head, latents[bpos]), N_SAMPLES, rng_np)
        supports = torch.as_tensor((pred_modes > 0).astype(np.float32))
        with torch.no_grad():
            lw, mu, sigma = cond_head(latents[bpos].expand(len(supports), -1),
                                      supports)
        w = np.exp(lw.numpy())
        mu_n, sigma_n = mu.numpy(), sigma.numpy()
        cond_samples = np.zeros((len(supports), len(COND_AXES)))
        for a in range(len(COND_AXES)):
            for s in range(len(supports)):
                ci = rng_np.choice(w.shape[-1], p=w[s, a] / w[s, a].sum())
                cond_samples[s, a] = rng_np.normal(mu_n[s, a, ci],
                                                   sigma_n[s, a, ci])
        # dominant true support's mixture params (rows share bulk + support
        # so the teacher-forced alphas are identical within the group)
        pat = presence[rows].astype(np.uint8)
        keys = np.ascontiguousarray(pat).view(
            np.dtype((np.void, pat.shape[1]))).ravel()
        uniq, counts = np.unique(keys, return_counts=True)
        dom = rows[np.flatnonzero(keys == uniq[np.argmax(counts)])]
        per_bulk[tag] = {
            "bulk_id": int(ds.bulk_ids[bpos]),
            "qap": str(ds.bulk_qap_names[bpos]),
            "rows": rows,
            "bulk_11": ds.bulk_11[rows].numpy(),
            "truth_modes": magnitude[rows],
            "pred_modes": pred_modes,
            "comp": comp,
            "cond_samples": cond_samples,
            "cond_truth": ds.cond_target[rows].numpy(),
            "gate_mean": np.exp(log_w[rows]).mean(0),
            "dom_rows": dom,
            "dom_alpha": alpha[dom[0]],
            "dom_log_w": log_w[dom[0]],
        }

    # amp K-curve groups: (bulk, support) groups at the rung with >= 30 rows,
    # amp present, ranked by within-group truth amp spread
    groups = []
    for b in range(n_bulks):
        m = np.flatnonzero((row_bulk == b) & (rung == RUNG))
        if len(m) == 0:
            continue
        pat = presence[m].astype(np.uint8)
        keys = np.ascontiguousarray(pat).view(
            np.dtype((np.void, pat.shape[1]))).ravel()
        for key in np.unique(keys):
            rows_g = m[np.flatnonzero(keys == key)]
            if len(rows_g) < 30 or presence[rows_g[0], AMP] == 0:
                continue
            vals = magnitude[rows_g, AMP]
            groups.append({
                "bulk_id": int(ds.bulk_ids[b]),
                "qap": str(ds.bulk_qap_names[b]),
                "amp_truth": vals,
                "alpha": alpha[rows_g[0]],
                "log_w": log_w[rows_g[0]],
                "support": presence[rows_g[0]] > 0,
                "spread": float(np.std(vals)),
            })
    groups.sort(key=lambda g: -g["spread"])

    data = {
        "n_components": int(alpha.shape[1]),
        "rung_mask": rung == RUNG,
        "presence_sum": presence.sum(1),
        "alpha0": (alpha * presence[:, None, :]).sum(-1),
        "gate_w": np.exp(log_w),
        "entropy": ent,
        "separation": sep,
        "per_bulk": per_bulk,
        "amp_groups": groups[:3],
    }
    WORK.mkdir(exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"cached {CACHE}")
    return data


def violin(ax, values, y, color, side, width=0.36, alpha=0.55, scale=1.0):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or np.ptp(v) < 1e-9:
        ax.plot(np.median(v) if len(v) else np.nan, y, "|", ms=9, color=color,
                mew=1.8)
        return
    grid = np.linspace(v.min(), v.max(), 256)
    dens = gaussian_kde(v)(grid)
    dens = dens / dens.max() * width * scale
    if side == "up":
        ax.fill_between(grid, y, y - dens, color=color, alpha=alpha, lw=0)
    else:
        ax.fill_between(grid, y, y + dens, color=color, alpha=alpha, lw=0)
    ax.plot([np.median(v)], [y], "|", ms=9, color=color, mew=1.8)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def card_k2(bk, name):
    """The inference card with the share panel's whiskers replaced by split
    violins of the sampled posterior (mixture-honest: bimodality shows)."""
    truth, pred, comp = bk["truth_modes"], bk["pred_modes"], bk["comp"]
    fig = plt.figure(figsize=(14.5, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.85, 0.55, 1.25, 1.05],
                          wspace=0.42)

    ax = fig.add_subplot(gs[0])
    med = np.median(bk["bulk_11"], axis=0)
    ax.barh(range(len(COMP_OXIDES)), med, color="tab:gray", alpha=0.8,
            height=0.6)
    h2o = COMP_OXIDES.index("H2O")
    lo, hi = np.quantile(bk["bulk_11"][:, h2o], [0.1, 0.9])
    ax.errorbar(med[h2o], h2o, xerr=[[med[h2o] - lo], [hi - med[h2o]]],
                fmt="none", ecolor="black", lw=0.9, capsize=2)
    ax.set_yticks(range(len(COMP_OXIDES)), COMP_OXIDES, fontsize=TICK_FS)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("mass fraction of the rock (log)", fontsize=XLABEL_FS)
    ax.set_title("Input: Bulk Chemistry", fontsize=TITLE_FS)

    show = [k for k in range(len(SOLID_PHASES))
            if truth[:, k].mean() > 1e-3 or pred[:, k].mean() > 1e-3]
    show.sort(key=lambda k: -truth[:, k].mean())
    genuine = [k for k in show if (truth[:, k] > 0).any()]

    ax_rate = fig.add_subplot(gs[1])
    for i, k in enumerate(show):
        fp = k not in genuine
        ax_rate.plot((pred[:, k] > 0).mean(), i - 0.15, "o", ms=4,
                     color="0.55" if fp else PRED, mfc="white")
        ax_rate.plot((truth[:, k] > 0).mean(), i + 0.15, "o", ms=4, color=TRUE)
    ax_rate.set_yticks(range(len(show)), [SOLID_PHASES[k] for k in show],
                       fontsize=TICK_FS)
    ax_rate.invert_yaxis()
    ax_rate.set_xlim(-0.05, 1.05)
    ax_rate.set_xticks([0, 0.5, 1])
    ax_rate.grid(axis="x", lw=0.3, alpha=0.5)
    ax_rate.set_xlabel("fraction of draws\ncontaining the mineral",
                       fontsize=XLABEL_FS)
    ax_rate.set_title("Phase Presence Rate", fontsize=TITLE_FS)

    ax_sh = fig.add_subplot(gs[2])
    for i, k in enumerate(genuine):
        pv = pred[:, k][pred[:, k] > 0]
        tv = truth[:, k][truth[:, k] > 0]
        violin(ax_sh, np.log10(np.maximum(pv, SHARE_FLOOR)), i, PRED, "up")
        violin(ax_sh, np.log10(np.maximum(tv, SHARE_FLOOR)), i, TRUE, "down")
    ax_sh.set_yticks(range(len(genuine)), [SOLID_PHASES[k] for k in genuine],
                     fontsize=TICK_FS)
    ax_sh.set_ylim(len(genuine) - 0.4, -0.7)
    ax_sh.set_xlim(FLOOR_LOG, 0.05)
    ticks = [0.001, 0.01, 0.1, 0.5]
    ax_sh.set_xticks(np.log10(ticks), ["0.1%", "1%", "10%", "50%"],
                     fontsize=TICK_FS)
    ax_sh.grid(axis="x", lw=0.3, alpha=0.5)
    ax_sh.set_xlabel("share of solids when present (log)\n"
                     "full sampled distribution, bar = median",
                     fontsize=XLABEL_FS)
    w = bk["gate_mean"]
    ax_sh.set_title("Phase Abundance When Present", fontsize=TITLE_FS)
    ax_sh.text(0.98, 0.02, f"gate: A {w[0]:.2f} / B {w[1]:.2f}",
               transform=ax_sh.transAxes, ha="right", va="bottom", fontsize=9,
               color="0.3")
    ax_sh.legend(handles=[
        Patch(color=PRED, alpha=0.55, label="prediction"),
        Patch(color=TRUE, alpha=0.55, label="truth draws"),
    ], fontsize=LEGEND_FS - 1, loc="upper left")

    inner = gs[3].subgridspec(len(COND_AXES), 1, hspace=0.55)
    labels = {"T": "temperature (C)", "P_kbar": "pressure (kbar)",
              "H2O": "water (wt%)"}
    for a, axis_name in enumerate(COND_AXES):
        axc = fig.add_subplot(inner[a])
        bins = np.histogram_bin_edges(
            np.concatenate([bk["cond_truth"][:, a], bk["cond_samples"][:, a]]),
            bins=35)
        axc.hist(bk["cond_truth"][:, a], bins=bins, density=True, color=TRUE,
                 alpha=0.35)
        axc.hist(bk["cond_samples"][:, a], bins=bins, density=True, color=PRED,
                 alpha=0.6, histtype="step", lw=1.4)
        axc.set_yticks([])
        axc.set_title(labels[axis_name], fontsize=TITLE_FS, pad=2)
        if a == len(COND_AXES) - 1:
            axc.legend(handles=[
                Patch(facecolor=TRUE, alpha=0.35, label="truth draws"),
                Line2D([], [], color=PRED, lw=1.4, label="posterior"),
            ], fontsize=LEGEND_FS, ncol=2, loc="upper center",
                bbox_to_anchor=(0.5, -0.45), frameon=False)
    frac_b = comp.mean()
    fig.suptitle(f"{bk['qap']} (bulk {bk['bulk_id']}), {RUNG}%-melt rung, "
                 f"K=2: predicted posterior vs {len(bk['rows'])}-draw truth; "
                 f"{1 - frac_b:.0%} of samples drew component A, "
                 f"{frac_b:.0%} component B", fontsize=SUPTITLE_FS)
    save(fig, name)


def log_beta_density(grid_log, a, b):
    """Beta pdf reexpressed on the log10 axis the cards plot on."""
    x = 10.0 ** grid_log
    dens = beta_dist.pdf(x, a, b) * x * np.log(10)
    return np.nan_to_num(dens)


def components_figure(bk):
    """The engaged bulk's dominant support, decomposed: analytic per-component
    Beta marginals (violin bodies) against the truth draws on that support."""
    alpha, log_w = bk["dom_alpha"], bk["dom_log_w"]
    rows = bk["dom_rows"]
    # dom_rows index into the full dataset; recover their positions in the
    # bulk's row block to reuse the cached truth slice
    pos = np.array([np.flatnonzero(bk["rows"] == r)[0] for r in rows])
    truth_dom = bk["truth_modes"][pos]
    support = truth_dom[0] > 0
    phases = np.flatnonzero(support)
    phases = phases[np.argsort(-truth_dom[:, phases].mean(0))]
    w = np.exp(log_w)
    a0 = (alpha * support).sum(-1)

    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(phases) + 1.8))
    grid = np.linspace(FLOOR_LOG, 0.0, 500)
    for i, k in enumerate(phases):
        for c in range(alpha.shape[0]):
            dens = log_beta_density(grid, alpha[c, k], a0[c] - alpha[c, k])
            if dens.max() > 0:
                dens = dens / dens.max() * 0.38 * max(w[c], 0.25)
                ax.fill_between(grid, i, i - dens, color=COMP_COLORS[c],
                                alpha=0.45, lw=0)
        tv = np.log10(np.maximum(truth_dom[:, k], SHARE_FLOOR))
        violin(ax, tv, i, TRUE, "down", width=0.30)
    ax.set_yticks(range(len(phases)), [SOLID_PHASES[k] for k in phases],
                  fontsize=TICK_FS)
    ax.set_ylim(len(phases) - 0.4, -0.8)
    ticks = [0.001, 0.01, 0.1, 0.5, 1.0]
    ax.set_xticks(np.log10(ticks), ["0.1%", "1%", "10%", "50%", "100%"],
                  fontsize=TICK_FS)
    ax.set_xlim(FLOOR_LOG, 0.05)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.set_xlabel("share of solids (log); component bodies scaled by gate "
                  "weight", fontsize=XLABEL_FS)
    ax.legend(handles=[
        Patch(color=COMP_COLORS[0], alpha=0.45,
              label=f"{COMP_NAMES[0]} (w={w[0]:.2f})"),
        Patch(color=COMP_COLORS[1], alpha=0.45,
              label=f"{COMP_NAMES[1]} (w={w[1]:.2f})"),
        Patch(color=TRUE, alpha=0.55, label="truth draws"),
    ], fontsize=LEGEND_FS - 1, loc="upper left")
    ax.set_title(f"{bk['qap']} (bulk {bk['bulk_id']}), dominant support "
                 f"({len(rows)} of {len(bk['rows'])} draws): the two magnitude "
                 f"regimes the gate mixes", fontsize=TITLE_FS)
    save(fig, "card_k2_components.png")


def beta_overlay_figure(groups):
    """amp K-curves: truth histogram + weighted component Betas + mixture."""
    fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 3.6),
                             sharey=False)
    grid = np.linspace(FLOOR_LOG, 0.0, 500)
    for ax, g in zip(np.atleast_1d(axes), groups):
        alpha, w = g["alpha"], np.exp(g["log_w"])
        a0 = (alpha * g["support"]).sum(-1)
        tv = np.log10(np.maximum(g["amp_truth"], SHARE_FLOOR))
        ax.hist(tv, bins=24, density=True, color=TRUE, alpha=0.35,
                label=f"truth ({len(tv)} draws)")
        mix = np.zeros_like(grid)
        for c in range(alpha.shape[0]):
            dens = log_beta_density(grid, alpha[c, AMP], a0[c] - alpha[c, AMP])
            mix += w[c] * dens
            ax.plot(grid, w[c] * dens, color=COMP_COLORS[c], lw=1.3, ls="--",
                    label=f"{COMP_NAMES[c]} (w={w[c]:.2f})")
        ax.plot(grid, mix, color=PRED, lw=1.8, label="mixture")
        ticks = [0.001, 0.01, 0.1, 0.5]
        ax.set_xticks(np.log10(ticks), ["0.1%", "1%", "10%", "50%"],
                      fontsize=TICK_FS)
        ax.set_xlim(FLOOR_LOG, 0.0)
        ax.set_yticks([])
        ax.set_xlabel("amp share of solids (log)", fontsize=XLABEL_FS)
        ax.set_title(f"{g['qap']} (bulk {g['bulk_id']})", fontsize=TITLE_FS)
        ax.legend(fontsize=LEGEND_FS - 3, loc="upper left")
    fig.suptitle("amp magnitude under the mixture, one support group each: "
                 "gate-weighted component Betas (dashed) and their sum "
                 "(solid), groups picked by truth spread", fontsize=SUPTITLE_FS)
    fig.tight_layout()
    save(fig, "beta_overlay_k2_amp.png")


def concentration_figure(data):
    """Per-component sharpness and the engagement map, val rows at the rung."""
    m = data["rung_mask"] & (data["presence_sum"] > 1)
    alpha0, gate = data["alpha0"][m], data["gate_w"][m]
    ent, sep = data["entropy"][m], data["separation"][m]
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.5, 3.9))

    bins = np.linspace(0, np.quantile(alpha0, 0.99), 60)
    for c in range(alpha0.shape[1]):
        ax0.hist(alpha0[:, c], bins=bins, density=True, histtype="step",
                 lw=1.6, color=COMP_COLORS[c], label=COMP_NAMES[c])
    ax0.set_xlabel("alpha0 on the true support", fontsize=XLABEL_FS)
    ax0.set_yticks([])
    ax0.set_title("Component Concentration\n(sharpness of each regime)",
                  fontsize=TITLE_FS)
    ax0.legend(fontsize=LEGEND_FS - 1)

    ax1.hist(gate[:, 1], bins=50, density=True, color="tab:gray", alpha=0.8)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel(f"gate weight on {COMP_NAMES[1]}", fontsize=XLABEL_FS)
    ax1.set_yticks([])
    ax1.set_title("Gate Weight Distribution\n(0.5 = maximally hedged)",
                  fontsize=TITLE_FS)

    h = ax2.hist2d(ent, sep, bins=60, cmap="Greys", cmin=1)
    fig.colorbar(h[3], ax=ax2, label="rows")
    ax2.set_xlabel("gate entropy (nats, max 0.693)", fontsize=XLABEL_FS)
    ax2.set_ylabel("component separation (L1)", fontsize=XLABEL_FS)
    ax2.set_title("Engagement Map\n(top-right = mixture doing real work)",
                  fontsize=TITLE_FS)
    fig.suptitle(f"mixture engagement, val rows at the {RUNG}%-melt rung "
                 f"(teacher-forced supports, n={int(m.sum())})",
                 fontsize=SUPTITLE_FS)
    fig.tight_layout()
    save(fig, "alpha_concentration_k2.png")


def scenario_figure_k2(bk):
    """Scenario view where each support scenario splits into its two
    magnitude regimes: one thin stacked bar per component."""
    pred, comp, cond = bk["pred_modes"], bk["comp"], bk["cond_samples"]
    pat = (pred > 0).astype(np.uint8)
    keys = np.ascontiguousarray(pat).view(
        np.dtype((np.void, pat.shape[1]))).ravel()
    uniq, inverse, counts = np.unique(keys, return_inverse=True,
                                      return_counts=True)
    top = np.argsort(-counts)[:2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4),
                                   gridspec_kw={"width_ratios": [1.5, 1.0]})
    cmap = plt.get_cmap("tab20")
    phase_color = {k: cmap(k % 20) for k in range(len(SOLID_PHASES))}
    scen_colors = ["tab:green", "tab:orange"]
    yticks, ylabels = [], []
    for i, gid in enumerate(top):
        rows = np.flatnonzero(inverse == gid)
        share = len(rows) / len(pred)
        for c in range(2):
            rows_c = rows[comp[rows] == c]
            y = i * 2.6 + c
            yticks.append(y)
            w_c = len(rows_c) / max(len(rows), 1)
            ylabels.append(f"scenario {i + 1}{'AB'[c]}\n"
                           f"({COMP_NAMES[c][-1]}, w={w_c:.2f})")
            if len(rows_c) == 0:
                continue
            mean_modes = pred[rows_c].mean(0)
            left = 0.0
            for k in np.flatnonzero(pat[rows[0]]):
                f = mean_modes[k]
                ax1.barh(y, f, left=left, height=0.8, color=phase_color[k],
                         edgecolor="white")
                if f > 0.05:
                    ax1.text(left + f / 2, y, SOLID_PHASES[k], ha="center",
                             va="center", fontsize=8)
                left += f
        ax1.text(1.02, i * 2.6 + 0.5, f"p={share:.2f}", va="center",
                 fontsize=9)
    ax1.set_yticks(yticks, ylabels, fontsize=8)
    ax1.set_xlim(0, 1.18)
    ax1.invert_yaxis()
    ax1.set_xlabel("mean share of the solids", fontsize=XLABEL_FS)
    ax1.set_title("same rock: each support scenario x its two magnitude "
                  "regimes", fontsize=TITLE_FS - 1)

    for i, gid in enumerate(top):
        rows = np.flatnonzero(inverse == gid)
        share = len(rows) / len(pred)
        ax2.scatter(cond[rows, 0], cond[rows, 1], s=5, alpha=0.35,
                    color=scen_colors[i],
                    label=f"scenario {i + 1} (p={share:.2f})")
    ax2.set_xlabel("temperature (C)", fontsize=XLABEL_FS)
    ax2.set_ylabel("pressure (kbar)", fontsize=XLABEL_FS)
    ax2.set_title("conditions posterior per scenario", fontsize=TITLE_FS - 1)
    ax2.legend(fontsize=8)
    fig.suptitle(f"{bk['qap']} (bulk {bk['bulk_id']}), K=2 scenario view: "
                 "support scenarios split by drawn component; identical bars "
                 "= the regimes coincide on that support", fontsize=SUPTITLE_FS)
    fig.tight_layout()
    save(fig, "scenario_view_k2.png")


def main():
    data = build_cache()
    print(f"K={data['n_components']}; engaged bulk "
          f"{data['per_bulk']['engaged']['bulk_id']} "
          f"({data['per_bulk']['engaged']['qap']}), median bulk "
          f"{data['per_bulk']['median']['bulk_id']} "
          f"({data['per_bulk']['median']['qap']})")
    card_k2(data["per_bulk"]["median"], "card_k2_median.png")
    card_k2(data["per_bulk"]["engaged"], "card_k2_engaged.png")
    components_figure(data["per_bulk"]["engaged"])
    beta_overlay_figure(data["amp_groups"])
    concentration_figure(data)
    scenario_figure_k2(data["per_bulk"]["engaged"])


if __name__ == "__main__":
    main()
