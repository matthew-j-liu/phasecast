"""Interim stand-in for the stable inference API (issue #70): the validation,
sampling and drawing that `inference.ipynb` calls, kept out of the notebook so
it reads as the four questions an application needs answered.  `predict` is
the call #70 promotes into the package; nothing here is production code."""

import cnmineral  # first: sets KMP_DUPLICATE_LIB_OK before torch loads (macOS OpenMP collision)

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, NullLocator
from scipy.stats import gaussian_kde

from cnmineral.data.bulk_chemistry import IRON_SOURCE_COLS, MAJOR_TOKEN_TO_COL, TRACE_TOKEN_TO_COL, tokenize_bulk_frame
from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS, MODEL_TO_GROUP, PHASE_INDEX
from cnmineral.data.tokens import VOCAB
from cnmineral.eval.real_rock import coarsen_alpha, coarsen_presence
from cnmineral.models.cond_head import COND_AXES

MAJOR_COLS = [c for c in MAJOR_TOKEN_TO_COL.values() if c != "cr2o3"]
TRACE_COLS = list(TRACE_TOKEN_TO_COL.values())
OXIDE_LABEL = {"sio2": "SiO2", "al2o3": "Al2O3", "cao": "CaO", "mgo": "MgO",
               "k2o": "K2O", "na2o": "Na2O", "tio2": "TiO2"}
# the pretraining corpus kept rows with at least this many populated tokens
# (`min_populated_tokens` in configs/pretrain/pretrain_allrock_redox.yaml)
MIN_ELEMENTS = 4
GROUP_NAME = {"fsp": "feldspar", "px": "pyroxene", "amp": "amphibole", "bi": "biotite",
              "ol": "olivine", "opq": "Fe-Ti oxides", "q": "quartz"}
GROUP_CODE = {name: code for code, name in GROUP_NAME.items()}


class Heads(NamedTuple):
    mode: torch.nn.Module
    cond: torch.nn.Module


def check_input(rocks):
    """`(errors, warnings)` for a composition table, every problem at once so a
    form can show them together.  An error refuses that rock; a warning lets
    it through with the message beside its result."""
    errors, warnings = [], []
    src, _ = tokenize_bulk_frame(rocks)
    oxides = rocks.reindex(columns=[*MAJOR_COLS, *IRON_SOURCE_COLS])
    traces = rocks.reindex(columns=TRACE_COLS)
    for i, rock in enumerate(rocks.index):
        ox, tr = oxides.loc[rock], traces.loc[rock]
        n_errors = len(errors)
        if (src[i] > 0).sum() < MIN_ELEMENTS:
            errors.append(f"{rock}: enter at least {MIN_ELEMENTS} measured elements.")
        for c in ox.index[(ox < 0) | (ox > 100)]:
            errors.append(f"{rock}: {c} is outside the valid range (0-100 wt%).")
        for c in tr.index[(tr < 0) | (tr > 1e6)]:
            errors.append(f"{rock}: {c} is outside the valid range (0-1,000,000 ppm).")
        if len(errors) > n_errors:
            continue
        missing = [OXIDE_LABEL[c] for c in MAJOR_COLS if not ox[c] > 0]
        if not (ox[list(IRON_SOURCE_COLS)] > 0).any():
            missing.append("iron (FeO or Fe2O3)")
        if missing:
            warnings.append(f"{rock}: {', '.join(missing)} not measured. The prediction uses the "
                            "elements provided and has not been checked against rocks with incomplete majors.")
        # iron counted once, in the tokenizer's order of preference
        iron = next((ox[c].sum() for c in (["feo", "fe2o3"], ["feo_tot"], ["fe2o3_tot"]) if (ox[c] > 0).any()), 0.0)
        total = ox[MAJOR_COLS].sum() + iron
        if not 85 <= total <= 103:
            warnings.append(f"{rock}: major oxides sum to {total:.1f} wt% (expected 85-103). "
                            "Check units (wt%, not a fraction) and missing values.")
    return errors, warnings


def _posterior(heads, latent_row, p21_row, n, seed, top_k=5):
    rng = np.random.default_rng(seed)
    draws = (rng.random((n, len(p21_row))) < p21_row).astype(np.float64)
    support = torch.as_tensor(draws, dtype=torch.float32)
    with torch.no_grad():
        alpha21 = heads.mode.magnitude_alpha(latent_row.expand(n, -1), support).numpy()
        log_w, mu, sigma = heads.cond(latent_row.expand(n, -1), support)
    alpha7, _ = coarsen_alpha(alpha21, draws)

    present = np.zeros((n, len(GROUPS)), dtype=bool)
    for phase, group in MODEL_TO_GROUP.items():
        present[:, GROUP_INDEX[group]] |= draws[:, PHASE_INDEX[phase]] > 0
    gamma = rng.gamma(np.clip(alpha7, 1e-9, None)) * present
    total = gamma.sum(1, keepdims=True)
    share = np.divide(gamma, total, out=np.zeros_like(gamma), where=total > 0) * 100

    keys = pd.Series([tuple(row) for row in present])
    counts = keys.value_counts()
    scenarios = []
    for key, count in counts.head(top_k).items():
        rows = (keys == key).to_numpy()
        scenarios.append({"groups": [GROUP_NAME[g] for g, on in zip(GROUPS, key) if on],
                          "prob": round(count / n, 3),
                          "share": np.round(share[rows].mean(0), 1).tolist(),
                          "share_p25": np.round(np.percentile(share[rows], 25, axis=0), 1).tolist(),
                          "share_p75": np.round(np.percentile(share[rows], 75, axis=0), 1).tolist()})
    rest = counts.iloc[top_k:]
    other = {"count": len(rest), "prob": round(rest.sum() / n, 3),
             "max_prob": round(rest.max() / n, 3) if len(rest) else 0.0}

    w = np.exp(log_w.numpy())
    w /= w.sum(-1, keepdims=True)
    mu, sigma = mu.numpy(), sigma.numpy()
    conditions = {}
    for a, axis in enumerate(COND_AXES):
        component = np.minimum((rng.random(n)[:, None] > w[:, a].cumsum(1)).sum(1), w.shape[-1] - 1)
        conditions[axis] = np.round(rng.normal(mu[np.arange(n), a, component], sigma[np.arange(n), a, component]), 1).tolist()

    return {
        "presence": np.round(coarsen_presence(p21_row[None])[0], 3).tolist(),
        "share_mean": np.round(share.mean(0), 1).tolist(),
        "share_p25": np.round(np.percentile(share, 25, axis=0), 1).tolist(),
        "share_p75": np.round(np.percentile(share, 75, axis=0), 1).tolist(),
        "scenarios": scenarios,
        "other_scenarios": other,
        "conditions": conditions,
    }


def predict(rocks, encoder, heads, n_samples=512, seed=0):
    """One JSON-ready record per rock: the input chemistry, then the zero-shot
    posterior with its scenarios and sampled conditions."""
    src, frac = tokenize_bulk_frame(rocks)
    with torch.no_grad():
        latent = encoder(src=torch.from_numpy(src), frac=torch.from_numpy(frac))
        p21 = torch.sigmoid(heads.mode.presence_logits(latent)).numpy()

    records = []
    for i, rock in enumerate(rocks.index):
        records.append({
            "rock": str(rock),
            "input_cation_wt_pct": {sym: round(float(frac[i, k]) * 100, 5)
                                    for k, (sym, _) in enumerate(VOCAB) if src[i, k] > 0},
            "groups": [GROUP_NAME[g] for g in GROUPS],
            "posterior": _posterior(heads, latent[i], p21[i], n_samples, seed),
        })
    return records


# The card follows the paper's inference card (Fig. 2) for the chemistry,
# presence and abundance panels and the overview deck's output card for the
# assemblages and conditions panels. The paper's figure code under
# paper/figures/ is not tracked, so its conventions are carried here: navy is
# the model everywhere, the input ramp is a chroma-boosted cividis, axes are a
# closed box with outward ticks and one minor tick between majors.
INK = "#0b0b0b"
NEUTRAL = "#6b6a66"
PREDICTED = "#002657"
GRADIENT = [
    "#002657", "#002857", "#002D58", "#0D3958", "#1F4658", "#31535A",
    "#435F57", "#506952", "#5E724F", "#6C7B4A", "#7A8547", "#898F42",
    "#97983E", "#A4A03A", "#B1A933", "#BDB22D", "#CAB928", "#D6C022",
    "#E2C91C", "#F0D117", "#FCD911",
]
# the deck's cividis-as-categorical mineral palette (quartz, feldspar, biotite,
# Fe-Ti oxides are its four) extended to the seven groups with the steps that
# keep likely neighbours in a bar apart; segments are labelled, so identity
# never rests on color alone
MINERAL_COLOR = {"feldspar": "#575D6D", "pyroxene": "#123570", "amphibole": "#FEE838",
                 "biotite": "#A59C74", "olivine": "#8A8678", "Fe-Ti oxides": "#E1CC55",
                 "quartz": "#00224E"}
LIGHT_MINERALS = {"amphibole", "Fe-Ti oxides"}
DENSITY = LinearSegmentedColormap.from_list("navy", ["#DCE3EF", "#8A9BB8", "#3B496C", "#00224E"])

CARD_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": INK,
    "axes.linewidth": 1.3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "legend.frameon": False,
    "mathtext.default": "regular",
    "figure.dpi": 150,
}


def _frame(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(which="both", top=False, right=False)


def _chemistry_panel(ax, chem):
    """Cation wt% as log10 on a linear axis, as the paper draws it."""
    chem = sorted(chem.items(), key=lambda kv: -kv[1])
    colors = [GRADIENT[int(round(i * (len(GRADIENT) - 1) / (len(chem) - 1)))] for i in range(len(chem))]
    base = -4.05
    values = np.log10(np.clip([v for _, v in chem], 1e-4, None)) - base
    ax.barh(np.arange(len(chem)), values, left=base, color=colors, height=0.62, alpha=0.7)
    ax.set_yticks(np.arange(len(chem)), [k for k, _ in chem])
    ax.invert_yaxis()
    ax.set_xlim(base, 1.8)
    ax.set_xticks([-4, -2, 0], ["$10^{-4}$", "$10^{-2}$", "$10^{0}$"])


def _presence_panel(ax, presence):
    ax.plot(presence, np.arange(len(presence)), "o", ms=6.0, color=PREDICTED, zorder=4)
    ax.axvline(0, color=INK, lw=1.0, ls=":", zorder=0)
    ax.set_xlim(-0.07, 1.05)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])


def _abundance_panel(ax, mean, lo, hi):
    """Share on a linear axis so one rock's column reads as a mass budget; a
    group absent from most sampled assemblages has both quantiles at zero and
    shows a point with no band."""
    for i, (m, l, h) in enumerate(zip(mean, lo, hi)):
        if h > l:
            ax.add_patch(Rectangle((l / 100, i - 0.28), (h - l) / 100, 0.56,
                                   facecolor=PREDICTED, alpha=0.18, edgecolor="none", zorder=1))
        ax.plot(m / 100, i, "o", ms=6.0, color=PREDICTED, zorder=4)
    ax.axvline(0, color=INK, lw=1.0, ls=":", zorder=0)
    ax.set_xlim(-0.05, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])


def _assemblages_panel(ax, scenarios, other, groups):
    """The deck's stacked bars: one per scenario, its probability at the end,
    and a line for the assemblages past the top few."""
    for k, sc in enumerate(scenarios):
        y, left = len(scenarios) - 1 - k, 0.0
        for g in sc["groups"]:
            s = sc["share"][groups.index(g)] / 100
            ax.barh(y, s, left=left, height=0.55, color=MINERAL_COLOR[g], edgecolor="white", lw=0.7)
            if s > 0.07:
                ax.text(left + s / 2, y, GROUP_CODE[g], ha="center", va="center", fontsize=6.5,
                        color=INK if g in LIGHT_MINERALS else "white")
            left += s
        ax.text(left + 0.03, y, f"{sc['prob']:.0%}", va="center", ha="left", color=INK)
    if other["count"]:
        ax.text(0, -0.9, f"{other['count']} other assemblages, {other['prob']:.0%} combined, "
                f"none above {other['max_prob']:.0%}", va="center", ha="left", color=NEUTRAL)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-1.2, len(scenarios) - 0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _conditions_panel(ax, conditions):
    """The deck's kernel density over the sampled (P, T) pairs.  The pressure
    axis starts at 0: the condition head's Gaussian tails put a few samples
    just below it, which are kept in the density but not drawn."""
    pressure, temperature = np.asarray(conditions["P_kbar"]), np.asarray(conditions["T"])
    t_lo, t_hi = np.floor(temperature.min() / 50) * 50, np.ceil(temperature.max() / 50) * 50
    kde = gaussian_kde(np.vstack([pressure, temperature]))
    pp, tt = np.meshgrid(np.linspace(0, 14, 200), np.linspace(t_lo, t_hi, 200))
    z = kde(np.vstack([pp.ravel(), tt.ravel()])).reshape(pp.shape)
    ax.contourf(pp, tt, z, levels=np.linspace(z.max() * 0.10, z.max(), 6), cmap=DENSITY, extend="max")
    ax.set_xlim(0, 14)
    ax.set_ylim(t_lo, t_hi)
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel("P (kbar)")
    ax.set_ylabel("T (\N{DEGREE SIGN}C)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def card(record):
    """The per-rock figure, drawn from the record alone.

    Panels (a) to (c) are the paper's inference card: the input chemistry,
    presence and abundance. Panels (d) and (e) are the deck's output card: the
    ranked assemblages and the conditions density. Water content stays in the
    record but is not drawn.
    """
    groups = record["groups"]
    codes = [GROUP_CODE[g] for g in groups]
    post = record["posterior"]
    with plt.rc_context(CARD_RC):
        fig = plt.figure(figsize=(5.5, 5.6))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.9, 1.2], height_ratios=[1.2, 1],
                              wspace=0.42, hspace=0.5, left=0.09, right=0.97, top=0.91, bottom=0.09)
        fig.suptitle(record["rock"], x=0.09, y=0.985, ha="left", fontsize=9)

        ax_chem = fig.add_subplot(gs[0, 0])
        _chemistry_panel(ax_chem, record["input_cation_wt_pct"])
        ax_chem.set_xlabel("Cation wt%")
        ax_chem.set_title("(a) Geochemical input")

        ax_pres, ax_abund = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
        _presence_panel(ax_pres, post["presence"])
        _abundance_panel(ax_abund, post["share_mean"], post["share_p25"], post["share_p75"])
        ax_pres.set_title("(b) Presence")
        ax_abund.set_title("(c) Abundance")
        ax_pres.set_xlabel("Presence probability")
        ax_abund.set_xlabel("Fraction of crystalline mass")
        for ax in (ax_pres, ax_abund):
            ax.set_yticks(np.arange(len(codes)), codes)
            ax.set_ylim(len(codes) - 0.5, -0.5)
        for ax in (ax_chem, ax_pres, ax_abund):
            _frame(ax)

        ax_scen = fig.add_subplot(gs[1, :2])
        _assemblages_panel(ax_scen, post["scenarios"], post["other_scenarios"], groups)
        ax_scen.set_title("(d) Possible assemblages", loc="left")
        ax_cond = fig.add_subplot(gs[1, 2])
        _conditions_panel(ax_cond, post["conditions"])
        ax_cond.set_title("(e) Formation conditions")
    return fig
