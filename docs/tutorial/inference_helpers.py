"""Interim stand-in for the stable inference API (issue #70): the sampling,
recalibration and drawing that `inference.ipynb` calls, kept out of the
notebook so it reads as the four questions an application needs answered.
`predict` is the call #70 promotes into the package; nothing here is
production code."""

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
from cnmineral.data.magemin_corpus import COMP_OXIDES
from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS, MODEL_TO_GROUP, PHASE_INDEX, SOLID_PHASES
from cnmineral.data.tokens import VOCAB
from cnmineral.eval.latent import bulk_latents
from cnmineral.eval.real_rock import coarsen_alpha, coarsen_presence, load_benchmark, split_mask
from cnmineral.models.cond_head import COND_AXES
from cnmineral.training.real_rock_recalibrator import build_group_predictions, fit_recalibrator, lift_group_presence_to_phases

MAJOR_COLS = [c for c in MAJOR_TOKEN_TO_COL.values() if c != "cr2o3"]
TRACE_COLS = list(TRACE_TOKEN_TO_COL.values())
GROUP_NAME = {"fsp": "feldspar", "px": "pyroxene", "amp": "amphibole", "bi": "biotite",
              "ol": "olivine", "opq": "Fe-Ti oxides", "q": "quartz"}
GROUP_CODE = {name: code for code, name in GROUP_NAME.items()}


class Heads(NamedTuple):
    mode: torch.nn.Module
    comp: torch.nn.Module
    cond: torch.nn.Module


def check_input(rocks):
    """Problems with a composition table, all of them, so a form can show them at once."""
    problems = []
    oxides = rocks.reindex(columns=[*MAJOR_COLS, *IRON_SOURCE_COLS])
    traces = rocks.reindex(columns=TRACE_COLS)
    for rock in rocks.index:
        blank = [c for c in MAJOR_COLS if not oxides.loc[rock, c] > 0]
        if blank:
            problems.append(f"{rock}: no value for {', '.join(blank)}")
        if not (oxides.loc[rock, list(IRON_SOURCE_COLS)] > 0).any():
            problems.append(f"{rock}: no iron (feo + fe2o3, feo_tot, or fe2o3_tot)")
        if ((oxides.loc[rock] < 0) | (oxides.loc[rock] > 100)).any():
            problems.append(f"{rock}: an oxide outside 0-100 wt%")
        if ((traces.loc[rock] < 0) | (traces.loc[rock] > 1e6)).any():
            problems.append(f"{rock}: a trace element outside 0-1e6 ppm")
        total = oxides.loc[rock].sum()
        if not 85 <= total <= 103:
            problems.append(f"{rock}: major-oxide total {total:.1f} wt% looks wrong")
    return problems


def fit_production_recalibrator(encoder, mode_head):
    """The matrix recalibrator on the benchmark's 372-rock train split, the production fit."""
    benchmark = load_benchmark()
    with torch.no_grad():
        predictions = build_group_predictions(mode_head, benchmark, bulk_latents(encoder, benchmark, torch.device("cpu")))
    fitting = split_mask(benchmark, "train") & benchmark.clean & benchmark.has_magnitude
    return fit_recalibrator(predictions.subset(fitting), tier="matrix", weight_decay=1.0).eval()


def _posterior(heads, latent_row, p21_row, correct_alpha7, n, seed, top_k=3):
    rng = np.random.default_rng(seed)
    draws = (rng.random((n, len(p21_row))) < p21_row).astype(np.float64)
    support = torch.as_tensor(draws, dtype=torch.float32)
    with torch.no_grad():
        alpha21 = heads.mode.magnitude_alpha(latent_row.expand(n, -1), support).numpy()
        log_w, mu, sigma = heads.cond(latent_row.expand(n, -1), support)
    alpha7, _ = coarsen_alpha(alpha21, draws)
    if correct_alpha7 is not None:
        alpha7 = correct_alpha7(alpha7)

    present = np.zeros((n, len(GROUPS)), dtype=bool)
    for phase, group in MODEL_TO_GROUP.items():
        present[:, GROUP_INDEX[group]] |= draws[:, PHASE_INDEX[phase]] > 0
    gamma = rng.gamma(np.clip(alpha7, 1e-9, None)) * present
    total = gamma.sum(1, keepdims=True)
    share = np.divide(gamma, total, out=np.zeros_like(gamma), where=total > 0) * 100

    keys = pd.Series([tuple(row) for row in present])
    scenarios = []
    for key, count in keys.value_counts().head(top_k).items():
        rows = (keys == key).to_numpy()
        scenarios.append({"groups": [GROUP_NAME[g] for g, on in zip(GROUPS, key) if on],
                          "prob": round(count / n, 3), "share": np.round(share[rows].mean(0), 1).tolist()})

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
        "conditions": conditions,
    }


def _corrector(recalibrator, logits7_row):
    def correct(alpha7):
        with torch.no_grad():
            _, corrected, _ = recalibrator(logits7_row.expand(len(alpha7), -1), torch.as_tensor(alpha7, dtype=torch.float64))
        return corrected.numpy()
    return correct


def _composition(alpha_row, p21_row, top=3):
    phases = [SOLID_PHASES[k] for k in np.argsort(-p21_row)[:top]]
    mean = alpha_row.astype(float) / alpha_row.sum(-1, keepdims=True) * 100
    table = pd.DataFrame(mean, index=SOLID_PHASES, columns=COMP_OXIDES).loc[phases]
    # the simulator carries ferric iron as excess oxygen; recast to Fe2O3 for display
    table["Fe2O3"] = table["O"] * 159.688 / 15.999
    table["FeO"] = (table["FeO"] - table["O"] * 2 * 71.844 / 15.999).clip(lower=0)
    return table.drop(columns="O").round(2).to_dict(orient="index")


def predict(rocks, encoder, heads, recalibrator, n_samples=512, seed=0):
    """One JSON-ready record per rock: both arms, scenarios, conditions and compositions."""
    src, frac = tokenize_bulk_frame(rocks)
    with torch.no_grad():
        latent = encoder(src=torch.from_numpy(src), frac=torch.from_numpy(frac))
        p21 = torch.sigmoid(heads.mode.presence_logits(latent)).numpy()
        comp_alpha = heads.comp(latent).numpy()
    p7 = np.clip(coarsen_presence(p21), 1e-7, 1 - 1e-7)
    logits7 = torch.as_tensor(np.log(p7 / (1 - p7)), dtype=torch.float64)
    p21_recal = lift_group_presence_to_phases(recalibrator, logits7, p21)

    records = []
    for i, rock in enumerate(rocks.index):
        records.append({
            "rock": str(rock),
            "input_cation_wt_pct": {sym: round(float(frac[i, k]) * 100, 5)
                                    for k, (sym, _) in enumerate(VOCAB) if src[i, k] > 0},
            "groups": [GROUP_NAME[g] for g in GROUPS],
            "arms": {
                "zero-shot": _posterior(heads, latent[i], p21[i], None, n_samples, seed),
                "recalibrated": _posterior(heads, latent[i], p21_recal[i], _corrector(recalibrator, logits7[i]), n_samples, seed),
            },
            "composition_wt_pct": _composition(comp_alpha[i], p21_recal[i]),
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


def _assemblages_panel(ax, scenarios, groups):
    """The deck's stacked bars: one per scenario, its probability at the end."""
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
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-0.55, len(scenarios) - 0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _conditions_panel(ax, conditions):
    """The deck's kernel density over the sampled (P, T) pairs."""
    pressure, temperature = np.asarray(conditions["P_kbar"]), np.asarray(conditions["T"])
    t_lo, t_hi = np.floor(temperature.min() / 50) * 50, np.ceil(temperature.max() / 50) * 50
    kde = gaussian_kde(np.vstack([pressure, temperature]))
    pp, tt = np.meshgrid(np.linspace(-1.5, 14, 200), np.linspace(t_lo, t_hi, 200))
    z = kde(np.vstack([pp.ravel(), tt.ravel()])).reshape(pp.shape)
    ax.contourf(pp, tt, z, levels=np.linspace(z.max() * 0.10, z.max(), 6), cmap=DENSITY, extend="max")
    ax.set_xlim(-1.5, 14)
    ax.set_ylim(t_lo, t_hi)
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel("P (kbar)")
    ax.set_ylabel("T (\N{DEGREE SIGN}C)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def card(record):
    """The per-rock figure, drawn from the record alone.

    Panels (a) to (e) are the paper's inference card: the input chemistry,
    then presence and abundance per arm. Panels (f) and (g) are the deck's
    output card: the ranked assemblages and the conditions density, both from
    the recalibrated arm. Water content stays in the record but is not drawn.
    """
    groups = record["groups"]
    codes = [GROUP_CODE[g] for g in groups]
    arms = record["arms"]
    with plt.rc_context(CARD_RC):
        fig = plt.figure(figsize=(5.5, 7.0))
        gs = fig.add_gridspec(3, 3, width_ratios=[1, 0.9, 1.2], height_ratios=[1, 1, 1.05],
                              wspace=0.42, hspace=0.55, left=0.09, right=0.97, top=0.93, bottom=0.07)
        fig.suptitle(record["rock"], x=0.09, y=0.985, ha="left", fontsize=9)

        ax_chem = fig.add_subplot(gs[:2, 0])
        _chemistry_panel(ax_chem, record["input_cation_wt_pct"])
        ax_chem.set_xlabel("Cation wt%")
        ax_chem.set_title("(a) Geochemical input")
        boxed = [ax_chem]

        for r, (arm, left, right) in enumerate((("zero-shot", "(b)", "(c)"), ("recalibrated", "(d)", "(e)"))):
            post = arms[arm]
            ax1, ax2 = fig.add_subplot(gs[r, 1]), fig.add_subplot(gs[r, 2])
            _presence_panel(ax1, post["presence"])
            _abundance_panel(ax2, post["share_mean"], post["share_p25"], post["share_p75"])
            ax1.set_title(f"{left} Presence, {arm}")
            ax2.set_title(f"{right} Abundance, {arm}")
            for ax in (ax1, ax2):
                ax.set_yticks(np.arange(len(codes)), codes)
                ax.set_ylim(len(codes) - 0.5, -0.5)
            if arm == "recalibrated":
                ax1.set_xlabel("Presence probability")
                ax2.set_xlabel("Fraction of crystalline mass")
            boxed += [ax1, ax2]
        for ax in boxed:
            _frame(ax)

        ax_scen = fig.add_subplot(gs[2, :2])
        _assemblages_panel(ax_scen, arms["recalibrated"]["scenarios"], groups)
        ax_scen.set_title("(f) Possible assemblages, recalibrated", loc="left")
        ax_cond = fig.add_subplot(gs[2, 2])
        _conditions_panel(ax_cond, arms["recalibrated"]["conditions"])
        ax_cond.set_title("(g) Formation conditions")
    return fig
