"""Zero-shot sim-to-real eval: the MAGEMin-trained ModeHead on real volcanic rocks.

Scores the trained heads against `data/real_mineralogy/` (point-counted and
visually-estimated phenocryst modes with matching whole-rock chemistry) with
nothing fit -- a fixed coarsening of the 21-phase vocabulary onto the 7 groups
real modal data actually resolves, then the same hurdle NLL the corpus eval
reports. That number is the sealed transfer gap and the blueprint for the thin
real-world adapter that comes next: its residual shape decides which per-group
corrections the adapter needs.

The coarsening is exact, not an approximation. Presence composes as a noisy-OR
because the head emits independent Bernoulli marginals; magnitude composes by
Dirichlet aggregation, and dropping the phases real modal data cannot resolve
is the Dirichlet sub-composition property that `masked_dirichlet_nll` already
rests on.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import betainc
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cnmineral.data.bulk_chemistry import tokenize_bulk_frame
from cnmineral.data.phase_vocab import PHASE_INDEX, SOLID_PHASES
from cnmineral.eval.mode_head import (
    conditional_mean,
    magnitude_parity_stats,
    presence_scores,
    reliability_table,
)
from cnmineral.losses.mode_loss import masked_dirichlet_nll

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPO_ROOT / "data" / "real_mineralogy"
RUNS_ROOT = REPO_ROOT / "runs"

# the vocabulary real modal data resolves. Coarser than the model's 21 solid
# phases in three places, each forced by the truth side: du Bray reports
# pyroxene and Fe-Ti oxides unsplit, and no source splits feldspar into
# plagioclase / alkali feldspar the way the training labels do.
GROUPS: tuple[str, ...] = ("fsp", "px", "amp", "bi", "ol", "opq", "q")
GROUP_INDEX: dict[str, int] = {g: i for i, g in enumerate(GROUPS)}

# model phase -> comparison group. The 11 phases absent from this map
# (g, ru, sph, sill, cd, mu, ep, ne, ky, and, cor) are unscoreable: no source
# reports them. They are not forced absent -- the model resolves them, and
# their aggregate mass is reported as the off-vocabulary diagnostic.
MODEL_TO_GROUP: dict[str, str] = {
    "plag": "fsp", "afs": "fsp",
    "cpx": "px", "opx": "px",
    "amp": "amp",
    "bi": "bi",
    "ol": "ol",
    "spl": "opq", "ilm": "opq",
    "q": "q",
}

# `modes.csv` magemin_phase -> comparison group. `groundmass` is dropped
# rather than mapped: the label is the crystals-renormalized solid assemblage,
# so the truth vector is renormalized to the phenocryst cargo the same way.
TRUTH_TO_GROUP: dict[str, str] = {
    "fsp": "fsp",
    "px_unsplit": "px", "cpx": "px", "opx": "px",
    "amp": "amp",
    "bi": "bi",
    "ol": "ol",
    "opq_unsplit": "opq",
    "q": "q",
}
DROPPED_TRUTH_PHASES: frozenset[str] = frozenset({"groundmass"})

# Representative densities (g/cm3, Deer/Howie/Zussman; Klein for the oxides)
# keyed by the as-reported mineral name, so a lumped "pyroxene" and a specific
# "hypersthene" convert differently. Point counts and visual estimates are
# volume fractions; MAGEMin modes are mass fractions, so truth converts before
# it is coarsened. Only ratios matter -- the one with real leverage is Fe-Ti
# oxides against the silicates, roughly 1.8x. Solid-solution names take a
# mid-series value (arc olivine near Fo80, andesine plagioclase), which is an
# assumption, not a measurement.
MINERAL_DENSITY: dict[str, float] = {
    "plagioclase": 2.68,
    "alkali feldspar": 2.57,
    "pyroxene": 3.40,
    "augite": 3.40,
    "hypersthene": 3.45,
    "bronzite": 3.35,
    "hornblende": 3.15,
    "cummingtonite": 3.30,
    "biotite": 3.05,
    "olivine": 3.50,
    "Fe-Ti oxides": 5.00,
    "quartz": 2.65,
    "silica phases": 2.40,
}

# du Bray's numeric grid bottoms out at 0.5 vol% (p10 of every abundance it
# does report), so a TR / <1 row means "present, below that grid". Used as the
# upper bound of the left-censored magnitude.
TRACE_BOUND_VOL_PCT: float = 0.5

# `alteration_extent` is graded 1 (least altered) to 4, with +/- modifiers,
# plus a literal "fresh" for the precision-tier eruptions.
CLEAN_GRADES: frozenset[int] = frozenset({1, 2})

PRECISION_SOURCES: frozenset[str] = frozenset(
    {"msh_pallister2008", "pinatubo_pallister1996"}
)


@dataclass(frozen=True)
class RealRockBenchmark:
    """One row per scoreable real rock, aligned across every array.

    Duck-types the attributes `eval.latent.bulk_latents` reads (`bulk_ids`,
    `bulk_src`, `bulk_frac`), so the frozen encoder runs over real rocks
    through the same path as the corpus.

    Truth splits three ways per group, because the sources report three
    different things: `presence` is what is there, `numeric` is where an
    abundance was actually quantified (the magnitude vector lives on that
    sub-composition and sums to 1 over it), and `censored` marks
    present-but-unquantified trace phases, whose share is known only to lie
    below `censor_bound`.
    """

    bulk_ids: np.ndarray
    bulk_src: torch.Tensor
    bulk_frac: torch.Tensor
    presence: np.ndarray
    magnitude: np.ndarray
    numeric: np.ndarray
    censored: np.ndarray
    censor_bound: np.ndarray
    has_magnitude: np.ndarray
    source: np.ndarray
    precision_tier: np.ndarray
    grade: np.ndarray
    clean: np.ndarray
    sio2: np.ndarray

    def __len__(self) -> int:
        return len(self.bulk_ids)

    def subset(self, mask: np.ndarray) -> "RealRockBenchmark":
        """The rocks selected by a boolean mask, arrays kept aligned."""
        mask = np.asarray(mask, dtype=bool)
        return RealRockBenchmark(
            bulk_ids=self.bulk_ids[mask],
            bulk_src=self.bulk_src[torch.as_tensor(mask)],
            bulk_frac=self.bulk_frac[torch.as_tensor(mask)],
            presence=self.presence[mask],
            magnitude=self.magnitude[mask],
            numeric=self.numeric[mask],
            censored=self.censored[mask],
            censor_bound=self.censor_bound[mask],
            has_magnitude=self.has_magnitude[mask],
            source=self.source[mask],
            precision_tier=self.precision_tier[mask],
            grade=self.grade[mask],
            clean=self.clean[mask],
            sio2=self.sio2[mask],
        )


def parse_alteration_grade(values: pd.Series) -> np.ndarray:
    """Alteration grade as an int, -1 where ungraded.

    du Bray grades carry modifiers (`1+`, `2-`), which qualify the grade
    rather than change it, so the leading digit is the grade. The
    precision-tier eruptions report a literal `fresh`, which is grade 1 by
    any reading.
    """
    text = values.astype(str).str.strip()
    grade = pd.to_numeric(text.str.extract(r"^(\d)")[0], errors="coerce")
    grade = grade.where(text.str.lower() != "fresh", 1.0)
    return grade.fillna(-1).to_numpy(dtype=np.int64)


def _truth_arrays(modes: pd.DataFrame, keys: np.ndarray) -> dict[str, np.ndarray]:
    """Build the per-group truth arrays for `keys`, in that order.

    Volume percent converts to mass before coarsening (per-raw-name density),
    the numeric entries renormalize to a composition over the groups that
    carry a number, and each censored group gets the renormalized bound its
    trace rows imply.
    """
    n, p = len(keys), len(GROUPS)
    presence = np.zeros((n, p), dtype=np.float64)
    mass = np.zeros((n, p), dtype=np.float64)
    trace_row = np.zeros((n, p), dtype=bool)
    censor_mass = np.zeros((n, p), dtype=np.float64)

    row_of = {k: i for i, k in enumerate(keys)}
    scored = modes[modes["magemin_phase"].isin(TRUTH_TO_GROUP)]
    for key, phase, magemin_phase, abundance in zip(
        scored["key"], scored["phase"], scored["magemin_phase"],
        scored["abundance_pct"],
    ):
        i = row_of.get(key)
        if i is None:
            continue
        # a reported 0.0 means the phase was looked for and not found, which
        # is the one case where a row in this table is not a presence.
        if pd.notna(abundance) and abundance <= 0:
            continue
        j = GROUP_INDEX[TRUTH_TO_GROUP[magemin_phase]]
        presence[i, j] = 1.0
        density = MINERAL_DENSITY[phase]
        if pd.notna(abundance):
            mass[i, j] += abundance * density
        else:
            trace_row[i, j] = True
            censor_mass[i, j] += TRACE_BOUND_VOL_PCT * density

    # coarsening can put a quantified and a trace mineral in the same group
    # (a numeric hornblende beside a trace cummingtonite). The group then has
    # a number, so it is scored as observed and its bound is discarded --
    # censored means no number for the group at all.
    numeric = mass > 0
    censored = trace_row & ~numeric
    censor_mass[~censored] = 0.0

    total = mass.sum(1, keepdims=True)
    has_magnitude = total[:, 0] > 0
    magnitude = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)

    # the bound is expressed against the mass the rock would have if every
    # trace phase sat exactly at its bound, so it is the conservative reading
    # (a slightly smaller share than renormalizing by the numeric mass alone).
    full = total + censor_mass.sum(1, keepdims=True)
    censor_bound = np.divide(
        censor_mass, full, out=np.zeros_like(censor_mass), where=full > 0
    )
    return {
        "presence": presence,
        "magnitude": magnitude,
        "numeric": numeric,
        "censored": censored,
        "censor_bound": censor_bound,
        "has_magnitude": has_magnitude,
    }


def load_benchmark(benchmark_dir: Path | str = BENCHMARK_DIR) -> RealRockBenchmark:
    """Every real rock carrying both whole-rock chemistry and modal mineralogy.

    Rocks without chemistry (Pinatubo, pending a table transcription) or
    without mode rows drop out here; the freshness cut is a mask on the
    result, not a filter, so the altered rocks stay available as a diagnostic.
    """
    benchmark_dir = Path(benchmark_dir)
    samples = pd.read_csv(benchmark_dir / "samples.csv")
    modes = pd.read_csv(benchmark_dir / "modes.csv")
    for frame in (samples, modes):
        frame["key"] = frame["source"] + "|" + frame["sample_id"].astype(str)

    unknown = set(modes["phase"]) - set(MINERAL_DENSITY) - {"groundmass"}
    if unknown:
        raise ValueError(
            f"modes.csv reports minerals with no density: {sorted(unknown)}; "
            "add them to MINERAL_DENSITY"
        )

    scoreable = samples["sio2"].notna() & samples["key"].isin(set(modes["key"]))
    samples = samples[scoreable].reset_index(drop=True)

    src, frac = tokenize_bulk_frame(samples)
    keys = samples["key"].to_numpy()
    truth = _truth_arrays(modes, keys)
    grade = parse_alteration_grade(samples["alteration_extent"])

    return RealRockBenchmark(
        bulk_ids=keys,
        bulk_src=torch.from_numpy(src),
        bulk_frac=torch.from_numpy(frac),
        source=samples["source"].to_numpy(),
        precision_tier=samples["source"].isin(PRECISION_SOURCES).to_numpy(),
        grade=grade,
        clean=np.isin(grade, list(CLEAN_GRADES)),
        sio2=samples["sio2"].to_numpy(dtype=np.float64),
        **truth,
    )


def coarsen_presence(p: np.ndarray) -> np.ndarray:
    """Per-group presence probability from the 21 per-phase marginals.

    A group is present when any member phase is. The head's presence branch
    emits independent Bernoulli marginals given the latent, so the noisy-OR
    `1 - prod(1 - p_i)` is that probability exactly, not an approximation.
    """
    out = np.ones((len(p), len(GROUPS)), dtype=np.float64)
    for phase, group in MODEL_TO_GROUP.items():
        out[:, GROUP_INDEX[group]] *= 1.0 - p[:, PHASE_INDEX[phase]]
    return 1.0 - out


def coarsen_alpha(
    alpha: np.ndarray, support: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group Dirichlet concentrations, plus the off-vocabulary total.

    Merging components of a Dirichlet sums their concentrations, so the
    coarsened magnitude distribution is exact. Only the phases on `support`
    take part -- the distribution lives on the support, and an off-support
    concentration is not a component of it. The phases no source reports are
    returned separately rather than discarded: their share is the
    off-vocabulary mass, which reads as how much of the predicted assemblage
    the benchmark structurally cannot see.
    """
    on = alpha * support
    out = np.zeros((len(alpha), len(GROUPS)), dtype=np.float64)
    for phase, group in MODEL_TO_GROUP.items():
        out[:, GROUP_INDEX[group]] += on[:, PHASE_INDEX[phase]]
    off_vocab = np.array(
        [on[:, PHASE_INDEX[ph]] for ph in SOLID_PHASES if ph not in MODEL_TO_GROUP]
    ).sum(0)
    return out, off_vocab


def truth_forced_support(presence7: np.ndarray, p21: np.ndarray) -> np.ndarray:
    """A 21-phase support consistent with the group-level truth.

    The magnitude branch is support-conditioned, but real modal data
    constrains only the group level -- "pyroxene is present" does not say
    whether that is cpx, opx, or both. So truth forces what it knows and the
    model resolves what it does not: within a truth-present group, the member
    phases the model puts above even odds turn on (its most likely member if
    none clears the bar); truth-absent groups turn fully off. The phases no
    source reports are left to the model, since truth never asserted their
    absence.
    """
    support = (p21 > 0.5).astype(np.float64)
    for group, gi in GROUP_INDEX.items():
        members = [PHASE_INDEX[ph] for ph, g in MODEL_TO_GROUP.items() if g == group]
        present = presence7[:, gi] > 0
        support[np.ix_(~present, members)] = 0.0
        empty = present & (support[:, members].sum(1) == 0)
        if empty.any():
            best = np.asarray(members)[p21[np.ix_(empty, members)].argmax(1)]
            support[np.flatnonzero(empty), best] = 1.0
    return support


def real_rock_nll(
    p7: np.ndarray,
    alpha7: np.ndarray,
    presence: np.ndarray,
    magnitude: np.ndarray,
    numeric: np.ndarray,
) -> dict[str, np.ndarray]:
    """The hurdle NLL in group space -- the headline, and the adapter's loss.

    Assembled exactly as the corpus eval assembles it, on the coarsened
    vocabulary: presence BCE summed over groups, plus the Dirichlet NLL of the
    observed sub-composition. The adapter that comes next rescales presence
    logits and magnitude concentrations per group, and both of those are fit
    by minimizing this -- Platt scaling *is* Bernoulli-NLL fitting -- so the
    zero-shot value is that loss at the identity correction and the adapter's
    gain reads as nats recovered on one scale.

    Rocks whose observed sub-composition has a single group contribute exactly
    0 to the Dirichlet term (a degenerate point mass, not a free win); the
    caller reports how many.
    """
    p = torch.as_tensor(np.clip(p7, 1e-7, 1 - 1e-7), dtype=torch.float64)
    bern = F.binary_cross_entropy(
        p, torch.as_tensor(presence, dtype=torch.float64), reduction="none",
    ).sum(-1)
    # off-mask concentrations must stay positive going in: the masked NLL
    # forms `mask * lgamma(alpha)`, and lgamma(0) is +inf, so a zero there
    # gives 0 * inf = NaN instead of the intended no-op. Coarsening produces
    # exact zeros wherever a group has no member on the support, and those
    # groups are excluded by the mask anyway, so any positive filler works.
    alpha = np.where(numeric, alpha7, 1.0)
    dirichlet = masked_dirichlet_nll(
        torch.as_tensor(alpha, dtype=torch.float64),
        torch.as_tensor(magnitude, dtype=torch.float64),
        torch.as_tensor(numeric, dtype=torch.float64),
        reduction="none",
    )
    return {
        "bernoulli": bern.numpy(),
        "dirichlet": dirichlet.numpy(),
        "joint": (bern + dirichlet).numpy(),
    }


def censored_consistency(
    alpha7: np.ndarray,
    support: np.ndarray,
    censored: np.ndarray,
    censor_bound: np.ndarray,
) -> dict[str, np.ndarray]:
    """How well the model respects the bounds on present-but-unquantified phases.

    A quarter of amphibole's real appearances (a third of quartz's) are
    reported only as a trace, so scoring magnitude on quantified entries alone
    would silently condition the number on the phase being abundant. Those
    entries are left-censored instead: truth says the share sits below a known
    bound, and the Dirichlet marginal on the support is a Beta, so
    `P(share < bound)` is closed-form.

    Kept beside the headline rather than added into it: the exact censored
    likelihood needs an integral over the censored region, and multiplying a
    density on the observed groups by a marginal tail probability is not a
    joint likelihood, because the Dirichlet couples the two. Returns the
    per-instance probabilities and its own NLL, which is what an adapter would
    add to the objective if these instances should inform the fit.
    """
    alpha0 = (alpha7 * support).sum(-1, keepdims=True)
    beta = np.clip(alpha0 - alpha7, 1e-9, None)
    prob = np.full(alpha7.shape, np.nan)
    where = censored & (censor_bound > 0)
    prob[where] = betainc(
        alpha7[where], beta[where], np.clip(censor_bound[where], 1e-12, 1 - 1e-12)
    )
    mean_share = np.divide(
        alpha7, np.clip(alpha0, 1e-12, None), out=np.zeros_like(alpha7),
        where=alpha0 > 0,
    )
    return {
        "prob_below_bound": prob,
        "nll": -np.log(np.clip(prob, 1e-12, None)),
        "mean_below_bound": (mean_share < censor_bound) & where,
        "scored": where,
    }


def presence_group_stats(p7: np.ndarray, presence: np.ndarray) -> dict[str, np.ndarray]:
    """Per-group AUROC and the threshold maximizing F1.

    AUROC leads because it is threshold-free: the head's probabilities were
    calibrated against how often MAGEMin grows a phase, which is not how often
    a geologist logs it, so any fixed cutoff conflates ranking quality with
    that offset. The best-F1 threshold separates them -- it is a direct read
    on the per-group intercept a Platt correction would learn.
    """
    n_groups = p7.shape[1]
    auroc = np.full(n_groups, np.nan)
    best_threshold = np.full(n_groups, np.nan)
    best_f1 = np.full(n_groups, np.nan)
    for i in range(n_groups):
        y = presence[:, i] > 0
        if y.all() or not y.any():
            continue
        auroc[i] = float(roc_auc_score(y, p7[:, i]))
        grid = np.unique(np.round(p7[:, i], 3))
        f1 = np.array([_f1(p7[:, i] > t, y) for t in grid])
        best = int(np.nanargmax(f1))
        best_threshold[i], best_f1[i] = float(grid[best]), float(f1[best])
    return {"auroc": auroc, "best_threshold": best_threshold, "best_f1": best_f1}


def _f1(pred: np.ndarray, true: np.ndarray) -> float:
    tp = float((pred & true).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (pred.sum() + true.sum())


def magnitude_group_stats(
    pred: np.ndarray, magnitude: np.ndarray, numeric: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-group R2 and Spearman on the groups that carry a reported number.

    Conditional on quantifiably-present by construction: trace-only entries
    have no number to correlate against and are scored by
    `censored_consistency` instead.
    """
    n_groups = magnitude.shape[1]
    r2 = np.full(n_groups, np.nan)
    rho = np.full(n_groups, np.nan)
    n = np.zeros(n_groups, dtype=np.int64)
    for i in range(n_groups):
        m = numeric[:, i]
        n[i] = int(m.sum())
        true, p = magnitude[m, i], pred[m, i]
        if n[i] < 3 or true.std() == 0:
            continue
        r2[i] = float(1.0 - ((true - p) ** 2).sum() / ((true - true.mean()) ** 2).sum())
        if p.std() > 0:
            rho[i] = float(spearmanr(true, p).statistic)
    return {"r2": r2, "spearman": rho, "n": n}


def _group_grid(height: float = 2.7):
    fig, axes = plt.subplots(2, 4, figsize=(4 * 3.1, 2 * height), squeeze=False)
    axes.flat[len(GROUPS)].set_visible(False)
    return fig, axes.flat


def _finish(fig, title: str, out_path: Path) -> None:
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_presence_calibration(
    p7, presence, stats, out_path: Path, subtitle: str
) -> None:
    """Predicted presence probability against observed frequency, per group.

    The diagonal is what calibration on MAGEMin would buy if the simulator's
    phase frequencies matched a geologist's logs. Systematic departure is the
    offset the adapter exists to absorb, so read the vertical gap, not just
    the scatter.
    """
    fig, axes = _group_grid()
    for i, group in enumerate(GROUPS):
        ax = axes[i]
        rel = reliability_table(p7[:, i], presence[:, i])
        ax.plot([0, 1], [0, 1], lw=0.8, color="0.6", ls="--")
        ok = rel["count"] > 0
        ax.plot(rel["mean_p"][ok], rel["freq"][ok], "o-", ms=4, color="tab:orange")
        ax.axhline(presence[:, i].mean(), lw=0.7, color="tab:blue", alpha=0.6)
        ax.set_title(
            f"{group}   AUROC {stats['auroc'][i]:.2f}   ECE {rel['ece']:.2f}",
            fontsize=9,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("observed frequency", fontsize=8)
    _finish(fig, f"presence calibration on real rocks -- {subtitle}\n"
                 "blue line: the group's base rate", out_path)


def plot_magnitude_parity(pred, magnitude, numeric, stats, out_path, subtitle) -> None:
    """Predicted against measured share of the crystal cargo, per group."""
    fig, axes = _group_grid()
    for i, group in enumerate(GROUPS):
        ax, m = axes[i], numeric[:, i]
        ax.plot([0, 1], [0, 1], lw=0.8, color="0.6", ls="--")
        ax.plot(magnitude[m, i], pred[m, i], "o", ms=2.5, alpha=0.35,
                color="tab:orange")
        ax.set_title(
            f"{group}   R2 {stats['r2'][i]:.2f}   rho {stats['spearman'][i]:.2f}"
            f"   n={stats['n'][i]}", fontsize=9,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("measured share", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("predicted share", fontsize=8)
    _finish(fig, f"magnitude parity on real rocks -- {subtitle}\n"
                 "quantified entries only; trace-only appearances scored by their "
                 "censoring bound", out_path)


def plot_residual_vs_sio2(
    p7, pred, benchmark, out_path: Path, subtitle: str
) -> None:
    """Where the gap lives along the differentiation trend -- the adapter's blueprint.

    A flat residual means one constant per group corrects it; a sloped one
    means the correction needs a silica term. The amphibole-pyroxene
    changeover sits near 52-63 wt% SiO2, so that band is where a slope would
    show up if anything does.
    """
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(2.5 * len(GROUPS), 5.4),
                             squeeze=False, sharex=True)
    for i, group in enumerate(GROUPS):
        top, bottom = axes[0][i], axes[1][i]
        top.axhline(0, lw=0.7, color="0.5")
        top.plot(benchmark.sio2, benchmark.presence[:, i] - p7[:, i], "o", ms=2,
                 alpha=0.3, color="tab:purple")
        top.axvspan(52, 63, color="0.85", zorder=0)
        top.set_title(group, fontsize=9)
        top.set_ylim(-1.05, 1.05)

        m = benchmark.numeric[:, i]
        bottom.axhline(0, lw=0.7, color="0.5")
        bottom.plot(benchmark.sio2[m], benchmark.magnitude[m, i] - pred[m, i], "o",
                    ms=2, alpha=0.3, color="tab:green")
        bottom.axvspan(52, 63, color="0.85", zorder=0)
        bottom.set_xlabel("SiO2 (wt%)", fontsize=8)
        bottom.set_ylim(-1.05, 1.05)
    axes[0][0].set_ylabel("presence residual\n(truth - predicted)", fontsize=8)
    axes[1][0].set_ylabel("magnitude residual\n(truth - predicted)", fontsize=8)
    _finish(fig, f"where the sim-to-real gap sits along the silica trend -- "
                 f"{subtitle}\nshaded: the 52-63 wt% band where amphibole and "
                 "pyroxene trade places", out_path)


def plot_gap_summary(p7, pred, benchmark, out_path: Path, subtitle: str) -> None:
    """The two corrections an adapter would make, per group, on one axis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    y = np.arange(len(GROUPS))
    presence_bias = benchmark.presence.mean(0) - p7.mean(0)
    ax1.barh(y, presence_bias, color=np.where(presence_bias > 0, "tab:blue", "tab:red"))
    ax1.axvline(0, lw=0.8, color="0.3")
    ax1.set_yticks(y, GROUPS)
    ax1.invert_yaxis()
    ax1.set_xlabel("observed rate - mean predicted probability")
    ax1.set_title("presence offset (positive: model under-calls)", fontsize=10)

    scale = np.full(len(GROUPS), np.nan)
    for i in range(len(GROUPS)):
        m = benchmark.numeric[:, i]
        if m.sum() >= 3 and pred[m, i].mean() > 0:
            scale[i] = np.log(benchmark.magnitude[m, i].mean() / pred[m, i].mean())
    ax2.barh(y, scale, color=np.where(scale > 0, "tab:blue", "tab:red"))
    ax2.axvline(0, lw=0.8, color="0.3")
    ax2.set_yticks(y, GROUPS)
    ax2.invert_yaxis()
    ax2.set_xlabel("log(mean measured share / mean predicted share)")
    ax2.set_title("magnitude scale offset (positive: model under-sizes)", fontsize=10)
    _finish(fig, f"the corrections a thin adapter would learn -- {subtitle}", out_path)


def real_rock_card(
    p7, pred, benchmark, row: int, out_path: Path
) -> None:
    """One real rock end to end: chemistry in, assemblage out, measurement beside it."""
    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.0, 0.9, 1.2]}
    )
    src, frac = benchmark.bulk_src[row].numpy(), benchmark.bulk_frac[row].numpy()
    populated = np.flatnonzero(src > 0)
    ax0.barh(np.arange(len(populated)), frac[populated] * 100, color="tab:gray")
    ax0.set_yticks(np.arange(len(populated)),
                   [str(int(z)) for z in src[populated]], fontsize=7)
    ax0.invert_yaxis()
    ax0.set_xscale("log")
    ax0.set_xlabel("cation wt% (log)", fontsize=8)
    ax0.set_title("input: whole-rock chemistry\n(y: atomic number)", fontsize=9)

    y = np.arange(len(GROUPS))
    ax1.barh(y - 0.19, p7[row], 0.36, color="tab:orange", label="predicted")
    ax1.barh(y + 0.19, benchmark.presence[row], 0.36, color="tab:blue",
             label="measured")
    ax1.set_yticks(y, GROUPS, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("presence probability", fontsize=8)
    ax1.set_title("is it there?", fontsize=9)
    ax1.legend(fontsize=7)

    for i in range(len(GROUPS)):
        if benchmark.numeric[row, i]:
            ax2.plot(benchmark.magnitude[row, i], i + 0.19, "o", ms=5,
                     color="tab:blue")
        elif benchmark.censored[row, i]:
            ax2.annotate("", xy=(benchmark.censor_bound[row, i], i + 0.19),
                         xytext=(1.0, i + 0.19),
                         arrowprops={"arrowstyle": "->", "color": "tab:blue",
                                     "lw": 0.9, "alpha": 0.7})
        ax2.plot(pred[row, i], i - 0.19, "o", ms=5, color="tab:orange", mfc="white")
    ax2.set_yticks(y, GROUPS, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.02)
    ax2.set_xlabel("share of the crystal cargo", fontsize=8)
    ax2.set_title("how much?\n(arrow: measured only as a trace, so bounded above)",
                  fontsize=9)
    _finish(fig, f"{benchmark.bulk_ids[row]} -- SiO2 {benchmark.sio2[row]:.1f} wt%, "
                 "zero-shot (nothing fit on real rocks)", out_path)


def _headline(nll: dict, mask: np.ndarray, numeric: np.ndarray) -> dict:
    """Mean NLL terms over the rocks selected by `mask`, unweighted per rock.

    Carries `mean_composition_size` because the Dirichlet term is a density
    over however many groups a rock's composition spans, so its scale moves
    with that count. Two subsets are only comparable when their sizes are --
    a subset of simpler rocks scores lower without predicting anything better.
    """
    if not mask.any():
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "joint_nll": float(nll["joint"][mask].mean()),
        "bernoulli_nll": float(nll["bernoulli"][mask].mean()),
        "dirichlet_nll": float(nll["dirichlet"][mask].mean()),
        "mean_composition_size": float(numeric[mask].sum(1).mean()),
    }


def predict(head, latents: torch.Tensor, benchmark: RealRockBenchmark) -> dict:
    """Coarsened zero-shot predictions for every rock in `benchmark`.

    Presence is the model's own call. The magnitude branch is
    support-conditioned, so its support is teacher-forced at the group level
    the truth actually resolves (`truth_forced_support`) and left to the model
    below that. `alpha_model_support` repeats it on the model's own support --
    the same exposure-bias probe the corpus eval reports, which here also says
    how much of the gap is the support being wrong rather than the sizes.
    """
    with torch.no_grad():
        p21 = torch.sigmoid(head.presence_logits(latents)).cpu().numpy()
    support = truth_forced_support(benchmark.presence, p21)
    model_support = (p21 > 0.5).astype(np.float64)
    with torch.no_grad():
        alpha = head.magnitude_alpha(
            latents, torch.as_tensor(support, dtype=latents.dtype,
                                     device=latents.device),
        ).cpu().numpy()
        alpha_model = head.magnitude_alpha(
            latents, torch.as_tensor(model_support, dtype=latents.dtype,
                                     device=latents.device),
        ).cpu().numpy()
    alpha7, off_vocab = coarsen_alpha(alpha, support)
    alpha7_model, _ = coarsen_alpha(alpha_model, model_support)
    return {
        "p7": coarsen_presence(p21),
        "alpha7": alpha7,
        "alpha7_model_support": alpha7_model,
        "off_vocab_alpha": off_vocab,
        "support21": support,
    }


@torch.no_grad()
def run_real_rock_eval(
    ckpt,
    benchmark: RealRockBenchmark,
    latents: torch.Tensor,
    *,
    device: torch.device,
    tag: str,
) -> dict:
    """Score a joint checkpoint on the real-rock benchmark, zero-shot.

    Writes figures + `metrics.json` under `runs/real_rock/<tag>/` and returns
    the metrics. The headline is the freshest rocks only; altered rocks and
    the point-counted precision tier are reported beside it so a weak number
    can be attributed rather than just observed.
    """
    output_dir = RUNS_ROOT / "real_rock" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output:  {output_dir}")

    head = ckpt.build_mode_head().to(device)
    pred = predict(head, latents, benchmark)
    p7, alpha7 = pred["p7"], pred["alpha7"]
    cond_mean = conditional_mean(alpha7, benchmark.numeric.astype(np.float64))

    clean = benchmark.clean & benchmark.has_magnitude
    nll = real_rock_nll(
        p7, alpha7, benchmark.presence, benchmark.magnitude, benchmark.numeric
    )
    nll_model_support = real_rock_nll(
        p7, pred["alpha7_model_support"], benchmark.presence, benchmark.magnitude,
        benchmark.numeric,
    )
    # the exposure probe asks what the magnitudes would cost if the model had
    # to stand on its own assemblage rather than the measured one. It is only
    # defined where the model's support covers every group that carries a
    # measured share: a group the model calls absent gets no concentration at
    # all, and the density of an observed share under it is undefined, not
    # merely bad. Reported with its coverage so the conditioning is visible.
    covered = clean & (
        (benchmark.numeric & (pred["alpha7_model_support"] > 0)).sum(1)
        == benchmark.numeric.sum(1)
    )
    censoring = censored_consistency(
        alpha7, (benchmark.numeric | benchmark.censored).astype(np.float64),
        benchmark.censored, benchmark.censor_bound,
    )

    subtitle = f"{int(clean.sum())} least-altered rocks"
    p_stats = presence_group_stats(p7[benchmark.clean],
                                   benchmark.presence[benchmark.clean])
    plot_presence_calibration(
        p7[benchmark.clean], benchmark.presence[benchmark.clean], p_stats,
        output_dir / "presence_calibration.png", subtitle,
    )
    clean_bench = benchmark.subset(benchmark.clean)
    mag_stats = magnitude_group_stats(
        cond_mean[benchmark.clean], clean_bench.magnitude, clean_bench.numeric
    )
    parity = magnitude_parity_stats(
        cond_mean[benchmark.clean], clean_bench.magnitude,
        clean_bench.numeric.astype(np.float64),
    )
    plot_magnitude_parity(
        cond_mean[benchmark.clean], clean_bench.magnitude, clean_bench.numeric,
        mag_stats, output_dir / "magnitude_parity.png", subtitle,
    )
    plot_residual_vs_sio2(
        p7[benchmark.clean], cond_mean[benchmark.clean], clean_bench,
        output_dir / "residual_vs_sio2.png", subtitle,
    )
    plot_gap_summary(
        p7[benchmark.clean], cond_mean[benchmark.clean], clean_bench,
        output_dir / "gap_summary.png", subtitle,
    )
    card_row = int(np.flatnonzero(clean)[np.argsort(nll["joint"][clean])[
        int(clean.sum()) // 2]])
    # the card normalizes over every measured group, trace ones included, so
    # a mineral logged only as a trace still shows what the model says about
    # it next to the bound it has to sit under.
    real_rock_card(
        p7,
        conditional_mean(alpha7, (benchmark.numeric | benchmark.censored).astype(
            np.float64)),
        benchmark, card_row, output_dir / "real_rock_card.png",
    )

    thresholded = presence_scores(p7[benchmark.clean],
                                  benchmark.presence[benchmark.clean])
    alpha0 = (alpha7 * (benchmark.numeric | benchmark.censored)).sum(1)
    off_vocab_share = pred["off_vocab_alpha"] / np.clip(
        alpha0 + pred["off_vocab_alpha"], 1e-12, None
    )

    per_group = []
    for i, group in enumerate(GROUPS):
        scored = censoring["scored"][benchmark.clean, i]
        per_group.append({
            "group": group,
            "observed_rate": float(benchmark.presence[benchmark.clean, i].mean()),
            "mean_predicted_p": float(p7[benchmark.clean, i].mean()),
            "auroc": _maybe(p_stats["auroc"][i]),
            "best_threshold": _maybe(p_stats["best_threshold"][i]),
            "best_f1": _maybe(p_stats["best_f1"][i]),
            "recall_at_0.5": _maybe(thresholded["recall"][i]),
            "precision_at_0.5": _maybe(thresholded["precision"][i]),
            "magnitude_r2": _maybe(mag_stats["r2"][i]),
            "magnitude_spearman": _maybe(mag_stats["spearman"][i]),
            "magnitude_mae": _maybe(parity["magnitude_mae"][i]),
            "magnitude_pearson_r": _maybe(parity["magnitude_r"][i]),
            "n_quantified": int(mag_stats["n"][i]),
            "n_trace_only": int(scored.sum()),
            "censored_prob_below_bound": _maybe(
                np.nanmean(censoring["prob_below_bound"][benchmark.clean, i])
                if scored.any() else np.nan
            ),
            "censored_mean_below_bound_rate": _maybe(
                censoring["mean_below_bound"][benchmark.clean, i].sum()
                / scored.sum() if scored.any() else np.nan
            ),
        })

    metrics = {
        "checkpoint_tag": tag,
        "encoder_checkpoint": str(ckpt.encoder_checkpoint),
        "benchmark": {
            "dir": str(BENCHMARK_DIR),
            "groups": list(GROUPS),
            "n_with_chemistry_and_modes": int(len(benchmark)),
            "n_headline": int(clean.sum()),
            "n_altered_or_ungraded": int((~benchmark.clean).sum()),
            "n_precision_tier": int(benchmark.precision_tier.sum()),
            "n_no_quantified_phase": int((~benchmark.has_magnitude).sum()),
            "n_single_group_composition": int(
                (benchmark.numeric.sum(1) == 1).sum()
            ),
            "trace_bound_vol_pct": TRACE_BOUND_VOL_PCT,
            "clean_grades": sorted(CLEAN_GRADES),
        },
        "headline": _headline(nll, clean, benchmark.numeric),
        "altered": _headline(nll, ~benchmark.clean & benchmark.has_magnitude,
                             benchmark.numeric),
        "all_rocks": _headline(nll, benchmark.has_magnitude, benchmark.numeric),
        "precision_tier": _headline(
            nll, benchmark.precision_tier & benchmark.has_magnitude,
            benchmark.numeric,
        ),
        "exposure_bias": {
            "n": int(covered.sum()),
            "coverage": float(covered.sum() / max(int(clean.sum()), 1)),
            "dirichlet_nll_truth_support": float(nll["dirichlet"][covered].mean())
            if covered.any() else None,
            "dirichlet_nll_model_support": float(
                nll_model_support["dirichlet"][covered].mean()
            ) if covered.any() else None,
            "delta": float(
                nll_model_support["dirichlet"][covered].mean()
                - nll["dirichlet"][covered].mean()
            ) if covered.any() else None,
        },
        "censored": {
            "n_instances": int(censoring["scored"][benchmark.clean].sum()),
            "mean_prob_below_bound": float(
                np.nanmean(censoring["prob_below_bound"][benchmark.clean])
            ),
            "mean_nll": float(
                censoring["nll"][benchmark.clean][
                    censoring["scored"][benchmark.clean]
                ].mean()
            ),
        },
        "off_vocabulary_mass": {
            "mean": float(off_vocab_share[benchmark.clean].mean()),
            "p90": float(np.quantile(off_vocab_share[benchmark.clean], 0.9)),
        },
        "per_group": per_group,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"headline joint NLL: {metrics['headline']['joint_nll']:.4f} nats "
          f"over {metrics['headline']['n']} rocks")
    return metrics


def _maybe(value) -> float | None:
    """JSON-safe scalar: NaN becomes null rather than an invalid literal."""
    value = float(value)
    return None if np.isnan(value) else value
