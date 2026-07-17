"""Interrogation report for a trained hurdle ModeHead.

Rebuilds the checkpoint's dataset and provisional val split, caches one
`(p, alpha)` prediction per bulk, and scores it per the heads.md eval
philosophy: presence calibration and the joint hurdle NLL (stratified by
phase rarity, rung, qap class, support size) as the headline, presence
recall and false-positive mass as guardrails, plus PIT-based magnitude
calibration. The statistics live as pure array functions; `run_mode_head_eval`
orchestrates and emits figures + `metrics.json` under `results/mode_head/<tag>/`.
All numbers are provisional until the valtest tier replaces the provisional
split.
"""

import json
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import betainc

from cnmineral.data.mineralogy_dataset import ModeHeadDataset
from cnmineral.data.phase_vocab import SOLID_PHASES
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.losses.mode_loss import masked_dirichlet_nll
from cnmineral.training import pick_device
from cnmineral.training.mode_head import (
    ModeHeadCheckpoint,
    _bulk_latents,
    _row_split_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"

# phases whose within-bulk presence frequency hovers intermediate by design
# (path-dependent stability); recall-at-0.5 is the wrong lens for these
PATH_SENSITIVE = ("amp", "opx", "ne", "g", "ep")

RARITY_BANDS = (("common", 0.10, 1.01), ("mid", 0.01, 0.10), ("rare", 0.0, 0.01))
FREQUENCY_BANDS = ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01))


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Fixed-width reliability bins for one phase.

    `p` predicted probabilities, `y` 0/1 outcomes, both [N]. Returns bin
    counts, per-bin mean predicted probability and empirical frequency
    (NaN for empty bins), and the count-weighted ECE.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        mean_p = np.where(count > 0, sum_p / count, np.nan)
        freq = np.where(count > 0, sum_y / count, np.nan)
    nonempty = count > 0
    ece = float(
        (count[nonempty] * np.abs(mean_p[nonempty] - freq[nonempty])).sum()
        / count.sum()
    )
    return {"count": count, "mean_p": mean_p, "freq": freq, "ece": ece}


def within_bulk_frequency(
    presence: np.ndarray, row_bulk_idx: np.ndarray, n_bulks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bulk presence frequency over that bulk's rows.

    Returns `(f, n_rows)`: `f` [n_bulks, P] with NaN rows for bulks that
    contribute no rows, `n_rows` [n_bulks].
    """
    n_rows = np.bincount(row_bulk_idx, minlength=n_bulks).astype(np.float64)
    sums = np.zeros((n_bulks, presence.shape[1]))
    np.add.at(sums, row_bulk_idx, presence)
    with np.errstate(invalid="ignore"):
        f = sums / n_rows[:, None]
    return f, n_rows


def within_bulk_conditional_sigma(
    magnitude: np.ndarray, presence: np.ndarray, row_bulk_idx: np.ndarray
) -> np.ndarray:
    """Pooled within-bulk std of the conditional magnitude, per phase.

    For each phase, the std of `magnitude` around its own bulk's mean over
    truly-present rows: the vertical parity scatter a bulk-level predictor
    cannot remove (path variation, not model error). NaN for phases with no
    present rows.
    """
    n_bulks = int(row_bulk_idx.max()) + 1 if row_bulk_idx.size else 0
    out = np.full(magnitude.shape[1], np.nan)
    for i in range(magnitude.shape[1]):
        m = presence[:, i] > 0
        if not m.any():
            continue
        t, b = magnitude[m, i], row_bulk_idx[m]
        cnt = np.bincount(b, minlength=n_bulks).astype(np.float64)
        mean = np.divide(
            np.bincount(b, weights=t, minlength=n_bulks), cnt,
            out=np.zeros(n_bulks), where=cnt > 0,
        )
        out[i] = float((t - mean[b]).std())
    return out


def within_bulk_sigma_profile(
    magnitude: np.ndarray,
    presence: np.ndarray,
    row_bulk_idx: np.ndarray,
    n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Within-bulk sd of conditional magnitude, binned by the bulk's mean level.

    Same deviations as `within_bulk_conditional_sigma`, but pooled inside
    `n_bins` quantile bins of the bulk's own mean magnitude. Returns
    `(levels, sigmas)`, each [P, n_bins] (NaN for empty bins): the
    magnitude-dependent parity-band width -- bounded fractions scatter more
    at mid levels than near 0 or 1, so a flat band overstates the floor at
    low magnitudes.
    """
    n_bulks = int(row_bulk_idx.max()) + 1 if row_bulk_idx.size else 0
    n_phases = magnitude.shape[1]
    levels = np.full((n_phases, n_bins), np.nan)
    sigmas = np.full((n_phases, n_bins), np.nan)
    for i in range(n_phases):
        m = presence[:, i] > 0
        if not m.any():
            continue
        t, b = magnitude[m, i], row_bulk_idx[m]
        cnt = np.bincount(b, minlength=n_bulks).astype(np.float64)
        mean = np.divide(
            np.bincount(b, weights=t, minlength=n_bulks), cnt,
            out=np.zeros(n_bulks), where=cnt > 0,
        )
        lvl = mean[b]
        dev = t - lvl
        edges = np.quantile(lvl, np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.digitize(lvl, edges[1:-1]), 0, n_bins - 1)
        for k in range(n_bins):
            mk = idx == k
            if mk.sum() > 1:
                levels[i, k] = float(lvl[mk].mean())
                sigmas[i, k] = float(dev[mk].std())
    return levels, sigmas


def hurdle_nll_rows(
    logits: torch.Tensor,
    alpha: torch.Tensor,
    presence: torch.Tensor,
    magnitude: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Per-row terms of the joint hurdle NLL.

    Returns `bernoulli` [R] (summed over phases), `dirichlet` [R], `joint`
    [R], and the per-phase Bernoulli decomposition `bce_per_phase` [R, P].
    """
    with torch.no_grad():
        bce = F.binary_cross_entropy_with_logits(logits, presence, reduction="none")
        dir_nll = masked_dirichlet_nll(alpha, magnitude, presence, reduction="none")
    bern = bce.sum(-1)
    return {
        "bernoulli": bern.cpu().numpy(),
        "dirichlet": dir_nll.cpu().numpy(),
        "joint": (bern + dir_nll).cpu().numpy(),
        "bce_per_phase": bce.cpu().numpy(),
    }


def stratified_mean(values: np.ndarray, labels: np.ndarray) -> dict:
    """Mean and count of `values` per distinct label."""
    out = {}
    for lab in np.unique(labels):
        m = labels == lab
        out[str(lab)] = {"mean": float(values[m].mean()), "n": int(m.sum())}
    return out


def conditional_mean(alpha: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dirichlet mean restricted to `mask` (sub-composition); zero off-mask.

    Rows with an empty mask return all zeros.
    """
    masked = alpha * mask
    denom = masked.sum(-1, keepdims=True)
    return np.divide(masked, denom, out=np.zeros_like(masked), where=denom > 0)


def beta_pit(
    alpha: np.ndarray, presence: np.ndarray, magnitude: np.ndarray
) -> np.ndarray:
    """PIT of each present phase's magnitude under its Beta marginal.

    Dirichlet marginals are Betas: on the true support, `magnitude_i ~
    Beta(alpha_i, alpha0_S - alpha_i)`. Returns [R, P] PIT values, NaN at
    absent phases and on single-phase supports (degenerate point mass).
    """
    alpha0 = (alpha * presence).sum(-1, keepdims=True)
    b = alpha0 - alpha
    valid = (presence > 0) & (presence.sum(-1, keepdims=True) > 1)
    pit = np.full(alpha.shape, np.nan)
    x = np.clip(magnitude, 1e-12, 1.0 - 1e-12)
    pit[valid] = betainc(alpha[valid], b[valid], x[valid])
    return pit


def fp_fn_mass(
    p: np.ndarray,
    alpha: np.ndarray,
    presence: np.ndarray,
    magnitude: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Mass-weighted guardrails under the thresholded presence view.

    FP-mass: conditional-mean mass the predicted support puts on
    truly-absent phases. FN-mass: true mass on phases the view missed.
    Returns per-row totals and the per-phase FP-mass matrix [R, P].
    """
    pred = (p > threshold).astype(alpha.dtype)
    cm = conditional_mean(alpha, pred)
    fp = cm * pred * (1.0 - presence)
    fn = magnitude * presence * (1.0 - pred)
    return {
        "fp_mass": fp.sum(-1),
        "fn_mass": fn.sum(-1),
        "fp_mass_per_phase": fp,
    }


def presence_scores(
    p: np.ndarray, presence: np.ndarray, threshold: float = 0.5
) -> dict[str, np.ndarray]:
    """Per-phase recall and precision at `threshold` (NaN where undefined)."""
    pred = p > threshold
    true = presence > 0
    tp = (pred & true).sum(0).astype(np.float64)
    n_true = true.sum(0)
    n_pred = pred.sum(0)
    with np.errstate(invalid="ignore"):
        recall = np.where(n_true > 0, tp / n_true, np.nan)
        precision = np.where(n_pred > 0, tp / n_pred, np.nan)
    return {"recall": recall, "precision": precision}


def recall_by_band(
    p: np.ndarray,
    presence: np.ndarray,
    f_rows: np.ndarray,
    bands: tuple = FREQUENCY_BANDS,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Recall per phase within bands of the bulk's own presence frequency.

    `f_rows` [R, P] is each row's bulk-level frequency; rows where the phase
    is truly present are grouped by that frequency, exposing how much of the
    apparent recall miss sits at intermediate (path-sensitive) frequencies.
    Returns `recall` and `n_present`, each [n_bands, P].
    """
    n_bands, n_phases = len(bands), p.shape[1]
    recall = np.full((n_bands, n_phases), np.nan)
    n_present = np.zeros((n_bands, n_phases), dtype=np.int64)
    detected = p > threshold
    true = presence > 0
    for k, (lo, hi) in enumerate(bands):
        in_band = true & (f_rows >= lo) & (f_rows < hi)
        n = in_band.sum(0)
        n_present[k] = n
        with np.errstate(invalid="ignore"):
            recall[k] = np.where(n > 0, (detected & in_band).sum(0) / n, np.nan)
    return {"recall": recall, "n_present": n_present}


PROVISIONAL = "provisional (partial corpus, provisional split)"


def _phase_grid(ncols: int = 5) -> tuple[plt.Figure, np.ndarray]:
    nrows = ceil(len(SOLID_PHASES) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False,
    )
    for ax in axes.flat[len(SOLID_PHASES):]:
        ax.set_visible(False)
    return fig, axes


def _finish(fig: plt.Figure, title: str, out_path: Path) -> None:
    fig.suptitle(f"{title} -- {PROVISIONAL}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(
    p_rows: np.ndarray, presence: np.ndarray, out_path: Path, n_bins: int = 10
) -> list[dict]:
    """Per-phase reliability diagrams; returns each phase's reliability table."""
    fig, axes = _phase_grid()
    tables = []
    for i, (ax, name) in enumerate(zip(axes.flat, SOLID_PHASES)):
        t = reliability_table(p_rows[:, i], presence[:, i], n_bins)
        tables.append(t)
        centers = (np.arange(n_bins) + 0.5) / n_bins
        twin = ax.twinx()
        twin.bar(centers, t["count"], width=1 / n_bins, color="0.85", zorder=0)
        twin.set_yticks([])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, zorder=1)
        ax.plot(t["mean_p"], t["freq"], "o-", color="#d95f02", ms=4, zorder=2)
        ax.set_zorder(twin.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name}  ECE={t['ece']:.3f}", fontsize=9)
        ax.set_xlabel("predicted p", fontsize=8)
        ax.set_ylabel("empirical freq", fontsize=8)
    _finish(fig, "presence reliability (bars: bin counts)", out_path)
    return tables


REGIME_PANELS = (
    ("plag", "plagioclase"),
    ("amp", "amphibole"),
    ("g", "garnet"),
    ("ne", "nepheline"),
)


def plot_reliability_subset(
    p_rows: np.ndarray,
    presence: np.ndarray,
    out_path: Path,
    panels: tuple = REGIME_PANELS,
    n_bins: int = 10,
) -> None:
    """One reliability panel per behavior regime, 2x2 (slide-sized)."""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0))
    for ax, (name, full_name) in zip(axes.flat, panels):
        i = SOLID_PHASES.index(name)
        t = reliability_table(p_rows[:, i], presence[:, i], n_bins)
        centers = (np.arange(n_bins) + 0.5) / n_bins
        twin = ax.twinx()
        twin.bar(centers, t["count"], width=1 / n_bins, color="0.85", zorder=0)
        twin.set_yticks([])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, zorder=1)
        ax.plot(t["mean_p"], t["freq"], "o-", color="#d95f02", ms=5, zorder=2)
        ax.set_zorder(twin.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{full_name}  ECE={t['ece']:.3f}", fontsize=12)
        ax.set_xlabel("predicted p", fontsize=10)
        ax.set_ylabel("empirical frequency", fontsize=10)
    _finish(fig, "presence reliability: the four behavior regimes", out_path)


def plot_pf_scatter(
    p_bulk: np.ndarray, f_bulk: np.ndarray, out_path: Path
) -> dict[str, np.ndarray]:
    """Per-phase p_i vs within-bulk frequency f_i; returns per-phase r and MAE."""
    fig, axes = _phase_grid()
    r = np.full(len(SOLID_PHASES), np.nan)
    mae = np.full(len(SOLID_PHASES), np.nan)
    for i, (ax, name) in enumerate(zip(axes.flat, SOLID_PHASES)):
        f, p = f_bulk[:, i], p_bulk[:, i]
        ok = ~np.isnan(f)
        f, p = f[ok], p[ok]
        if f.size and f.std() > 0 and p.std() > 0:
            r[i] = float(np.corrcoef(f, p)[0, 1])
        if f.size:
            mae[i] = float(np.abs(p - f).mean())
        ax.scatter(f, p, s=5, alpha=0.3, edgecolors="none", color="#1b9e77")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        tag = " [path-sensitive]" if name in PATH_SENSITIVE else ""
        ax.set_title(f"{name}{tag}  r={r[i]:.2f} MAE={mae[i]:.3f}", fontsize=9)
        ax.set_xlabel("within-bulk freq f", fontsize=8)
        ax.set_ylabel("predicted p", fontsize=8)
    _finish(fig, "presence p vs within-bulk frequency (one point per bulk)", out_path)
    return {"pf_r": r, "pf_mae": mae}


PF_REGIME_PANELS = (
    ("plag", "plagioclase"),
    ("amp", "amphibole"),
    ("spl", "spinel"),
    ("g", "garnet"),
)


def plot_pf_subset(
    p_bulk: np.ndarray,
    f_bulk: np.ndarray,
    out_path: Path,
    panels: tuple = PF_REGIME_PANELS,
) -> None:
    """One p-vs-frequency panel per behavior regime, 2x2 (slide-sized)."""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0))
    for ax, (name, full_name) in zip(axes.flat, panels):
        i = SOLID_PHASES.index(name)
        f, p = f_bulk[:, i], p_bulk[:, i]
        ok = ~np.isnan(f)
        f, p = f[ok], p[ok]
        r = float(np.corrcoef(f, p)[0, 1]) if f.size and f.std() > 0 else float("nan")
        mae = float(np.abs(p - f).mean()) if f.size else float("nan")
        ax.scatter(f, p, s=8, alpha=0.35, edgecolors="none", color="#1b9e77")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{full_name}  r={r:.2f}  MAE={mae:.3f}", fontsize=12)
        ax.set_xlabel("within-bulk frequency f", fontsize=10)
        ax.set_ylabel("predicted p", fontsize=10)
    _finish(fig, "presence p vs within-bulk frequency: the four behavior regimes",
            out_path)


def _parity_band(ax, levels: np.ndarray, sigmas: np.ndarray) -> None:
    ok = np.isfinite(levels) & np.isfinite(sigmas)
    if not ok.any():
        return
    x = np.linspace(0, 1, 101)
    s = np.interp(x, levels[ok], sigmas[ok])
    ax.fill_between(x, x - s, x + s, color="#ff6b6b", alpha=0.3, lw=0, zorder=2)


def plot_magnitude_parity(
    cond_mean: np.ndarray,
    magnitude: np.ndarray,
    presence: np.ndarray,
    band: tuple[np.ndarray, np.ndarray],
    out_path: Path,
) -> dict[str, np.ndarray]:
    """True vs predicted conditional magnitude on truly-present rows.

    Band around the diagonal is +/-1 within-bulk sd at that magnitude level
    (`band` from `within_bulk_sigma_profile`): the scatter a bulk-level
    predictor cannot remove.
    """
    fig, axes = _phase_grid()
    mae = np.full(len(SOLID_PHASES), np.nan)
    r = np.full(len(SOLID_PHASES), np.nan)
    for i, (ax, name) in enumerate(zip(axes.flat, SOLID_PHASES)):
        m = presence[:, i] > 0
        true, pred = magnitude[m, i], cond_mean[m, i]
        if true.size:
            mae[i] = float(np.abs(pred - true).mean())
            if true.std() > 0 and pred.std() > 0:
                r[i] = float(np.corrcoef(true, pred)[0, 1])
        if true.size > 5000:
            ax.hexbin(pred, true, gridsize=40, cmap="viridis", bins="log",
                      extent=(0, 1, 0, 1))
        else:
            ax.scatter(pred, true, s=5, alpha=0.3, edgecolors="none",
                       color="#7570b3")
        _parity_band(ax, band[0][i], band[1][i])
        ax.plot([0, 1], [0, 1], "w--" if true.size > 5000 else "k--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name}  MAE={mae[i]:.3f} r={r[i]:.2f} n={true.size}",
                     fontsize=9)
        ax.set_xlabel("predicted cond. mean", fontsize=8)
        ax.set_ylabel("true magnitude", fontsize=8)
    _finish(fig, "conditional magnitude parity (true support; band = +/-1 "
            "within-bulk sd at that magnitude level)", out_path)
    return {"magnitude_mae": mae, "magnitude_r": r}


PARITY_REGIME_PANELS = (
    ("afs", "alkali feldspar"),
    ("amp", "amphibole"),
    ("spl", "spinel"),
)


def plot_magnitude_parity_subset(
    cond_mean: np.ndarray,
    magnitude: np.ndarray,
    presence: np.ndarray,
    sigma_wb: np.ndarray,
    band: tuple[np.ndarray, np.ndarray],
    out_path: Path,
    panels: tuple = PARITY_REGIME_PANELS,
) -> None:
    """One magnitude-parity panel per behavior regime (slide-sized)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4))
    for ax, (name, full_name) in zip(np.atleast_1d(axes), panels):
        i = SOLID_PHASES.index(name)
        m = presence[:, i] > 0
        true, pred = magnitude[m, i], cond_mean[m, i]
        mae = float(np.abs(pred - true).mean())
        r = (float(np.corrcoef(true, pred)[0, 1])
             if true.std() > 0 and pred.std() > 0 else float("nan"))
        ax.hexbin(pred, true, gridsize=40, cmap="viridis", bins="log",
                  extent=(0, 1, 0, 1))
        _parity_band(ax, band[0][i], band[1][i])
        ax.plot([0, 1], [0, 1], "w--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{full_name}  MAE={mae:.3f}  r={r:.2f}  "
                     f"wb-sd={sigma_wb[i]:.2f}", fontsize=12)
        ax.set_xlabel("predicted cond. mean", fontsize=10)
        ax.set_ylabel("true magnitude", fontsize=10)
    _finish(fig, "conditional magnitude parity: behavior regimes "
            "(band = +/-1 within-bulk sd at that magnitude level)", out_path)


def plot_pit(pit: np.ndarray, out_path: Path, n_bins: int = 20) -> None:
    """Per-phase PIT histograms; flat = calibrated magnitude spread."""
    fig, axes = _phase_grid()
    for i, (ax, name) in enumerate(zip(axes.flat, SOLID_PHASES)):
        vals = pit[:, i]
        vals = vals[~np.isnan(vals)]
        if vals.size:
            ax.hist(vals, bins=n_bins, range=(0, 1), density=True,
                    color="#d95f02", alpha=0.8)
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_title(f"{name}  n={vals.size}", fontsize=9)
        ax.set_xlabel("PIT", fontsize=8)
    _finish(fig, "magnitude PIT (flat=calibrated, U=overconfident)", out_path)


PIT_REGIME_PANELS = (
    ("q", "quartz"),
    ("amp", "amphibole"),
    ("ru", "rutile"),
)


def plot_pit_subset(
    pit: np.ndarray,
    out_path: Path,
    panels: tuple = PIT_REGIME_PANELS,
    n_bins: int = 20,
) -> None:
    """One PIT panel per behavior regime (slide-sized)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4))
    for ax, (name, full_name) in zip(np.atleast_1d(axes), panels):
        i = SOLID_PHASES.index(name)
        v = pit[:, i]
        v = v[~np.isnan(v)]
        tails = float(((v < 0.05) | (v > 0.95)).mean())
        ax.hist(v, bins=n_bins, range=(0, 1), density=True,
                color="#d95f02", alpha=0.8)
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_title(f"{full_name}  mean={v.mean():.2f}  tails={tails:.2f}",
                     fontsize=12)
        ax.set_xlabel("PIT", fontsize=10)
        ax.set_ylabel("density", fontsize=10)
    _finish(fig, "magnitude PIT: behavior regimes "
            "(flat = honest spread; tails expected 0.10)", out_path)


def plot_alpha_concentration(
    alpha0: np.ndarray, support_size: np.ndarray, out_path: Path
) -> None:
    """Support-summed alpha_0 distributions by support size."""
    sizes = np.unique(support_size)
    groups = [alpha0[support_size == s] for s in sizes]
    fig, ax = plt.subplots(figsize=(max(6, len(sizes) * 0.8), 4.5))
    ax.boxplot(groups, tick_labels=[str(int(s)) for s in sizes], showfliers=False)
    ax.set_yscale("log")
    ax.set_xlabel("true support size")
    ax.set_ylabel("support-summed alpha_0")
    _finish(fig, "Dirichlet concentration by support size", out_path)


def plot_rung_frequency(
    f50: np.ndarray, f30: np.ndarray, p_bulk: np.ndarray, out_path: Path
) -> None:
    """f_i at rung 50 vs rung 30 per bulk, colored by the single per-bulk p_i."""
    ncols = 5
    nrows = ceil(len(SOLID_PHASES) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False,
        layout="constrained",
    )
    for ax in axes.flat[len(SOLID_PHASES):]:
        ax.set_visible(False)
    sc = None
    for i, (ax, name) in enumerate(zip(axes.flat, SOLID_PHASES)):
        ok = ~(np.isnan(f50[:, i]) | np.isnan(f30[:, i]))
        sc = ax.scatter(
            f50[ok, i], f30[ok, i], c=p_bulk[ok, i], s=6, alpha=0.5,
            edgecolors="none", cmap="coolwarm", vmin=0, vmax=1,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("f at rung 50", fontsize=8)
        ax.set_ylabel("f at rung 30", fontsize=8)
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="predicted p", shrink=0.5)
    fig.suptitle(f"presence frequency by rung, colored by p -- {PROVISIONAL}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


RUNG_REGIME_PANELS = (
    ("q", "quartz"),
    ("opx", "orthopyroxene"),
    ("cpx", "clinopyroxene"),
    ("g", "garnet"),
)


def plot_rung_subset(
    f50: np.ndarray,
    f30: np.ndarray,
    p_bulk: np.ndarray,
    out_path: Path,
    panels: tuple = RUNG_REGIME_PANELS,
) -> None:
    """One rung-comparison panel per behavior regime, 2x2 (slide-sized)."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), layout="constrained")
    sc = None
    for ax, (name, full_name) in zip(axes.flat, panels):
        i = SOLID_PHASES.index(name)
        ok = ~(np.isnan(f50[:, i]) | np.isnan(f30[:, i]))
        d = f30[ok, i] - f50[ok, i]
        sc = ax.scatter(
            f50[ok, i], f30[ok, i], c=p_bulk[ok, i], s=10, alpha=0.55,
            edgecolors="none", cmap="coolwarm", vmin=0, vmax=1,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{full_name}  mean(f30-f50)={d.mean():+.2f}", fontsize=12)
        ax.set_xlabel("f at rung 50", fontsize=10)
        ax.set_ylabel("f at rung 30", fontsize=10)
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="predicted p", shrink=0.6)
    fig.suptitle("presence frequency by rung: the four behavior regimes -- "
                 f"{PROVISIONAL}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mass_guardrails(
    fp_mass: np.ndarray, fn_mass: np.ndarray, out_path: Path
) -> None:
    """Per-row FP-mass and FN-mass distributions (log counts)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, label in (
        (axes[0], fp_mass, "false-positive mass"),
        (axes[1], fn_mass, "false-negative mass"),
    ):
        ax.hist(vals, bins=50, range=(0, max(0.01, float(vals.max()))),
                color="#d95f02")
        ax.set_yscale("log")
        ax.set_xlabel(f"{label} per row (fraction of solid mass)")
        ax.set_ylabel("rows")
        ax.set_title(f"mean={vals.mean():.4f}  p95={np.quantile(vals, 0.95):.4f}",
                     fontsize=9)
    _finish(fig, "thresholded-view mass guardrails", out_path)


def _rarity_band(frequency: float) -> str:
    for name, lo, hi in RARITY_BANDS:
        if lo <= frequency < hi:
            return name
    return "rare"


def _qap_labels(names: np.ndarray, top_k: int = 8) -> np.ndarray:
    """Collapse qap names to the top_k most frequent + 'other'/'unknown'."""
    out = np.where(pd.isna(names) | (names == ""), "unknown", names.astype(object))
    top = pd.Series(out[out != "unknown"]).value_counts().head(top_k).index
    return np.where(np.isin(out, top) | (out == "unknown"), out, "other")


@torch.no_grad()
def run_mode_head_eval(
    head_checkpoint_path: Path,
    *,
    seed: int = 0,
    val_fraction: float = 0.20,
    device: torch.device | None = None,
) -> dict:
    """Full interrogation of a ModeHead checkpoint on its provisional val split.

    `seed` / `val_fraction` must match the values the checkpoint was trained
    with (the split is rebuilt, not stored). Writes figures + `metrics.json`
    to `results/mode_head/<checkpoint dir name>/` and returns the metrics.
    """
    head_checkpoint_path = Path(head_checkpoint_path)
    ckpt = ModeHeadCheckpoint.load(head_checkpoint_path)
    device = device or pick_device()
    tag = head_checkpoint_path.parent.name
    output_dir = RESULTS_ROOT / "mode_head" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device:  {device}")
    print(f"head:    {head_checkpoint_path}")
    print(f"encoder: {ckpt.encoder_checkpoint}")
    print(f"output:  {output_dir}")

    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    head = ckpt.build_head().to(device)

    lc = ckpt.label_config
    dataset = ModeHeadDataset(
        corpus_dir=lc["corpus_dir"],
        rungs=tuple(lc["rungs"]),
        presence_cutoff=lc["presence_cutoff"],
    )
    latents = _bulk_latents(encoder, dataset, device)
    _, val_rows = _row_split_indices(dataset, val_fraction, seed)
    print(f"bulks={len(dataset.bulk_ids)} rows={len(dataset)} "
          f"n_val={len(val_rows)} (provisional split, seed={seed})")

    logits_bulk, alpha_bulk = head(latents)
    p_bulk = torch.sigmoid(logits_bulk).cpu().numpy()
    alpha_bulk_np = alpha_bulk.cpu().numpy()

    row_bulk = dataset.row_bulk_idx[val_rows].numpy()
    presence = dataset.presence[val_rows].numpy()
    magnitude = dataset.magnitude[val_rows].numpy()
    rung = dataset.rung[val_rows].numpy()
    qap = _qap_labels(dataset.bulk_qap_names[row_bulk])
    p_rows = p_bulk[row_bulk]
    alpha_rows = alpha_bulk_np[row_bulk]
    support_size = presence.sum(-1).astype(np.int64)

    nll = hurdle_nll_rows(
        logits_bulk[torch.as_tensor(row_bulk, device=device)],
        alpha_bulk[torch.as_tensor(row_bulk, device=device)],
        torch.as_tensor(presence, device=device),
        torch.as_tensor(magnitude, device=device),
    )

    n_bulks = len(dataset.bulk_ids)
    f_all, _ = within_bulk_frequency(presence, row_bulk, n_bulks)
    f50, _ = within_bulk_frequency(presence[rung == 50], row_bulk[rung == 50], n_bulks)
    f30, _ = within_bulk_frequency(presence[rung == 30], row_bulk[rung == 30], n_bulks)
    val_bulks = np.unique(row_bulk)
    f_rows = f_all[row_bulk]

    rel_tables = plot_reliability(p_rows, presence, output_dir / "reliability.png")
    plot_reliability_subset(p_rows, presence, output_dir / "reliability_subset.png")
    pf = plot_pf_scatter(
        p_bulk[val_bulks], f_all[val_bulks], output_dir / "p_vs_frequency.png"
    )
    plot_pf_subset(
        p_bulk[val_bulks], f_all[val_bulks], output_dir / "p_vs_frequency_subset.png"
    )
    cm_true = conditional_mean(alpha_rows, presence)
    sigma_wb = within_bulk_conditional_sigma(magnitude, presence, row_bulk)
    band = within_bulk_sigma_profile(magnitude, presence, row_bulk)
    parity = plot_magnitude_parity(
        cm_true, magnitude, presence, band, output_dir / "magnitude_parity.png"
    )
    plot_magnitude_parity_subset(
        cm_true, magnitude, presence, sigma_wb, band,
        output_dir / "magnitude_parity_subset.png",
    )
    pit = beta_pit(alpha_rows, presence, magnitude)
    plot_pit(pit, output_dir / "magnitude_pit.png")
    plot_pit_subset(pit, output_dir / "magnitude_pit_subset.png")
    alpha0 = (alpha_rows * presence).sum(-1)
    plot_alpha_concentration(alpha0, support_size, output_dir / "alpha_concentration.png")
    plot_rung_frequency(
        f50[val_bulks], f30[val_bulks], p_bulk[val_bulks],
        output_dir / "rung_frequency.png",
    )
    plot_rung_subset(
        f50[val_bulks], f30[val_bulks], p_bulk[val_bulks],
        output_dir / "rung_frequency_subset.png",
    )
    masses = fp_fn_mass(p_rows, alpha_rows, presence, magnitude)
    plot_mass_guardrails(
        masses["fp_mass"], masses["fn_mass"], output_dir / "mass_guardrails.png"
    )

    scores = presence_scores(p_rows, presence)
    band_recall = recall_by_band(p_rows, presence, f_rows)
    frequency = presence.mean(0)
    bce_phase = nll["bce_per_phase"].mean(0)

    per_phase = []
    for i, name in enumerate(SOLID_PHASES):
        per_phase.append({
            "phase": name,
            "frequency": float(frequency[i]),
            "rarity": _rarity_band(float(frequency[i])),
            "path_sensitive": name in PATH_SENSITIVE,
            "ece": rel_tables[i]["ece"],
            "pf_r": float(pf["pf_r"][i]),
            "pf_mae": float(pf["pf_mae"][i]),
            "bernoulli_nll": float(bce_phase[i]),
            "magnitude_mae": float(parity["magnitude_mae"][i]),
            "magnitude_r": float(parity["magnitude_r"][i]),
            "within_bulk_sd": float(sigma_wb[i]),
            "recall": float(scores["recall"][i]),
            "precision": float(scores["precision"][i]),
            "fp_mass_mean": float(masses["fp_mass_per_phase"][:, i].mean()),
        })

    metrics = {
        "head_checkpoint": str(head_checkpoint_path),
        "encoder_checkpoint": str(ckpt.encoder_checkpoint),
        "label_config": dict(lc),
        "split": {"kind": "provisional", "seed": seed, "val_fraction": val_fraction},
        "n_val_rows": int(len(val_rows)),
        "n_val_bulks": int(len(val_bulks)),
        "joint_nll": float(nll["joint"].mean()),
        "bernoulli_nll": float(nll["bernoulli"].mean()),
        "dirichlet_nll": float(nll["dirichlet"].mean()),
        "joint_nll_by_rung": stratified_mean(nll["joint"], rung),
        "joint_nll_by_qap": stratified_mean(nll["joint"], qap),
        "joint_nll_by_support_size": stratified_mean(nll["joint"], support_size),
        "fp_mass": {"mean": float(masses["fp_mass"].mean()),
                    "p95": float(np.quantile(masses["fp_mass"], 0.95))},
        "fn_mass": {"mean": float(masses["fn_mass"].mean()),
                    "p95": float(np.quantile(masses["fn_mass"], 0.95))},
        "recall_by_frequency_band": {
            f"[{lo},{hi})": {
                "recall": [None if np.isnan(v) else float(v)
                           for v in band_recall["recall"][k]],
                "n_present": band_recall["n_present"][k].tolist(),
            }
            for k, (lo, hi) in enumerate(FREQUENCY_BANDS)
        },
        "per_phase": per_phase,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    table = pd.DataFrame(per_phase).sort_values("frequency", ascending=False)
    print(f"\njoint hurdle NLL: {metrics['joint_nll']:.4f} nats/row "
          f"(bernoulli {metrics['bernoulli_nll']:.4f} + "
          f"dirichlet {metrics['dirichlet_nll']:.4f})")
    print("by rung:", {k: round(v["mean"], 4)
                       for k, v in metrics["joint_nll_by_rung"].items()})
    print(f"fp mass mean={metrics['fp_mass']['mean']:.4f} "
          f"p95={metrics['fp_mass']['p95']:.4f}  "
          f"fn mass mean={metrics['fn_mass']['mean']:.4f} "
          f"p95={metrics['fn_mass']['p95']:.4f}")
    print(table.round(4).to_string(index=False))
    print(f"metrics -> {output_dir / 'metrics.json'}")
    return metrics
