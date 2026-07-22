"""Interrogation report for the CompHead of a trained joint checkpoint.

Predicts one oxide Dirichlet per (bulk, phase) and scores it on present rows
only: the masked comp Dirichlet NLL as the headline (per phase, per rung),
per-(phase, oxide) parity, Beta-marginal PIT calibration, the
shared-concentration profile, the mass-balance closure residual, and a plag
An# face-validity check. The closure here uses the TRUE modes (not
ModeHead's), isolating composition error from mode error; melt/fluid fixings
are teacher-forced as in training. `predicted_modes_closure` /
`run_predicted_closure_eval` are the predicted-modes variant (the cross-epoch
selection gate and OOD diagnostic). Statistics live as pure array functions;
`run_comp_head_eval` takes the shared `(ckpt, dataset, latents)` built by
`eval.joint_heads` and emits figures + `metrics.json` under
`runs/comp_head/<tag>/`.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import betainc

from cnmineral.data.magemin_corpus import COMP_OXIDES
from cnmineral.data.mineralogy_dataset import JointHeadDataset
from cnmineral.data.phase_vocab import PHASE_INDEX, SOLID_PHASES
from cnmineral.eval.mode_head import (
    RUNS_ROOT,
    _finish,
    _phase_grid,
    magnitude_alpha_rows,
    stratified_mean,
)
from cnmineral.losses.comp_loss import masked_comp_dirichlet_nll
from cnmineral.training.joint_heads import JointHeadCheckpoint

PARITY_OXIDES = ("SiO2", "Al2O3", "CaO", "FeO")
SUBSET_PHASES = ("plag", "cpx", "amp")


def comp_nll_matrix(
    alpha: np.ndarray, comp_target: np.ndarray, presence: np.ndarray
) -> np.ndarray:
    """Per-(row, phase) Dirichlet NLL [R, P], NaN at absent phases."""
    nll = (
        masked_comp_dirichlet_nll(
            torch.as_tensor(alpha),
            torch.as_tensor(comp_target),
            torch.as_tensor(presence),
            reduction="none",
        )
        .numpy()
    )
    return np.where(presence > 0, nll, np.nan)


def comp_beta_pit(
    alpha: np.ndarray, comp_target: np.ndarray, presence: np.ndarray
) -> np.ndarray:
    """PIT of each oxide fraction under its Beta marginal, [R, P, O].

    Dirichlet marginals are Betas: `target_o ~ Beta(alpha_o, alpha0 -
    alpha_o)`. NaN at absent phases.
    """
    alpha0 = alpha.sum(-1, keepdims=True)
    b = alpha0 - alpha
    x = np.clip(comp_target, 1e-12, 1.0 - 1e-12)
    pit = betainc(alpha, b, x)
    return np.where(presence[..., None] > 0, pit, np.nan)


def comp_parity_stats(
    mean: np.ndarray, comp_target: np.ndarray, presence: np.ndarray
) -> dict:
    """Per-(phase, oxide) MAE and correlation of predicted means on present rows.

    Returns `mae` and `r` [P, O]; NaN where a phase never occurs (or its
    target has zero variance, for `r`).
    """
    n_phases, n_ox = mean.shape[1], mean.shape[2]
    mae = np.full((n_phases, n_ox), np.nan)
    r = np.full((n_phases, n_ox), np.nan)
    for p in range(n_phases):
        m = presence[:, p] > 0
        if not m.any():
            continue
        err = mean[m, p] - comp_target[m, p]
        mae[p] = np.abs(err).mean(0)
        for o in range(n_ox):
            t = comp_target[m, p, o]
            if t.std() > 0 and mean[m, p, o].std() > 0:
                r[p, o] = np.corrcoef(mean[m, p, o], t)[0, 1]
    return {"mae": mae, "r": r}


def an_number(comp: np.ndarray) -> np.ndarray:
    """Molar An# = CaO / (CaO + Na2O + K2O) from oxide mass fractions [..., O]."""
    mw = {"CaO": 56.08, "Na2O": 61.98, "K2O": 94.20}
    idx = {o: COMP_OXIDES.index(o) for o in mw}
    ca = comp[..., idx["CaO"]] / mw["CaO"]
    na = comp[..., idx["Na2O"]] / mw["Na2O"]
    k = comp[..., idx["K2O"]] / mw["K2O"]
    denom = ca + na + k
    return np.divide(ca, denom, out=np.full_like(ca, np.nan), where=denom > 0)


def plot_comp_parity(
    mean: np.ndarray, comp_target: np.ndarray, presence: np.ndarray,
    oxide: str, out_path: Path, split_tag: str,
) -> None:
    """21-phase grid of predicted-vs-true parity for one oxide."""
    o = COMP_OXIDES.index(oxide)
    fig, axes = _phase_grid()
    for i, (name, ax) in enumerate(zip(SOLID_PHASES, axes.flat)):
        m = presence[:, i] > 0
        if m.any():
            ax.hexbin(comp_target[m, i, o], mean[m, i, o], gridsize=30,
                      cmap="viridis", mincnt=1)
        lim = (0, max(0.05, comp_target[:, i, o].max() if m.any() else 1.0))
        ax.plot(lim, lim, color="red", lw=0.8)
        ax.set_title(name, fontsize=9)
    _finish(fig, f"CompHead {oxide} parity (present rows)", out_path, split_tag)


def plot_comp_parity_subset(
    mean: np.ndarray, comp_target: np.ndarray, presence: np.ndarray,
    out_path: Path, split_tag: str,
) -> None:
    """`SUBSET_PHASES` x `PARITY_OXIDES` parity panels."""
    fig, axes = plt.subplots(
        len(SUBSET_PHASES), len(PARITY_OXIDES),
        figsize=(3.2 * len(PARITY_OXIDES), 3.0 * len(SUBSET_PHASES)),
        squeeze=False,
    )
    for row, phase in enumerate(SUBSET_PHASES):
        i = PHASE_INDEX[phase]
        m = presence[:, i] > 0
        for col, oxide in enumerate(PARITY_OXIDES):
            o = COMP_OXIDES.index(oxide)
            ax = axes[row, col]
            if m.any():
                ax.hexbin(comp_target[m, i, o], mean[m, i, o], gridsize=30,
                          cmap="viridis", mincnt=1)
                lim = (0, max(0.05, comp_target[m, i, o].max()))
                ax.plot(lim, lim, color="red", lw=0.8)
            ax.set_title(f"{phase} {oxide}", fontsize=9)
    _finish(fig, "CompHead parity (subset)", out_path, split_tag)


def plot_comp_pit(pit: np.ndarray, out_path: Path, split_tag: str) -> None:
    """21-phase grid of PIT histograms pooled over the 11 oxides."""
    fig, axes = _phase_grid()
    for i, (name, ax) in enumerate(zip(SOLID_PHASES, axes.flat)):
        v = pit[:, i, :].ravel()
        v = v[~np.isnan(v)]
        if len(v):
            ax.hist(v, bins=20, range=(0, 1), density=True, color="tab:blue")
        ax.axhline(1.0, color="red", lw=0.8, ls="--")
        ax.set_title(name, fontsize=9)
    _finish(fig, "CompHead oxide PIT (pooled over oxides)", out_path, split_tag)


def plot_comp_concentration(
    alpha0: np.ndarray, presence: np.ndarray, out_path: Path, split_tag: str
) -> None:
    """Per-phase distribution of the shared concentration on present rows."""
    fig, ax = plt.subplots(figsize=(10, 4))
    data = []
    for i in range(len(SOLID_PHASES)):
        m = presence[:, i] > 0
        data.append(np.log10(alpha0[m, i]) if m.any() else np.array([np.nan]))
    ax.boxplot(data, labels=SOLID_PHASES, showfliers=False)
    ax.set_ylabel("log10 alpha0")
    ax.tick_params(axis="x", rotation=60)
    _finish(fig, "CompHead shared concentration", out_path, split_tag)


@torch.no_grad()
def predicted_modes_closure(
    mode_head, comp_head, dataset, latents: torch.Tensor
) -> np.ndarray:
    """Per-row signed closure residual [R, O] from predicted modes x comps.

    Composes the two heads exactly as the training closure does -- true-
    presence-masked mode alphas renormalized to modes, times comp simplex
    means, plus the corpus-pinned melt/fluid fixings -- so it measures
    whether a (mode, comp) pairing stays mass-balance consistent.  This is
    the cross-epoch gate instrument for per-head checkpoint selection and
    the OOD diagnostic's numerator; the closure in `run_comp_head_eval`
    instead uses TRUE modes to isolate composition error.
    """
    device = latents.device
    row_bulk = dataset.row_bulk_idx.to(device)
    presence = dataset.presence.to(device)
    mode_alpha = magnitude_alpha_rows(mode_head, latents, row_bulk, presence)
    comp_alpha = comp_head(latents)
    masked = presence * mode_alpha
    modes = masked / masked.sum(-1, keepdim=True).clamp_min(1e-12)
    comp_mean = comp_alpha[row_bulk]
    comp_mean = comp_mean / comp_mean.sum(-1, keepdim=True)
    solids = torch.einsum("rp,rpo->ro", modes, comp_mean)
    f_liq = dataset.f_liq.to(device)
    f_fl = dataset.f_fl.to(device)
    recon = (
        (1.0 - f_liq - f_fl).unsqueeze(-1) * solids
        + f_liq.unsqueeze(-1) * dataset.c_liq.to(device)
        + f_fl.unsqueeze(-1) * dataset.c_fl.to(device)
    )
    return (recon - dataset.bulk_11.to(device)).cpu().numpy()


def plot_closure_residual(
    residual: np.ndarray, out_path: Path, split_tag: str
) -> None:
    """Per-row L2 norm histogram + per-oxide signed residual boxes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(np.linalg.norm(residual, axis=1), bins=60, color="tab:blue")
    ax1.set_yscale("log")
    ax1.set_xlabel("per-row L2 residual")
    ax2.boxplot(list(residual.T), labels=COMP_OXIDES, showfliers=False)
    ax2.axhline(0.0, color="red", lw=0.8)
    ax2.tick_params(axis="x", rotation=60)
    ax2.set_ylabel("signed residual (mass frac)")
    _finish(fig, "Mass-balance closure residual", out_path, split_tag)


def plot_an_parity(
    an_pred: np.ndarray, an_true: np.ndarray, out_path: Path, split_tag: str
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.hexbin(an_true, an_pred, gridsize=40, cmap="viridis", mincnt=1)
    ax.plot([0, 1], [0, 1], color="red", lw=0.8)
    ax.set_xlabel("true An#")
    ax.set_ylabel("predicted An#")
    _finish(fig, "plag An# parity (derived from oxide means)", out_path, split_tag)


@torch.no_grad()
def run_predicted_closure_eval(
    ckpt: JointHeadCheckpoint,
    dataset: JointHeadDataset,
    latents: torch.Tensor,
    *,
    device: torch.device,
    tag: str,
) -> dict:
    """Predicted-modes closure summary for one checkpoint's (mode, comp) pair.

    Builds both heads at their stamped epochs and summarizes
    `predicted_modes_closure`: per-row L2 mean/p95 plus per-oxide signed-mean
    residuals.  Run on two checkpoints to compare pairings -- e.g. per-head
    bests against a joint-epoch snapshot for the cross-epoch selection gate.
    """
    mode_head = ckpt.build_mode_head().to(device)
    comp_head = ckpt.build_comp_head().to(device)
    residual = predicted_modes_closure(mode_head, comp_head, dataset, latents)
    l2 = np.linalg.norm(residual, axis=1)
    summary = {
        "checkpoint_tag": tag,
        "n_val_rows": int(len(dataset)),
        "closure_l2": {
            "mean": float(l2.mean()),
            "p95": float(np.quantile(l2, 0.95)),
        },
        "signed_mean_by_oxide": {
            o: float(residual[:, j].mean()) for j, o in enumerate(COMP_OXIDES)
        },
    }
    print(
        f"predicted-modes closure ({tag}): "
        f"l2 mean={summary['closure_l2']['mean']:.4f} "
        f"p95={summary['closure_l2']['p95']:.4f}"
    )
    return summary


@torch.no_grad()
def run_comp_head_eval(
    ckpt: JointHeadCheckpoint,
    dataset: JointHeadDataset,
    latents: torch.Tensor,
    *,
    device: torch.device,
    tag: str,
) -> dict:
    """Full CompHead interrogation on the valtest val roster.

    `ckpt`, `dataset`, and `latents` come from the shared setup in
    `eval.joint_heads`. Writes figures + `metrics.json` to
    `runs/comp_head/<tag>/` and returns the metrics.
    """
    output_dir = RUNS_ROOT / "comp_head" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output:  {output_dir}")

    comp_head = ckpt.build_comp_head().to(device)
    split_tag = f"val ({len(dataset.bulk_ids)} bulks, 128 draws/bulk)"

    alpha_bulk = comp_head(latents).cpu().numpy()
    row_bulk = dataset.row_bulk_idx.numpy()
    alpha = alpha_bulk[row_bulk]
    mean = alpha / alpha.sum(-1, keepdims=True)
    presence = dataset.presence.numpy()
    comp_target = dataset.comp_target.numpy()
    rung = dataset.rung.numpy()

    nll = comp_nll_matrix(alpha, comp_target, presence)
    row_nll = np.nanmean(nll, axis=1)
    pit = comp_beta_pit(alpha, comp_target, presence)
    parity = comp_parity_stats(mean, comp_target, presence)

    cond_modes = dataset.magnitude.numpy()
    solids = np.einsum("rp,rpo->ro", cond_modes, mean)
    f_liq = dataset.f_liq.numpy()
    f_fl = dataset.f_fl.numpy()
    recon = (
        (1.0 - f_liq - f_fl)[:, None] * solids
        + f_liq[:, None] * dataset.c_liq.numpy()
        + f_fl[:, None] * dataset.c_fl.numpy()
    )
    residual = recon - dataset.bulk_11.numpy()
    residual_l2 = np.linalg.norm(residual, axis=1)

    plag = PHASE_INDEX["plag"]
    m_plag = presence[:, plag] > 0
    an_pred = an_number(mean[m_plag, plag])
    an_true = an_number(comp_target[m_plag, plag])

    for oxide in PARITY_OXIDES:
        plot_comp_parity(
            mean, comp_target, presence, oxide,
            output_dir / f"comp_parity_{oxide}.png", split_tag,
        )
    plot_comp_parity_subset(
        mean, comp_target, presence,
        output_dir / "comp_parity_subset.png", split_tag,
    )
    plot_comp_pit(pit, output_dir / "comp_pit.png", split_tag)
    plot_comp_concentration(
        alpha.sum(-1), presence, output_dir / "comp_concentration.png", split_tag
    )
    plot_closure_residual(residual, output_dir / "closure_residual.png", split_tag)
    plot_an_parity(an_pred, an_true, output_dir / "an_parity.png", split_tag)

    tail_lo = np.nanmean(pit < 0.1, axis=0)
    tail_hi = np.nanmean(pit > 0.9, axis=0)
    per_phase = []
    for i, name in enumerate(SOLID_PHASES):
        n_present = int((presence[:, i] > 0).sum())
        per_phase.append({
            "phase": name,
            "n_present": n_present,
            "comp_nll": float(np.nanmean(nll[:, i])) if n_present else None,
            "mae_by_oxide": {
                o: None if np.isnan(parity["mae"][i, j]) else float(parity["mae"][i, j])
                for j, o in enumerate(COMP_OXIDES)
            },
            "r_by_oxide": {
                o: None if np.isnan(parity["r"][i, j]) else float(parity["r"][i, j])
                for j, o in enumerate(COMP_OXIDES)
            },
            "pit_tail_lo": float(np.nanmean(tail_lo[i])),
            "pit_tail_hi": float(np.nanmean(tail_hi[i])),
        })

    metrics = {
        "checkpoint_tag": tag,
        "encoder_checkpoint": str(ckpt.encoder_checkpoint),
        "label_config": dict(ckpt.label_config),
        "n_val_rows": int(len(dataset)),
        "n_val_bulks": int(len(dataset.bulk_ids)),
        "comp_nll": float(np.nanmean(nll)),
        "comp_nll_by_rung": stratified_mean(row_nll, rung),
        "closure_l2": {
            "mean": float(residual_l2.mean()),
            "p95": float(np.quantile(residual_l2, 0.95)),
        },
        "closure_abs_by_oxide": {
            o: float(np.abs(residual[:, j]).mean())
            for j, o in enumerate(COMP_OXIDES)
        },
        "an_parity": {
            "mae": float(np.nanmean(np.abs(an_pred - an_true))),
            "r": float(np.corrcoef(an_pred, an_true)[0, 1]) if len(an_pred) > 1 else None,
        },
        "per_phase": per_phase,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    table = pd.DataFrame(
        [{
            "phase": r["phase"], "n_present": r["n_present"],
            "comp_nll": r["comp_nll"],
            "mae_SiO2": r["mae_by_oxide"]["SiO2"],
            "pit_lo": r["pit_tail_lo"], "pit_hi": r["pit_tail_hi"],
        } for r in per_phase]
    ).sort_values("n_present", ascending=False)
    print(f"\ncomp NLL: {metrics['comp_nll']:.4f} nats per present (row, phase)")
    print(f"closure L2 mean={metrics['closure_l2']['mean']:.4f} "
          f"p95={metrics['closure_l2']['p95']:.4f}")
    print(f"plag An# mae={metrics['an_parity']['mae']:.4f} "
          f"r={metrics['an_parity']['r']:.4f}")
    print(table.round(4).to_string(index=False))
    print(f"metrics -> {output_dir / 'metrics.json'}")
    return metrics
