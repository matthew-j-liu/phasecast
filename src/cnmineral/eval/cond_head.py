"""Interrogation report for the CondHead of a trained joint checkpoint.

Scores the support-conditioned per-axis GMM over (T_end, P_kbar, H2O):
per-axis NLL (stratified by rung and support size) as the headline,
mixture-CDF PIT and central-interval coverage for calibration,
component-usage diagnostics, and the exposure-bias measurement -- the same
rows scored under the teacher-forced true support vs supports sampled from
the checkpoint's own ModeHead, the decision input for mask-noising
mitigations (heads.md). Statistics live as pure array functions;
`run_cond_head_eval` takes the shared `(ckpt, dataset, latents)` built by
`eval.joint_heads` and emits figures + `metrics.json` under
`runs/cond_head/<tag>/`.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from cnmineral.data.mineralogy_dataset import JointHeadDataset
from cnmineral.data.phase_vocab import PHASE_INDEX
from cnmineral.eval.mode_head import RUNS_ROOT, _finish, stratified_mean
from cnmineral.models.cond_head import COND_AXES
from cnmineral.training.joint_heads import JointHeadCheckpoint

COVERAGE_LEVELS = (0.5, 0.68, 0.9)
DIAGNOSTIC_PHASES = ("amp", "q", "ne")


def gmm_nll_per_axis(
    log_w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Per-(row, axis) mixture NLL [R, A]; parameters [R, A, K]."""
    t = target[..., None]
    log_norm = (
        -0.5 * ((t - mu) / sigma) ** 2
        - np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
    )
    return -np.log(np.exp(log_w + log_norm).sum(-1).clip(1e-300))


def gmm_pit(
    log_w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Mixture-CDF PIT per (row, axis) [R, A]."""
    t = target[..., None]
    return (np.exp(log_w) * norm.cdf((t - mu) / sigma)).sum(-1)


def central_coverage(pit: np.ndarray, levels: tuple[float, ...]) -> dict:
    """Empirical mass of central intervals per axis, keyed by nominal level."""
    out = {}
    for lv in levels:
        inside = np.abs(pit - 0.5) < lv / 2
        out[str(lv)] = [float(v) for v in inside.mean(0)]
    return out


def effective_components(log_w: np.ndarray) -> np.ndarray:
    """exp(entropy of mixture weights) per (row, axis) [R, A]."""
    w = np.exp(log_w)
    ent = -(w * np.where(w > 0, np.log(w.clip(1e-12)), 0.0)).sum(-1)
    return np.exp(ent)


def mass_below_zero(
    log_w: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Mixture mass below 0 per (row, axis) [R, A] -- a diagnostic, not a bug."""
    return (np.exp(log_w) * norm.cdf((0.0 - mu) / sigma)).sum(-1)


def sample_supports(
    p: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One Bernoulli support draw per row from presence probabilities [R, P]."""
    return (rng.random(p.shape) < p).astype(np.float32)


def plot_cond_pit(pit: np.ndarray, out_path: Path, split_tag: str) -> None:
    fig, axes = plt.subplots(1, len(COND_AXES), figsize=(11, 3.4))
    for a, (name, ax) in enumerate(zip(COND_AXES, axes)):
        ax.hist(pit[:, a], bins=20, range=(0, 1), density=True, color="tab:blue")
        ax.axhline(1.0, color="red", lw=0.8, ls="--")
        ax.set_title(name, fontsize=10)
    _finish(fig, "CondHead mixture PIT (teacher-forced support)", out_path, split_tag)


def plot_cond_coverage(coverage: dict, out_path: Path, split_tag: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    levels = [float(k) for k in coverage]
    for a, name in enumerate(COND_AXES):
        ax.plot(levels, [coverage[str(lv)][a] for lv in levels],
                marker="o", label=name)
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--")
    ax.set_xlabel("nominal central interval")
    ax.set_ylabel("empirical coverage")
    ax.legend()
    _finish(fig, "CondHead central-interval coverage", out_path, split_tag)


def plot_cond_components(
    log_w: np.ndarray, mu: np.ndarray, eff: np.ndarray,
    out_path: Path, split_tag: str,
) -> None:
    """(T, P) component-mean scatter colored by weight + effective-K hists."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    w = np.exp(log_w)
    t_idx, p_idx = COND_AXES.index("T"), COND_AXES.index("P_kbar")
    sc = ax1.scatter(
        mu[:, t_idx, :].ravel(), mu[:, p_idx, :].ravel(),
        c=w[:, t_idx, :].ravel(), s=4, alpha=0.3, cmap="viridis",
    )
    fig.colorbar(sc, ax=ax1, label="T-axis weight")
    ax1.set_xlabel("component mu_T (C)")
    ax1.set_ylabel("component mu_P (kbar)")
    for a, name in enumerate(COND_AXES):
        ax2.hist(eff[:, a], bins=30, histtype="step", label=name)
    ax2.set_xlabel("effective components exp(H(w))")
    ax2.legend()
    _finish(fig, "CondHead mixture components", out_path, split_tag)


def plot_cond_by_phase(
    pit: np.ndarray, presence: np.ndarray, out_path: Path, split_tag: str
) -> None:
    """PIT per axis, split by presence of each diagnostic phase."""
    fig, axes = plt.subplots(
        len(DIAGNOSTIC_PHASES), len(COND_AXES),
        figsize=(11, 3.0 * len(DIAGNOSTIC_PHASES)), squeeze=False,
    )
    for r, phase in enumerate(DIAGNOSTIC_PHASES):
        m = presence[:, PHASE_INDEX[phase]] > 0
        for a, name in enumerate(COND_AXES):
            ax = axes[r, a]
            ax.hist(pit[m, a], bins=20, range=(0, 1), density=True,
                    histtype="step", label=f"{phase} present")
            ax.hist(pit[~m, a], bins=20, range=(0, 1), density=True,
                    histtype="step", label=f"{phase} absent")
            ax.set_title(f"{name} | {phase}", fontsize=9)
            if a == 0:
                ax.legend(fontsize=7)
    _finish(fig, "CondHead PIT by diagnostic-phase presence", out_path, split_tag)


def plot_nll_by_support_size(
    nll: np.ndarray, support_size: np.ndarray, out_path: Path, split_tag: str
) -> None:
    fig, axes = plt.subplots(1, len(COND_AXES), figsize=(11, 3.4))
    sizes = np.unique(support_size)
    for a, (name, ax) in enumerate(zip(COND_AXES, axes)):
        means = [nll[support_size == s, a].mean() for s in sizes]
        ax.plot(sizes, means, marker="o")
        ax.set_xlabel("support size")
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel("NLL (nats)")
    _finish(fig, "CondHead NLL by support size", out_path, split_tag)


@torch.no_grad()
def run_cond_head_eval(
    ckpt: JointHeadCheckpoint,
    dataset: JointHeadDataset,
    latents: torch.Tensor,
    *,
    device: torch.device,
    tag: str,
    exposure_seed: int = 0,
) -> dict:
    """Full CondHead interrogation on the valtest val roster.

    `ckpt`, `dataset`, and `latents` come from the shared setup in
    `eval.joint_heads`. Writes figures + `metrics.json` to
    `runs/cond_head/<tag>/` and returns the metrics.
    """
    output_dir = RUNS_ROOT / "cond_head" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output:  {output_dir}")

    lc = ckpt.label_config
    cond_head = ckpt.build_cond_head().to(device)
    mode_head = ckpt.build_mode_head().to(device)
    split_tag = f"val ({len(dataset.bulk_ids)} bulks, 128 draws/bulk)"

    row_bulk = dataset.row_bulk_idx.numpy()
    presence = dataset.presence.numpy()
    target = dataset.cond_target.numpy()
    rung = dataset.rung.numpy()
    support_size = presence.sum(-1).astype(np.int64)

    def _predict(support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outs = ([], [], [])
        bs = 8192
        for start in range(0, len(row_bulk), bs):
            h = latents[torch.as_tensor(row_bulk[start:start + bs], device=device)]
            s = torch.as_tensor(support[start:start + bs], device=device)
            for buf, t in zip(outs, cond_head(h, s)):
                buf.append(t.cpu().numpy())
        return tuple(np.concatenate(b) for b in outs)

    log_w, mu, sigma = _predict(presence)
    nll = gmm_nll_per_axis(log_w, mu, sigma, target)
    pit = gmm_pit(log_w, mu, sigma, target)
    coverage = central_coverage(pit, COVERAGE_LEVELS)
    eff = effective_components(log_w)
    below_zero = mass_below_zero(log_w, mu, sigma)

    logits_bulk, _ = mode_head(latents)
    p_rows = torch.sigmoid(logits_bulk).cpu().numpy()[row_bulk]
    rng = np.random.default_rng(exposure_seed)
    sampled = sample_supports(p_rows, rng)
    log_w_s, mu_s, sigma_s = _predict(sampled)
    nll_s = gmm_nll_per_axis(log_w_s, mu_s, sigma_s, target)
    pit_s = gmm_pit(log_w_s, mu_s, sigma_s, target)
    coverage_s = central_coverage(pit_s, COVERAGE_LEVELS)

    plot_cond_pit(pit, output_dir / "cond_pit.png", split_tag)
    plot_cond_coverage(coverage, output_dir / "cond_coverage.png", split_tag)
    plot_cond_components(log_w, mu, eff, output_dir / "cond_components.png", split_tag)
    plot_cond_by_phase(pit, presence, output_dir / "cond_pit_by_phase.png", split_tag)
    plot_nll_by_support_size(
        nll, support_size, output_dir / "cond_nll_by_support_size.png", split_tag
    )

    per_axis = []
    for a, name in enumerate(COND_AXES):
        per_axis.append({
            "axis": name,
            "nll": float(nll[:, a].mean()),
            "nll_sampled_support": float(nll_s[:, a].mean()),
            "exposure_bias_nll_delta": float(nll_s[:, a].mean() - nll[:, a].mean()),
            "coverage_68": coverage["0.68"][a],
            "coverage_68_sampled_support": coverage_s["0.68"][a],
            "pit_tail_lo": float((pit[:, a] < 0.1).mean()),
            "pit_tail_hi": float((pit[:, a] > 0.9).mean()),
            "effective_components_mean": float(eff[:, a].mean()),
            "mass_below_zero_mean": float(below_zero[:, a].mean()),
        })

    metrics = {
        "checkpoint_tag": tag,
        "encoder_checkpoint": str(ckpt.encoder_checkpoint),
        "label_config": dict(lc),
        "n_val_rows": int(len(dataset)),
        "n_val_bulks": int(len(dataset.bulk_ids)),
        "cond_nll": float(nll.sum(-1).mean()),
        "cond_nll_by_rung": stratified_mean(nll.sum(-1), rung),
        "cond_nll_by_support_size": stratified_mean(nll.sum(-1), support_size),
        "coverage": coverage,
        "coverage_sampled_support": coverage_s,
        "exposure_seed": exposure_seed,
        "per_axis": per_axis,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    table = pd.DataFrame(per_axis)
    print(f"\ncond NLL: {metrics['cond_nll']:.4f} nats/row (summed over axes)")
    print(table.round(4).to_string(index=False))
    print(f"metrics -> {output_dir / 'metrics.json'}")
    return metrics
