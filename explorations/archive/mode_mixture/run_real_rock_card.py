"""One real rock, full posterior, K=2 vs the K=1 control on the same axes.

The single-rock card in the corpus inference-card style, drawn for a real
benchmark rock zero-shot: input chemistry, presence in the 7-group space,
and the share panel with the mixture's gate-weighted Beta marginals drawn in
full (components dashed, mixture solid) against the K=1 control's single
Beta (dotted) and the measured share.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_dist

from cnmineral.data.tokens import VOCAB
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import (
    CARD_SHARE_FLOOR,
    GROUPS,
    LEGEND_FS,
    SUPTITLE_FS,
    TICK_FS,
    TITLE_FS,
    XLABEL_FS,
    load_benchmark,
    split_mask,
)
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    score_recalibrator,
)

OUT = Path(__file__).resolve().parent / "results"
MIX = Path("checkpoints/joint_heads_allrock_mixture_seed0/heads_best.pt")
CTRL = Path("checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt")

PRED, TRUE = "tab:orange", "tab:blue"
COMP_COLORS = ("tab:purple", "tab:green")
FLOOR_LOG = np.log10(CARD_SHARE_FLOOR)


def log_beta_density(grid_log, a, b):
    x = 10.0 ** grid_log
    return np.nan_to_num(beta_dist.pdf(x, a, b) * x * np.log(10))


def main():
    device = torch.device("cpu")
    bm = load_benchmark()
    ckpt_mix = JointHeadCheckpoint.load(MIX)
    ckpt_ctrl = JointHeadCheckpoint.load(CTRL)
    encoder, _ = load_encoder_from_checkpoint(ckpt_mix.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), bm, device)
    gp_mix = build_group_predictions(ckpt_mix.build_mode_head().to(device),
                                     bm, latents)
    gp_ctrl = build_group_predictions(ckpt_ctrl.build_mode_head().to(device),
                                      bm, latents)

    scoreable = bm.clean & bm.has_magnitude & split_mask(bm, "val")
    joint_mix = score_recalibrator(None, gp_mix)["joint"]
    joint_ctrl = score_recalibrator(None, gp_ctrl)["joint"]
    cand = np.flatnonzero(scoreable)
    row = int(cand[np.argmin(joint_mix[cand])])
    key = bm.bulk_ids[row]
    print(f"picked {key}: zero-shot joint mix {joint_mix[row]:.2f} "
          f"vs control {joint_ctrl[row]:.2f} (best-of-val under the mixture)")

    numeric = gp_mix.numeric.numpy().astype(bool)
    mag = gp_mix.magnitude.numpy()
    presence = gp_mix.presence.numpy()
    p7_mix = torch.sigmoid(gp_mix.logits7).numpy()
    a_mix = gp_mix.alpha7.numpy()[row]
    w = np.exp(gp_mix.log_w.numpy()[row])
    a_ctrl = gp_ctrl.alpha7.numpy()[row]

    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.0, 0.9, 1.4]}
    )

    symbol = {z: sym for sym, z in VOCAB}
    src, frac = bm.bulk_src[row].numpy(), bm.bulk_frac[row].numpy()
    populated = np.flatnonzero(src > 0)
    ax0.barh(np.arange(len(populated)), frac[populated] * 100,
             color="tab:gray", alpha=0.8, height=0.6)
    ax0.set_yticks(np.arange(len(populated)),
                   [symbol[int(z)] for z in src[populated]], fontsize=TICK_FS)
    ax0.invert_yaxis()
    ax0.set_xscale("log")
    ax0.set_xlabel("cation wt% (log)", fontsize=XLABEL_FS)
    ax0.set_title("Input: Whole-Rock Chemistry", fontsize=TITLE_FS)

    y = np.arange(len(GROUPS))
    ax1.plot(p7_mix[row], y - 0.15, "o", ms=5, color=PRED, mfc="white")
    ax1.plot(presence[row], y + 0.15, "o", ms=5, color=TRUE)
    ax1.set_yticks(y, GROUPS, fontsize=TICK_FS)
    ax1.invert_yaxis()
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_xticks([0, 0.5, 1])
    ax1.grid(axis="x", lw=0.3, alpha=0.5)
    ax1.set_xlabel("presence probability", fontsize=XLABEL_FS)
    ax1.set_title("Phase Presence", fontsize=TITLE_FS)
    ax1.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=5, color=PRED, mfc="white",
               label="predicted"),
        Line2D([], [], marker="o", ls="none", ms=5, color=TRUE,
               label="measured"),
    ], fontsize=LEGEND_FS - 2, loc="center right", framealpha=0.9)

    grid = np.linspace(FLOOR_LOG, 0.0, 500)
    a0_mix = (a_mix * numeric[row]).sum(-1)
    a0_ctrl = (a_ctrl * numeric[row]).sum(-1)
    for i in np.flatnonzero(numeric[row]):
        mix = np.zeros_like(grid)
        for c in range(a_mix.shape[0]):
            dens = log_beta_density(grid, a_mix[c, i], a0_mix[c] - a_mix[c, i])
            mix += w[c] * dens
            dc = dens / max(dens.max(), 1e-12) * 0.36 * w[c]
            ax2.plot(grid, i - dc, color=COMP_COLORS[c], lw=1.0, ls="--",
                     alpha=0.9)
        if mix.max() > 0:
            dm = mix / mix.max() * 0.40
            ax2.fill_between(grid, i, i - dm, color=PRED, alpha=0.40, lw=0)
        ctrl = log_beta_density(grid, a_ctrl[i], a0_ctrl - a_ctrl[i])
        if ctrl.max() > 0:
            dc = ctrl / ctrl.max() * 0.40
            ax2.plot(grid, i - dc, color="0.35", lw=1.2, ls=":")
        ax2.plot(np.log10(max(mag[row, i], CARD_SHARE_FLOOR)), i + 0.15, "o",
                 ms=5, color=TRUE)
    ax2.set_yticks(y, GROUPS, fontsize=TICK_FS)
    ax2.set_ylim(len(GROUPS) - 0.4, -0.8)
    ax2.set_xlim(FLOOR_LOG, 0.05)
    ticks = [0.001, 0.01, 0.1, 0.5]
    ax2.set_xticks(np.log10(ticks), ["0.1%", "1%", "10%", "50%"],
                   fontsize=TICK_FS)
    ax2.grid(axis="x", lw=0.3, alpha=0.5)
    ax2.set_xlabel("share of the crystal cargo (log)", fontsize=XLABEL_FS)
    ax2.set_title("Phase Share", fontsize=TITLE_FS)
    ax2.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=PRED, alpha=0.40,
                      label="K=2 mixture"),
        Line2D([], [], color=COMP_COLORS[0], lw=1.0, ls="--",
               label=f"component A (w={w[0]:.2f})"),
        Line2D([], [], color=COMP_COLORS[1], lw=1.0, ls="--",
               label=f"component B (w={w[1]:.2f})"),
        Line2D([], [], color="0.35", lw=1.2, ls=":", label="K=1 control"),
        Line2D([], [], marker="o", ls="none", ms=5, color=TRUE,
               label="measured"),
    ], fontsize=LEGEND_FS - 3, loc="upper left", framealpha=0.9)

    fig.suptitle(
        f"{key}, SiO2 {bm.sio2[row]:.1f} wt%, zero-shot: per-rock joint NLL "
        f"{joint_mix[row]:.2f} (K=2) vs {joint_ctrl[row]:.2f} (K=1 control)",
        fontsize=SUPTITLE_FS)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "real_rock_card_k2.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
