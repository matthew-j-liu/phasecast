"""One global gate-logit offset: can re-pointing the gate close the gap?

The mixture's recalibrated val deficit against the K=1 control sits entirely
in the magnitude term, and the real amp population is single-humped where
the corpus is two-humped. A single scalar added to component B's gate logit
(then renormalized) is the smallest possible fix: it can shift the
population-level component rates but cannot commit per rock. Grid over the
offset, matrix recalibrator refit on the train roster at every point,
selection on train NLL, and the pre-registered paired val contrast against
the recalibrated control at the selected offset. Large negative/positive
offsets collapse the mixture onto one component, so the grid ends contain
"K=1 via component A/B" as special cases.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, split_mask
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    GroupPredictions,
    build_group_predictions,
    fit_recalibrator,
    paired_bootstrap,
    score_recalibrator,
)

MIX = Path("checkpoints/joint_heads_allrock_mixture_seed0/heads_best.pt")
CTRL = Path("checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt")
WEIGHT_DECAY = 1.0
DELTAS = np.linspace(-6.0, 6.0, 25)


def shifted(gp, delta):
    lw = F.log_softmax(
        gp.log_w + torch.tensor([0.0, delta], dtype=gp.log_w.dtype), dim=-1)
    return GroupPredictions(
        logits7=gp.logits7, alpha7=gp.alpha7, sio2=gp.sio2,
        presence=gp.presence, magnitude=gp.magnitude, numeric=gp.numeric,
        unit=gp.unit, log_w=lw,
    )


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

    scored = bm.clean & bm.has_magnitude
    train = split_mask(bm, "train") & scored
    val = split_mask(bm, "val") & scored

    rec_c = fit_recalibrator(gp_ctrl.subset(train), tier="matrix",
                             weight_decay=WEIGHT_DECAY)
    ctrl_val = score_recalibrator(rec_c, gp_ctrl.subset(val))["joint"]
    print(f"control adapted val joint: {ctrl_val.mean():.3f}")

    best = None
    for delta in DELTAS:
        gp_d = shifted(gp_mix, float(delta))
        rec = fit_recalibrator(gp_d.subset(train), tier="matrix",
                               weight_decay=WEIGHT_DECAY)
        tr = score_recalibrator(rec, gp_d.subset(train))["joint"].mean()
        va = score_recalibrator(rec, gp_d.subset(val))["joint"].mean()
        w_b = float(torch.exp(
            F.log_softmax(gp_mix.log_w
                          + torch.tensor([0.0, float(delta)]), dim=-1)
        )[:, 1].mean())
        marker = ""
        if best is None or tr < best["train"]:
            best = {"delta": float(delta), "train": tr, "gp": gp_d, "rec": rec}
            marker = " <- train best so far"
        print(f"delta {delta:+5.2f}  mean w_B {w_b:.3f}  train {tr:.3f}  "
              f"val {va:.3f}{marker}")

    gp_b = best["gp"].subset(val)
    mix_val = score_recalibrator(best["rec"], gp_b)["joint"]
    boot = paired_bootstrap(ctrl_val, mix_val,
                            bm.unit[val])
    print(f"\nselected on train: delta {best['delta']:+.2f}")
    print(f"val adapted joint at selected delta: {mix_val.mean():.3f} "
          f"(was 1.642 at delta 0)")
    print(f"paired val contrast, control minus gate-shifted mixture: "
          f"{boot['mean']:+.3f} [{boot['lo']:+.3f}, {boot['hi']:+.3f}]")


if __name__ == "__main__":
    main()
