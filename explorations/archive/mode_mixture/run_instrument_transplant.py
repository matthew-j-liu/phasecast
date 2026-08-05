"""Score corpus val rows through the real-rock instrument's coarsening.

Population held fixed (simulator draws, true supports), instrument changed:
the 21-phase teacher-forced dirichlet NLL vs the same rows coarsened onto
the 7 comparison groups exactly as the real-rock eval coarsens them
(on-support concentration sums, off-vocabulary mass dropped, sub-composition
renormalized). Whatever fraction of the mixture-vs-control gap vanishes is
attributable to the instrument, not to sim-to-real shift. Per-group marginal
Beta NLLs isolate amp, which maps to its own group.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from pathlib import Path
from scipy.stats import beta as beta_dist

from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.data.phase_vocab import GROUPS, MODEL_TO_GROUP, PHASE_INDEX, SOLID_PHASES
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import magnitude_mixture_rows
from cnmineral.eval.real_rock import coarsen_alpha, coarsen_support
from cnmineral.losses.mode_loss import mixture_dirichlet_nll
from cnmineral.training.joint_heads import JointHeadCheckpoint

MIX = Path("checkpoints/joint_heads_allrock_mixture_seed0/heads_best.pt")
CTRL = Path("checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt")


def mixture_nll(log_w, alpha, magnitude, mask):
    """Per-row mixture dirichlet NLL with the off-mask filler the real eval
    uses (positive filler where mask is off, excluded by the mask anyway)."""
    m = mask[:, None, :]
    safe = np.where(m > 0, alpha, 1.0)
    return mixture_dirichlet_nll(
        torch.as_tensor(log_w, dtype=torch.float64),
        torch.as_tensor(safe, dtype=torch.float64),
        torch.as_tensor(magnitude, dtype=torch.float64),
        torch.as_tensor(mask, dtype=torch.float64),
        reduction="none",
    ).numpy()


def marginal_beta_nll(log_w, alpha, magnitude, mask, col):
    """Per-row NLL of `col`'s share under the gate-weighted Beta marginal.

    NaN where `col` is off-mask or the mask has a single member (point mass).
    """
    a = alpha[:, :, col]
    a0 = (alpha * mask[:, None, :]).sum(-1)
    b = a0 - a
    valid = (mask[:, col] > 0) & (mask.sum(-1) > 1)
    x = np.clip(magnitude[:, col], 1e-12, 1 - 1e-12)
    dens = beta_dist.pdf(x[:, None], a, b)
    mix = (np.exp(log_w) * dens).sum(-1)
    out = np.full(len(alpha), np.nan)
    out[valid] = -np.log(np.clip(mix[valid], 1e-300, None))
    return out


def main():
    ckpt_mix = JointHeadCheckpoint.load(MIX)
    ckpt_ctrl = JointHeadCheckpoint.load(CTRL)
    encoder, _ = load_encoder_from_checkpoint(ckpt_mix.encoder_checkpoint)
    encoder.eval()
    ds = JointHeadDataset(
        corpus_dir=ckpt_mix.label_config["valtest_corpus_dir"],
        rungs=tuple(ckpt_mix.label_config["rungs"]),
        presence_cutoff=ckpt_mix.label_config["presence_cutoff"],
        comp_eps=ckpt_mix.label_config["comp_eps"],
        bulk_ids=load_valtest_split("val"),
    )
    latents = bulk_latents(encoder, ds, torch.device("cpu"))
    row_bulk = ds.row_bulk_idx.numpy()
    presence = ds.presence.numpy()
    magnitude = ds.magnitude.numpy()

    on_vocab = np.zeros(len(SOLID_PHASES), dtype=bool)
    for ph in MODEL_TO_GROUP:
        on_vocab[PHASE_INDEX[ph]] = True
    on_mass = (magnitude * on_vocab).sum(-1)
    support7 = coarsen_support(presence)
    mag7 = np.zeros((len(magnitude), len(GROUPS)))
    for ph, g in MODEL_TO_GROUP.items():
        mag7[:, GROUPS.index(g)] += magnitude[:, PHASE_INDEX[ph]]
    keep = on_mass > 1e-6
    mag7[keep] /= on_mass[keep, None]
    multi21 = presence.sum(-1) > 1
    multi7 = support7.sum(-1) > 1
    included = keep & multi21
    print(f"rows {len(magnitude)}, included {included.sum()} "
          f"(all-off-vocab or single-phase dropped); "
          f"single-group after coarsening {int((included & ~multi7).sum())} "
          f"(score 0 in group space, kept as the instrument keeps them)")
    print(f"off-vocab truth mass on included rows: "
          f"mean {1 - on_mass[included].mean():.3f}")

    arms = {}
    for name, ckpt in (("mixture", ckpt_mix), ("control", ckpt_ctrl)):
        head = ckpt.build_mode_head()
        alpha_t, log_w_t = magnitude_mixture_rows(head, latents, row_bulk,
                                                  ds.presence)
        alpha, log_w = alpha_t.numpy().astype(np.float64), log_w_t.numpy()
        alpha7, _ = coarsen_alpha(alpha, presence)
        arms[name] = {
            "nll21": mixture_nll(log_w, alpha, magnitude, presence),
            "nll7": mixture_nll(log_w, alpha7, mag7, support7.astype(float)),
            "amp21": marginal_beta_nll(log_w, alpha, magnitude, presence,
                                       PHASE_INDEX["amp"]),
            "amp7": marginal_beta_nll(log_w, alpha7, mag7,
                                      support7.astype(float),
                                      GROUPS.index("amp")),
        }

    inc = included
    g21 = arms["mixture"]["nll21"][inc].mean() - arms["control"]["nll21"][inc].mean()
    g7 = arms["mixture"]["nll7"][inc].mean() - arms["control"]["nll7"][inc].mean()
    print("\ndirichlet NLL, included corpus val rows (teacher-forced, "
          "mixture minus control; negative = mixture better):")
    print(f"  21-phase space: {arms['mixture']['nll21'][inc].mean():.4f} - "
          f"{arms['control']['nll21'][inc].mean():.4f} -> gap {g21:+.4f}")
    print(f"  7-group space:  {arms['mixture']['nll7'][inc].mean():.4f} - "
          f"{arms['control']['nll7'][inc].mean():.4f} -> gap {g7:+.4f}")
    print(f"  gap surviving the instrument: {g7 / g21:.1%}"
          if g21 != 0 else "")

    print("\nper-group marginal Beta NLL gap (mixture minus control), "
          "rows where the phase/group is present:")
    for ph, col21, col7 in (("amp", PHASE_INDEX["amp"], GROUPS.index("amp")),):
        m21 = inc & np.isfinite(arms["mixture"]["amp21"]) \
                  & np.isfinite(arms["control"]["amp21"])
        m7 = inc & np.isfinite(arms["mixture"]["amp7"]) \
                 & np.isfinite(arms["control"]["amp7"])
        d21 = (arms["mixture"]["amp21"][m21]
               - arms["control"]["amp21"][m21]).mean()
        d7 = (arms["mixture"]["amp7"][m7]
              - arms["control"]["amp7"][m7]).mean()
        print(f"  {ph}: 21-phase {d21:+.4f} (n={m21.sum()}) vs "
              f"7-group {d7:+.4f} (n={m7.sum()})")


if __name__ == "__main__":
    main()
