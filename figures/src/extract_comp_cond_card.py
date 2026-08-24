"""Dump the comp/cond appendix-card slices to CSV.

Runs the composition and conditions heads on one simulated validation
composition (with its per-draw labels as truth) and on the Fig 2 rhyolite
(predictions only), with supports sampled from the presence branch, never
teacher-forced. Writes two tables into `figures/data/` so the figure script
never opens `data/` or `checkpoints/`. Run from repo root; only needed when
the production checkpoint or the rendered rocks change.

`comp_cond_card_comp.csv` carries, per card and panel, the quartile summary
of 2048 Dirichlet draws from the predicted oxide concentration (and of the
simulated labels where the card has truth), on the Fe-recast display oxides.
`comp_cond_card_cond.csv` carries the raw per-assemblage condition samples
and the simulated per-draw condition labels that the histograms bin.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmineral.data.magemin_corpus import COMP_OXIDES
from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.data.phase_vocab import SOLID_PHASES
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import support_alpha_fn
from cnmineral.eval.real_rock import load_benchmark
from cnmineral.eval.summary_cards import sample_assemblages
from cnmineral.models.cond_head import COND_AXES
from cnmineral.training.joint_heads import JointHeadCheckpoint
from style import DATADIR, ROOT

CKPT = ROOT / "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt"
SIM_BULK = 926801
REAL_ID = "dubray_ds764|203378"
RUNG = 50
N_SAMPLES = 512
COMP_DRAWS = 2048
SEED = 0

# phases whose MAGEMin solution models carry structural water
HYDROUS = {"amp", "bi", "mu", "ep", "cd"}

# display recast of the simulator's excess-oxygen component: Fe2O3 = 2 FeO + O
# by moles, so each unit mass of O converts 8.98 of FeO into 9.98 of Fe2O3
FE_O_TO_FE2O3 = 159.688 / 15.999
FE_O_TO_FEO = 2 * 71.844 / 15.999
DISPLAY_OXIDES = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3",
                  "K2O", "Na2O", "TiO2", "Cr2O3", "H2O"]


def recast_fe(vals):
    cols = {o: vals[..., i] for i, o in enumerate(COMP_OXIDES)}
    cols["Fe2O3"] = FE_O_TO_FE2O3 * cols["O"]
    cols["FeO"] = cols["FeO"] - FE_O_TO_FEO * cols["O"]
    return np.stack([np.clip(cols[o], 1e-3, None) for o in DISPLAY_OXIDES],
                    axis=-1)


def predict(latent, mode_head, comp_head, cond_head, rng):
    with torch.no_grad():
        p = torch.sigmoid(mode_head.presence_logits(latent[None]))[0].numpy()
        comp_alpha = comp_head(latent[None])[0].numpy()
    pred_modes = sample_assemblages(
        p, support_alpha_fn(mode_head, latent), N_SAMPLES, rng)
    supports = torch.as_tensor((pred_modes > 0).astype(np.float32))
    with torch.no_grad():
        log_w, mu, sigma = cond_head(latent.expand(len(supports), -1), supports)
    w = np.exp(log_w.numpy())
    mu_n, sigma_n = mu.numpy(), sigma.numpy()
    cond = np.zeros((len(supports), len(COND_AXES)))
    for a in range(len(COND_AXES)):
        for s in range(len(supports)):
            ci = rng.choice(w.shape[-1], p=w[s, a] / w[s, a].sum())
            cond[s, a] = rng.normal(mu_n[s, a, ci], sigma_n[s, a, ci])
    return p, comp_alpha, cond


def pick_phases(p):
    ranked = np.argsort(-p)
    anhydrous = next(int(k) for k in ranked
                     if SOLID_PHASES[k] not in HYDROUS and SOLID_PHASES[k] != "q")
    hydrous = next(int(k) for k in ranked if SOLID_PHASES[k] in HYDROUS)
    return [anhydrous, hydrous]


def quartiles(draws, center):
    mid = center(draws, axis=0)
    lo, hi = np.quantile(draws, [0.25, 0.75], axis=0)
    return mid, lo, hi


def main():
    ckpt = JointHeadCheckpoint.load(CKPT)
    mode_head = ckpt.build_mode_head()
    comp_head = ckpt.build_comp_head()
    cond_head = ckpt.build_cond_head()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.eval()
    device = torch.device("cpu")

    ds = JointHeadDataset(
        corpus_dir=ckpt.label_config["valtest_corpus_dir"],
        rungs=tuple(ckpt.label_config["rungs"]),
        presence_cutoff=ckpt.label_config["presence_cutoff"],
        comp_eps=ckpt.label_config["comp_eps"],
        bulk_ids=load_valtest_split("val"),
    )
    latents = bulk_latents(encoder, ds, device)
    bpos = int(np.flatnonzero(ds.bulk_ids == SIM_BULK)[0])
    rows = np.flatnonzero(
        (ds.row_bulk_idx.numpy() == bpos) & (ds.rung.numpy() == RUNG))

    benchmark = load_benchmark()
    latents_b = bulk_latents(encoder, benchmark, device)
    brow = int(np.flatnonzero(np.asarray(benchmark.bulk_ids) == REAL_ID)[0])

    rng = np.random.default_rng(SEED)
    sim_p, sim_alpha, sim_cond = predict(
        latents[bpos], mode_head, comp_head, cond_head, rng)
    real_p, real_alpha, real_cond = predict(
        latents_b[brow], mode_head, comp_head, cond_head, rng)

    for name, p in (("sim", sim_p), ("real", real_p)):
        top = np.argsort(-p)[:6]
        print(f"{name} top presence:",
              ", ".join(f"{SOLID_PHASES[k]} {p[k]:.2f}" for k in top))
    for name, cond in (("sim", sim_cond), ("real", real_cond)):
        neg = (cond < 0).mean(axis=0)
        print(f"{name} sampled mass below zero per axis:",
              np.round(neg, 3).tolist())

    presence = ds.presence.numpy()
    comp_rows = []
    for card, p, alpha, with_truth in (("simulated", sim_p, sim_alpha, True),
                                       ("real", real_p, real_alpha, False)):
        for r, k in enumerate(pick_phases(p)):
            draws = recast_fe(100 * rng.dirichlet(
                np.clip(alpha[k], 1e-6, None), size=COMP_DRAWS))
            mean, lo, hi = quartiles(draws, np.mean)
            truth = None
            if with_truth:
                on = rows[presence[rows, k] > 0]
                if len(on):
                    truth = quartiles(
                        recast_fe(100 * ds.comp_target[on, k, :].numpy()),
                        np.median)
            comp_rows += [{
                "card": card,
                "panel": "ab"[r],
                "phase": SOLID_PHASES[k],
                "oxide": oxide,
                "pred_mean": round(float(mean[i]), 6),
                "pred_lo": round(float(lo[i]), 6),
                "pred_hi": round(float(hi[i]), 6),
                "truth_med": round(float(truth[0][i]), 6) if truth else np.nan,
                "truth_lo": round(float(truth[1][i]), 6) if truth else np.nan,
                "truth_hi": round(float(truth[2][i]), 6) if truth else np.nan,
            } for i, oxide in enumerate(DISPLAY_OXIDES)]
    pd.DataFrame(comp_rows).to_csv(DATADIR / "comp_cond_card_comp.csv",
                                   index=False)

    cond_truth = ds.cond_target[rows].numpy()
    cond_rows = []
    for card, cond in (("simulated", sim_cond), ("real", real_cond)):
        for a, axis_name in enumerate(COND_AXES):
            cond_rows += [{"card": card, "axis": axis_name,
                           "kind": "predicted", "value": round(float(v), 6)}
                          for v in cond[:, a]]
            if card == "simulated":
                cond_rows += [{"card": card, "axis": axis_name,
                               "kind": "label", "value": round(float(v), 6)}
                              for v in cond_truth[:, a]]
    pd.DataFrame(cond_rows).to_csv(DATADIR / "comp_cond_card_cond.csv",
                                   index=False)
    print(f"wrote {DATADIR}/comp_cond_card_{{comp,cond}}.csv")


if __name__ == "__main__":
    main()
