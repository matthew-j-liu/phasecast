"""Is the real amp-share distribution bimodal where the corpus says it is?

Chemistry-neighborhoods among clean benchmark rocks (k-NN in the frozen
all-rock encoder latent, per-unit contribution capped), the within-
neighborhood distribution of measured amp share among amp-bearing rocks,
and a 1-vs-2-component GMM BIC test on log share. Paired reference: each
neighborhood center's nearest corpus val bulk, same test on its within-bulk
amp shares across the 128 hidden-(P, H2O) draws at the 50 rung. The real
neighborhoods mix rock-to-rock chemistry variation on top of any path
variation, which biases them TOWARD extra spread and humps -- so a
narrower/unimodal real read is conservative evidence that nature's
conditional lacks the corpus's second hump. SiO2 windows repeat the test
without the encoder in the loop.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from pathlib import Path
from sklearn.mixture import GaussianMixture

from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.data.phase_vocab import GROUPS, PHASE_INDEX
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark
from cnmineral.training.joint_heads import JointHeadCheckpoint

MIX = Path("checkpoints/joint_heads_allrock_mixture_seed0/heads_best.pt")
AMP_G = GROUPS.index("amp")
RUNG = 50
K_NEIGHBORS = 30
UNIT_CAP = 3
MIN_VALUES = 15
STRONG_BIC = 10.0


def gmm_delta_bic(values, rng_seed=0):
    """BIC(1 component) - BIC(2 components) on log10 shares; positive = the
    two-component fit is preferred. Also returns the 2-GMM mean separation."""
    x = np.log10(np.clip(np.asarray(values, dtype=float), 1e-4, 1.0))[:, None]
    if len(x) < MIN_VALUES or np.ptp(x) < 1e-6:
        return np.nan, np.nan
    g1 = GaussianMixture(1, random_state=rng_seed).fit(x)
    g2 = GaussianMixture(2, n_init=3, random_state=rng_seed).fit(x)
    sep = float(np.abs(g2.means_[0, 0] - g2.means_[1, 0]))
    return float(g1.bic(x) - g2.bic(x)), sep


def summarize(name, results):
    d = np.array([r[0] for r in results])
    d = d[np.isfinite(d)]
    if len(d) == 0:
        print(f"{name}: no testable groups")
        return
    print(f"{name}: {len(d)} testable groups; "
          f"2-component preferred {np.mean(d > 0):.1%}, "
          f"strongly (dBIC>{STRONG_BIC:.0f}) {np.mean(d > STRONG_BIC):.1%}, "
          f"median dBIC {np.median(d):+.1f}")


def main():
    ckpt = JointHeadCheckpoint.load(MIX)
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.eval()
    device = torch.device("cpu")

    bm = load_benchmark()
    rock_lat = bulk_latents(encoder, bm, device).numpy()
    amp_ok = bm.clean & bm.numeric[:, AMP_G] & (bm.magnitude[:, AMP_G] > 0)
    idx = np.flatnonzero(amp_ok)
    print(f"clean rocks with numeric nonzero amp: {len(idx)} "
          f"across {len(np.unique(bm.unit[idx]))} units")

    ds = JointHeadDataset(
        corpus_dir=ckpt.label_config["valtest_corpus_dir"],
        rungs=tuple(ckpt.label_config["rungs"]),
        presence_cutoff=ckpt.label_config["presence_cutoff"],
        comp_eps=ckpt.label_config["comp_eps"],
        bulk_ids=load_valtest_split("val"),
    )
    bulk_lat = bulk_latents(encoder, ds, device).numpy()
    row_bulk = ds.row_bulk_idx.numpy()
    rung = ds.rung.numpy()
    amp_col = PHASE_INDEX["amp"]
    mag = ds.magnitude.numpy()
    pres = ds.presence.numpy()

    # real neighborhoods: k-NN among amp-bearing clean rocks, unit-capped
    lat = rock_lat[idx]
    lat = (lat - lat.mean(0)) / lat.std(0).clip(1e-9)
    d2 = ((lat[:, None, :] - lat[None, :, :]) ** 2).sum(-1)
    real_results, real_spreads, matched = [], [], []
    for c in range(len(idx)):
        order = np.argsort(d2[c])[:K_NEIGHBORS * 3]
        chosen, per_unit = [], {}
        for j in order:
            u = bm.unit[idx[j]]
            if per_unit.get(u, 0) >= UNIT_CAP:
                continue
            per_unit[u] = per_unit.get(u, 0) + 1
            chosen.append(j)
            if len(chosen) == K_NEIGHBORS:
                break
        vals = bm.magnitude[idx[np.array(chosen)], AMP_G]
        vals = vals[vals > 0]
        if len(vals) >= MIN_VALUES:
            real_results.append(gmm_delta_bic(vals))
            real_spreads.append(np.std(np.log10(np.clip(vals, 1e-4, 1))))
            matched.append(c)
    summarize("real neighborhoods", real_results)

    # matched corpus bulks: nearest val bulk in the same latent space,
    # within-bulk amp shares across draws at the rung
    blat = (bulk_lat - bulk_lat.mean(0)) / bulk_lat.std(0).clip(1e-9)
    corpus_results, corpus_spreads = [], []
    seen = set()
    for c in matched:
        r = rock_lat[idx[c]]
        r = (r - bulk_lat.mean(0)) / bulk_lat.std(0).clip(1e-9)
        b = int(np.argmin(((blat - r) ** 2).sum(-1)))
        if b in seen:
            continue
        seen.add(b)
        rows = np.flatnonzero((row_bulk == b) & (rung == RUNG)
                              & (pres[:, amp_col] > 0))
        vals = mag[rows, amp_col]
        if len(vals) >= MIN_VALUES:
            corpus_results.append(gmm_delta_bic(vals))
            corpus_spreads.append(np.std(np.log10(np.clip(vals, 1e-4, 1))))
    summarize(f"matched corpus bulks ({len(seen)} distinct)", corpus_results)

    print(f"log10-share spread (std): real neighborhoods median "
          f"{np.median(real_spreads):.3f} vs matched corpus within-bulk "
          f"median {np.median(corpus_spreads):.3f}")

    # model-free check: overlapping SiO2 windows
    sio2 = bm.sio2[idx]
    windows = [(lo, lo + 4.0) for lo in np.arange(45, 76, 2.0)]
    win_results = []
    for lo, hi in windows:
        m = (sio2 >= lo) & (sio2 < hi)
        vals = bm.magnitude[idx[m], AMP_G]
        vals = vals[vals > 0]
        if len(vals) >= MIN_VALUES:
            win_results.append(gmm_delta_bic(vals))
    summarize("SiO2 windows (4 wt%, step 2)", win_results)


if __name__ == "__main__":
    main()
