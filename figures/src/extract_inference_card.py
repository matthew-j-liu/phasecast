"""Dump the inference-card slices for one held-out real rock to CSV.

Fits the canonical tailed recalibrator on a 25-rock draw of whole
stratigraphic units from the train roster, scores the sealed test slice, and
writes three small tables into `figures/data/` so the figure script never
opens `data/` or `checkpoints/`. Run from repo root; only needed when the
production checkpoint, the benchmark or the fit budget changes.

`inference_card_chemistry.csv` is the rendered rock's cation input.
`inference_card_groups.csv` carries both arms of the marginal mode: for the
zero-shot and recalibrated models, each group's presence probability and its
share of the crystalline mass with the absences kept in, so a column sums to
one and is comparable to a measured mode. `inference_card_meta.csv` holds the
identifiers and fit sizes the caption quotes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmineral.data.phase_vocab import GROUP_INDEX, MODEL_TO_GROUP, PHASE_INDEX
from cnmineral.data.tokens import VOCAB
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import GROUPS, coarsen_alpha, load_benchmark, split_mask
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    draw_units,
    fit_recalibrator,
    lift_group_presence_to_phases,
    score_recalibrator,
)
from style import DATADIR, ROOT

CKPT = ROOT / "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt"
BUDGET = 25
DRAW_SEED = 0
# the value selected on val for the tailed tier; see run_curve.WEIGHT_DECAY
WEIGHT_DECAY = 1.0
N_SAMPLES = 2048
INTERVAL_Q = (0.25, 0.75)


def sampled_mode(head, latent, recalibrator, gp_row, p21, corrected):
    """Marginal group shares over assemblages drawn from `p21`.

    Absences are kept as zeros rather than conditioned away, so every sample's
    shares sum to one and the mean over samples is the posterior mean mode.
    Conditioning on inclusion instead -- the corpus card's convention -- gives
    a column that sums past one and cannot be read against a measured mode.
    """
    rng = np.random.default_rng(DRAW_SEED)
    draws = (rng.random((N_SAMPLES, p21.shape[-1])) < p21).astype(np.float64)
    with torch.no_grad():
        alpha21 = head.magnitude_alpha(
            latent.expand(N_SAMPLES, -1),
            torch.as_tensor(draws, dtype=latent.dtype, device=latent.device),
        ).cpu().numpy()
    alpha7, _ = coarsen_alpha(alpha21, draws)

    present = np.zeros((N_SAMPLES, len(GROUPS)))
    for phase, group in MODEL_TO_GROUP.items():
        present[:, GROUP_INDEX[group]] = np.maximum(
            present[:, GROUP_INDEX[group]], draws[:, PHASE_INDEX[phase]])

    if not corrected:
        on = alpha7 * present
    else:
        with torch.no_grad():
            _, corr, log_w = recalibrator(
                gp_row.logits7.expand(N_SAMPLES, -1),
                torch.as_tensor(alpha7, dtype=torch.float64),
                gp_row.sio2.expand(N_SAMPLES),
            )
        # the tailed tier returns a two-component mixture even on
        # single-Dirichlet input; pick a component per sample by its gate
        pick = (rng.random(N_SAMPLES) < np.exp(log_w.numpy())[:, 1]).astype(int)
        on = corr.numpy()[np.arange(N_SAMPLES), pick] * present

    gam = rng.gamma(np.clip(on, 1e-9, None)) * present
    total = gam.sum(-1, keepdims=True)
    shares = np.divide(gam, total, out=np.zeros_like(gam), where=total > 0)
    return shares


def main():
    ckpt = JointHeadCheckpoint.load(CKPT)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder, benchmark, device)
    head = ckpt.build_mode_head().to(device)
    assert head.n_components == 1
    gp = build_group_predictions(head, benchmark, latents)

    scored = benchmark.clean & benchmark.has_magnitude
    train_mask, test_mask = (split_mask(benchmark, s) & scored
                             for s in ("train", "test"))
    train = gp.subset(train_mask)

    drawn = draw_units(train, BUDGET, np.random.default_rng(DRAW_SEED))
    n_rocks, n_units = int(drawn.sum()), len(np.unique(train.unit[drawn]))
    recalibrator = fit_recalibrator(train.subset(drawn), tier="tailed",
                                    weight_decay=WEIGHT_DECAY)

    test = gp.subset(test_mask)
    before, after = score_recalibrator(None, test), score_recalibrator(recalibrator, test)
    print(f"fit on {n_rocks} rocks / {n_units} units, {len(test)} test rocks")
    for term in ("bernoulli", "dirichlet", "joint"):
        print(f"  {term:10s} {before[term].mean():7.3f} -> {after[term].mean():7.3f}")

    # The unrestricted median: representative by construction, and the only
    # rule that needs no clause in the caption. Restricting to rocks with more
    # quantified groups was considered -- the adapted NLL is a density over
    # however many groups a rock spans, so small compositions score
    # degenerately -- but every such pool picks a rock the model fits worse.
    order = np.argsort(after["joint"])
    local = int(order[len(order) // 2])
    row = int(np.flatnonzero(test_mask)[local])
    print(f"median rock {benchmark.bulk_ids[row]}, "
          f"NLL {before['joint'][local]:.2f} -> {after['joint'][local]:.2f}")

    symbol = {z: sym for sym, z in VOCAB}
    src, frac = benchmark.bulk_src[row].numpy(), benchmark.bulk_frac[row].numpy()
    populated = np.flatnonzero(src > 0)
    pd.DataFrame({
        "element": [symbol[int(z)] for z in src[populated]],
        "cat_wtpct": np.round(frac[populated] * 100, 6),
    }).to_csv(DATADIR / "inference_card_chemistry.csv", index=False)

    gp_row = test.subset(np.arange(len(test)) == local)
    with torch.no_grad():
        adj_logits, _, _ = recalibrator(gp_row.logits7, gp_row.alpha7, gp_row.sio2)
        p21_zero = torch.sigmoid(
            head.presence_logits(latents[row:row + 1])).cpu().numpy()
    p21_adj = lift_group_presence_to_phases(
        recalibrator, gp_row.logits7, p21_zero, gp_row.sio2,
    )

    rows = []
    for arm, p21, logits, corrected in (
        ("zero_shot", p21_zero, gp_row.logits7, False),
        ("recalibrated", p21_adj, adj_logits, True),
    ):
        shares = sampled_mode(head, latents[row], recalibrator, gp_row, p21,
                              corrected)
        p7 = torch.sigmoid(logits).numpy()[0]
        lo, hi = np.quantile(shares, INTERVAL_Q, axis=0)
        rows += [{
            "arm": arm,
            "group": g,
            "p_pred": round(float(p7[i]), 6),
            "presence_measured": int(benchmark.presence[row, i]),
            "share_pred": round(float(shares[:, i].mean()), 6),
            "share_lo": round(float(lo[i]), 6),
            "share_hi": round(float(hi[i]), 6),
            "measured_share": round(float(benchmark.magnitude[row, i]), 6)
            if benchmark.numeric[row, i] else np.nan,
            "censor_bound": round(float(benchmark.censor_bound[row, i]), 6)
            if benchmark.censored[row, i] else np.nan,
        } for i, g in enumerate(GROUPS)]
    pd.DataFrame(rows).to_csv(DATADIR / "inference_card_groups.csv", index=False)

    pd.DataFrame([{
        "bulk_id": str(benchmark.bulk_ids[row]),
        "sio2": round(float(benchmark.sio2[row]), 2),
        "n_fit_rocks": n_rocks,
        "n_fit_units": n_units,
        "draw_seed": DRAW_SEED,
        "n_samples": N_SAMPLES,
        "n_test": len(test),
        "n_measured_groups": int(benchmark.numeric[row].sum()),
        "test_joint_zero_shot": round(float(before["joint"].mean()), 3),
        "test_joint_recalibrated": round(float(after["joint"].mean()), 3),
        "test_bernoulli_zero_shot": round(float(before["bernoulli"].mean()), 3),
        "test_bernoulli_recalibrated": round(float(after["bernoulli"].mean()), 3),
        "test_dirichlet_zero_shot": round(float(before["dirichlet"].mean()), 3),
        "test_dirichlet_recalibrated": round(float(after["dirichlet"].mean()), 3),
        "test_opq_rate": round(float(np.asarray(
            benchmark.presence)[test_mask, GROUP_INDEX["opq"]].mean()), 3),
    }]).to_csv(DATADIR / "inference_card_meta.csv", index=False)
    print(f"wrote {DATADIR}/inference_card_{{chemistry,groups,meta}}.csv")


if __name__ == "__main__":
    main()
