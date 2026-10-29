"""The paper's cohort card and inference card, drawn on the plutonic benchmark.

Extracts the same tables the paper's `extract_cohort_card.py` and
`extract_inference_card.py` write, from the plutonic train and val rocks (test
stays sealed), then draws them with the paper's own `fig_cohort_card` and
`fig_inference_card`, imported from the gitignored `paper/figures/src/` and
pointed at this exploration's results, so the style is the paper's by
construction.  The extraction follows the current recalibrator API rather than
the paper's: the canonical matrix tier on the paper's 25-rock budget, and the
median val rock rather than a test rock.

    conda run -n mineral python explorations/real_mineralogy/run_plutonic_figures.py \
        --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper" / "figures" / "src"))

import fig_cohort_card
import fig_inference_card
import style

from cnmineral.data.phase_vocab import GROUP_INDEX, MODEL_TO_GROUP, PHASE_INDEX
from cnmineral.data.tokens import VOCAB
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import (
    GROUPS,
    coarsen_alpha,
    predict,
    split_mask,
    unit_bootstrap,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    draw_units,
    fit_recalibrator,
    lift_group_presence_to_phases,
    score_recalibrator,
)
from explorations.real_mineralogy.plutonic import plutonic_benchmark

RESULTS = Path(__file__).resolve().parent / "results"

# the paper's inference card: fit budget, draw seed, marginal-mode samples and
# the share interval it draws
BUDGET = 25
DRAW_SEED = 0
N_SAMPLES = 2048
INTERVAL_Q = (0.25, 0.75)
TIER = "matrix"
# the value selected on val for the matrix tier; see
# explorations/real_rock_fewshot/run_curve.py
WEIGHT_DECAY = 1.0


def cohort_tables(head, bench, latents, out: Path) -> None:
    """`extract_cohort_card.py`'s three tables, on `bench`."""
    pred = predict(head, latents, bench)
    p7 = pred["p7"]
    forced = conditional_mean(pred["alpha7"], bench.numeric.astype(np.float64))

    symbol = {z: sym for sym, z in VOCAB}
    src, frac = bench.bulk_src.numpy(), bench.bulk_frac.numpy()
    pd.DataFrame([
        {"element": symbol[int(z)], "cat_wtpct": round(v * 100, 6)}
        for z in np.unique(src[src > 0])
        for v in frac[src == z]
    ]).to_csv(out / "cohort_card_elements.csv", index=False)

    present = bench.presence > 0

    def auroc(rows):
        result = np.full(len(GROUPS), np.nan)
        for i in range(len(GROUPS)):
            truth = present[rows, i]
            if truth.min() != truth.max():
                result[i] = roc_auc_score(truth, p7[rows, i])
        return result

    q25, q75 = unit_bootstrap(auroc, bench.unit, q=(0.25, 0.75))
    pd.DataFrame({
        "group": GROUPS,
        "auroc": auroc(np.arange(len(bench))).round(4),
        "auroc_q25": q25.round(4),
        "auroc_q75": q75.round(4),
        "n_quantified": bench.numeric.sum(0).astype(int),
    }).to_csv(out / "cohort_card_groups.csv", index=False)

    rows = []
    for i, group in enumerate(GROUPS):
        quantified = bench.numeric[:, i]
        rows += [{"group": group, "kind": "measured", "share": round(v, 6)}
                 for v in bench.magnitude[quantified, i]]
        rows += [{"group": group, "kind": "predicted", "share": round(v, 6)}
                 for v in forced[quantified, i]]
    pd.DataFrame(rows).to_csv(out / "cohort_card_abundance.csv", index=False)


def sampled_mode(head, latent, recalibrator, gp_row, p21, corrected):
    """Marginal group shares over assemblages drawn from `p21`.

    Absences are kept as zeros rather than conditioned away, so every sample's
    shares sum to one and the mean over samples is the posterior mean mode, the
    quantity a measured mode can be read against.
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

    if corrected:
        with torch.no_grad():
            _, alpha7 = recalibrator(
                gp_row.logits7.expand(N_SAMPLES, -1),
                torch.as_tensor(alpha7, dtype=gp_row.alpha7.dtype),
            )
        alpha7 = alpha7.numpy()
    on = alpha7 * present

    gam = rng.gamma(np.clip(on, 1e-9, None)) * present
    total = gam.sum(-1, keepdims=True)
    return np.divide(gam, total, out=np.zeros_like(gam), where=total > 0)


def inference_tables(head, bench, latents, out: Path) -> None:
    """`extract_inference_card.py`'s three tables, for the median val rock."""
    gp = build_group_predictions(head, bench, latents)
    train_mask, val_mask = (split_mask(bench, s) for s in ("train", "val"))
    train = gp.subset(train_mask)

    drawn = draw_units(train, BUDGET, np.random.default_rng(DRAW_SEED))
    n_rocks, n_units = int(drawn.sum()), len(np.unique(train.unit[drawn]))
    recalibrator = fit_recalibrator(train.subset(drawn), tier=TIER,
                                    weight_decay=WEIGHT_DECAY)

    val = gp.subset(val_mask)
    before = score_recalibrator(None, val)
    after = score_recalibrator(recalibrator, val)
    print(f"recalibrator fit on {n_rocks} rocks / {n_units} intrusions; "
          f"val joint nll {before['joint'].mean():.3f} -> "
          f"{after['joint'].mean():.3f} over {len(val)} rocks")

    order = np.argsort(after["joint"])
    local = int(order[len(order) // 2])
    row = int(np.flatnonzero(val_mask)[local])
    print(f"median val rock {bench.bulk_ids[row]}, joint nll "
          f"{before['joint'][local]:.2f} -> {after['joint'][local]:.2f}")

    symbol = {z: sym for sym, z in VOCAB}
    src, frac = bench.bulk_src[row].numpy(), bench.bulk_frac[row].numpy()
    populated = np.flatnonzero(src > 0)
    pd.DataFrame({
        "element": [symbol[int(z)] for z in src[populated]],
        "cat_wtpct": np.round(frac[populated] * 100, 6),
    }).to_csv(out / "inference_card_chemistry.csv", index=False)

    gp_row = val.subset(np.arange(len(val)) == local)
    with torch.no_grad():
        adj_logits, _ = recalibrator(gp_row.logits7, gp_row.alpha7)
        p21_zero = torch.sigmoid(
            head.presence_logits(latents[row:row + 1])).cpu().numpy()
    p21_adj = lift_group_presence_to_phases(recalibrator, gp_row.logits7, p21_zero)

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
            "presence_measured": int(bench.presence[row, i]),
            "share_pred": round(float(shares[:, i].mean()), 6),
            "share_lo": round(float(lo[i]), 6),
            "share_hi": round(float(hi[i]), 6),
            "measured_share": round(float(bench.magnitude[row, i]), 6)
            if bench.numeric[row, i] else np.nan,
            "censor_bound": round(float(bench.censor_bound[row, i]), 6)
            if bench.censored[row, i] else np.nan,
        } for i, g in enumerate(GROUPS)]
    pd.DataFrame(rows).to_csv(out / "inference_card_groups.csv", index=False)

    pd.DataFrame([{
        "bulk_id": str(bench.bulk_ids[row]),
        "unit": str(bench.unit[row]),
        "sio2": round(float(bench.sio2[row]), 2),
        "tier": TIER,
        "n_fit_rocks": n_rocks,
        "n_fit_units": n_units,
        "draw_seed": DRAW_SEED,
        "n_samples": N_SAMPLES,
        "n_val": len(val),
        "n_measured_groups": int(bench.numeric[row].sum()),
        "val_joint_zero_shot": round(float(before["joint"].mean()), 3),
        "val_joint_recalibrated": round(float(after["joint"].mean()), 3),
        "val_bernoulli_zero_shot": round(float(before["bernoulli"].mean()), 3),
        "val_bernoulli_recalibrated": round(float(after["bernoulli"].mean()), 3),
        "val_dirichlet_zero_shot": round(float(before["dirichlet"].mean()), 3),
        "val_dirichlet_recalibrated": round(float(after["dirichlet"].mean()), 3),
    }]).to_csv(out / "inference_card_meta.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    out = RESULTS / args.checkpoint.parent.name
    data = out / "plutonic_figure_data"
    data.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder = encoder.to(device)
    head = ckpt.build_mode_head().to(device)

    with plutonic_benchmark() as benchmark:
        bench = benchmark.subset(
            split_mask(benchmark, "train") | split_mask(benchmark, "val")
        )
        latents = bulk_latents(encoder, bench, device)
        cohort_tables(head, bench, latents, data)
        inference_tables(head, bench, latents, data)
    print(f"wrote {data}")

    style.FIGDIR = out
    fig_cohort_card.DATADIR = data
    fig_inference_card.DATADIR = data
    fig_cohort_card.cohort_card("plutonic_cohort_card")
    fig_inference_card.inference_card("plutonic_inference_card")


if __name__ == "__main__":
    main()
