"""Dump the cohort-card slices of the real-rock benchmark to CSV.

Runs the production checkpoint zero-shot over the eruption-v2 benchmark and
writes three small tables into `figures/data/` so the figure script never
opens `data/` or `checkpoints/`. Run from repo root; only needed when the
production checkpoint or the benchmark changes.

`cohort_card_elements.csv` is the input-chemistry panel (cation wt% per rock,
one column per element). `cohort_card_groups.csv` carries per-group presence
AUROC, its interquartile band under a unit-clustered bootstrap over the rocks,
and quantified counts. `cohort_card_abundance.csv` is the teacher-forced
abundance panel: measured shares and predicted conditional-mean shares over
the quantified entries.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmineral.data.tokens import VOCAB
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import (
    GROUPS, load_benchmark, predict, unit_bootstrap,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from style import DATADIR, ROOT

CKPT = ROOT / "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt"


def main():
    ckpt = JointHeadCheckpoint.load(CKPT)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder, benchmark, device)
    head = ckpt.build_mode_head().to(device)
    assert head.n_components == 1
    pred = predict(head, latents, benchmark)
    forced = conditional_mean(pred["alpha7"], benchmark.numeric.astype(np.float64))

    # the card slice; tab:percat computes on benchmark.clean (650), a known
    # 10-rock mismatch the owner has parked
    clean = benchmark.clean & benchmark.has_magnitude
    bench = benchmark.subset(clean)
    p7, forced = pred["p7"][clean], forced[clean]
    print(f"{ckpt.encoder_checkpoint}: {len(bench)} rocks "
          f"(clean alone {int(benchmark.clean.sum())})")

    # long format: rocks do not all populate the same elements
    symbol = {z: sym for sym, z in VOCAB}
    src, frac = bench.bulk_src.numpy(), bench.bulk_frac.numpy()
    pd.DataFrame([
        {"element": symbol[int(z)], "cat_wtpct": round(v * 100, 6)}
        for z in np.unique(src[src > 0])
        for v in frac[src == z]
    ]).to_csv(DATADIR / "cohort_card_elements.csv", index=False)

    present = bench.presence > 0

    def auroc(rows):
        out = np.full(len(GROUPS), np.nan)
        for i in range(len(GROUPS)):
            truth = present[rows, i]
            if truth.min() != truth.max():
                out[i] = roc_auc_score(truth, p7[rows, i])
        return out

    # rows index into `bench`, so the clustering array must be its own units
    q25, q75 = unit_bootstrap(auroc, bench.unit, q=(0.25, 0.75))
    pd.DataFrame({
        "group": GROUPS,
        "auroc": auroc(np.arange(len(bench))).round(4),
        "auroc_q25": q25.round(4),
        "auroc_q75": q75.round(4),
        "n_quantified": bench.numeric.sum(0).astype(int),
    }).to_csv(DATADIR / "cohort_card_groups.csv", index=False)

    rows = []
    for i, group in enumerate(GROUPS):
        tm = bench.numeric[:, i]
        rows += [{"group": group, "kind": "measured", "share": round(v, 6)}
                 for v in bench.magnitude[tm, i]]
        rows += [{"group": group, "kind": "predicted", "share": round(v, 6)}
                 for v in forced[tm, i]]
    pd.DataFrame(rows).to_csv(DATADIR / "cohort_card_abundance.csv",
                              index=False)
    print(f"wrote {DATADIR}/cohort_card_{{elements,groups,abundance}}.csv")


if __name__ == "__main__":
    main()
