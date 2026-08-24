"""Dump the amphibole-vs-sampled-water slices of the label corpus to CSV.

Reads `data/corpus/valtest` (production corpus, val roster only, one rung) and
writes two small tables into `figures/data/` so the figure script never touches
the corpus. Run from repo root; only needed when the corpus changes.

`amp_water_bins.csv` is what the figure plots. `amp_water_rates.csv` is
per-composition and carries the caption's population statistic: how often
water alone decides whether a composition crystallizes amphibole.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.data.phase_vocab import PHASE_INDEX
from cnmineral.models.cond_head import COND_AXES
from style import DATADIR, ROOT

RUNG = 50
H2O_EDGES = np.arange(0.0, 8.5, 0.5)


def main():
    ds = JointHeadDataset(
        corpus_dir=ROOT / "data/corpus/valtest",
        rungs=(50, 30),
        bulk_ids=load_valtest_split("val"),
    )
    amp = PHASE_INDEX["amp"]
    rows = ds.rung.numpy() == RUNG
    bulk = ds.row_bulk_idx.numpy()[rows]
    share = ds.magnitude.numpy()[rows, amp]
    present = ds.presence.numpy()[rows, amp] > 0
    h2o = ds.cond_target.numpy()[rows, COND_AXES.index("H2O")]
    print(f"{len(bulk)} draws over {len(np.unique(bulk))} bulks at rung {RUNG}")

    idx = np.clip(np.digitize(h2o, H2O_EDGES) - 1, 0, len(H2O_EDGES) - 2)
    bins = []
    for i in range(len(H2O_EDGES) - 1):
        m = idx == i
        if not m.any():
            continue
        pres = present[m]
        bins.append({
            "h2o_lo": H2O_EDGES[i],
            "h2o_hi": H2O_EDGES[i + 1],
            "n_draws": int(m.sum()),
            "presence_rate": round(float(pres.mean()), 4),
            "median_share_present": round(
                float(np.median(share[m][pres])) if pres.any() else np.nan, 4
            ),
        })
    pd.DataFrame(bins).to_csv(DATADIR / "amp_water_bins.csv", index=False)

    rates = []
    for b in np.unique(bulk):
        m = bulk == b
        rates.append({
            "bulk_id": int(ds.bulk_ids[b]),
            "qap": ds.bulk_qap_names[b],
            "n_draws": int(m.sum()),
            "presence_rate": round(float(present[m].mean()), 4),
        })
    rate_df = pd.DataFrame(rates)
    rate_df.to_csv(DATADIR / "amp_water_rates.csv", index=False)
    undecided = rate_df.presence_rate.between(0.05, 0.95).mean()
    print(f"water alone decides amphibole for {undecided:.1%} of compositions")


if __name__ == "__main__":
    main()
