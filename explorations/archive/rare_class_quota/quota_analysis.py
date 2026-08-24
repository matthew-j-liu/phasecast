"""Rare-class quota analysis behind the regen's +20k train expansion (#45).

Fits the class-abundance power law and compares candidate allocations of the
new-bulk budget: temperature-smoothed sampling (quota ~ pool^alpha) at the
literature-default alpha=0.5 and the scaling-law-derived alpha=0.82, plus a
take-all-tail hybrid. Writes the figure and per-class table the locked
alpha=0.5 decision rests on. The production draw is
`scripts/build/build_corpus_bulk_tables.py --expand`.

    conda run -n mineral python explorations/rare_class_quota/quota_analysis.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnmineral.data.volcanic_subset import build_subset, expansion_quotas

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"

FOIDITE_FAMILY = {"foidite", "phonolitic foidite", "tephritic foidite"}
BUDGET = 20000
# alpha = 1/(1+b) with the #27 corpus-scaling exponent b = 0.225: the
# marginal-utility optimum under deployment-frequency class weighting
ALPHA_MICRO = 0.82
HYBRID_TAIL_CUTOFF = 500


def eligible_pool():
    df = build_subset()
    df["qap_name"] = df.qap_name.astype(str)
    df = df[~df.qap_name.isin(FOIDITE_FAMILY)]
    flagged = df.flag_oxidesum | df.flag_supersilicic | df.flag_ultramafic
    return df[~flagged]


def powerlaw_figure(counts, out_path):
    rank = np.arange(1, len(counts) + 1)
    n = counts.to_numpy(dtype=float)
    logr, logn = np.log10(rank), np.log10(n)
    slope, intercept = np.polyfit(logr, logn, 1)
    pred = slope * logr + intercept
    r2 = 1 - ((logn - pred) ** 2).sum() / ((logn - logn.mean()) ** 2).sum()

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.loglog(rank, n, "o", color="tab:blue")
    ax.loglog(rank, 10 ** pred, "-", color="tab:red", lw=1,
              label=f"n = {10**intercept:.0f} * rank^{slope:.2f}  (R2 {r2:.3f})")
    for i in (0, 1, 2, 3, len(counts) - 1):
        ax.annotate(counts.index[i], (rank[i], n[i]), fontsize=6,
                    textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel("class rank")
    ax.set_ylabel("bulks in eligible pool")
    ax.set_title("qap class abundance, rank-frequency", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return slope, r2


def hybrid_quotas(pool, have, avail, budget):
    """Take-all on availability-bound classes, alpha=0.82 on the rest."""
    new = avail * 0
    tail = avail.index[avail <= HYBRID_TAIL_CUTOFF]
    new[tail] = avail[tail]
    head = avail.index[avail > HYBRID_TAIL_CUTOFF]
    new[head] = expansion_quotas(
        pool[head], have[head], avail[head], budget - int(new.sum()), ALPHA_MICRO
    )
    return new


def main():
    RESULTS.mkdir(exist_ok=True)
    df = eligible_pool()
    train = pd.read_csv(REPO / "data" / "corpus" / "train_bulks.csv")
    valtest = pd.read_csv(REPO / "data" / "corpus" / "valtest_bulks.csv")
    used = set(train.sample_id) | set(valtest.sample_id)
    rem = df[~df.sample_id.isin(used)]

    pool = df.qap_name.value_counts()
    have = train.qap_name.value_counts().reindex(pool.index).fillna(0).astype(int)
    avail = rem.qap_name.value_counts().reindex(pool.index).fillna(0).astype(int)
    fe_pair = df.feo.notna() & df.fe2o3.notna()
    fe_share = fe_pair.groupby(df.qap_name).mean().reindex(pool.index)

    slope, r2 = powerlaw_figure(pool, RESULTS / "class_abundance_powerlaw.png")
    print(f"pool: {len(df):,} bulks, {len(pool)} classes; "
          f"rank-frequency slope {slope:.2f}, R2 {r2:.3f}")

    out = pd.DataFrame({
        "pool": pool, "in_train_10k": have, "available": avail,
        "fe_pair_share": fe_share.round(3),
        "new_alpha0.82": expansion_quotas(pool, have, avail, BUDGET, ALPHA_MICRO),
        "new_alpha0.5": expansion_quotas(pool, have, avail, BUDGET, 0.5),
        "new_hybrid": hybrid_quotas(pool, have, avail, BUDGET),
    })
    out["final_alpha0.5"] = out.in_train_10k + out["new_alpha0.5"]
    out.to_csv(RESULTS / "allocation_candidates.csv")
    print(out.to_string())

    for col in ("new_alpha0.82", "new_alpha0.5", "new_hybrid"):
        tail = avail.index[avail <= HYBRID_TAIL_CUTOFF]
        exhausted = int((out.loc[tail, col] == avail[tail]).sum())
        print(f"{col}: big4 {int(out[col].iloc[:4].sum()):,}, "
              f"tail(avail<={HYBRID_TAIL_CUTOFF}) {int(out.loc[tail, col].sum()):,} "
              f"({exhausted}/{len(tail)} exhausted), "
              f"smallest final {int((have + out[col]).min())}")


if __name__ == "__main__":
    main()
