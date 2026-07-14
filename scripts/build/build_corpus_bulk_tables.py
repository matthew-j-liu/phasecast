"""Build the stratified bulk tables for corpus generation (corpus_regen_spec "Scale").

Two disjoint tiers from the volcanic + has-iron core (25 aphanitic QAPF names,
foidite family excluded for `ig` non-convergence, QC-flagged bulks weighted out
of the draw): a val/test precision tier drawn first, then the training tier
from the remainder. Allocation per qap class is proportional to class
abundance with a rare-class floor; each table is shuffled (seeded) so any
prefix is class-representative -- the ~500-bulk pilot is just the first chunks
of the training table (`--max-chunks 5` on the orchestrator).

    conda run -n alignn_fix python scripts/build/build_corpus_bulk_tables.py

Writes data/corpus/train_bulks.csv and data/corpus/valtest_bulks.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from explorations.equilibrium_multimodality.selection import build_subset

FOIDITE_FAMILY = {"foidite", "phonolitic foidite", "tephritic foidite"}
KEEP_COLS = ["sample_id", "qap_name", "series", "sio2", "fe_measured"]


def allocate(counts, total, floor):
    """Per-class quotas: a rare-class floor, then largest-remainder proportional."""
    n = {c: min(int(v), floor) for c, v in counts.items()}
    avail = {c: int(counts[c]) - n[c] for c in counts}
    remaining = total - sum(n.values())
    pool = sum(avail.values())
    if remaining <= 0 or pool <= 0:
        return n
    exact = {c: remaining * avail[c] / pool for c in counts}
    for c in counts:
        n[c] += min(avail[c], int(exact[c]))
        avail[c] = int(counts[c]) - n[c]
    short = total - sum(n.values())
    for c in sorted(counts, key=lambda c: exact[c] % 1, reverse=True):
        if short == 0:
            break
        if avail[c] > 0:
            n[c] += 1
            avail[c] -= 1
            short -= 1
    return n


def draw_tier(pool, total, floor, rng):
    counts = pool.qap_name.value_counts().to_dict()
    quotas = allocate(counts, total, floor)
    picks = []
    for qap, k in quotas.items():
        g = pool[pool.qap_name == qap]
        picks.append(g.iloc[rng.permutation(len(g))[:k]])
    out = pd.concat(picks, ignore_index=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--valtest", type=int, default=500)
    ap.add_argument("--train-floor", type=int, default=30)
    ap.add_argument("--valtest-floor", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()

    df = build_subset()
    df["qap_name"] = df.qap_name.astype(str)
    df = df[~df.qap_name.isin(FOIDITE_FAMILY)]
    flagged = df.flag_oxidesum | df.flag_supersilicic | df.flag_ultramafic
    df = df[~flagged].copy()
    df["fe_measured"] = df.feo.notna() & df.fe2o3.notna()
    print(f"volcanic core after foidite + QC exclusions: {len(df):,} bulks, "
          f"{df.qap_name.nunique()} classes ({int(flagged.sum()):,} flagged out)")

    rng = np.random.default_rng(args.seed)
    valtest = draw_tier(df, args.valtest, args.valtest_floor, rng)
    remainder = df[~df.sample_id.isin(valtest.sample_id)]
    train = draw_tier(remainder, args.train, args.train_floor, rng)

    out = REPO / "data" / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    for name, tab in [("train", train), ("valtest", valtest)]:
        path = out / f"{name}_bulks.csv"
        tab[KEEP_COLS].to_csv(path, index=False)
        print(f"\n{name}: {len(tab)} bulks -> {path.relative_to(REPO)}")
        print(tab.qap_name.value_counts().to_string())


if __name__ == "__main__":
    main()
