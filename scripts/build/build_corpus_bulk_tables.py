"""Build the stratified bulk tables for corpus generation (corpus_regen_spec "Scale").

Two disjoint tiers from the volcanic + has-iron core (25 aphanitic QAPF names,
foidite family excluded for `ig` non-convergence, QC-flagged bulks weighted out
of the draw): a val/test precision tier drawn first, then the training tier
from the remainder. Allocation per qap class is proportional to class
abundance with a rare-class floor; each table is shuffled (seeded) so any
prefix is class-representative -- the ~500-bulk pilot is just the first chunks
of the training table (`--max-chunks 5` on the orchestrator).

    conda run -n mineral python scripts/build/build_corpus_bulk_tables.py

Writes data/corpus/train_bulks.csv and data/corpus/valtest_bulks.csv.

`--expand N` instead draws N new train bulks (excluding every tier table
already on disk) under temperature-smoothed rare-class quotas
(`expansion_quotas`, quota ~ pool^alpha) and writes
data/corpus/train_bulks_expansion.csv; the existing tables are left untouched.
Rationale and candidate comparison in explorations/archive/rare_class_quota/.
`--expand-block M` orders the emitted table as two shuffled blocks -- the
M-budget quota selection first, then the increment -- so an orchestrator
stopped mid-run still realizes the quota shape at the achieved size.

`--valtest-expand N` grows the valtest tier by the same quota rule and rewrites
valtest_bulks.csv in place, held roster first then the increment. Because
`expansion_quotas` targets the alpha-shape of the *expanded* tier, the grown
table carries that shape for all its bulks, not just the new ones -- except
where a class is exhausted, whose unmet quota respills onto the classes that
still have bulks left.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cnmineral.data.volcanic_subset import build_subset, expansion_quotas

REPO = Path(__file__).resolve().parents[2]

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


def draw_expansion(df, budget, alpha, rng, have, used, block=None):
    """Draw `budget` new bulks for a tier that already holds `have`.

    `used` is every sample_id spoken for by any tier -- the contamination
    guard, computed once by the caller so both expansion modes share one
    definition. `have` is the expanding tier's own per-class counts, which
    set the quota target.
    """
    rem = df[~df.sample_id.isin(used)]
    pool = df.qap_name.value_counts()
    have = have.reindex(pool.index).fillna(0).astype(int)
    avail = rem.qap_name.value_counts().reindex(pool.index).fillna(0).astype(int)
    quotas = expansion_quotas(pool, have, avail, budget, alpha)
    inner = (expansion_quotas(pool, have, avail, block, alpha)
             if block is not None else None)
    first, rest = [], []
    for qap, k in quotas.items():
        if k == 0:
            continue
        g = rem[rem.qap_name == qap]
        picks = g.iloc[rng.permutation(len(g))[:k]]
        # quotas are monotone in budget up to largest-remainder rounding;
        # min() absorbs the +/-1 wobble
        k1 = k if inner is None else min(int(inner[qap]), k)
        first.append(picks.iloc[:k1])
        rest.append(picks.iloc[k1:])
    blocks = [pd.concat(fr, ignore_index=True) for fr in (first, rest)]
    blocks = [blk.iloc[rng.permutation(len(blk))] for blk in blocks]
    return pd.concat(blocks, ignore_index=True)


def tier_bulks(out_dir):
    """Every tier table already on disk, keyed by stem."""
    names = ["train_bulks", "train_bulks_expansion", "valtest_bulks"]
    return {n: pd.read_csv(out_dir / f"{n}.csv")
            for n in names if (out_dir / f"{n}.csv").exists()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--valtest", type=int, default=500)
    ap.add_argument("--train-floor", type=int, default=30)
    ap.add_argument("--valtest-floor", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--expand", type=int, default=None,
                    help="draw N new train bulks under rare-class quotas "
                         "instead of building fresh tiers")
    ap.add_argument("--valtest-expand", type=int, default=None,
                    help="grow the valtest tier by N bulks under the same "
                         "quotas, rewriting valtest_bulks.csv in place")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--expand-block", type=int, default=None,
                    help="nested-block size: the first rows of the expansion "
                         "realize the quota shape at this smaller budget")
    args = ap.parse_args()
    if args.expand and args.valtest_expand:
        ap.error("--expand and --valtest-expand are separate runs")
    if args.expand_block is not None:
        if not args.expand:
            ap.error("--expand-block requires --expand")
        if args.expand_block >= args.expand:
            ap.error("--expand-block must be smaller than --expand")

    df = build_subset()
    df["qap_name"] = df.qap_name.astype(str)
    df = df[~df.qap_name.isin(FOIDITE_FAMILY)]
    flagged = df.flag_oxidesum | df.flag_supersilicic | df.flag_ultramafic
    df = df[~flagged].copy()
    df["fe_measured"] = df.feo.notna() & df.fe2o3.notna()
    print(f"volcanic core after foidite + QC exclusions: {len(df):,} bulks, "
          f"{df.qap_name.nunique()} classes ({int(flagged.sum()):,} flagged out)")

    rng = np.random.default_rng(args.seed)
    out = REPO / "data" / "corpus"

    if args.expand or args.valtest_expand:
        tiers = tier_bulks(out)
        grow = "train_bulks" if args.expand else "valtest_bulks"
        if grow not in tiers:
            ap.error(f"no {grow}.csv in {out}; build the fresh tiers first")
        used = set().union(*(set(t.sample_id) for t in tiers.values()))

    if args.expand:
        have = tiers["train_bulks"].qap_name.value_counts()
        expansion = draw_expansion(df, args.expand, args.alpha, rng, have, used,
                                   block=args.expand_block)
        path = out / "train_bulks_expansion.csv"
        expansion[KEEP_COLS].to_csv(path, index=False)
        note = (f", nested block {args.expand_block}"
                if args.expand_block is not None else "")
        print(f"\nexpansion (alpha={args.alpha:g}{note}): {len(expansion)} "
              f"bulks -> {path.relative_to(REPO)}")
        print(expansion.qap_name.value_counts().to_string())
        return

    if args.valtest_expand:
        held = tiers["valtest_bulks"]
        have = held.qap_name.value_counts()
        added = draw_expansion(df, args.valtest_expand, args.alpha, rng, have, used)
        # the held roster leads so a generation run cut short still completes
        # the sealed bulks; each block is separately class-representative
        grown = pd.concat([held.iloc[rng.permutation(len(held))][KEEP_COLS],
                           added[KEEP_COLS]], ignore_index=True)
        path = out / "valtest_bulks.csv"
        grown.to_csv(path, index=False)
        print(f"\nvaltest: {len(held)} + {len(added)} = {len(grown)} bulks "
              f"(alpha={args.alpha:g}) -> {path.relative_to(REPO)}")
        tally = pd.DataFrame({"held": have, "added": added.qap_name.value_counts()})
        tally = tally.fillna(0).astype(int)
        tally["total"] = tally.held + tally.added
        print(tally.sort_values("total", ascending=False).to_string())
        return

    valtest = draw_tier(df, args.valtest, args.valtest_floor, rng)
    remainder = df[~df.sample_id.isin(valtest.sample_id)]
    train = draw_tier(remainder, args.train, args.train_floor, rng)

    out.mkdir(parents=True, exist_ok=True)
    for name, tab in [("train", train), ("valtest", valtest)]:
        path = out / f"{name}_bulks.csv"
        tab[KEEP_COLS].to_csv(path, index=False)
        print(f"\n{name}: {len(tab)} bulks -> {path.relative_to(REPO)}")
        print(tab.qap_name.value_counts().to_string())


if __name__ == "__main__":
    main()
