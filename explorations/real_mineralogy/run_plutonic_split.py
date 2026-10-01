"""Generate the fixed plutonic train/val/test split, one time.

Assigns every plutonic rock to a subset as a whole intrusion, balanced on the
evaluated rocks by the same `split_balance` the eruption split was selected
on, and writes the tracked `plutonic_split.csv`.  Generated once and never
recomputed, like the eruption split: the test subset stays sealed until final
numbers, and a split that moved between runs would compare nothing.  Refuses
to overwrite an existing split unless --force.

The dealer differs from the eruption split's `unit_grouped_split` in one
respect.  That one walks units largest-first, and here a few intrusions are
large enough (Rader Creek alone is 89 rocks) that they always land in train,
so its candidates barely differ and cannot balance the mineral categories
across subsets.  Dealing in random order lets a large intrusion land in any
subset; the floor on intrusions per subset keeps a handful of them from
deciding what a subset measures.

    conda run -n mineral python explorations/real_mineralogy/run_plutonic_split.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.eval.real_rock import (
    GROUPS,
    SPLIT_FRACTIONS,
    SPLIT_SUBSETS,
    RealRockBenchmark,
    _assign_units,
    _unit_sizes,
    split_balance,
)
from explorations.real_mineralogy.plutonic import PLUTONIC_SPLIT_PATH, load_plutonic

N_CANDIDATES = 20000
MIN_UNITS_PER_SUBSET = 15


def random_order_split(benchmark: RealRockBenchmark, seed: int = 0) -> np.ndarray:
    """The best-balanced of `N_CANDIDATES` random-order deals of whole units."""
    rng = np.random.default_rng(seed)
    units = _unit_sizes(benchmark.unit)
    evaluated = benchmark.clean & benchmark.has_magnitude
    best, best_score = None, np.inf
    for _ in range(N_CANDIDATES):
        order = [units[i] for i in rng.permutation(len(units))]
        assignment = _assign_units(order, SPLIT_FRACTIONS, int(evaluated.sum()), rng)
        subsets = np.array([assignment[u] for u in benchmark.unit])
        score = split_balance(subsets[evaluated], benchmark.presence[evaluated],
                              benchmark.sio2[evaluated])["score"]
        fewest = min(len(set(benchmark.unit[subsets == s])) for s in SPLIT_SUBSETS)
        if score < best_score and fewest >= MIN_UNITS_PER_SUBSET:
            best, best_score = subsets, score
    if best is None:
        raise ValueError(f"no candidate put {MIN_UNITS_PER_SUBSET} units in every subset")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing split")
    args = parser.parse_args()

    if PLUTONIC_SPLIT_PATH.exists() and not args.force:
        sys.exit(f"{PLUTONIC_SPLIT_PATH} exists; the split is fixed. "
                 "Pass --force to regenerate.")

    benchmark = load_plutonic()
    subsets = random_order_split(benchmark)
    source = np.array([b.split("|", 1)[0] for b in benchmark.bulk_ids])
    sample_id = np.array([b.split("|", 1)[1] for b in benchmark.bulk_ids])
    pd.DataFrame({
        "source": source,
        "sample_id": sample_id,
        "unit": benchmark.unit,
        "subset": subsets,
    }).sort_values(["subset", "unit", "sample_id"]).to_csv(
        PLUTONIC_SPLIT_PATH, index=False)
    print(f"wrote {PLUTONIC_SPLIT_PATH} ({len(subsets)} rocks)")

    evaluated = benchmark.clean & benchmark.has_magnitude
    balance = split_balance(subsets[evaluated], benchmark.presence[evaluated],
                            benchmark.sio2[evaluated])
    print(f"balance: rate_dev {balance['rate_dev']:.3f}, "
          f"sio2_dev {balance['sio2_dev']:.3f}")
    print("base rates (evaluated): " + ", ".join(
        f"{g} {r:.2f}" for g, r in zip(GROUPS, benchmark.presence[evaluated].mean(0))))
    for subset in SPLIT_SUBSETS:
        here = subsets == subset
        m = here & evaluated
        rates = ", ".join(f"{g} {r:.2f}"
                          for g, r in zip(GROUPS, benchmark.presence[m].mean(0)))
        print(f"{subset}: {int(m.sum())} evaluated of {int(here.sum())} rocks, "
              f"{len(set(benchmark.unit[here]))} units, "
              f"sio2 {benchmark.sio2[m].min():.1f}-{benchmark.sio2[m].max():.1f}; "
              f"{rates}")


if __name__ == "__main__":
    main()
