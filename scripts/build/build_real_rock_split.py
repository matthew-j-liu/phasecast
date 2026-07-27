"""Generate the fixed real-rock train/val/test roster, one time.

Assigns every benchmark rock to a subset as a whole stratigraphic unit, so no
eruption straddles a boundary, and writes
src/cnmineral/data/real_rock_unit_split.csv (committed; read at runtime by
`cnmineral.eval.real_rock.load_real_rock_split`). Generated once and versioned
rather than recomputed at load time: the carve-out is the board where
adaptation strategies get compared, and a board that moves between runs
compares nothing. Refuses to overwrite an existing roster unless --force.

    conda run -n alignn_fix python scripts/build/build_real_rock_split.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cnmineral.eval.real_rock import (
    GROUPS,
    SPLIT_FRACTIONS,
    SPLIT_PATH,
    SPLIT_SUBSETS,
    load_benchmark,
    split_balance,
    unit_grouped_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing roster")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=400)
    args = parser.parse_args()

    if SPLIT_PATH.exists() and not args.force:
        sys.exit(f"{SPLIT_PATH} exists; the carve-out is fixed. "
                 f"Pass --force to regenerate.")

    benchmark = load_benchmark()
    subsets = unit_grouped_split(
        benchmark, seed=args.seed, fractions=SPLIT_FRACTIONS,
        n_candidates=args.candidates,
    )
    scored = benchmark.clean & benchmark.has_magnitude

    source = np.array([b.split("|", 1)[0] for b in benchmark.bulk_ids])
    sample_id = np.array([b.split("|", 1)[1] for b in benchmark.bulk_ids])
    roster = pd.DataFrame({
        "source": source,
        "sample_id": sample_id,
        "unit": benchmark.unit,
        "subset": subsets,
    }).sort_values(["subset", "unit", "sample_id"])
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    roster.to_csv(SPLIT_PATH, index=False)
    print(f"wrote {SPLIT_PATH} ({len(roster)} rocks)")

    balance = split_balance(
        subsets[scored], benchmark.presence[scored], benchmark.sio2[scored]
    )
    print(f"\nbalance: rate_dev {balance['rate_dev']:.3f}  "
          f"sio2_dev {balance['sio2_dev']:.3f}")
    print(f"base rates (all): "
          + "  ".join(f"{g} {r:.2f}" for g, r
                      in zip(GROUPS, benchmark.presence[scored].mean(0))))
    for subset in SPLIT_SUBSETS:
        m = (subsets == subset) & scored
        rates = "  ".join(f"{g} {r:.2f}" for g, r
                          in zip(GROUPS, benchmark.presence[m].mean(0)))
        print(f"{subset:5s} n={int(m.sum()):3d} "
              f"units={len(set(benchmark.unit[subsets == subset])):3d}  "
              f"SiO2 {benchmark.sio2[m].min():.1f}-{benchmark.sio2[m].max():.1f}  "
              f"{rates}")


if __name__ == "__main__":
    main()
