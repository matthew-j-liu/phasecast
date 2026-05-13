"""Expand filtered bulks to per-bulk (T, P) samples for MAGEMin.

Input:  data/magemin_inputs.csv  (N bulks, no T/P columns; from 02_filter_complete.py)
Output: data/magemin_inputs_tp.csv  (N × K rows with T_celsius, P_kbar appended)

Each output row is one MAGEMin call. Downstream `run_magemin.jl` is stateless:
it reads T, P, oxides per row and runs minimization once per row.

T/P strategy is toggle-able via --strategy:
  range  (default)  — Putirka 2008 per-class ranges, uniform sampling, K=20 per bulk
  grid              — Stage 1 brute grid (7×4=28 per bulk)
  const:T:P         — single (T_C, P_kbar) per bulk for sensitivity baseline

Run:
    python scripts/02b_sample_tp.py
    python scripts/02b_sample_tp.py --k 30
    python scripts/02b_sample_tp.py --strategy grid
    python scripts/02b_sample_tp.py --strategy const:1175:2.0 --k 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cnmineral.data.tp_priors import (
    ConstantTPStrategy,
    TPGridStrategy,
    TPRangeStrategy,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def make_strategy(spec: str, k: int):
    """Parse `--strategy` CLI value → strategy instance."""
    if spec == "range":
        return TPRangeStrategy(k_per_bulk=k)
    if spec == "grid":
        return TPGridStrategy()
    if spec.startswith("const:"):
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"const strategy needs 'const:T:P', got '{spec}'")
        T_C = float(parts[1])
        P_kbar = float(parts[2])
        return ConstantTPStrategy(T_C=T_C, P_kbar=P_kbar, k_per_bulk=k)
    raise ValueError(f"unknown --strategy '{spec}'; expected 'range', 'grid', or 'const:T:P'")


def expand_with_tp(bulks: pd.DataFrame, strategy, rng: np.random.Generator) -> pd.DataFrame:
    """For each bulk row, sample K (T, P) and emit K output rows."""
    rows = []
    for _, bulk in bulks.iterrows():
        rock_class = (bulk.get("qap_name") or "").strip().lower()
        samples = strategy.sample(rock_class, rng)
        for T_C, P_kbar in samples:
            new = bulk.to_dict()
            new["T_celsius"] = float(T_C)
            new["P_kbar"] = float(P_kbar)
            rows.append(new)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=REPO_ROOT / "data" / "magemin_inputs.csv",
        help="Filtered bulks CSV (output of 02_filter_complete.py)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "data" / "magemin_inputs_tp.csv",
        help="Output CSV with T_celsius + P_kbar columns",
    )
    parser.add_argument(
        "--strategy", default="range",
        help="T/P sampling strategy: 'range' (default, Putirka 2008), 'grid' (Stage 1 brute), or 'const:T:P'",
    )
    parser.add_argument("--k", type=int, default=20,
                        help="Samples per bulk (used by range and const; grid ignores this)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    strategy = make_strategy(args.strategy, args.k)

    print(f"loading {args.input}")
    bulks = pd.read_csv(args.input)
    print(f"  {len(bulks):,} bulks")

    print(f"strategy: {args.strategy}  (k_per_bulk={strategy.k_per_bulk})")
    expanded = expand_with_tp(bulks, strategy, rng)
    print(f"  expanded to {len(expanded):,} rows (N_bulks × k_per_bulk)")

    print("T/P range in output:")
    print(f"  T: {expanded['T_celsius'].min():.0f} – {expanded['T_celsius'].max():.0f} °C")
    print(f"  P: {expanded['P_kbar'].min():.1f} – {expanded['P_kbar'].max():.1f} kbar")

    print("\nT range per class (top 10 classes):")
    summary = (
        expanded.groupby("qap_name")
        .agg(n=("T_celsius", "size"),
             T_mean=("T_celsius", "mean"),
             T_std=("T_celsius", "std"),
             P_mean=("P_kbar", "mean"),
             P_std=("P_kbar", "std"))
        .sort_values("n", ascending=False)
        .head(10)
    )
    print(summary.to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
