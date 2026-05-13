"""Recover most rows from MAGEMin chunks that segfaulted deterministically.

When MAGEMin's libMAGEMin.dylib crashes mid-chunk (SIGSEGV in reset_SS, for
example), the whole 5000-row chunk's output is lost. Crashes are deterministic
on specific (composition, T, P) rows. Subdividing into smaller sub-chunks
isolates the bad row(s) and recovers ~90% of the rows we'd otherwise drop.

Approach:
  1. Identify input_chunk_NN.csv files in data/magemin_cache/ that lack a
     matching result_chunk_NN.csv.
  2. For each such chunk: split into sub-chunks of --subchunk-size rows.
  3. Run Julia on each sub-chunk independently. Some will segfault again
     (the ones containing the problematic rows); most won't.
  4. Concatenate successful sub-chunk results into the expected
     result_chunk_NN.csv so the orchestrator's final merge can proceed.

After this script, re-run scripts/orchestrate_magemin.py (no --clean) — it
will see all result_chunk files present and do the merge into
magemin_results.csv.

Run:
    python scripts/retry_failed_chunks.py [--subchunk-size 500]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import time

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
JULIA_DIR = REPO_ROOT / "scripts" / "julia"
JULIA_SCRIPT = JULIA_DIR / "run_magemin.jl"
CACHE_DIR_DEFAULT = REPO_ROOT / "data" / "magemin_cache"


def find_failed_chunks(cache_dir: Path) -> list[int]:
    """Return chunk indices whose input CSV exists but result CSV doesn't."""
    inputs = {
        int(p.stem.split("_")[-1]): p
        for p in cache_dir.glob("input_chunk_*.csv")
    }
    results = {
        int(p.stem.split("_")[-1])
        for p in cache_dir.glob("result_chunk_*.csv")
    }
    return sorted(set(inputs) - results)


def run_julia(input_csv: Path, output_csv: Path, threads: str = "auto") -> int:
    """Spawn julia for one sub-chunk. Returns return code."""
    cmd = [
        "julia",
        f"--project={JULIA_DIR}",
        f"-t{threads}",
        str(JULIA_SCRIPT),
        "--input", str(input_csv),
        "--output", str(output_csv),
        "--database", "ig",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode


def retry_one_chunk(
    chunk_idx: int,
    cache_dir: Path,
    subchunk_size: int,
) -> tuple[int, int, int]:
    """Subdivide and re-run one failed chunk.

    Returns (n_rows_recovered, n_rows_lost, n_subchunks_failed).
    """
    input_csv = cache_dir / f"input_chunk_{chunk_idx:05d}.csv"
    sub_dir = cache_dir / "subchunks" / f"{chunk_idx:05d}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    n_rows = len(df)
    n_sub = (n_rows + subchunk_size - 1) // subchunk_size

    print(f"  chunk {chunk_idx}: {n_rows} rows → {n_sub} sub-chunks of {subchunk_size}")
    sub_results: list[pd.DataFrame] = []
    n_sub_failed = 0

    for s in range(n_sub):
        start = s * subchunk_size
        end = min(start + subchunk_size, n_rows)
        sub_in = sub_dir / f"sub_input_{s:03d}.csv"
        sub_out = sub_dir / f"sub_result_{s:03d}.csv"

        if sub_out.exists():
            sub_results.append(pd.read_csv(sub_out))
            print(f"    sub {s:3d}: cached ({end - start} rows)")
            continue

        df.iloc[start:end].to_csv(sub_in, index=False)
        t0 = time()
        rc = run_julia(sub_in, sub_out)
        elapsed = time() - t0

        if rc != 0:
            n_sub_failed += 1
            print(f"    sub {s:3d}: FAILED (rc={rc}) in {elapsed:.1f}s — {end - start} rows lost")
        else:
            sub_df = pd.read_csv(sub_out)
            sub_results.append(sub_df)
            print(f"    sub {s:3d}: ok ({end - start} rows) in {elapsed:.1f}s")

    # Concatenate successful sub-results into the canonical result_chunk_NN.csv
    if not sub_results:
        print(f"  chunk {chunk_idx}: NO sub-chunks succeeded; result file not written")
        return 0, n_rows, n_sub_failed

    merged = pd.concat(sub_results, ignore_index=True)
    result_csv = cache_dir / f"result_chunk_{chunk_idx:05d}.csv"
    merged.to_csv(result_csv, index=False)

    n_recovered = len(merged)
    n_lost = n_rows - n_recovered
    print(f"  chunk {chunk_idx}: recovered {n_recovered}/{n_rows} rows ({n_lost} lost across {n_sub_failed} failed sub-chunks)")
    return n_recovered, n_lost, n_sub_failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--subchunk-size", type=int, default=500,
                        help="Rows per sub-chunk during retry (default 500). Smaller = finer "
                             "isolation of bad rows but more JIT overhead per sub-chunk.")
    args = parser.parse_args()

    if not JULIA_SCRIPT.exists():
        sys.exit(f"ERROR: julia script not found at {JULIA_SCRIPT}")

    failed = find_failed_chunks(args.cache_dir)
    print(f"found {len(failed)} failed chunks: {failed}")
    if not failed:
        print("nothing to retry. (orchestrate_magemin.py should be ready to merge.)")
        return

    total_recovered = 0
    total_lost = 0
    total_sub_failed = 0
    t0 = time()

    for chunk_idx in failed:
        n_rec, n_lost, n_fail = retry_one_chunk(
            chunk_idx, args.cache_dir, args.subchunk_size
        )
        total_recovered += n_rec
        total_lost += n_lost
        total_sub_failed += n_fail

    elapsed = time() - t0
    print(f"\n=== summary ===")
    print(f"  chunks attempted:           {len(failed)}")
    print(f"  total rows recovered:       {total_recovered:,}")
    print(f"  total rows lost (deterministic crashes): {total_lost:,}")
    print(f"  sub-chunks that segfaulted: {total_sub_failed}")
    print(f"  wall time:                  {elapsed:.1f}s")
    print(f"\nnext: re-run orchestrate_magemin.py (no --clean) to merge to magemin_results.csv")


if __name__ == "__main__":
    main()
