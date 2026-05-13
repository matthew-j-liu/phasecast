"""Chunked MAGEMin dispatch + crash recovery.

Splits the input CSV (per-row T/P from 02b_sample_tp.py) into chunks of size
`--chunk-size`, calls run_magemin.jl per chunk, accumulates results into a
single output CSV.

Crash recovery: per-chunk results are written to data/magemin_cache/. On
re-run, chunks whose result file already exists are skipped — so a run that
dies at chunk K can be resumed by simply re-running this script.

Run:
    # smoke (~2.6k points at K=5 per bulk × 522 bulks)
    python scripts/orchestrate_magemin.py

    # bigger chunk size for fewer Julia invocations on a long run
    python scripts/orchestrate_magemin.py --chunk-size 5000

    # force-rerun everything (delete cache first)
    python scripts/orchestrate_magemin.py --clean

Assumes Julia + MAGEMin_C.jl env is installed at scripts/julia/Project.toml.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from time import time

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
JULIA_DIR = REPO_ROOT / "scripts" / "julia"
JULIA_SCRIPT = JULIA_DIR / "run_magemin.jl"
CACHE_DIR_DEFAULT = REPO_ROOT / "data" / "magemin_cache"


def chunk_iter(df: pd.DataFrame, chunk_size: int):
    """Yield (chunk_index, sub_df) for each chunk."""
    n = len(df)
    for i, start in enumerate(range(0, n, chunk_size)):
        yield i, df.iloc[start : start + chunk_size].copy()


def run_julia_chunk(chunk_csv: Path, chunk_out: Path, database: str, threads: str) -> int:
    """Spawn julia subprocess for one chunk. Returns subprocess return code."""
    cmd = [
        "julia",
        f"--project={JULIA_DIR}",
        f"-t{threads}",
        str(JULIA_SCRIPT),
        "--input", str(chunk_csv),
        "--output", str(chunk_out),
        "--database", database,
        "--quiet",
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=REPO_ROOT / "data" / "magemin_inputs_tp.csv",
        help="Per-row (T, P, oxides) CSV from 02b_sample_tp.py",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "data" / "magemin_results.csv",
        help="Final merged results CSV",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=CACHE_DIR_DEFAULT,
        help="Directory for per-chunk intermediate results (gitignored)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Rows per chunk. Larger = fewer Julia JIT overheads; smaller = finer crash recovery.",
    )
    parser.add_argument(
        "--database", default="ig",
        help="MAGEMin thermodynamic database (ig=igneous, mp=metapelite)",
    )
    parser.add_argument(
        "--threads", default="auto",
        help="Julia thread count (passed as -tN). 'auto' uses all cores.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete cache dir before running (force-rerun all chunks)",
    )
    args = parser.parse_args()

    if not JULIA_SCRIPT.exists():
        sys.exit(f"ERROR: julia script not found at {JULIA_SCRIPT}")
    if not args.input.exists():
        sys.exit(f"ERROR: input CSV not found at {args.input}")

    if args.clean and args.cache_dir.exists():
        print(f"cleaning cache dir {args.cache_dir}")
        shutil.rmtree(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.input}")
    df = pd.read_csv(args.input)
    n_total = len(df)
    n_chunks = (n_total + args.chunk_size - 1) // args.chunk_size
    print(f"  {n_total:,} rows → {n_chunks} chunks of {args.chunk_size}")

    n_skipped = 0
    n_run = 0
    n_failed = 0
    t_overall = time()

    for chunk_i, sub_df in chunk_iter(df, args.chunk_size):
        chunk_csv = args.cache_dir / f"input_chunk_{chunk_i:05d}.csv"
        chunk_out = args.cache_dir / f"result_chunk_{chunk_i:05d}.csv"

        if chunk_out.exists():
            n_skipped += 1
            continue

        print(f"\n[chunk {chunk_i+1}/{n_chunks}] {len(sub_df)} rows")
        sub_df.to_csv(chunk_csv, index=False)
        t0 = time()
        rc = run_julia_chunk(chunk_csv, chunk_out, args.database, args.threads)
        elapsed = time() - t0
        if rc != 0:
            n_failed += 1
            print(f"  FAILED with exit code {rc} after {elapsed:.1f}s")
        else:
            n_run += 1
            print(f"  done in {elapsed:.1f}s")

    print(f"\n=== summary ===")
    print(f"  chunks skipped (cached):  {n_skipped}")
    print(f"  chunks run:               {n_run}")
    print(f"  chunks failed:            {n_failed}")
    print(f"  wall time:                {time() - t_overall:.1f}s")

    if n_failed > 0:
        sys.exit(f"\n{n_failed} chunk(s) failed; re-run to retry just the missing ones")

    print(f"\nmerging chunk results → {args.output}")
    result_files = sorted(args.cache_dir.glob("result_chunk_*.csv"))
    if not result_files:
        sys.exit("ERROR: no chunk result files found")

    parts = [pd.read_csv(f) for f in result_files]
    merged = pd.concat(parts, ignore_index=True)
    merged.to_csv(args.output, index=False)
    print(f"  {len(merged):,} rows written")
    print(f"  next step: python scripts/parse_magemin.py")


if __name__ == "__main__":
    main()
