"""Generic chunk-parquet merger for the production pipeline.

The orchestrator scripts (01_find_T_FC_end_per_bulk.py, 02_generate_corpus.py)
write per-chunk parquets incrementally to a cache directory. Their in-script
merge step often hangs at mp.Pool.__exit__ when juliacall workers refuse to
SIGTERM cleanly. This utility merges the cached chunks independently of the
orchestrator's lifecycle.

Usage:
    python scripts/production/merge_chunks.py \\
        --cache-dir data/fc_corpus_cache \\
        --output data/train_fc_v1.parquet

    # Pre-pass merger (same pattern):
    python scripts/production/merge_chunks.py \\
        --cache-dir data/T_FC_end_cache \\
        --output data/bulks_v1_with_T.parquet
"""
import argparse
from pathlib import Path

import pandas as pd


# Columns that are metadata, not modal phase mass fractions. Used to compute
# the modal-only subset for fillna(0). Anything else is presumed to be a phase.
META_COLS = {
    "sample_id", "path_idx", "qap_name", "P_kbar", "h2o_path_wt",
    "T_FC_end_used", "T_FC_end_attrition", "T_FC_end_bulk",
    "liq_at_T_FC_end_wt", "drop_reason",
    "fe_provenance", "h2o_provenance", "fc_status",
    "sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
    "k2o", "na2o", "tio2", "cr2o3", "h2o_tot",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pattern", default="chunk_*.parquet")
    args = ap.parse_args()

    parts = sorted(args.cache_dir.glob(args.pattern))
    if not parts:
        print(f"no chunks matching {args.pattern} in {args.cache_dir}")
        return
    print(f"{len(parts)} chunk parquets")

    n_corrupt = 0
    frames = []
    for p in parts:
        try:
            frames.append(pd.read_parquet(p))
        except Exception as e:
            print(f"  CORRUPT: {p.name}: {type(e).__name__}: {e}")
            n_corrupt += 1
    if n_corrupt:
        print(f"  ({n_corrupt} corrupt chunks skipped)")
    if not frames:
        print("no readable chunks")
        return

    merged = pd.concat(frames, ignore_index=True, sort=False)
    # Modal columns: anything that isn't metadata. NaN means "phase absent at this row".
    modal_cols = [c for c in merged.columns if c not in META_COLS]
    if modal_cols:
        merged[modal_cols] = merged[modal_cols].fillna(0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    print(f"wrote {args.output}: {merged.shape}")

    if "fc_status" in merged.columns:
        vc = merged["fc_status"].value_counts()
        print("\nfc_status counts:")
        for k, v in vc.items():
            print(f"  {k:25s} {v}")
    if "liq" in merged.columns and "fc_status" in merged.columns:
        ok = merged[merged["fc_status"] == "ok"]
        if len(ok):
            print(f"\nmean liq mass frac (ok rows): {ok['liq'].mean():.4f}")
    if modal_cols:
        print(f"\nmodal phase count: {len(modal_cols)}")
        print(f"modal phases:      {sorted(modal_cols)}")


if __name__ == "__main__":
    main()
