"""Side-by-side compare two training runs by per-phase R² and presence.

Reads metrics.jsonl files from two checkpoint dirs and reports the per-phase
delta. Useful for "smoke vs full" type comparisons.

Run:
    python scripts/compare_runs.py \
        --baseline checkpoints/base_v0/ \
        --new      checkpoints/base_v0_full/ \
        --baseline-h5 data/train_magemin_smoke.h5 \
        --new-h5      data/train_magemin.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_ckpt(ckpt_dir: Path) -> dict:
    """Load best.pt + phase_names + final metrics."""
    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    return {
        "phase_names": ckpt["phase_names"],
        "n_minerals": ckpt["n_minerals"],
        "best_epoch": ckpt["best_epoch"],
        "best_val_loss": ckpt["best_val_loss"],
        "config": ckpt["config"],
    }


def load_final_metrics(metrics_path: Path) -> dict:
    """Get the final-epoch JSON record from metrics.jsonl."""
    with open(metrics_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return {}
    return rows[-1]


def phase_summary_from_h5(h5_path: Path) -> dict[str, float]:
    """Return phase presence (fraction of samples) by phase name."""
    with h5py.File(h5_path, "r") as f:
        Y_wt = f["Y_wt"][:]
        phase_names = [c.decode() if isinstance(c, bytes) else c for c in f["Y_cols"][:]]
    presence = (Y_wt > 0.01).mean(axis=0)
    return dict(zip(phase_names, presence.astype(float).tolist()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True, help="checkpoint dir 1 (e.g. smoke)")
    p.add_argument("--new", type=Path, required=True, help="checkpoint dir 2 (e.g. full scale)")
    p.add_argument("--baseline-h5", type=Path, default=None)
    p.add_argument("--new-h5", type=Path, default=None)
    args = p.parse_args()

    base = load_ckpt(args.baseline)
    new = load_ckpt(args.new)

    print(f"{'Property':<30s} {'baseline':>15s} {'new':>15s}")
    print("-" * 65)
    print(f"{'best epoch':<30s} {base['best_epoch']:>15} {new['best_epoch']:>15}")
    print(f"{'best val loss':<30s} {base['best_val_loss']:>15.5f} {new['best_val_loss']:>15.5f}")
    print(f"{'n_minerals':<30s} {base['n_minerals']:>15} {new['n_minerals']:>15}")

    # Final-epoch metrics (no R² in JSONL by default; pull from H5 + recomputed in compare script if needed)
    if args.baseline_h5 and args.baseline_h5.exists():
        pres_base = phase_summary_from_h5(args.baseline_h5)
    else:
        pres_base = {}
    if args.new_h5 and args.new_h5.exists():
        pres_new = phase_summary_from_h5(args.new_h5)
    else:
        pres_new = {}

    if pres_base or pres_new:
        print("\nPer-phase val-set presence (fraction of samples with wt > 0.01):")
        all_phases = sorted(set(pres_base) | set(pres_new))
        # Sort by max presence in either run
        all_phases.sort(key=lambda p: -max(pres_base.get(p, 0), pres_new.get(p, 0)))
        print(f"  {'phase':<10s} {'baseline':>10s} {'new':>10s}  {'delta':>8s}")
        for ph in all_phases:
            b = pres_base.get(ph, float("nan"))
            n = pres_new.get(ph, float("nan"))
            d = n - b if not (np.isnan(b) or np.isnan(n)) else float("nan")
            print(f"  {ph:<10s} {b:>10.3f} {n:>10.3f}  {d:>+8.3f}")

    # The richest per-phase R² info would require re-running eval on each
    # checkpoint, since it isn't logged per-epoch in jsonl currently. Print
    # next steps to do that manually if needed:
    print("\nNote: per-phase R² is logged to stdout during training but not "
          "to metrics.jsonl. Re-run evaluate() on each checkpoint to get a "
          "side-by-side R² table.")


if __name__ == "__main__":
    main()
