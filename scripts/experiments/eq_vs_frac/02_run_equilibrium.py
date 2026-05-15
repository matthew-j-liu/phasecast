"""Equilibrium leg: run existing run_magemin.jl on the experiment bulks, parse
the results into a per-bulk modal vector (one row per sample), write parquet.

Reuses scripts/parse_magemin.py helpers for vocabulary discovery and the
duplicate-phase-name aggregation fix.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from parse_magemin import discover_vocabularies, parse_phases_row  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = REPO / cfg["experiment"]["output_dir"]

    eq_inputs = out_dir / "eq_inputs.csv"
    eq_results = out_dir / "eq_results.csv"
    eq_parquet = out_dir / "equilibrium.parquet"

    if not eq_results.exists():
        cmd = [
            "julia",
            f"--project={REPO/'scripts'/'julia'}",
            "-t", "auto",
            str(REPO / "scripts" / "julia" / "run_magemin.jl"),
            "--input", str(eq_inputs),
            "--output", str(eq_results),
            "--database", "ig",
        ]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"reusing existing {eq_results}")

    results = pd.read_csv(eq_results)
    n_total = len(results)
    results = results[results["status"] == 0].copy()
    print(f"converged: {len(results)}/{n_total}")
    results["phases_json"] = results["phases_json"].fillna("")
    results["ss_json"] = results["ss_json"].fillna("")

    phases, _ = discover_vocabularies(results)
    print(f"phases: {phases}")

    Y = np.zeros((len(results), len(phases)), dtype=np.float32)
    for i, ph_json in enumerate(results["phases_json"]):
        Y[i] = parse_phases_row(ph_json, phases)

    sums = Y.sum(axis=1)
    print(f"phase sum: mean={sums.mean():.4f} min={sums.min():.4f} max={sums.max():.4f}")

    df = pd.DataFrame(Y, columns=phases)
    df.insert(0, "sample_id", results["sample_id"].to_numpy())
    df.to_parquet(eq_parquet, index=False)
    print(f"wrote {eq_parquet}: {df.shape}")


if __name__ == "__main__":
    main()
