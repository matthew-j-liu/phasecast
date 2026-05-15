"""Stratified bulk sample for eq-vs-frac experiment.

Reads magemin_inputs.csv, samples up to n_per_class per qap_name from the
configured class list, attaches per-class T_celsius (midpoint of near-solidus
range) and P_kbar, writes a parquet plus a sidecar CSV in the schema
run_magemin.jl expects.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[3]
OXIDES = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
          "k2o", "na2o", "tio2", "cr2o3", "h2o_tot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    exp = cfg["experiment"]
    classes = cfg["classes"]

    inputs_path = REPO / exp["inputs_csv"]
    out_dir = REPO / exp["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {inputs_path}")
    df = pd.read_csv(inputs_path)
    print(f"  {len(df):,} rows")

    rng = np.random.default_rng(exp["seed"])
    picks = []
    for cls, params in classes.items():
        pool = df[df["qap_name"] == cls]
        n_want = exp["n_per_class"]
        if len(pool) == 0:
            print(f"  {cls}: 0 in pool, skipping")
            continue
        n_take = min(n_want, len(pool))
        idx = rng.choice(len(pool), size=n_take, replace=False)
        sub = pool.iloc[idx].copy()
        sub["T_celsius"] = 0.5 * (params["T_lo"] + params["T_hi"])
        sub["T_lo"] = params["T_lo"]
        sub["T_hi"] = params["T_hi"]
        sub["P_kbar"] = params["P_kbar"]
        picks.append(sub)
        print(f"  {cls}: {n_take}/{len(pool)} taken")

    bulks = pd.concat(picks, ignore_index=True)
    print(f"total bulks: {len(bulks)}")

    bulks_path = out_dir / "bulks.parquet"
    bulks.to_parquet(bulks_path, index=False)
    print(f"wrote {bulks_path}")

    # Sidecar CSV in the column order run_magemin.jl expects (it ignores extras).
    eq_csv_cols = ["sample_id", "T_celsius", "P_kbar"] + OXIDES + \
                  ["fe_provenance", "h2o_provenance", "qap_name"]
    eq_csv = bulks[eq_csv_cols].copy()
    eq_csv_path = out_dir / "eq_inputs.csv"
    eq_csv.to_csv(eq_csv_path, index=False)
    print(f"wrote {eq_csv_path}")


if __name__ == "__main__":
    main()
