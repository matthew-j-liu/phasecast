"""v2 Phase 3: eq leg at T_FC_end_class per bulk's class.

Builds an inputs CSV in run_magemin.jl's schema using T_FC_end_class as the
per-row T_celsius, shells out to julia run_magemin.jl, parses to a flat modal
parquet keeping liq and fl (in-situ comparison).

Output: equilibrium_v2.parquet
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

OXIDES = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
          "k2o", "na2o", "tio2", "cr2o3", "h2o_tot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    v2_cfg = cfg["v2"]
    out_dir = REPO / v2_cfg["output_dir"]

    T_table = pd.read_csv(out_dir / "T_FC_end_table.csv")
    T_map = dict(zip(T_table["class"], T_table["T_FC_end_class"]))
    bulks = pd.read_parquet(REPO / cfg["experiment"]["output_dir"] / "bulks.parquet")
    bulks["T_celsius"] = bulks["qap_name"].map(T_map)
    bulks = bulks.dropna(subset=["T_celsius"])
    bulks["T_celsius"] = bulks["T_celsius"].astype(float)

    eq_inputs = out_dir / "eq_inputs_v2.csv"
    eq_results = out_dir / "eq_results_v2.csv"
    eq_parquet = out_dir / "equilibrium_v2.parquet"

    cols = ["sample_id", "T_celsius", "P_kbar"] + OXIDES + ["fe_provenance", "h2o_provenance", "qap_name"]
    bulks[cols].to_csv(eq_inputs, index=False)
    print(f"wrote {eq_inputs} ({len(bulks)} bulks)")

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
    df.insert(1, "T_used", results["T_celsius"].to_numpy().astype(int))
    df.to_parquet(eq_parquet, index=False)
    print(f"wrote {eq_parquet}: {df.shape}")


if __name__ == "__main__":
    main()
