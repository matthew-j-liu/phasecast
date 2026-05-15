"""v2 Phase 1: per-bulk FC sweep from T_start down to T_floor_v2.

One ptt.isobaric_crystallisation call per bulk, dt_C=25, Frac_solid=True.
Saves the full mass_g trajectory across cooling steps. Downstream picks
T_FC_end_class per class from the trajectory and extracts the modal vector
at that T.

Output: data/experiments/eq_vs_frac_v2/fc_sweep.parquet
Schema: one row per (bulk, cooling step), columns include sample_id,
qap_name, step_idx, T_C, plus every column from mass_g (instantaneous and
*_cumsum mass per phase, plus liquid1 and fluid1 if present).
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "eq_vs_frac"))
from _phase_map import to_short  # noqa: E402

MW_FEO = 71.844
MW_FE2O3 = 159.69


def to_ptt_bulk(row: pd.Series) -> dict:
    feo, fe2o3 = float(row["feo"]), float(row["fe2o3"])
    feot = feo + fe2o3 * (2 * MW_FEO / MW_FE2O3)
    mol_fe2 = feo / MW_FEO
    mol_fe3 = 2 * fe2o3 / MW_FE2O3
    fe3fet = mol_fe3 / (mol_fe2 + mol_fe3) if (mol_fe2 + mol_fe3) > 0 else 0.0
    return {
        "SiO2": float(row["sio2"]), "Al2O3": float(row["al2o3"]),
        "CaO": float(row["cao"]), "MgO": float(row["mgo"]),
        "FeOt": feot, "K2O": float(row["k2o"]), "Na2O": float(row["na2o"]),
        "TiO2": float(row["tio2"]), "Cr2O3": float(row["cr2o3"]),
        "H2O": float(row["h2o_tot"]), "Fe3Fet": fe3fet,
    }


def main():
    import petthermotools as ptt
    ptt.activate_petthermotools_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--classes", type=str, default=None,
                    help="comma-separated class names to restrict the sweep (default: all)")
    ap.add_argument("--append", action="store_true",
                    help="append to existing fc_sweep.parquet, replacing rows for swept classes")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cls_cfg = cfg["classes"]
    v2_cfg = cfg["v2"]
    fc_cfg = cfg["fc"]

    out_dir = REPO / v2_cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse v1's bulk sample (same 550 bulks).
    bulks = pd.read_parquet(REPO / cfg["experiment"]["output_dir"] / "bulks.parquet")
    if args.classes:
        keep = set(c.strip() for c in args.classes.split(","))
        bulks = bulks[bulks["qap_name"].isin(keep)].reset_index(drop=True)
    if args.limit:
        bulks = bulks.groupby("qap_name", group_keys=False).head(args.limit).reset_index(drop=True)
    print(f"sweeping FC for {len(bulks)} bulks across {bulks['qap_name'].nunique()} classes")

    chunks: list[pd.DataFrame] = []
    n_fail = 0
    t0 = time.time()

    for _, row in tqdm(bulks.iterrows(), total=len(bulks), leave=True):
        cls = row["qap_name"]
        params = cls_cfg[cls]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ptt.isobaric_crystallisation(
                    Model=fc_cfg["model"],
                    bulk=to_ptt_bulk(row),
                    Frac_solid=True,
                    T_start_C=float(params["T_start"]),
                    T_end_C=float(params["T_floor_v2"]),
                    dt_C=float(v2_cfg["dt_C"]),
                    P_bar=float(params["P_kbar"] * 1000.0),
                    Suppress=fc_cfg["suppress"],
                    multi_processing=False,
                    timeout=fc_cfg["timeout_sec_per_bulk"],
                )
            mg = res.get("mass_g")
            if mg is None or len(mg) == 0:
                n_fail += 1
                continue
            mg = mg.reset_index(drop=True).copy()
            mg.insert(0, "sample_id", row["sample_id"])
            mg.insert(1, "qap_name", cls)
            mg.insert(2, "step_idx", np.arange(len(mg)))
            # Conditions DataFrame holds T_C / P_bar per step.
            cond = res.get("Conditions")
            if cond is not None and "T_C" in cond.columns:
                mg["T_C"] = cond["T_C"].values
            chunks.append(mg)
        except Exception as e:
            n_fail += 1
            tqdm.write(f"  fail {row['sample_id']} ({cls}): {type(e).__name__}: {e}")

    dt = time.time() - t0
    print(f"\nwall: {dt:.0f}s ({dt/max(len(bulks),1):.1f}s/bulk)")
    print(f"converged: {len(chunks)}/{len(bulks)}  fails: {n_fail}")

    if not chunks:
        print("no successful sweeps")
        return

    # Different bulks have different phase sets; align via concat with fill.
    df = pd.concat(chunks, ignore_index=True, sort=False)
    sweep_path = out_dir / "fc_sweep.parquet"
    if args.append and sweep_path.exists():
        existing = pd.read_parquet(sweep_path)
        swept = set(df["qap_name"].unique())
        existing = existing[~existing["qap_name"].isin(swept)]
        df = pd.concat([existing, df], ignore_index=True, sort=False)
        print(f"appended; combined sweep: {df.shape}")
    else:
        print(f"combined sweep: {df.shape}")
    df.to_parquet(sweep_path, index=False)
    print(f"wrote {sweep_path}")


if __name__ == "__main__":
    main()
