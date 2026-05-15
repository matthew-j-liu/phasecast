"""Fractional crystallization leg via PetThermoTools.

Per-bulk serial loop (ptt's MAGEMin backend doesn't accept DataFrame bulks).
Skips find_liquidus by using a class-conditional fixed T_start_C — empirically
this matches find_liquidus=True endpoints to modal-L1 < 0.01 at ~6x speedup.

Output: fractional.parquet — one row per converged bulk; columns are canonical
short-name phases (cpx, ol, fsp, ...) plus residual 'liq' at termination;
each row sums to 1.0.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
        "SiO2": float(row["sio2"]),
        "Al2O3": float(row["al2o3"]),
        "CaO": float(row["cao"]),
        "MgO": float(row["mgo"]),
        "FeOt": feot,
        "K2O": float(row["k2o"]),
        "Na2O": float(row["na2o"]),
        "TiO2": float(row["tio2"]),
        "Cr2O3": float(row["cr2o3"]),
        "H2O": float(row["h2o_tot"]),
        "Fe3Fet": fe3fet,
    }


def final_modal(res: dict) -> dict:
    """Accumulated FC solids + residual melt at the final cooling step.

    With Frac_solid=True, mass_g has '<phase>_cumsum' columns (cumulative
    removed solids) plus instantaneous columns. Endpoint = last-row cumsum
    for solids + last-row liquid1 (residual melt). Phases sharing a short
    code (plagioclase + alkali-feldspar -> fsp) are summed.
    """
    out: dict[str, float] = {}
    mg = res.get("mass_g")
    if mg is None or len(mg) == 0:
        return out
    last = mg.iloc[-1]
    for col in mg.columns:
        if col.endswith("_cumsum"):
            short = to_short(col[: -len("_cumsum")])
            if short is None:
                continue
            v = float(last[col])
            if v > 0:
                out[short] = out.get(short, 0.0) + v
    if "liquid1" in mg.columns:
        v = float(last["liquid1"])
        if v > 0:
            out["liq"] = out.get("liq", 0.0) + v
    return out


def main():
    import petthermotools as ptt
    ptt.activate_petthermotools_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="optional cap for smoke runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    exp_cfg = cfg["experiment"]
    fc_cfg = cfg["fc"]
    cls_cfg = cfg["classes"]
    out_dir = REPO / exp_cfg["output_dir"]

    bulks = pd.read_parquet(out_dir / "bulks.parquet")
    if args.limit:
        bulks = bulks.groupby("qap_name", group_keys=False).head(args.limit).reset_index(drop=True)
    print(f"running FC on {len(bulks)} bulks")

    all_modal: list[dict] = []
    all_ids: list = []
    all_classes: list = []
    n_fail = 0
    t0 = time.time()

    pbar = tqdm(bulks.iterrows(), total=len(bulks), leave=True)
    for _, row in pbar:
        cls = row["qap_name"]
        params = cls_cfg[cls]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ptt.isobaric_crystallisation(
                    Model=fc_cfg["model"],
                    bulk=to_ptt_bulk(row),
                    Frac_solid=fc_cfg["frac_solid"],
                    T_start_C=float(params["T_start"]),
                    T_end_C=float(params["T_lo"]),
                    dt_C=fc_cfg["dt_C"],
                    P_bar=float(params["P_kbar"] * 1000.0),
                    Suppress=fc_cfg["suppress"],
                    multi_processing=False,
                    timeout=fc_cfg["timeout_sec_per_bulk"],
                )
            modal = final_modal(res)
        except Exception as e:
            modal = {}
            pbar.write(f"  fail {row['sample_id']} ({cls}): {e}")

        if not modal:
            n_fail += 1
            continue
        all_modal.append(modal)
        all_ids.append(row["sample_id"])
        all_classes.append(cls)

    dt = time.time() - t0
    print(f"\nwall: {dt:.0f}s ({dt/len(bulks):.1f}s/bulk)")
    print(f"converged: {len(all_modal)}/{len(bulks)}; failures: {n_fail}")

    if not all_modal:
        print("no successful runs")
        return

    all_phases = sorted({p for m in all_modal for p in m})
    Y = np.zeros((len(all_modal), len(all_phases)), dtype=np.float32)
    for i, m in enumerate(all_modal):
        for j, p in enumerate(all_phases):
            Y[i, j] = m.get(p, 0.0)
    sums = Y.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    Y = Y / sums

    df = pd.DataFrame(Y, columns=all_phases)
    df.insert(0, "sample_id", all_ids)
    df.insert(1, "qap_name", all_classes)
    out_path = out_dir / "fractional.parquet"
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {df.shape}")
    print(f"phases: {all_phases}")


if __name__ == "__main__":
    main()
