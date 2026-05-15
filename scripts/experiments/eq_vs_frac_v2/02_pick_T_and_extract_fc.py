"""v2 Phase 2 + 4: from fc_sweep.parquet, derive T_FC_end_class per class and
extract the FC in-situ modal vector at that T per bulk.

Rule: class-aggregate mean liq < threshold (default 5 wt%) at the highest T
that satisfies it. Quantize to nearest 25C. Per-bulk audit flags bulks whose
liq > threshold at T_FC_end_class.

In-situ modal at T_FC_end_class per bulk:
  solids   = sum of all <phase>_cumsum at that step (mass g)
  liq      = liquid1 at that step
  fl       = fluid1  at that step (NaN -> 0)
  total    ~ 100 g (mass conserved by FC); normalize to 1.0
Phases with the same short code (plagioclase + alkali-feldspar -> fsp,
spinel1+spinel2 -> spl) are summed via _phase_map.to_short.

Outputs:
  T_FC_end_table.csv      class, T_FC_end_class, n_bulks, n_flagged, mean_liq_at_T
  audit_flags.parquet     per-bulk: sample_id, qap_name, T_used, liq_at_T, flagged
  fractional_v2.parquet   per-bulk: sample_id, qap_name, T_used, phase columns (normalized)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "eq_vs_frac"))
from _phase_map import to_short  # noqa: E402


def quantize(T: float, step: int) -> int:
    return int(np.floor(T / step) * step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    v2_cfg = cfg["v2"]
    cls_cfg = cfg["classes"]
    threshold = float(v2_cfg["liq_threshold"]) * 100.0  # wt% (since mass = 100g)
    quant = int(v2_cfg["T_quantize_C"])

    out_dir = REPO / v2_cfg["output_dir"]
    sweep = pd.read_parquet(out_dir / "fc_sweep.parquet")
    print(f"loaded sweep: {sweep.shape}, classes: {sweep['qap_name'].unique().tolist()}")

    sweep["liquid1"] = sweep["liquid1"].fillna(0.0)
    if "fluid1" in sweep.columns:
        sweep["fluid1"] = sweep["fluid1"].fillna(0.0)
    else:
        sweep["fluid1"] = 0.0

    # 1. T_FC_end_class per class
    rows = []
    T_choice: dict[str, int] = {}
    for cls in cls_cfg:
        g = sweep[sweep["qap_name"] == cls]
        if g.empty:
            print(f"  {cls}: no sweep data")
            continue
        # Mean liq at each T across bulks in class
        per_T = g.groupby("T_C")["liquid1"].mean().sort_index(ascending=False)
        passing = per_T[per_T < threshold]
        if passing.empty:
            T_C_picked = int(per_T.index.min())
            note = "FALLBACK: never below threshold; using floor"
        else:
            T_C_picked = quantize(passing.index.max(), quant)
            note = ""
        T_choice[cls] = T_C_picked
        # Per-bulk liq at the picked T
        at_T = g[g["T_C"] == T_C_picked]
        if at_T.empty:
            nearest = g.iloc[(g["T_C"] - T_C_picked).abs().argsort()].head(50)
            T_C_picked = int(nearest["T_C"].iloc[0])
            T_choice[cls] = T_C_picked
            at_T = g[g["T_C"] == T_C_picked]
        n_flagged = int((at_T["liquid1"] > threshold).sum())
        rows.append({
            "class": cls,
            "T_FC_end_class": T_C_picked,
            "n_bulks": len(at_T),
            "n_flagged_over_thresh": n_flagged,
            "mean_liq_at_T_wt%": float(at_T["liquid1"].mean()),
            "p75_liq_at_T_wt%": float(at_T["liquid1"].quantile(0.75)),
            "p95_liq_at_T_wt%": float(at_T["liquid1"].quantile(0.95)),
            "note": note,
        })
    T_table = pd.DataFrame(rows)
    print("\nT_FC_end per class:")
    print(T_table.to_string(index=False))
    T_table.to_csv(out_dir / "T_FC_end_table.csv", index=False)

    # 2. Per-bulk extraction at T_FC_end_class
    audit_rows = []
    modal_rows = []
    cumsum_cols = [c for c in sweep.columns if c.endswith("_cumsum")]
    for cls, T_C in T_choice.items():
        g = sweep[(sweep["qap_name"] == cls) & (sweep["T_C"] == T_C)]
        for _, r in g.iterrows():
            liq_g = float(r["liquid1"])
            fl_g = float(r.get("fluid1", 0.0) or 0.0)
            phase_g: dict[str, float] = {}
            for col in cumsum_cols:
                short = to_short(col[: -len("_cumsum")])
                if short is None:
                    continue
                v = r[col]
                v = float(v) if not pd.isna(v) else 0.0
                if v > 0:
                    phase_g[short] = phase_g.get(short, 0.0) + v
            if liq_g > 0:
                phase_g["liq"] = phase_g.get("liq", 0.0) + liq_g
            if fl_g > 0:
                phase_g["fl"] = phase_g.get("fl", 0.0) + fl_g
            total = sum(phase_g.values())
            if total <= 0:
                continue
            modal = {k: v / total for k, v in phase_g.items()}
            modal["sample_id"] = r["sample_id"]
            modal["qap_name"] = cls
            modal["T_used"] = T_C
            modal_rows.append(modal)
            audit_rows.append({
                "sample_id": r["sample_id"], "qap_name": cls, "T_used": T_C,
                "liq_at_T_wt%": liq_g, "fl_at_T_wt%": fl_g,
                "flagged": bool(liq_g > threshold),
            })

    audit = pd.DataFrame(audit_rows)
    audit.to_parquet(out_dir / "audit_flags.parquet", index=False)
    print(f"\naudit: {audit['flagged'].sum()} of {len(audit)} bulks flagged (liq>{threshold:.0f}wt%)")
    print("  by class:")
    print(audit.groupby("qap_name")["flagged"].agg(["sum", "count"]).to_string())

    # Build wide modal table
    all_phases = sorted({k for r in modal_rows for k in r if k not in ("sample_id", "qap_name", "T_used")})
    mat = np.zeros((len(modal_rows), len(all_phases)), dtype=np.float32)
    for i, r in enumerate(modal_rows):
        for j, p in enumerate(all_phases):
            mat[i, j] = r.get(p, 0.0)
    fr_df = pd.DataFrame(mat, columns=all_phases)
    fr_df.insert(0, "sample_id", [r["sample_id"] for r in modal_rows])
    fr_df.insert(1, "qap_name", [r["qap_name"] for r in modal_rows])
    fr_df.insert(2, "T_used", [r["T_used"] for r in modal_rows])
    fr_df.to_parquet(out_dir / "fractional_v2.parquet", index=False)
    print(f"\nwrote fractional_v2.parquet: {fr_df.shape}; phases: {all_phases}")


if __name__ == "__main__":
    main()
