"""v2 Phase 5: L1 comparison of equilibrium_v2 vs fractional_v2.

In-situ L1: phase union INCLUDES liq + fl (we're comparing system state at T,
not eventual surface mineralogy). Audit-flagged bulks (liq_at_T > threshold)
are dropped to keep the comparison clean.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    v2_cfg = cfg["v2"]
    out_dir = REPO / v2_cfg["output_dir"]
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    eq = pd.read_parquet(out_dir / "equilibrium_v2.parquet")
    fr = pd.read_parquet(out_dir / "fractional_v2.parquet")
    audit = pd.read_parquet(out_dir / "audit_flags.parquet")
    bulks = pd.read_parquet(REPO / cfg["experiment"]["output_dir"] / "bulks.parquet")[["sample_id", "qap_name"]]

    eq = eq.merge(bulks, on="sample_id", how="left")
    flagged = set(audit[audit["flagged"]]["sample_id"])
    print(f"audit-flagged: {len(flagged)} bulks (excluded from L1)")

    ids = (set(eq["sample_id"]) & set(fr["sample_id"])) - flagged
    print(f"eq: {len(eq)}  fr: {len(fr)}  intersect (post-audit): {len(ids)}")

    phases = sorted(set(eq.columns).union(fr.columns) - {"sample_id", "qap_name", "T_used"})
    print(f"union phases ({len(phases)}): {phases}")

    rows = []
    for sid in ids:
        e = eq[eq["sample_id"] == sid].iloc[0]
        f = fr[fr["sample_id"] == sid].iloc[0]
        cls = e["qap_name"]
        v_eq = np.array([float(e.get(p, 0.0) or 0.0) for p in phases])
        v_fr = np.array([float(f.get(p, 0.0) or 0.0) for p in phases])
        if v_eq.sum() > 0: v_eq /= v_eq.sum()
        if v_fr.sum() > 0: v_fr /= v_fr.sum()
        rows.append({
            "sample_id": sid, "qap_name": cls,
            "l1": float(np.abs(v_eq - v_fr).sum()),
            "v_eq": v_eq, "v_fr": v_fr,
        })
    df = pd.DataFrame(rows)
    print(f"\noverall L1: median={df['l1'].median():.3f}  mean={df['l1'].mean():.3f}")

    per_class = df.groupby("qap_name")["l1"].agg(
        ["count", "median", "mean",
         lambda s: s.quantile(0.25),
         lambda s: s.quantile(0.75)]
    ).rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
    print("\nL1 by class:")
    print(per_class.sort_values("median").to_string())
    per_class.to_csv(out_dir / "results_v2.csv")

    deltas = {}
    for cls, g in df.groupby("qap_name"):
        v_eq = np.stack(g["v_eq"].values)
        v_fr = np.stack(g["v_fr"].values)
        deltas[cls] = (v_fr - v_eq).mean(axis=0)
    delta_df = pd.DataFrame(deltas, index=phases).T

    fig, ax = plt.subplots(figsize=(10, 5))
    order = per_class.sort_values("median").index.tolist()
    data = [df[df["qap_name"] == c]["l1"].values for c in order]
    ax.boxplot(data, labels=order)
    ax.axhline(0.10, color="g", linestyle="--", lw=1, label="adopt eq (<0.10)")
    ax.axhline(0.25, color="r", linestyle="--", lw=1, label="adopt fc (>0.25)")
    ax.set_ylabel("L1 modal distance (in-situ eq vs fc at same T)")
    ax.set_title("v2: anchored terminal T, liq < 5% threshold")
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(plots_dir / "l1_by_class_v2.png", dpi=130)
    print(f"wrote {plots_dir/'l1_by_class_v2.png'}")

    fig, ax = plt.subplots(figsize=(0.5 * len(phases) + 2, 0.5 * len(order) + 2))
    vmax = float(np.nanmax(np.abs(delta_df.values)))
    im = ax.imshow(delta_df.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(phases))); ax.set_xticklabels(phases, rotation=45, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_title("v2 per-phase signed delta (fractional - equilibrium), mean per class")
    plt.colorbar(im, ax=ax, label="modal fraction")
    plt.tight_layout()
    fig.savefig(plots_dir / "phase_delta_v2.png", dpi=130)
    print(f"wrote {plots_dir/'phase_delta_v2.png'}")

    med_max = per_class["median"].max()
    med_min = per_class["median"].min()
    print()
    if med_max < 0.10:
        print(f"DECISION: near-solidus equilibrium is fine (worst class median={med_max:.3f})")
    elif med_max > 0.25:
        worst = per_class["median"].idxmax()
        print(f"DECISION: adopt fractional crystallization (worst='{worst}' median={med_max:.3f})")
    else:
        big = per_class[per_class["median"] > 0.15].index.tolist()
        print(f"DECISION: hybrid (range [{med_min:.3f}, {med_max:.3f}]; FC for: {big})")


if __name__ == "__main__":
    main()
