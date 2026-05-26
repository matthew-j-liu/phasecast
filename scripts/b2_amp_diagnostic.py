"""Why is amphibole the weak major phase? (HANDOFF B2 -- pure analysis.)

amp per-path R^2 on test is ~0.22 -- low for a phase present in a third of the
corpus. Before designing CompHead (B1), pin down the cause: metric artifact,
irreducible signal loss, or genuine model deficiency.

The v1 corpus is many-to-one: each bulk has 8 FC paths at different (P, H2O)
but all 8 share an identical model input (the 10 anhydrous oxides are unchanged
across paths). The honest model therefore emits exactly one amp value per bulk
(verified: per-bulk std of pred_amp = 0). Per-path R^2 then penalizes the model
for within-bulk path spread it structurally cannot see.

This script decomposes amp's per-path R^2 as:

    per-path R^2  =  ceiling  x  R2_bulk

  - ceiling = SS_between / SS_tot: the highest per-path R^2 any bulk-only model
    can reach. The rest (SS_within / SS_tot) is within-bulk path spread.
  - R2_bulk: the model's genuine skill at predicting per-bulk-mean amp -- the
    fair metric.

It also measures how much the (leaked) h2o_tot input bought for amp by
comparing against the archived 11-oxide base_v1_ensemble_h2oleak ensemble, and
contrasts amp against the strong anhydrous majors cpx and ol.

Run:
    python scripts/b2_amp_diagnostic.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = REPO_ROOT / "data" / "diagnostics" / "b2_amp"


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    """R^2 against the target mean, matching scripts/06_eval_ensemble.py."""
    ss_res = np.sum((pred - target) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def variance_decomposition(df: pd.DataFrame, phase: str) -> dict:
    """Split SS_tot of target_<phase> into within-bulk and between-bulk parts.

    Returns the two sums of squares, the per-path R^2 ceiling, the n_b-weighted
    bulk-level R^2 of the model, and the model's per-path R^2.
    """
    col, pred_col = f"target_{phase}", f"pred_{phase}"
    y = df[col].to_numpy()
    pred = df[pred_col].to_numpy()
    grand = y.mean()

    g = df.groupby("sample_id")
    ybar_b = g[col].transform("mean").to_numpy()      # per-row bulk mean target
    pred_b = g[pred_col].transform("mean").to_numpy()  # per-row bulk mean pred

    ss_tot = float(np.sum((y - grand) ** 2))
    ss_within = float(np.sum((y - ybar_b) ** 2))
    ss_between = float(np.sum((ybar_b - grand) ** 2))
    ceiling = ss_between / ss_tot

    ss_res_bulk = float(np.sum((pred_b - ybar_b) ** 2))
    r2_bulk = 1.0 - ss_res_bulk / (ss_between + 1e-12)
    r2_path = r2(pred, y)

    return {
        "phase": phase,
        "presence": float((y > 0.01).mean()),
        "ss_tot": ss_tot,
        "ss_within": ss_within,
        "ss_between": ss_between,
        "within_frac": ss_within / ss_tot,
        "ceiling": ceiling,
        "r2_bulk": r2_bulk,
        "r2_path": r2_path,
        "r2_path_check": ceiling * r2_bulk,
    }


def per_class_amp(df: pd.DataFrame, min_presence: float = 0.05) -> pd.DataFrame:
    """Per-class amp presence and per-path R^2 (class-local target mean)."""
    rows = []
    for cls, sub in df.groupby("qap_name"):
        y = sub["target_amp"].to_numpy()
        presence = float((y > 0.01).mean())
        if presence < min_presence:
            continue
        rows.append({"qap_name": cls, "n": len(sub), "presence": presence,
                     "r2_path": r2(sub["pred_amp"].to_numpy(), y)})
    return pd.DataFrame(rows).sort_values("presence", ascending=False)


def plot_ceilings(decomps: list[dict], out: Path, min_presence: float = 0.05) -> None:
    """Per-phase within/between SS decomposition with the model's R^2 overlaid.

    Restricted to phases with presence > min_presence so rare-phase noise
    doesn't dominate. R^2 is clipped to [-0.1, 1] for legibility.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decomps = [d for d in decomps if d["presence"] >= min_presence]
    decomps = sorted(decomps, key=lambda d: d["ceiling"])
    names = [d["phase"] for d in decomps]
    between = np.array([d["ceiling"] for d in decomps])
    within = 1.0 - between
    r2_path = np.clip([d["r2_path"] for d in decomps], -0.1, 1.0)
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, between, color="tab:blue", label="between-bulk (per-path R^2 ceiling)")
    ax.bar(x, within, bottom=between, color="tab:gray", alpha=0.55,
           label="within-bulk (path spread -- unreachable)")
    ax.plot(x, r2_path, "o", color="black", ms=6, label="model per-path R^2")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("fraction of SS_tot  /  R^2")
    ax.set_title(f"per-phase variance: within-bulk path spread caps per-path R^2 "
                 f"(presence >= {min_presence:.2f})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_amp_parity(df: pd.DataFrame, dec: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df.groupby("sample_id")
    ybar_b = g["target_amp"].mean()
    pred_b = g["pred_amp"].mean()

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(df["pred_amp"], df["target_amp"], s=4, alpha=0.10, color="tab:gray",
               label=f"path-rows  (R^2 = {dec['r2_path']:.2f})")
    ax.scatter(pred_b, ybar_b, s=10, alpha=0.5, color="tab:blue",
               label=f"bulk means  (R^2 = {dec['r2_bulk']:.2f})")
    hi = max(float(df["target_amp"].max()), float(df["pred_amp"].max()))
    ax.plot([0, hi], [0, hi], "k--", lw=1.0)
    ax.set_xlabel("predicted amp wt%")
    ax.set_ylabel("true amp wt%")
    ax.set_title("amp: one prediction per bulk vs an 8-path target spread")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_per_class(pc: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(pc))
    ax.bar(x, pc["r2_path"], color="tab:blue")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pc["qap_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("amp per-path R^2")
    ax.set_title("amp per-path R^2 by rock class (classes with >5% amp presence)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--honest", type=Path,
                        default=REPO_ROOT / "checkpoints/base_v1_ensemble/predictions_test.parquet")
    parser.add_argument("--h2oleak", type=Path,
                        default=REPO_ROOT / "checkpoints/base_v1_ensemble_h2oleak/predictions_test.parquet")
    args = parser.parse_args()

    hon = pd.read_parquet(args.honest)
    leak = pd.read_parquet(args.h2oleak)
    same_split = set(hon["sample_id"]) == set(leak["sample_id"])
    print(f"honest test rows: {len(hon)}  bulks: {hon['sample_id'].nunique()}")
    print(f"h2oleak shares the same test split: {same_split}")

    pred_std = hon.groupby("sample_id")["pred_amp"].std().max()
    print(f"honest pred_amp per-bulk std (max): {pred_std:.2e}  "
          f"-> model emits one amp value per bulk\n")

    contrast = ["amp", "cpx", "ol", "opx", "bi", "q", "fsp"]
    decomps = {ph: variance_decomposition(hon, ph) for ph in contrast}

    print("variance decomposition (test, per-path R^2 = ceiling x R2_bulk):")
    print(f"  {'phase':6s} {'presence':>9s} {'within%':>8s} {'ceiling':>8s} "
          f"{'R2_bulk':>8s} {'R2_path':>8s} {'check':>8s}")
    for ph in contrast:
        d = decomps[ph]
        print(f"  {ph:6s} {d['presence']:9.3f} {d['within_frac']:8.3f} "
              f"{d['ceiling']:8.3f} {d['r2_bulk']:8.3f} {d['r2_path']:8.3f} "
              f"{d['r2_path_check']:8.3f}")

    amp = decomps["amp"]
    leak_r2 = r2(leak["pred_amp"].to_numpy(), leak["target_amp"].to_numpy())
    print(f"\namp per-path R^2: honest 10-oxide = {amp['r2_path']:.3f}  "
          f"|  h2oleak 11-oxide = {leak_r2:.3f}  "
          f"(leaked h2o_tot buys +{leak_r2 - amp['r2_path']:.3f})")

    pc = per_class_amp(hon)
    print(f"\namp per-path R^2 by class (>5% presence), {len(pc)} classes:")
    for _, r in pc.iterrows():
        print(f"  {r['qap_name']:28s} n={r['n']:5d} presence={r['presence']:.3f} "
              f"R2_path={r['r2_path']:+.3f}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    all_phases = [c[len("target_"):] for c in hon.columns if c.startswith("target_")
                  and c[len("target_"):] not in ("T", "P", "H2O")]
    all_decomps = [variance_decomposition(hon, ph) for ph in all_phases]
    plot_ceilings(all_decomps, PLOT_DIR / "phase_variance_ceilings.png")
    plot_amp_parity(hon, amp, PLOT_DIR / "amp_parity.png")
    plot_per_class(pc, PLOT_DIR / "amp_per_class_r2.png")
    print(f"\nplots -> {PLOT_DIR}")

    print("\n--- summary ---")
    print(f"amp per-path R^2 = {amp['r2_path']:.2f} = ceiling {amp['ceiling']:.2f} "
          f"x R2_bulk {amp['r2_bulk']:.2f}")
    print(f"{amp['within_frac']*100:.0f}% of amp's total variance is within-bulk "
          f"path spread -- structurally invisible to a bulk-only model.")
    print(f"contrast: cpx ceiling {decomps['cpx']['ceiling']:.2f}, "
          f"ol ceiling {decomps['ol']['ceiling']:.2f} -- the anhydrous majors "
          f"have little within-bulk spread, so amp's low ceiling is amp-specific.")


if __name__ == "__main__":
    main()
