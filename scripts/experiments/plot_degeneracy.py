"""Plot the per-bulk degeneracy spread from visualize_degeneracy.py output.

Reads:
  data/diagnostics/degeneracy_viz/modal_per_path.parquet
  data/diagnostics/degeneracy_viz/composition_per_path.parquet

Produces (all PNGs in same dir):
  fig1_modal_violin.png        -- distribution of modal wt% per phase across 100 paths
  fig2_T_FC_end_heatmap.png    -- T_FC_end as a function of (P, H2O) — shows T is path-dependent
  fig3_modal_heatmap_PH2O.png  -- modal wt% per phase, gridded over (P, H2O), for key phases
  fig4_composition_violin.png  -- per-phase oxide composition distributions for major phases
  fig5_composition_vs_P.png    -- key compositional axes (cpx jadeite, amp tschermak, plag An) vs P
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "data" / "diagnostics" / "degeneracy_viz"

modal = pd.read_parquet(DIR / "modal_per_path.parquet")
comp = pd.read_parquet(DIR / "composition_per_path.parquet")
print(f"loaded {len(modal)} paths; {modal.shape[1]-3} phases")
print(f"composition table: {len(comp)} (path × phase × oxide) entries")

# Identify phase columns (everything not P_kbar, H2O_wt, T_FC_end)
meta_cols = {"P_kbar", "H2O_wt", "T_FC_end"}
phase_cols = [c for c in modal.columns if c not in meta_cols]

# Sort phases by mean modal (descending) for readable plotting
phase_order = sorted(phase_cols, key=lambda p: -modal[p].mean())
print(f"phases by mean modal: {phase_order}")


def fig1_modal_violin():
    """Distribution of modal wt% per phase, across 100 paths.
    Each violin shows the within-bulk spread for that phase.
    Mean = where the ModeHead point prediction would land.
    Spread = the degeneracy variance the model cannot capture from a single bulk input.
    """
    data = [modal[p].to_numpy() for p in phase_order]
    fig, ax = plt.subplots(figsize=(max(8, len(phase_order) * 0.55), 5))
    parts = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("#4682B4")
        body.set_alpha(0.7)
    ax.set_xticks(range(1, len(phase_order)+1))
    ax.set_xticklabels(phase_order, rotation=45, ha="right")
    ax.set_ylabel("mass fraction of cumulate")
    ax.set_title(f"Modal-abundance spread across {len(modal)} (P, H2O) paths\n"
                 f"(one Arc basalt; the within-bulk variance ModeHead must average over)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIR / "fig1_modal_violin.png", dpi=130)
    plt.close(fig)


def fig2_T_FC_end_heatmap():
    """T_FC_end as a function of (P, H2O). Demonstrates that T_FC_end is
    path-dependent (not a single value per bulk as the current pipeline
    assumes)."""
    pivot = modal.pivot_table(index="H2O_wt", columns="P_kbar", values="T_FC_end")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto",
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.min(), pivot.index.max()],
                   cmap="coolwarm")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("T_FC_end (°C) where liq < 5 wt%")
    # Annotate
    for i, h in enumerate(pivot.index):
        for j, p in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                ax.text(p, h, f"{int(v)}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v - pivot.values[~np.isnan(pivot.values)].mean()) > 50 else "black")
    ax.set_xlabel("P (kbar)")
    ax.set_ylabel("H2O (wt%)")
    ax.set_title(f"T_FC_end depends on (P, H2O) — range across paths: "
                 f"{int(modal['T_FC_end'].min())}-{int(modal['T_FC_end'].max())} °C "
                 f"(spread = {int(modal['T_FC_end'].max() - modal['T_FC_end'].min())} °C)")
    fig.tight_layout()
    fig.savefig(DIR / "fig2_T_FC_end_heatmap.png", dpi=130)
    plt.close(fig)


def fig3_modal_heatmap_PH2O():
    """For the most P/H2O-sensitive phases (amp, ol, fl, liq), show modal
    abundance gridded over (P, H2O). Shows where the within-bulk variance
    actually lives."""
    sensitive = [p for p in ["amp", "ol", "cpx", "fsp", "liq", "fl", "bi", "opx"]
                 if p in phase_cols]
    n = len(sensitive)
    fig, axes = plt.subplots(2, (n+1)//2, figsize=(4 * ((n+1)//2), 8))
    axes = axes.flatten()
    for i, ph in enumerate(sensitive):
        pivot = modal.pivot_table(index="H2O_wt", columns="P_kbar", values=ph)
        im = axes[i].imshow(pivot.values, origin="lower", aspect="auto",
                            extent=[pivot.columns.min(), pivot.columns.max(),
                                    pivot.index.min(), pivot.index.max()],
                            cmap="viridis")
        plt.colorbar(im, ax=axes[i], label="mass frac")
        axes[i].set_title(f"{ph} modal — mean {modal[ph].mean():.3f}, "
                          f"std {modal[ph].std():.3f}")
        axes[i].set_xlabel("P (kbar)")
        axes[i].set_ylabel("H2O (wt%)")
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Per-path modal abundance vs (P, H2O) — shows where degeneracy lives", y=1.0)
    fig.tight_layout()
    fig.savefig(DIR / "fig3_modal_heatmap_PH2O.png", dpi=130)
    plt.close(fig)


def fig4_composition_violin():
    """Per-phase oxide composition distributions across paths.
    For each major phase, show the distribution of each oxide.
    Tight distributions = high CompHead per-path R² ceiling.
    Wide distributions = composition is path-sensitive (CompHead degeneracy).
    """
    # Pick major phases that exist in the data and have at least 50 path-rows
    phase_counts = comp.groupby("phase").size()
    major_phases = [p for p in ["cpx", "fsp", "ol", "opx", "amp", "bi", "spl", "ilm"]
                    if p in phase_counts.index and phase_counts[p] >= 50 * 5]  # >=50 paths x 5 oxides
    if not major_phases:
        print("no major phases with enough data; skipping fig4")
        return

    fig, axes = plt.subplots(len(major_phases), 1,
                              figsize=(11, 2.5 * len(major_phases)))
    if len(major_phases) == 1:
        axes = [axes]
    for ax, ph in zip(axes, major_phases):
        sub = comp[comp["phase"] == ph]
        oxides = sorted(sub["oxide"].unique())
        data = [sub[sub["oxide"] == ox]["wt_pct"].to_numpy() for ox in oxides]
        parts = ax.violinplot(data, showmeans=True, widths=0.8)
        for body in parts["bodies"]:
            body.set_facecolor("#cd5c5c")
            body.set_alpha(0.7)
        ax.set_xticks(range(1, len(oxides)+1))
        ax.set_xticklabels(oxides)
        ax.set_ylabel("wt%")
        n_paths_for_ph = sub.groupby("P_kbar").size().shape[0]
        # Compute coefficient of variation per oxide (std/mean), reporting median
        cvs = []
        for ox in oxides:
            vals = sub[sub["oxide"] == ox]["wt_pct"].to_numpy()
            if len(vals) > 1 and vals.mean() > 0:
                cvs.append(vals.std() / vals.mean())
        median_cv = float(np.median(cvs)) if cvs else float("nan")
        ax.set_title(f"{ph}  (present in {n_paths_for_ph} P-values; "
                     f"median CV across oxides = {median_cv:.3f})")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-phase composition spread — CompHead must average over the within-bulk variance shown",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(DIR / "fig4_composition_violin.png", dpi=130)
    plt.close(fig)


def fig5_composition_vs_P():
    """For 3 known-P-sensitive endmember-like oxides, plot composition vs P
    colored by H2O. Shows the systematic P-dependence of composition that
    CompHead would have to learn.

    Picks:
      Cpx Al2O3 (jadeite + tschermak proxy)
      Amp Al2O3 (tschermak proxy)
      Plag CaO (anorthite proxy)
    """
    panels = [("cpx", "Al2O3"), ("amp", "Al2O3"), ("fsp", "CaO"), ("ol", "MgO")]
    panels = [(ph, ox) for ph, ox in panels
              if ph in comp["phase"].values
              and ox in comp[comp["phase"] == ph]["oxide"].values]
    if not panels:
        print("no compositional panels available; skipping fig5")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (ph, ox) in zip(axes, panels):
        sub = comp[(comp["phase"] == ph) & (comp["oxide"] == ox)]
        sc = ax.scatter(sub["P_kbar"], sub["wt_pct"], c=sub["H2O_wt"],
                        cmap="viridis", s=30, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="H2O (wt%)")
        # Add a linear fit to show trend
        if len(sub) > 1:
            slope, intercept = np.polyfit(sub["P_kbar"], sub["wt_pct"], 1)
            x_fit = np.linspace(sub["P_kbar"].min(), sub["P_kbar"].max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, "k--", alpha=0.5, lw=1)
            ax.text(0.05, 0.95, f"slope = {slope:+.3f} wt%/kbar",
                    transform=ax.transAxes, fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        ax.set_xlabel("P (kbar)")
        ax.set_ylabel(f"{ph} {ox} (wt%)")
        ax.set_title(f"{ph} {ox} across {len(sub)} paths")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Composition vs P (colored by H2O) — shows the structured (P, H2O) → composition mapping CompHead would predict")
    fig.tight_layout()
    fig.savefig(DIR / "fig5_composition_vs_P.png", dpi=130)
    plt.close(fig)


def summary_stats():
    """Print a numerical summary of within-bulk variances per phase / oxide."""
    print("\n=== Modal abundance summary (ModeHead degeneracy structure) ===")
    print(f"{'phase':>8s}  {'mean':>7s}  {'std':>7s}  {'cv':>5s}  {'min':>6s}  {'max':>6s}  {'n_paths':>7s}")
    for ph in phase_order:
        vals = modal[ph]
        mean = vals.mean()
        std = vals.std()
        cv = std / mean if mean > 0 else 0
        n_present = int((vals > 0.001).sum())
        print(f"{ph:>8s}  {mean:7.4f}  {std:7.4f}  {cv:5.2f}  {vals.min():6.3f}  {vals.max():6.3f}  {n_present:>7d}")

    print("\n=== Composition spread per phase / oxide (the CompHead degeneracy structure) ===")
    for ph in sorted(comp["phase"].unique()):
        sub = comp[comp["phase"] == ph]
        oxides = sorted(sub["oxide"].unique())
        if len(sub.groupby("P_kbar")) < 10:  # phase not present in enough paths
            continue
        print(f"\n{ph}:")
        print(f"  {'oxide':>6s}  {'mean':>7s}  {'std':>6s}  {'cv':>5s}  {'within-bulk variance fraction':>30s}")
        for ox in oxides:
            v = sub[sub["oxide"] == ox]["wt_pct"]
            if v.mean() > 0 and len(v) > 5:
                cv = v.std() / v.mean()
                ceiling = 1.0 - (v.std() ** 2) / max(v.std() ** 2 + 1.0, 1.0)  # placeholder
                print(f"  {ox:>6s}  {v.mean():7.3f}  {v.std():6.3f}  {cv:5.2f}  (range: {v.min():.2f}-{v.max():.2f})")


def main():
    fig1_modal_violin()
    fig2_T_FC_end_heatmap()
    fig3_modal_heatmap_PH2O()
    fig4_composition_violin()
    fig5_composition_vs_P()
    summary_stats()
    print(f"\nplots -> {DIR}")


if __name__ == "__main__":
    main()
