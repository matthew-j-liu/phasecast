"""Exploratory plots on the real-mineralogy benchmark, for the endpoint/label
decision. The crux plot is granularity: at what SiO2 do quartz / biotite /
alkali feldspar actually appear as phenocrysts (vs being groundmass-only)? If
they only turn on in felsic rocks, a phenocryst-regime synthetic label captures
them where they matter and legitimately omits them where they aren't phenocrysts.

Reads data/real_mineralogy/{samples,modes}.csv, writes eda/*.png + eda/README.md.
    conda run -n alignn_fix python explorations/real_mineralogy/eda.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "real_mineralogy"
OUT = Path(__file__).resolve().parent / "eda"

# mineral-level phases we track; the three "granularity" minerals first
GRANULARITY = ["quartz", "biotite", "alkali feldspar"]
AMPHIBOLE = ["hornblende", "cummingtonite"]
PYROXENE = ["pyroxene", "augite", "hypersthene", "bronzite"]
SILICA = ["quartz", "silica phases"]


def load():
    s = pd.read_csv(DATA / "samples.csv")
    m = pd.read_csv(DATA / "modes.csv")
    m = m.merge(s[["source", "sample_id", "sio2", "total_crystals_pct", "mode_basis"]],
                on=["source", "sample_id"], how="left")
    m["present"] = m.abundance_pct.notna() | m.qualifier.notna()
    return s, m


def sio2_binned_presence(m, phases, denom_samples):
    """Presence frequency of each phase vs SiO2 bin, over samples that carry SiO2."""
    bins = np.arange(45, 82, 3)
    cent = (bins[:-1] + bins[1:]) / 2
    denom = denom_samples.dropna(subset=["sio2"]).copy()
    denom["bin"] = pd.cut(denom.sio2, bins)
    n_per_bin = denom.groupby("bin", observed=False).size()
    out = {}
    for ph in phases:
        present_ids = set(m[(m.phase == ph) & m.present].sample_id)
        d = denom.copy()
        d["has"] = d.sample_id.isin(present_ids)
        freq = d.groupby("bin", observed=False).has.mean()
        out[ph] = freq.values
    return cent, out, n_per_bin.values


def fig_coverage(s):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for src, g in s.groupby("source"):
        ax[0].hist(g.sio2.dropna(), bins=np.arange(45, 82, 2), alpha=0.55, label=src)
    ax[0].set_xlabel("SiO2 wt%"); ax[0].set_ylabel("samples"); ax[0].set_title("composition coverage")
    ax[0].legend(fontsize=7)
    for src, g in s.groupby("source"):
        ax[1].scatter(g.sio2, g.na2o + g.k2o, s=12, alpha=0.5, label=src)
    ax[1].set_xlabel("SiO2 wt%"); ax[1].set_ylabel("Na2O + K2O wt%"); ax[1].set_title("TAS-style")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "1_coverage.png", dpi=120); plt.close(fig)


def fig_granularity(s, m):
    """The crux: where do quartz / biotite / alkali feldspar appear as phenocrysts?"""
    dubray = s[s.source.str.startswith("dubray")]
    cent, freqs, nbin = sio2_binned_presence(m[m.source.str.startswith("dubray")],
                                             GRANULARITY + ["plagioclase", "hornblende", "pyroxene", "olivine"],
                                             dubray)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for ph in GRANULARITY:
        ax[0].plot(cent, freqs[ph], "o-", label=ph)
    ax[0].axvline(63, ls=":", color="grey"); ax[0].text(63.3, 0.05, "andesite/dacite", fontsize=7, color="grey")
    ax[0].set_xlabel("bulk SiO2 wt%"); ax[0].set_ylabel("phenocryst presence frequency")
    ax[0].set_title("granularity minerals: where do they turn on?\n(du Bray, N=%d)" % dubray.sio2.notna().sum())
    ax[0].legend(fontsize=8); ax[0].set_ylim(-0.03, 1.03)
    for ph in ["plagioclase", "hornblende", "pyroxene", "olivine"]:
        ax[1].plot(cent, freqs[ph], "s-", label=ph, alpha=0.8)
    ax[1].set_xlabel("bulk SiO2 wt%"); ax[1].set_ylabel("phenocryst presence frequency")
    ax[1].set_title("mafic/intermediate phenocryst phases (contrast)")
    ax[1].legend(fontsize=8); ax[1].set_ylim(-0.03, 1.03)
    fig.tight_layout(); fig.savefig(OUT / "2_granularity.png", dpi=120); plt.close(fig)


def fig_crystallinity(s):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for src, g in s.groupby("source"):
        ax[0].scatter(g.sio2, g.total_crystals_pct, s=12, alpha=0.5, label=src)
    ax[0].set_xlabel("SiO2 wt%"); ax[0].set_ylabel("total crystals %")
    ax[0].set_title("crystallinity vs composition\n(groundmass = 100 - this)"); ax[0].legend(fontsize=7)
    ax[1].hist(s.total_crystals_pct.dropna(), bins=np.arange(0, 105, 5))
    ax[1].set_xlabel("total crystals %"); ax[1].set_ylabel("samples")
    ax[1].set_title("crystallinity spread\n(the axis chemistry can't predict)")
    fig.tight_layout(); fig.savefig(OUT / "3_crystallinity.png", dpi=120); plt.close(fig)


def fig_amp_px(s, m):
    def present_ids(phases):
        return set(m[m.phase.isin(phases) & m.present].sample_id)
    d = s[s.source.str.startswith("dubray")].dropna(subset=["sio2"]).copy()
    bins = np.arange(45, 82, 3); cent = (bins[:-1] + bins[1:]) / 2
    d["bin"] = pd.cut(d.sio2, bins)
    amp, px = present_ids(AMPHIBOLE), present_ids(PYROXENE)
    d["amp"] = d.sample_id.isin(amp); d["px"] = d.sample_id.isin(px)
    fa = d.groupby("bin", observed=False).amp.mean(); fp = d.groupby("bin", observed=False).px.mean()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(cent, fa.values, "o-", label="amphibole (hydrous)")
    ax.plot(cent, fp.values, "s-", label="pyroxene (anhydrous)")
    ax.set_xlabel("bulk SiO2 wt%"); ax.set_ylabel("phenocryst presence frequency")
    ax.set_title("amphibole vs pyroxene footprint in real rocks (du Bray)")
    ax.legend(); ax.set_ylim(-0.03, 1.03)
    fig.tight_layout(); fig.savefig(OUT / "4_amp_vs_px.png", dpi=120); plt.close(fig)


def fig_richness(s, m):
    mm = m[(m.phase != "groundmass") & m.present]
    rich = mm.groupby(["source", "sample_id"]).phase.nunique().rename("n_phases").reset_index()
    rich = rich.merge(s[["source", "sample_id", "sio2"]], on=["source", "sample_id"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].scatter(rich.sio2, rich.n_phases, s=12, alpha=0.5)
    ax[0].set_xlabel("SiO2 wt%"); ax[0].set_ylabel("# distinct phenocryst phases")
    ax[0].set_title("assemblage richness vs composition")
    phases = ["plagioclase", "alkali feldspar", "quartz", "biotite", "hornblende",
              "pyroxene", "olivine", "Fe-Ti oxides"]
    ids = {p: set(mm[mm.phase == p].apply(lambda r: (r.source, r.sample_id), axis=1)) for p in phases}
    n = len(phases); co = np.zeros((n, n))
    allids = set().union(*ids.values())
    for i, a in enumerate(phases):
        for j, b in enumerate(phases):
            co[i, j] = len(ids[a] & ids[b]) / max(len(ids[a]), 1)
    im = ax[1].imshow(co, cmap="viridis", vmin=0, vmax=1)
    ax[1].set_xticks(range(n)); ax[1].set_xticklabels(phases, rotation=90, fontsize=7)
    ax[1].set_yticks(range(n)); ax[1].set_yticklabels(phases, fontsize=7)
    ax[1].set_title("co-occurrence P(col present | row present)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / "5_richness_cooccurrence.png", dpi=120); plt.close(fig)


def fig_visual_vs_pointcount(m):
    common = ["plagioclase", "hornblende"]
    fig, ax = plt.subplots(1, len(common), figsize=(11, 4.2))
    for k, ph in enumerate(common):
        vis = m[(m.phase == ph) & m.source.str.startswith("dubray")].abundance_pct.dropna()
        pc = m[(m.phase == ph) & ~m.source.str.startswith("dubray")].abundance_pct.dropna()
        ax[k].hist(vis, bins=np.arange(0, 70, 5), alpha=0.55, density=True, label="du Bray (visual)")
        ax[k].hist(pc, bins=np.arange(0, 70, 5), alpha=0.55, density=True, label="MSH+Pin (pointcount)")
        ax[k].set_xlabel(f"{ph} abundance %"); ax[k].set_ylabel("density"); ax[k].legend(fontsize=7)
    fig.suptitle("visual estimate vs point count, shared phases")
    fig.tight_layout(); fig.savefig(OUT / "6_visual_vs_pointcount.png", dpi=120); plt.close(fig)


def granularity_numbers(s, m):
    d = m[m.source.str.startswith("dubray") & m.present].merge(
        s[["source", "sample_id", "sio2"]], on=["source", "sample_id"], suffixes=("", "_s"))
    d = d.dropna(subset=["sio2_s"])
    lines = ["granularity minerals: SiO2 distribution of phenocryst occurrences (du Bray)\n"]
    for ph in GRANULARITY + ["olivine", "plagioclase"]:
        occ = d[d.phase == ph].sio2_s
        if len(occ) == 0:
            continue
        lines.append(f"{ph:18s} n={len(occ):4d}  median SiO2={occ.median():4.1f}  "
                     f"p10={occ.quantile(.1):4.1f}  frac<63%={np.mean(occ < 63):.2f}  frac<52%={np.mean(occ < 52):.2f}")
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    s, m = load()
    print(f"samples {len(s)}, mode rows {len(m)}; SiO2 present {s.sio2.notna().sum()}")
    fig_coverage(s); fig_granularity(s, m); fig_crystallinity(s)
    fig_amp_px(s, m); fig_richness(s, m); fig_visual_vs_pointcount(m)
    summary = granularity_numbers(s, m)
    print("\n" + summary)
    (OUT / "granularity_numbers.txt").write_text(summary + "\n")
    print("\nwrote eda/*.png + granularity_numbers.txt")


if __name__ == "__main__":
    main()
