"""
Embed the MagmaDB arc-experiment compilation into the frozen SSL encoder latent
alongside the gard2019 igneous corpus, then measure what real-rock analogs with
measured (P, T, fO2) buy as transparent evidence + a coarse OOD flag.

MagmaDB (Candioti et al. 2026, G-cubed; the MineRS+MagmaDB merge shipped with
the ArcThermoML repo) is 510 experiment rows over 78 distinct starting bulks,
each run at several (P, T, fO2). Starting bulks are majors-only -- they tokenize
into 8 of our 9 major slots (Cr stays padded, matching production
`include_cr2o3=False`) with all 13 trace slots padded. Majors-only is in
distribution: ~24% of the gard pretrain corpus is itself majors-only.

Run (CSV ships outside the repo, in the ArcThermoML download):

    conda run -n alignn_fix python explorations/magmadb_analogs/run_analogs.py \
        --magmadb ~/Downloads/lcandiot-ArcThermoML-676d770/data/lab_experiments/MineRS_MagmaDB_merged_v2_H2Ocalc.csv

Writes results/magmadb_vs_gard.png and results/coverage_metrics.json.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from cnmineral.data.bulk_chemistry import OXIDE_TO_CATION_FACTOR
from cnmineral.data.tokens import MAJORS, N_TOKENS
from cnmineral.eval.latent import extract_latents, load_encoder_from_checkpoint
from cnmineral.training import create_dataset

CKPT = "checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt"
HERE = Path(__file__).resolve().parent

MODE_COLS = ["ol", "opx", "cpx", "plag", "amph", "mag", "sp", "ilm", "bt", "ksp", "gt", "qz"]
START_MAJORS = ["SiO2_start", "TiO2_start", "Al2O3_start", "FeO_start",
                "Fe2O3_start", "MgO_start", "CaO_start", "Na2O_start", "K2O_start"]
START_OXIDE_COL = {
    "Si": "SiO2_start", "Al": "Al2O3_start", "Ca": "CaO_start", "Mg": "MgO_start",
    "K": "K2O_start", "Na": "Na2O_start", "Ti": "TiO2_start",
}


def load_magmadb(path: Path) -> pd.DataFrame:
    """Read the semicolon-delimited, latin-1 MagmaDB merge with numeric coercion."""
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    for c in MODE_COLS + START_MAJORS + ["P_kbar", "T_C", "delta_buffer", "melt_fraction"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def magmadb_tokens(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize MagmaDB starting bulks the way the production gard pipeline does.

    8 majors populated (Si Al Ca Mg Fe K Na Ti); Cr padded (production trains
    `include_cr2o3=False`, so populating it would be OOD); all 13 traces padded
    (MagmaDB carries none). Fe blends FeO + Fe2O3 cation contributions exactly
    like `_resolve_iron`.
    """
    n = len(df)
    src = np.zeros((n, N_TOKENS), dtype=np.int64)
    frac = np.zeros((n, N_TOKENS), dtype=np.float32)
    for tok_idx, (sym, z) in enumerate(MAJORS):
        if sym == "Cr":
            continue
        if sym == "Fe":
            feo = pd.to_numeric(df["FeO_start"], errors="coerce").to_numpy(np.float32)
            fe2o3 = pd.to_numeric(df["Fe2O3_start"], errors="coerce").to_numpy(np.float32)
            values = (np.nan_to_num(feo) * OXIDE_TO_CATION_FACTOR["feo"]
                      + np.nan_to_num(fe2o3) * OXIDE_TO_CATION_FACTOR["fe2o3"])
        else:
            col = START_OXIDE_COL[sym]
            oxide_key = col.lower().replace("_start", "")
            raw = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float32)
            values = raw * OXIDE_TO_CATION_FACTOR[oxide_key]
        pop = np.isfinite(values) & (values > 0)
        src[pop, tok_idx] = z
        frac[pop, tok_idx] = values[pop] / 100.0
    return torch.from_numpy(src), torch.from_numpy(frac)


def magmadb_mg_number(df: pd.DataFrame) -> np.ndarray:
    fe_cat = (df["FeO_start"].fillna(0) * OXIDE_TO_CATION_FACTOR["feo"]
              + df["Fe2O3_start"].fillna(0) * OXIDE_TO_CATION_FACTOR["fe2o3"])
    mg_cat = df["MgO_start"] * OXIDE_TO_CATION_FACTOR["mgo"]
    mol_mg, mol_fe = mg_cat / 24.305, fe_cat / 55.845
    return (mol_mg / (mol_mg + mol_fe)).to_numpy()


class _TensorDS(torch.utils.data.Dataset):
    def __init__(self, src, frac):
        self.src, self.frac = src, frac

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, i):
        return {"src": self.src[i].clone(), "frac": self.frac[i].clone()}


def _assemblage(row) -> str:
    present = [(m, row[m]) for m in MODE_COLS if pd.notna(row[m]) and row[m] > 0]
    present.sort(key=lambda x: -x[1])
    return " ".join(f"{m}{v:.0f}" for m, v in present) or "(glass only)"


def _unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--magmadb", required=True, type=Path,
                    help="path to MineRS_MagmaDB_merged_v2_H2Ocalc.csv")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    encoder, ck = load_encoder_from_checkpoint(args.checkpoint)
    print(f"encoder epoch={ck.epoch} d_model={ck.config['d_model']}")

    mdf = load_magmadb(args.magmadb)
    msrc, mfrac = magmadb_tokens(mdf)
    mlat = extract_latents(encoder, _TensorDS(msrc, mfrac), device=args.device).numpy()

    gard = create_dataset(ck.config)
    glat = extract_latents(encoder, gard, device=args.device).numpy()
    ax = gard.petrological_axes()
    gsi, gmg, galk = ax["SiO2 (wt%)"], ax["Mg#"], ax["Na2O+K2O (wt%)"]
    qaps = np.array([str(q).lower() for q in gard.qap_names])

    mU, gU = _unit(mlat), _unit(glat)
    nn = (gU @ mU.T).max(1)  # nearest-analog cosine per gard bulk

    # distinct starting compositions (round majors to 2 dp)
    key = mdf[START_MAJORS].round(2).astype(str).agg("|".join, axis=1)
    rows_per_bulk = key.value_counts()

    rocktypes = {
        "all_igneous": np.ones(len(qaps), bool),
        "andesite": np.char.find(qaps, "andesite") >= 0,
        "basalt": np.char.find(qaps, "basalt") >= 0,
        "rhyolite_granite": (np.char.find(qaps, "rhyolite") >= 0) | (np.char.find(qaps, "granite") >= 0),
        "ultramafic_SiO2_lt45": gsi < 45,
    }
    coverage = {
        name: {"n": int(m.sum()),
               "frac_usable_cos_ge_0.7": round(float((nn[m] >= 0.7).mean()), 3),
               "median_cos": round(float(np.median(nn[m])), 3)}
        for name, m in rocktypes.items() if m.sum()
    }
    band = (gsi >= 52) & (gsi <= 58)
    sub_lo, sub_hi = band & (galk < 5), band & (galk > 8)
    alkaline_gap = {
        "band": "SiO2 52-58",
        "subalkaline_alk_lt5": {"n": int(sub_lo.sum()), "median_cos": round(float(np.median(nn[sub_lo])), 3)},
        "alkaline_alk_gt8": {"n": int(sub_hi.sum()), "median_cos": round(float(np.median(nn[sub_hi])), 3)},
    }

    # andesite analogs (sample_id 6875 is filtered out of the corpus; pick deterministically)
    and_idx = np.where(rocktypes["andesite"])[0]
    queries = sorted(rng.choice(and_idx, size=3, replace=False).tolist())
    qax = gard.petrological_axes(np.array(queries))
    retrievals = []
    for qi, qrow in enumerate(queries):
        sims_q = gU[qrow] @ mU.T
        topk = np.argsort(-sims_q)[:5]
        neighbors = [{
            "cos": round(float(sims_q[j]), 3),
            "SiO2_start": round(float(mdf.iloc[j]["SiO2_start"]), 1),
            "P_kbar": round(float(mdf.iloc[j]["P_kbar"]), 2),
            "T_C": round(float(mdf.iloc[j]["T_C"]), 0),
            "delta_NNO": round(float(mdf.iloc[j]["delta_buffer"]), 1),
            "melt_frac": round(float(mdf.iloc[j]["melt_fraction"]), 1),
            "assemblage": _assemblage(mdf.iloc[j]),
        } for j in topk]
        retrievals.append({
            "sample_id": int(gard.sample_ids[qrow]),
            "qap": qaps[qrow],
            "SiO2": round(float(qax["SiO2 (wt%)"][qi]), 1),
            "Mg#": round(float(qax["Mg#"][qi]), 2),
            "neighbors": neighbors,
        })

    metrics = {
        "encoder_checkpoint": args.checkpoint,
        "magmadb_rows": int(len(mdf)),
        "magmadb_distinct_bulks": int(key.nunique()),
        "rows_per_bulk_median": int(rows_per_bulk.median()),
        "rows_per_bulk_max": int(rows_per_bulk.max()),
        "magmadb_populated_tokens_per_row": int((msrc > 0).sum(1).median()),
        "coverage_by_rocktype": coverage,
        "alkaline_gap": alkaline_gap,
        "andesite_retrievals": retrievals,
    }
    out_json = HERE / "results" / "coverage_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"metrics -> {out_json}")

    _make_figure(gsi, gmg, galk, qaps, nn, mdf, mU, gU, rng,
                 HERE / "results" / "magmadb_vs_gard.png")


def _make_figure(gsi, gmg, galk, qaps, nn, mdf, mU, gU, rng, out_path):
    msi = mdf["SiO2_start"].to_numpy()
    malk = (mdf["Na2O_start"] + mdf["K2O_start"]).to_numpy()
    mmg = magmadb_mg_number(mdf)
    sub = rng.choice(len(gU), size=min(20000, len(gU)), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    a = axes[0, 0]
    a.hexbin(gsi[sub], galk[sub], gridsize=60, cmap="Greys", bins="log", mincnt=1)
    a.scatter(msi, malk, s=14, c="#d62728", edgecolors="k", linewidths=0.3, label="MagmaDB (n=510)")
    a.set_xlim(40, 80); a.set_ylim(0, 14)
    a.set_xlabel("SiO2 (wt%)"); a.set_ylabel("Na2O+K2O (wt%)")
    a.set_title("(a) TAS -- gard density (grey) vs MagmaDB starting bulks")
    a.legend(loc="upper left")

    b = axes[0, 1]
    b.hexbin(gsi[sub], gmg[sub], gridsize=60, cmap="Greys", bins="log", mincnt=1)
    b.scatter(msi, mmg, s=14, c="#d62728", edgecolors="k", linewidths=0.3)
    b.set_xlim(40, 80); b.set_ylim(0, 1)
    b.set_xlabel("SiO2 (wt%)"); b.set_ylabel("Mg#")
    b.set_title("(b) SiO2 vs Mg# -- gard density vs MagmaDB")

    c = axes[1, 0]
    pca = PCA(n_components=2, random_state=0).fit(gU)
    gp, mp2 = pca.transform(gU[sub]), pca.transform(mU)
    c.scatter(gp[:, 0], gp[:, 1], s=3, c="lightgrey", alpha=0.4, edgecolors="none", label="gard")
    c.scatter(mp2[:, 0], mp2[:, 1], s=16, c="#d62728", edgecolors="k", linewidths=0.3, label="MagmaDB")
    ev = pca.explained_variance_ratio_
    c.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)"); c.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    c.set_title("(c) encoder latent (PCA of unit latents) -- retrieval space")
    c.legend(loc="best")

    d = axes[1, 1]
    bins = np.linspace(0, 1, 41)
    for name, m, col in [
        ("all igneous", np.ones(len(qaps), bool), "0.4"),
        ("andesite", np.char.find(qaps, "andesite") >= 0, "#1f77b4"),
        ("basalt", np.char.find(qaps, "basalt") >= 0, "#2ca02c"),
        ("rhyolite/granite", (np.char.find(qaps, "rhyolite") >= 0) | (np.char.find(qaps, "granite") >= 0), "#ff7f0e"),
    ]:
        d.hist(nn[m], bins=bins, density=True, histtype="step", lw=1.8, color=col,
               label=f"{name} (n={m.sum()})")
    d.axvline(0.7, color="k", ls="--", lw=1, label="usable-analog cut (cos=0.7)")
    d.set_xlabel("cosine sim to nearest MagmaDB analog"); d.set_ylabel("density")
    d.set_title("(d) coverage / OOD: nearest-analog cosine per gard bulk")
    d.legend(fontsize=8)

    fig.suptitle("MagmaDB (484 arc experiments, 78 distinct bulks) vs gard2019 igneous corpus", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"figure -> {out_path}")


if __name__ == "__main__":
    main()
