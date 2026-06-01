"""Companion to 09_visualize_latent.py.

Two new diagnostics requested in JOURNAL discussion 2026-06-01:

  (1) Recolor the SSL-pretrained UMAP by continuous chemical axes (SiO2 wt%,
      Mg#, total alkalis) instead of categorical supergroup. The argument:
      bulk rock chemistry is continuous (basalt -> andesite -> dacite ->
      rhyolite is a differentiation continuum), so smooth gradients across the
      latent are the right evidence of structure, not discrete clusters.

  (2) 3D UMAP of the SSL-pretrained latent, colored by supergroup, saved as
      three orthographic views + an interactive plotly HTML. 2D UMAP has a
      bottleneck; 3D often resolves apparent cluster overlap.

Run after 09_visualize_latent.py has produced the pretrain encoder; uses the
same exclude-supervised-splits + seed=42 sampling so the 5000 bulks here are
the same set the 2D figure plotted.

Outputs:
  data/diagnostics/latent_v2/umap_by_chem_axes.png        (3 panels: SiO2, Mg#, alkalis)
  data/diagnostics/latent_v2/umap_3d_supergroup.png       (3 views of 3D UMAP)
  data/diagnostics/latent_v2/umap_3d_supergroup.html      (interactive)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse pieces from 09_visualize_latent.py
from importlib import import_module
viz = import_module("09_visualize_latent")

from cnmineral.data.unlabeled import UnlabeledV2Dataset

PLOT_DIR = REPO_ROOT / "data" / "diagnostics" / "latent_v2"
PRETRAIN_CKPT = REPO_ROOT / "checkpoints" / "pretrain_v2" / "encoder_best.pt"
SPLITS_PATH = REPO_ROOT / "data" / "splits_v1.json"
N_VIS = 5000          # match 09_visualize_latent.py default
SEED = 42             # match 09_visualize_latent.py default


def load_vis_subset(device: torch.device):
    """Replicate 09_visualize_latent.py's data flow: same exclusions, same seed,
    so the 5000 bulks selected here align with the existing 2D figures."""
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    exclude_ids = (splits["train_sample_ids"] + splits["val_sample_ids"]
                   + splits["test_sample_ids"]
                   + splits.get("dropped_class_sample_ids", []))
    print(f"loading UnlabeledV2Dataset (excluding {len(exclude_ids)} supervised bulks)")
    unl = UnlabeledV2Dataset(
        REPO_ROOT / "data/complete.csv", exclude_sample_ids=exclude_ids,
    )
    rng = np.random.default_rng(SEED)
    vis_idx = rng.choice(len(unl), size=min(N_VIS, len(unl)), replace=False)
    vis_subset = Subset(unl, vis_idx.tolist())
    print(f"  selected {len(vis_idx)} bulks for visualization")
    return unl, vis_idx, vis_subset


def chemical_axes(unl: UnlabeledV2Dataset, vis_idx: np.ndarray) -> dict:
    """SiO2 wt%, Mg# (mol), total alkalis (wt%), all per-bulk."""
    majors = unl._majors.iloc[vis_idx]
    sio2 = majors["sio2"].to_numpy(dtype=float)
    mgo_wt = majors["mgo"].to_numpy(dtype=float)
    feo_wt = majors["feo"].to_numpy(dtype=float)  # already merged Fe in unlabeled
    na2o = majors["na2o"].to_numpy(dtype=float)
    k2o = majors["k2o"].to_numpy(dtype=float)
    # Mg# = mol MgO / (mol MgO + mol FeO_total)
    mol_mgo = mgo_wt / 40.30
    mol_feo = feo_wt / 71.84
    denom = mol_mgo + mol_feo
    mg_num = np.where(denom > 0, mol_mgo / np.maximum(denom, 1e-9), np.nan)
    alkalis = na2o + k2o
    return {"SiO2 (wt%)": sio2, "Mg#": mg_num, "Na2O+K2O (wt%)": alkalis}


def umap_2d(latents: np.ndarray) -> np.ndarray:
    """Same UMAP params as 09_visualize_latent.py for consistency with existing
    2D figures (so the chem-axis recolor is on the same projection)."""
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        random_state=SEED, init="random")
    return reducer.fit_transform(latents)


def umap_3d(latents: np.ndarray) -> np.ndarray:
    """3D UMAP with the same neighbor and dist params, so structure is comparable."""
    import umap
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                        random_state=SEED, init="random")
    return reducer.fit_transform(latents)


def plot_chem_axes(emb2: np.ndarray, axes_dict: dict, out_path: Path):
    fig, axes = plt.subplots(1, len(axes_dict), figsize=(5 * len(axes_dict), 4.5))
    if len(axes_dict) == 1:
        axes = [axes]
    for ax, (name, vals) in zip(axes, axes_dict.items()):
        finite = np.isfinite(vals)
        lo, hi = np.percentile(vals[finite], [2, 98])
        sc = ax.scatter(emb2[finite, 0], emb2[finite, 1], c=vals[finite],
                        cmap="viridis", s=4, alpha=0.6, vmin=lo, vmax=hi,
                        rasterized=True)
        plt.colorbar(sc, ax=ax, label=name)
        ax.set_title(f"SSL latent UMAP -- {name}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_3d_views(emb3: np.ndarray, supergroups: np.ndarray, out_path: Path):
    """Three orthographic-ish viewpoints into the 3D UMAP, colored by supergroup."""
    fig = plt.figure(figsize=(15, 5))
    views = [(20, 30), (20, 120), (60, 60)]
    for i, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        for sg, color in viz.SG_COLORS.items():
            mask = supergroups == sg
            if mask.sum() == 0:
                continue
            ax.scatter(emb3[mask, 0], emb3[mask, 1], emb3[mask, 2],
                       s=4, c=color, alpha=0.55, label=f"{sg} (n={int(mask.sum())})",
                       rasterized=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"view elev={elev} azim={azim}")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        if i == 3:
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=7)
    fig.suptitle("SSL-pretrained encoder -- 3D UMAP, colored by petrogenetic supergroup")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_3d_interactive(emb3: np.ndarray, supergroups: np.ndarray,
                        sample_ids: np.ndarray, qap_names: np.ndarray,
                        out_path: Path):
    """Interactive plotly HTML so you can rotate / hover for class identity."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed; skipping interactive 3D HTML")
        return
    traces = []
    for sg, color in viz.SG_COLORS.items():
        mask = supergroups == sg
        if mask.sum() == 0:
            continue
        text = [f"sg={sg}<br>qap={q}<br>sample_id={s}"
                for q, s in zip(qap_names[mask], sample_ids[mask])]
        traces.append(go.Scatter3d(
            x=emb3[mask, 0], y=emb3[mask, 1], z=emb3[mask, 2],
            mode="markers",
            marker=dict(size=2.2, color=color, opacity=0.6),
            name=f"{sg} (n={int(mask.sum())})",
            text=text, hoverinfo="text",
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="SSL-pretrained latent -- 3D UMAP (rotate / hover)",
        scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  wrote {out_path}")


def main():
    device = viz.pick_device()
    print(f"device: {device}")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    unl, vis_idx, vis_subset = load_vis_subset(device)
    sg_vis = unl.supergroups[vis_idx]
    qap_vis = unl.qap_names[vis_idx]
    sid_vis = unl.sample_ids[vis_idx]

    print("encoding with SSL-pretrained encoder...")
    enc = viz.build_encoder(PRETRAIN_CKPT, device)
    lat = viz.pooled_latents(enc, vis_subset, batch_size=512, device=device)
    print(f"  latents: {lat.shape}")

    print("fitting UMAP-2D...")
    emb2 = umap_2d(lat)
    axes_dict = chemical_axes(unl, vis_idx)
    plot_chem_axes(emb2, axes_dict, PLOT_DIR / "umap_by_chem_axes.png")

    print("fitting UMAP-3D...")
    emb3 = umap_3d(lat)
    np.save(PLOT_DIR / "umap_3d_coords.npy", emb3)
    plot_3d_views(emb3, sg_vis, PLOT_DIR / "umap_3d_supergroup.png")
    plot_3d_interactive(emb3, sg_vis, sid_vis, qap_vis,
                        PLOT_DIR / "umap_3d_supergroup.html")

    print(f"\nDone. Outputs in {PLOT_DIR}")


if __name__ == "__main__":
    main()
