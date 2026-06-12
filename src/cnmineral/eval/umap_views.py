"""UMAP figures for the SSL-pretrained latent.

Five plots, all rendered headless to disk:

- `plot_umap_supergroup` -- 2D UMAP, points colored by the 9 supergroups.
- `plot_umap_qap` -- 2D UMAP, top-N qap_name classes colored, rest grey.
  70 classes is too many for a categorical palette; the top-N coloring
  is the practical compromise.
- `plot_umap_chem_axes` -- 2D UMAP recolored by three continuous axes
  (SiO2 wt%, Mg#, total alkalis).  v2's headline figure; the proof
  that the latent organizes along petrological gradients.
- `plot_umap_3d_supergroup` -- 3D UMAP, three orthographic views,
  colored by supergroup.  Resolves cluster overlap that 2D collapses.
- `plot_knn_qap_purity` -- bar plot of mean k-NN qap_name purity at
  k=5/10/50, with a random-baseline line.

Two UMAP fits per call set (2D and 3D), since reusing a fit across
different `n_components` requires refitting anyway.  Embeddings are
returned alongside the saved PNG path so the caller can persist them.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import umap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers 3d projection
from cnmineral.eval.supergroup import SUPERGROUP_COLORS, SUPERGROUPS

UMAP_DEFAULTS: dict = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "cosine",
    "random_state": 42,
    "init": "random",
}


def fit_umap_2d(latents: np.ndarray, **kwargs) -> np.ndarray:
    """2D UMAP fit with project-wide defaults (cosine metric, 15 neighbors)."""
    params = {**UMAP_DEFAULTS, **kwargs}
    reducer = umap.UMAP(n_components=2, **params)
    return reducer.fit_transform(latents)


def fit_umap_3d(latents: np.ndarray, **kwargs) -> np.ndarray:
    """3D UMAP fit; same defaults as the 2D path so structure is comparable."""
    params = {**UMAP_DEFAULTS, **kwargs}
    reducer = umap.UMAP(n_components=3, **params)
    return reducer.fit_transform(latents)


def plot_umap_supergroup(
    emb2: np.ndarray,
    supergroups: np.ndarray,
    out_path: Path,
    *,
    title: str = "SSL latent UMAP - supergroup",
) -> Path:
    """2D UMAP scatter, one color per supergroup, legend ordered by SUPERGROUPS."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for sg in SUPERGROUPS:
        mask = supergroups == sg
        if not mask.any():
            continue
        ax.scatter(
            emb2[mask, 0], emb2[mask, 1],
            s=4, alpha=0.55,
            color=SUPERGROUP_COLORS.get(sg, "#444"),
            label=f"{sg} (n={int(mask.sum())})",
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_umap_qap(
    emb2: np.ndarray,
    qap_names: np.ndarray,
    out_path: Path,
    *,
    top_n: int = 12,
    title: str = "SSL latent UMAP - qap_name (top classes)",
) -> Path:
    """2D UMAP scatter; top-N qap_names colored, remainder grey.

    Categorical palettes top out around 10-12 distinct colors; with
    70 qap_name classes, coloring all of them produces an unreadable
    legend and visually indistinguishable buckets.  Top-N-by-count
    + grey rest preserves the dominant structure cleanly.
    """
    qap_arr = np.asarray(qap_names)
    classes, counts = np.unique(qap_arr, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_classes = classes[order][:top_n]
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    rest_mask = ~np.isin(qap_arr, top_classes)
    if rest_mask.any():
        ax.scatter(
            emb2[rest_mask, 0], emb2[rest_mask, 1],
            s=3, alpha=0.25, color="#bbbbbb",
            label=f"other ({len(classes) - len(top_classes)} classes, n={int(rest_mask.sum())})",
            rasterized=True,
        )
    for i, cls in enumerate(top_classes):
        mask = qap_arr == cls
        ax.scatter(
            emb2[mask, 0], emb2[mask, 1],
            s=5, alpha=0.7, color=cmap(i % 20),
            label=f"{cls} (n={int(mask.sum())})",
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_umap_chem_axes(
    emb2: np.ndarray,
    axes_dict: dict[str, np.ndarray],
    out_path: Path,
) -> Path:
    """Three-panel UMAP recolor; viridis on each continuous axis.

    Clips to the 2nd/98th percentile of finite values per panel so a
    single outlier doesn't compress the rest of the colorbar.
    """
    keys = list(axes_dict.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5))
    if len(keys) == 1:
        axes = [axes]
    for ax, name in zip(axes, keys):
        vals = axes_dict[name]
        finite = np.isfinite(vals)
        if not finite.any():
            ax.set_title(f"{name} (no finite values)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        lo, hi = np.percentile(vals[finite], [2, 98])
        sc = ax.scatter(
            emb2[finite, 0], emb2[finite, 1],
            c=vals[finite], cmap="viridis",
            s=4, alpha=0.6, vmin=lo, vmax=hi,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=name)
        ax.set_title(f"SSL latent UMAP - {name}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_umap_3d_supergroup(
    emb3: np.ndarray,
    supergroups: np.ndarray,
    out_path: Path,
) -> Path:
    """Three orthographic-ish 3D scatter views, colored by supergroup."""
    fig = plt.figure(figsize=(16, 5.5))
    views = [(20, 30), (20, 120), (60, 60)]
    for i, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        for sg in SUPERGROUPS:
            mask = supergroups == sg
            if not mask.any():
                continue
            ax.scatter(
                emb3[mask, 0], emb3[mask, 1], emb3[mask, 2],
                s=4, alpha=0.55,
                c=SUPERGROUP_COLORS.get(sg, "#444"),
                label=f"{sg} (n={int(mask.sum())})",
                rasterized=True,
            )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev} azim={azim}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if i == 3:
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=7)
    fig.suptitle("SSL-pretrained encoder - 3D UMAP, colored by supergroup")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_knn_qap_purity(
    purity_by_k: dict[int, float],
    random_baseline: float,
    out_path: Path,
    *,
    title: str = "k-NN qap_name purity (cosine on latent)",
) -> Path:
    """Bar plot of per-k mean qap-name purity with a random-baseline reference."""
    ks = sorted(purity_by_k.keys())
    values = [purity_by_k[k] for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(ks))
    bars = ax.bar(positions, values, color="#4c72b0", width=0.55)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax.axhline(
        random_baseline, color="k", ls="--", lw=1,
        label=f"class-balanced random ({random_baseline:.3f})",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylim(0, max(1.05, max(values) * 1.15))
    ax.set_ylabel("mean fraction of neighbors with matching qap_name")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
