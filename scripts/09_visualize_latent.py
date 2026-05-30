"""Latent-space diagnostics for the v2 SSL pretrain (HANDOFF Phase 4).

Produces:
  1. UMAP + PCA of pooled latents, random-init vs pretrained encoder, colored
     by petrogenetic supergroup. Separate UMAPs per encoder (don't share a
     fitted transform across encoders -- artificial).
  2. UMAP recolored by tectonic-discriminant ratios (Sr/Y, La/Yb, Nb/Y, Th/Yb).
  3. Three-panel UMAP (random / pretrained / fine-tuned) once the v2 cell-d
     fine-tune checkpoint exists.
  4. k-NN mineralogy sanity check: for each test bulk, k=10 nearest training
     bulks in 512-D pretrained latent; mean modal vs true modal, vs random
     baseline. Quantitative check that the latent is task-relevant.
  5. OOD detection: distance to k=10th-nearest training neighbor distribution.

Outputs go to data/diagnostics/latent_v2/.

Run:
    python scripts/09_visualize_latent.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnmineral.data.unlabeled import (
    SUPERGROUPS, UnlabeledV2Dataset, V2_TRACE_PPM_COLS,
)
from cnmineral.models.dataset import TrainFCv2Dataset
from cnmineral.models.encoder import Encoder
from cnmineral.models.model import BaseModelV1
from cnmineral.models.training import make_loaders_v1_bulk_split


PLOT_DIR = REPO_ROOT / "data" / "diagnostics" / "latent_v2"
PRETRAIN_CKPT = REPO_ROOT / "checkpoints" / "pretrain_v2" / "encoder_best.pt"
FINETUNE_CKPT = REPO_ROOT / "checkpoints" / "base_v2_d_full" / "best.pt"
SPLITS_PATH = REPO_ROOT / "data" / "splits_v1.json"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()):
        return torch.device("mps")
    return torch.device("cpu")


def build_encoder(state_dict_path: Path | None, device: torch.device) -> Encoder:
    """Build an Encoder. If state_dict_path is None: random init. Otherwise
    load encoder weights (from a pretrain checkpoint or a fine-tuned
    BaseModelV1 checkpoint).
    """
    enc = Encoder(d_model=512, N=4, heads=16, dropout=0.0, elem_prop="mat2vec")
    if state_dict_path is not None:
        ck = torch.load(state_dict_path, map_location="cpu", weights_only=False)
        if "encoder_state_dict" in ck:
            enc.load_state_dict(ck["encoder_state_dict"])
        elif "model_state_dict" in ck:
            full_sd = ck["model_state_dict"]
            enc_sd = {k[len("encoder."):]: v for k, v in full_sd.items()
                      if k.startswith("encoder.")}
            enc.load_state_dict(enc_sd)
        else:
            raise ValueError(f"Don't know how to load encoder from {state_dict_path}")
    enc.eval().to(device)
    return enc


@torch.no_grad()
def pooled_latents(encoder: Encoder, dataset, batch_size: int,
                   device: torch.device) -> np.ndarray:
    """Run encoder on every dataset row; return [N, d_model] pooled latents."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        h = encoder(batch["src"].to(device), batch["frac"].to(device))
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0)


def project_2d(X: np.ndarray, method: str, seed: int = 42) -> np.ndarray:
    """UMAP or PCA to 2D. UMAP is non-deterministic across machines, so we
    fix random_state for within-encoder reproducibility."""
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=seed, init="random")
        return reducer.fit_transform(X)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    raise ValueError(method)


# Stable color order for the 9 supergroups (8 real + 'other').
SG_COLORS: dict[str, str] = {
    "ultramafic": "#7f3b08",
    "mafic-volcanic": "#b35806",
    "mafic-plutonic": "#e08214",
    "intermediate-volcanic": "#fdb863",
    "intermediate-plutonic": "#b2abd2",
    "felsic-volcanic": "#8073ac",
    "felsic-plutonic": "#542788",
    "foid-bearing": "#2d004b",
    "other": "#bbbbbb",
}


def scatter_by_supergroup(ax, coords: np.ndarray, supergroups: np.ndarray,
                          title: str) -> None:
    for sg in SUPERGROUPS:
        m = supergroups == sg
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=3, alpha=0.55,
                   color=SG_COLORS.get(sg, "#444"), label=f"{sg} (n={int(m.sum())})")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def panel_pair_plot(coords_random: np.ndarray, coords_pretrained: np.ndarray,
                    supergroups: np.ndarray, method: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    scatter_by_supergroup(axes[0], coords_random, supergroups,
                          f"random-init encoder ({method.upper()})")
    scatter_by_supergroup(axes[1], coords_pretrained, supergroups,
                          f"SSL-pretrained encoder ({method.upper()})")
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def panel_three_plot(coords_random, coords_pretrained, coords_finetuned,
                     supergroups: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    scatter_by_supergroup(axes[0], coords_random, supergroups, "random init")
    scatter_by_supergroup(axes[1], coords_pretrained, supergroups, "SSL pretrained")
    scatter_by_supergroup(axes[2], coords_finetuned, supergroups,
                          "pretrained + fine-tuned (cell d)")
    axes[2].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def ratio_plot(coords: np.ndarray, ratios: dict[str, np.ndarray],
               method: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in ratios if np.isfinite(ratios[k]).any()]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5))
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        r = ratios[k]
        valid = np.isfinite(r) & (r > 0)
        # log scale on a positive-ratio axis; clip extremes for visibility
        log_r = np.log10(r[valid])
        vmin, vmax = np.percentile(log_r, [5, 95])
        sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=3, alpha=0.6,
                        c=np.clip(log_r, vmin, vmax), cmap="viridis")
        plt.colorbar(sc, ax=ax, label=f"log10({k})")
        ax.set_title(f"{k} on pretrained {method.upper()}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def knn_mineralogy_check(
    train_latents: np.ndarray, train_modal: np.ndarray,
    test_latents: np.ndarray, test_modal: np.ndarray,
    k: int = 10, seed: int = 42,
) -> dict:
    """For each test bulk, find k=10 nearest training bulks in the FULL 512-D
    pretrained latent. Mean modal of those neighbors vs true modal.
    Cosine similarity averaged across test bulks.
    Compare against random-neighbor baseline.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_latents)
    _, nbr_idx = nn.kneighbors(test_latents)
    nbr_modal = train_modal[nbr_idx].mean(axis=1)  # [N_test, n_phases]

    rng = np.random.default_rng(seed)
    rand_idx = rng.integers(0, len(train_latents), size=nbr_idx.shape)
    rand_modal = train_modal[rand_idx].mean(axis=1)

    def cos(a, b):
        num = (a * b).sum(axis=-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
        return num / den

    cos_knn = cos(nbr_modal, test_modal).mean()
    cos_rnd = cos(rand_modal, test_modal).mean()
    return {
        "k": k,
        "cosine_similarity_knn": float(cos_knn),
        "cosine_similarity_random": float(cos_rnd),
        "delta_vs_random": float(cos_knn - cos_rnd),
        "n_test": int(len(test_latents)),
        "n_train": int(len(train_latents)),
    }


def ood_distance_distribution(
    train_latents: np.ndarray, test_latents: np.ndarray,
    held_train: np.ndarray, k: int = 10,
) -> dict:
    """Distance from each point to its k=10-th nearest TRAINING neighbor.

    Compares the distribution for held-out train (in-distribution baseline)
    to the test set (target). If the two distributions overlap, test is
    in-distribution.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_latents)
    d_test, _ = nn.kneighbors(test_latents)
    d_held, _ = nn.kneighbors(held_train)
    d_test = d_test[:, -1]   # distance to the k-th neighbor
    d_held = d_held[:, -1]
    flag_threshold = float(np.percentile(d_held, 99))
    ood_test = int((d_test > flag_threshold).sum())
    return {
        "k": k,
        "ood_flag_threshold_99p_train": flag_threshold,
        "ood_test_n_flagged": ood_test,
        "ood_test_frac_flagged": float(ood_test / len(d_test)),
        "test_distances": d_test.tolist(),
        "held_train_distances": d_held.tolist(),
    }


def ood_distance_plot(stats: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(stats["held_train_distances"], bins=60, alpha=0.55,
            color="tab:gray", label="held-out train (in-distribution)", density=True)
    ax.hist(stats["test_distances"], bins=60, alpha=0.55,
            color="tab:red", label="test", density=True, histtype="step", lw=2)
    ax.axvline(stats["ood_flag_threshold_99p_train"], color="k", ls="--",
               label=f"99% train threshold = {stats['ood_flag_threshold_99p_train']:.2f}")
    ax.set_xlabel(f"distance to k={stats['k']}-th nearest training latent")
    ax.set_ylabel("density")
    ax.set_title(f"OOD diagnostic (frac flagged in test: "
                 f"{stats['ood_test_frac_flagged']:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-vis", type=int, default=5000,
                        help="held-out unlabeled bulks to project (default 5k)")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Unlabeled held-out set for UMAP/PCA (random vs pretrained) ----
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    exclude_ids = (splits["train_sample_ids"] + splits["val_sample_ids"]
                   + splits["test_sample_ids"]
                   + splits.get("dropped_class_sample_ids", []))
    print(f"loading UnlabeledV2Dataset (excluding {len(exclude_ids)} supervised bulks)")
    unl = UnlabeledV2Dataset(
        REPO_ROOT / "data/complete.csv", exclude_sample_ids=exclude_ids,
    )
    rng = np.random.default_rng(42)
    vis_idx = rng.choice(len(unl), size=min(args.n_vis, len(unl)), replace=False)
    vis_subset = Subset(unl, vis_idx.tolist())
    print(f"  selected {len(vis_idx)} bulks for visualization")

    sg_vis = unl.supergroups[vis_idx]

    print("encoding with random-init encoder...")
    enc_random = build_encoder(None, device)
    lat_random = pooled_latents(enc_random, vis_subset, args.batch_size, device)

    print("encoding with SSL-pretrained encoder...")
    enc_pretrained = build_encoder(PRETRAIN_CKPT, device)
    lat_pretrained = pooled_latents(enc_pretrained, vis_subset, args.batch_size, device)

    print("UMAP 2D (random)...")
    coords_random_umap = project_2d(lat_random, "umap")
    print("UMAP 2D (pretrained)...")
    coords_pretrained_umap = project_2d(lat_pretrained, "umap")
    print("PCA 2D (both)...")
    coords_random_pca = project_2d(lat_random, "pca")
    coords_pretrained_pca = project_2d(lat_pretrained, "pca")

    panel_pair_plot(coords_random_umap, coords_pretrained_umap, sg_vis, "umap",
                    PLOT_DIR / "umap_pretrain_vs_random.png")
    panel_pair_plot(coords_random_pca, coords_pretrained_pca, sg_vis, "pca",
                    PLOT_DIR / "pca_pretrain_vs_random.png")
    print(f"saved umap_pretrain_vs_random.png and pca_pretrain_vs_random.png")

    # ---- Ratio coloring on the pretrained UMAP ----
    # Pull the trace ppm cols for the vis bulks from complete.csv (already
    # loaded into the dataset; just use the cached _traces).
    trace_arr = unl._traces.to_numpy()[vis_idx]
    cols = [c.replace("_ppm", "") for c in V2_TRACE_PPM_COLS]
    col_idx = {c: i for i, c in enumerate(cols)}

    def ratio(a, b):
        ai, bi = col_idx[a], col_idx[b]
        num = trace_arr[:, ai].astype(np.float64)
        den = trace_arr[:, bi].astype(np.float64)
        out = np.full_like(num, np.nan, dtype=np.float64)
        valid = (num > 0) & (den > 0) & np.isfinite(num) & np.isfinite(den)
        out[valid] = num[valid] / den[valid]
        return out

    ratios = {"Sr/Y": ratio("sr", "y"),
              "La/Yb": ratio("la", "yb"),
              "Nb/Y": ratio("nb", "y"),
              "Th/Yb": ratio("th", "yb")}
    ratio_plot(coords_pretrained_umap, ratios, "umap",
               PLOT_DIR / "umap_by_ratio_pretrained.png")
    print("saved umap_by_ratio_pretrained.png")

    # ---- Three-panel UMAP if cell-d fine-tuned checkpoint exists ----
    if FINETUNE_CKPT.exists():
        print("encoding with fine-tuned (cell d) encoder...")
        enc_ft = build_encoder(FINETUNE_CKPT, device)
        lat_ft = pooled_latents(enc_ft, vis_subset, args.batch_size, device)
        coords_ft_umap = project_2d(lat_ft, "umap")
        panel_three_plot(coords_random_umap, coords_pretrained_umap,
                         coords_ft_umap, sg_vis,
                         PLOT_DIR / "umap_three_stage.png")
        print("saved umap_three_stage.png")
    else:
        print(f"(skipping three-panel: {FINETUNE_CKPT.name} not yet present)")

    # ---- k-NN sanity check on the SUPERVISED corpus, pretrained encoder ----
    print("\nrunning k-NN mineralogy sanity check on supervised train/test...")
    sup = TrainFCv2Dataset(
        parquet_path=REPO_ROOT / "data/train_fc_v1.parquet",
        complete_csv_path=REPO_ROOT / "data/complete.csv",
        liq_threshold=0.10,
    )
    train_ids = np.asarray(splits["train_sample_ids"], dtype=np.int64)
    test_ids = np.asarray(splits["test_sample_ids"], dtype=np.int64)
    train_mask = np.isin(sup.sample_ids, train_ids)
    test_mask = np.isin(sup.sample_ids, test_ids)
    print(f"  train rows: {int(train_mask.sum())}, test rows: {int(test_mask.sum())}")

    sup_train_subset = Subset(sup, np.where(train_mask)[0].tolist())
    sup_test_subset = Subset(sup, np.where(test_mask)[0].tolist())
    print("  encoding supervised train...")
    train_lat = pooled_latents(enc_pretrained, sup_train_subset, args.batch_size, device)
    print("  encoding supervised test...")
    test_lat = pooled_latents(enc_pretrained, sup_test_subset, args.batch_size, device)

    train_modal = sup._y_modal[train_mask]
    test_modal = sup._y_modal[test_mask]

    knn_stats = knn_mineralogy_check(train_lat, train_modal, test_lat, test_modal)
    print(f"  cosine vs true modal: kNN={knn_stats['cosine_similarity_knn']:.3f}  "
          f"random={knn_stats['cosine_similarity_random']:.3f}  "
          f"delta={knn_stats['delta_vs_random']:.3f}")

    # ---- OOD distance distribution ----
    print("\nrunning OOD distance distribution...")
    rng2 = np.random.default_rng(123)
    n_held = min(2000, len(train_lat) // 4)
    held_idx = rng2.choice(len(train_lat), n_held, replace=False)
    held_train_lat = train_lat[held_idx]
    train_lat_for_ood = np.delete(train_lat, held_idx, axis=0)
    ood_stats = ood_distance_distribution(train_lat_for_ood, test_lat, held_train_lat)
    print(f"  threshold (99p of held train): {ood_stats['ood_flag_threshold_99p_train']:.3f}")
    print(f"  test bulks flagged as OOD: {ood_stats['ood_test_n_flagged']} "
          f"({ood_stats['ood_test_frac_flagged']:.3f})")
    ood_distance_plot(ood_stats, PLOT_DIR / "ood_distance.png")
    print("  saved ood_distance.png")

    summary = {
        "n_vis_unlabeled": int(len(vis_idx)),
        "knn_mineralogy_check": knn_stats,
        "ood_distance": {k: v for k, v in ood_stats.items()
                         if k not in ("test_distances", "held_train_distances")},
    }
    with open(PLOT_DIR / "knn_mineralogy.json", "w") as f:
        json.dump(knn_stats, f, indent=2)
    with open(PLOT_DIR / "ood_flags_summary.json", "w") as f:
        json.dump(summary["ood_distance"], f, indent=2)
    print(f"\nplots + json -> {PLOT_DIR}")


if __name__ == "__main__":
    main()

