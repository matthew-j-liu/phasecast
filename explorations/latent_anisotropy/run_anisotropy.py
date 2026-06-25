"""Latent anisotropy of the canonical encoder -- both facets, for the record.

`input_knn_ceiling` argued the weak cosine-kNN purity is "not an anisotropy
artifact" only indirectly (the same gap exists in the raw input).  This measures
the latent's anisotropy head-on so the claim rests on a direct number.

Two independent ways a distribution departs from isotropy (looking the same in
every direction):

- first-order (mean / cone): the cloud sits off-origin in a narrow cone -- a
  shared dominant direction.  Seen by cosine, removed by centering.  Scored by
  the mean random-pair cosine and the mean-vector dominance.
- second-order (shape): the covariance ellipsoid is elongated -- variance spread
  unevenly across directions.  Computed on centered data, blind to the mean.
  Scored by the covariance spectrum's effective rank / participation ratio;
  dimensional collapse is its extreme tail.

The canonical latent is anchored against its own raw-input representations
(`raw_frac`, `std_log`, reused from `input_knn_ceiling`) and an isotropic
Gaussian reference at matched N, d.

Run (module form so the cross-exploration import resolves from repo root):
    conda run -n alignn_fix python -m explorations.latent_anisotropy.run_anisotropy \
        --checkpoint checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
from pathlib import Path
from time import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cnmineral.eval.latent import extract_latents, load_encoder_from_checkpoint
from cnmineral.training import create_dataset, pick_device, reserve_test
from cnmineral.training.checkpoint import checkpoint_tag
from explorations.input_knn_ceiling.run_ceiling import build_raw_representations

RESULTS_DIR = Path(__file__).resolve().parent / "results"

_EPS = 1e-12
_HIST_PAIRS = 1_000_000  # random pairs sampled for the cosine histogram only
_SEED = 0
_DEFAULT_CHECKPOINT = "checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt"


def first_order_anisotropy(X: np.ndarray) -> dict[str, float]:
    """Mean random-pair cosine (closed form) and mean-vector dominance.

    The mean over all ordered i!=j pairs of `cos(x_i, x_j)` equals
    `(N*||u_bar||^2 - 1)/(N-1)` where `u_bar` is the mean of the L2-normalized
    rows -- no pair sampling needed for the scalar.
    """
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1)
    units = X / np.maximum(norms, _EPS)[:, None]
    u_bar = units.mean(axis=0)
    mean_cos = (n * float(u_bar @ u_bar) - 1.0) / (n - 1)
    mean_vec_dominance = float(np.linalg.norm(X.mean(axis=0)) / norms.mean())
    return {"mean_pair_cosine": mean_cos, "mean_vector_dominance": mean_vec_dominance}


def second_order_anisotropy(X: np.ndarray) -> dict:
    """Covariance-spectrum shape metrics on centered `X`.

    `effective_rank` is the Roy-Vetterli entropy rank `exp(-sum p ln p)` over the
    normalized eigenvalues; `participation_ratio` is `(sum lam)^2 / sum(lam^2)`.
    Both live in `[1, d]`; their `_frac` versions divide by `d` for cross-rep
    comparison (the latent is 256-d, the raw reps 22-d).
    """
    d = X.shape[1]
    Xc = X - X.mean(axis=0)
    sv = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lam = sv**2 / (X.shape[0] - 1)
    evr = lam / lam.sum()
    p = evr[evr > 0]
    effective_rank = float(np.exp(-(p * np.log(p)).sum()))
    participation_ratio = float(lam.sum() ** 2 / (lam**2).sum())
    cum = np.cumsum(evr)
    return {
        "dim": int(d),
        "effective_rank": effective_rank,
        "effective_rank_frac": effective_rank / d,
        "participation_ratio": participation_ratio,
        "participation_ratio_frac": participation_ratio / d,
        "top1_var_frac": float(evr[0]),
        "top10_var_frac": float(cum[min(10, d) - 1]),
        "explained_variance_ratio": evr.tolist(),
    }


def _sample_pair_cosines(
    X: np.ndarray, n_pairs: int, rng: np.random.Generator,
) -> np.ndarray:
    n = X.shape[0]
    units = X / np.maximum(np.linalg.norm(X, axis=1), _EPS)[:, None]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    return np.einsum("ij,ij->i", units[i[keep]], units[j[keep]])


def run_anisotropy(
    checkpoint_path: Path, *, device: torch.device | None = None,
) -> dict:
    """Compute first- and second-order anisotropy and write `results/<tag>/`."""
    device = device or pick_device()
    tag = checkpoint_tag(checkpoint_path)
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")

    print("\nloading encoder (config comes from ckpt.config)...")
    encoder, ckpt = load_encoder_from_checkpoint(checkpoint_path)
    config = ckpt.config

    print("building dataset + reproducing test split...")
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    train_idx = list(splits["trainval"])
    test_idx = list(splits["test"])
    print(f"  n_test={len(test_idx)} (trainval={len(train_idx)})")

    print("stacking raw-input representations...")
    reps = build_raw_representations(dataset, test_idx, train_idx)

    print("extracting latents over the test split...")
    t0 = time()
    reps["latent"] = extract_latents(
        encoder, dataset, indices=test_idx, device=device,
    ).numpy()
    print(f"  latent {reps['latent'].shape} in {time() - t0:.1f}s")

    rng = np.random.default_rng(_SEED)
    n, d = reps["latent"].shape
    reps["isotropic_ref"] = rng.standard_normal((n, d)).astype(np.float32)

    spaces = ("latent", "raw_frac", "std_log", "isotropic_ref")
    metrics: dict[str, dict] = {}
    for space in spaces:
        X = reps[space]
        entry = {
            "first_order": first_order_anisotropy(X),
            "first_order_centered": first_order_anisotropy(X - X.mean(axis=0)),
            "second_order": second_order_anisotropy(X),
        }
        metrics[space] = entry

    _print_table(metrics, spaces)
    _plot_spectrum(metrics, spaces, out_dir / "spectrum.png")
    _plot_cosine_hist(reps["latent"], reps["isotropic_ref"], rng, out_dir / "cosine_hist.png")

    payload = {
        "checkpoint": str(checkpoint_path),
        "tag": tag,
        "device": str(device),
        "n_test": n,
        "latent_dim": d,
        "seed": _SEED,
        "metrics": metrics,
    }
    out_path = out_dir / "anisotropy_metrics.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nmetrics -> {out_path}")
    return payload


def _print_table(metrics: dict, spaces: tuple[str, ...]) -> None:
    print("\nfirst-order (mean / cone) anisotropy")
    print(f"  {'space':<16}{'pair_cos':<12}{'pair_cos(ctr)':<16}{'mean_vec_dom':<14}")
    for space in spaces:
        fo = metrics[space]["first_order"]
        foc = metrics[space]["first_order_centered"]
        print(
            f"  {space:<16}{fo['mean_pair_cosine']:<12.4f}"
            f"{foc['mean_pair_cosine']:<16.4f}{fo['mean_vector_dominance']:<14.4f}"
        )
    print("\nsecond-order (covariance shape) anisotropy")
    print(f"  {'space':<16}{'eff_rank':<12}{'eff_rank/d':<12}{'top1':<10}{'top10':<10}")
    for space in spaces:
        so = metrics[space]["second_order"]
        print(
            f"  {space:<16}{so['effective_rank']:<12.2f}{so['effective_rank_frac']:<12.4f}"
            f"{so['top1_var_frac']:<10.4f}{so['top10_var_frac']:<10.4f}"
        )


def _plot_spectrum(metrics: dict, spaces: tuple[str, ...], path: Path) -> None:
    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(11, 4.2))
    for space in spaces:
        evr = np.asarray(metrics[space]["second_order"]["explained_variance_ratio"])
        x = np.arange(1, len(evr) + 1) / len(evr)
        ax_s.plot(x, evr, label=space)
        ax_c.plot(x, np.cumsum(evr), label=space)
    ax_s.set_yscale("log")
    ax_s.set_xlabel("component index / d")
    ax_s.set_ylabel("explained variance ratio")
    ax_s.set_title("covariance spectrum (log)")
    ax_c.set_xlabel("component index / d")
    ax_c.set_ylabel("cumulative explained variance")
    ax_c.set_title("cumulative")
    ax_c.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_cosine_hist(
    latent: np.ndarray,
    isotropic: np.ndarray,
    rng: np.random.Generator,
    path: Path,
) -> None:
    series = {
        "latent": _sample_pair_cosines(latent, _HIST_PAIRS, rng),
        "latent (centered)": _sample_pair_cosines(
            latent - latent.mean(axis=0), _HIST_PAIRS, rng
        ),
        "isotropic ref": _sample_pair_cosines(isotropic, _HIST_PAIRS, rng),
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(-1, 1, 200)
    for name, cos in series.items():
        ax.hist(cos, bins=bins, density=True, histtype="step", label=name)
    ax.axvline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("random-pair cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("first-order anisotropy (cone)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(_DEFAULT_CHECKPOINT),
        help="encoder checkpoint (defaults to the canonical log-MSE encoder)",
    )
    args = parser.parse_args()
    run_anisotropy(checkpoint_path=args.checkpoint.resolve())


if __name__ == "__main__":
    main()
