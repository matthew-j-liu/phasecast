"""Where does a padded redox slot sit relative to the populated ones?

  python explorations/redox_padding/run_geometry.py
  python explorations/redox_padding/run_geometry.py --limit 200

Run from the repo root. Label-free, so it runs on every corpus bulk whose
source reported a real FeO + Fe2O3 pair (`fe_provenance == measured`), not
just the val roster -- nothing here is scored against a label and no split
discipline applies.

Each bulk is encoded once with the redox slot padded and once per ratio on a
fixed grid, tracing the curve the token sweeps out in latent space. The
padded latent is then located against that curve: how far the sweep moves the
representation at all, which ratio the padded state most resembles, whether
that ratio tracks the bulk's own measured value or sits at one global
default, and whether the leftover displacement is a constant offset (the
arithmetic of pooling over one fewer token) or something rock-specific.

Writes `results/geometry.json` and `results/geometry.png`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cnmineral.data.bulk_chemistry import _load_raw_dataframe, tokenize_bulk_frame
from cnmineral.data.tokens import REDOX_TOKEN_ID, REDOX_TOKEN_INDEX
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.training import pick_device

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"

ENCODER = "checkpoints/pretrain_allrock_redox_seed0/epoch_best.pt"
CORPUS_DIRS = ("data/corpus/train", "data/corpus/valtest")
GARD_CSV = "data/bulks/gard2019.csv"

# The grid spans the plausible range with room on both ends: the measured
# tier's own ratios run 0.04-0.96 (median 0.33), and the class lookup that
# covers the other 70% of the corpus tops out at 0.45.
RATIO_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Bulk pairs sampled for the two pairwise summaries (the between-rock
# distance scale and the displacement-direction cosine).  The full pairwise
# matrix over ~3000 bulks is affordable but both medians are stable long
# before that, and this keeps the run flat if the tier ever grows.
_SCALE_PAIRS = 20000


def measured_bulk_ids() -> np.ndarray:
    """Corpus bulk ids whose generator input came from a measured Fe pair.

    `derived_partial` is excluded deliberately: only 81 of its bulks derive a
    real ratio from a flavour plus a total, and the rest are a lone flavour
    split by the same class lookup `assumed_class` gets (journal 2026-08-06).
    The `measured` tier coincides exactly with the tokenizer's populated
    condition, so it is the one tier where corpus and encoder agree on what
    the slot means.
    """
    frames = []
    for corpus_dir in CORPUS_DIRS:
        files = sorted(glob.glob(str(REPO / corpus_dir / "chunk_*_draws.parquet")))
        if not files:
            raise FileNotFoundError(f"no draws tables under {corpus_dir}")
        frames += [
            pd.read_parquet(f, columns=["bulk", "fe_provenance"]) for f in files
        ]
    per_bulk = pd.concat(frames).groupby("bulk")["fe_provenance"].first()
    return np.sort(per_bulk.index[per_bulk == "measured"].to_numpy())


def bulk_tokens(bulk_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """23-token (src, frac) for the given gard sample ids, in that order."""
    gard = _load_raw_dataframe(REPO / GARD_CSV).set_index("sample_id")
    missing = np.setdiff1d(bulk_ids, gard.index.to_numpy())
    if missing.size:
        raise KeyError(f"{missing.size} corpus bulks absent from gard2019")
    src, frac = tokenize_bulk_frame(gard.loc[bulk_ids], include_redox=True)
    populated = src[:, REDOX_TOKEN_INDEX] == REDOX_TOKEN_ID
    if not populated.all():
        raise ValueError(
            f"{(~populated).sum()} measured-tier bulks tokenize to a padded "
            "redox slot; the tier no longer coincides with the tokenizer rule"
        )
    return torch.from_numpy(src), torch.from_numpy(frac)


@torch.no_grad()
def _encode(encoder, src, frac, device, batch_size=512) -> torch.Tensor:
    out = []
    for start in range(0, len(src), batch_size):
        out.append(
            encoder(
                src=src[start : start + batch_size].to(device),
                frac=frac[start : start + batch_size].to(device),
            ).cpu()
        )
    return torch.cat(out)


def sweep_latents(encoder, src, frac, device) -> tuple[np.ndarray, np.ndarray]:
    """Latents with the slot padded, and one per grid ratio.

    Returns `padded` [N, d] and `swept` [N, len(RATIO_GRID), d].
    """
    pad_src = src.clone()
    pad_frac = frac.clone()
    pad_src[:, REDOX_TOKEN_INDEX] = 0
    pad_frac[:, REDOX_TOKEN_INDEX] = 0.0
    padded = _encode(encoder, pad_src, pad_frac, device).numpy()

    swept = np.empty((len(src), len(RATIO_GRID), padded.shape[1]), dtype=np.float32)
    for j, r in enumerate(RATIO_GRID):
        r_frac = frac.clone()
        r_frac[:, REDOX_TOKEN_INDEX] = float(r)
        swept[:, j] = _encode(encoder, src, r_frac, device).numpy()
    return padded, swept


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float | None:
    """Correlation, or None when one side is constant.

    `json.dumps` writes a bare NaN, which is not valid JSON, so an
    undefined correlation has to become null rather than propagate.
    """
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return round(float(pd.Series(a).corr(pd.Series(b), method=method)), 4)


def _between_rock_scale(padded: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(padded)
    i = rng.integers(0, n, _SCALE_PAIRS)
    j = rng.integers(0, n, _SCALE_PAIRS)
    keep = i != j
    return float(np.median(np.linalg.norm(padded[i[keep]] - padded[j[keep]], axis=1)))


def geometry(padded: np.ndarray, swept: np.ndarray, r_true: np.ndarray) -> dict:
    # how far the token moves a rock, end to end and along the path
    span = np.linalg.norm(swept[:, -1] - swept[:, 0], axis=1)
    arc = np.linalg.norm(np.diff(swept, axis=1), axis=2).sum(axis=1)
    scale = _between_rock_scale(padded)

    # which populated ratio the padded state most resembles
    gaps = np.linalg.norm(swept - padded[:, None, :], axis=2)
    nearest = gaps.argmin(axis=1)
    r_star = RATIO_GRID[nearest]
    d_pad = gaps[np.arange(len(gaps)), nearest]

    # the leftover displacement: constant across rocks (pooling over one
    # fewer token) or rock-specific (something semantic)?
    disp = padded - swept[np.arange(len(swept)), nearest]
    unit = disp / np.clip(np.linalg.norm(disp, axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(0)
    a = rng.integers(0, len(unit), _SCALE_PAIRS)
    b = rng.integers(0, len(unit), _SCALE_PAIRS)
    keep = a != b
    cos = (unit[a[keep]] * unit[b[keep]]).sum(axis=1)

    # Padding drops a token from the pool, which moves the latent whatever
    # the ratio means -- so some of the off-curve distance is arithmetic,
    # not semantics.  One refinement pass: strip the displacement shared by
    # every bulk, then re-ask which ratio the remainder sits nearest.
    offset = disp.mean(axis=0)
    gaps_corr = np.linalg.norm(swept - (padded - offset)[:, None, :], axis=2)
    nearest_corr = gaps_corr.argmin(axis=1)
    r_star_corr = RATIO_GRID[nearest_corr]
    d_pad_corr = gaps_corr[np.arange(len(gaps_corr)), nearest_corr]

    q = lambda x: dict(zip(("p10", "p50", "p90"), np.round(np.quantile(x, [0.1, 0.5, 0.9]), 4)))
    return {
        "n_bulks": int(len(padded)),
        "between_rock_distance_median": round(scale, 4),
        "sweep_span": q(span),
        "sweep_span_over_between_rock": q(span / scale),
        "sweep_arc_length": q(arc),
        "sweep_arc_over_span": q(arc / np.clip(span, 1e-12, None)),
        "r_star": q(r_star),
        "r_star_mode": float(pd.Series(r_star).mode().iat[0]),
        "r_star_unique_fraction": round(float(len(np.unique(r_star)) / len(RATIO_GRID)), 3),
        "r_true": q(r_true),
        "r_star_vs_r_true_pearson": _corr(r_star, r_true, "pearson"),
        "r_star_vs_r_true_spearman": _corr(r_star, r_true, "spearman"),
        "padded_to_curve_distance": q(d_pad),
        "padded_to_curve_over_span": q(d_pad / np.clip(span, 1e-12, None)),
        "displacement_direction_cosine": q(cos),
        "displacement_direction_cosine_mean": round(float(cos.mean()), 4),
        "shared_offset_norm": round(float(np.linalg.norm(offset)), 4),
        "shared_offset_over_median_displacement": round(
            float(
                np.linalg.norm(offset)
                / max(float(np.median(np.linalg.norm(disp, axis=1))), 1e-12)
            ), 4,
        ),
        "offset_corrected": {
            "r_star": q(r_star_corr),
            "r_star_mode": float(pd.Series(r_star_corr).mode().iat[0]),
            "r_star_vs_r_true_pearson": _corr(r_star_corr, r_true, "pearson"),
            "r_star_vs_r_true_spearman": _corr(r_star_corr, r_true, "spearman"),
            "padded_to_curve_over_span": q(d_pad_corr / np.clip(span, 1e-12, None)),
        },
    }


def figure(padded, swept, r_true, r_star, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # left: the sweep as the encoder sees it, projected on the plane that
    # best captures the token's own effect (PCA of the swept displacements)
    centred = swept - swept.mean(axis=1, keepdims=True)
    flat = centred.reshape(-1, centred.shape[-1])
    _, _, vt = np.linalg.svd(flat[:: max(1, len(flat) // 20000)], full_matrices=False)
    basis = vt[:2].T
    ax = axes[0]
    show = np.linspace(0, len(swept) - 1, 8, dtype=int)
    for i in show:
        pts = (swept[i] - swept[i].mean(axis=0)) @ basis
        ax.plot(pts[:, 0], pts[:, 1], "-", color="0.75", lw=1, zorder=1)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=RATIO_GRID, cmap="viridis", s=18, zorder=2)
        p = (padded[i] - swept[i].mean(axis=0)) @ basis
        ax.scatter(p[0], p[1], marker="x", color="crimson", s=70, zorder=3)
    fig.colorbar(sc, ax=ax, label="Fe3+/total-Fe in the slot")
    ax.set_xlabel("direction 1 of the token's own effect (arb.)")
    ax.set_ylabel("direction 2 (arb.)")
    ax.set_title("Sweep vs padded: eight bulks\n(x = padded)")

    # right: does the padded state track the rock, or sit at one default?
    ax = axes[1]
    jitter = np.random.default_rng(0).normal(0, 0.006, len(r_star))
    ax.scatter(r_true, r_star + jitter, s=8, alpha=0.25, color="#1f5fa9")
    lim = [0, 1]
    ax.plot(lim, lim, "--", color="0.5", lw=1, label="padded imitates this rock's own ratio")
    ax.axhline(np.median(r_star), color="crimson", lw=1, label="median nearest ratio")
    ax.set_xlim(lim)
    ax.set_ylim(0, 1)
    ax.set_xlabel("measured Fe3+/total-Fe")
    ax.set_ylabel("nearest populated ratio to the padded latent")
    ax.set_title("What does the blank slot imitate?")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="score only the first N measured bulks (smoke runs)",
    )
    parser.add_argument("--encoder", default=ENCODER)
    args = parser.parse_args()

    bulk_ids = measured_bulk_ids()
    if args.limit:
        bulk_ids = bulk_ids[: args.limit]
    src, frac = bulk_tokens(bulk_ids)
    r_true = frac[:, REDOX_TOKEN_INDEX].numpy().copy()
    print(f"measured-tier bulks: {len(bulk_ids)}")

    device = pick_device()
    encoder, ckpt = load_encoder_from_checkpoint(REPO / args.encoder)
    encoder.to(device)
    print(f"encoder {args.encoder} (epoch {ckpt.epoch}) on {device}")

    # the invariant the production config rests on, checked against the
    # real weights before anything is read off the geometry
    narrow = _encode(encoder, src[:64, :REDOX_TOKEN_INDEX], frac[:64, :REDOX_TOKEN_INDEX], device)
    wide_src = src[:64].clone()
    wide_frac = frac[:64].clone()
    wide_src[:, REDOX_TOKEN_INDEX] = 0
    wide_frac[:, REDOX_TOKEN_INDEX] = 0.0
    wide = _encode(encoder, wide_src, wide_frac, device)
    max_gap = float((narrow - wide).abs().max())
    if max_gap > 1e-4:
        raise AssertionError(f"padded-23 differs from 22-token by {max_gap:.2e}")
    print(f"padded-23 == 22-token to {max_gap:.2e}")

    padded, swept = sweep_latents(encoder, src, frac, device)
    stats = geometry(padded, swept, r_true)
    stats["encoder"] = args.encoder
    stats["ratio_grid"] = RATIO_GRID.tolist()
    stats["padded_equals_22_token_max_abs"] = max_gap

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "geometry.json").write_text(json.dumps(stats, indent=1))
    gaps = np.linalg.norm(swept - padded[:, None, :], axis=2)
    figure(padded, swept, r_true, RATIO_GRID[gaps.argmin(axis=1)], RESULTS / "geometry.png")

    print(f"\nsweep span / between-rock distance: {stats['sweep_span_over_between_rock']}")
    print(f"nearest ratio to padded: {stats['r_star']} (mode {stats['r_star_mode']})")
    print(f"r_star vs r_true: pearson {stats['r_star_vs_r_true_pearson']}, "
          f"spearman {stats['r_star_vs_r_true_spearman']}")
    print(f"padded-to-curve / span: {stats['padded_to_curve_over_span']}")
    print(f"displacement cosine: mean {stats['displacement_direction_cosine_mean']}")
    corr = stats["offset_corrected"]
    print(f"after removing the shared offset: r_star {corr['r_star']} "
          f"(mode {corr['r_star_mode']}), vs r_true pearson "
          f"{corr['r_star_vs_r_true_pearson']}, distance/span "
          f"{corr['padded_to_curve_over_span']}")
    print(f"\nwrote {RESULTS / 'geometry.json'}")


if __name__ == "__main__":
    main()
