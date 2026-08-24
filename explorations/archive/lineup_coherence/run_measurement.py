"""Lineup-coherence measurement for the scenario sampler (#47 step 0).

`sample_assemblages` draws each phase's Bernoulli independently, so it can
compose support patterns the corpus never pairs. This script computes, per
val-roster bulk, the exact probability mass the sampler puts on patterns
outside the bulk's own truth draws and outside the corpus-observed pattern
library -- closed-form Bernoulli products, no Monte-Carlo noise -- plus the
mitigation's cost side: the fraction of truth draws whose pattern is absent
from the train-tier library.

    conda run -n mineral python explorations/lineup_coherence/run_measurement.py \\
        --checkpoint checkpoints/joint_heads_supportmag_seed0/heads_best.pt
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.data.mineralogy_labels import hurdle_targets, load_joint_labels
from cnmineral.data.phase_vocab import SOLID_PHASES
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import _qap_labels
from cnmineral.training.joint_heads import JointHeadCheckpoint

RESULTS_DIR = Path(__file__).resolve().parent / "results"

AMBIGUITY_BINS = ((0, 2), (2, 4), (4, 6), (6, 22))
P_INTERMEDIATE = (0.1, 0.9)
LOG_EPS = 1e-12


def pattern_codes(presence: np.ndarray) -> np.ndarray:
    """Bit-packed integer code per support pattern ([N, P] bool -> [N] int64)."""
    bits = 1 << np.arange(presence.shape[1], dtype=np.int64)
    return presence.astype(np.int64) @ bits


def codes_to_patterns(codes: np.ndarray, n_phases: int) -> np.ndarray:
    """Inverse of `pattern_codes` ([U] int64 -> [U, P] uint8)."""
    bits = 1 << np.arange(n_phases, dtype=np.int64)
    return ((codes[:, None] & bits) > 0).astype(np.uint8)


def library_mass(codes: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Exact per-bulk sampler mass on a pattern library, empty-support-rejected.

    `codes` are unique pattern codes, `p` [B, P] presence probabilities.
    The deployed sampler rejects the all-absent draw, so masses are
    renormalized by 1 - prod(1 - p). Returns [B].
    """
    Z = codes_to_patterns(np.unique(codes), p.shape[1]).astype(np.float64)
    logp = np.log(np.clip(p, LOG_EPS, 1.0))
    log1mp = np.log(np.clip(1.0 - p, LOG_EPS, 1.0))
    log_mass = Z @ logp.T + (1.0 - Z) @ log1mp.T
    norm = 1.0 - np.exp(log1mp.sum(1))
    return np.exp(log_mass).sum(0) / norm


def sample_support_codes(p: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` pattern codes from the deployed Bernoulli + empty-rejection scheme."""
    draws = rng.random((n, len(p))) < p
    for _ in range(100):
        empty = ~draws.any(1)
        if not empty.any():
            break
        draws[empty] = rng.random((int(empty.sum()), len(p))) < p
    return pattern_codes(draws)


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to a joint-heads checkpoint (.pt)")
    parser.add_argument("--train-corpus-dir", type=Path, default=None,
                        help="train-tier corpus dir (default: the checkpoint's)")
    parser.add_argument("--n-mc", type=int, default=2000,
                        help="Monte-Carlo support draws per bulk for the cross-check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    lc = ckpt.label_config
    mode_head = ckpt.build_mode_head()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.eval()

    ds = JointHeadDataset(
        corpus_dir=lc["valtest_corpus_dir"],
        rungs=tuple(lc["rungs"]),
        presence_cutoff=lc["presence_cutoff"],
        comp_eps=lc["comp_eps"],
        bulk_ids=load_valtest_split("val"),
    )
    latents = bulk_latents(encoder, ds, torch.device("cpu"))
    with torch.no_grad():
        p = torch.sigmoid(mode_head.presence_logits(latents)).numpy().astype(np.float64)

    presence = ds.presence.numpy() > 0
    row_bulk = ds.row_bulk_idx.numpy()
    n_bulks = len(ds.bulk_ids)
    val_codes = pattern_codes(presence)

    train_dir = args.train_corpus_dir or lc["corpus_dir"]
    print(f"loading train-tier labels from {train_dir} ...")
    train_labels = load_joint_labels(train_dir, tuple(lc["rungs"]), lc["comp_eps"])
    train_presence, _ = hurdle_targets(train_labels["modes"], lc["presence_cutoff"])
    train_codes = np.unique(pattern_codes(train_presence > 0))
    union_codes = np.union1d(train_codes, val_codes)
    print(f"library sizes: train {len(train_codes)} unique patterns, "
          f"val truth {len(np.unique(val_codes))}, union {len(union_codes)}")

    on_train = library_mass(train_codes, p)
    on_union = library_mass(union_codes, p)
    off_own = np.zeros(n_bulks)
    truth_excluded = np.zeros(n_bulks)
    rng = np.random.default_rng(args.seed)
    mc_dev = np.zeros(n_bulks)
    for b in range(n_bulks):
        own = val_codes[row_bulk == b]
        off_own[b] = 1.0 - library_mass(own, p[b : b + 1])[0]
        truth_excluded[b] = float((~np.isin(own, train_codes)).mean())
        mc_codes = sample_support_codes(p[b], args.n_mc, rng)
        mc_dev[b] = abs(float((~np.isin(mc_codes, own)).mean()) - off_own[b])
    off_train = 1.0 - on_train
    off_union = 1.0 - on_union

    assert np.all((off_own > -1e-9) & (off_own < 1 + 1e-9))
    assert np.all((off_train > -1e-9) & (off_train < 1 + 1e-9))
    # a bulk's own truth patterns are a subset of the union library
    assert np.all(off_union <= off_own + 1e-9)

    n_intermediate = ((p > P_INTERMEDIATE[0]) & (p < P_INTERMEDIATE[1])).sum(1)
    qap = _qap_labels(ds.bulk_qap_names)
    by_qap = {}
    for cls in np.unique(qap):
        m = qap == cls
        by_qap[str(cls)] = {
            "n": int(m.sum()),
            "off_own_mean": float(off_own[m].mean()),
            "off_train_mean": float(off_train[m].mean()),
        }
    by_ambiguity = {}
    for lo, hi in AMBIGUITY_BINS:
        m = (n_intermediate >= lo) & (n_intermediate < hi)
        if m.any():
            by_ambiguity[f"[{lo},{hi})"] = {
                "n": int(m.sum()),
                "off_own_mean": float(off_own[m].mean()),
                "off_train_mean": float(off_train[m].mean()),
            }
    ambiguity_r = float(np.corrcoef(n_intermediate, off_own)[0, 1])

    mc_binom = float(np.sqrt(0.25 / args.n_mc))
    print(f"\noff-own-menu mass:   mean {off_own.mean():.3f}  median "
          f"{np.median(off_own):.3f}  p90 {np.quantile(off_own, 0.9):.3f}  "
          f"max {off_own.max():.3f}")
    print(f"off-train-library:   mean {off_train.mean():.3f}  median "
          f"{np.median(off_train):.3f}  p90 {np.quantile(off_train, 0.9):.3f}  "
          f"max {off_train.max():.3f}")
    print(f"off-union (corpus-wide): mean {off_union.mean():.3f}  "
          f"max {off_union.max():.3f}")
    print(f"truth draws outside train library: mean {truth_excluded.mean():.3f}  "
          f"max {truth_excluded.max():.3f}")
    print(f"ambiguity: r(off_own, n intermediate p) = {ambiguity_r:.2f}")
    print("by ambiguity:", {k: round(v["off_own_mean"], 3)
                            for k, v in by_ambiguity.items()})
    print(f"mc cross-check ({args.n_mc} draws): max |mc - exact| = "
          f"{mc_dev.max():.4f} (binomial scale ~{mc_binom:.4f})")

    metrics = {
        "checkpoint": str(args.checkpoint),
        "train_corpus_dir": str(train_dir),
        "n_val_bulks": n_bulks,
        "n_phases": len(SOLID_PHASES),
        "presence_cutoff": lc["presence_cutoff"],
        "library_sizes": {
            "train": int(len(train_codes)),
            "val_truth": int(len(np.unique(val_codes))),
            "union": int(len(union_codes)),
        },
        "off_own_menu_mass": summarize(off_own),
        "off_train_library_mass": summarize(off_train),
        "off_union_library_mass": summarize(off_union),
        "truth_mass_excluded_by_train_library": summarize(truth_excluded),
        "by_qap": by_qap,
        "by_ambiguity": by_ambiguity,
        "ambiguity_pearson_r": ambiguity_r,
        "mc_check": {"n_draws": args.n_mc, "seed": args.seed,
                     "max_abs_deviation": float(mc_dev.max()),
                     "binomial_scale": mc_binom},
        "per_bulk": [
            {
                "bulk": int(ds.bulk_ids[b]),
                "qap": str(qap[b]),
                "n_intermediate_p": int(n_intermediate[b]),
                "off_own": float(off_own[b]),
                "off_train": float(off_train[b]),
                "off_union": float(off_union[b]),
                "truth_excluded": float(truth_excluded[b]),
            }
            for b in range(n_bulks)
        ],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "lineup_coherence.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics -> {out}")


if __name__ == "__main__":
    main()
