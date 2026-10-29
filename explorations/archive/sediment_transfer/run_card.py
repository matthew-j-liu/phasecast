"""The five real-rock headline figures, pointed at the Andrews XRD sediments.

Mirrors `scripts/view/real_rock_card.py` call for call so the pictures are
comparable to the eruption-v2 set; every plot function is the shipped one. Two
things differ, both forced by the source: the climatology null is fit on a
random carve-out rather than a unit-grouped one, and the cohort is all 605
rocks because no freshness grading exists.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval import real_rock
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import (
    climatology,
    mixture_share_interval,
    murphy,
    plot_cohort_card,
    plot_coverage_sharpness,
    plot_magnitude_ratio,
    plot_nll_distribution,
    plot_presence_brier,
    predict,
    real_rock_nll,
    share_interval,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint

from load_andrews import load_andrews, write_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_CHECKPOINT = "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt"


def src_stamp() -> dict:
    """Which code produced this. `src/` has had concurrent editors."""
    def run(*args):
        return subprocess.run(args, cwd=REPO, capture_output=True,
                              text=True).stdout
    diff = run("git", "diff", "--", "src/")
    return {
        "head": run("git", "rev-parse", "HEAD").strip(),
        "src_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:16],
        "src_dirty": bool(diff.strip()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    benchmark, info = load_andrews()
    # climatology reads the committed roster by path; point it at ours.
    real_rock.SPLIT_PATH = write_split(benchmark, HERE / "split.csv")

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)
    pred = predict(ckpt.build_mode_head().to(device), latents, benchmark)

    clean = benchmark.clean & benchmark.has_magnitude
    p7, alpha7 = pred["p7"], pred["alpha7"]
    numeric = benchmark.numeric.astype(np.float64)
    forced_share = conditional_mean(alpha7, numeric)
    model_share = conditional_mean(
        pred["alpha7_model_support"], pred["model_support7"].astype(np.float64)
    )
    nll = real_rock_nll(p7, alpha7, benchmark.presence, benchmark.magnitude,
                        benchmark.numeric)
    null = climatology(benchmark)
    null_nll = real_rock_nll(null["p7"], null["alpha7"], benchmark.presence,
                             benchmark.magnitude, benchmark.numeric)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    subtitle = (f"{int(clean.sum())} sedimentary rocks, XRD modes, all grades "
                f"(no freshness grading exists)")

    # the two interval figures need the caveat on the figure, not just in a
    # sidecar: no locality metadata means the unit-clustered bootstrap
    # degenerates to a per-rock one. `run_bootstrap_width.py` prices that.
    width = out_dir / "bootstrap_width.json"
    ratio = (json.loads(width.read_text())["summary_ratio_mean"]
             if width.exists() else None)
    interval_subtitle = subtitle + (
        f"; per-rock bootstrap (no units to cluster on), "
        f"~{ratio:.1f}x too narrow on eruption-v2" if ratio else
        "; per-rock bootstrap, not comparable to the eruption benchmark's"
    )

    plot_cohort_card(
        p7[clean], forced_share[clean], model_share[clean],
        pred["model_support7"][clean], benchmark.subset(clean),
        out_dir / "cohort_card.png",
    )
    plot_magnitude_ratio(
        forced_share[clean], benchmark.magnitude[clean],
        benchmark.numeric[clean], out_dir / "magnitude_ratio.png", subtitle,
    )
    plot_nll_distribution(
        {k: v[clean] for k, v in nll.items()},
        {k: v[clean] for k, v in null_nll.items()},
        out_dir / "nll_distribution.png", subtitle,
    )
    plot_presence_brier(
        p7[clean], null["p7"][clean], benchmark.presence[clean],
        benchmark.unit[clean], out_dir / "presence_brier.png", interval_subtitle,
    )
    if log_w is None:
        lo, hi = share_interval(alpha7, numeric)
    else:
        lo, hi = mixture_share_interval(alpha7, log_w, numeric)
    null_lo, null_hi = share_interval(null["alpha7"], numeric)
    plot_coverage_sharpness(
        {"model": (lo[clean], hi[clean]),
         "climatology": (null_lo[clean], null_hi[clean])},
        benchmark.magnitude[clean], benchmark.numeric[clean],
        benchmark.unit[clean], out_dir / "coverage_sharpness.png",
        interval_subtitle,
    )

    # the shipped Brier panel averages reliability / resolution over all seven
    # groups. Four of the seven have zero presence variance here (fsp and q in
    # every rock, ol and bi in none), so the resolution term is uninformative
    # for model and null alike; the five-group version is recorded beside it.
    y = (benchmark.presence[clean] > 0).astype(float)
    scoreable = [i for i, g in enumerate(GROUPS) if g not in ("ol", "bi")]
    murphy_terms = {}
    for label, probs in (("model", p7[clean]), ("climatology", null["p7"][clean])):
        allg = np.array([murphy(probs[:, i], y[:, i]) for i in range(len(GROUPS))])
        murphy_terms[label] = {
            "all7": {"reliability": float(allg[:, 0].mean()),
                     "resolution": float(allg[:, 1].mean())},
            "scoreable5": {"reliability": float(allg[scoreable, 0].mean()),
                           "resolution": float(allg[scoreable, 1].mean())},
        }

    off = info["off_vocab_share"]
    manifest = {
        "checkpoint": str(args.checkpoint),
        "code": src_stamp(),
        "n_rocks": int(clean.sum()),
        "headline_nll": {
            term: {"model": float(nll[term][clean].mean()),
                   "climatology": float(null_nll[term][clean].mean())}
            for term in ("bernoulli", "dirichlet", "joint")
        },
        "murphy": murphy_terms,
        "bootstrap": {"kind": "per-rock (no units available)",
                      "eruption_v2_width_ratio": ratio},
        "climatology_concentration": null["concentration"],
        "climatology_n_fit": null["n_fit"],
        "truth": {
            "groups_absent": info["groups_absent_from_truth"],
            "presence_base_rate": {
                g: float((benchmark.presence[clean][:, i] > 0).mean())
                for i, g in enumerate(GROUPS)
            },
            "off_vocab_share_p5_50_95": [
                float(np.percentile(off, q)) for q in (5, 50, 95)
            ],
            "dropped_minerals": info["dropped_minerals"],
        },
        "inputs": {
            "populated_tokens": int((benchmark.bulk_src.numpy() > 0).sum(1)[0]),
            "elements_with_no_token": info["unused_elements"],
        },
        "caveats": [
            "provenance unrecovered: anonymised Rock_N, no locality or source "
            "publication identified, so these are not reportable numbers",
            "error bars are a per-rock bootstrap, not the unit-clustered one "
            "the eruption benchmark uses; see bootstrap_width.json",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"figures -> {out_dir}")
    for term in ("bernoulli", "dirichlet", "joint"):
        m, c = manifest["headline_nll"][term].values()
        print(f"  {term:10s} model {m:7.3f}   climatology {c:7.3f}   "
              f"delta {m - c:+.3f}")


if __name__ == "__main__":
    main()
