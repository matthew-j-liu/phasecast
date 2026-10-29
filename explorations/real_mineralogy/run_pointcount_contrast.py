"""How much of the sim-to-real gap is the model, and how much is the ruler.

70 du Bray rocks carry two modal measurements of the same thin section: the
visual phenocryst estimate every du Bray rock has, and the microscope point
count in DS 1099's Appendix 10.  Every real-rock verdict on record was scored
against the first.  This scores the frozen production stack against both and
reads the difference, which is the part of the gap that belongs to the
measurement rather than to the model.

Three things make the comparison honest:

  - Appendix 10 tallies five of the seven scored groups and never looks for
    pyroxene or olivine, so the visual arm is restricted to the same five
    before anything is compared.  Otherwise the point-count arm is charged for
    absences nobody measured, and the two Dirichlet terms live on
    sub-compositions of different size.
  - `predict` teacher-forces the magnitude support from the truth presence, so
    changing the ruler changes the conditioning as well as the target.  These
    are two full re-scorings, and the presence and magnitude terms are reported
    apart so the delta is not attributed wholesale to sizes.
  - The 70 rocks are two stratigraphic units, so the unit-clustered interval
    every other contrast in this project reports is close to vacuous here.  It
    is printed for consistency and the paired per-rock distribution is the read.

(A last section once asked whether the rocks the tailed recalibrator's
contamination component claimed were the rocks the geologist eyeballed worst:
Spearman +0.22, journal 2026-08-06.  It went with the tier on 2026-09-03.)

    conda run -n mineral python explorations/real_mineralogy/run_pointcount_contrast.py \
        --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt
"""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cnmineral.data.phase_vocab import GROUP_INDEX
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import (
    BENCHMARK_DIR,
    POINTCOUNT_BASIS,
    VISUAL_BASIS,
    load_benchmark,
    preferred_basis,
)
from cnmineral.eval.summary_cards import LEGEND_FS, SUPTITLE_FS, TICK_FS, TITLE_FS
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    paired_bootstrap,
    score_recalibrator,
)

RESULTS = Path(__file__).resolve().parent / "results"

# Appendix 10's counting scheme, as the groups it resolves. Pyroxene and
# olivine are absent from the scheme, so a rock's silence about them is not a
# measured absence and both arms drop them before scoring.
COVERED_GROUPS: tuple[str, ...] = ("fsp", "amp", "bi", "opq", "q")
UNCOVERED_PHASES: tuple[str, ...] = ("pyroxene", "olivine")


def _keys(frame: pd.DataFrame) -> pd.Series:
    return frame["source"] + "|" + frame["sample_id"].astype(str)


def paired_keys() -> set[str]:
    """The rocks measured both ways, keyed as `load_benchmark` keys them."""
    modes = pd.read_csv(BENCHMARK_DIR / "modes.csv")
    counts = modes.assign(key=_keys(modes)).groupby("key")["basis"].nunique()
    return set(counts[counts > 1].index)


def restricted_benchmark_dir(out: Path, paired: set[str], basis: str) -> Path:
    """A benchmark whose paired rocks report only what Appendix 10 counts,
    each rock read on `basis` where it carries both rulers.

    Written to disk and loaded back through the shipped loader rather than
    edited in memory, so the restriction goes through the same renormalization,
    censoring and support-forcing path as everything else.
    """
    samples = pd.read_csv(BENCHMARK_DIR / "samples.csv")
    modes = pd.read_csv(BENCHMARK_DIR / "modes.csv")
    drop = _keys(modes).isin(paired) & modes["phase"].isin(UNCOVERED_PHASES)
    kept = modes[~drop].assign(key=_keys(modes[~drop]))
    out.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out / "samples.csv", index=False)
    preferred_basis(kept, basis).drop(columns="key").to_csv(out / "modes.csv", index=False)
    return out


def section_totals() -> pd.Series:
    """Points counted per rock, keyed like the benchmark."""
    samples = pd.read_csv(BENCHMARK_DIR / "samples.csv")
    return samples.set_index(_keys(samples))["n_points"]


def parent_unit(unit: np.ndarray) -> np.ndarray:
    """The eruption a paired rock came from, not the string the source typed.

    Appendix 10 covers two ignimbrites, and the benchmark's unit field spells
    them nine ways -- a missing space, a trailing space, a query, and sub-facies
    suffixes like `, pumice`.  A cluster bootstrap over those nine would treat
    one eruption's rocks as four independent draws and report an interval far
    narrower than the data supports.

    The merge is deliberately local to this contrast: the same fragmentation
    exists benchmark-wide, but fixing it there moves committed split
    assignments, which is a separate decision.  Every merge this makes is
    printed so it can be checked by eye.
    """
    stem = pd.Series(unit).str.replace(r"\s+", " ", regex=True).str.strip()
    stem = stem.str.split(",").str[0].str.rstrip(" ?")
    stem = stem.str.replace("FractionTuff", "Fraction Tuff")
    stem = stem.str.replace(r"\s*\(\?\)", "", regex=True).str.strip()
    return stem.to_numpy()


def truth_contrast(visual, counted, mask: np.ndarray) -> list[dict]:
    """Per-group disagreement between the two rulers, with no model involved.

    This is the measurement floor: how far apart two readings of the same thin
    section land before anything is predicted.
    """
    rows = []
    for group in COVERED_GROUPS:
        g = GROUP_INDEX[group]
        pv, pc = visual.presence[mask, g] > 0, counted.presence[mask, g] > 0
        both = visual.numeric[mask, g] & counted.numeric[mask, g]
        sv, sc = visual.magnitude[mask, g], counted.magnitude[mask, g]
        rows.append({
            "group": group,
            "present_visual": int(pv.sum()),
            "present_pointcount": int(pc.sum()),
            "presence_agree": float((pv == pc).mean()),
            "n_quantified_both": int(both.sum()),
            "median_share_visual": float(np.median(sv[both])) if both.any() else None,
            "median_share_pointcount": float(np.median(sc[both])) if both.any() else None,
            "median_signed_diff": float(np.median(sv[both] - sc[both])) if both.any() else None,
        })
    return rows


def counting_noise(counted, mask: np.ndarray, n_points: np.ndarray) -> dict:
    """The binomial floor a point count carries by construction.

    A count of `n` points estimates a share with a standard error that shrinks
    only as `sqrt(n)`, so part of any residual against point-counted truth is
    the count itself.  Reported as the median per-group standard error in share
    units, which is directly comparable to the disagreement above.
    """
    share = counted.magnitude[mask]
    n = n_points[:, None]
    se = np.sqrt(np.clip(share * (1 - share), 0, None) / n)
    observed = counted.numeric[mask]
    return {
        "median_points_counted": float(np.median(n_points)),
        "median_share_se": {
            group: float(np.median(se[observed[:, GROUP_INDEX[group]],
                                      GROUP_INDEX[group]]))
            for group in COVERED_GROUPS
            if observed[:, GROUP_INDEX[group]].any()
        },
    }


def total_variation(visual, counted, mask: np.ndarray) -> np.ndarray:
    """Per-rock distance between the two truth vectors, in share units.

    Half the summed absolute difference, so 0 is identical readings and 1 is
    disjoint ones -- one number per rock to correlate the model's behaviour
    against.
    """
    idx = [GROUP_INDEX[g] for g in COVERED_GROUPS]
    return 0.5 * np.abs(visual.magnitude[mask][:, idx]
                        - counted.magnitude[mask][:, idx]).sum(1)


def plot_contrast(rows, per_rock, out_path: Path, subtitle: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

    ax = axes[0]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["median_share_visual"] for r in rows], 0.4,
           color="#d9622b", label="visual estimate")
    ax.bar(x + 0.2, [r["median_share_pointcount"] for r in rows], 0.4,
           color="#2a78d6", label="point count")
    ax.set_xticks(x, [r["group"] for r in rows], fontsize=TICK_FS)
    ax.set_ylabel("median share of crystal cargo", fontsize=TICK_FS)
    ax.set_title("what the two rulers report", fontsize=TITLE_FS)
    ax.legend(fontsize=LEGEND_FS)

    ax = axes[1]
    lo = min(per_rock["nll_visual"].min(), per_rock["nll_pointcount"].min())
    hi = max(per_rock["nll_visual"].max(), per_rock["nll_pointcount"].max())
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=1)
    ax.scatter(per_rock["nll_visual"], per_rock["nll_pointcount"], s=22,
               color="#2a78d6", alpha=0.75)
    ax.set_xlabel("joint NLL vs visual truth", fontsize=TICK_FS)
    ax.set_ylabel("joint NLL vs point count", fontsize=TICK_FS)
    ax.set_title("same rock, same model, two rulers", fontsize=TITLE_FS)

    fig.suptitle(subtitle, fontsize=SUPTITLE_FS)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    paired = paired_keys()
    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    head = ckpt.build_mode_head().to(device)
    encoder = encoder.to(device)

    with tempfile.TemporaryDirectory() as tmp:
        visual = load_benchmark(
            restricted_benchmark_dir(Path(tmp) / "visual", paired, VISUAL_BASIS))
        counted = load_benchmark(
            restricted_benchmark_dir(Path(tmp) / "counted", paired, POINTCOUNT_BASIS))
    full = load_benchmark()

    # restricting the mode rows must not change which rocks are scoreable, or
    # the two arms would be answering on different populations
    assert np.array_equal(visual.bulk_ids, counted.bulk_ids)
    assert np.array_equal(visual.bulk_ids, full.bulk_ids)
    assert torch.equal(visual.bulk_frac, counted.bulk_frac)
    latents = bulk_latents(encoder, visual, device)

    mask = (
        np.isin(visual.bulk_ids, sorted(paired))
        & visual.clean & visual.has_magnitude
        & counted.clean & counted.has_magnitude
    )
    n_points = section_totals().reindex(visual.bulk_ids[mask]).to_numpy(dtype=float)

    print(f"paired rocks {int(mask.sum())} across "
          f"{len(np.unique(visual.unit[mask]))} unit strings / "
          f"{len(np.unique(parent_unit(visual.unit[mask])))} eruptions, "
          f"median {np.median(n_points):.0f} points counted")

    gp_visual = build_group_predictions(head, visual, latents).subset(mask)
    gp_counted = build_group_predictions(head, counted, latents).subset(mask)

    zero_visual = score_recalibrator(None, gp_visual)
    zero_counted = score_recalibrator(None, gp_counted)
    print("\nzero-shot on the paired rocks (joint / dirichlet / bernoulli):")
    for name, s in (("visual", zero_visual), ("pointcount", zero_counted)):
        print(f"  {name:11s} {s['joint'].mean():7.3f} "
              f"{s['dirichlet'].mean():7.3f} {s['bernoulli'].mean():7.3f} "
              f"(composition size {s['mean_composition_size']:.2f})")

    parent = parent_unit(gp_visual.unit)
    merges = pd.DataFrame({"parent": parent, "raw": gp_visual.unit})
    print("\nunit merge for the cluster bootstrap:")
    for name, group in merges.groupby("parent"):
        print(f"  {name}: {sorted(set(group['raw']))}")

    contrasts = {}
    for term in ("joint", "dirichlet", "bernoulli"):
        boot = paired_bootstrap(zero_visual[term], zero_counted[term], parent)
        delta = zero_visual[term] - zero_counted[term]
        contrasts[f"visual_minus_pointcount_{term}"] = {
            **boot,
            "median_delta": float(np.median(delta)),
            "frac_pointcount_better": float((delta > 0).mean()),
        }
        print(f"  visual minus point count, {term:10s} mean {boot['mean']:+.3f} "
              f"[{boot['lo']:+.3f}, {boot['hi']:+.3f}] over {boot['n_units']} "
              f"eruptions; median {np.median(delta):+.3f}, "
              f"{(delta > 0).mean():.0%} of rocks fit the point count better")
    # Two clusters admit three resamples, so the interval above is degenerate
    # rather than conservative -- it can come out narrower than the per-rock
    # spread justifies.  The paired median and win rate are the read.
    print(f"  ({len(np.unique(parent))} eruptions: the cluster interval is "
          f"reported for form, not for inference)")

    rows = truth_contrast(visual, counted, mask)
    print("\nper-group truth disagreement:")
    for r in rows:
        print(f"  {r['group']:4s} present {r['present_visual']:3d} visual vs "
              f"{r['present_pointcount']:3d} counted "
              f"(agree {r['presence_agree']:.2f})   median share "
              f"{r['median_share_visual']:.3f} vs {r['median_share_pointcount']:.3f}")

    noise = counting_noise(counted, mask, n_points)
    print(f"\nbinomial floor at {noise['median_points_counted']:.0f} points: "
          + ", ".join(f"{g} +/-{se:.3f}" for g, se in noise["median_share_se"].items()))

    per_rock = {
        "bulk_id": visual.bulk_ids[mask].tolist(),
        "nll_visual": zero_visual["joint"],
        "nll_pointcount": zero_counted["joint"],
        "truth_distance": total_variation(visual, counted, mask),
    }
    tag = args.checkpoint.parent.name
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_contrast(rows, per_rock, out_dir / "pointcount_contrast.png",
                  f"{tag}: visual estimate vs point count, {int(mask.sum())} paired rocks")
    (out_dir / "pointcount_contrast.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "n_paired": int(mask.sum()),
        "n_unit_strings": int(len(np.unique(visual.unit[mask]))),
        "n_eruptions": int(len(np.unique(parent))),
        "covered_groups": list(COVERED_GROUPS),
        "zero_shot": {
            name: {
                "joint": float(s["joint"].mean()),
                "dirichlet": float(s["dirichlet"].mean()),
                "bernoulli": float(s["bernoulli"].mean()),
                "share_mae_mean": s["share_mae_mean"],
                "mean_composition_size": s["mean_composition_size"],
            }
            for name, s in (("visual", zero_visual), ("pointcount", zero_counted))
        },
        "contrasts": contrasts,
        "truth_disagreement": rows,
        "counting_noise": noise,
        "per_rock": {k: (v if isinstance(v, list) else np.asarray(v).tolist())
                     for k, v in per_rock.items()},
    }, indent=2))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
