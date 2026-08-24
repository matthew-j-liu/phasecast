"""Does the analytical-total convention move the real-rock score (issue #56)?

du Bray publishes major oxides recalculated to sum to 100 on a volatile-free
basis, which is the petrological standard for classifying a rock.  gard2019 --
the corpus the heads were trained on -- carries raw analyses whose remainder is
volatiles the tokenizer never sees.  Nothing renormalizes a row on the way in,
so a real rock and a training rock of identical relative chemistry reach the
encoder at slightly different overall scale.

The workbooks record what each analysis summed to *before* that recalculation
(`Total_I_pct`, the full analytical total including volatiles) and how much of
it was volatile (`Volatile_pct`), so the recalculation is invertible per rock:

    raw major = published major * (Total_I_pct - Volatile_pct) / 100

This scores the frozen stack on both conventions and reads the difference.  The
undo is a translation into the training corpus's convention, not a correction of
an error du Bray made.

Zero-shot only, which is what issue #56 asks for: a fitted recalibrator could
absorb a uniform input shift and hide the effect being measured.  Test rocks are
excluded -- that slice is spent.

    conda run -n mineral python explorations/real_mineralogy/run_scale_convention.py \
        --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import BENCHMARK_DIR, load_benchmark, split_mask
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    paired_bootstrap,
    score_recalibrator,
)

RESULTS = Path(__file__).resolve().parent / "results"

# The columns du Bray recalculated.  Traces stay in as-measured ppm and are not
# part of the volatile-free recalculation, so they do not move; `loi` is a
# volatile and carries no token.
RECALCULATED_COLS: tuple[str, ...] = (
    "sio2", "tio2", "al2o3", "feo_tot", "mno", "mgo", "cao", "na2o", "k2o", "p2o5",
)

# The subset the encoder actually sees: MnO and P2O5 are recalculated by the
# source but carry no token, so a "total" measured over them would not be the
# scale the model is exposed to.  This is the set issue #56's quoted totals are
# measured over, which is why the benchmark's median lands at 99.7 rather than
# the 100.0 the full recalculated major set sums to.
TOKENIZED_MAJOR_COLS: tuple[str, ...] = (
    "sio2", "al2o3", "cao", "mgo", "k2o", "na2o", "tio2",
)
IRON_COLS: tuple[str, ...] = ("feo_tot", "fe2o3_tot", "feo", "fe2o3")

DUBRAY_CHEM = {
    "dubray_ds764": "dubray_ds764/Appendix3.xls",
    "dubray_ds1099": "dubray_ds1099/Appendix03.xlsx",
}

# gard2019's own major columns, for the question of which convention the undo
# moves the benchmark toward.
GARD_PATH = Path(__file__).resolve().parents[2] / "data" / "bulks" / "gard2019.parquet"


def scale_factors() -> pd.Series:
    """Per-rock `(Total_I_pct - Volatile_pct)/100`, keyed as the benchmark keys.

    Under 1.0 wherever the published analysis was scaled up to reach 100, which
    is nearly every rock.  Rocks whose workbook row lacks either column get no
    factor and stay on the published convention.
    """
    out = {}
    for source, path in DUBRAY_CHEM.items():
        geo = pd.read_excel(BENCHMARK_DIR / path, header=1)
        geo = geo.dropna(subset=["Field_ID"]).drop_duplicates(subset=["Field_ID"])
        total = pd.to_numeric(geo["Total_I_pct"], errors="coerce")
        volatile = pd.to_numeric(geo["Volatile_pct"], errors="coerce")
        factor = (total - volatile) / 100.0
        keys = source + "|" + geo["Field_ID"].astype(str)
        out.update(dict(zip(keys[factor.notna()], factor[factor.notna()])))
    return pd.Series(out, dtype=float)


def unrecalculated_dir(out: Path, factors: pd.Series) -> tuple[Path, np.ndarray]:
    """A benchmark whose du Bray majors are put back on the raw analytical total.

    Written to disk and read back through the shipped loader, so the alternate
    convention goes through the same tokenization, freshness and truth path as
    the published one.  Returns the directory and the keys that actually moved.
    """
    samples = pd.read_csv(BENCHMARK_DIR / "samples.csv")
    modes = pd.read_csv(BENCHMARK_DIR / "modes.csv")
    keys = samples["source"] + "|" + samples["sample_id"].astype(str)
    factor = keys.map(factors)
    moved = factor.notna().to_numpy()
    for col in RECALCULATED_COLS:
        if col in samples.columns:
            samples[col] = samples[col] * factor.where(factor.notna(), 1.0)
    samples.to_csv(out / "samples.csv", index=False)
    modes.to_csv(out / "modes.csv", index=False)
    return out, keys[moved].to_numpy()


def tokenized_totals(frame: pd.DataFrame) -> pd.Series:
    """Sum of the majors the encoder is exposed to, iron from whichever column reports it.

    Iron takes the max across the reported columns rather than a sum, since
    `feo` / `fe2o3` / `*_tot` are alternative expressions of the same element
    and adding them would double-count -- the same stand-in issue #56 measured
    gard2019 with.
    """
    base = [c for c in TOKENIZED_MAJOR_COLS if c in frame.columns]
    total = frame[base].sum(axis=1, min_count=len(base))
    iron = [c for c in IRON_COLS if c in frame.columns]
    if iron:
        total = total + frame[iron].max(axis=1)
    return total


def major_totals(benchmark_dir: Path) -> pd.Series:
    """Tokenized-major totals keyed as `load_benchmark` keys, so they reindex onto it.

    samples.csv holds rocks the loader drops (no chemistry, or no mode rows), so
    positional alignment against a benchmark mask would silently misalign.
    """
    samples = pd.read_csv(benchmark_dir / "samples.csv")
    keys = samples["source"] + "|" + samples["sample_id"].astype(str)
    return pd.Series(tokenized_totals(samples).to_numpy(), index=keys.to_numpy())


def gard_major_totals() -> pd.Series:
    """gard2019's tokenized-major sums, from the parquet the corpus builds from."""
    gard = pd.read_parquet(GARD_PATH)
    return tokenized_totals(gard).dropna()


def describe(values: np.ndarray, name: str) -> dict:
    q = np.nanpercentile(values, [5, 50, 95])
    return {"name": name, "p5": float(q[0]), "median": float(q[1]),
            "p95": float(q[2]), "n": int(np.isfinite(values).sum())}


def latent_shift(published: torch.Tensor, undone: torch.Tensor) -> dict:
    """How far the encoder's representation moves, in its own units.

    Relative L2 is the shift against the typical latent norm, so it is readable
    without knowing the embedding scale: 0 means the encoder cannot see the
    change at all, 1 means it moved the representation as far as an average rock
    sits from the origin.
    """
    delta = (undone - published).norm(dim=1)
    scale = published.norm(dim=1).median()
    cosine = torch.nn.functional.cosine_similarity(published, undone, dim=1)
    return {
        "median_relative_l2": float((delta / scale).median()),
        "p95_relative_l2": float((delta / scale).quantile(0.95)),
        "median_cosine": float(cosine.median()),
        "min_cosine": float(cosine.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    factors = scale_factors()
    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    head = ckpt.build_mode_head().to(device)
    encoder = encoder.to(device)

    published = load_benchmark()
    with tempfile.TemporaryDirectory() as tmp:
        undone_dir, moved_keys = unrecalculated_dir(Path(tmp), factors)
        undone = load_benchmark(undone_dir)
        totals_undone = major_totals(undone_dir)
    totals_published = major_totals(BENCHMARK_DIR)

    # only the chemistry may differ: same rocks, same truth, same freshness.
    assert np.array_equal(published.bulk_ids, undone.bulk_ids)
    for field in ("presence", "magnitude", "numeric", "censored", "has_magnitude",
                  "clean"):
        assert np.array_equal(getattr(published, field), getattr(undone, field)), field
    assert not torch.equal(published.bulk_frac, undone.bulk_frac)

    scored = published.clean & published.has_magnitude
    rescaled = np.isin(published.bulk_ids, moved_keys)
    # test is spent; the contrast reads on the rocks that actually moved
    mask = scored & rescaled & ~split_mask(published, "test")
    val = mask & split_mask(published, "val")

    factor_on_scored = factors.reindex(published.bulk_ids[mask]).to_numpy()
    print(f"scored non-test rocks {int((scored & ~split_mask(published, 'test')).sum())}, "
          f"of which rescaled {int(mask.sum())} ({int(val.sum())} in val)")
    print(f"recalculation factor on those rocks: median "
          f"{np.median(factor_on_scored):.4f}, p5 {np.percentile(factor_on_scored, 5):.4f}, "
          f"p95 {np.percentile(factor_on_scored, 95):.4f}")

    print("\nmajor-oxide totals, the quantity whose convention is in question:")
    gard = gard_major_totals().to_numpy()
    for row in (describe(totals_published.reindex(published.bulk_ids[mask]).to_numpy(), "benchmark as published"),
                describe(totals_undone.reindex(published.bulk_ids[mask]).to_numpy(), "benchmark un-recalculated"),
                describe(gard, "gard2019 (training corpus)")):
        print(f"  {row['name']:28s} p5 {row['p5']:6.2f}  median {row['median']:6.2f}  "
              f"p95 {row['p95']:6.2f}  (n={row['n']})")

    lat_published = bulk_latents(encoder, published, device)
    lat_undone = bulk_latents(encoder, undone, device)
    shift = latent_shift(lat_published[torch.as_tensor(mask)],
                         lat_undone[torch.as_tensor(mask)])
    print(f"\nencoder latent shift: relative L2 median "
          f"{shift['median_relative_l2']:.4f} (p95 {shift['p95_relative_l2']:.4f}), "
          f"cosine median {shift['median_cosine']:.5f} (min {shift['min_cosine']:.5f})")

    gp_published = build_group_predictions(head, published, lat_published).subset(mask)
    gp_undone = build_group_predictions(head, undone, lat_undone).subset(mask)
    zero_published = score_recalibrator(None, gp_published)
    zero_undone = score_recalibrator(None, gp_undone)

    print("\nzero-shot on the rescaled non-test rocks (joint / dirichlet / bernoulli):")
    for name, s in (("as published", zero_published), ("un-recalculated", zero_undone)):
        print(f"  {name:16s} {s['joint'].mean():7.4f} {s['dirichlet'].mean():7.4f} "
              f"{s['bernoulli'].mean():7.4f}")

    contrasts = {}
    for term in ("joint", "dirichlet", "bernoulli"):
        delta = zero_published[term] - zero_undone[term]
        boot = paired_bootstrap(zero_published[term], zero_undone[term],
                                published.unit[mask])
        contrasts[f"published_minus_undone_{term}"] = {
            **boot,
            "median_delta": float(np.median(delta)),
            "frac_undone_better": float((delta > 0).mean()),
        }
        print(f"  published minus un-recalculated, {term:10s} mean {boot['mean']:+.4f} "
              f"[{boot['lo']:+.4f}, {boot['hi']:+.4f}] over {boot['n_units']} units; "
              f"median {np.median(delta):+.4f}, "
              f"{(delta > 0).mean():.0%} of rocks better un-recalculated")

    # A uniform shift that merely happened to help would improve every rock
    # about equally.  If instead the gain tracks how large a correction the rock
    # needed, the convention is doing the work rather than a tail of outliers.
    correction = 1.0 - factor_on_scored
    joint_delta = zero_published["joint"] - zero_undone["joint"]
    dose = float(pd.Series(correction).corr(pd.Series(joint_delta), method="spearman"))
    edges = np.quantile(correction, [0, 1 / 3, 2 / 3, 1.0])
    tercile = np.clip(np.digitize(correction, edges[1:-1]), 0, 2)
    print(f"\ndose-response: spearman {dose:+.3f} between correction size and "
          f"nats gained")
    bands = []
    for t, label in enumerate(("smallest", "middle", "largest")):
        sel = tercile == t
        bands.append({
            "band": label,
            "n": int(sel.sum()),
            "median_correction_pct": float(np.median(correction[sel]) * 100),
            "mean_joint_delta": float(joint_delta[sel].mean()),
            "frac_improved": float((joint_delta[sel] > 0).mean()),
        })
        print(f"  {label:8s} correction n={int(sel.sum()):3d}  "
              f"median {np.median(correction[sel]) * 100:5.2f}%  "
              f"mean gain {joint_delta[sel].mean():+.4f}  "
              f"improved {(joint_delta[sel] > 0).mean():.0%}")

    val_delta = (zero_published["joint"][val[mask]] - zero_undone["joint"][val[mask]])
    print(f"\nval slice alone ({int(val.sum())} rocks): joint "
          f"{zero_published['joint'][val[mask]].mean():.4f} as published vs "
          f"{zero_undone['joint'][val[mask]].mean():.4f} un-recalculated "
          f"(mean delta {val_delta.mean():+.4f})")

    tag = args.checkpoint.parent.name
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scale_convention.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "n_rescaled_scored_nontest": int(mask.sum()),
        "n_val": int(val.sum()),
        "factor": describe(factor_on_scored, "recalculation factor"),
        "major_totals": [
            describe(totals_published.reindex(published.bulk_ids[mask]).to_numpy(), "benchmark as published"),
            describe(totals_undone.reindex(published.bulk_ids[mask]).to_numpy(), "benchmark un-recalculated"),
            describe(gard, "gard2019"),
        ],
        "latent_shift": shift,
        "zero_shot": {
            name: {term: float(s[term].mean())
                   for term in ("joint", "dirichlet", "bernoulli")}
            for name, s in (("as_published", zero_published),
                            ("un_recalculated", zero_undone))
        },
        "contrasts": contrasts,
        "dose_response": {"spearman": dose, "bands": bands},
        "val_joint": {
            "as_published": float(zero_published["joint"][val[mask]].mean()),
            "un_recalculated": float(zero_undone["joint"][val[mask]].mean()),
        },
    }, indent=2))
    print(f"\nwrote {out_dir}/scale_convention.json")


if __name__ == "__main__":
    main()
