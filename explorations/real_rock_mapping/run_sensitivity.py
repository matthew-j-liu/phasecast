"""
How far does the zero-shot real-rock number move under the truth-mapping
judgment calls?  The eval scores measured modal mineralogy against the frozen
heads, but every number it reports passes through assumptions nobody measured:
mineral densities are mid-series stand-ins for solid solutions, the 0.5 vol%
trace bound is read off a reporting grid, the freshness cut is a leading-digit
parse of a 1-4 grade, and the support forcing invents a rule for splitting
"pyroxene is present" into cpx/opx.

The question is not how many nats the headline moves.  It is whether any
defensible mapping choice changes a decision: the recalibrator's parameterization
(per-group constants plus SiO2 slopes on q and amp), amphibole being a
discrimination failure rather than a calibration miss, and the headline being
quotable as the zero-shot gap.  Sign flips in the per-group offset vectors are
what would invalidate the first; everything else is an error bar.

Nothing here fits anything, so the recalibrator's held-out estimate is untouched.

    conda run -n alignn_fix python explorations/real_rock_mapping/run_sensitivity.py \
        --checkpoint checkpoints/joint_heads_supportmag_seed0/heads_best.pt
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import (
    BENCHMARK_DIR,
    GROUPS,
    GROUP_INDEX,
    MINERAL_DENSITY,
    MODEL_TO_GROUP,
    RealRockBenchmark,
    coarsen_alpha,
    coarsen_presence,
    load_benchmark,
    presence_group_stats,
    real_rock_nll,
    truth_forced_support,
)
from cnmineral.data.phase_vocab import PHASE_INDEX
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# Petrologically plausible range per reported mineral name, for arc volcanic
# rocks.  Endpoints are the solid-solution limits a geologist would actually
# meet in this suite, not the full mineral-group range: olivine spans Fo90 to
# Fo50 rather than Fo100-Fa100, and quartz has no solid solution so it barely
# moves.  Drawn uniformly and independently for the Monte Carlo band.
DENSITY_RANGE: dict[str, tuple[float, float]] = {
    "plagioclase": (2.62, 2.76),
    "alkali feldspar": (2.55, 2.63),
    "pyroxene": (3.20, 3.60),
    "augite": (3.25, 3.55),
    "hypersthene": (3.30, 3.60),
    "bronzite": (3.25, 3.45),
    "hornblende": (3.00, 3.40),
    "cummingtonite": (3.10, 3.45),
    "biotite": (2.80, 3.30),
    "olivine": (3.30, 3.90),
    "Fe-Ti oxides": (4.70, 5.30),
    "quartz": (2.64, 2.66),
    "silica phases": (2.20, 2.65),
}

# independent draws understate the spread if the whole suite is systematically
# iron-rich, which moves every ferromagnesian the same way at once.  These two
# corners hold the mafic/felsic mass ratio at its extremes, which is the ratio
# the magnitude comparison is actually about.
MAFIC_NAMES = frozenset({
    "pyroxene", "augite", "hypersthene", "bronzite", "hornblende",
    "cummingtonite", "biotite", "olivine", "Fe-Ti oxides",
})

MC_DRAWS = 200

# `?` and `??` qualify the identification, not the abundance; `xenos` marks a
# xenocryst, which is not part of the equilibrium assemblage the model predicts.
UNCERTAIN_QUALIFIERS = frozenset({"?", "??", "tr?"})
XENOCRYST_QUALIFIER = "xenos"


def scaled_densities(**overrides: float) -> dict[str, float]:
    """The shipped density table with named minerals replaced."""
    table = dict(MINERAL_DENSITY)
    table.update(overrides)
    return table


def endpoint_densities(mafic_high: bool) -> dict[str, float]:
    """Every ferromagnesian at one end of its range, every felsic at the other."""
    return {
        name: (high if (name in MAFIC_NAMES) == mafic_high else low)
        for name, (low, high) in DENSITY_RANGE.items()
    }


def monte_carlo_densities(rng: np.random.Generator) -> dict[str, float]:
    return {
        name: float(rng.uniform(low, high))
        for name, (low, high) in DENSITY_RANGE.items()
    }


def benchmark_without_mode_rows(drop: pd.Series) -> RealRockBenchmark:
    """Reload the benchmark with some `modes.csv` rows removed.

    Goes through a filtered copy on disk rather than editing arrays, so the
    variant exercises the shipped loader end to end -- an audit of a
    reimplementation audits nothing.
    """
    modes = pd.read_csv(BENCHMARK_DIR / "modes.csv")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(BENCHMARK_DIR / "samples.csv", tmp / "samples.csv")
        modes[~drop.to_numpy()].to_csv(tmp / "modes.csv", index=False)
        return load_benchmark(tmp)


def uncertain_id_masks() -> dict[str, pd.Series]:
    """Which `modes.csv` rows each uncertain-identification variant drops."""
    modes = pd.read_csv(BENCHMARK_DIR / "modes.csv")
    qualifier = modes["qualifier"].astype(str).str.strip().str.lower()
    return {
        "drop_uncertain_id": qualifier.isin(UNCERTAIN_QUALIFIERS),
        "drop_xenocrysts": qualifier == XENOCRYST_QUALIFIER,
    }


def imputed_at_bound(benchmark: RealRockBenchmark) -> RealRockBenchmark:
    """Trace phases quantified at their bound instead of left-censored.

    The structural alternative to censoring, and an exact transform of the
    loaded arrays rather than a second mapping: `censor_bound` is already the
    trace mass as a share of the mass the rock would have with every trace
    sitting at its bound, so the numeric shares only have to give up the room
    those bounds occupy.  `f` is what is left for them.

    Imputing would also hand a composition to the 11 rocks whose every phase
    was reported as a trace, which no other variant scores.  `has_magnitude`
    stays the base benchmark's so the comparison keeps the headline's row set --
    the variant is meant to isolate the policy, not to also enlarge the
    benchmark.
    """
    f = 1.0 - benchmark.censor_bound.sum(1, keepdims=True)
    magnitude = np.where(
        benchmark.censored, benchmark.censor_bound, benchmark.magnitude * f
    )
    numeric = benchmark.numeric | benchmark.censored
    return replace(
        benchmark,
        magnitude=np.where(benchmark.has_magnitude[:, None], magnitude, 0.0),
        numeric=numeric & benchmark.has_magnitude[:, None],
        censored=np.zeros_like(benchmark.censored),
        censor_bound=np.zeros_like(benchmark.censor_bound),
    )


def group_members(group: str) -> list[int]:
    return [PHASE_INDEX[ph] for ph, g in MODEL_TO_GROUP.items() if g == group]


def coarse_support(presence7: np.ndarray, p21: np.ndarray, rule: str) -> np.ndarray:
    """The two structural alternatives to the shipped resolution rule.

    `all_members` turns on every member of a truth-present group, spreading the
    group's concentration over phases the model may not believe in;
    `single_member` keeps only its favourite.  Both leave the phases no source
    reports to the model, as the shipped rule does -- truth never asserted
    their absence, so forcing them off would be the mapping inventing evidence.
    """
    if rule not in ("all_members", "single_member"):
        raise ValueError(f"unknown support rule {rule!r}")
    support = (p21 > 0.5).astype(np.float64)
    for group, gi in GROUP_INDEX.items():
        members = group_members(group)
        present = presence7[:, gi] > 0
        support[np.ix_(~present, members)] = 0.0
        if rule == "all_members":
            support[np.ix_(present, members)] = 1.0
        else:
            support[np.ix_(present, members)] = 0.0
            best = np.asarray(members)[p21[np.ix_(present, members)].argmax(1)]
            support[np.flatnonzero(present), best] = 1.0
    return support


def score(
    head, latents: torch.Tensor, p21: np.ndarray, p7: np.ndarray,
    benchmark: RealRockBenchmark, support: np.ndarray, mask: np.ndarray,
) -> dict:
    """Every number this audit compares across variants, for one mapping.

    Presence prediction is the model's own call and never sees the mapping, so
    `p7` is computed once by the caller; only the magnitude branch is re-run,
    because its concentrations are conditioned on the support the mapping
    forces.  The two offset vectors are the ones `plot_gap_summary` draws --
    they are the recalibrator's blueprint, so they are what a sign flip would
    invalidate.
    """
    with torch.no_grad():
        alpha = head.magnitude_alpha(
            latents,
            torch.as_tensor(support, dtype=latents.dtype, device=latents.device),
        ).cpu().numpy()
    alpha7, _ = coarsen_alpha(alpha, support)
    nll = real_rock_nll(
        p7, alpha7, benchmark.presence, benchmark.magnitude, benchmark.numeric
    )
    pred = conditional_mean(alpha7, benchmark.numeric.astype(np.float64))

    presence_offset = benchmark.presence[mask].mean(0) - p7[mask].mean(0)
    magnitude_offset = np.full(len(GROUPS), np.nan)
    for i in range(len(GROUPS)):
        m = mask & benchmark.numeric[:, i]
        if m.sum() >= 3 and pred[m, i].mean() > 0:
            magnitude_offset[i] = np.log(
                benchmark.magnitude[m, i].mean() / pred[m, i].mean()
            )
    auroc = presence_group_stats(p7[mask], benchmark.presence[mask])["auroc"]
    return {
        "n": int(mask.sum()),
        "joint_nll": float(nll["joint"][mask].mean()),
        "bernoulli_nll": float(nll["bernoulli"][mask].mean()),
        "dirichlet_nll": float(nll["dirichlet"][mask].mean()),
        "mean_composition_size": float(benchmark.numeric[mask].sum(1).mean()),
        "presence_offset": presence_offset.tolist(),
        "magnitude_offset": magnitude_offset.tolist(),
        "auroc": [None if np.isnan(a) else float(a) for a in auroc],
    }


def sample_column(column: str) -> dict[str, object]:
    """`samples.csv` values keyed the way `load_benchmark` keys its rocks.

    Lets a variant reach a column the benchmark does not carry (the raw
    alteration string, the reporting basis) by unique-key lookup rather than
    by re-deriving which rows survived the scoreable filter.
    """
    samples = pd.read_csv(BENCHMARK_DIR / "samples.csv")
    key = samples["source"] + "|" + samples["sample_id"].astype(str)
    return dict(zip(key, samples[column]))


def row_index(base_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Where each of `ids` sits in `base_ids`.

    A variant that drops mode rows can drop a whole rock (if every row it had
    was uncertain), which shortens the benchmark.  The latents were computed on
    the base rocks, so they have to be re-indexed rather than assumed aligned.
    """
    position = {key: i for i, key in enumerate(base_ids)}
    return np.array([position[key] for key in ids], dtype=np.int64)


def alteration_masks(benchmark: RealRockBenchmark) -> dict[str, np.ndarray]:
    """The freshness cuts, from strictest to none at all.

    `modifier_strict` reads a `+` as "worse than the digit", which demotes the
    25 `2+` rocks out of clean.  du Bray does not document the convention, so
    the audit scores both readings instead of guessing at one.
    """
    lookup = sample_column("alteration_extent")
    raw = np.array([str(lookup[key]) for key in benchmark.bulk_ids])
    strict = benchmark.grade + np.char.endswith(raw, "+").astype(np.int64)
    graded = benchmark.grade > 0
    return {
        "grade_1_only": benchmark.grade == 1,
        "grades_1_2": benchmark.clean,
        "modifier_strict": np.isin(strict, [1, 2]),
        "grades_1_2_3": np.isin(benchmark.grade, [1, 2, 3]),
        "all_graded": graded,
        "all_rocks_incl_ungraded": np.ones(len(benchmark), dtype=bool),
    }


def run_variants(
    head, latents, base: RealRockBenchmark, p21, p7
) -> tuple[list[dict], list[dict]]:
    """Every mapping variant, scored on the shipped headline's terms.

    The row set is the headline's own -- least-altered rocks carrying at least
    one quantified phase -- except where a variant is about the row set.
    Returns the named variants and the Monte Carlo density draws separately.
    """
    rows: list[dict] = []

    def add(knob: str, name: str, benchmark, support, mask, idx=None, **extra):
        rows_in = slice(None) if idx is None else idx
        row = {"knob": knob, "variant": name}
        row.update(score(head, latents[rows_in], p21[rows_in], p7[rows_in],
                         benchmark, support, mask))
        row.update(extra)
        rows.append(row)
        print(f"  {name:<28s} n={row['n']:<4d} joint {row['joint_nll']:8.4f}"
              f"  size {row['mean_composition_size']:.2f}")

    def headline_mask(benchmark):
        return benchmark.clean & benchmark.has_magnitude

    shipped_support = truth_forced_support(base.presence, p21)
    base_mask = headline_mask(base)

    print("densities")
    density_tables = {
        "shipped": MINERAL_DENSITY,
        "no_conversion": {name: 1.0 for name in MINERAL_DENSITY},
        "oxides_light": scaled_densities(**{"Fe-Ti oxides": 4.70}),
        "oxides_heavy": scaled_densities(**{"Fe-Ti oxides": 5.30}),
        "olivine_forsteritic": scaled_densities(olivine=3.30),
        "olivine_fayalitic": scaled_densities(olivine=3.90),
        "mafic_felsic_contrast_high": endpoint_densities(mafic_high=True),
        "mafic_felsic_contrast_low": endpoint_densities(mafic_high=False),
    }
    for name, table in density_tables.items():
        b = load_benchmark(densities=table)
        add("densities", name, b, shipped_support, headline_mask(b))

    print(f"monte carlo densities ({MC_DRAWS} draws)")
    rng = np.random.default_rng(0)
    mc: list[dict] = []
    for _ in range(MC_DRAWS):
        b = load_benchmark(densities=monte_carlo_densities(rng))
        mc.append(score(head, latents, p21, p7, b, shipped_support,
                        headline_mask(b)))

    print("trace bound")
    for bound in (0.1, 0.25, 0.5, 1.0):
        b = load_benchmark(trace_bound_vol_pct=bound)
        add("trace_bound", f"bound_{bound}_volpct", b, shipped_support,
            headline_mask(b), censored_bound_mass=float(
                b.censor_bound[headline_mask(b)].sum(1).mean()))
    imputed = imputed_at_bound(base)
    add("trace_bound", "impute_at_bound", imputed, shipped_support,
        headline_mask(imputed))

    print("alteration rule")
    for name, mask in alteration_masks(base).items():
        add("alteration", name, base, shipped_support, mask & base.has_magnitude)

    print("support forcing")
    for threshold in (0.3, 0.5, 0.7):
        add("support", f"resolve_at_{threshold}", base,
            truth_forced_support(base.presence, p21, member_threshold=threshold),
            base_mask)
    for rule in ("all_members", "single_member"):
        add("support", rule, base, coarse_support(base.presence, p21, rule),
            base_mask)

    print("uncertain identifications")
    for name, drop in uncertain_id_masks().items():
        b = benchmark_without_mode_rows(drop)
        idx = row_index(base.bulk_ids, b.bulk_ids)
        add("uncertain_id", name, b,
            truth_forced_support(b.presence, p21[idx]), headline_mask(b),
            idx=idx, n_mode_rows_dropped=int(drop.sum()),
            n_rocks_lost=int(len(base) - len(b)))

    print("reporting basis")
    basis = sample_column("mode_basis")
    pointcounted = np.array(
        [basis[key] == "optical_vol_pointcount" for key in base.bulk_ids]
    )
    add("reporting_basis", "drop_pointcounted", base, shipped_support,
        base_mask & ~pointcounted, n_pointcounted=int(pointcounted.sum()))
    add("reporting_basis", "pointcounted_only", base, shipped_support,
        base_mask & pointcounted)
    return rows, mc


def plot_headline_spread(rows, mc, baseline, out_path: Path) -> None:
    """Every variant's headline against the shipped one, grouped by knob.

    The Monte Carlo band is the honest spread from densities; the named
    variants are corners chosen to be interpretable.  Rows whose composition
    size moved are annotated, because the Dirichlet term is a density over
    however many groups a rock spans -- a variant that changes that count is
    not comparable on this axis, however it lands.
    """
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(rows) + 1.8))
    colors = {knob: f"C{i}" for i, knob in enumerate(dict.fromkeys(
        row["knob"] for row in rows))}
    y = np.arange(len(rows))
    ax.axvline(baseline, lw=1.0, color="0.3", zorder=3)
    band = np.quantile([m["joint_nll"] for m in mc], [0.05, 0.95])
    ax.axvspan(band[0], band[1], color="C0", alpha=0.15, zorder=0)
    for i, row in enumerate(rows):
        ax.plot(row["joint_nll"], i, "o", ms=5, color=colors[row["knob"]])
    size0 = baseline_size(rows)
    labels = []
    for row in rows:
        moved = abs(row["mean_composition_size"] - size0) > 0.05
        labels.append(f"{row['variant']}  (n={row['n']})"
                      + ("  *size" if moved else ""))
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("headline joint NLL (nats per rock)")
    ax.set_title(
        f"truth-mapping sensitivity -- shipped {baseline:.3f} nats (grey line)\n"
        f"shaded: 5-95% of {len(mc)} Monte Carlo density draws;  "
        "*size = composition size moved, so the Dirichlet term is not comparable",
        fontsize=10,
    )
    for knob, color in colors.items():
        ax.plot([], [], "o", color=color, label=knob)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def baseline_size(rows) -> float:
    return next(r["mean_composition_size"] for r in rows if r["variant"] == "shipped")


def plot_offset_stability(rows, mc, out_path: Path) -> None:
    """The recalibrator's blueprint under every mapping -- the figure that decides it.

    The thin recalibrator is per-group constants plus two SiO2 slopes, fit to
    exactly these two vectors.  A variant that moves an offset a little rescales
    the correction; one that flips its sign says the correction points the wrong
    way, which is the only outcome here that would change the recalibrator's design.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    shipped = next(r for r in rows if r["variant"] == "shipped")
    for ax, key, title in (
        (axes[0], "presence_offset",
         "presence offset (observed rate - mean predicted p)\n"
         "density band is zero-width by construction"),
        (axes[1], "magnitude_offset",
         "magnitude log-scale offset (log measured/predicted)"),
    ):
        x = np.arange(len(GROUPS))
        mc_values = np.array([m[key] for m in mc], dtype=np.float64)
        lo, hi = np.nanquantile(mc_values, [0.05, 0.95], axis=0)
        # per-group segments, not a filled band: a zero-width band has to
        # render as nothing.  `fill_between(..., step="mid")` fills the span
        # between neighbouring groups when lo == hi, which would draw density
        # uncertainty onto presence, where density provably has none.
        ax.vlines(x, lo, hi, color="C0", lw=7, alpha=0.3,
                  label="density Monte Carlo 5-95%")
        for row in rows:
            ax.plot(x, row[key], "-", lw=0.6, color="0.6", alpha=0.6)
        ax.plot(x, shipped[key], "o-", lw=1.8, color="tab:red", ms=6,
                label="shipped mapping")
        ax.axhline(0, lw=1.0, color="0.2")
        ax.set_xticks(x, GROUPS)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("offset")
    fig.suptitle("does the recalibrator's blueprint survive the mapping? "
                 "grey: every variant; a sign flip is the only disqualifying move",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def sign_flips(
    rows, key: str, materiality: float, min_n: int = 100
) -> list[dict]:
    """Groups where some variant reverses a correction the recalibrator would make.

    Both sides have to clear `materiality`, because a group whose shipped
    offset sits at zero has no correction to reverse -- amphibole's magnitude
    offset is -0.002, so every perturbation flips its sign while changing
    nothing a recalibrator would do.  Subsets below `min_n` rocks are excluded
    outright: a per-group offset estimated on a handful of rocks is sampling
    noise, not evidence about the parameterization.
    """
    shipped = np.array(next(r for r in rows if r["variant"] == "shipped")[key])
    flips = []
    for i, group in enumerate(GROUPS):
        if np.isnan(shipped[i]) or abs(shipped[i]) < materiality:
            continue
        offending = [
            {"variant": row["variant"], "offset": float(row[key][i])}
            for row in rows
            if row["n"] >= min_n and not np.isnan(row[key][i])
            and np.sign(row[key][i]) != np.sign(shipped[i])
            and abs(row[key][i]) >= materiality
        ]
        if offending:
            flips.append({"group": group, "shipped": float(shipped[i]),
                          "flipped_by": offending})
    return flips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    ckpt = JointHeadCheckpoint.load(checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    base = load_benchmark()
    latents = bulk_latents(encoder, base, device)
    head = ckpt.build_mode_head().to(device)
    with torch.no_grad():
        p21 = torch.sigmoid(head.presence_logits(latents)).cpu().numpy()
    p7 = coarsen_presence(p21)
    print(f"device:  {device}")
    print(f"heads:   {checkpoint}")
    print(f"rocks:   {len(base)} with chemistry + modes\n")

    rows, mc = run_variants(head, latents, base, p21, p7)
    shipped = next(row for row in rows if row["variant"] == "shipped")

    output_dir = RESULTS_ROOT / checkpoint.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_headline_spread(rows, mc, shipped["joint_nll"],
                         output_dir / "headline_spread.png")
    plot_offset_stability(rows, mc, output_dir / "offset_stability.png")

    headlines = [row["joint_nll"] for row in rows]
    mc_headlines = [m["joint_nll"] for m in mc]
    # a group everybody has (or nobody has) on a subset gets no AUROC, so the
    # per-variant entry is null rather than a number to take a range over.
    amp_auroc = [
        np.nan if row["auroc"][GROUP_INDEX["amp"]] is None
        else row["auroc"][GROUP_INDEX["amp"]] for row in rows
    ]
    comparable = [
        row["joint_nll"] for row in rows
        if row["n"] == shipped["n"]
        and abs(row["mean_composition_size"] - shipped["mean_composition_size"]) < 0.05
    ]
    by_knob: dict[str, list[float]] = {}
    for row in rows:
        by_knob.setdefault(row["knob"], []).append(row["joint_nll"])
    summary = {
        "checkpoint_tag": checkpoint.parent.name,
        "shipped_joint_nll": shipped["joint_nll"],
        "range_over_named_variants": [min(headlines), max(headlines)],
        # the only range that is a like-for-like read: same rocks, same
        # composition size, so neither the subset nor the Dirichlet term's
        # dimensionality is doing the moving.
        "range_over_comparable_variants": [min(comparable), max(comparable)],
        "range_by_knob": {
            knob: [min(v), max(v)] for knob, v in by_knob.items()
        },
        "monte_carlo_density_5_95": list(
            np.quantile(mc_headlines, [0.05, 0.95])
        ),
        "monte_carlo_draws": len(mc),
        "monte_carlo_offset_5_95": {
            key: np.nanquantile(
                np.array([m[key] for m in mc], dtype=np.float64), [0.05, 0.95],
                axis=0,
            ).T.tolist()
            for key in ("presence_offset", "magnitude_offset")
        },
        # materiality floors: 5 percentage points of presence probability, and
        # a 10% magnitude rescale. Below those a recalibrator learns approximately
        # no correction, so the sign it learns is not a design question.
        "presence_offset_sign_flips": sign_flips(rows, "presence_offset", 0.05),
        "magnitude_offset_sign_flips": sign_flips(rows, "magnitude_offset", 0.10),
        "amp_auroc_range": [
            float(np.nanmin(amp_auroc)), float(np.nanmax(amp_auroc)),
        ],
        "groups": list(GROUPS),
        "variants": rows,
    }
    (output_dir / "sensitivity.json").write_text(json.dumps(summary, indent=2))

    print(f"\nshipped headline      {shipped['joint_nll']:.4f} nats")
    print(f"comparable variants   {min(comparable):.4f} .. {max(comparable):.4f}"
          "  (same rocks, same composition size)")
    print(f"all named variants    {min(headlines):.4f} .. {max(headlines):.4f}"
          "  (mixes subsets and sizes -- not a like-for-like range)")
    for knob, values in by_knob.items():
        print(f"  {knob:<18s} {min(values):8.4f} .. {max(values):8.4f}")
    print(f"density monte carlo   {summary['monte_carlo_density_5_95'][0]:.4f} "
          f".. {summary['monte_carlo_density_5_95'][1]:.4f}  (5-95%)")
    print(f"amp AUROC             {summary['amp_auroc_range'][0]:.3f} .. "
          f"{summary['amp_auroc_range'][1]:.3f}")
    for key in ("presence_offset_sign_flips", "magnitude_offset_sign_flips"):
        flips = summary[key]
        print(f"{key:<28s} "
              + ("none" if not flips
                 else ", ".join(f"{f['group']} ({len(f['flipped_by'])} variants)"
                                for f in flips)))
    print(f"\nwrote {output_dir}")


if __name__ == "__main__":
    main()
