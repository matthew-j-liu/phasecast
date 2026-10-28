"""Does fractional crystallization of each benchmark rock's own composition
give simulated crystals closer to its measured mineralogy than equilibrium
does? Both runs share their rocks and their (P, H2O) draws, so the recipe is
the only difference. Each is read at ten rungs (90 to 10 and 5% melt), at
each draw's last converged step, at the pooled 50 + 30 rungs the corpus
label uses and, on the volcanic benchmark, at each rock's own melt fraction
(1 - `total_crystals_pct` / 100, a volume fraction against a mass fraction;
the plutonic rocks are fully crystalline and have none), on the draws both
recipes reach at that read. Every source is the equal-weight draw mean of
each draw's share over the rock's measured mineral list, evaluated on the
train and val rocks with no fit (raw simulator error), beside the production
head's marginal read and the CIPW norm; test stays sealed. Bands are a unit
bootstrap (eruptive units, or intrusions), which also counts how often each
read has the lowest mean-of-seven error per recipe.

  python explorations/generator_reach/run_recipe_rungs.py --benchmark volcanic
  python explorations/generator_reach/run_recipe_rungs.py --benchmark plutonic
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.magemin_oxides import FE2O3_TO_FEO_MASS
from cnmineral.data.phase_vocab import GROUPS, PRESENCE_CUTOFF
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import BENCHMARK_DIR, load_benchmark
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from explorations.corpus_scaling.fig_zero_shot_scaling import (
    LABELS, MIN_SPAN, PANELS, panel_grid, setup_style)
from explorations.corpus_scaling.run_depth_grid import MEANS, share_metrics, with_means
from explorations.generator_reach import run_read
from explorations.generator_reach.compositions import load_compositions
from explorations.generator_reach.run_signal import (
    PRODUCTION, cipw_shares, marginal_shares, presence_auroc, renormalize, spearman)
from explorations.real_mineralogy.plutonic import plutonic_benchmark

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORK = HERE / "work"
BENCHMARKS = {
    "volcanic": dict(compositions="compositions.csv", works=("broad128", "broad128_fc"), own_melt=True),
    "plutonic": dict(compositions="plutonic_compositions.csv", works=("plutonic128", "plutonic128_fc"),
                     own_melt=False)}
RECIPES = ("equilibrium", "fractional")
TABLES = {"equilibrium": "ladder", "fractional": "pile"}
RUNGS = ("90", "80", "70", "60", "50", "40", "30", "20", "10", "5")
READS = (*RUNGS, "last", "own_melt", "50+30")
READ_LABELS = {**{r: r for r in RUNGS}, "last": "last step", "own_melt": "own melt", "50+30": "50+30"}
BASELINES = ("head_marginal", "cipw_norm")
SPLITS = ("train", "val")
N_BOOT = 2000
OUT_LABELS = (*GROUPS, *MEANS)


def rung_ends(points, target):
    """Per (bulk, draw), the highest-T point with melt below `target`, a
    melt fraction or a per-bulk Series of them; a draw that never gets below
    has no row."""
    if isinstance(target, pd.Series):
        below = points["liq_frac"] < points["bulk"].map(target)
    else:
        below = points["liq_frac"] < target
    return points[below].groupby(["bulk", "draw"])["T_celsius"].max().reset_index()


def last_ends(points):
    """Per (bulk, draw), its lowest-T converged point."""
    return points.groupby(["bulk", "draw"])["T_celsius"].min().reset_index()


def matched(ends):
    """The (bulk, draw) pairs every end table in `ends` reaches."""
    keys = ends[0][["bulk", "draw"]]
    for e in ends[1:]:
        keys = keys.merge(e[["bulk", "draw"]], on=["bulk", "draw"])
    return keys


def read_rows(points, ends, keys):
    """The points at `ends`, restricted to the matched draws `keys`."""
    return points.merge(ends.merge(keys, on=["bulk", "draw"]), on=run_read.KEYS)


def rock_reads(rows, n_rocks):
    """Per bulk 0..n_rocks-1, the equal-weight mean over its rows of the
    share over the measured mineral list and of presence (the category above
    the label cutoff of the seven-category crystal mass); NaN for a bulk with
    no rows."""
    mass = rows[[f"m_{g}" for g in GROUPS]].to_numpy()
    total = mass.sum(1, keepdims=True)
    seven = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
    frame = pd.DataFrame(np.c_[rows[[f"s_{g}" for g in GROUPS]].to_numpy(), seven > PRESENCE_CUTOFF],
                         index=rows["bulk"].to_numpy())
    mean = frame.groupby(level=0).mean().reindex(range(n_rocks)).to_numpy(dtype=float)
    k = len(GROUPS)
    return mean[:, :k], mean[:, k:]


def recipe_reads(points, reads, own_melt, n_rocks):
    """Per read and recipe, the per-rock share and presence on the draws both
    recipes reach there, with the draw counts behind them. 50+30 pools each
    draw's two rung rows over the draws both recipes bring to both rungs."""
    out, counts = {}, {}
    ends = {}
    for rec in RECIPES:
        for r in reads:
            if r in RUNGS:
                ends[rec, r] = rung_ends(points[rec], int(r) / 100)
            elif r == "last":
                ends[rec, r] = last_ends(points[rec])
            elif r == "own_melt":
                ends[rec, r] = rung_ends(points[rec], own_melt)
        if "50+30" in reads:
            for p in ("50", "30"):
                ends.setdefault((rec, p), rung_ends(points[rec], int(p) / 100))
    for r in reads:
        parts = ("50", "30") if r == "50+30" else (r,)
        keys = matched([ends[rec, p] for rec in RECIPES for p in parts])
        counts[r] = {**{rec: int(len(matched([ends[rec, p] for p in parts]))) for rec in RECIPES},
                     "matched": int(len(keys))}
        for rec in RECIPES:
            rows = pd.concat([read_rows(points[rec], ends[rec, p], keys) for p in parts])
            out[rec, r] = rock_reads(rows, n_rocks)
    return out, counts


def read_set(reads, rows):
    """The rocks in `rows` with a read at every (recipe, read)."""
    ok = rows.copy()
    for share, _ in reads.values():
        ok &= np.isfinite(share).all(1)
    return ok


def align(benchmark, comps):
    """The benchmark's truth placed on the composition table's rows (by key),
    so every array indexes by `bulk`; rows the benchmark does not evaluate
    carry no quantified category and `evaluated` False."""
    n, k = len(comps), len(GROUPS)
    at = comps.set_index("key").loc[benchmark.bulk_ids, "bulk"].to_numpy()
    out = {"magnitude": np.zeros((n, k)), "numeric": np.zeros((n, k), dtype=bool),
           "presence": np.zeros((n, k)), "measured": np.zeros((n, k), dtype=bool),
           "unit": np.full(n, None, dtype=object), "evaluated": np.zeros(n, dtype=bool)}
    for name in ("magnitude", "numeric", "presence", "measured", "unit"):
        out[name][at] = getattr(benchmark, name)
    out["evaluated"][at] = True
    return out, at


def errors(shares, magnitude, numeric, idx):
    """Per source, the absolute error per category and its means on `idx`."""
    return {name: with_means(share_metrics(s[idx], magnitude[idx], numeric[idx])[0])
            for name, s in shares.items()}


def bootstrap(shares, magnitude, numeric, units, n_boot=N_BOOT, seed=0):
    """Unit-bootstrap replicates of every source's error (the resampling of
    `cnmineral.eval.real_rock.unit_bootstrap`, kept here because the win
    counts need the replicates): returns the estimate, the replicates per
    source (n_boot x labels) and, per recipe, the fraction of replicates in
    which each read has the lowest mean-of-seven error."""
    est = errors(shares, magnitude, numeric, np.arange(len(units)))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(units == u) for u in np.unique(units)]
    reps = {name: np.empty((n_boot, len(OUT_LABELS))) for name in shares}
    for b in range(n_boot):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        for name, e in errors(shares, magnitude, numeric, idx).items():
            reps[name][b] = e
    m7 = OUT_LABELS.index("mean7")
    wins = {}
    for rec in RECIPES:
        reads = [r for (source, r) in (k for k in shares if isinstance(k, tuple)) if source == rec]
        best = np.argmin(np.stack([reps[rec, r][:, m7] for r in reads], 1), 1)
        wins[rec] = {r: float((best == i).mean()) for i, r in enumerate(reads)}
    return est, reps, wins


def band(reps, q=(2.5, 97.5)):
    with np.errstate(invalid="ignore"):
        return np.nanpercentile(reps, q[0], axis=0), np.nanpercentile(reps, q[1], axis=0)


def table(shares, presence, est, reps, magnitude, numeric, present, measured):
    """One row per (source, read, category or mean): error with its band,
    fractional minus equilibrium with its band, Spearman and presence AUROC
    (presence only where the source carries a probability)."""
    rows = []
    for name, share in shares.items():
        source, read = (name if isinstance(name, tuple) else (name, ""))
        lo, hi = band(reps[name])
        q25, q75 = band(reps[name], (25, 75))
        rho = with_means(spearman(share, magnitude, numeric))
        auc = (with_means(presence_auroc(presence[name], present, measured)) if name in presence
               else np.full(len(OUT_LABELS), np.nan))
        n = share_metrics(share, magnitude, numeric)[2]
        diff = None
        if source == "fractional":
            d = reps[name] - reps["equilibrium", read]
            diff = (est[name] - est["equilibrium", read], *band(d))
        for i, label in enumerate(OUT_LABELS):
            row = dict(source=source, read=read, category=label,
                       n_rocks=int(n[i]) if i < len(GROUPS) else None,
                       mae=est[name][i], mae_lo=lo[i], mae_hi=hi[i], mae_q25=q25[i], mae_q75=q75[i],
                       spearman=rho[i], auroc=auc[i])
            if diff is not None:
                row.update(fc_minus_eq=diff[0][i], fc_minus_eq_lo=diff[1][i], fc_minus_eq_hi=diff[2][i])
            rows.append(row)
    return pd.DataFrame(rows)


def norm_input(comps):
    """The compositions with total iron as FeO on every row, since
    `cipw_shares` reads `feo_tot` and a few rocks report their iron as Fe2O3."""
    out = comps.copy()
    out["feo_tot"] = out["feo_tot"].fillna(out["fe2o3_tot"] * FE2O3_TO_FEO_MASS)
    return out


def head_marginal(benchmark, device):
    ckpt = JointHeadCheckpoint.load(REPO / "checkpoints" / PRODUCTION / "heads_best.pt")
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)
    head = ckpt.build_mode_head().to(device)
    return marginal_shares(head, latents, benchmark.numeric.astype(np.float64))


def run_facts(work):
    """Counts any run carries, whichever rung read-outs its draws table has."""
    draws = run_read.load_draws(work)
    return dict(work=str(work.relative_to(REPO)), n_chunks=len(run_read.read_manifest(work)),
                n_rocks=int(draws["bulk"].nunique()), n_draws=int(len(draws)),
                mean_failed_per_draw=round(float(draws["n_failed"].mean()), 3))


def load(name):
    """The evaluated benchmark and its composition table."""
    if name == "volcanic":
        benchmark = load_benchmark(BENCHMARK_DIR)
        evaluated = benchmark.subset(benchmark.clean & benchmark.has_magnitude)
    else:
        with plutonic_benchmark() as benchmark:
            evaluated = benchmark.subset(benchmark.has_magnitude)
    return evaluated, load_compositions(RESULTS / BENCHMARKS[name]["compositions"])


def plot_recipe_rungs(tab, reads, benchmark, out):
    style, spec = setup_style()
    colors = {"equilibrium": style.CIVIDIS[2], "fractional": style.CIVIDIS[8]}
    line = [r for r in reads if r in RUNGS]
    x = {r: float(i) for i, r in enumerate(line)}
    for i, r in enumerate(r for r in reads if r not in RUNGS):
        x[r] = len(line) + 0.7 + i
    offset = {"equilibrium": -0.17, "fractional": 0.17}
    by = tab.set_index(["source", "read", "category"])
    fig, axes = panel_grid(style)
    for i, (ax, c) in enumerate(zip(axes.ravel(), PANELS)):
        for rec in RECIPES:
            y = np.array([100 * by.loc[(rec, r, c), "mae"] for r in reads])
            lo = np.array([100 * by.loc[(rec, r, c), "mae_q25"] for r in reads])
            hi = np.array([100 * by.loc[(rec, r, c), "mae_q75"] for r in reads])
            xs = np.array([x[r] for r in reads]) + offset[rec]
            k = len(line)
            ax.plot(xs[:k], y[:k], "-", color=colors[rec], lw=1.2)
            ax.errorbar(xs, y, yerr=[y - lo, hi - y], fmt="o", color=colors[rec],
                        ms=3.5, elinewidth=0.9, capsize=1.5)
        ax.set_xticks([x[r] for r in reads], [READ_LABELS[r] for r in reads], rotation=90)
        ax.set_xlim(-0.7, max(x.values()) + 0.7)
        lo_, hi_ = ax.get_ylim()
        if hi_ - lo_ < MIN_SPAN:
            mid = (hi_ + lo_) / 2
            ax.set_ylim(mid - MIN_SPAN / 2, mid + MIN_SPAN / 2)
        ax.set_title(f"({chr(97 + i)}) {LABELS[c]}")
        style.frame(ax)
        ax.xaxis.set_minor_locator(plt.NullLocator())
    fig.suptitle(f"Raw simulator error on train and validation rocks ({benchmark})", y=0.995,
                 fontsize=spec["label"] + 2)
    fig.supxlabel("Melt fraction at the read (%)", y=0.085, fontsize=spec["label"])
    fig.supylabel("Abundance MAE (%)", x=0.012, fontsize=spec["label"])
    handles = [Line2D([], [], color=colors[rec], marker="o", ms=3.5, label=label)
               for rec, label in (("equilibrium", "Equilibrium"),
                                  ("fractional", "Fractional crystallization; bars span the unit-bootstrap interquartile range"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=2,
               handlelength=2.6)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=tuple(BENCHMARKS), default="volcanic")
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    config = BENCHMARKS[args.benchmark]
    works = dict(zip(RECIPES, (WORK / w for w in config["works"])))
    reads = tuple(r for r in READS if r != "own_melt" or config["own_melt"])

    evaluated, comps = load(args.benchmark)
    n = len(comps)
    if not (comps["bulk"].to_numpy() == np.arange(n)).all():
        raise SystemExit(f"{config['compositions']} bulk ids are not row positions")
    truth, at = align(evaluated, comps)
    magnitude, numeric = truth["magnitude"], truth["numeric"]

    points = {rec: run_read.shares_and_distances(run_read.build_points(work, TABLES[rec]), magnitude, numeric)
              for rec, work in works.items()}
    # the recipe comparison pairs draws by index, so both runs must have drawn
    # the same conditions for every (bulk, draw)
    conditions = [run_read.load_draws(work).set_index(["bulk", "draw"])[["P_kbar", "H2O"]].sort_index()
                  for work in works.values()]
    if set(conditions[0].index.get_level_values("bulk")) != set(range(n)):
        raise SystemExit(f"the runs do not cover the {n} rocks of {config['compositions']}")
    if not conditions[0].equals(conditions[1]):
        raise SystemExit("the two runs did not draw the same (P, H2O) for every (bulk, draw)")
    own_melt = pd.Series(1 - comps["total_crystals_pct"].to_numpy() / 100, index=comps["bulk"]).dropna()
    rock_level, counts = recipe_reads(points, reads, own_melt, n)

    split = truth["evaluated"] & comps["subset"].isin(SPLITS).to_numpy()
    rows = read_set(rock_level, split)
    dropped = comps.loc[split & ~rows, ["key", "total_crystals_pct"]]
    shares = {key: share for key, (share, _) in rock_level.items()}
    presence = {key: p for key, (_, p) in rock_level.items()}
    shares["head_marginal"] = np.full((n, len(GROUPS)), np.nan)
    shares["head_marginal"][at] = head_marginal(evaluated, pick_device())
    shares["cipw_norm"] = renormalize(cipw_shares(norm_input(comps)), numeric)
    for b in BASELINES:
        if not np.isfinite(shares[b][rows]).all():
            raise SystemExit(f"{b} has no read for some rocks in the read set")

    sub = {k: v[rows] for k, v in shares.items()}
    units = truth["unit"][rows].astype(str)
    est, reps, wins = bootstrap(sub, magnitude[rows], numeric[rows], units)
    tab = table(sub, {k: v[rows] for k, v in presence.items()}, est, reps, magnitude[rows],
                numeric[rows], truth["presence"][rows], truth["measured"][rows])

    m7 = OUT_LABELS.index("mean7")
    print(f"{args.benchmark}: {int(rows.sum())} train and val rocks in {len(np.unique(units))} units; "
          f"{len(dropped)} dropped with no read somewhere")
    for rec in RECIPES:
        print(f"{rec}, mean of seven: " + ", ".join(
            f"{READ_LABELS[r]} {est[rec, r][m7]:.4f}" for r in reads))
        print(f"{rec}, lowest in: " + ", ".join(
            f"{READ_LABELS[r]} {w:.2f}" for r, w in wins[rec].items() if w > 0))
    fc = tab[(tab["source"] == "fractional") & (tab["category"] == "mean7")].set_index("read")
    print("fractional minus equilibrium, mean of seven: " + ", ".join(
        f"{READ_LABELS[r]} {fc.loc[r, 'fc_minus_eq']:+.4f} [{fc.loc[r, 'fc_minus_eq_lo']:+.4f}, "
        f"{fc.loc[r, 'fc_minus_eq_hi']:+.4f}]" for r in reads))
    print("baselines, mean of seven: " + ", ".join(f"{b} {est[b][m7]:.4f}" for b in BASELINES))

    args.results.mkdir(parents=True, exist_ok=True)
    stem = args.results / f"recipe_rungs_{args.benchmark}"
    tab.to_csv(stem.with_suffix(".csv"), index=False)
    payload = {
        "settings": {"benchmark": args.benchmark, "rocks": int(rows.sum()), "units": int(len(np.unique(units))),
                     "splits": list(SPLITS), "test": "sealed", "reads": list(reads), "head": PRODUCTION,
                     "presence_cutoff": PRESENCE_CUTOFF, "n_boot": N_BOOT, "bootstrap_seed": 0,
                     "tables": TABLES},
        "draws": counts,
        "dropped_rocks": dropped.astype(object).where(dropped.notna(), None).to_dict(orient="records"),
        "lowest_mean7_fraction": wins,
        "runs": {rec: run_facts(work) for rec, work in works.items()},
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=1, default=float))
    plot_recipe_rungs(tab, reads, args.benchmark, stem)
    print(f"wrote {stem}.csv, .json, .png and .pdf")


if __name__ == "__main__":
    main()
