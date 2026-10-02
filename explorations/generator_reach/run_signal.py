"""Does the equilibrium label track rocks (#85)? On the #78 rows, whether the
simulator's per-rock share follows the measured share beyond a constant once
its offset is removed, beside the #79 prior rows, the 5,000-composition head
at 224 steps and a hint at the ruler's own error. No MAGEMin, no training.

  python explorations/generator_reach/run_signal.py

Every source is a per-rock share over the rock's measured mineral list. Per
category on the val rocks quantifying it: `mae`, `pearson_r` across rocks,
and `mae_linear_fit`, the error after a least-squares line from the source's
share to the measured share, fit on the train rocks, clipped and
renormalized over the mineral list (sources already fit to train labels get
none). Intervals are a unit bootstrap over the val units with the fits held
fixed. The ruler read compares du Bray's visual estimate with the DS 1099
point count of the same thin sections (train and val only): two tuff
eruptions, so a hint at the ruler's error, not a floor. Writes
`signal.csv`, `signal.json` and `signal.png` under `--results`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import mixture_conditional_mean
from cnmineral.eval.real_rock import (
    BENCHMARK_DIR, POINTCOUNT_BASIS, VISUAL_BASIS, load_benchmark, split_mask, unit_bootstrap)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.measured_mode_head import ablate_latent, fit_measured_head
from explorations.corpus_scaling.run_depth_grid import MEANS, head_shares, share_metrics, with_means
from explorations.generator_reach import run_read
from explorations.generator_reach.compositions import load_compositions
from explorations.prior_search import run_predictability as predict
from explorations.prior_search.run_prior_screen import NAVY, ORANGE, reweighted_marginals
from explorations.prior_search.schemes import GRID, WIDTHS, density
from explorations.real_mineralogy.run_pointcount_contrast import (
    COVERED_GROUPS, counting_noise, paired_keys, parent_unit, restricted_benchmark_dir,
    section_totals)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORK = HERE / "work" / "broad128"
PRIOR = REPO / "explorations" / "prior_search" / "results"
HEAD = "depth224_allrock_n5000"
SPLIT_COUNTS = {"train": 372, "val": 106, "test": 162}
N_BOOT = 2000
SOURCES = ("simulator", "prior_best_rectangle", "prior_forest_window", "prior_own_hit_window",
           "prior_hit_rows", "constant", "head_zero_shot", "head_recalibrated", "head_no_chemistry")
ORACLE = ("prior_own_hit_window", "prior_hit_rows")
FIT_ON_TRAIN_LABELS = ("constant", "head_recalibrated", "head_no_chemistry")
LABELS = (*GROUPS, *MEANS)


def renormalize(share, numeric):
    """Shares clipped to [0, 1] and renormalized over the mineral list; a row
    with nothing left is zero, a NaN row stays NaN."""
    masked = np.where(numeric, np.clip(share, 0.0, 1.0), 0.0)
    total = masked.sum(1, keepdims=True)
    out = np.divide(masked, total, out=np.zeros_like(masked), where=total > 0)
    out[np.isnan(masked).any(1)] = np.nan
    return out


def constant_vector(magnitude, numeric, rows):
    """The mean measured share per category over `rows` quantifying it."""
    return np.array([magnitude[rows & numeric[:, j], j].mean() if (rows & numeric[:, j]).any()
                     else 0.0 for j in range(len(GROUPS))])


def linear_fit(share, magnitude, numeric, train):
    """Per category, least squares measured ~ a + b x share over the train
    rocks quantifying it (the train mean where the share does not vary),
    applied to every rock, clipped and renormalized over its mineral list."""
    fitted, coef = np.full(share.shape, np.nan), {}
    for j, g in enumerate(GROUPS):
        m = train & numeric[:, j] & np.isfinite(share[:, j])
        x, y = share[m, j], magnitude[m, j]
        if m.sum() >= 3 and np.ptp(x) > 1e-9:
            b, a = np.polyfit(x, y, 1)
        else:
            a, b = (y.mean() if m.any() else 0.0), 0.0
        coef[g] = (float(a), float(b))
        fitted[:, j] = a + b * share[:, j]
    return renormalize(fitted, numeric), coef


def marginal_shares(rungs, weights, bulks):
    """The weighted mean share per category over each rock's rung rows, in
    `bulks` order; NaN for a rock no weighted draw reaches."""
    marg = reweighted_marginals(rungs, weights)
    return marg.reindex(bulks)[[f"mean_{g}" for g in GROUPS]].to_numpy()


def table_stats(shares, fitted, magnitude, numeric, idx):
    """{(source, metric): values over `LABELS`} on the rows `idx`, with every
    source's difference from the constant on the same rows."""
    out = {}
    for name, share in shares.items():
        mae, r, _ = share_metrics(share[idx], magnitude[idx], numeric[idx])
        out[name, "mae"], out[name, "pearson_r"] = with_means(mae), with_means(r)
        if name in fitted:
            out[name, "mae_linear_fit"] = with_means(
                share_metrics(fitted[name][idx], magnitude[idx], numeric[idx])[0])
    base = out["constant", "mae"]
    for name in shares:
        out[name, "mae_minus_constant"] = out[name, "mae"] - base
        if name in fitted:
            out[name, "mae_linear_fit_minus_constant"] = out[name, "mae_linear_fit"] - base
    return out


def bootstrap_table(shares, fitted, magnitude, numeric, units, n_boot=N_BOOT):
    """One row per (source, label): every metric with its unit-bootstrap
    band. The arrays are the val subset's own, so the bootstrap's positions
    index them directly."""
    everything = np.arange(len(units))
    keys = list(table_stats(shares, fitted, magnitude, numeric, everything))

    def stat(idx):
        s = table_stats(shares, fitted, magnitude, numeric, idx)
        return np.concatenate([s[k] for k in keys])

    est = stat(everything)
    lo, hi = unit_bootstrap(stat, units, n_boot=n_boot)
    k = len(LABELS)
    band = {key: (est[i * k:(i + 1) * k], lo[i * k:(i + 1) * k], hi[i * k:(i + 1) * k])
            for i, key in enumerate(keys)}
    counts = {name: share_metrics(share, magnitude, numeric)[2] for name, share in shares.items()}
    quantifies = {label: numeric[:, [GROUP_INDEX[g] for g in groups]].any(1).sum()
                  for label, groups in MEANS.items()}
    rows = []
    for name in shares:
        for i, label in enumerate(LABELS):
            row = dict(source=name, category=label, oracle=name in ORACLE,
                       n_rocks=int(counts[name][i]) if i < len(GROUPS) else int(quantifies[label]))
            for (source, metric), (e, a, b) in band.items():
                if source == name:
                    row.update({metric: e[i], f"{metric}_lo": a[i], f"{metric}_hi": b[i]})
            rows.append(row)
    return pd.DataFrame(rows)


def ruler_disagreement(visual_mag, visual_num, counted_mag, counted_num):
    """Per covered category over the rocks both rulers quantify it on: the
    mean absolute and the signed (visual minus point count) share difference."""
    out = {}
    for g in COVERED_GROUPS:
        j = GROUP_INDEX[g]
        both = visual_num[:, j] & counted_num[:, j]
        diff = visual_mag[both, j] - counted_mag[both, j]
        out[g] = {"n_rocks": int(both.sum()),
                  "ruler_disagreement": float(np.abs(diff).mean()) if both.any() else None,
                  "signed_mean": float(diff.mean()) if both.any() else None,
                  "signed_median": float(np.median(diff)) if both.any() else None}
    return out


def broad_weights(draws):
    return draws[["bulk", "draw"]].assign(w=1.0)


def prior_sources(points, draws, comps, numeric, train):
    """The #79 table's rows as per-rock shares: the broad box (the simulator's
    own average), the best rectangle, and the predictability read's windows,
    rebuilt with the prior_search functions."""
    rungs = run_read.rung_rows(points, draws)
    bulks = comps["bulk"].to_numpy()
    schemes = pd.read_csv(PRIOR / "schemes.csv")
    best = schemes[schemes["kind"] == "grid"]["name"].iloc[0]
    scheme = next(s for s in GRID if s.name == best)
    p, h, q = (draws[c].to_numpy(dtype=float) for c in ("P_kbar", "H2O", "q"))
    rectangle = draws[["bulk", "draw"]].assign(w=density(scheme, p, h) / q)

    rungs["e_sil"] = predict.silicate_error(rungs, numeric)
    hits, summary = predict.hit_rows(rungs, draws, tau=0.05, fallback=8)
    summary = summary.reindex(comps["bulk"])
    preds = predict.predictors(predict.features(comps), comps, summary, train)
    wide = WIDTHS["wide"]
    shares = {
        "simulator": marginal_shares(rungs, broad_weights(draws), bulks),
        "prior_best_rectangle": marginal_shares(rungs, rectangle, bulks),
        "prior_forest_window": marginal_shares(
            rungs, predict.window_weights(draws, preds["forest"], *wide), bulks),
        "prior_own_hit_window": marginal_shares(
            rungs, predict.window_weights(draws, summary[list(predict.CENTER)], *wide), bulks),
        "prior_hit_rows": marginal_shares(rungs, predict.hit_weights(draws, hits), bulks),
    }
    return shares, best


def head_sources(benchmark, mask, train_mask, device):
    """The 5,000-composition head at 224 steps, zero-shot and recalibrated,
    and the head given no chemistry, as per-rock shares on the evaluated
    rocks."""
    ckpt = JointHeadCheckpoint.load(REPO / "checkpoints" / HEAD / "heads_best.pt")
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)
    gp, _, zero, recal = head_shares(ckpt, benchmark, latents, train_mask, device)
    nochem = fit_measured_head(ablate_latent(gp.subset(train_mask)), weight_decay=10.0)
    with torch.no_grad():
        _, alpha, log_w = nochem(torch.zeros_like(gp.latent), gp.presence)
    none = mixture_conditional_mean(alpha.numpy(), log_w.numpy(), benchmark.numeric.astype(np.float64))
    return {"head_zero_shot": zero[mask], "head_recalibrated": recal[mask],
            "head_no_chemistry": none[mask]}


def reproduction(table):
    """The rebuilt #79 rows against the numbers prior_search and run_read
    committed; returns the largest absolute difference per check."""
    sil = table[(table["category"] == "silicate")].set_index("source")["mae"]
    schemes = pd.read_csv(PRIOR / "schemes.csv").set_index("name")
    best = schemes[schemes["kind"] == "grid"].index[0]
    pred = pd.read_csv(PRIOR / "predictability.csv").set_index("name")["silicate_val"]
    dist = pd.read_csv(HERE / "results" / "distances.csv")
    ref = dist[dist["subset"] == "val"].groupby("category")["d_marginal"].mean()
    per_cat = table[(table["source"] == "simulator") & table["category"].isin(GROUPS)].set_index("category")["mae"]
    return {"broad box": abs(sil["simulator"] - schemes.loc["broad", "silicate_val"]),
            "best rectangle": abs(sil["prior_best_rectangle"] - schemes.loc[best, "silicate_val"]),
            "hit rows": abs(sil["prior_hit_rows"] - pred["hit rows"]),
            "own hit window": abs(sil["prior_own_hit_window"] - pred["own hit center, wide"]),
            "forest window": abs(sil["prior_forest_window"] - pred["forest, wide"]),
            "distances.csv per category": float((per_cat - ref.reindex(per_cat.index)).abs().max())}


def ruler_read(benchmark, mask, comps, raw_points, draws):
    """Visual estimate against point count on the paired rocks outside test,
    and the simulator's and a constant's error against each."""
    paired = paired_keys()
    with tempfile.TemporaryDirectory() as tmp:
        restricted = restricted_benchmark_dir(Path(tmp), paired)
        visual = load_benchmark(restricted, prefer_basis=VISUAL_BASIS)
        counted = load_benchmark(restricted, prefer_basis=POINTCOUNT_BASIS)
    for b in (visual, counted):
        if not np.array_equal(b.bulk_ids, benchmark.bulk_ids):
            raise SystemExit("the restricted benchmark reorders the rocks")
    keys, subset = benchmark.bulk_ids[mask], comps["subset"].to_numpy()
    in_both = (visual.clean & visual.has_magnitude & counted.clean & counted.has_magnitude)[mask]
    rows = np.isin(keys, sorted(paired)) & in_both & (subset != "test")
    fit_rows = (subset == "train") & ~np.isin(keys, sorted(paired))
    arms = {"visual": visual, "pointcount": counted}
    truth = {name: (b.magnitude[mask], b.numeric[mask]) for name, b in arms.items()}
    const = constant_vector(*truth["visual"], fit_rows)
    bulks = comps["bulk"].to_numpy()
    raw_points = raw_points[raw_points["bulk"].isin(bulks[rows])]
    draws = draws[draws["bulk"].isin(bulks[rows])]

    error = {}
    for name, (mag, num) in truth.items():
        pts = run_read.shares_and_distances(raw_points, mag, num)
        sim = marginal_shares(run_read.rung_rows(pts, draws), broad_weights(draws), bulks)
        for source, share in (("simulator", sim), ("constant", renormalize(np.tile(const, (len(mag), 1)), num))):
            mae, _, n = share_metrics(share[rows], mag[rows], num[rows])
            error[f"{source}_mae_vs_{name}"] = {g: float(mae[GROUP_INDEX[g]]) for g in COVERED_GROUPS}
            error[f"n_rocks_{name}"] = {g: int(n[GROUP_INDEX[g]]) for g in COVERED_GROUPS}

    full = np.zeros(len(benchmark), dtype=bool)
    full[np.flatnonzero(mask)[rows]] = True
    n_points = section_totals().reindex(benchmark.bulk_ids[full]).to_numpy(dtype=float)
    return {"n_rocks": int(rows.sum()),
            "n_train": int((rows & (subset == "train")).sum()),
            "n_val": int((rows & (subset == "val")).sum()),
            "n_eruptions": int(len(np.unique(parent_unit(benchmark.unit[full])))),
            "constant_fit_on": "train rocks outside the paired set, visual estimate",
            "disagreement": ruler_disagreement(truth["visual"][0][rows], truth["visual"][1][rows],
                                               truth["pointcount"][0][rows], truth["pointcount"][1][rows]),
            "counting_noise": counting_noise(counted, full, n_points),
            **error}


def print_table(table):
    sil = table[table["category"] == "silicate"].set_index("source")
    print("four silicates on val: mae [interval], pearson_r, mae_linear_fit, mae minus constant")
    for name in SOURCES:
        r = sil.loc[name]
        lf = (f", linear fit {r['mae_linear_fit']:.3f} [{r['mae_linear_fit_lo']:.3f}, "
              f"{r['mae_linear_fit_hi']:.3f}]" if np.isfinite(r.get("mae_linear_fit", np.nan)) else "")
        print(f"{name}{' (oracle)' if r['oracle'] else ''}: mae {r['mae']:.3f} [{r['mae_lo']:.3f}, "
              f"{r['mae_hi']:.3f}], r {r['pearson_r']:.2f} [{r['pearson_r_lo']:.2f}, "
              f"{r['pearson_r_hi']:.2f}]{lf}, minus constant {r['mae_minus_constant']:+.3f} "
              f"[{r['mae_minus_constant_lo']:+.3f}, {r['mae_minus_constant_hi']:+.3f}]")


def print_ruler(ruler):
    print(f"ruler: {ruler['n_rocks']} paired rocks ({ruler['n_train']} train, {ruler['n_val']} val), "
          f"{ruler['n_eruptions']} eruptions; a hint, not a floor")
    for g in COVERED_GROUPS:
        d = ruler["disagreement"][g]
        if d["n_rocks"] == 0:
            continue
        print(f"{g}: disagreement {d['ruler_disagreement']:.3f} (signed {d['signed_mean']:+.3f}, "
              f"n {d['n_rocks']}); simulator {ruler['simulator_mae_vs_visual'][g]:.3f} vs visual, "
              f"{ruler['simulator_mae_vs_pointcount'][g]:.3f} vs point count; constant "
              f"{ruler['constant_mae_vs_visual'][g]:.3f} / {ruler['constant_mae_vs_pointcount'][g]:.3f}")


def draw_figure(shares, coef, magnitude, numeric, subset, table, out):
    by = table.set_index(["source", "category"])
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4))
    val, train = subset == "val", subset == "train"
    for ax, g in zip(axes.flat, GROUPS):
        j = GROUP_INDEX[g]
        m = train & numeric[:, j]
        ax.scatter(shares["simulator"][m, j], magnitude[m, j], s=6, color="#c8c8c8", lw=0,
                   label="simulator, train rocks")
        m = val & numeric[:, j]
        for name, color, label in (("simulator", NAVY, "simulator, val"),
                                   ("head_zero_shot", ORANGE, "head zero-shot, val")):
            ax.scatter(shares[name][m, j], magnitude[m, j], s=12, color=color, alpha=0.75, lw=0,
                       label=label)
        a, b = coef["simulator"][g]
        ax.plot([0, 1], [a, a + b], color=NAVY, ls="--", lw=1, label="simulator linear fit (train)")
        ax.plot([0, 1], [0, 1], color="#8a8a8a", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        sim, head = by.loc[("simulator", g)], by.loc[("head_zero_shot", g)]
        ax.set_title(f"{g} (n {int(sim['n_rocks'])}): r {sim['pearson_r']:.2f} simulator, "
                     f"{head['pearson_r']:.2f} head", fontsize=9)
    for ax in axes[1, :3]:
        ax.set_xlabel("simulated or predicted share", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("measured share", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="upper left")

    ax = axes[1, 3]
    sil = table[table["category"] == "silicate"].set_index("source").loc[list(SOURCES)]
    y = np.arange(len(sil))
    ax.errorbar(sil["mae"], y, xerr=[sil["mae"] - sil["mae_lo"], sil["mae_hi"] - sil["mae"]],
                fmt="o", color=NAVY, capsize=2, label="mae")
    fit = sil["mae_linear_fit"].notna()
    ax.errorbar(sil.loc[fit, "mae_linear_fit"], y[fit.to_numpy()],
                xerr=[(sil["mae_linear_fit"] - sil["mae_linear_fit_lo"])[fit],
                      (sil["mae_linear_fit_hi"] - sil["mae_linear_fit"])[fit]],
                fmt="o", mfc="none", color=ORANGE, capsize=2, label="mae_linear_fit")
    ax.axvline(sil.loc["constant", "mae"], color="#8a8a8a", ls="--", lw=1, label="constant")
    ax.set_yticks(y, [f"{n}{' *' if n in ORACLE else ''}" for n in sil.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("four-silicate absolute error, val (* oracle)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle("does the equilibrium label track rocks? measured against simulated share per "
                 "category, val rocks; bars = unit-bootstrap 95% over the 39 val units", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "signal.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    benchmark = load_benchmark(BENCHMARK_DIR)
    mask = benchmark.clean & benchmark.has_magnitude
    comps = load_compositions(HERE / "results" / "compositions.csv")
    if list(comps["key"]) != list(benchmark.bulk_ids[mask]):
        raise SystemExit("results/compositions.csv does not match the benchmark's evaluated rocks")
    subset = comps["subset"].to_numpy()
    counts = {s: int((subset == s).sum()) for s in SPLIT_COUNTS}
    if counts != SPLIT_COUNTS:
        raise SystemExit(f"split counts {counts} are not the eruption benchmark's {SPLIT_COUNTS}")
    magnitude, numeric = benchmark.magnitude[mask], benchmark.numeric[mask]
    train, val = subset == "train", subset == "val"

    raw_points = run_read.build_points(args.work)
    points = run_read.shares_and_distances(raw_points, magnitude, numeric)
    draws = run_read.load_draws(args.work)
    shares, best = prior_sources(points, draws, comps, numeric, train)
    const = constant_vector(magnitude, numeric, train)
    shares["constant"] = renormalize(np.tile(const, (len(magnitude), 1)), numeric)
    train_mask = split_mask(benchmark, "train") & mask
    shares.update(head_sources(benchmark, mask, train_mask, pick_device()))
    shares = {name: shares[name] for name in SOURCES}

    fitted, coef = {}, {}
    for name in SOURCES:
        if name not in FIT_ON_TRAIN_LABELS:
            fitted[name], coef[name] = linear_fit(shares[name], magnitude, numeric, train)

    units = benchmark.unit[mask][val]
    table = bootstrap_table({k: v[val] for k, v in shares.items()},
                            {k: v[val] for k, v in fitted.items()},
                            magnitude[val], numeric[val], units)
    checks = reproduction(table)
    print("reproduction of the committed #79 numbers, max abs diff:",
          {k: f"{v:.1e}" for k, v in checks.items()})
    if max(checks.values()) > 1e-9:
        raise SystemExit("the rebuilt prior rows do not reproduce the committed numbers")

    ruler = ruler_read(benchmark, mask, comps, raw_points, draws)
    print_table(table)
    print_ruler(ruler)

    table.to_csv(args.results / "signal.csv", index=False)
    (args.results / "signal.json").write_text(json.dumps({
        "settings": {"head": HEAD, "best_rectangle": best, "n_boot": N_BOOT, "bootstrap_seed": 0,
                     "val_rocks": int(val.sum()), "val_units": int(len(np.unique(units))),
                     "split_counts": counts, "hit_rows": {"tau": 0.05, "fallback": 8},
                     "fits_on": "train rocks", "reads_on": "val rocks", "test": "sealed"},
        "reproduction_max_abs_diff": checks,
        "constant_train_mean": dict(zip(GROUPS, const.tolist())),
        "linear_fit_coefficients": {name: {g: {"a": a, "b": b} for g, (a, b) in c.items()}
                                    for name, c in coef.items()},
        "ruler": ruler,
    }, indent=1, default=float))
    draw_figure(shares, coef, magnitude, numeric, subset, table, args.results)
    print(f"wrote {args.results / 'signal.csv'}, signal.json, signal.png")


if __name__ == "__main__":
    main()
