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

`--benchmark plutonic` is #83's step 0 and Stage A: the production head on
the plutonic benchmark, zero-shot and under the matrix recalibrator, against
a ladder of baselines that use no simulator (the constant, the head given no
chemistry, the CIPW norm read like any source, the head given raw
geochemistry), scored by share error, Spearman, presence AUROC and joint NLL
with unit-bootstrap bands over the intrusions. No straight line. Every
fitted source is cross-fit: grouped 5-fold over the train and val rocks,
intrusions kept whole, each rock read once out of fold; test stays sealed.
Rocks whose Fe-Ti oxides were lumped with the accessories are read with the
oxides unmeasured. Stage A adds the generator run under `--work` (default
`work/plutonic128`) read at 50, 30, 10 and 5% melt and the last converged
step (the rung ends re-derived from the kept ladder and checked against the
run's own), each zero-shot and recalibrated, two joint fits asking whether
the generator adds information beyond the CIPW norm (a two-source matrix
recalibrator on the norm whose generator terms start and shrink at zero, and
the measured head given the norm, the generator, and both), the head's
marginal read (its share averaged over assemblages sampled from its own
presence probabilities, the same question the generator's draw mean
answers, beside the teacher-forced read the production figures use), and
the head's copying error: that marginal against the generator's at each
rung on the same rocks. It writes only the `plutonic` key of
`signal.json`, which the eruption read keeps, and `plutonic_stage_a.png`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import logging
import sys
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pyrolite.mineral.normative import CIPW_norm
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS, PRESENCE_CUTOFF
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean, mixture_conditional_mean
from cnmineral.eval.real_rock import (
    BENCHMARK_DIR, POINTCOUNT_BASIS, VISUAL_BASIS, coarsen_alpha, load_benchmark, split_mask,
    unit_bootstrap)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.measured_mode_head import (
    ablate_latent, fit_measured_head, score_measured_head, use_raw_chemistry)
from cnmineral.models.real_rock_recalibrator import RealRockRecalibrator
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions, fit_recalibrator, recalibrator_nll, score_predictions,
    score_recalibrator)
from explorations.corpus_scaling.run_depth_grid import MEANS, head_shares, share_metrics, with_means
from explorations.generator_reach import run_read
from explorations.generator_reach.compositions import load_compositions
from explorations.prior_search import run_predictability as predict
from explorations.prior_search.run_prior_screen import NAVY, ORANGE, reweighted_marginals
from explorations.prior_search.schemes import GRID, WIDTHS, density
from explorations.real_mineralogy.harvest import PLUTONIC_DIR
from explorations.real_mineralogy.plutonic import plutonic_benchmark
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

PRODUCTION = "joint_heads_allrock_tuned_seed0"
PLUTONIC_SPLIT_COUNTS = {"train": 320, "val": 84, "test": 171}
N_FOLDS = 5
STEP0_SOURCES = ("head_zero_shot", "head_marginal", "head_recalibrated", "cipw_zero_shot",
                 "cipw_recalibrated", "constant", "head_no_chemistry", "head_raw_chemistry")
# support samples behind the head's marginal share
N_SUPPORT_SAMPLES = 32
PLUTONIC_WORK = HERE / "work" / "plutonic128"
STAGE_A_RUNGS = ("50", "30", "10", "5", "last")
# the reach read keeps #83's near-solidus rows, so its number is comparable
# across reads that add rungs
SOLIDUS_RUNGS = ("10", "5", "last")
# each joint source against the single-source arms it nests
JOINT_PAIRS = {
    **{f"joint_recal_norm_generator_{r}": ("cipw_recalibrated", f"generator_{r}_recalibrated")
       for r in STAGE_A_RUNGS},
    **{f"head_given_norm_generator_{r}": ("head_given_norm", f"head_given_generator_{r}")
       for r in STAGE_A_RUNGS}}
BASELINES = ("constant", "head_no_chemistry", "cipw_recalibrated", "head_raw_chemistry")
# `explorations/real_rock_fewshot/run_curve.py`'s WEIGHT_DECAY, selected on the eruption val
# split; not re-selected here, where val is the split being read
MEASURED_WEIGHT_DECAY = {"head_no_chemistry": 10.0, "head_raw_chemistry": 1.0}
PACKAGE_FLOOR = 1e-3
# CIPW summary minerals (not their end-member components, which would count twice) -> group;
# apatite, corundum, titanite, zircon and the feldspathoids fall outside the seven groups
CIPW_TO_GROUP = {"quartz": "q", "albite": "fsp", "anorthite": "fsp", "orthoclase": "fsp",
                 "diopside": "px", "hypersthene": "px", "olivine": "ol",
                 "magnetite": "opq", "ilmenite": "opq", "hematite": "opq"}
CIPW_OXIDES = {"sio2": "SiO2", "tio2": "TiO2", "al2o3": "Al2O3", "feo_tot": "FeO", "mno": "MnO",
               "mgo": "MgO", "cao": "CaO", "na2o": "Na2O", "k2o": "K2O", "p2o5": "P2O5"}


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


def spearman(share, magnitude, numeric):
    """Per category over the rows quantifying it, as `share_metrics` counts
    them: Pearson on average ranks; NaN where either side does not vary."""
    out = np.full(len(GROUPS), np.nan)
    for j in range(len(GROUPS)):
        m = numeric[:, j] & np.isfinite(share[:, j])
        if m.sum() < 3:
            continue
        x, y = rankdata(share[m, j]), rankdata(magnitude[m, j])
        if np.ptp(x) > 0 and np.ptp(y) > 0:
            out[j] = np.corrcoef(x, y)[0, 1]
    return out


def presence_auroc(p, presence, measured):
    """Per category, the rank (Mann-Whitney) AUROC of `p` against measured
    presence over the rocks that looked for it; NaN where every such rock or
    none carries the category."""
    out = np.full(len(GROUPS), np.nan)
    for j in range(len(GROUPS)):
        m = measured[:, j]
        y = presence[m, j] > 0.5
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        if n_pos and n_neg:
            out[j] = (rankdata(p[m, j])[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def cipw_shares(samples):
    """The CIPW norm of each row's majors as group mass fractions over the
    seven groups; weight percent in, so on the benchmark's mass basis. Iron is
    all FeO in these sources, so pyrolite splits it with its default Le Maitre
    correction, and an unreported minor oxide counts as zero."""
    chem = samples[list(CIPW_OXIDES)].rename(columns=CIPW_OXIDES).reset_index(drop=True)
    chem[["TiO2", "MnO", "P2O5"]] = chem[["TiO2", "MnO", "P2O5"]].fillna(0.0)
    # pyrolite warns on every call that Fe2O3 is absent, which is this input by construction
    logging.getLogger("pyrolite").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm = CIPW_norm(chem)
    out = np.zeros((len(chem), len(GROUPS)))
    for mineral, group in CIPW_TO_GROUP.items():
        out[:, GROUP_INDEX[group]] += norm[mineral].fillna(0.0).to_numpy()
    total = out.sum(1, keepdims=True)
    return np.divide(out, total, out=np.full_like(out, np.nan), where=total > 0)


def packaged(gp, share, presence_p, kappa, floor=PACKAGE_FLOOR):
    """`gp` with its forecast swapped for a source that has shares but no
    probability model: presence `presence_p` and concentrations `kappa` times
    the floored shares. Truth, units and inputs stay `gp`'s, so the
    recalibrator fits and scores it on the same rocks. The floor keeps every
    group revivable, since the recalibrator corrects multiplicatively and a
    zero concentration stays zero."""
    p = np.clip(presence_p, floor, 1 - floor)
    s = np.maximum(np.nan_to_num(share), floor)
    s = s / s.sum(1, keepdims=True)
    return replace(gp, logits7=torch.as_tensor(np.log(p / (1 - p)), dtype=torch.float64),
                   alpha7=torch.as_tensor(kappa * s, dtype=torch.float64), log_w=None)


class TwoSourceRecalibrator(nn.Module):
    """The production matrix correction of source A's forecast, plus 7x7
    terms reading source B's presence logits and log concentrations. B's
    terms start at zero and weight decay pulls them back there, so the
    nested null is exactly A recalibrated alone and any gain is what B adds
    beyond A under a linear combination in log space."""

    def __init__(self):
        super().__init__()
        n = len(GROUPS)
        self.base = RealRockRecalibrator("matrix")
        self.presence_mix_b = nn.Parameter(torch.zeros(n, n))
        self.magnitude_mix_b = nn.Parameter(torch.zeros(n, n))

    def forward(self, logits_a, alpha_a, logits_b, alpha_b):
        logits, alpha, _ = self.base(logits_a, alpha_a)
        return (logits + logits_b @ self.presence_mix_b,
                alpha * (alpha_b.log() @ self.magnitude_mix_b).exp())


def fit_two_source(a, b, weight_decay=1.0, epochs=400, lr=0.05, seed=0):
    """`fit_recalibrator`'s protocol on the two-source module; `a` carries
    the truth, and both are packaged forecasts with every concentration
    floored above zero."""
    torch.manual_seed(seed)
    model = TwoSourceRecalibrator().to(a.logits7.dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        logits, alpha = model(a.logits7, a.alpha7, b.logits7, b.alpha7)
        loss = recalibrator_nll(logits, alpha, a.presence, a.magnitude, a.numeric)["joint"].mean()
        loss.backward()
        opt.step()
    return model.eval()


def generator_reads(work, comps, benchmark, table="ladder"):
    """Per Stage A rung, on the evaluated rocks in benchmark order: the
    zero-shot share (the mean, over the draws reaching the rung, of each
    draw's share over the measured mineral list), the fraction of those draws
    carrying each category above the label presence cutoff, and their mean
    share of the seven-category crystal mass. A rung's end is the highest-T
    converged point of each draw with melt below it, read from the kept
    ladder and checked against the run's own read-outs where it recorded
    them. A draw that never reaches a rung drops out; a rock no draw brings
    to it is read at its draws' last converged steps. Returns the reads, the
    rung rows for the reach read and the attrition counts."""
    bulk = comps.set_index("key").loc[benchmark.bulk_ids, "bulk"].to_numpy()
    magnitude = np.zeros((len(comps), len(GROUPS)))
    numeric = np.zeros((len(comps), len(GROUPS)), dtype=bool)
    magnitude[bulk], numeric[bulk] = benchmark.magnitude, benchmark.numeric
    draws = run_read.load_draws(work)
    draws = draws[draws["bulk"].isin(bulk)]
    if set(draws["bulk"]) != set(bulk):
        raise SystemExit(f"{work} lacks {len(set(bulk) - set(draws['bulk']))} evaluated rocks")
    raw = run_read.build_points(work, table)
    points = run_read.shares_and_distances(raw[raw["bulk"].isin(bulk)], magnitude, numeric)
    mass = points[[f"m_{g}" for g in GROUPS]].to_numpy()
    total = mass.sum(1, keepdims=True)
    seven = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
    for i, g in enumerate(GROUPS):
        points[f"t_{g}"] = seven[:, i]
        points[f"p_{g}"] = (seven[:, i] > PRESENCE_CUTOFF).astype(float)

    subset = comps.set_index("bulk")["subset"]
    last = draws[["bulk", "draw", "T_last"]].rename(columns={"T_last": "T_celsius"}).dropna()
    reads, rung_rows, attrition = {}, [], {}
    for rung in STAGE_A_RUNGS:
        if rung == "last":
            ends = last
        else:
            ends = (points[points["liq_frac"] < int(rung) / 100]
                    .groupby(["bulk", "draw"])["T_celsius"].max().reset_index())
            col = f"T_end_{rung}"
            if col in draws:
                run = draws[["bulk", "draw", col]].merge(ends, on=["bulk", "draw"], how="left")
                if not np.allclose(run[col].fillna(-1.0), run["T_celsius"].fillna(-1.0)):
                    raise SystemExit(f"rung {rung} read from the ladder differs from the run's {col}")
        unreached = sorted(set(bulk) - set(ends["bulk"]))
        rows = points.merge(pd.concat([ends, last[last["bulk"].isin(unreached)]]), on=run_read.KEYS)
        rung_rows.append(rows.assign(rung=rung))
        by = rows.groupby("bulk")
        mean = {prefix: by[[f"{prefix}_{g}" for g in GROUPS]].mean().reindex(bulk).to_numpy()
                for prefix in ("s", "p", "t")}
        reads[rung] = {"zero_shot": mean["s"], "presence": mean["p"], "share": mean["t"]}
        attrition[rung] = {
            "draws_unreached": int(len(draws) - len(ends)), "draws": int(len(draws)),
            "rocks_on_last_step": {s: int((subset.loc[unreached] == s).sum())
                                   for s in PLUTONIC_SPLIT_COUNTS}}
    return reads, pd.concat(rung_rows, ignore_index=True), attrition


def marginal_shares(head, latents, numeric, n_samples=N_SUPPORT_SAMPLES, seed=0):
    """The head read the way the generator is read: its share over each
    rock's measured mineral list averaged over assemblages sampled from its
    own presence probabilities, rather than conditioned on every listed
    mineral being present (`predict`'s teacher-forced read, which the
    production figures use). A rock's sampled support never comes up empty:
    its most likely phase turns on when every draw says absent."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        p21 = torch.sigmoid(head.presence_logits(latents)).cpu().numpy()
    out = np.zeros((len(p21), len(GROUPS)))
    for _ in range(n_samples):
        support = (rng.uniform(size=p21.shape) < p21).astype(np.float64)
        empty = support.sum(1) == 0
        support[np.flatnonzero(empty), p21[empty].argmax(1)] = 1.0
        with torch.no_grad():
            alpha = head.magnitude_alpha(
                latents, torch.as_tensor(support, dtype=latents.dtype, device=latents.device)).cpu().numpy()
        alpha7, _ = coarsen_alpha(alpha, support)
        out += conditional_mean(alpha7, numeric)
    return out / n_samples


def copying_error(head_share, generator_shares, numeric):
    """Per rung, the production head's marginal share against the generator's
    zero-shot share on the same rocks, both over each rock's measured mineral
    list and both marginals over assemblages, as `share_metrics` scores a
    source against the measured share: the head's error against what it was
    trained to reproduce, on the benchmark's chemistry. Values over `LABELS`."""
    return {rung: dict(zip(LABELS, with_means(share_metrics(head_share, share, numeric)[0])))
            for rung, share in generator_shares.items()}


def reach(rung_rows, benchmark, comps, tau=0.05):
    """Fitting-free, over the near-solidus rows (`SOLIDUS_RUNGS`): per
    category, the fraction of evaluated rocks where some (draw, rung) row
    comes within `tau` of the measured share; and the fraction where one
    row's mean error over the quantified silicates does (#79's joint reach,
    62% on the eruption benchmark), on every evaluated rock and on the train
    and val rocks the cross-fitted read covers."""
    rung_rows = rung_rows[rung_rows["rung"].isin(SOLIDUS_RUNGS)]
    bulk = comps.set_index("key").loc[benchmark.bulk_ids, "bulk"].to_numpy()
    read = np.isin(bulk, comps.loc[comps["subset"].isin(["train", "val"]), "bulk"])
    best = rung_rows.groupby("bulk")[[f"d_{g}" for g in GROUPS]].min().reindex(bulk)
    out = {"tau": tau, "per_category": {}}
    for g in GROUPS:
        d = best[f"d_{g}"].to_numpy()
        quantified = np.isfinite(d)
        out["per_category"][g] = {"n_rocks": int(quantified.sum()),
                                  "reached": float((d[quantified] <= tau).mean())}
    sil = (predict.silicate_error(rung_rows, None).groupby(rung_rows["bulk"]).min()
           .reindex(bulk).to_numpy())
    has, hit = np.isfinite(sil), sil <= tau
    out["joint_silicate"] = {"n_rocks": int(has.sum()), "reached": float(hit[has].mean()),
                             "n_read": int((has & read).sum()),
                             "reached_read": float(hit[has & read].mean())}
    return out


def plutonic_inputs(benchmark, device, reads):
    """What every plutonic source is built from, computed once: the
    production head's group predictions and marginal share, the CIPW norm
    and the generator reads per rung."""
    ckpt = JointHeadCheckpoint.load(REPO / "checkpoints" / PRODUCTION / "heads_best.pt")
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)
    head = ckpt.build_mode_head().to(device)
    gp = build_group_predictions(head, benchmark, latents)
    samples = pd.read_csv(PLUTONIC_DIR / "samples.csv")
    samples.index = samples["source"] + "|" + samples["sample_id"].astype(str)
    return {"gp": gp, "marginal": marginal_shares(head, latents, benchmark.numeric.astype(np.float64)),
            "norm": cipw_shares(samples.loc[benchmark.bulk_ids]), "reads": reads}


def fit_sources(inputs, fit_rows):
    """Every source of the plutonic read with its fits on `fit_rows`, on
    every rock: shares over each rock's measured mineral list, presence
    probabilities where the source has them, and per-rock joint NLL where it
    has one. Step 0's sources come first, then per rung the generator and the
    joint fits."""
    gp, norm, reads = inputs["gp"], inputs["norm"], inputs["reads"]
    numeric = gp.numeric.numpy()
    fit_gp = gp.subset(fit_rows)
    fit = fit_recalibrator(fit_gp, tier="matrix", weight_decay=1.0)
    with torch.no_grad():
        logits, alpha, _ = fit(gp.logits7, gp.alpha7, log_w=gp.log_w)
    shares = {"head_zero_shot": conditional_mean(gp.alpha7.numpy(), numeric),
              "head_marginal": inputs["marginal"],
              "head_recalibrated": conditional_mean(alpha.numpy(), numeric)}
    presence = {"head_zero_shot": torch.sigmoid(gp.logits7).numpy(),
                "head_recalibrated": torch.sigmoid(logits).numpy()}
    nll = {"head_zero_shot": score_recalibrator(None, gp)["joint"],
           "head_recalibrated": score_recalibrator(fit, gp)["joint"]}

    shares["cipw_zero_shot"] = renormalize(norm, numeric > 0)
    kappa = float(np.median(fit_gp.alpha7.sum(1).numpy()))
    norm_gp = packaged(gp, norm, (norm > 0).astype(float), kappa)
    norm_fit = fit_recalibrator(norm_gp.subset(fit_rows), tier="matrix", weight_decay=1.0)
    with torch.no_grad():
        logits, alpha, _ = norm_fit(norm_gp.logits7, norm_gp.alpha7)
    shares["cipw_recalibrated"] = conditional_mean(alpha.numpy(), numeric)
    presence["cipw_recalibrated"] = torch.sigmoid(logits).numpy()
    nll["cipw_recalibrated"] = score_recalibrator(norm_fit, norm_gp)["joint"]

    magnitude = gp.magnitude.numpy()
    const = constant_vector(magnitude, numeric > 0, fit_rows)
    shares["constant"] = renormalize(np.tile(const, (len(gp), 1)), numeric > 0)

    for name, swap in (("head_no_chemistry", ablate_latent), ("head_raw_chemistry", use_raw_chemistry)):
        head = fit_measured_head(swap(fit_gp), weight_decay=MEASURED_WEIGHT_DECAY[name])
        every = swap(gp)
        with torch.no_grad():
            logits, alpha, log_w = head(every.latent, every.presence)
        shares[name] = mixture_conditional_mean(alpha.numpy(), log_w.numpy(), numeric)
        presence[name] = torch.sigmoid(logits).numpy()
        nll[name] = score_measured_head(head, every)["joint"]
    order = list(STEP0_SOURCES)

    for rung, read in reads.items():
        name = f"generator_{rung}"
        shares[f"{name}_zero_shot"], presence[f"{name}_zero_shot"] = read["zero_shot"], read["presence"]
        gen_gp = packaged(gp, read["share"], read["presence"], kappa)
        gen_fit = fit_recalibrator(gen_gp.subset(fit_rows), tier="matrix", weight_decay=1.0)
        with torch.no_grad():
            logits, alpha, _ = gen_fit(gen_gp.logits7, gen_gp.alpha7)
        shares[f"{name}_recalibrated"] = conditional_mean(alpha.numpy(), numeric)
        presence[f"{name}_recalibrated"] = torch.sigmoid(logits).numpy()
        nll[f"{name}_recalibrated"] = score_recalibrator(gen_fit, gen_gp)["joint"]

        joint = fit_two_source(norm_gp.subset(fit_rows), gen_gp.subset(fit_rows))
        with torch.no_grad():
            logits, alpha = joint(norm_gp.logits7, norm_gp.alpha7, gen_gp.logits7, gen_gp.alpha7)
        key = f"joint_recal_norm_generator_{rung}"
        shares[key] = conditional_mean(alpha.numpy(), numeric)
        presence[key] = torch.sigmoid(logits).numpy()
        nll[key] = score_predictions(logits, alpha, None, norm_gp)["joint"]
        order += [f"{name}_zero_shot", f"{name}_recalibrated", key]

    norm7 = np.nan_to_num(norm)
    features = {"head_given_norm": norm7}
    for rung, read in reads.items():
        features[f"head_given_generator_{rung}"] = np.hstack([read["presence"], read["share"]])
        features[f"head_given_norm_generator_{rung}"] = np.hstack([norm7, read["presence"], read["share"]])
    for name, x in features.items():
        data = replace(gp, latent=torch.as_tensor(x, dtype=torch.float64))
        head = fit_measured_head(data.subset(fit_rows), weight_decay=1.0)
        with torch.no_grad():
            logits, alpha, log_w = head(data.latent, data.presence)
        shares[name] = mixture_conditional_mean(alpha.numpy(), log_w.numpy(), numeric)
        presence[name] = torch.sigmoid(logits).numpy()
        nll[name] = score_measured_head(head, data)["joint"]
        order.append(name)
    return {name: shares[name] for name in order}, presence, nll, kappa


def cross_fit(inputs, rows, units, k=N_FOLDS):
    """Grouped k-fold over the rocks `rows` selects: each fold's sources are
    fit on the other folds and its rocks keep their held-out predictions, so
    every rock is read once out of fold and no intrusion straddles a fit and
    its read. Returns the held-out shares, presence and NLL on those rocks in
    order, and per fold its rock and intrusion counts and kappa."""
    idx = np.flatnonzero(rows)
    shares, presence, nll, folds = {}, {}, {}, []
    for fit_pos, read_pos in GroupKFold(n_splits=k).split(idx, groups=units[idx]):
        fit_rows = np.zeros(len(rows), dtype=bool)
        fit_rows[idx[fit_pos]] = True
        s, p, n, kappa = fit_sources(inputs, fit_rows)
        for out, part in ((shares, s), (presence, p), (nll, n)):
            for name, v in part.items():
                out.setdefault(name, np.full((len(idx),) + v.shape[1:], np.nan))[read_pos] = v[idx[read_pos]]
        folds.append({"rocks": int(len(read_pos)), "intrusions": int(len(np.unique(units[idx[read_pos]]))),
                      "kappa": kappa})
    return shares, presence, nll, folds


def plutonic_stats(shares, presence_p, nll, magnitude, numeric, presence, measured, idx):
    """{(source, metric): values} on the rows `idx`: share error, Spearman and
    presence AUROC over `LABELS`, mean joint NLL as a single value, and every
    source's share error and NLL minus each baseline's on the same rows."""
    out = {}
    for name, share in shares.items():
        out[name, "share_mae"] = with_means(share_metrics(share[idx], magnitude[idx], numeric[idx])[0])
        out[name, "spearman"] = with_means(spearman(share[idx], magnitude[idx], numeric[idx]))
        if name in presence_p:
            out[name, "auroc"] = with_means(presence_auroc(presence_p[name][idx], presence[idx],
                                                           measured[idx]))
        if name in nll:
            out[name, "joint_nll"] = np.array([nll[name][idx].mean()])
    for name in shares:
        for base in BASELINES:
            out[name, f"share_mae_minus_{base}"] = out[name, "share_mae"] - out[base, "share_mae"]
            if name in nll and base in nll:
                out[name, f"joint_nll_minus_{base}"] = out[name, "joint_nll"] - out[base, "joint_nll"]
    for name, arms in JOINT_PAIRS.items():
        if name not in shares:
            continue
        for arm in arms:
            out[name, f"share_mae_minus_{arm}"] = out[name, "share_mae"] - out[arm, "share_mae"]
            out[name, f"joint_nll_minus_{arm}"] = out[name, "joint_nll"] - out[arm, "joint_nll"]
    return out


def plutonic_table(shares, presence_p, nll, magnitude, numeric, presence, measured, units,
                   n_boot=N_BOOT):
    """One row per (source, label) with each per-label metric and its
    unit-bootstrap band, and per source the NLL terms with theirs. The arrays
    are the read rocks' own, so the bootstrap's positions index them."""
    args = (shares, presence_p, nll, magnitude, numeric, presence, measured)
    est = plutonic_stats(*args, np.arange(len(units)))
    keys = list(est)
    lo, hi = unit_bootstrap(lambda idx: np.concatenate(
        [v for v in plutonic_stats(*args, idx).values()]), units, n_boot=n_boot)
    bounds = np.cumsum([0, *(len(est[k]) for k in keys)])
    band = {k: (est[k], lo[a:b], hi[a:b]) for k, a, b in zip(keys, bounds[:-1], bounds[1:])}
    quantifies = {label: int(numeric[:, [GROUP_INDEX[g] for g in groups]].any(1).sum())
                  for label, groups in MEANS.items()}
    rows, nll_terms = [], {}
    for name, share in shares.items():
        n = share_metrics(share, magnitude, numeric)[2]
        for i, label in enumerate(LABELS):
            row = dict(source=name, category=label,
                       n_rocks=int(n[i]) if i < len(GROUPS) else quantifies[label])
            for (source, metric), (e, a, b) in band.items():
                if source == name and len(e) == len(LABELS):
                    row.update({metric: e[i], f"{metric}_lo": a[i], f"{metric}_hi": b[i]})
            rows.append(row)
        nll_terms[name] = {metric: [float(e[0]), float(a[0]), float(b[0])]
                           for (source, metric), (e, a, b) in band.items() if source == name and len(e) == 1}
    return pd.DataFrame(rows), nll_terms


def print_plutonic(table, nll_terms):
    by = table.set_index(["source", "category"])
    print("plutonic train and val, read out of fold: four-silicate share error [interval], all seven, "
          "silicate spearman, joint nll")
    for name in table["source"].unique():
        sil, every = by.loc[(name, "silicate")], by.loc[(name, "mean7")]
        j = nll_terms[name].get("joint_nll")
        print(f"{name}: {sil['share_mae']:.3f} [{sil['share_mae_lo']:.3f}, {sil['share_mae_hi']:.3f}], "
              f"all {every['share_mae']:.3f}, spearman {sil['spearman']:.2f}"
              + (f", nll {j[0]:.2f} [{j[1]:.2f}, {j[2]:.2f}]" if j else ""))
    shown = ["cipw_recalibrated", *(f"generator_{r}_{k}" for r in STAGE_A_RUNGS
                                    for k in ("zero_shot", "recalibrated"))]
    print("per category share error (spearman):")
    for g in GROUPS:
        print(f"{g}: " + ", ".join(f"{name} {by.loc[(name, g), 'share_mae']:.3f} "
                                   f"({by.loc[(name, g), 'spearman']:.2f})" for name in shown))
    print("joint fits, silicate share error and joint nll minus each nested arm [interval]:")
    for name, arms in JOINT_PAIRS.items():
        sil = by.loc[(name, "silicate")]
        parts = [f"{arm} {sil[f'share_mae_minus_{arm}']:+.4f} [{sil[f'share_mae_minus_{arm}_lo']:+.4f}, "
                 f"{sil[f'share_mae_minus_{arm}_hi']:+.4f}], nll "
                 + "{:+.3f} [{:+.3f}, {:+.3f}]".format(*nll_terms[name][f"joint_nll_minus_{arm}"])
                 for arm in arms]
        print(f"{name}: " + "; ".join(parts))


def run_facts(work):
    recs = run_read.read_manifest(work)
    return dict(work=str(work), n_chunks=len(recs), n_rocks=int(sum(r["n_bulks"] for r in recs.values())),
                n_draws_per_rock=int(recs[min(recs)]["n_draws_per_bulk"]),
                mean_failed_per_draw=round(float(np.mean([r["mean_failed_per_draw"] for r in recs.values()])), 3),
                solver_crashes=int(sum(len(r["solver_crashes"]) for r in recs.values())),
                wall_h=round(sum(r["wall_s"] for r in recs.values()) / 3600, 2))


def draw_stage_a(table, nll_terms, out, name="plutonic_stage_a"):
    sil = table[table["category"] == "silicate"].set_index("source")
    names = list(sil.index)
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * len(names) + 1.6), sharey=True)
    ax = axes[0]
    ax.errorbar(sil["share_mae"], y, xerr=[sil["share_mae"] - sil["share_mae_lo"],
                                           sil["share_mae_hi"] - sil["share_mae"]],
                fmt="o", color=NAVY, capsize=2)
    ax.axvline(sil.loc["cipw_recalibrated", "share_mae"], color=ORANGE, ls="--", lw=1,
               label="CIPW norm, recalibrated")
    ax.set_xlabel("four-silicate share absolute error, train and val out of fold", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    has = [n for n in names if "joint_nll" in nll_terms[n]]
    e = np.array([nll_terms[n]["joint_nll"] for n in has])
    ax.errorbar(e[:, 0], [names.index(n) for n in has], xerr=[e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]],
                fmt="o", color=NAVY, capsize=2)
    ax.axvline(nll_terms["cipw_recalibrated"]["joint_nll"][0], color=ORANGE, ls="--", lw=1)
    ax.set_xlabel("joint NLL per rock, train and val out of fold", fontsize=9)
    axes[0].set_yticks(y, names, fontsize=7)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("plutonic train and val rocks, grouped 5-fold: the simulator near the solidus against "
                 "the baselines and the joint fits; bars = unit-bootstrap 95% over the intrusions",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)


def read_plutonic(args):
    with plutonic_benchmark() as evaluated:
        benchmark = evaluated.subset(evaluated.has_magnitude)
        masks = {s: split_mask(benchmark, s) for s in PLUTONIC_SPLIT_COUNTS}
    counts = {s: int(m.sum()) for s, m in masks.items()}
    if counts != PLUTONIC_SPLIT_COUNTS:
        raise SystemExit(f"split counts {counts} are not the plutonic benchmark's {PLUTONIC_SPLIT_COUNTS}")
    rows = masks["train"] | masks["val"]

    work = args.work or PLUTONIC_WORK
    comps = load_compositions(RESULTS / "plutonic_compositions.csv")
    reads, rung_rows, attrition = generator_reads(work, comps, benchmark, args.table)
    inputs = plutonic_inputs(benchmark, pick_device(), reads)
    shares, presence_p, nll, folds = cross_fit(inputs, rows, benchmark.unit)
    units = benchmark.unit[rows]
    print("folds: " + ", ".join(f"{f['rocks']} rocks / {f['intrusions']} intrusions" for f in folds))
    table, nll_terms = plutonic_table(shares, presence_p, nll, benchmark.magnitude[rows],
                                      benchmark.numeric[rows], benchmark.presence[rows],
                                      benchmark.measured[rows], units)
    print_plutonic(table, nll_terms)
    copying = copying_error(shares["head_marginal"], {r: v["zero_shot"][rows] for r, v in reads.items()},
                            benchmark.numeric[rows])
    print("head marginal against the generator, four silicates: " + ", ".join(
        f"rung {r} {c['silicate']:.3f}" for r, c in copying.items()))
    reached = reach(rung_rows, benchmark, comps)
    print("attrition: " + "; ".join(
        f"rung {r}: {a['draws_unreached']} of {a['draws']} draws unreached, rocks on the last "
        f"step {a['rocks_on_last_step']}" for r, a in attrition.items() if r != "last"))
    j = reached["joint_silicate"]
    print(f"reach at tau 0.05: joint four-silicate {j['reached']:.2f} of {j['n_rocks']} rocks "
          f"({j['reached_read']:.2f} of {j['n_read']} train and val); per category "
          + ", ".join(f"{g} {v['reached']:.2f}" for g, v in reached["per_category"].items()))

    key = "plutonic" if args.recipe == "equilibrium" else f"plutonic_{args.recipe}"
    path = args.results / "signal.json"
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload[key] = {
        "settings": {"head": PRODUCTION, "recalibrator": "matrix, weight decay 1.0",
                     "measured_head_weight_decay": MEASURED_WEIGHT_DECAY,
                     "package_floor": PACKAGE_FLOOR, "cipw_to_group": CIPW_TO_GROUP,
                     "baselines": list(BASELINES), "n_boot": N_BOOT, "bootstrap_seed": 0,
                     "protocol": f"grouped {N_FOLDS}-fold cross-validation over the train and val "
                                 "rocks, intrusions kept whole (sklearn GroupKFold); every rock read "
                                 "once out of fold",
                     "folds": folds, "read_rocks": int(rows.sum()),
                     "read_units": int(len(np.unique(units))), "split_counts": counts,
                     "oxides_unmeasured_rocks": int((~benchmark.measured[:, GROUP_INDEX["opq"]]).sum()),
                     "test": "sealed"},
        "stage_a": {"run": run_facts(work), "recipe": args.recipe, "table": args.table,
                    "rungs": list(STAGE_A_RUNGS),
                    "presence_cutoff": PRESENCE_CUTOFF, "attrition": attrition,
                    "joint_pairs": {k: list(v) for k, v in JOINT_PAIRS.items()},
                    "joint_recalibrator": "matrix on the CIPW norm plus zero-initialized 7x7 "
                                          "generator terms, weight decay 1.0, 400 epochs",
                    "joint_head_weight_decay": 1.0, "reach": reached,
                    "reach_rungs": list(SOLIDUS_RUNGS), "head_copying_error": copying},
        "nll": nll_terms,
        "table": table.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=1, default=float))
    name = "plutonic_stage_a" if args.recipe == "equilibrium" else f"plutonic_stage_a_{args.recipe}"
    draw_stage_a(table, nll_terms, args.results, name)
    print(f"wrote the plutonic read into {path} under {key!r} and {name}.png")


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


def read_eruption(args):
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

    work = args.work or WORK
    raw_points = run_read.build_points(work)
    points = run_read.shares_and_distances(raw_points, magnitude, numeric)
    draws = run_read.load_draws(work)
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
    path = args.results / "signal.json"
    plutonic = json.loads(path.read_text()).get("plutonic") if path.exists() else None
    payload = {
        "settings": {"head": HEAD, "best_rectangle": best, "n_boot": N_BOOT, "bootstrap_seed": 0,
                     "val_rocks": int(val.sum()), "val_units": int(len(np.unique(units))),
                     "split_counts": counts, "hit_rows": {"tau": 0.05, "fallback": 8},
                     "fits_on": "train rocks", "reads_on": "val rocks", "test": "sealed"},
        "reproduction_max_abs_diff": checks,
        "constant_train_mean": dict(zip(GROUPS, const.tolist())),
        "linear_fit_coefficients": {name: {g: {"a": a, "b": b} for g, (a, b) in c.items()}
                                    for name, c in coef.items()},
        "ruler": ruler,
    }
    if plutonic is not None:
        payload["plutonic"] = plutonic
    path.write_text(json.dumps(payload, indent=1, default=float))
    draw_figure(shares, coef, magnitude, numeric, subset, table, args.results)
    print(f"wrote {args.results / 'signal.csv'}, signal.json, signal.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=("eruption", "plutonic"), default="eruption")
    ap.add_argument("--work", type=Path, help="default work/broad128, or work/plutonic128 "
                                              "with --benchmark plutonic")
    ap.add_argument("--table", default="ladder",
                    help="the run's ladder table to read: ladder (equilibrium), or a fractional "
                         "run's step (the cargo as it forms) or pile (the accumulated crystals)")
    ap.add_argument("--recipe", choices=("equilibrium", "fractional"), default="equilibrium",
                    help="names the plutonic read's key in signal.json and its figure")
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    (read_plutonic if args.benchmark == "plutonic" else read_eruption)(args)


if __name__ == "__main__":
    main()
