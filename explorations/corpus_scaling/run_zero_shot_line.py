"""The zero-shot corpus-size line on a benchmark's train and val rocks: the
last-epoch heads at 56, 224 and 896 gradient steps
(`final{D}_allrock_n{N}[_seed{K}]`, ten sizes, three seeds, all trained on
the 50 + 30 equilibrium labels) read teacher-forced, as the depth grid
reads them, with no fit to any measured rock. On the plutonic benchmark the
rocks are also read by pretraining membership: DS 942 rocks whose chemistry
the encoder trained on (pretraining train or val), DS 942 rocks it held out
(pretraining test), and PP 1793, which pretraining barely touched (journal
2026-10-19). Membership matches each rock's eight majors, renormalized to
100 with total iron as FeO, against the rebuilt pretraining set: every oxide
within 0.02 wt% of one row. The roster replicates
(`run_roster_replicates.py`, `final224_allrock_n{N}_roster{r}`) are read beside
the nested-roster heads. Per depth and rock group, a unit bootstrap (whole
eruptive units or intrusions, the same resample at every size) bands the
nested-roster line, seed-averaged per rock, and its rise (78 - 10), fall
(10,000 - 78) and net (10,000 - 10); and the val rocks' rise, fall and net
are placed among random draws of as many train and val units. Writes
`results/zero_shot_line_{benchmark}.csv`, `zero_shot_line_{benchmark}_bands.json`
and, for the plutonic benchmark, `plutonic_pretraining_membership.csv`;
`fig_zero_shot_scaling.py` draws the figures.

  python explorations/corpus_scaling/run_zero_shot_line.py --benchmark volcanic
  python explorations/corpus_scaling/run_zero_shot_line.py --benchmark plutonic
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.data.magemin_oxides import FE2O3_TO_FEO_MASS
from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import load_benchmark, split_mask
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import build_group_predictions
from explorations.corpus_scaling.run_depth_grid import MEANS, PRODUCTION, SIZES, share_metrics, with_means, with_seeds
from explorations.real_mineralogy.plutonic import plutonic_benchmark

RESULTS = Path(__file__).resolve().parent / "results"
PRETRAIN_CONFIG = REPO / "configs" / "pretrain" / "pretrain_allrock_redox.yaml"
DEPTHS = (56, 224, 896)
SPLITS = ("train", "val")
LABELS = (*GROUPS, *MEANS)
MAJORS = ("sio2", "tio2", "al2o3", "fe_t", "mgo", "cao", "na2o", "k2o")
MATCH_TOL = 0.02
DS942, PP1793 = "dubray_ds942", "dubray_pp1793"
ROSTER_REPLICATES = (1, 2, 3, 4, 5)
N_BOOT = 2000


def renormalized_majors(frame):
    """The eight majors with total iron as FeO (`feo_tot`, else `fe2o3_tot`
    converted, else `feo` plus converted `fe2o3`), a missing oxide as zero,
    renormalized to 100."""
    fe = frame["feo_tot"].copy()
    fe = fe.fillna(frame["fe2o3_tot"] * FE2O3_TO_FEO_MASS)
    if "feo" in frame:
        split = frame["feo"].fillna(0.0) + frame["fe2o3"].fillna(0.0) * FE2O3_TO_FEO_MASS
        fe = fe.fillna(split.where(frame[["feo", "fe2o3"]].notna().any(axis=1)))
    x = frame.assign(fe_t=fe)[list(MAJORS)].fillna(0.0).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * x / x.sum(1, keepdims=True)


def match(rocks, pool, tol=MATCH_TOL):
    """Per rock, every pool row whose majors all sit within `tol` wt% (the
    largest single-oxide difference), and the distance to the nearest row.
    Pool rows with no majors (trace-only rows) cannot match."""
    usable = np.flatnonzero(np.isfinite(pool).all(1))
    tree = cKDTree(pool[usable])
    dist, _ = tree.query(rocks, p=np.inf)
    return [usable[np.asarray(h, dtype=int)] for h in tree.query_ball_point(rocks, r=tol, p=np.inf)], dist


def membership_status(matched_ids, test_ids):
    """Per rock from its matched sample ids: "absent" for none, "seen" when
    any copy sat in pretraining train or val (near-copies of one analysis
    can straddle the splits), "held_out" when every copy sat in test."""
    test = set(np.asarray(test_ids).tolist())
    return np.array(["absent" if len(ids) == 0 else
                     "held_out" if all(i in test for i in ids.tolist()) else "seen" for ids in matched_ids])


def pretraining_membership(comps):
    """Each rock's pretraining status, rebuilding the pretraining set from
    its config and reading the held-out sample ids the encoder stamped."""
    cfg = yaml.safe_load(PRETRAIN_CONFIG.read_text())
    kwargs = {k: (str(REPO / v) if k.endswith("_path") and v else v) for k, v in cfg["data_kwargs"].items()}
    ids = BulkChemistryDataset(**kwargs).sample_ids
    # the dataset reads gard2019 through this parquet sidecar of the csv
    gard = pd.read_parquet(Path(kwargs["csv_path"]).with_suffix(".parquet"),
                           columns=["sample_id", "sio2", "tio2", "al2o3", "feo", "fe2o3",
                                    "feo_tot", "fe2o3_tot", "mgo", "cao", "na2o", "k2o"])
    gard = gard.set_index("sample_id").loc[ids].reset_index()
    ckpt = JointHeadCheckpoint.load(REPO / "checkpoints" / PRODUCTION / "heads_best.pt")
    test_ids = np.asarray(torch.load(REPO / ckpt.encoder_checkpoint, map_location="cpu",
                                     weights_only=False)["test_sample_ids"])
    hits, dist = match(renormalized_majors(comps), renormalized_majors(gard))
    matched = [gard["sample_id"].to_numpy()[h] for h in hits]
    return pd.DataFrame({"key": comps["key"], "source": comps["source"],
                         "status": membership_status(matched, test_ids),
                         "n_copies": [len(m) for m in matched],
                         "n_copies_held_out": [int(np.isin(m, test_ids).sum()) for m in matched],
                         "nearest_max_oxide_difference_wt": dist.round(4)})


def group_metrics(share, magnitude, numeric, groups):
    """Per named rock mask: the error per category and its means, and the
    rock count per category."""
    out = {}
    for name, m in groups.items():
        mae, _, n = share_metrics(share[m], magnitude[m], numeric[m])
        out[name] = {**{f"mae_{label}": v for label, v in zip(LABELS, with_means(mae))},
                     **{f"n_{g}": int(c) for g, c in zip(GROUPS, n)}, "n_rocks": int(m.sum())}
    return out


def rock_errors(share, magnitude, numeric):
    """Per rock and category, the absolute share error, NaN where the rock
    does not quantify the category."""
    return np.where(numeric, np.abs(share - magnitude), np.nan)


def line_values(errors, idx):
    """Per corpus size, the error per category and its means over the rows
    `idx` of `errors` (sizes x rocks x categories); the mean over a category's
    quantified rows is what `share_metrics` reports."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mae = np.nanmean(errors[:, idx], axis=1)
    return np.stack([with_means(m) for m in mae])


def unit_resamples(units, n_boot=N_BOOT, seed=0):
    """Row positions of whole units drawn with replacement, the resampling
    of `cnmineral.eval.real_rock.unit_bootstrap`."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(units == u) for u in np.unique(units)]
    for _ in range(n_boot):
        yield np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])


def differences(values):
    """The line's rise (78 - 10), fall (10,000 - 78) and net (10,000 - 10)
    per label, from per-size values."""
    at = {n: values[..., SIZES.index(n), :] for n in (10, 78, 10000)}
    return {"rise_78_minus_10": at[78] - at[10], "fall_10000_minus_78": at[10000] - at[78],
            "net_10000_minus_10": at[10000] - at[10]}


def paired_bands(errors, units, n_boot=N_BOOT):
    """The line per size with unit-bootstrap 95% bands and interquartile
    ranges, and its rise, fall and net with 95% bands; every size reads the same resample, so the
    differences are paired."""
    est = line_values(errors, np.arange(len(units)))
    reps = np.stack([line_values(errors, idx) for idx in unit_resamples(units, n_boot)])
    lo, hi = np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0)
    q25, q75 = np.nanpercentile(reps, 25, axis=0), np.nanpercentile(reps, 75, axis=0)
    out = {"points": {int(n): {label: [est[i, j], lo[i, j], hi[i, j]] for j, label in enumerate(LABELS)}
                      for i, n in enumerate(SIZES)},
           "points_interquartile": {int(n): {label: [q25[i, j], q75[i, j]] for j, label in enumerate(LABELS)}
                                    for i, n in enumerate(SIZES)}}
    d_est, d_reps = differences(est), differences(reps)
    out["differences"] = {name: {label: [d_est[name][j], *np.nanpercentile(d_reps[name][:, j], [2.5, 97.5])]
                                 for j, label in enumerate(LABELS)} for name in d_est}
    return out


def subset_percentiles(errors, units, subset, n_draw=N_BOOT, seed=0):
    """Where the line's rise, fall and net (mean of seven) on the rows
    `subset` fall among random draws of as many whole units, without
    replacement, from all of `units`: a percentile near 50 is an ordinary
    subset."""
    rng = np.random.default_rng(seed)
    pool = np.unique(units)
    k = len(np.unique(units[subset]))
    m7 = LABELS.index("mean7")
    own = differences(line_values(errors, np.flatnonzero(subset)))
    draws = {name: [] for name in own}
    for _ in range(n_draw):
        idx = np.flatnonzero(np.isin(units, rng.choice(pool, size=k, replace=False)))
        for name, v in differences(line_values(errors, idx)).items():
            draws[name].append(v[m7])
    return {name: {"subset": float(own[name][m7]), "percentile": float((np.array(draws[name]) < own[name][m7]).mean() * 100),
                   "draws_median": float(np.median(draws[name])), "units_drawn": int(k)} for name in own}


def load(name):
    """The evaluated benchmark and its train-or-val and val masks; the plutonic split
    roster is only in place inside its context manager."""
    if name == "volcanic":
        benchmark = load_benchmark()
        benchmark = benchmark.subset(benchmark.clean & benchmark.has_magnitude)
        rows = split_mask(benchmark, "train") | split_mask(benchmark, "val")
        val = split_mask(benchmark, "val")
    else:
        with plutonic_benchmark() as evaluated:
            benchmark = evaluated.subset(evaluated.has_magnitude)
            rows = split_mask(benchmark, "train") | split_mask(benchmark, "val")
            val = split_mask(benchmark, "val")
    return benchmark, rows, val


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=("volcanic", "plutonic"), required=True)
    args = ap.parse_args()
    benchmark, rows, val = load(args.benchmark)
    groups = {"all": rows}
    if args.benchmark == "plutonic":
        comps = pd.read_csv(REPO / "explorations" / "generator_reach" / "results" / "plutonic_compositions.csv",
                            dtype={"sample_id": str})
        membership = pretraining_membership(comps)
        membership.to_csv(RESULTS / "plutonic_pretraining_membership.csv", index=False)
        status = membership.set_index("key").loc[benchmark.bulk_ids]
        print("pretraining membership, evaluated rocks:",
              status.groupby(["source", "status"]).size().to_dict())
        is_ds942 = (status["source"] == DS942).to_numpy()
        groups.update({
            "ds942_seen": rows & is_ds942 & (status["status"] == "seen").to_numpy(),
            "ds942_held_out": rows & is_ds942 & (status["status"] == "held_out").to_numpy(),
            "pp1793": rows & (status["source"] == PP1793).to_numpy()})

    device = pick_device()
    prod = JointHeadCheckpoint.load(REPO / "checkpoints" / PRODUCTION / "heads_best.pt")
    encoder, _ = load_encoder_from_checkpoint(prod.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)
    numeric = benchmark.numeric.astype(np.float64)

    records, errors = [], {}
    for depth in DEPTHS:
        for n in SIZES:
            cells = [(0, seed, tag) for seed, tag in with_seeds(f"final{depth}_allrock_n{n}")]
            cells += [(r, 0, f"final{depth}_allrock_n{n}_roster{r}") for r in ROSTER_REPLICATES
                      if (REPO / "checkpoints" / f"final{depth}_allrock_n{n}_roster{r}" / "heads_best.pt").exists()]
            nested = []
            for roster, seed, tag in cells:
                ckpt = JointHeadCheckpoint.load(REPO / "checkpoints" / tag / "heads_best.pt")
                gp = build_group_predictions(ckpt.build_mode_head().to(device), benchmark, latents)
                share = conditional_mean(gp.alpha7.numpy(), numeric)
                if roster == 0:
                    nested.append(rock_errors(share, benchmark.magnitude, benchmark.numeric))
                for name, metrics in group_metrics(share, benchmark.magnitude, benchmark.numeric,
                                                   groups).items():
                    records.append(dict(benchmark=args.benchmark, group=name, depth=depth, n_bulks=n,
                                        roster=roster, seed=seed, tag=tag, **metrics))
            errors[depth, n] = np.mean(nested, axis=0)
        line = [r for r in records if r["depth"] == depth and r["group"] == "all" and r["roster"] == 0]
        print(f"{depth} steps, mean of seven by size: " + ", ".join(
            f"{n:,} {np.mean([r['mae_mean7'] for r in line if r['n_bulks'] == n]):.4f}" for n in SIZES))
    out = RESULTS / f"zero_shot_line_{args.benchmark}.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"{int(rows.sum())} train and val rocks; wrote {out}")

    units = np.asarray(benchmark.unit).astype(str)
    m7 = LABELS.index("mean7")
    bands = {"settings": {"n_boot": N_BOOT, "seed": 0, "rocks": "train and val", "basis": "teacher-forced",
                          "seed_mean": "per-rock errors averaged over the nested roster's seeds"}}
    for depth in DEPTHS:
        stack = np.stack([errors[depth, n] for n in SIZES])
        entry = {}
        for name, m in groups.items():
            entry[name] = paired_bands(stack[:, m], units[m])
            if name == "all":
                entry[name]["val_subset"] = subset_percentiles(stack[:, rows], units[rows], val[rows])
        bands[str(depth)] = entry
        d = entry["all"]["differences"]
        print(f"{depth} steps, mean of seven: " + "; ".join(
            f"{k} {100 * v['mean7'][0]:+.2f} [{100 * v['mean7'][1]:+.2f}, {100 * v['mean7'][2]:+.2f}]"
            for k, v in d.items()))
        print(f"{depth} steps, val among random {entry['all']['val_subset']['net_10000_minus_10']['units_drawn']}"
              "-unit subsets, percentile: " + ", ".join(
                  f"{k} {v['percentile']:.0f}" for k, v in entry["all"]["val_subset"].items()))
    path = RESULTS / f"zero_shot_line_{args.benchmark}_bands.json"
    path.write_text(json.dumps(bands, indent=1, default=float))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
