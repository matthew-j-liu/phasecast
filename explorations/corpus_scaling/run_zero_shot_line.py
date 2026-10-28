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
within 0.02 wt% of one row. Writes `results/zero_shot_line_{benchmark}.csv`
and, for the plutonic benchmark, `results/plutonic_pretraining_membership.csv`;
`fig_zero_shot_scaling.py` draws the figures.

  python explorations/corpus_scaling/run_zero_shot_line.py --benchmark volcanic
  python explorations/corpus_scaling/run_zero_shot_line.py --benchmark plutonic
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import sys
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
PRETRAINING_GROUPS = ("ds942_seen", "ds942_held_out", "pp1793")


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


def load(name):
    """The evaluated benchmark and its train-or-val mask; the plutonic split
    roster is only in place inside its context manager."""
    if name == "volcanic":
        benchmark = load_benchmark()
        benchmark = benchmark.subset(benchmark.clean & benchmark.has_magnitude)
        rows = split_mask(benchmark, "train") | split_mask(benchmark, "val")
    else:
        with plutonic_benchmark() as evaluated:
            benchmark = evaluated.subset(evaluated.has_magnitude)
            rows = split_mask(benchmark, "train") | split_mask(benchmark, "val")
    return benchmark, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=("volcanic", "plutonic"), required=True)
    args = ap.parse_args()
    benchmark, rows = load(args.benchmark)
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

    records = []
    for depth in DEPTHS:
        for n in SIZES:
            for seed, tag in with_seeds(f"final{depth}_allrock_n{n}"):
                ckpt = JointHeadCheckpoint.load(REPO / "checkpoints" / tag / "heads_best.pt")
                gp = build_group_predictions(ckpt.build_mode_head().to(device), benchmark, latents)
                share = conditional_mean(gp.alpha7.numpy(), numeric)
                for name, metrics in group_metrics(share, benchmark.magnitude, benchmark.numeric,
                                                   groups).items():
                    records.append(dict(benchmark=args.benchmark, group=name, depth=depth, n_bulks=n,
                                        seed=seed, tag=tag, **metrics))
        line = [r for r in records if r["depth"] == depth and r["group"] == "all"]
        print(f"{depth} steps, mean of seven by size: " + ", ".join(
            f"{n:,} {np.mean([r['mae_mean7'] for r in line if r['n_bulks'] == n]):.4f}" for n in SIZES))
    out = RESULTS / f"zero_shot_line_{args.benchmark}.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"{int(rows.sum())} train and val rocks; wrote {out}")


if __name__ == "__main__":
    main()
