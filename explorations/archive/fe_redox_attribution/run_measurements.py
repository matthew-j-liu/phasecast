"""Is the spinel ceiling redox invisibility? -- the two #46 measurements.
The generator saw an Fe3+/Fe2+ split per bulk (measured or Middlemost class
lookup, provenance stamped in the draws tables); the model input carries only
total Fe. This script (1) slices per-bulk presence-fraction errors by
`fe_provenance` and (2) re-runs the XGBoost ceiling arm of
`presence_fraction_ceiling` under the generator's own anhydrous oxide basis,
with and without its excess-oxygen redox column, to price what the input
representation loses to redox. Read-only against production models.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import sys
from pathlib import Path

# running as a script puts only this directory on sys.path; the probe helpers
# live one package over, so the repo root has to be importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from cnmineral.data.mineralogy_dataset import JointHeadDataset
from cnmineral.data.phase_vocab import SOLID_PHASES
from cnmineral.data.tokens import VOCAB
from cnmineral.eval.joint_heads import val_dataset_and_latents
from cnmineral.eval.supergroup import map_qap_array
from cnmineral.training import checkpoint_tag, pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from explorations.archive.presence_fraction_ceiling.run_probe import (
    HEADLINE,
    _fit_xgb,
    binomial_floor,
    bulk_frequency_targets,
    log_feature_matrix,
    score_arm,
    tune_xgb,
)

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
RECORDED_TOKEN22_METRICS = (
    Path(__file__).resolve().parents[1]
    / "presence_fraction_ceiling" / "results"
    / "joint_heads_seed0_heads_best" / "metrics.json"
)
ANHYDROUS_OXIDES = (
    "SiO2", "Al2O3", "CaO", "MgO", "FeO_t",
    "K2O", "Na2O", "TiO2", "excess_O", "Cr2O3",
)
EXCESS_O_COL = ANHYDROUS_OXIDES.index("excess_O")
ELEMENT_SYMBOLS = [s for s, _z in VOCAB]
DEFAULT_CHECKPOINT = Path("checkpoints/joint_heads_supportmag_seed0/heads_best.pt")


def anhydrous_bulk_basis(
    bulk_11: np.ndarray, row_bulk_idx: np.ndarray, n_bulks: int, atol: float = 1e-5
) -> np.ndarray:
    """Per-bulk generator input in the 10-oxide anhydrous basis, [n_bulks, 10].

    `bulk_11` is per-row and normalized including the per-draw H2O; dropping
    H2O and renormalizing must recover a bulk-constant vector (H2O is the
    sampled condition, everything else is the bulk). The constancy check is the
    contract: if it fires, the corpus's in_* columns are not per-bulk and this
    feature basis is invalid. `atol` sits above the dataset's float32 rounding
    (~1e-7 observed) and far below any real chemistry variation.
    """
    anhyd = bulk_11[:, : len(ANHYDROUS_OXIDES)]
    anhyd = anhyd / anhyd.sum(axis=1, keepdims=True)
    uniq, first_idx = np.unique(row_bulk_idx, return_index=True)
    if not np.array_equal(uniq, np.arange(n_bulks)):
        raise ValueError("row_bulk_idx does not cover 0..n_bulks-1")
    per_bulk = anhyd[first_idx]
    dev = np.abs(anhyd - per_bulk[row_bulk_idx]).max()
    if dev > atol:
        raise ValueError(
            f"anhydrous basis varies within a bulk (max dev {dev:.2e} > {atol:.0e})"
        )
    return per_bulk


def without_excess_o(bulk10: np.ndarray) -> np.ndarray:
    """Drop the excess-O column and renormalize to sum 1, [n, 9].

    Without the renormalization the remaining columns sum to 1 - excess_O and
    a tree could read the redox column right back off the deficit.
    """
    out = np.delete(bulk10, EXCESS_O_COL, axis=1)
    return out / out.sum(axis=1, keepdims=True)


def provenance_by_bulk(corpus_dir: str | Path) -> pd.Series:
    """bulk id -> fe_provenance from the draws tables; one value per bulk."""
    files = sorted(Path(corpus_dir).glob("chunk_*_draws.parquet"))
    if not files:
        raise FileNotFoundError(f"no chunk_*_draws.parquet under {corpus_dir}")
    df = pd.concat(
        [pd.read_parquet(f, columns=["bulk", "fe_provenance"]) for f in files],
        ignore_index=True,
    ).drop_duplicates()
    counts = df.groupby("bulk").size()
    if (counts > 1).any():
        bad = counts[counts > 1].index.tolist()[:5]
        raise ValueError(f"bulks with multiple fe_provenance values: {bad}")
    return df.set_index("bulk")["fe_provenance"]


def group_slices(
    pred: np.ndarray,
    truth: np.ndarray,
    n_draws: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Per-group score_arm + mean truth + binomial floor over [B, P] arrays."""
    out = {}
    for g in np.unique(groups):
        m = groups == g
        floors = binomial_floor(truth[m], n_draws[m])
        out[str(g)] = {
            "n": int(m.sum()),
            "scores": score_arm(pred[m], truth[m]),
            "mean_f": {p: float(truth[m, i].mean()) for i, p in enumerate(SOLID_PHASES)},
            "binomial_floor": {p: float(floors[i]) for i, p in enumerate(SOLID_PHASES)},
        }
    return out


def bootstrap_delta_ci(
    abs_err_a: np.ndarray,
    abs_err_b: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile CI on MAE(a) - MAE(b), resampling bulks independently."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(abs_err_a), size=(n_boot, len(abs_err_a)))
    ib = rng.integers(0, len(abs_err_b), size=(n_boot, len(abs_err_b)))
    deltas = abs_err_a[ia].mean(axis=1) - abs_err_b[ib].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def fit_basis_arm(
    x_train: np.ndarray,
    x_val: np.ndarray,
    f_train: np.ndarray,
    fit_idx: np.ndarray,
    hold_idx: np.ndarray,
    params: dict,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One XGB per phase; predictions on the val roster and the train holdout.

    Holdout predictions are mildly optimistic (it is the early-stopping set);
    it is used for the provenance split because the val roster alone is thin.
    """
    n_phases = f_train.shape[1]
    p_val = np.zeros((len(x_val), n_phases))
    p_hold = np.zeros((len(hold_idx), n_phases))
    importances = {}
    for i, phase in enumerate(SOLID_PHASES):
        model = _fit_xgb(
            params,
            x_train[fit_idx], f_train[fit_idx, i],
            x_train[hold_idx], f_train[hold_idx, i],
            seed=seed,
        )
        p_val[:, i] = np.clip(model.predict(x_val), 0.0, 1.0)
        p_hold[:, i] = np.clip(model.predict(x_train[hold_idx]), 0.0, 1.0)
        importances[phase] = [float(v) for v in model.feature_importances_]
    return p_val, p_hold, importances


def provenance_report(
    preds: dict[str, np.ndarray],
    truth: np.ndarray,
    n_draws: np.ndarray,
    prov: np.ndarray,
    supergroups: np.ndarray,
    *,
    seed: int,
) -> dict:
    """Per-arm provenance slices + spl measured-minus-assumed bootstrap CI +
    supergroup composition and within-supergroup contrast (n >= 20 both sides)."""
    spl = SOLID_PHASES.index("spl")
    report = {"by_arm": {}, "supergroup_composition": {}}
    for g in np.unique(prov):
        m = prov == g
        comp = {sg: int(((supergroups == sg) & m).sum()) for sg in np.unique(supergroups)}
        report["supergroup_composition"][str(g)] = comp
    meas, assm = prov == "measured", prov == "assumed_class"
    for arm, pred in preds.items():
        entry = {"groups": group_slices(pred, truth, n_draws, prov)}
        if meas.any() and assm.any():
            err = np.abs(pred[:, spl] - truth[:, spl])
            lo, hi = bootstrap_delta_ci(err[meas], err[assm], seed=seed)
            entry["spl_measured_minus_assumed"] = {
                "delta_mae": float(err[meas].mean() - err[assm].mean()),
                "ci95": [lo, hi],
            }
            within = {}
            for sg in np.unique(supergroups):
                a, b = meas & (supergroups == sg), assm & (supergroups == sg)
                if a.sum() >= 20 and b.sum() >= 20:
                    within[str(sg)] = {
                        "measured": {"mae": float(err[a].mean()), "n": int(a.sum())},
                        "assumed_class": {"mae": float(err[b].mean()), "n": int(b.sum())},
                    }
            entry["spl_within_supergroup"] = within
        report["by_arm"][arm] = entry
    return report


def _plot_spl_by_provenance(reports: dict[str, dict], tag: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(reports), figsize=(6.5 * len(reports), 4), squeeze=False)
    for ax, (roster, report) in zip(axes[0], reports.items()):
        arms = list(report["by_arm"])
        groups = sorted(report["by_arm"][arms[0]]["groups"])
        x = np.arange(len(groups))
        w = 0.8 / len(arms)
        for j, arm in enumerate(arms):
            slices = report["by_arm"][arm]["groups"]
            vals = [slices[g]["scores"]["spl"]["mae"] for g in groups]
            ax.bar(x + (j - (len(arms) - 1) / 2) * w, vals, w, label=arm)
        floors = [report["by_arm"][arms[0]]["groups"][g]["binomial_floor"]["spl"]
                  for g in groups]
        ax.scatter(x, floors, marker="_", s=220, color="black", zorder=3,
                   label="binomial floor")
        ns = [report["by_arm"][arms[0]]["groups"][g]["n"] for g in groups]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(groups, ns)])
        ax.set_ylabel("spl per-bulk MAE")
        ax.set_title(roster)
        ax.legend(fontsize=8)
    fig.suptitle(f"spinel error by fe_provenance -- {tag}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_headline_by_basis(
    scores: dict[str, dict], floors: np.ndarray, tag: str, out_path: Path
) -> None:
    x = np.arange(len(HEADLINE))
    arms = list(scores)
    w = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, arm in enumerate(arms):
        vals = [scores[arm][p]["mae"] for p in HEADLINE]
        ax.bar(x + (j - (len(arms) - 1) / 2) * w, vals, w, label=arm)
    floor_vals = [floors[SOLID_PHASES.index(p)] for p in HEADLINE]
    ax.scatter(x, floor_vals, marker="_", s=220, color="black", zorder=3,
               label="binomial floor (val draws)")
    ax.set_xticks(x)
    ax.set_xticklabels(HEADLINE)
    ax.set_ylabel("per-bulk MAE of presence fraction")
    ax.set_title(f"headline ceiling by feature basis -- {tag}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_importances(
    importances: dict[str, dict[str, list]],
    feature_names: dict[str, list[str]],
    tag: str,
    out_path: Path,
) -> None:
    bases = list(importances)
    fig, axes = plt.subplots(
        len(bases), len(HEADLINE),
        figsize=(3.5 * len(HEADLINE), 3.0 * len(bases)),
        squeeze=False,
    )
    for r, basis in enumerate(bases):
        names = feature_names[basis]
        x = np.arange(len(names))
        for c, phase in enumerate(HEADLINE):
            ax = axes[r][c]
            ax.bar(x, importances[basis][phase])
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=90, fontsize=6)
            if c == 0:
                ax.set_ylabel(f"{basis}\nXGB gain")
            if r == 0:
                ax.set_title(phase)
    fig.suptitle(f"what the true-input bases lean on -- {tag}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_measurements(
    checkpoint_path: Path,
    *,
    device: torch.device | None = None,
    n_trials: int = 30,
    seed: int = 0,
) -> dict:
    """Both #46 measurements; metrics + figures to `results/<tag>/`."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tag = checkpoint_tag(checkpoint_path)
    output_dir = RESULTS_ROOT / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    print(f"device:     {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output:     {output_dir}")

    ckpt = JointHeadCheckpoint.load(checkpoint_path)
    val_ds, val_lat = val_dataset_and_latents(ckpt, device)
    f_val, val_draws = bulk_frequency_targets(val_ds)
    floors_val = binomial_floor(f_val, val_draws)
    print(f"val roster: {len(val_ds.bulk_ids)} bulks, "
          f"{int(np.median(val_draws))} draws/bulk (median)")

    mode_head = ckpt.build_mode_head().to(device)
    with torch.no_grad():
        p_head = torch.sigmoid(mode_head.presence_logits(val_lat)).cpu().numpy()

    lc = ckpt.label_config
    train_ds = JointHeadDataset(
        corpus_dir=lc["corpus_dir"],
        rungs=tuple(lc["rungs"]),
        presence_cutoff=lc["presence_cutoff"],
        comp_eps=lc["comp_eps"],
    )
    f_train, train_draws = bulk_frequency_targets(train_ds)
    print(f"train tier: {len(train_ds.bulk_ids)} bulks, "
          f"{int(np.median(train_draws))} draws/bulk (median)")

    fit_idx, hold_idx = train_test_split(
        np.arange(len(train_ds.bulk_ids)), test_size=0.2, random_state=seed,
    )

    bulk10_train = anhydrous_bulk_basis(
        train_ds.bulk_11.numpy(), train_ds.row_bulk_idx.numpy(),
        len(train_ds.bulk_ids),
    )
    bulk10_val = anhydrous_bulk_basis(
        val_ds.bulk_11.numpy(), val_ds.row_bulk_idx.numpy(), len(val_ds.bulk_ids),
    )
    token22_train = log_feature_matrix(
        train_ds.bulk_frac.numpy(), train_ds.bulk_src.numpy() != 0)
    token22_val = log_feature_matrix(
        val_ds.bulk_frac.numpy(), val_ds.bulk_src.numpy() != 0)
    bases = {
        "token22": (token22_train, token22_val),
        "bulk10": (bulk10_train, bulk10_val),
        "bulk9": (without_excess_o(bulk10_train), without_excess_o(bulk10_val)),
        "token22_xo": (
            np.column_stack([token22_train, bulk10_train[:, EXCESS_O_COL]]),
            np.column_stack([token22_val, bulk10_val[:, EXCESS_O_COL]]),
        ),
    }
    feature_names = {
        "token22": ELEMENT_SYMBOLS,
        "bulk10": list(ANHYDROUS_OXIDES),
        "bulk9": [o for o in ANHYDROUS_OXIDES if o != "excess_O"],
        "token22_xo": ELEMENT_SYMBOLS + ["excess_O"],
    }

    with open(RECORDED_TOKEN22_METRICS) as fh:
        recorded = json.load(fh)
    params_by_basis = {"token22": recorded["optuna"]["best_params"]}
    headline_cols = [SOLID_PHASES.index(p) for p in HEADLINE]
    for basis in ("bulk10", "bulk9", "token22_xo"):
        x_train, _ = bases[basis]
        print(f"tuning {basis} ({n_trials} trials, shared over {list(HEADLINE)}) ...")
        params_by_basis[basis] = tune_xgb(
            x_train[fit_idx], f_train[fit_idx],
            x_train[hold_idx], f_train[hold_idx],
            headline_cols, n_trials=n_trials, seed=seed,
        )
        print(f"  best: {params_by_basis[basis]}")

    preds_val, preds_hold, importances = {"head": p_head}, {}, {}
    for basis, (x_train, x_val) in bases.items():
        print(f"fitting {basis} (21 phases) ...")
        p_val, p_hold, imp = fit_basis_arm(
            x_train, x_val, f_train, fit_idx, hold_idx,
            params_by_basis[basis], seed=seed,
        )
        preds_val[basis] = p_val
        preds_hold[basis] = p_hold
        importances[basis] = imp

    scores_val = {arm: score_arm(p, f_val) for arm, p in preds_val.items()}

    anchor_recorded = recorded["scores"]["raw_xgb"]["spl"]["mae"]
    anchor_now = scores_val["token22"]["spl"]["mae"]
    anchor_ok = abs(anchor_now - anchor_recorded) <= 0.01
    print(f"\nsanity anchor: token22 spl MAE {anchor_now:.3f} vs recorded "
          f"{anchor_recorded:.3f} -> {'ok' if anchor_ok else 'MISMATCH'}")
    if not anchor_ok:
        print("  (instrument drifted from the presence_fraction_ceiling run; "
              "treat the basis comparison with suspicion)")
    head_headline = float(np.mean([scores_val["head"][p]["mae"] for p in HEADLINE]))
    head_recorded = recorded["verdict"]["headline_mean_mae"]["head"]
    print(f"head headline mean MAE {head_headline:.3f} vs "
          f"{head_recorded:.3f} recorded (different presence branch since the "
          "support-conditioned magnitude change; drift here is not an "
          "instrument fault)")

    print(f"\n{'phase':<6s} {'floor':>6s}   " + "".join(f"{a:>18s}" for a in scores_val))
    for i, phase in enumerate(SOLID_PHASES):
        star = "*" if phase in HEADLINE else " "
        cells = "".join(
            f"  {scores_val[a][phase]['mae']:.3f}/"
            f"{scores_val[a][phase]['pearson_r']:+.2f}".rjust(18)
            for a in scores_val
        )
        print(f"{phase:<5s}{star} {floors_val[i]:.3f} {cells}")
    print("  (cells are per-bulk MAE / pearson r on the val roster; * = headline)")

    prov_train = provenance_by_bulk(lc["corpus_dir"])
    prov_valtest = provenance_by_bulk(lc["valtest_corpus_dir"])
    prov_val = prov_valtest.reindex(val_ds.bulk_ids).to_numpy(dtype=object)
    prov_hold = prov_train.reindex(train_ds.bulk_ids[hold_idx]).to_numpy(dtype=object)
    if pd.isna(prov_val).any() or pd.isna(prov_hold).any():
        raise ValueError("bulks without a fe_provenance stamp in the draws tables")
    sg_val = map_qap_array(val_ds.bulk_qap_names)
    sg_hold = map_qap_array(train_ds.bulk_qap_names[hold_idx])

    reports = {
        "val": provenance_report(
            preds_val, f_val, val_draws, prov_val, sg_val, seed=seed),
        "train_holdout": provenance_report(
            preds_hold, f_train[hold_idx], train_draws[hold_idx],
            prov_hold, sg_hold, seed=seed),
    }

    for roster, report in reports.items():
        print(f"\nspl by fe_provenance ({roster}):")
        for arm, entry in report["by_arm"].items():
            cells = "  ".join(
                f"{g}: {s['scores']['spl']['mae']:.3f} (n={s['n']}, "
                f"f={s['mean_f']['spl']:.2f})"
                for g, s in sorted(entry["groups"].items())
            )
            print(f"  {arm:<11s} {cells}")
            delta = entry.get("spl_measured_minus_assumed")
            if delta:
                lo, hi = delta["ci95"]
                print(f"  {'':<11s} measured - assumed_class: "
                      f"{delta['delta_mae']:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    metrics = {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "n_val_bulks": int(len(val_ds.bulk_ids)),
        "n_holdout_bulks": int(len(hold_idx)),
        "optuna": {"n_trials": n_trials, "params_by_basis": params_by_basis},
        "sanity_anchor": {
            "token22_spl_mae": anchor_now,
            "recorded_spl_mae": anchor_recorded,
            "ok": bool(anchor_ok),
        },
        "binomial_floor_val": {
            p: float(floors_val[i]) for i, p in enumerate(SOLID_PHASES)
        },
        "scores_val": scores_val,
        "provenance": reports,
        "feature_names": feature_names,
        "feature_importances": importances,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    _plot_spl_by_provenance(reports, tag, output_dir / "spl_by_provenance.png")
    _plot_headline_by_basis(
        {a: s for a, s in scores_val.items() if a != "head"},
        floors_val, tag, output_dir / "headline_by_basis.png")
    _plot_importances(
        {b: importances[b] for b in ("bulk10", "token22_xo")},
        feature_names, tag, output_dir / "importance_true_input.png")
    print(f"\nmetrics -> {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="joint-heads checkpoint (heads_best.pt)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna trials per new feature basis")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_measurements(
        checkpoint_path=args.checkpoint.resolve(),
        n_trials=args.n_trials,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
