"""Four-cell ablation evaluation for v2 (HANDOFF Phase 4 decision-rule input).

Compares (a) v1 baseline, (b) v2 traces + random init, (c) v2 majors-only +
pretrained init, (d) full v2 (traces + pretrained), on the SAME frozen
bulk-level test split. Per-phase R^2_bulk (the fair metric established by
B2 -- per-path R^2 is misleading for hydrous phases) plus joint-NLL and T MAE.

For (a) the existing checkpoints/base_v1_ensemble/predictions_test.parquet is
read directly. For (b/c/d), this script runs each model forward on the test
loader and writes predictions_test.parquet into the cell's checkpoint dir,
then computes the same R^2_bulk decomposition off it.

Run:
    python scripts/10_ablation_v2_eval.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnmineral.models.dataset import TrainFCv1Dataset, TrainFCv2Dataset
from cnmineral.models.model import BaseModelV1
from cnmineral.models.training import make_loaders_v1_bulk_split


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()):
        return torch.device("mps")
    return torch.device("cpu")


# Cells to compare. Each entry: (label, checkpoint_dir, requires_v2_dataset)
CELLS = [
    ("(a) v1 baseline",          "checkpoints/base_v1_ensemble", False),
    ("(b) v2 random init",       "checkpoints/base_v2_b_trace_nopretrain", True),
    ("(c) majors-only pretrain", "checkpoints/base_v2_c_pretrain_majorsonly", True),
    ("(d) full v2",              "checkpoints/base_v2_d_full", True),
]


def r2_bulk(pred_b: np.ndarray, target_b: np.ndarray, n_b: np.ndarray) -> float:
    """n_b-weighted R^2 at the per-bulk level.

    pred_b, target_b: per-bulk mean prediction/target (length n_bulks).
    n_b: paths per bulk (weights).
    """
    n_b = n_b.astype(np.float64)
    grand = (target_b * n_b).sum() / n_b.sum()
    ss_tot = (n_b * (target_b - grand) ** 2).sum() + 1e-12
    ss_res = (n_b * (pred_b - target_b) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def r2_path(pred: np.ndarray, target: np.ndarray) -> float:
    """Standard per-path R^2 (path-rows treated as independent test points)."""
    ss_tot = ((target - target.mean()) ** 2).sum() + 1e-12
    ss_res = ((pred - target) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def metrics_from_predictions(df: pd.DataFrame, phases: list[str]) -> dict:
    """Per-phase R^2_path and R^2_bulk + sanity-check ceilings."""
    out = {"r2_path": {}, "r2_bulk": {}, "ceiling": {}, "presence": {}}
    grouped = df.groupby("sample_id")
    n_b_full = grouped.size().to_numpy()
    for ph in phases:
        y = df[f"target_{ph}"].to_numpy()
        p = df[f"pred_{ph}"].to_numpy()
        out["r2_path"][ph] = r2_path(p, y)
        ybar_b = grouped[f"target_{ph}"].mean().to_numpy()
        pbar_b = grouped[f"pred_{ph}"].mean().to_numpy()
        out["r2_bulk"][ph] = r2_bulk(pbar_b, ybar_b, n_b_full)
        # Ceiling = SS_between / SS_tot (same for all cells — corpus property)
        grand = y.mean()
        ss_tot = ((y - grand) ** 2).sum() + 1e-12
        # SS_between: sum over rows of (ybar_b[row's bulk] - grand)^2
        ybar_b_per_row = grouped[f"target_{ph}"].transform("mean").to_numpy()
        ss_between = ((ybar_b_per_row - grand) ** 2).sum()
        out["ceiling"][ph] = float(ss_between / ss_tot)
        out["presence"][ph] = float((y > 0.01).mean())
    return out


def cond_mae_raw(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for ax in ("T", "P", "H2O"):
        out[ax] = float(np.abs(df[f"pred_{ax}"].to_numpy()
                               - df[f"target_{ax}"].to_numpy()).mean())
    return out


def build_predictions_for_v2_cell(
    cell_dir: Path, device: torch.device, parquet_path: Path,
    complete_csv_path: Path, splits_path: Path,
) -> pd.DataFrame:
    """Run a v2 cell's model forward on the test split, return per-row DataFrame
    with sample_id, path_idx, qap_name, pred_<phase>, target_<phase>,
    pred_<T,P,H2O>, target_<T,P,H2O>.
    """
    ckpt_path = cell_dir / "best.pt"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]

    dataset = TrainFCv2Dataset(
        parquet_path=parquet_path,
        complete_csv_path=complete_csv_path,
        liq_threshold=ck["liq_threshold"],
        majors_only=ck.get("majors_only", False),
    )
    _, _, test_loader, split_info = make_loaders_v1_bulk_split(
        dataset,
        val_frac=cfg["data"].get("val_frac", 0.16),
        test_frac=cfg["data"]["test_frac"],
        min_bulks_per_class=cfg["data"]["min_bulks_per_class"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
        num_workers=cfg["data"]["num_workers"],
        splits_path=str(splits_path),
    )
    test_set_arr = np.asarray(split_info["test_sample_ids"], dtype=np.int64)
    test_row_idx = np.where(np.isin(dataset.sample_ids, test_set_arr))[0]

    model = BaseModelV1(
        n_minerals=ck["n_minerals"], K_gmm=ck["K_gmm"],
        d_model=cfg["model"]["d_model"], N=cfg["model"]["N"],
        heads=cfg["model"]["heads"], dropout=cfg["model"]["dropout"],
        elem_prop=cfg["model"]["elem_prop"],
        mode_hidden=cfg["model"]["mode_hidden"],
        cond_hidden=cfg["model"]["cond_hidden"],
        alpha_floor=cfg["model"]["alpha_floor"],
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    cond_mean = np.asarray(ck["cond_mean"], dtype=np.float64)
    cond_std = np.asarray(ck["cond_std"], dtype=np.float64)
    phases = ck["phase_names"]

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["src"].to(device), batch["frac"].to(device))
            alpha = out["alpha"].cpu().numpy()
            modal_pred = alpha / alpha.sum(axis=-1, keepdims=True)
            modal_target = batch["y_modal"].numpy()
            # GMM cond mean = sum(softmax(mix_logits) * means).
            cm = out["cond_means"].cpu().numpy()
            cl = out["cond_mix_logits"].cpu().numpy()
            w = np.exp(cl - cl.max(axis=-1, keepdims=True))
            w = w / w.sum(axis=-1, keepdims=True)
            cond_pred_z = (w * cm).sum(axis=-1)
            cond_target_z = batch["y_cond_z"].numpy()
            cond_pred_raw = cond_pred_z * cond_std + cond_mean
            cond_target_raw = cond_target_z * cond_std + cond_mean
            for i in range(modal_pred.shape[0]):
                rows.append((modal_pred[i], modal_target[i],
                             cond_pred_raw[i], cond_target_raw[i]))

    data = {
        "sample_id": dataset.sample_ids[test_row_idx],
        "path_idx": dataset.path_idx[test_row_idx],
        "qap_name": dataset.qap_names[test_row_idx],
    }
    for j, ph in enumerate(phases):
        data[f"pred_{ph}"] = np.array([r[0][j] for r in rows])
        data[f"target_{ph}"] = np.array([r[1][j] for r in rows])
    for j, ax in enumerate(("T", "P", "H2O")):
        data[f"pred_{ax}"] = np.array([r[2][j] for r in rows])
        data[f"target_{ax}"] = np.array([r[3][j] for r in rows])
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases-of-interest",
                        default="amp,fl,liq,bi,mu,ne,cpx,ol,fsp,q",
                        help="Comma-separated phase names to print in the report.")
    args = parser.parse_args()
    phases_of_interest = args.phases_of_interest.split(",")

    device = pick_device()
    print(f"device: {device}")

    parquet_path = REPO_ROOT / "data/train_fc_v1.parquet"
    complete_csv_path = REPO_ROOT / "data/complete.csv"
    splits_path = REPO_ROOT / "data/splits_v1.json"

    all_cell_metrics = {}
    for label, cell_dir, is_v2 in CELLS:
        full_dir = REPO_ROOT / cell_dir
        pred_path = full_dir / "predictions_test.parquet"
        if not full_dir.exists():
            print(f"\n{label}: missing dir {full_dir}, skipping")
            continue
        if pred_path.exists():
            print(f"\n{label}: reading {pred_path.name}")
            df = pd.read_parquet(pred_path)
        elif is_v2:
            print(f"\n{label}: building predictions for v2 cell at {cell_dir}")
            df = build_predictions_for_v2_cell(
                full_dir, device, parquet_path, complete_csv_path, splits_path,
            )
            df.to_parquet(pred_path)
            print(f"  wrote {pred_path}")
        else:
            print(f"\n{label}: no predictions parquet and not a v2 cell, skipping")
            continue

        phases = [c[len("target_"):] for c in df.columns
                  if c.startswith("target_") and c[len("target_"):]
                  not in ("T", "P", "H2O")]
        metrics = metrics_from_predictions(df, phases)
        metrics["cond_mae_raw"] = cond_mae_raw(df)
        metrics["n_rows"] = int(len(df))
        metrics["n_bulks"] = int(df["sample_id"].nunique())
        all_cell_metrics[label] = metrics

    # Comparison table
    print("\n=== R^2_bulk per phase (the fair metric, B2-aware) ===")
    cell_labels = list(all_cell_metrics.keys())
    header = f"{'phase':>6s} {'ceiling':>9s}  " + "  ".join(f"{lbl[:18]:>18s}" for lbl in cell_labels)
    print(header)
    # Use ceiling from cell (a) since the v1 baseline shares the same labels
    # (corpus structure is fixed across cells).
    ref_ceil = next(iter(all_cell_metrics.values()))["ceiling"]
    for ph in phases_of_interest:
        row = f"{ph:>6s} {ref_ceil.get(ph, float('nan')):>9.3f}  "
        for lbl in cell_labels:
            v = all_cell_metrics[lbl]["r2_bulk"].get(ph, float("nan"))
            row += f"  {v:>+18.3f}"
        print(row)

    print("\n=== R^2_path per phase (per-path; the misleading-for-hydrous metric) ===")
    for ph in phases_of_interest:
        row = f"{ph:>6s} {'':>9s}  "
        for lbl in cell_labels:
            v = all_cell_metrics[lbl]["r2_path"].get(ph, float("nan"))
            row += f"  {v:>+18.3f}"
        print(row)

    print("\n=== Conditions MAE (raw units) ===")
    for ax in ("T", "P", "H2O"):
        row = f"{ax:>6s} {'':>9s}  "
        for lbl in cell_labels:
            v = all_cell_metrics[lbl]["cond_mae_raw"].get(ax, float("nan"))
            row += f"  {v:>+18.3f}"
        print(row)

    out_path = REPO_ROOT / "data/diagnostics/ablation_v2/comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_cell_metrics, f, indent=2)
    print(f"\nfull comparison -> {out_path}")


if __name__ == "__main__":
    main()
