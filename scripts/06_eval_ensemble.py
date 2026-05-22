"""Evaluate the K-member v1 ensemble on val and test splits.

Loads all member checkpoints, runs each through both val and test loaders,
aggregates Dirichlet alphas and GMM mixture parameters across members, and
writes a single `eval_v1.json` summary.

Aggregation conventions:
- Modal: ensemble Dirichlet alpha = mean(alpha_k) across K members. Ensemble
  predictive mean = alpha_ens / alpha_ens.sum(). Epistemic sigma per phase =
  std across member modal-mean predictions (per-sample), then averaged.
- Conditions: ensemble GMM = uniform-weight mixture of all K * K_gmm Gaussians
  per axis (each member contributes K_gmm components; final mixture weight is
  member's mix_weight / K). Predictive mean = sum(weights * means).

Run:
    python scripts/06_eval_ensemble.py \\
        --ensemble-dir checkpoints/base_v1_ensemble \\
        --out eval_v1.json

Optional --compare-dir runs the same eval on a second ensemble (e.g.,
checkpoints/base_v1_ensemble_rowsplit) and reports the leakage gap.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnmineral.models.dataset import TrainFCv1Dataset
from cnmineral.models.model import BaseModelV1
from cnmineral.models.training import make_loaders_v1_bulk_split


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
            and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_member(ckpt_path: Path, device: torch.device) -> tuple[BaseModelV1, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = BaseModelV1(
        n_minerals=ck["n_minerals"],
        K_gmm=ck["K_gmm"],
        d_model=cfg["model"]["d_model"],
        N=cfg["model"]["N"],
        heads=cfg["model"]["heads"],
        dropout=cfg["model"]["dropout"],
        elem_prop=cfg["model"]["elem_prop"],
        mode_hidden=cfg["model"].get("mode_hidden"),
        cond_hidden=cfg["model"].get("cond_hidden"),
        alpha_floor=cfg["model"].get("alpha_floor", 1e-3),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def collect_member_outputs(
    model: BaseModelV1, loader: DataLoader, device: torch.device,
) -> dict[str, np.ndarray]:
    alphas, means, log_sigmas, mix_logits = [], [], [], []
    y_modal, y_cond_z = [], []
    for batch in loader:
        src = batch["src"].to(device)
        frac = batch["frac"].to(device)
        out = model(src, frac)
        alphas.append(out["alpha"].cpu().numpy())
        means.append(out["cond_means"].cpu().numpy())
        log_sigmas.append(out["cond_log_sigmas"].cpu().numpy())
        mix_logits.append(out["cond_mix_logits"].cpu().numpy())
        y_modal.append(batch["y_modal"].numpy())
        y_cond_z.append(batch["y_cond_z"].numpy())
    return {
        "alpha": np.concatenate(alphas, 0),
        "means": np.concatenate(means, 0),
        "log_sigmas": np.concatenate(log_sigmas, 0),
        "mix_logits": np.concatenate(mix_logits, 0),
        "y_modal": np.concatenate(y_modal, 0),
        "y_cond_z": np.concatenate(y_cond_z, 0),
    }


def gmm_mean_var_1d(means: np.ndarray, log_sigmas: np.ndarray,
                    mix_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis mean + variance of a 1D GMM defined by component (mean, sigma, mix_logit).

    Inputs shape [..., K]. Returns mean, variance of same shape minus last K dim.
    """
    w = np.exp(mix_logits - mix_logits.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    sigmas = np.exp(log_sigmas)
    mu = (w * means).sum(axis=-1)
    second_moment = (w * (means ** 2 + sigmas ** 2)).sum(axis=-1)
    var = second_moment - mu ** 2
    return mu, var


def stack_gmm_components_across_members(
    member_means: list[np.ndarray],
    member_log_sigmas: list[np.ndarray],
    member_mix_logits: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate K members' (K_gmm components) into one (K * K_gmm)-component GMM.

    Each member contributes its own K_gmm Gaussians; the per-component mixture
    weight is divided by K so the ensemble's mixture sums to 1.
    """
    K = len(member_means)
    means = np.concatenate(member_means, axis=-1)
    log_sigmas = np.concatenate(member_log_sigmas, axis=-1)
    # convert each member's mix_logits to weights, divide by K, take log
    member_logits = np.stack(member_mix_logits, axis=0)
    member_w = np.exp(member_logits - member_logits.max(axis=-1, keepdims=True))
    member_w = member_w / member_w.sum(axis=-1, keepdims=True)
    ens_w = member_w / K
    ens_w_flat = np.moveaxis(ens_w, 0, -2)
    ens_w_flat = ens_w_flat.reshape(ens_w_flat.shape[:-2] + (-1,))
    ens_mix_logits = np.log(ens_w_flat + 1e-12)
    return means, log_sigmas, ens_mix_logits


def gmm_nll_numpy(target: np.ndarray, means: np.ndarray,
                   log_sigmas: np.ndarray, mix_logits: np.ndarray) -> np.ndarray:
    """Per-sample, per-axis 1D GMM NLL.

    target: [N, A]
    means/log_sigmas/mix_logits: [N, A, K_total]
    Returns: [N, A] negative log likelihood.
    """
    sigmas = np.exp(log_sigmas)
    log_w = mix_logits - np.logaddexp.reduce(mix_logits, axis=-1, keepdims=True)
    diff = target[..., None] - means
    log_normal = -0.5 * (np.log(2 * np.pi) + 2 * log_sigmas + (diff / sigmas) ** 2)
    log_mixed = log_w + log_normal
    log_p = np.logaddexp.reduce(log_mixed, axis=-1)
    return -log_p


def coverage_at_k_sigma(target: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                         k: float = 1.0) -> float:
    """Fraction of samples within mu +/- k*sigma."""
    inside = np.abs(target - mu) <= k * sigma
    return float(inside.mean())


def dirichlet_nll_numpy(alpha: np.ndarray, y: np.ndarray,
                         eps_smooth: float = 1e-3) -> float:
    """Mean Dirichlet NLL across the batch (matches losses.dirichlet_nll).

    Label-smooths y into the open simplex by eps_smooth so log(y) is finite.
    """
    from scipy.special import gammaln
    P = y.shape[-1]
    y_s = (1.0 - eps_smooth) * y + eps_smooth / P
    y_s = y_s / y_s.sum(axis=-1, keepdims=True)
    log_p = (gammaln(alpha.sum(axis=-1))
             - gammaln(alpha).sum(axis=-1)
             + ((alpha - 1.0) * np.log(y_s)).sum(axis=-1))
    return float(-log_p.mean())


def eval_split(member_outs: list[dict], cond_mean: np.ndarray, cond_std: np.ndarray,
               phase_names: list[str], sparsity_threshold: float = 0.01) -> dict:
    K = len(member_outs)
    A = np.stack([m["alpha"] for m in member_outs], axis=0)
    ens_alpha = A.mean(axis=0)
    ens_modal = ens_alpha / ens_alpha.sum(axis=-1, keepdims=True)
    y_modal = member_outs[0]["y_modal"]

    member_modal = A / A.sum(axis=-1, keepdims=True)
    epistemic_modal = member_modal.std(axis=0)
    member_modal_mean = member_modal.mean(axis=0)
    ens_alpha_sum = ens_alpha.sum(axis=-1, keepdims=True)
    aleatoric_modal_var = (ens_modal * (1.0 - ens_modal)) / (ens_alpha_sum + 1.0)

    ss_res = ((ens_modal - y_modal) ** 2).sum(axis=0)
    ss_tot = ((y_modal - y_modal.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    presence = (y_modal > sparsity_threshold).mean(axis=0)

    member_r2 = []
    for k in range(K):
        mm = A[k] / A[k].sum(axis=-1, keepdims=True)
        ss_r = ((mm - y_modal) ** 2).sum(axis=0)
        member_r2.append(1.0 - ss_r / ss_tot)
    member_r2 = np.stack(member_r2, axis=0)

    means = [m["means"] for m in member_outs]
    log_sigmas = [m["log_sigmas"] for m in member_outs]
    mix_logits = [m["mix_logits"] for m in member_outs]
    ens_means, ens_log_sigmas, ens_mix_logits = stack_gmm_components_across_members(
        means, log_sigmas, mix_logits,
    )
    cond_target_z = member_outs[0]["y_cond_z"]
    ens_mu_z, ens_var_z = gmm_mean_var_1d(ens_means, ens_log_sigmas, ens_mix_logits)
    ens_sigma_z = np.sqrt(np.clip(ens_var_z, 1e-12, None))
    cond_mae_z = np.abs(ens_mu_z - cond_target_z).mean(axis=0)
    cond_mae_raw = cond_mae_z * cond_std

    nll_per_sample_axis = gmm_nll_numpy(cond_target_z, ens_means,
                                         ens_log_sigmas, ens_mix_logits)
    cond_nll_per_axis = nll_per_sample_axis.mean(axis=0)
    nll_gmm = float(nll_per_sample_axis.mean())
    nll_dir = dirichlet_nll_numpy(ens_alpha, y_modal)
    joint_nll_simple = 0.5 * nll_dir + 0.5 * nll_gmm

    cov_1sigma = [coverage_at_k_sigma(cond_target_z[:, a], ens_mu_z[:, a],
                                       ens_sigma_z[:, a], 1.0)
                   for a in range(3)]
    cov_2sigma = [coverage_at_k_sigma(cond_target_z[:, a], ens_mu_z[:, a],
                                       ens_sigma_z[:, a], 2.0)
                   for a in range(3)]

    return {
        "n_samples": int(y_modal.shape[0]),
        "nll_dir": nll_dir,
        "nll_gmm": nll_gmm,
        "joint_nll_simple": joint_nll_simple,
        "r2_per_phase_ensemble": r2.tolist(),
        "r2_per_phase_members": member_r2.tolist(),
        "phase_presence": presence.tolist(),
        "phase_names": phase_names,
        "epistemic_sigma_modal_mean": epistemic_modal.mean(axis=0).tolist(),
        "aleatoric_sigma_modal_mean": np.sqrt(aleatoric_modal_var).mean(axis=0).tolist(),
        "cond_mae_z": cond_mae_z.tolist(),
        "cond_mae_raw": cond_mae_raw.tolist(),
        "cond_nll_per_axis_z": cond_nll_per_axis.tolist(),
        "coverage_1sigma": cov_1sigma,
        "coverage_2sigma": cov_2sigma,
    }


def compute_per_sample_predictions(
    member_outs: list[dict],
    dataset,
    row_indices: np.ndarray,
    phase_names: list[str],
    cond_axes: tuple,
) -> "pd.DataFrame":
    """Build a per-row DataFrame of ensemble predictions for a split.

    One row per path-row in the split, with sample_id / path_idx / qap_name
    plus ensemble Dirichlet mean + epistemic sigma per phase, ensemble GMM
    mean + sigma per condition axis (de-normalized to raw units), and the
    ground-truth modal vector + raw conditions for parity plotting.
    """
    import pandas as pd

    K = len(member_outs)
    A = np.stack([m["alpha"] for m in member_outs], axis=0)
    ens_alpha = A.mean(axis=0)
    ens_modal = ens_alpha / ens_alpha.sum(axis=-1, keepdims=True)
    member_modal = A / A.sum(axis=-1, keepdims=True)
    epistemic_modal = member_modal.std(axis=0)
    y_modal = member_outs[0]["y_modal"]

    means = [m["means"] for m in member_outs]
    log_sigmas = [m["log_sigmas"] for m in member_outs]
    mix_logits = [m["mix_logits"] for m in member_outs]
    ens_means, ens_log_sigmas, ens_mix_logits = stack_gmm_components_across_members(
        means, log_sigmas, mix_logits,
    )
    ens_mu_z, ens_var_z = gmm_mean_var_1d(ens_means, ens_log_sigmas, ens_mix_logits)
    ens_sigma_z = np.sqrt(np.clip(ens_var_z, 1e-12, None))

    cond_std = np.asarray(dataset.cond_std, dtype=np.float64)
    cond_mean = np.asarray(dataset.cond_mean, dtype=np.float64)
    ens_mu_raw = ens_mu_z * cond_std + cond_mean
    ens_sigma_raw = ens_sigma_z * cond_std
    y_cond_z = member_outs[0]["y_cond_z"]
    y_cond_raw = y_cond_z * cond_std + cond_mean

    sample_ids = dataset.sample_ids[row_indices]
    path_idx = dataset.path_idx[row_indices]
    qap_names_per_row = dataset.qap_names[row_indices]

    data = {"sample_id": sample_ids, "path_idx": path_idx, "qap_name": qap_names_per_row}
    for i, ph in enumerate(phase_names):
        data[f"pred_{ph}"] = ens_modal[:, i]
        data[f"epi_{ph}"] = epistemic_modal[:, i]
        data[f"target_{ph}"] = y_modal[:, i]
    axis_keys = ("T", "P", "H2O")
    for i, ax in enumerate(axis_keys):
        data[f"pred_{ax}"] = ens_mu_raw[:, i]
        data[f"sigma_{ax}"] = ens_sigma_raw[:, i]
        data[f"target_{ax}"] = y_cond_raw[:, i]
    return pd.DataFrame(data)


def _per_class_metrics(
    member_outs: list[dict],
    qap_names_per_row: np.ndarray,
    phase_names: list[str],
    cond_std: np.ndarray,
    sparsity_threshold: float = 0.01,
    min_presence_for_r2: float = 0.01,
) -> dict:
    """Per-qap_name breakdown of R², condition MAE, and coverage.

    Reuses the same ensemble aggregation as `eval_split` but groups rows by
    class. Per-phase R² for a class is only reported when that class's
    presence for the phase is at least `min_presence_for_r2` (otherwise the
    denominator is too small to be interpretable).
    """
    K = len(member_outs)
    A = np.stack([m["alpha"] for m in member_outs], axis=0)
    ens_alpha = A.mean(axis=0)
    ens_modal = ens_alpha / ens_alpha.sum(axis=-1, keepdims=True)
    y_modal = member_outs[0]["y_modal"]

    means = [m["means"] for m in member_outs]
    log_sigmas = [m["log_sigmas"] for m in member_outs]
    mix_logits = [m["mix_logits"] for m in member_outs]
    ens_means, ens_log_sigmas, ens_mix_logits = stack_gmm_components_across_members(
        means, log_sigmas, mix_logits,
    )
    ens_mu_z, ens_var_z = gmm_mean_var_1d(ens_means, ens_log_sigmas, ens_mix_logits)
    ens_sigma_z = np.sqrt(np.clip(ens_var_z, 1e-12, None))
    y_cond_z = member_outs[0]["y_cond_z"]

    per_class = {}
    for cls in np.unique(qap_names_per_row):
        mask = qap_names_per_row == cls
        n = int(mask.sum())
        if n == 0:
            continue
        ym = y_modal[mask]
        pm = ens_modal[mask]
        ss_res = ((pm - ym) ** 2).sum(axis=0)
        ss_tot = ((ym - ym.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        presence = (ym > sparsity_threshold).mean(axis=0)
        per_phase_r2 = {ph: float(r2[i]) for i, ph in enumerate(phase_names)
                        if presence[i] >= min_presence_for_r2}

        cmu = ens_mu_z[mask]
        csig = ens_sigma_z[mask]
        ct = y_cond_z[mask]
        cond_mae_z = np.abs(cmu - ct).mean(axis=0)
        cond_mae_raw = cond_mae_z * cond_std
        cov_1sigma = [float((np.abs(ct[:, a] - cmu[:, a]) <= csig[:, a]).mean())
                      for a in range(ct.shape[1])]

        per_class[str(cls)] = {
            "n_samples": n,
            "per_phase_r2": per_phase_r2,
            "phase_presence": {ph: float(presence[i])
                               for i, ph in enumerate(phase_names)},
            "cond_mae_z": cond_mae_z.tolist(),
            "cond_mae_raw": cond_mae_raw.tolist(),
            "coverage_1sigma": cov_1sigma,
        }
    return per_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path, default=None,
                        help="Optional second ensemble (e.g. row-leaked) to "
                             "re-evaluate on the SAME bulk-level splits, for a "
                             "leakage-gap data point. Adds a `leakage_gap` "
                             "block to the output JSON.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/base_v1.yaml")
    parser.add_argument("--splits-path", type=Path,
                        default=REPO_ROOT / "data/splits_v1.json")
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--min-bulks-per-class", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = pick_device()
    print(f"device: {device}")

    parquet_path = REPO_ROOT / cfg["data"]["parquet_path"]
    dataset = TrainFCv1Dataset(parquet_path, liq_threshold=cfg["data"]["liq_threshold"])
    train_loader, val_loader, test_loader, split_info = make_loaders_v1_bulk_split(
        dataset,
        val_frac=cfg["data"].get("val_frac", 0.16),
        test_frac=args.test_frac,
        min_bulks_per_class=args.min_bulks_per_class,
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
        num_workers=cfg["data"]["num_workers"],
        splits_path=str(args.splits_path),
    )
    print(f"val: {split_info['n_val_rows']} rows / {split_info['n_val_bulks']} bulks  "
          f"test: {split_info['n_test_rows']} rows / {split_info['n_test_bulks']} bulks")

    val_set_arr = np.asarray(split_info["val_sample_ids"], dtype=np.int64)
    test_set_arr = np.asarray(split_info["test_sample_ids"], dtype=np.int64)
    val_row_idx = np.where(np.isin(dataset.sample_ids, val_set_arr))[0]
    test_row_idx = np.where(np.isin(dataset.sample_ids, test_set_arr))[0]

    cond_mean = np.asarray(dataset.cond_mean, dtype=np.float64)
    cond_std = np.asarray(dataset.cond_std, dtype=np.float64)

    def _eval_one_ensemble(ensemble_dir: Path) -> tuple[dict, dict, list, list]:
        ensemble_dir = ensemble_dir if ensemble_dir.is_absolute() else (REPO_ROOT / ensemble_dir).resolve()
        member_dirs = sorted(p for p in ensemble_dir.glob("seed_*") if p.is_dir())
        if not member_dirs:
            raise SystemExit(f"No seed_* subdirs in {ensemble_dir}")
        print(f"loading {len(member_dirs)} member checkpoints from {ensemble_dir}")
        v_outs, t_outs = [], []
        for d in member_dirs:
            ckpt = d / "best.pt"
            model, ck = load_member(ckpt, device)
            v_outs.append(collect_member_outputs(model, val_loader, device))
            t_outs.append(collect_member_outputs(model, test_loader, device))
            print(f"  loaded {d.name}: seed={ck.get('seed')} best_epoch={ck.get('best_epoch')}")
        eval_v = eval_split(v_outs, cond_mean, cond_std, dataset.phase_names)
        eval_t = eval_split(t_outs, cond_mean, cond_std, dataset.phase_names)
        eval_v["per_class"] = _per_class_metrics(
            v_outs, dataset.qap_names[val_row_idx], dataset.phase_names, cond_std,
        )
        eval_t["per_class"] = _per_class_metrics(
            t_outs, dataset.qap_names[test_row_idx], dataset.phase_names, cond_std,
        )
        return eval_v, eval_t, v_outs, t_outs

    ensemble_dir = args.ensemble_dir
    if not ensemble_dir.is_absolute():
        ensemble_dir = (REPO_ROOT / ensemble_dir).resolve()
    eval_val, eval_test, val_outs, test_outs = _eval_one_ensemble(args.ensemble_dir)

    pred_val_df = compute_per_sample_predictions(
        val_outs, dataset, val_row_idx, dataset.phase_names, dataset.cond_axes,
    )
    pred_test_df = compute_per_sample_predictions(
        test_outs, dataset, test_row_idx, dataset.phase_names, dataset.cond_axes,
    )

    summary = {
        "ensemble_dir": str(ensemble_dir),
        "K": len(val_outs),
        "device": str(device),
        "split_info": {k: split_info[k] for k in split_info
                       if k not in ("train_sample_ids", "val_sample_ids",
                                    "test_sample_ids", "trainval_pool_sample_ids",
                                    "dropped_class_sample_ids")},
        "phase_names": dataset.phase_names,
        "cond_axes": list(dataset.cond_axes),
        "cond_mean": dataset.cond_mean.tolist(),
        "cond_std": dataset.cond_std.tolist(),
        "val": eval_val,
        "test": eval_test,
    }

    if args.compare_dir is not None:
        print(f"\n=== running --compare-dir on {args.compare_dir} (leakage-gap eval) ===")
        cmp_val, cmp_test, _, _ = _eval_one_ensemble(args.compare_dir)
        summary["leakage_gap"] = {
            "compare_dir": str(args.compare_dir),
            "val": {
                "joint_nll_simple_bulk_split": eval_val["joint_nll_simple"],
                "joint_nll_simple_compare_reeval": cmp_val["joint_nll_simple"],
                "delta_joint_nll": cmp_val["joint_nll_simple"] - eval_val["joint_nll_simple"],
                "cond_mae_raw_bulk_split": eval_val["cond_mae_raw"],
                "cond_mae_raw_compare_reeval": cmp_val["cond_mae_raw"],
                "r2_per_phase_bulk_split": eval_val["r2_per_phase_ensemble"],
                "r2_per_phase_compare_reeval": cmp_val["r2_per_phase_ensemble"],
            },
            "test": {
                "joint_nll_simple_bulk_split": eval_test["joint_nll_simple"],
                "joint_nll_simple_compare_reeval": cmp_test["joint_nll_simple"],
                "delta_joint_nll": cmp_test["joint_nll_simple"] - eval_test["joint_nll_simple"],
                "cond_mae_raw_bulk_split": eval_test["cond_mae_raw"],
                "cond_mae_raw_compare_reeval": cmp_test["cond_mae_raw"],
                "r2_per_phase_bulk_split": eval_test["r2_per_phase_ensemble"],
                "r2_per_phase_compare_reeval": cmp_test["r2_per_phase_ensemble"],
            },
            "phase_names": dataset.phase_names,
        }

    out = args.out if args.out is not None else (ensemble_dir / "eval_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\neval summary -> {out}")

    pred_val_path = out.parent / "predictions_val.parquet"
    pred_test_path = out.parent / "predictions_test.parquet"
    pred_val_df.to_parquet(pred_val_path)
    pred_test_df.to_parquet(pred_test_path)
    print(f"predictions val/test -> {pred_val_path.name}, {pred_test_path.name}")

    print("\nval per-phase R^2 (ensemble) sorted by presence:")
    order = sorted(range(len(dataset.phase_names)),
                   key=lambda i: -eval_val["phase_presence"][i])
    for i in order:
        print(f"  {dataset.phase_names[i]:6s}  pres={eval_val['phase_presence'][i]:.3f}  "
              f"R2_val={eval_val['r2_per_phase_ensemble'][i]:+.3f}  "
              f"R2_test={eval_test['r2_per_phase_ensemble'][i]:+.3f}")
    print(f"\nT MAE: val={eval_val['cond_mae_raw'][0]:.2f}  test={eval_test['cond_mae_raw'][0]:.2f}")
    print(f"P MAE: val={eval_val['cond_mae_raw'][1]:.2f}  test={eval_test['cond_mae_raw'][1]:.2f}")
    print(f"H2O MAE: val={eval_val['cond_mae_raw'][2]:.3f}  test={eval_test['cond_mae_raw'][2]:.3f}")
    print(f"coverage 1-sigma (T, P, H2O): val={eval_val['coverage_1sigma']}  "
          f"test={eval_test['coverage_1sigma']}")
    print(f"coverage 2-sigma: val={eval_val['coverage_2sigma']}  test={eval_test['coverage_2sigma']}")


if __name__ == "__main__":
    main()
