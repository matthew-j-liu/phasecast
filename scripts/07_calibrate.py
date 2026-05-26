"""Post-hoc temperature scaling on the v1 ensemble GMM CondHead (HANDOFF A2).

The v1 ensemble's condition uncertainty is miscalibrated: 1-sigma coverage
T/P/H2O = 0.88/0.64/0.76 against a nominal 0.683 (T and H2O underconfident, P
overconfident). This script fits one scalar temperature per axis on the val
split and applies it to test. No retraining: calibration multiplies every GMM
component sigma by exp(t_axis), so log_sigma -> log_sigma + t_axis.

This is temperature scaling in the sense of Guo et al. 2017, "On Calibration
of Modern Neural Networks" -- a single post-hoc scalar fit on validation by
minimizing NLL. Guo et al. divide a classifier's logits by T; the CondHead
emits a continuous density (a GMM per axis), so the analog is scaling the
predictive spread. The adopted temperature is the NLL-fit one. The script also
reports, per axis, the temperature that would instead match 1-sigma coverage
to 0.683, as a side-by-side diagnostic (Guo et al.'s reliability-diagram
analog) -- it is not adopted.

Writes calibration_v1.json into the ensemble dir. At inference, add the stored
per-axis t to each CondHead component log_sigma before forming the predictive
distribution.

Run:
    python scripts/07_calibrate.py --ensemble-dir checkpoints/base_v1_ensemble
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kstest, norm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# 06_eval_ensemble.py starts with a digit, so load it by path rather than import.
_spec = importlib.util.spec_from_file_location(
    "eval_ensemble", REPO_ROOT / "scripts" / "06_eval_ensemble.py")
_ee = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ee)

from cnmineral.models.dataset import TrainFCv1Dataset
from cnmineral.models.training import make_loaders_v1_bulk_split

AXES = ("T", "P", "H2O")
def stacked_gmm(member_outs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack K members into one (K * K_gmm)-component GMM per axis.

    Returns (means, log_sigmas, mix_logits, target_z): the first three of
    shape [N, 3, K_total], target_z of shape [N, 3].
    """
    means = [m["means"] for m in member_outs]
    log_sigmas = [m["log_sigmas"] for m in member_outs]
    mix_logits = [m["mix_logits"] for m in member_outs]
    ens_means, ens_log_sigmas, ens_mix_logits = _ee.stack_gmm_components_across_members(
        means, log_sigmas, mix_logits)
    return ens_means, ens_log_sigmas, ens_mix_logits, member_outs[0]["y_cond_z"]


# central probability mass of a Gaussian within +/- 1 and 2 sigma.
P_1SIGMA = 2.0 * norm.cdf(1.0) - 1.0  # 0.6827
P_2SIGMA = 2.0 * norm.cdf(2.0) - 1.0  # 0.9545


def pit_values(target: np.ndarray, means: np.ndarray, log_sigmas: np.ndarray,
               mix_logits: np.ndarray) -> np.ndarray:
    """Probability integral transform: GMM predictive CDF evaluated at target.

    target [N], means/log_sigmas/mix_logits [N, K]. Returns PIT [N]. If the
    predictive distribution is calibrated, PIT ~ Uniform(0, 1) -- this is the
    proper calibration check for a non-Gaussian (mixture) predictive, where
    mean +/- k*sqrt(variance) coverage is only a Gaussian approximation.
    """
    log_w = mix_logits - np.logaddexp.reduce(mix_logits, axis=-1, keepdims=True)
    w = np.exp(log_w)
    cdf_k = norm.cdf((target[:, None] - means) / np.exp(log_sigmas))
    return (w * cdf_k).sum(axis=-1)


def central_interval_coverage(pit: np.ndarray, p: float) -> float:
    """Fraction of PIT values inside the central-p interval [(1-p)/2, (1+p)/2].

    The distribution-correct analog of k-sigma coverage: for a calibrated
    predictive, this equals p exactly.
    """
    lo, hi = 0.5 * (1.0 - p), 0.5 * (1.0 + p)
    return float(((pit >= lo) & (pit <= hi)).mean())


def axis_metrics(target_z: np.ndarray, means: np.ndarray, log_sigmas: np.ndarray,
                 mix_logits: np.ndarray, axis: int) -> dict:
    """NLL, Gaussian-approx coverage, and PIT-based calibration for one axis."""
    tz, mu_c, ls, ml = (target_z[:, axis], means[:, axis],
                        log_sigmas[:, axis], mix_logits[:, axis])
    nll = float(_ee.gmm_nll_numpy(tz, mu_c, ls, ml).mean())
    mu, var = _ee.gmm_mean_var_1d(mu_c, ls, ml)
    sigma = np.sqrt(np.clip(var, 1e-12, None))
    pit = pit_values(tz, mu_c, ls, ml)
    return {
        "nll": nll,
        "coverage_1sigma_gauss": _ee.coverage_at_k_sigma(tz, mu, sigma, 1.0),
        "coverage_2sigma_gauss": _ee.coverage_at_k_sigma(tz, mu, sigma, 2.0),
        "coverage_1sigma_pit": central_interval_coverage(pit, P_1SIGMA),
        "coverage_2sigma_pit": central_interval_coverage(pit, P_2SIGMA),
        "pit_ks_stat": float(kstest(pit, "uniform").statistic),
        "pit_mean": float(pit.mean()),
    }


def fit_temperature_nll(target_z: np.ndarray, means: np.ndarray, log_sigmas: np.ndarray,
                        mix_logits: np.ndarray, axis: int) -> float:
    """Fit log-temperature t for one axis by minimizing val GMM NLL (Guo et al.).

    Returned t is in log space: temperature = exp(t), applied as
    log_sigma + t. Bounds [-3, 3] span sigma factors ~0.05x to ~20x.
    """
    tz, mu, ls, ml = target_z[:, axis], means[:, axis], log_sigmas[:, axis], mix_logits[:, axis]

    def nll_at(t: float) -> float:
        return float(_ee.gmm_nll_numpy(tz, mu, ls + t, ml).mean())

    return float(minimize_scalar(nll_at, bounds=(-3.0, 3.0), method="bounded").x)


def fit_temperature_coverage(target_z: np.ndarray, means: np.ndarray, log_sigmas: np.ndarray,
                             mix_logits: np.ndarray, axis: int) -> float:
    """Diagnostic only: log-temperature that brings the PIT central-68% interval
    coverage to 0.683. A single scalar can only fix this when the
    miscalibration is pure scale; the residual gap measures shape miscalibration.
    """
    tz, mu, ls, ml = target_z[:, axis], means[:, axis], log_sigmas[:, axis], mix_logits[:, axis]

    def cov_err(t: float) -> float:
        pit = pit_values(tz, mu, ls + t, ml)
        return (central_interval_coverage(pit, P_1SIGMA) - P_1SIGMA) ** 2

    return float(minimize_scalar(cov_err, bounds=(-3.0, 3.0), method="bounded").x)


def plot_pit_histograms(target_z: np.ndarray, means: np.ndarray,
                        log_sigmas_before: np.ndarray, log_sigmas_after: np.ndarray,
                        mix_logits: np.ndarray, out_path: Path) -> None:
    """PIT histogram per axis, before vs after calibration (test split).

    A calibrated predictive gives a flat PIT histogram. A U-shape means
    under-confident (intervals too wide), a dome means over-confident.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    bins = np.linspace(0, 1, 21)
    for a, ax_name in enumerate(AXES):
        ax = axes[a]
        pit_b = pit_values(target_z[:, a], means[:, a], log_sigmas_before[:, a], mix_logits[:, a])
        pit_a = pit_values(target_z[:, a], means[:, a], log_sigmas_after[:, a], mix_logits[:, a])
        ax.hist(pit_b, bins=bins, density=True, alpha=0.45, color="tab:gray", label="before")
        ax.hist(pit_a, bins=bins, density=True, histtype="step", lw=2.0,
                color="tab:blue", label="after")
        ax.axhline(1.0, color="k", ls="--", lw=1.0)
        ax.set_title(f"{ax_name}  (PIT, test)")
        ax.set_xlabel("PIT")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path,
                        default=REPO_ROOT / "checkpoints" / "base_v1_ensemble")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/base_v1.yaml")
    parser.add_argument("--splits-path", type=Path,
                        default=REPO_ROOT / "data/splits_v1.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _ee.pick_device()
    print(f"device: {device}")

    dataset = TrainFCv1Dataset(REPO_ROOT / cfg["data"]["parquet_path"],
                               liq_threshold=cfg["data"]["liq_threshold"])
    _, val_loader, test_loader, split_info = make_loaders_v1_bulk_split(
        dataset,
        val_frac=cfg["data"].get("val_frac", 0.16),
        test_frac=cfg["data"]["test_frac"],
        min_bulks_per_class=cfg["data"]["min_bulks_per_class"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
        num_workers=cfg["data"]["num_workers"],
        splits_path=str(args.splits_path),
    )
    print(f"val: {split_info['n_val_rows']} rows  test: {split_info['n_test_rows']} rows")

    ensemble_dir = args.ensemble_dir
    if not ensemble_dir.is_absolute():
        ensemble_dir = (REPO_ROOT / ensemble_dir).resolve()
    member_dirs = sorted(p for p in ensemble_dir.glob("seed_*") if p.is_dir())
    if not member_dirs:
        raise SystemExit(f"No seed_* subdirs in {ensemble_dir}")
    print(f"loading {len(member_dirs)} members from {ensemble_dir}")

    val_outs, test_outs = [], []
    for d in member_dirs:
        model, ck = _ee.load_member(d / "best.pt", device)
        val_outs.append(_ee.collect_member_outputs(model, val_loader, device))
        test_outs.append(_ee.collect_member_outputs(model, test_loader, device))
        print(f"  {d.name}: seed={ck.get('seed')}")

    v_means, v_log_sigmas, v_mix_logits, v_target = stacked_gmm(val_outs)
    t_means, t_log_sigmas, t_mix_logits, t_target = stacked_gmm(test_outs)

    temps_log = [fit_temperature_nll(v_target, v_means, v_log_sigmas, v_mix_logits, a)
                 for a in range(3)]
    cov_temps_log = [fit_temperature_coverage(v_target, v_means, v_log_sigmas, v_mix_logits, a)
                     for a in range(3)]
    temps = [float(np.exp(t)) for t in temps_log]
    cov_temps = [float(np.exp(t)) for t in cov_temps_log]

    scale = np.asarray(temps_log).reshape(1, 3, 1)
    v_scaled = v_log_sigmas + scale
    t_scaled = t_log_sigmas + scale

    report = {
        "method": "temperature scaling (Guo et al. 2017), GMM sigma variant",
        "fit_objective": "minimize val GMM NLL",
        "fit_split": "val",
        "nominal_central_coverage": {"1sigma": P_1SIGMA, "2sigma": P_2SIGMA},
        "axes": list(AXES),
        "temperatures": temps,
        "log_temperatures": temps_log,
        "diagnostic_coverage_match_temperatures": cov_temps,
        "diagnostic_coverage_match_log_temperatures": cov_temps_log,
        "val": {},
        "test": {},
    }
    for split, tgt, means, ml, ls_before, ls_after in (
        ("val", v_target, v_means, v_mix_logits, v_log_sigmas, v_scaled),
        ("test", t_target, t_means, t_mix_logits, t_log_sigmas, t_scaled),
    ):
        for a, ax in enumerate(AXES):
            report[split][ax] = {
                "before": axis_metrics(tgt, means, ls_before, ml, a),
                "after": axis_metrics(tgt, means, ls_after, ml, a),
            }

    out = args.out if args.out is not None else (ensemble_dir / "calibration_v1.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    plot_path = REPO_ROOT / "data" / "diagnostics" / "calibration" / "pit_hist_v1.png"
    plot_pit_histograms(t_target, t_means, t_log_sigmas, t_scaled, t_mix_logits, plot_path)

    print("\nfitted temperatures (sigma multiplier, NLL-fit -- adopted):")
    print("  " + "  ".join(f"{ax}={t:.3f}" for ax, t in zip(AXES, temps)))
    print("PIT-coverage-match temperatures (diagnostic only, not adopted):")
    print("  " + "  ".join(f"{ax}={t:.3f}" for ax, t in zip(AXES, cov_temps)))
    print(f"\nnominal central coverage: 1sigma={P_1SIGMA:.3f}  2sigma={P_2SIGMA:.3f}")
    print("cov1_g/cov2_g = Gaussian-approx (mean +/- k*sqrt(var)); "
          "cov1_pit/cov2_pit = PIT central-interval (distribution-correct)")
    for split in ("val", "test"):
        print(f"\n{split}:")
        print(f"  {'axis':5s} {'NLL b':>8s} {'NLL a':>8s} "
              f"{'cov1_g b':>9s} {'cov1_g a':>9s} "
              f"{'cov1_pit b':>11s} {'cov1_pit a':>11s} "
              f"{'cov2_pit a':>11s} {'KS b':>7s} {'KS a':>7s}")
        for ax in AXES:
            b, a = report[split][ax]["before"], report[split][ax]["after"]
            print(f"  {ax:5s} {b['nll']:8.4f} {a['nll']:8.4f} "
                  f"{b['coverage_1sigma_gauss']:9.3f} {a['coverage_1sigma_gauss']:9.3f} "
                  f"{b['coverage_1sigma_pit']:11.3f} {a['coverage_1sigma_pit']:11.3f} "
                  f"{a['coverage_2sigma_pit']:11.3f} "
                  f"{b['pit_ks_stat']:7.3f} {a['pit_ks_stat']:7.3f}")
    print(f"\ncalibration -> {out}")
    print(f"PIT histogram -> {plot_path}")


if __name__ == "__main__":
    main()
