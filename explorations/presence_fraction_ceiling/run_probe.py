"""
Where is the ceiling on per-bulk presence-fraction discrimination? -- the #41
successor gate.  The ModeHead's accessory failure (spl/ilm/ru/sph) is not
marginal calibration but per-bulk discrimination: it knows base rates yet
hedges mid-range because it can't tell which rocks are the high ones.  f_bulk
(a phase's fraction of a bulk's draws) is a deterministic function of chemistry
up to binomial sampling noise, so three predictors bound the problem: the head
itself, an XGBoost on raw bulk chemistry (the input ceiling), and an MLP probe
on the frozen pooled latent.  raw ~ head -> data ceiling, enrich the corpus;
raw > head and latent > head -> head-side fix; raw > head but latent ~ head ->
the encoder dropped the signal, feed bulk_frac to the ModeHead directly.
Read-only -- nothing here trains or mutates production models.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from xgboost import XGBRegressor

from cnmineral.data.mineralogy_dataset import JointHeadDataset
from cnmineral.data.phase_vocab import N_PHASES, SOLID_PHASES
from cnmineral.data.tokens import VOCAB
from cnmineral.eval.joint_heads import val_dataset_and_latents
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import within_bulk_frequency
from cnmineral.training import checkpoint_tag, pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
HEADLINE = ("spl", "ilm", "ru", "sph")
ELEMENT_SYMBOLS = [s for s, _z in VOCAB]


def bulk_frequency_targets(dataset: JointHeadDataset) -> tuple[np.ndarray, np.ndarray]:
    """Per-bulk presence fraction `f` [n_bulks, 21] and draws per bulk [n_bulks].

    Rows are (draw, rung) pairs; `within_bulk_frequency` pools the rungs, so
    draws ~ rows / n_rungs.  The rungs of one draw are correlated, so the
    binomial floor is quoted at n = draws (conservative).
    """
    f, n_rows = within_bulk_frequency(
        dataset.presence.numpy(),
        dataset.row_bulk_idx.numpy(),
        len(dataset.bulk_ids),
    )
    if not np.isfinite(f).all():
        raise ValueError("some bulks contribute no rows; targets undefined")
    n_rungs = len(torch.unique(dataset.rung))
    return f, n_rows / n_rungs


def log_feature_matrix(frac: np.ndarray, populated: np.ndarray) -> np.ndarray:
    """`log10(frac)` where populated, NaN where absent (XGBoost native missing)."""
    out = np.full(frac.shape, np.nan, dtype=np.float64)
    out[populated] = np.log10(frac[populated])
    return out


def binomial_floor(f: np.ndarray, n_draws: np.ndarray) -> np.ndarray:
    """Per-phase expected MAE of the truth estimator itself, [P].

    A perfect predictor of the underlying rate still shows |f_hat - f| noise
    from the finite draw count; for a normal approximation E|x| = sqrt(2/pi)*sd.
    """
    sd = np.sqrt(f * (1.0 - f) / n_draws[:, None])
    return np.sqrt(2.0 / np.pi) * sd.mean(axis=0)


def score_arm(pred: np.ndarray, truth: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-phase per-bulk MAE + Pearson r of `pred` vs `truth`, both [B, P]."""
    out = {}
    for i, phase in enumerate(SOLID_PHASES):
        p, t = pred[:, i], truth[:, i]
        mae = float(np.abs(p - t).mean())
        if p.std() > 0 and t.std() > 0:
            r = float(np.corrcoef(p, t)[0, 1])
        else:
            r = float("nan")
        out[phase] = {"mae": mae, "pearson_r": r}
    return out


def _suggest_params(trial: optuna.trial.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def _fit_xgb(
    params: dict,
    x_fit: np.ndarray, y_fit: np.ndarray,
    x_hold: np.ndarray, y_hold: np.ndarray,
    *,
    seed: int,
) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=2000,
        early_stopping_rounds=50,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=seed,
        **params,
    )
    model.fit(x_fit, y_fit, eval_set=[(x_hold, y_hold)], verbose=False)
    return model


def tune_xgb(
    x_fit: np.ndarray, y_fit: np.ndarray,
    x_hold: np.ndarray, y_hold: np.ndarray,
    headline_cols: list[int],
    *,
    n_trials: int,
    seed: int,
) -> dict:
    """One shared study; objective = mean hold-out MSE over the headline phases."""

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(trial)
        mses = []
        for c in headline_cols:
            model = _fit_xgb(params, x_fit, y_fit[:, c], x_hold, y_hold[:, c], seed=seed)
            pred = np.clip(model.predict(x_hold), 0.0, 1.0)
            mses.append(float(np.mean((pred - y_hold[:, c]) ** 2)))
        return float(np.mean(mses))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


class FractionProbe(nn.Module):
    """1-hidden-layer read-out of f_bulk from the pooled latent (house probe depth)."""

    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_PHASES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_latent_probe(
    x_fit: np.ndarray, y_fit: np.ndarray,
    x_hold: np.ndarray, y_hold: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    epochs: int = 300,
    patience: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> FractionProbe:
    """BCE-with-logits on the soft f targets -- the head's own objective family,
    so the probe reads what a head of this shape could extract from the latent."""
    torch.manual_seed(seed)
    probe = FractionProbe(x_fit.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    xf = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    yf = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    xh = torch.as_tensor(x_hold, dtype=torch.float32, device=device)
    yh = torch.as_tensor(y_hold, dtype=torch.float32, device=device)
    best_loss, best_state, since_best = float("inf"), None, 0
    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(len(xf), device=device)
        for start in range(0, len(xf), batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(probe(xf[idx]), yf[idx])
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            hold = float(nn.functional.binary_cross_entropy_with_logits(probe(xh), yh))
        if hold < best_loss:
            best_loss, since_best = hold, 0
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    probe.load_state_dict(best_state)
    probe.eval()
    return probe


def knn_frequency(
    train_latents: np.ndarray, train_f: np.ndarray, query_latents: np.ndarray,
    *, k: int = 10,
) -> np.ndarray:
    """Mean f over the k cosine-nearest train bulks -- hyperparameter-free
    geometry read in the spirit of `knn_label_purity`."""
    a = train_latents / np.linalg.norm(train_latents, axis=1, keepdims=True)
    b = query_latents / np.linalg.norm(query_latents, axis=1, keepdims=True)
    sims = b @ a.T
    nn_idx = np.argpartition(-sims, k, axis=1)[:, :k]
    return train_f[nn_idx].mean(axis=1)


def build_verdict(
    scores: dict[str, dict], margin: float = 0.02
) -> dict:
    """Advisory routing on mean headline-phase MAE; the numbers are the decision.

    `margin` is the binomial-noise scale below which two arms are a tie."""
    means = {
        arm: float(np.mean([per_phase[p]["mae"] for p in HEADLINE]))
        for arm, per_phase in scores.items()
    }
    raw_gain = means["head"] - means["raw_xgb"]
    latent_gain = means["head"] - means["latent_mlp"]
    if raw_gain <= margin:
        route = (
            "raw chemistry does not beat the head: the corpus is the ceiling; "
            "enrichment (more draws / bulks) is the lever, stop architecture work"
        )
    elif latent_gain > margin:
        route = (
            "raw chemistry and the frozen latent both beat the head: the signal "
            "survives the encoder; fix the head side (per-phase presence-BCE "
            "weighting and/or capacity)"
        )
    else:
        route = (
            "raw chemistry beats the head but the frozen latent does not: the "
            "encoder dropped the signal; feed bulk_frac to the ModeHead via a "
            "skip connection"
        )
    return {
        "headline_phases": list(HEADLINE),
        "tie_margin_mae": margin,
        "headline_mean_mae": means,
        "raw_gain_over_head": raw_gain,
        "latent_gain_over_head": latent_gain,
        "route": route,
    }


def _plot_mae_by_phase(
    scores: dict[str, dict], floors: np.ndarray, tag: str, out_path: Path
) -> None:
    arms = [
        ("head", "ModeHead"),
        ("raw_xgb", "raw-chemistry XGB"),
        ("latent_mlp", "latent MLP probe"),
        ("latent_knn", "latent kNN"),
    ]
    x = np.arange(N_PHASES)
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for j, (arm, label) in enumerate(arms):
        vals = [scores[arm][p]["mae"] for p in SOLID_PHASES]
        ax.bar(x + (j - 1.5) * w, vals, w, label=label)
    ax.scatter(x, floors, marker="_", s=220, color="black", zorder=3,
               label="binomial floor (val draws)")
    ax.set_xticks(x)
    ax.set_xticklabels(SOLID_PHASES, rotation=45, ha="right")
    for tick in ax.get_xticklabels():
        if tick.get_text() in HEADLINE:
            tick.set_fontweight("bold")
    ax.set_ylabel("per-bulk MAE of presence fraction")
    ax.set_title(f"presence-fraction discrimination by arm -- {tag}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_importances(importances: dict[str, list], tag: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(4 * len(HEADLINE), 3.5))
    x = np.arange(len(ELEMENT_SYMBOLS))
    for ax, phase in zip(axes, HEADLINE):
        ax.bar(x, importances[phase])
        ax.set_xticks(x)
        ax.set_xticklabels(ELEMENT_SYMBOLS, rotation=90, fontsize=7)
        ax.set_title(phase)
    axes[0].set_ylabel("XGB feature importance (gain)")
    fig.suptitle(f"what chemistry drives f_bulk -- {tag}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_ceiling_probe(
    checkpoint_path: Path,
    *,
    device: torch.device | None = None,
    n_trials: int = 30,
    seed: int = 0,
    tie_margin: float = 0.02,
) -> dict:
    """Score the three arms on the val roster; write metrics + figures to
    `results/<tag>/` and return the metrics dict."""
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
        logits = mode_head.presence_logits(val_lat)
    p_head = torch.sigmoid(logits).cpu().numpy()

    lc = ckpt.label_config
    train_ds = JointHeadDataset(
        corpus_dir=lc["corpus_dir"],
        rungs=tuple(lc["rungs"]),
        presence_cutoff=lc["presence_cutoff"],
        comp_eps=lc["comp_eps"],
    )
    f_train, train_draws = bulk_frequency_targets(train_ds)
    print(f"train tier: {len(train_ds.bulk_ids)} bulks, "
          f"{int(np.median(train_draws))} draws/bulk (median); "
          f"targets carry ~{binomial_floor(f_train, train_draws).max():.3f} "
          f"worst-phase binomial MAE of their own")

    fit_idx, hold_idx = train_test_split(
        np.arange(len(train_ds.bulk_ids)), test_size=0.2, random_state=seed,
    )

    x_train = log_feature_matrix(
        train_ds.bulk_frac.numpy(), train_ds.bulk_src.numpy() != 0)
    x_val = log_feature_matrix(
        val_ds.bulk_frac.numpy(), val_ds.bulk_src.numpy() != 0)
    headline_cols = [SOLID_PHASES.index(p) for p in HEADLINE]

    print(f"tuning xgb ({n_trials} trials, shared over {list(HEADLINE)}) ...")
    best_params = tune_xgb(
        x_train[fit_idx], f_train[fit_idx],
        x_train[hold_idx], f_train[hold_idx],
        headline_cols, n_trials=n_trials, seed=seed,
    )
    print(f"  best: {best_params}")

    p_raw = np.zeros_like(p_head)
    importances = {}
    for i, phase in enumerate(SOLID_PHASES):
        model = _fit_xgb(
            best_params,
            x_train[fit_idx], f_train[fit_idx, i],
            x_train[hold_idx], f_train[hold_idx, i],
            seed=seed,
        )
        p_raw[:, i] = np.clip(model.predict(x_val), 0.0, 1.0)
        importances[phase] = [float(v) for v in model.feature_importances_]

    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    train_lat = bulk_latents(encoder, train_ds, device).cpu().numpy()
    val_lat_np = val_lat.cpu().numpy()

    probe = train_latent_probe(
        train_lat[fit_idx], f_train[fit_idx],
        train_lat[hold_idx], f_train[hold_idx],
        device=device, seed=seed,
    )
    with torch.no_grad():
        p_mlp = torch.sigmoid(
            probe(torch.as_tensor(val_lat_np, dtype=torch.float32, device=device))
        ).cpu().numpy()

    p_knn = knn_frequency(train_lat, f_train, val_lat_np)

    scores = {
        "head": score_arm(p_head, f_val),
        "raw_xgb": score_arm(p_raw, f_val),
        "latent_mlp": score_arm(p_mlp, f_val),
        "latent_knn": score_arm(p_knn, f_val),
    }
    verdict = build_verdict(scores, margin=tie_margin)

    print(f"\n{'phase':<6s} {'floor':>6s}   "
          + "".join(f"{arm:>18s}" for arm in scores))
    for i, phase in enumerate(SOLID_PHASES):
        star = "*" if phase in HEADLINE else " "
        cells = "".join(
            f"  {scores[arm][phase]['mae']:.3f}/"
            f"{scores[arm][phase]['pearson_r']:+.2f}".rjust(18)
            for arm in scores
        )
        print(f"{phase:<5s}{star} {floors_val[i]:.3f} {cells}")
    print("  (cells are per-bulk MAE / pearson r; * = headline phase)")

    print(f"\nheadline mean MAE over {list(HEADLINE)} (tie margin {tie_margin}):")
    for arm, mae in verdict["headline_mean_mae"].items():
        print(f"  {arm:<11s} {mae:.3f}")
    print(f"  route: {verdict['route']}")

    metrics = {
        "checkpoint": str(checkpoint_path),
        "encoder_checkpoint": ckpt.encoder_checkpoint,
        "device": str(device),
        "seed": seed,
        "n_val_bulks": int(len(val_ds.bulk_ids)),
        "n_train_bulks": int(len(train_ds.bulk_ids)),
        "val_draws_median": float(np.median(val_draws)),
        "train_draws_median": float(np.median(train_draws)),
        "optuna": {"n_trials": n_trials, "best_params": best_params},
        "binomial_floor_val": {
            p: float(floors_val[i]) for i, p in enumerate(SOLID_PHASES)
        },
        "scores": scores,
        "feature_importances": importances,
        "verdict": verdict,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    _plot_mae_by_phase(scores, floors_val, tag, output_dir / "mae_by_phase.png")
    _plot_importances(importances, tag, output_dir / "importance_headline.png")
    print(f"metrics -> {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="joint-heads checkpoint (heads_best.pt)",
    )
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna trials for the shared xgb study")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tie-margin", type=float, default=0.02,
                        help="MAE difference below which two arms are a tie")
    args = parser.parse_args()
    run_ceiling_probe(
        checkpoint_path=args.checkpoint.resolve(),
        n_trials=args.n_trials,
        seed=args.seed,
        tie_margin=args.tie_margin,
    )


if __name__ == "__main__":
    main()
