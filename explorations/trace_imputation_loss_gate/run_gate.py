"""
Trace-imputation loss-gate exploration: decides which layer to fix.

The frozen pretrain head imputes majors well but collapses on traces (a
loss-balance artifact, not an architecture limit -- see docs/JOURNAL.md
2026-06-15).  This gate runs a controlled pair per element on the Gard test
split, both scored in log space: 1a input-space XGBoost (predict each element
from the other measured elements, the ceiling) and 1b XGBoost on the frozen
latent with the target masked before encoding (leakage-guarded).  Only the
representation differs between the two arms.  Gate on 1b vs 1a over the traces:
if 1b stays near the 1a ceiling the frozen latent carries the trace signal (fix
= log-space read-out head on the frozen latent, encoder untouched); if 1b
collapses far below 1a the latent is trace-blind (fix = trace-rebalanced
re-pretrain).  Read-only -- nothing here trains or mutates the encoder.

Scoring/back-transform/plot helpers are reused from `cnmineral.eval.imputation`;
the canonical verdict lands in docs/JOURNAL.md (and an ADR if a re-pretrain is
indicated).
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
from torch.utils.data import DataLoader, Dataset, Subset
from xgboost import XGBRegressor

from cnmineral.data.tokens import MAJORS, MASK_TOKEN_ID, VOCAB
from cnmineral.eval.imputation import (
    _back_transform,
    _plot_group,
    _regression_metrics,
)
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.eval.supergroup import map_qap_array
from cnmineral.models.encoder import Encoder
from cnmineral.training import checkpoint_tag, create_dataset, pick_device, reserve_test

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

_N_MAJORS = len(MAJORS)
_TRACE_POSITIONS = list(range(_N_MAJORS, len(VOCAB)))


def _collect_frac(
    dataset: Dataset, indices: np.ndarray, *, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the `[len(indices), 22]` frac matrix and a populated mask.

    Populated is `src != 0`, the only padding signal (tokens convention C).
    """
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=batch_size,
        shuffle=False, num_workers=0,
    )
    fracs, srcs = [], []
    for batch in loader:
        fracs.append(batch["frac"].numpy())
        srcs.append(batch["src"].numpy())
    frac = np.concatenate(fracs, axis=0)
    src = np.concatenate(srcs, axis=0)
    return frac, src != 0


@torch.no_grad()
def _encode_with_masked_token(
    encoder: Encoder,
    dataset: Dataset,
    indices: np.ndarray,
    token_pos: int,
    *,
    device: torch.device | str,
    batch_size: int,
) -> np.ndarray:
    """Encode `Subset(dataset, indices)` with `token_pos` masked, return `[N, d_model]`.

    Same masking as `_impute_token` (`src=MASK_TOKEN_ID, frac=0`) but stops at
    the encoder output.  Masking is what keeps 1b honest: the latent is built
    from every token *except* the one being imputed.
    """
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=batch_size,
        shuffle=False, num_workers=0,
    )
    chunks = []
    for batch in loader:
        src = batch["src"].clone().to(device)
        frac = batch["frac"].clone().to(device)
        src[:, token_pos] = MASK_TOKEN_ID
        frac[:, token_pos] = 0.0
        chunks.append(encoder(src=src, frac=frac).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _log_feature_matrix(frac: np.ndarray, populated: np.ndarray) -> np.ndarray:
    """`log10(frac)` where populated, NaN where absent (XGBoost native missing)."""
    out = np.full(frac.shape, np.nan, dtype=np.float64)
    out[populated] = np.log10(frac[populated])
    return out


def _fit_predict_xgb(
    params: dict,
    x_fit: np.ndarray, y_fit: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_pred: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    model = XGBRegressor(
        n_estimators=2000,
        early_stopping_rounds=50,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=seed,
        **params,
    )
    model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
    return model.predict(x_pred)


def _xgb_val_mse(
    params: dict,
    x_fit: np.ndarray, y_fit: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    *,
    seed: int,
) -> float:
    pred = _fit_predict_xgb(params, x_fit, y_fit, x_val, y_val, x_val, seed=seed)
    return float(np.mean((pred - y_val) ** 2))


def _suggest_params(trial: optuna.trial.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def _tune_shared(
    trace_tasks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    n_trials: int,
    seed: int,
) -> dict:
    """One light study whose objective is mean val-MSE over the 13 trace tasks.

    `trace_tasks` is a list of `(x_fit, y_fit, x_val, y_val)` already built for
    the feature set being tuned.  The single best param set is reused for every
    element's final fit.
    """
    if not trace_tasks:
        raise ValueError(
            "no trace tasks to tune on -- the corpus has no trace coverage, so "
            "the 1b-vs-1a trace gate is undefined"
        )

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(trial)
        return float(np.mean([
            _xgb_val_mse(params, xf, yf, xv, yv, seed=seed)
            for xf, yf, xv, yv in trace_tasks
        ]))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def _plot_r2_by_element(per_element: dict, tag: str, out_path: Path) -> None:
    """Grouped bar of log-space R2 (1a / 1b), majors and traces split."""
    syms = [s for s, _z in VOCAB if s in per_element]
    majors = [s for s in syms if per_element[s]["token_pos"] < _N_MAJORS]
    traces = [s for s in syms if per_element[s]["token_pos"] >= _N_MAJORS]
    runs = [("run_1a", "1a input XGB"), ("run_1b", "1b latent XGB")]
    colors = {"run_1a": "#1b9e77", "run_1b": "#7570b3"}

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(syms) * 0.5), 8))
    for ax, group, gtitle in ((axes[0], majors, "majors"), (axes[1], traces, "traces")):
        x = np.arange(len(group))
        w = 0.4
        for k, (run, label) in enumerate(runs):
            vals = [per_element[s][run]["r2"] for s in group]
            ax.bar(x + (k - 0.5) * w, vals, w, label=label, color=colors[run])
        ax.set_xticks(x)
        ax.set_xticklabels(group)
        ax.set_ylabel("log-space R2")
        ax.set_title(gtitle)
        ax.set_ylim(0.0, 1.0)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"imputation diagnostic R2 (log space) -- {tag}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_imputation_diagnostic(
    checkpoint_path: Path,
    *,
    device: torch.device | None = None,
    n_trials: int = 40,
    study_subsample: int | None = None,
    batch_size: int = 512,
) -> dict:
    """Trace-imputation gate: 1a input XGB vs 1b latent XGB, scored in log space.

    `study_subsample` caps the per-split rows used during Optuna tuning; `None`
    (the default) tunes on all fit/val rows.  Final per-element fits always use
    the full splits.  Writes per-element log-space metrics, the gate verdict, an
    R2 bar chart, and trace parity grids to `results/<tag>/`; returns the
    metrics dict.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tag = checkpoint_tag(checkpoint_path)
    output_dir = RESULTS_ROOT / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    print(f"device:     {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output:     {output_dir}")

    encoder, ckpt = load_encoder_from_checkpoint(checkpoint_path)
    encoder.to(device)
    seed = int(ckpt.config.get("seed", 0))

    dataset = create_dataset(ckpt.config)
    splits = reserve_test(dataset, ckpt.config)
    trainval_idx = np.asarray(splits["trainval"], dtype=np.int64)
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    fit_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.2, random_state=seed,
    )
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    sg_all = map_qap_array(dataset.qap_names)
    print(f"dataset n={len(dataset)} fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)}")

    frac_fit, pop_fit = _collect_frac(dataset, fit_idx, batch_size=batch_size)
    frac_val, pop_val = _collect_frac(dataset, val_idx, batch_size=batch_size)
    frac_test, pop_test = _collect_frac(dataset, test_idx, batch_size=batch_size)
    xlog_fit = _log_feature_matrix(frac_fit, pop_fit)
    xlog_val = _log_feature_matrix(frac_val, pop_val)
    xlog_test = _log_feature_matrix(frac_test, pop_test)

    if study_subsample is None:
        sf = np.arange(len(fit_idx))
        sv = np.arange(len(val_idx))
    else:
        rng = np.random.default_rng(seed)
        sf = rng.choice(len(fit_idx), size=min(study_subsample, len(fit_idx)), replace=False)
        sv = rng.choice(len(val_idx), size=min(study_subsample, len(val_idx)), replace=False)
    print(f"tuning rows: fit={len(sf)} val={len(sv)}"
          f"{' (all)' if study_subsample is None else f' (subsample={study_subsample})'}")

    print("tuning 1a (input-space) ...")
    tasks_1a = []
    for e in _TRACE_POSITIONS:
        rf = sf[pop_fit[sf, e]]
        rv = sv[pop_val[sv, e]]
        if rf.size == 0 or rv.size == 0:
            continue
        tasks_1a.append((
            np.delete(xlog_fit, e, axis=1)[rf], np.log10(frac_fit[rf, e]),
            np.delete(xlog_val, e, axis=1)[rv], np.log10(frac_val[rv, e]),
        ))
    best_1a = _tune_shared(tasks_1a, n_trials=n_trials, seed=seed)
    print(f"  best 1a: {best_1a}")

    print("tuning 1b (frozen latent) ...")
    tasks_1b = []
    for e in _TRACE_POSITIONS:
        rf_mask = pop_fit[sf, e]
        rv_mask = pop_val[sv, e]
        if rf_mask.sum() == 0 or rv_mask.sum() == 0:
            continue
        h_fit = _encode_with_masked_token(
            encoder, dataset, fit_idx[sf], e, device=device, batch_size=batch_size)
        h_val = _encode_with_masked_token(
            encoder, dataset, val_idx[sv], e, device=device, batch_size=batch_size)
        tasks_1b.append((
            h_fit[rf_mask], np.log10(frac_fit[sf][rf_mask, e]),
            h_val[rv_mask], np.log10(frac_val[sv][rv_mask, e]),
        ))
    best_1b = _tune_shared(tasks_1b, n_trials=n_trials, seed=seed)
    print(f"  best 1b: {best_1b}")

    per_element: dict[str, dict] = {}
    trace_panels_1a: list[dict] = []
    trace_panels_1b: list[dict] = []
    for token_pos, (sym, _z) in enumerate(VOCAB):
        rows_fit = pop_fit[:, token_pos]
        rows_val = pop_val[:, token_pos]
        rows_test = pop_test[:, token_pos]
        if rows_fit.sum() == 0 or rows_test.sum() == 0:
            continue
        y_fit = np.log10(frac_fit[rows_fit, token_pos])
        y_val = np.log10(frac_val[rows_val, token_pos])
        y_test = np.log10(frac_test[rows_test, token_pos])

        pred_1a = _fit_predict_xgb(
            best_1a,
            np.delete(xlog_fit, token_pos, axis=1)[rows_fit], y_fit,
            np.delete(xlog_val, token_pos, axis=1)[rows_val], y_val,
            np.delete(xlog_test, token_pos, axis=1)[rows_test],
            seed=seed,
        )
        m_1a = _regression_metrics(y_test, pred_1a)

        h_fit = _encode_with_masked_token(
            encoder, dataset, fit_idx, token_pos, device=device, batch_size=batch_size)
        h_val = _encode_with_masked_token(
            encoder, dataset, val_idx, token_pos, device=device, batch_size=batch_size)
        h_test = _encode_with_masked_token(
            encoder, dataset, test_idx, token_pos, device=device, batch_size=batch_size)
        pred_1b = _fit_predict_xgb(
            best_1b, h_fit[rows_fit], y_fit, h_val[rows_val], y_val, h_test[rows_test],
            seed=seed,
        )
        m_1b = _regression_metrics(y_test, pred_1b)

        _bt_true, label = _back_transform(token_pos, frac_test[rows_test, token_pos])
        per_element[sym] = {
            "token_pos": token_pos,
            "unit": label,
            "n_train": int(rows_fit.sum()),
            "n_test": int(rows_test.sum()),
            "run_1a": m_1a,
            "run_1b": m_1b,
        }
        print(f"  {sym:<3s} n={int(rows_test.sum()):<6d} "
              f"R2 1a={m_1a['r2']:.3f} 1b={m_1b['r2']:.3f}")

        if token_pos in _TRACE_POSITIONS:
            kept_ds = test_idx[rows_test]
            sg = sg_all[kept_ds]
            bt_true, _ = _back_transform(token_pos, frac_test[rows_test, token_pos])
            bt_1a, _ = _back_transform(token_pos, np.power(10.0, pred_1a))
            bt_1b, _ = _back_transform(token_pos, np.power(10.0, pred_1b))
            trace_panels_1a.append({
                "label": label, "n": int(rows_test.sum()), "true_bt": bt_true,
                "pred_bt": bt_1a, "supergroup": sg, "metrics_bt": m_1a,
            })
            trace_panels_1b.append({
                "label": label, "n": int(rows_test.sum()), "true_bt": bt_true,
                "pred_bt": bt_1b, "supergroup": sg, "metrics_bt": m_1b,
            })

    gate = _build_gate(per_element)
    _plot_r2_by_element(per_element, tag, output_dir / "r2_by_element.png")
    if trace_panels_1a:
        _plot_group(trace_panels_1a, f"1a input-XGB traces (log) -- {tag}",
                    output_dir / "trace_parity_1a.png", log_scale=True)
        _plot_group(trace_panels_1b, f"1b latent-XGB traces (log) -- {tag}",
                    output_dir / "trace_parity_1b.png", log_scale=True)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "space": "log10",
        "checkpoint_epoch": ckpt.epoch,
        "n_test": int(len(test_idx)),
        "latent_dim": int(ckpt.config["d_model"]),
        "optuna": {"n_trials": n_trials, "best_1a": best_1a, "best_1b": best_1b},
        "per_element": per_element,
        "gate": gate,
    }
    metrics_path = output_dir / "diagnostic_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    med_a, med_b = gate["median_r2_1a"], gate["median_r2_1b"]
    frac = med_b / med_a if med_a > 0 else float("nan")
    print(f"\ntrace imputation gate -- {len(gate['per_trace'])} traces, log-space R2:")
    print(f"  input-element XGBoost ceiling: median {med_a:.3f}")
    print(f"  frozen-latent XGBoost: median {med_b:.3f} ({frac:.0%} of the ceiling)")
    print("  criterion: the latent carries the trace signal if its median R2 is within")
    print("    20% of the input ceiling (>= 0.80x); otherwise the latent is blind.")
    if gate["verdict"] == "latent_carries_signal":
        print("  conclusion: the frozen latent retains the trace signal (latent XGBoost near")
        print("    the input ceiling). the imputation failure is the pretrain head's")
        print("    linear-frac loss, not the encoder.")
        print("  action: train a log-space read-out head on the frozen latent; do not")
        print("    re-pretrain the encoder.")
    else:
        print("  conclusion: the frozen latent has lost the trace signal (latent XGBoost far")
        print("    below the input ceiling). a read-out head cannot recover it.")
        print("  action: re-pretrain the encoder with a trace-rebalanced loss.")
    print(f"metrics -> {metrics_path}")
    return metrics


def _build_gate(per_element: dict) -> dict:
    """Compare 1b vs 1a over the traces; advisory verdict a human confirms.

    `latent_carries_signal` => the frozen latent encodes the traces (median 1b
    R2 within ~20% of the 1a ceiling, ceiling positive): a log-space read-out
    head on the frozen latent suffices, encoder untouched.  `latent_blind` =>
    the latent has lost the trace signal: a trace-rebalanced re-pretrain is
    needed.  The numbers are the decision; the verdict is a heuristic summary.
    """
    per_trace = {}
    r2_1a, r2_1b = [], []
    for sym, rec in per_element.items():
        if rec["token_pos"] < _N_MAJORS:
            continue
        a, b = rec["run_1a"]["r2"], rec["run_1b"]["r2"]
        per_trace[sym] = {"r2_1a": a, "r2_1b": b, "delta": b - a}
        r2_1a.append(a)
        r2_1b.append(b)
    med_a = float(np.median(r2_1a)) if r2_1a else float("nan")
    med_b = float(np.median(r2_1b)) if r2_1b else float("nan")
    verdict = (
        "latent_carries_signal"
        if (med_a > 0 and med_b >= 0.8 * med_a)
        else "latent_blind"
    )
    return {
        "per_trace": per_trace,
        "mean_r2_1a": float(np.mean(r2_1a)) if r2_1a else float("nan"),
        "mean_r2_1b": float(np.mean(r2_1b)) if r2_1b else float("nan"),
        "median_r2_1a": med_a,
        "median_r2_1b": med_b,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="encoder+head checkpoint produced by scripts/training/pretrain.py",
    )
    parser.add_argument(
        "--n-trials", type=int, default=40,
        help="Optuna trials per shared study (one for 1a, one for 1b)",
    )
    parser.add_argument(
        "--study-subsample", type=int, default=None,
        help="if set, cap per-split rows used during Optuna tuning; "
             "default tunes on all rows (final fits always use all)",
    )
    args = parser.parse_args()
    run_imputation_diagnostic(
        checkpoint_path=args.checkpoint.resolve(),
        n_trials=args.n_trials,
        study_subsample=args.study_subsample,
    )


if __name__ == "__main__":
    main()
