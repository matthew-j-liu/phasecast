"""Optuna HPO over the log-space imputation head on a frozen encoder.

Single-stage TPE sweep of the head and its optimization (encoder frozen,
head-only) maximizing median trace leave-one-out log-R2 on a held-out val
split -- the same instrument `cnmineral.eval.imputation_logspace` ships on,
scored on val so the test split stays untouched until the final artifact.
Mirrors the probe-HPO scaffolding (`JournalStorage`, `load_if_exists` resume,
TPE sampler).  No pruner: the objective is an end-of-trial LOO computation with
no cheap per-epoch analogue, and each head trains in seconds on the frozen
latent, so there is little to prune.  The sweep emits `best_params`; the final
production head ships from `train_imputation_head` on the updated config, not
from here.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import yaml
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.visualization.matplotlib import plot_optimization_history

from cnmineral.eval.imputation_logspace import (
    _N_MAJORS,
    _summarize,
    score_head_per_element,
)
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.training import (
    checkpoint_tag,
    create_dataset,
    pick_device,
    reserve_test,
)
from cnmineral.training.hpo import _sample_hyperparameter
from cnmineral.training.imputation_head import _train_val_split, train_head_on_split


REPO_ROOT = Path(__file__).resolve().parents[3]


def _median_major_log_r2(per_element: dict) -> float:
    """Median log-R2 over the major positions, to watch majors not regressing."""
    vals = [
        rec["log_r2"] for rec in per_element.values()
        if rec["token_pos"] < _N_MAJORS
    ]
    return float(np.median(vals)) if vals else float("nan")


def run_hpo(
    config_path: Path,
    *,
    device: torch.device | None = None,
) -> dict:
    """Sweep the head's HPs against a frozen encoder; write + return the summary.

    Expected config keys: `encoder_checkpoint`, `study_name`, `storage_path`
    (a `{encoder_tag}` template), `output_dir`, `sampler_seed`, `n_trials`,
    `n_startup_trials`, `base` (locked knobs incl. `batch_size`, `epochs`
    ceiling, `early_stop_patience`, `val_split`, `standardize`, `seed`),
    `search_space` (per-knob spec for `_sample_hyperparameter`).  Optional
    `score_batch_size` (val LOO scoring) and `baseline` (a fixed HP dict scored
    once for an untuned reference point).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    encoder_ckpt_path = Path(cfg["encoder_checkpoint"])
    n_trials = cfg["n_trials"]
    device = device or pick_device()
    tag = checkpoint_tag(encoder_ckpt_path)

    storage_path = REPO_ROOT / cfg["storage_path"].format(encoder_tag=tag)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(storage_path)))
    output_dir = REPO_ROOT / cfg["output_dir"] / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device:           {device}")
    print(f"encoder:          {encoder_ckpt_path}")
    print(f"hpo config:       {config_path}")
    print(f"output:           {output_dir}")

    encoder, enc_ckpt = load_encoder_from_checkpoint(encoder_ckpt_path)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    dataset = create_dataset(enc_ckpt.config)
    trainval = reserve_test(dataset, enc_ckpt.config)["trainval"]
    base_config = dict(cfg["base"])
    seed = int(base_config.get("seed", enc_ckpt.config.get("seed", 0)))
    val_split = base_config.get("val_split", 0.20)
    train_idx, val_idx = _train_val_split(dataset, trainval, val_split, seed)
    val_arr = np.asarray(val_idx, dtype=np.int64)
    score_batch_size = cfg.get("score_batch_size", 512)
    print(
        f"  dataset n={len(dataset)} trainval={len(trainval)} "
        f"train={len(train_idx)} val={len(val_idx)}"
    )

    # untuned reference: the design-ablation fixed config, scored on the same
    # val split, so the journal can report tuned-vs-untuned on this encoder.
    baseline: dict | None = None
    if cfg.get("baseline"):
        baseline_config = {**base_config, **cfg["baseline"]}
        torch.manual_seed(seed)
        b_head, _mon, _tr, _vl = train_head_on_split(
            encoder, enc_ckpt.config, dataset,
            train_idx, val_idx, baseline_config, device, verbose=False,
        )
        b_per_element, _ = score_head_per_element(
            encoder, b_head, dataset, val_arr,
            device=device, batch_size=score_batch_size,
        )
        b_summary = _summarize(b_per_element)
        baseline = {
            "params": dict(cfg["baseline"]),
            "median_trace_log_r2": b_summary["median_head_log_r2"],
            "median_major_log_r2": _median_major_log_r2(b_per_element),
        }
        print(
            f"  baseline (untuned) val median trace log-R2="
            f"{baseline['median_trace_log_r2']:.4f}"
        )

    sampler = TPESampler(
        seed=cfg["sampler_seed"], n_startup_trials=cfg["n_startup_trials"],
    )
    study = optuna.create_study(
        study_name=cfg["study_name"],
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    n_done = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    trials_left = max(0, n_trials - n_done)
    print(
        f"resuming study {cfg['study_name']!r}: {n_done} complete, "
        f"running {trials_left} more to reach n_trials={n_trials}"
    )

    search_space = cfg.get("search_space", {})

    def objective(trial: optuna.trial.Trial) -> float:
        trial_hyperparameters = {
            name: _sample_hyperparameter(trial, name, spec)
            for name, spec in search_space.items()
        }
        merged = {**base_config, **trial_hyperparameters}
        # Same seed every trial: the only thing that differs across trials is
        # the sampled HPs, not the masking / shuffle draw.
        torch.manual_seed(seed)
        head, _monitor, _train_loss, _val_loss = train_head_on_split(
            encoder, enc_ckpt.config, dataset,
            train_idx, val_idx, merged, device, verbose=False,
        )
        per_element, _ = score_head_per_element(
            encoder, head, dataset, val_arr,
            device=device, batch_size=score_batch_size,
        )
        summary = _summarize(per_element)
        trial.set_user_attr("median_major_log_r2", _median_major_log_r2(per_element))
        trial.set_user_attr("n_traces", summary["n_traces"])
        return summary["median_head_log_r2"]

    if trials_left > 0:
        study.optimize(objective, n_trials=trials_left, gc_after_trial=True)

    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    best_params: dict = {}
    best_value: float | None = None
    best_major: float | None = None
    if completed:
        best_value = float(study.best_value)
        best_params = dict(study.best_params)
        best_major = study.best_trial.user_attrs.get("median_major_log_r2")
        print(f"\nbest val median trace log-R2={best_value:.4f}")
        print(f"best params: {best_params}")

    summary = {
        "encoder_checkpoint": str(encoder_ckpt_path),
        "encoder_tag": tag,
        "hpo_config": str(config_path),
        "storage_path": str(storage_path),
        "study_name": cfg["study_name"],
        "device": str(device),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "objective": "median_trace_log_r2_val",
        "n_trials_complete": len(completed),
        "best_value": best_value,
        "best_median_major_log_r2": best_major,
        "best_params": best_params,
        "baseline": baseline,
        "config": cfg,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsummary -> {summary_path}")

    # optuna's matplotlib module flags itself experimental on every call;
    # stable across 4.x, noise at our call site.  Empty studies also trip the
    # no-legend-artists warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        warnings.filterwarnings("ignore", message="No artists with labels found")
        ax = plot_optimization_history(study)
    history_path = output_dir / "hpo_history.png"
    ax.figure.savefig(history_path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"figure -> {history_path}")
    return summary
