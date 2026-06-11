"""Optuna probe-HPO sweep against a fixed encoder checkpoint.

`run_hpo()` orchestrates: latent extraction (once at study scope, since
all trials see the same latent), the fixed-protocol LR baseline (run
once), the Optuna sweep over MLP probe HPs, and the post-sweep
production MLP at `study.best_params`.  Writes a per-encoder
`summary.json` the `cnmineral.eval.probe.calibration` module then
aggregates across encoders.

The encoder is fixed throughout: probe HPO measures how well a
probe extracts the encoder's frozen representation.  It never
updates the encoder weights.
"""

import json
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.visualization.matplotlib import plot_optimization_history

from cnmineral.eval.latent import extract_latents, load_encoder_from_checkpoint
from cnmineral.eval.probe.primitives import train_logreg_probe, train_mlp_probe
from cnmineral.eval.supergroup import map_qap_array
from cnmineral.eval.training_curves import plot_run_loss
from cnmineral.training import create_dataset, pick_device, reserve_test
from cnmineral.training.checkpoint import checkpoint_tag
from cnmineral.training.hpo import _sample_hyperparameter


REPO_ROOT = Path(__file__).resolve().parents[4]


def _labels_for_target(dataset, target: str) -> np.ndarray:
    """Pull the per-row label array for the configured probe target."""
    if not hasattr(dataset, "qap_names"):
        raise SystemExit(
            "dataset has no qap_names attribute; probe HPO needs labels"
        )
    qap = np.asarray(dataset.qap_names)
    if target == "qap_name":
        return qap
    if target == "supergroup":
        return map_qap_array(qap)
    raise ValueError(
        f"unknown probe_target {target!r}; expected qap_name or supergroup"
    )


def run_hpo(
    checkpoint_path: Path,
    hpo_config_path: Path,
    *,
    device: torch.device | None = None,
) -> dict:
    with open(hpo_config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve required YAML keys before any I/O so a missing key
    # surfaces as a loud KeyError, not a half-written run.
    output_dir = REPO_ROOT / cfg["output_dir"]
    n_trials = cfg["n_trials"]

    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    print(f"device: {device}")
    print(f"checkpoint:       {checkpoint_path}")
    print(f"hpo config:       {hpo_config_path}")
    print(f"output:           {output_dir}")

    tag = checkpoint_tag(checkpoint_path)
    storage_path_template = cfg["storage_path"]
    storage_path = REPO_ROOT / storage_path_template.format(encoder_tag=tag)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(storage_path)))

    print("loading encoder (pretrain config comes from ckpt.config)...")
    encoder, ckpt = load_encoder_from_checkpoint(checkpoint_path)
    pretrain_config = ckpt.config

    print("building dataset + reproducing test split...")
    dataset = create_dataset(pretrain_config)
    splits = reserve_test(dataset, pretrain_config)
    n_train = len(splits["trainval"])
    n_test = len(splits["test"])
    print(f"  dataset n={len(dataset)} trainval={n_train} test={n_test}")

    print("extracting latents (once at study scope)...")
    t0 = time()
    latents = extract_latents(encoder, dataset, device=device).numpy()
    print(f"  latents {latents.shape} in {time() - t0:.1f}s")

    target = cfg.get("probe_target", "qap_name")
    labels_full = _labels_for_target(dataset, target)
    train_idx = np.asarray(splits["trainval"], dtype=np.int64)
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    X_train = latents[train_idx]
    X_test = latents[test_idx]
    y_train = labels_full[train_idx]
    y_test = labels_full[test_idx]
    print(
        f"  probe target={target}  n_classes={len(np.unique(labels_full))}"
    )

    # fixed-protocol LR baseline, run once, recorded alongside the
    # tuned MLP result so the calibration report can cross-reference.
    print("fixed-protocol LR baseline (no tuning)...")
    t0 = time()
    lr_baseline = train_logreg_probe(X_train, y_train, X_test, y_test)
    print(
        f"  lr_baseline test_acc={lr_baseline.test_accuracy:.3f} "
        f"in {time() - t0:.1f}s"
    )

    sampler = TPESampler(
        seed=cfg["sampler_seed"],
        n_startup_trials=cfg["n_startup_trials"],
    )
    pruner = MedianPruner(n_warmup_steps=cfg["pruner_warmup_steps"])
    study = optuna.create_study(
        study_name=cfg["study_name"],
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    n_done = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    trials_left = max(0, n_trials - n_done)
    print(
        f"resuming study {cfg['study_name']!r}: {n_done} complete, "
        f"running {trials_left} more to reach n_trials={n_trials}"
    )

    base_config = dict(cfg["base"])
    search_space = cfg.get("search_space", {})

    def objective(trial: optuna.trial.Trial) -> float:
        trial_hyperparameters = {
            name: _sample_hyperparameter(trial, name, spec)
            for name, spec in search_space.items()
        }
        merged = {**base_config, **trial_hyperparameters}
        metrics = train_mlp_probe(
            X_train, y_train, X_test, y_test,
            device=device,
            **merged,
        )
        return float(metrics.test_accuracy)

    if trials_left > 0:
        study.optimize(objective, n_trials=trials_left, gc_after_trial=True)

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    final_mlp_dict: dict | None = None
    best_params: dict = {}
    best_trial_value: float | None = None
    if completed:
        best_trial_value = float(study.best_value)
        best_params = dict(study.best_params)
        print(
            f"\nproduction MLP at best params (val={best_trial_value:.3f})"
        )
        production_config = {**base_config, **best_params}
        final_mlp = train_mlp_probe(
            X_train, y_train, X_test, y_test,
            device=device,
            **production_config,
        )
        final_mlp_dict = final_mlp.to_dict()
        print(
            f"  final_mlp test_acc={final_mlp.test_accuracy:.3f}  "
            f"baseline_lr test_acc={lr_baseline.test_accuracy:.3f}"
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "encoder_tag": tag,
        "hpo_config": str(hpo_config_path),
        "storage_path": str(storage_path),
        "study_name": cfg["study_name"],
        "probe_target": target,
        "n_classes": int(len(np.unique(labels_full))),
        "device": str(device),
        "n_trials_complete": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ),
        "n_trials_pruned": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ),
        "best_trial_value": best_trial_value,
        "best_params": best_params,
        "final_mlp": final_mlp_dict,
        "baseline_lr": lr_baseline.to_dict(),
        "checkpoint_metadata": {
            "epoch": ckpt.epoch,
            "val_loss": ckpt.val_loss,
            "test_loss": ckpt.test_loss,
        },
        "config": cfg,
    }
    summary_path = output_dir / tag / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsummary -> {summary_path}")

    # No hpo_trial_curves.png here: probe trials never call
    # trial.report, so the study has no intermediate values to plot.
    ax = plot_optimization_history(study)
    history_path = summary_path.parent / "hpo_history.png"
    ax.figure.savefig(history_path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"figure -> {history_path}")

    if final_mlp_dict is not None:
        fig, ax = plt.subplots()
        plot_run_loss(
            final_mlp_dict["monitor"], ax=ax,
            title=f"production MLP ({target}) -- {tag}",
            ylabel="cross-entropy",
        )
        curve_path = summary_path.parent / "loss_curve.png"
        fig.savefig(curve_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"figure -> {curve_path}")
    return summary
