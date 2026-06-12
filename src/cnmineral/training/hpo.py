"""
Optuna HPO sweep + post-sweep production run.

`run_hpo` orchestrates: TPE sampler + median pruner sweep against
the search space in the YAML, a post-sweep production run at
`study.best_params`, the test eval, and the summary JSON.

Test set is touched exactly once, in the post-sweep production
phase -- per-trial test scores would defeat the held-out set's
purpose.  Trials pass `store=None` so per-trial checkpoints never
collide in `checkpoint_dir`.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import torch
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.visualization.matplotlib import (
    plot_intermediate_values,
    plot_optimization_history,
)

from cnmineral.training.checkpoint import CheckpointStore
from cnmineral.training.build_from_config import create_dataset, create_encoder_and_head
from cnmineral.training.reconstruction_mse import evaluate_on_test
from cnmineral.training.ssl_pretrain import run_pretraining, write_run_results
from cnmineral.training.splits import reserve_test


REPO_ROOT = Path(__file__).resolve().parents[3]


def _sample_hyperparameter(
    trial: optuna.trial.Trial, name: str, spec: dict,
):
    """Sample one hyperparameter from a `search_space:` entry via `trial.suggest_*`."""
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False),
        )
    if kind == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"], log=spec.get("log", False),
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(
        f"unknown search_space type {kind!r} for {name!r}; "
        f"expected one of: float, int, categorical"
    )


def run_hpo(
    config_path: Path,
    device: torch.device,
) -> None:
    """Orchestrate an Optuna HPO sweep + a final production run on best params.

    Parameters
    ----------
    config_path
        YAML config.  Expected keys: `study_name`, `storage_path`,
        `output_dir` (summary JSON lands at
        `output_dir/<study_name>/summary.json`), `sampler_seed`,
        `n_trials`, `n_startup_trials`,
        `pruner_warmup_steps`, `hpo_run_epochs`, `base` (locked
        config keys), `search_space` (per-knob spec dict consumed by
        `_sample_hyperparameter`).  `base` and `search_space` are
        disjoint -- a key lives in exactly one section.  The YAML is
        authoritative; there is no CLI override for any of these.
    device
        Where to place models and batches.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    storage_path = Path(cfg["storage_path"])
    if not storage_path.is_absolute():
        storage_path = REPO_ROOT / storage_path
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(storage_path)))

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

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
        direction="minimize",
        load_if_exists=True,
    )

    n_done = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )

    trials_left = max(0, cfg["n_trials"] - n_done)
    print(
        f"resuming study {cfg['study_name']!r}: {n_done} complete, "
        f"running {trials_left} more to reach n_trials={cfg['n_trials']}"
    )

    base_config = dict(cfg["base"])
    base_config["epochs"] = cfg.get("hpo_run_epochs", 3)
    search_space = cfg.get("search_space", {})

    # build the dataset once at study scope: every trial, the
    # production run, and the test eval below all index into this
    # same instance.  `trial_hyperparameters` fills in the
    # search-space dims that `base_config` deliberately leaves out --
    # the two are disjoint, so merge is set union, not override.
    dataset = create_dataset(base_config)
    splits = reserve_test(dataset, base_config)
    print(
        f"split: trainval={len(splits['trainval'])} test={len(splits['test'])}"
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_hyperparameters = {
            name: _sample_hyperparameter(trial, name, spec)
            for name, spec in search_space.items()
        }
        merged = {**base_config, **trial_hyperparameters}
        # Trials NEVER save: their state lives in Optuna's journal,
        # the encoder is discarded, and concurrent trials sharing a
        # checkpoint_dir would clobber each other's files.  Discipline
        # is enforced by passing `store=None` explicitly -- not by
        # popping `checkpoint_dir` out of the config dict.
        result = run_pretraining(
            dataset, splits["trainval"], merged, device,
            trial=trial, store=None,
        )
        return result["val_loss"]

    if trials_left > 0:
        study.optimize(objective, n_trials=trials_left, gc_after_trial=True)

    # post-sweep production run + test eval on the best params.  If no
    # trials completed (all pruned, or n_trials=0), skip.
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    best_trial_value: float | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    final_test_loss: float | None = None
    best_params: dict = {}
    if completed:
        best_trial_value = float(study.best_value)
        best_params = study.best_params

        # full-epoch production config: restore base["epochs"] over
        # the hpo_run_epochs substitution we did at the top.
        production_config = {**base_config, **best_params}
        production_config["epochs"] = cfg["base"].get(
            "epochs", cfg.get("hpo_run_epochs", 3),
        )

        print(
            f"production run with best params: "
            f"epochs={production_config['epochs']} "
            f"best_trial_value={best_trial_value:.4f}"
        )
        # Single CheckpointStore construction for the production run.
        # Filename keys are required whenever checkpoint_dir is set;
        # KeyError surfaces a partial config immediately rather than
        # silently picking a default that overwrites a prior run.
        store = (
            CheckpointStore(
                production_config["checkpoint_dir"],
                best_filename=production_config["checkpoint_best_filename"],
                last_filename=production_config["checkpoint_last_filename"],
            )
            if production_config.get("checkpoint_dir") else None
        )
        prod_result = run_pretraining(
            dataset, splits["trainval"], production_config, device,
            store=store,
        )
        final_train_loss = prod_result["train_loss"]
        final_val_loss = prod_result["val_loss"]

        # Compute test_loss against the BEST-VAL model -- same pattern
        # as run_single.
        if store is not None and store.best_exists():
            best_encoder, best_head = create_encoder_and_head(production_config)
            store.load_best().apply_to(best_encoder, best_head)
            best_encoder.to(device)
            best_head.to(device)
            test_loss_raw = evaluate_on_test(
                best_encoder, best_head,
                dataset, splits["test"], production_config, device,
            )
        else:
            test_loss_raw = evaluate_on_test(
                prod_result["encoder"], prod_result["head"],
                dataset, splits["test"], production_config, device,
            )
        final_test_loss = (
            None if math.isnan(test_loss_raw) else test_loss_raw
        )

        # Stamp test_loss + test_indices onto the store's best, then
        # land the same per-run artifacts a run_single would.
        if store is not None and store.best_exists():
            store.stamp_test_results_on_best(
                final_test_loss, list(splits["test"]),
            )
            write_run_results(prod_result, store)

    summary = {
        "study_name": cfg["study_name"],
        "storage_path": str(storage_path),
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
        "final": (
            {
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "test_loss": final_test_loss,
            }
            if completed else None
        ),
        "device": str(device),
        "config": cfg,
    }

    summary_path = output_dir / cfg["study_name"] / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"summary -> {summary_path}")

    for plot_fn, filename in (
        (plot_optimization_history, "hpo_history.png"),
        (plot_intermediate_values, "hpo_trial_curves.png"),
    ):
        ax = plot_fn(study)
        fig_path = summary_path.parent / filename
        ax.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"figure -> {fig_path}")

    if completed:
        print(f"best trial val:   {best_trial_value:.4f}")
        print(f"best params:      {best_params}")
        test_str = (
            f"{final_test_loss:.4f}"
            if final_test_loss is not None else "n/a"
        )
        print(
            f"final production: train={final_train_loss:.4f} "
            f"val={final_val_loss:.4f} test={test_str}"
        )
