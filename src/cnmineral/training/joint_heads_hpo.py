"""Optuna HPO over the three joint heads on a frozen encoder.

Single-stage TPE sweep (ADR 006) of the shared training environment plus one
head's architecture, ranked on that head's raw selection metric at its own
best epoch (`selection_series`): `objective: mode` (assemblage NLL,
unweighted bce + Dirichlet) for the main sweep, `objective: comp | cond`
for the conditional architecture mini-sweeps.  Per-head best selection in
the trainer means trials never arbitrate stopping; every trial records all
three heads' bests, best epochs, and the predicted-modes closure as user
attrs -- the scatter that audits the single-objective choice.  Trial 0 is
enqueued from `baseline:` (the untuned reference, fully comparable and a
TPE seed point).  The winner is read post-hoc by `tiebreak_best`, not
`study.best_params`: mode ties within `tie_band` nats break to comp then
cond, and mini-sweep winners must hold mode within `tie_band` of
`mode_reference`.  JournalStorage + `load_if_exists` resume; no pruner --
pruning would cut cond's tail and the user attrs to save minutes on
already-cheap trials.  Trials never save checkpoints; production ships from
`train_joint_heads` on the updated heads config.
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
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)

from cnmineral.eval.comp_head import predicted_modes_closure
from cnmineral.eval.joint_heads import val_dataset_and_latents
from cnmineral.training import pick_device
from cnmineral.training.hpo import _sample_hyperparameter
from cnmineral.training.joint_heads import (
    HEAD_NAMES,
    selection_series,
    train_joint_heads,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def tiebreak_best(
    trials: list[optuna.trial.FrozenTrial],
    objective: str,
    tie_band: float,
    mode_reference: float | None = None,
) -> optuna.trial.FrozenTrial | None:
    """The locked winner read-out over completed trials.

    `objective: mode`: trials within `tie_band` nats of the incumbent are
    ties, broken by comp NLL then cond NLL.  `objective: comp | cond`
    (mini-sweeps): trials are admissible only if their mode best stays
    within `tie_band` of `mode_reference` (the main-sweep winner); the best
    admissible trial by the objective wins.  Returns None when no trial is
    admissible -- itself a finding (the head's architecture moves mode
    through the closure coupling).
    """
    complete = [
        t for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        return None
    if objective == "mode":
        leader = min(t.value for t in complete)
        tied = [t for t in complete if t.value <= leader + tie_band]
        return min(
            tied,
            key=lambda t: (t.user_attrs["comp_best"], t.user_attrs["cond_best"]),
        )
    pool = [
        t for t in complete
        if mode_reference is None
        or t.user_attrs["mode_best"] <= mode_reference + tie_band
    ]
    if not pool:
        return None
    return min(pool, key=lambda t: t.value)


def _scatter_mode_comp(study: optuna.Study, out_path: Path) -> None:
    """Mode-vs-comp bests across trials: the single-objective audit plot."""
    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        return
    mode = [t.user_attrs["mode_best"] for t in complete]
    comp = [t.user_attrs["comp_best"] for t in complete]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(mode, comp, s=18, color="tab:blue")
    zero = next((t for t in complete if t.number == 0), None)
    if zero is not None:
        ax.scatter(
            zero.user_attrs["mode_best"], zero.user_attrs["comp_best"],
            s=60, color="tab:orange", label="trial 0",
        )
        ax.legend()
    ax.set_xlabel("mode best (raw assemblage NLL)")
    ax.set_ylabel("comp best (raw NLL)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out_path}")


def run_hpo(
    config_path: Path,
    *,
    device: torch.device | None = None,
) -> dict:
    """Sweep the joint heads per the YAML; write + return the summary.

    Expected config keys: `study_name`, `storage_path`, `output_dir`,
    `objective` (mode | comp | cond), `tie_band`, `sampler_seed`,
    `n_trials`, `n_startup_trials`, `base` (locked `train_joint_heads`
    keys), `search_space` (per-knob spec for `_sample_hyperparameter`;
    disjoint from `base`), `baseline` (search-space values enqueued as
    trial 0).  Mini-sweeps additionally set `mode_reference` (the
    main-sweep winner's mode best).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    objective_head = cfg["objective"]
    if objective_head not in HEAD_NAMES:
        raise ValueError(
            f"objective must be one of {HEAD_NAMES}, got {objective_head!r}"
        )
    tie_band = float(cfg["tie_band"])
    mode_reference = cfg.get("mode_reference")
    device = device or pick_device()

    storage_path = REPO_ROOT / cfg["storage_path"]
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(storage_path)))
    output_dir = REPO_ROOT / cfg["output_dir"] / cfg["study_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device:     {device}")
    print(f"hpo config: {config_path}")
    print(f"objective:  {objective_head}")
    print(f"output:     {output_dir}")

    base_config = dict(cfg["base"])
    search_space = cfg["search_space"]

    sampler = TPESampler(
        seed=cfg["sampler_seed"], n_startup_trials=cfg["n_startup_trials"],
    )
    study = optuna.create_study(
        study_name=cfg["study_name"],
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    if not study.trials and cfg.get("baseline"):
        study.enqueue_trial(dict(cfg["baseline"]))

    n_done = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    trials_left = max(0, cfg["n_trials"] - n_done)
    print(
        f"resuming study {cfg['study_name']!r}: {n_done} complete, "
        f"running {trials_left} more toward the n_trials={cfg['n_trials']} "
        f"ceiling (stop anytime; the journal resumes)"
    )

    # val dataset + latents for the closure attr, built once from the first
    # trial's stamped label_config (identical across trials -- the search
    # space never touches label keys).
    closure_cache: dict = {}

    def objective(trial: optuna.trial.Trial) -> float:
        hp = {
            name: _sample_hyperparameter(trial, name, spec)
            for name, spec in search_space.items()
        }
        merged = {**base_config, **hp}
        ckpt = train_joint_heads(merged, device)
        sel = selection_series(ckpt.monitor)
        bests = {name: float(min(series)) for name, series in sel.items()}
        for name in HEAD_NAMES:
            trial.set_user_attr(f"{name}_best", bests[name])
            trial.set_user_attr(
                f"{name}_best_epoch", 1 + int(np.argmin(sel[name]))
            )
        if "data" not in closure_cache:
            closure_cache["data"] = val_dataset_and_latents(ckpt, device)
        dataset, latents = closure_cache["data"]
        with torch.no_grad():
            residual = predicted_modes_closure(
                ckpt.build_mode_head().to(device),
                ckpt.build_comp_head().to(device),
                dataset, latents,
            )
        trial.set_user_attr(
            "closure_l2_mean",
            float(np.linalg.norm(residual, axis=1).mean()),
        )
        return bests[objective_head]

    if trials_left > 0:
        study.optimize(objective, n_trials=trials_left, gc_after_trial=True)

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    winner = tiebreak_best(study.trials, objective_head, tie_band, mode_reference)
    summary = {
        "study_name": cfg["study_name"],
        "storage_path": str(storage_path),
        "objective": objective_head,
        "tie_band": tie_band,
        "mode_reference": mode_reference,
        "device": str(device),
        "n_trials_complete": len(completed),
        "optuna_best": (
            {"value": float(study.best_value), "params": dict(study.best_params)}
            if completed else None
        ),
        "tiebreak_winner": (
            {
                "number": winner.number,
                "value": float(winner.value),
                "params": dict(winner.params),
                "user_attrs": dict(winner.user_attrs),
            }
            if winner is not None else None
        ),
        "config": cfg,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsummary -> {summary_path}")
    if winner is not None:
        print(
            f"tiebreak winner: trial {winner.number} "
            f"{objective_head}_best={winner.value:.4f}"
        )
        print(f"params: {winner.params}")
    elif completed:
        print(
            "no admissible trial under the mode_reference constraint -- "
            "the swept architecture moves mode; see the scatter"
        )

    # optuna's matplotlib module flags itself experimental on every call;
    # stable across 4.x, noise at our call sites.
    if completed:
        plots = [(plot_optimization_history, "hpo_history.png")]
        # fANOVA needs a handful of trials before importances are defined
        if len(completed) >= 4:
            plots.append((plot_param_importances, "hpo_param_importances.png"))
        for plot_fn, filename in plots:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ExperimentalWarning)
                ax = plot_fn(study)
            fig_path = output_dir / filename
            ax.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(ax.figure)
            print(f"figure -> {fig_path}")
        _scatter_mode_comp(study, output_dir / "mode_vs_comp_scatter.png")
    return summary
