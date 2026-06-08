"""Loss-curve plotting for SSL pretrain runs and Optuna studies.

Three pure functions, all matplotlib-only:

- `plot_run_loss(monitor)` -- two-line train + val curves vs epoch,
  from the dict-shaped monitor stored on every checkpoint.
- `plot_study_history(study)` -- per-trial best-val-so-far across an
  Optuna study (wraps the Optuna matplotlib helper).
- `plot_trial_curves(study)` -- per-trial val-loss intermediate values
  (one faded line per trial, the per-epoch reports made by
  `trial.report` during `run_pretraining`).

All accept an optional `ax=` keyword and return the `Axes`.  No file
I/O lives here -- the CLI driver in `scripts/plot_training.py` is
where saving happens.
"""

import matplotlib.pyplot as plt
import optuna
from matplotlib.axes import Axes


def plot_run_loss(
    monitor: dict[str, list[float]],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Render train + val loss curves from a checkpoint's monitor dict.

    Parameters
    ----------
    monitor
        `{"train": [per-epoch], "val": [per-epoch]}` as written by
        `save_checkpoint` in `scripts/pretrain.py`.
    ax
        Optional matplotlib Axes.  Created if absent.
    title
        Optional axes title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    train = monitor.get("train", [])
    val = monitor.get("val", [])
    if train:
        ax.plot(range(1, len(train) + 1), train, marker="o", label="train")
    if val:
        ax.plot(range(1, len(val) + 1), val, marker="o", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked-element MSE")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_study_history(
    study: optuna.Study,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Per-trial best-val-so-far across an Optuna study.

    Renders one dot per completed trial + a running-min line.  Useful
    for "is the sweep converging?"  Skips pruned / failed trials.
    """
    if ax is None:
        _, ax = plt.subplots()
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not completed:
        ax.set_title("no completed trials")
        return ax
    numbers = [t.number for t in completed]
    values = [t.value for t in completed]
    best_so_far = [min(values[: i + 1]) for i in range(len(values))]
    ax.plot(numbers, values, "o", alpha=0.4, label="trial value")
    ax.plot(numbers, best_so_far, "-", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("final-epoch val loss")
    ax.legend()
    return ax


def plot_trial_curves(
    study: optuna.Study,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Per-trial val-loss curves -- one faded line per trial.

    Reads `trial.intermediate_values` populated by the
    `trial.report(val_loss, epoch)` call inside `run_pretraining`'s
    epoch loop.  Empty trials (no reports) are skipped.
    """
    if ax is None:
        _, ax = plt.subplots()
    for t in study.trials:
        if not t.intermediate_values:
            continue
        epochs = sorted(t.intermediate_values.keys())
        losses = [t.intermediate_values[e] for e in epochs]
        ax.plot(epochs, losses, alpha=0.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss")
    ax.set_title("per-trial val curves")
    return ax
