"""Smoke tests for the loss-curve plotting library.

These pin the basic contracts -- the plot functions don't raise and
they actually render the lines they advertise.  The Optuna integration
gets a quick smoke test against a tiny in-memory study.

`matplotlib.use("Agg")` is set at the top so the suite runs headless
under CI without a display.

Run with:
    pytest tests/test_training_curves.py -v
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import optuna
import pytest

from cnmineral.eval.training_curves import (
    plot_run_loss, plot_study_history, plot_trial_curves,
)


def test_plot_run_loss_returns_axes_with_two_lines() -> None:
    monitor = {"train": [0.5, 0.4, 0.3], "val": [0.6, 0.45, 0.35]}
    ax = plot_run_loss(monitor)
    lines = ax.get_lines()
    assert len(lines) == 2
    assert {l.get_label() for l in lines} == {"train", "val"}
    # x-axis = 1..n_epochs, not 0..n-1
    xs = lines[0].get_xdata()
    assert list(xs) == [1, 2, 3]
    plt.close("all")


def test_plot_run_loss_handles_train_only() -> None:
    """Renders cleanly even if monitor doesn't have a val key yet."""
    ax = plot_run_loss({"train": [0.5, 0.4]})
    assert len(ax.get_lines()) == 1
    plt.close("all")


def test_plot_study_history_smoke() -> None:
    """Build a tiny in-memory study, render history axes; just don't crash."""
    study = optuna.create_study(direction="minimize")

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        return (x - 0.5) ** 2

    study.optimize(objective, n_trials=3)
    ax = plot_study_history(study)
    assert ax.get_xlabel() == "trial"
    assert len(ax.get_lines()) >= 1
    plt.close("all")


def test_plot_trial_curves_smoke() -> None:
    """Trials that called trial.report should appear as faded lines."""
    study = optuna.create_study(direction="minimize")

    def objective(trial: optuna.trial.Trial) -> float:
        for epoch in range(3):
            trial.report(0.5 - 0.1 * epoch, epoch)
        return 0.2

    study.optimize(objective, n_trials=2)
    ax = plot_trial_curves(study)
    assert len(ax.get_lines()) == 2
    plt.close("all")
