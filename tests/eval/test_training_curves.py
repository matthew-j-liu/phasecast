"""Smoke tests for the loss-curve plotting library.

These pin the basic contracts -- the plot function doesn't raise and
actually renders the lines it advertises.

`matplotlib.use("Agg")` is set at the top so the suite runs headless
under CI without a display.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cnmineral.eval.training_curves import plot_run_loss


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


def test_plot_run_loss_custom_ylabel() -> None:
    ax = plot_run_loss({"train": [0.5]}, ylabel="cross-entropy")
    assert ax.get_ylabel() == "cross-entropy"
    plt.close("all")
