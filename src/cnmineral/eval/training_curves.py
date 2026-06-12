"""
Loss-curve plotting for training runs.

One pure function, matplotlib-only:

- `plot_run_loss(monitor)` -- two-line train + val curves vs epoch,
  from the dict-shaped monitor stored on every checkpoint (and on
  `ProbeMetrics` for MLP probes).

Accepts an optional `ax=` keyword and returns the `Axes`.  No file
I/O lives here -- saving happens at the call sites.

Optuna-study figures (optimization history, per-trial intermediate
values) are NOT wrapped here: callers use
`optuna.visualization.matplotlib.plot_optimization_history` and
`plot_intermediate_values` directly.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator


def plot_run_loss(
    monitor: dict[str, list[float]],
    *,
    ax: Axes | None = None,
    title: str | None = None,
    ylabel: str = "loss",
) -> Axes:
    """Render train + val loss curves from a run's monitor dict.

    Parameters
    ----------
    monitor
        `{"train": [per-epoch], "val": [per-epoch]}` as carried on
        every `cnmineral.training.Checkpoint`.
    ax
        Optional matplotlib Axes.  Created if absent.
    title
        Optional axes title.
    ylabel
        Y-axis label naming the loss being plotted (e.g.
        `"masked-element MSE"` for pretrain, `"cross-entropy"` for
        the MLP probe).

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
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title:
        ax.set_title(title)
    ax.legend()
    return ax
