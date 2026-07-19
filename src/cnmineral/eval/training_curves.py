"""
Loss-curve plotting for training runs.

Two pure functions, matplotlib-only:

- `plot_run_loss(monitor)` -- two-line train + val curves vs epoch,
  from the dict-shaped monitor stored on every checkpoint (and on
  `ProbeMetrics` for MLP probes).
- `plot_joint_run_loss(monitor)` -- the multi-panel variant for
  monitors carrying per-term series (`train_<term>` / `val_<term>`,
  as on `JointHeadCheckpoint`): one panel for the weighted total plus
  one per term, since the terms live on very different scales.

No file I/O lives here -- saving happens at the call sites.

Optuna-study figures (optimization history, per-trial intermediate
values) are NOT wrapped here: callers use
`optuna.visualization.matplotlib.plot_optimization_history` and
`plot_intermediate_values` directly.
"""

import math

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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


def plot_joint_run_loss(
    monitor: dict[str, list[float]],
    *,
    title: str | None = None,
) -> Figure:
    """Render per-term train + val curves from a joint-heads monitor.

    Parameters
    ----------
    monitor
        Monitor dict with `train` / `val` totals plus `train_<term>` /
        `val_<term>` series, as carried on `JointHeadCheckpoint`.
        Term panels are laid out in the monitor's key order.
    title
        Optional figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    terms = [k.removeprefix("train_") for k in monitor if k.startswith("train_")]
    n_panels = 1 + len(terms)
    ncols = 3
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    flat = axes.ravel()
    plot_run_loss(monitor, ax=flat[0], title="total", ylabel="weighted total")
    for ax, term in zip(flat[1:], terms):
        sub = {
            "train": monitor.get(f"train_{term}", []),
            "val": monitor.get(f"val_{term}", []),
        }
        plot_run_loss(sub, ax=ax, title=term)
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
