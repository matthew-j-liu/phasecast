"""Cross-version trajectory figure: raw headline NLLs over shipped versions.

The private progress summary (results/README.md). Reads
`results/ledger.yaml` (one entry per shipped model version) and renders
per-head small-multiple panels of raw headline NLL with the y-axis
inverted (up = better). Versions measured on different instruments (the
ledger's `instrument` stamp) are drawn as disconnected segments -- NLLs
are comparable only within one instrument, never across the boundary.
Regenerates from the ledger alone, so it survives checkpoint and runs/
cleaning. Thin CLI: `scripts/view/trajectory.py`.
"""

from itertools import groupby
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

HEADS = ("mode", "comp", "cond")
HEADLINE_KEY = {"mode": "joint_nll", "comp": "comp_nll", "cond": "cond_nll"}
PANEL_TITLE = {
    "mode": "ModeHead (hurdle NLL / row)",
    "comp": "CompHead (Dirichlet NLL / present (row, phase))",
    "cond": "CondHead (GMM NLL / row, 3 axes)",
}


def load_ledger(path):
    """Validated version entries, ledger order preserved."""
    with open(path) as f:
        payload = yaml.safe_load(f)
    entries = payload.get("versions") if isinstance(payload, dict) else None
    if not entries:
        raise ValueError(f"{path}: expected a top-level `versions` list")
    instruments = payload.get("instruments") or {}
    for e in entries:
        missing = {
            "version", "date", "checkpoint_tag", "instrument", "changes", "heads"
        } - set(e)
        if missing:
            raise ValueError(f"ledger entry {e.get('version', '?')}: missing {sorted(missing)}")
        if e["instrument"] not in instruments:
            raise ValueError(
                f"ledger entry {e['version']}: instrument `{e['instrument']}` "
                f"not defined under `instruments:`"
            )
        unknown = set(e["heads"]) - set(HEADS)
        if unknown:
            raise ValueError(f"ledger entry {e['version']}: unknown heads {sorted(unknown)}")
        for head, metrics in e["heads"].items():
            if HEADLINE_KEY[head] not in metrics:
                raise ValueError(
                    f"ledger entry {e['version']}: head `{head}` lacks "
                    f"`{HEADLINE_KEY[head]}`"
                )
    return entries


def build_trajectory_figure(entries, out_path):
    """Per-head NLL panels over versions, one disconnected segment per instrument."""
    names = [e["version"] for e in entries]
    segments = []
    start = 0
    for instrument, run in groupby(entries, key=lambda e: e["instrument"]):
        stop = start + len(list(run))
        segments.append((instrument, start, stop))
        start = stop
    fig, axes = plt.subplots(1, len(HEADS), figsize=(11.5, 3.8))
    for head, ax in zip(HEADS, axes):
        for instrument, seg_start, seg_stop in segments:
            xs = [
                i for i in range(seg_start, seg_stop) if head in entries[i]["heads"]
            ]
            ys = [entries[i]["heads"][head][HEADLINE_KEY[head]] for i in xs]
            ax.plot(xs, ys, marker="o", color="tab:blue")
        for _, seg_start, _ in segments[1:]:
            ax.axvline(seg_start - 0.5, ls=":", color="0.6", lw=0.8)
        ax.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=8)
        ax.set_xlim(-0.4, len(names) - 0.6)
        ax.set_title(PANEL_TITLE[head], fontsize=9)
        ax.invert_yaxis()
    if len(segments) > 1:
        for ax in axes:
            for instrument, seg_start, seg_stop in segments:
                ax.text(
                    (seg_start + seg_stop - 1) / 2, 0.03, instrument,
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6, color="0.4",
                )
    axes[0].set_ylabel("NLL (nats; up = better)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
