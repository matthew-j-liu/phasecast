"""Cross-version trajectory figure on the real-rock axis.

The private progress summary (results/README.md). Reads
`results/real_rock_ledger.yaml` (one zero-shot entry per model version on
the eruption benchmark) and renders joint NLL over versions with the y-axis
inverted (up = better). Entries scored on different benchmark versions are
drawn as disconnected segments -- numbers compare only within one benchmark.
The real rocks are the one instrument that survives corpus-era churn, which
is why the trajectory lives on this axis; per-era corpus numbers stay in
`results/ledger.yaml`, comparable only within their own instrument. When the
tracked checkpoint contrast (`explorations/real_rock_fewshot/results/
checkpoint_contrast_<tier>.json`) covers a version, its post-recalibration
val-slice level is overlaid as a second series. Regenerates from tracked
files alone, so it survives checkpoint and runs/ cleaning. Thin CLI:
`scripts/view/trajectory.py`.
"""

import json
from itertools import groupby
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REQUIRED = {"version", "date", "checkpoint_tag", "benchmark", "changes",
            "n", "joint_nll"}


def load_ledger(path):
    """Validated real-rock version entries, ledger order preserved."""
    with open(path) as f:
        payload = yaml.safe_load(f)
    entries = payload.get("versions") if isinstance(payload, dict) else None
    if not entries:
        raise ValueError(f"{path}: expected a top-level `versions` list")
    benchmarks = payload.get("benchmarks") or {}
    for e in entries:
        missing = REQUIRED - set(e)
        if missing:
            raise ValueError(
                f"ledger entry {e.get('version', '?')}: missing {sorted(missing)}"
            )
        if e["benchmark"] not in benchmarks:
            raise ValueError(
                f"ledger entry {e['version']}: benchmark `{e['benchmark']}` "
                f"not defined under `benchmarks:`"
            )
    return entries


def load_adapted_levels(path):
    """(benchmark, {tag: post-recalibration val joint NLL}), or (None, {}).

    The contrast JSON is a tracked exploration artifact; absence just means
    no overlay, never an error -- the zero-shot trajectory must regenerate
    from the ledger alone. The benchmark rides along so the overlay only
    attaches to ledger entries scored on the same benchmark version.
    """
    path = Path(path)
    if not path.exists():
        return None, {}
    contrast = json.loads(path.read_text())
    levels = {
        tag: by_slice["val"]["adapted_joint"]
        for tag, by_slice in contrast.get("levels", {}).items()
        if "val" in by_slice
    }
    return contrast.get("benchmark"), levels


def build_trajectory_figure(entries, out_path, adapted=None,
                            adapted_benchmark=None):
    """Joint NLL over versions, one disconnected segment per benchmark."""
    names = [f"{e['version']}\n({e['benchmark']})" for e in entries]
    segments = []
    start = 0
    for benchmark, run in groupby(entries, key=lambda e: e["benchmark"]):
        stop = start + len(list(run))
        segments.append((benchmark, start, stop))
        start = stop
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for benchmark, seg_start, seg_stop in segments:
        xs = list(range(seg_start, seg_stop))
        ax.plot(xs, [entries[i]["joint_nll"] for i in xs], marker="o",
                color="tab:blue",
                label="zero-shot (headline rocks)" if seg_start == 0 else None)
    for _, seg_start, _ in segments[1:]:
        ax.axvline(seg_start - 0.5, ls=":", color="0.6", lw=0.8)
    adapted = adapted or {}
    pts = [(i, adapted[e["checkpoint_tag"]]) for i, e in enumerate(entries)
           if e["checkpoint_tag"] in adapted
           and (adapted_benchmark is None or e["benchmark"] == adapted_benchmark)]
    if pts:
        ax.plot(*zip(*pts), marker="s", ls="--", color="tab:orange",
                label="after matrix recalibration (val slice)")
    ax.legend(fontsize=8)
    ax.set_xticks(range(len(names)), names, fontsize=7.5, rotation=12)
    ax.set_xlim(-0.4, len(names) - 0.6)
    ax.set_ylabel("real-rock joint NLL (nats; up = better)")
    ax.set_title("model versions on the eruption benchmark", fontsize=10)
    ax.invert_yaxis()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
