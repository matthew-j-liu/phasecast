"""Caches corpus label statistics to runs/corpus_stats.json for
scripts/build/build_numbers.py: per-phase presence rates over the production
training labels. The corpus pass takes about a minute, so it runs here once
rather than inside every numbers rebuild."""

import json
from pathlib import Path

import pandas as pd

from cnmineral.data.mineralogy_labels import _route_tokens
from cnmineral.data.phase_vocab import (
    DROPPED_PHASES, NON_SOLID_PHASES, PRESENCE_CUTOFF, SOLID_PHASES)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "train"
RUNGS = (50, 30)
OUT = ROOT / "runs" / "corpus_stats.json"


def main():
    cols = ["bulk", "draw", "rung", "phase", "mode_wt",
            "wt_CaO", "wt_Na2O", "wt_K2O"]
    parts = []
    for f in sorted(CORPUS.glob("chunk_*_rungs.parquet")):
        part = pd.read_parquet(f, columns=cols)
        parts.append(part[part["rung"].isin(RUNGS)])
    df = pd.concat(parts, ignore_index=True)

    solids = df[~df["phase"].isin(NON_SOLID_PHASES)
                & ~df["phase"].isin(DROPPED_PHASES)].copy()
    solids["token"] = _route_tokens(solids)
    modes = (solids.groupby(["bulk", "draw", "rung", "token"], sort=False)
             ["mode_wt"].sum().unstack("token", fill_value=0.0)
             .reindex(columns=list(SOLID_PHASES), fill_value=0.0))
    modes = modes[modes.sum(axis=1) > 0]
    presence = modes.div(modes.sum(axis=1), axis=0) > PRESENCE_CUTOFF

    stats = {
        "corpus": str(CORPUS.relative_to(ROOT)),
        "rungs": list(RUNGS),
        "n_rows": int(len(presence)),
        "presence_rate_pct": {
            p: float(presence[p].mean() * 100) for p in SOLID_PHASES
        },
    }
    OUT.write_text(json.dumps(stats, indent=2))
    print(f"{len(presence)} rows -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
