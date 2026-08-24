"""Dump the pretraining and joint-heads loss histories to CSV.

Reads the two production checkpoints' `monitor` dicts and writes two small
tables into `figures/data/` so the figure scripts never open `checkpoints/`.
Run from repo root; only needed when either checkpoint changes.

`pretrain_loss.csv` is the masked-element reconstruction loss per epoch. This
run's best epoch is its last, so the checkpoint's monitor (which truncates at
best) is the full history; a run whose best preceded its last would need the
untruncated monitor from `runs/.../run_summary.json`. `heads_loss.csv` carries
the three per-head selection curves (assemblage = presence + mode, comp, cond),
each tagged with the head's best (saved) epoch.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmineral.training.joint_heads import selection_series
from style import DATADIR, ROOT

PRETRAIN = ROOT / "checkpoints/pretrain_allrock_redox_seed0/epoch_best.pt"
HEADS = ROOT / "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt"

HEAD_TERMS = {
    "assemblage": ["presence_bce", "mode_nll"],
    "comp": ["comp_nll"],
    "cond": ["cond_nll"],
}
SELECT = {"assemblage": "mode", "comp": "comp", "cond": "cond"}


def _sum(mon, split, terms):
    series = [mon[f"{split}_{t}"] for t in terms]
    return [sum(vals) for vals in zip(*series)]


def main():
    pmon = torch.load(PRETRAIN, map_location="cpu", weights_only=False)["monitor"]
    pd.DataFrame({
        "epoch": range(1, len(pmon["train"]) + 1),
        "train": pmon["train"],
        "val": pmon["val"],
    }).to_csv(DATADIR / "pretrain_loss.csv", index=False)
    print(f"pretrain: {len(pmon['train'])} epochs")

    hmon = torch.load(HEADS, map_location="cpu", weights_only=False)["monitor"]
    sel = selection_series(hmon)
    n = len(hmon["train"])
    best = {}
    rows = []
    for head, terms in HEAD_TERMS.items():
        s = sel[SELECT[head]]
        best[head] = 1 + min(range(len(s)), key=s.__getitem__)
        train, val = _sum(hmon, "train", terms), _sum(hmon, "val", terms)
        for e in range(n):
            rows.append({"head": head, "epoch": e + 1, "train": train[e],
                         "val": val[e], "best_epoch": best[head]})
    pd.DataFrame(rows).to_csv(DATADIR / "heads_loss.csv", index=False)
    print(f"heads: {n} epochs, best {best}")


if __name__ == "__main__":
    main()
