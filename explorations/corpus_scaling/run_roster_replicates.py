"""Roster replicates of the 224-step corpus-size line: at each size below the
full corpus, five rosters drawn at random from the training corpus (not the
nested md5 prefix every `final224_allrock_n{N}` cell uses), each trained
exactly as those cells were (seed 0, last-epoch state, 224 gradient steps of
1,024 rows) and evaluated zero-shot. A replicate differs from the nested
seed-0 head only by which compositions it was trained on, so the spread
across replicates measures what roster choice does to a point of the line.

  python explorations/corpus_scaling/run_roster_replicates.py
  python explorations/corpus_scaling/run_roster_replicates.py --sizes 10 --replicates 1

Run from the repo root. Each cell trains through `run_variant.py` in its own
process and is skipped when its checkpoint and zero-shot run exist.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from explorations.corpus_scaling.run_ladder import CHECKPOINTS, TRAIN_BULKS

SIZES = (10, 20, 40, 78, 156, 312, 625, 1250, 5000)
REPLICATES = (1, 2, 3, 4, 5)
# below 78 compositions an epoch holds fewer than the 8 truncated batches,
# so the epoch count scales to hold 224 gradient steps (corpus_scaling README)
EPOCHS = {10: 224, 20: 112, 40: 56}


def replicate_tag(n, r):
    return f"final224_allrock_n{n}_roster{r}"


def replicate_roster(ids, n, r):
    """`n` corpus ids drawn without replacement, the same for the same
    (n, r), sorted as the training config reads them."""
    return np.sort(np.random.default_rng((n, r)).choice(np.asarray(ids), size=n, replace=False))


def train(n, r, ids):
    tag = replicate_tag(n, r)
    roster = CHECKPOINTS / tag / "train_bulk_ids.csv"
    roster.parent.mkdir(parents=True, exist_ok=True)
    if not roster.exists():
        pd.DataFrame({"sample_id": replicate_roster(ids, n, r)}).to_csv(roster, index=False)
    cmd = [sys.executable, str(REPO / "explorations" / "corpus_scaling" / "run_variant.py"), "--tag", tag,
           "--roster", str(roster), "--batches-per-epoch", "8", "--set", f"epochs={EPOCHS.get(n, 28)}",
           "--mode-selection", "final", "--seed", "0"]
    start = time.time()
    subprocess.run(cmd, check=True, cwd=REPO, capture_output=True)
    return time.time() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=list(SIZES))
    ap.add_argument("--replicates", type=lambda s: [int(x) for x in s.split(",")], default=list(REPLICATES))
    args = ap.parse_args()
    ids = pd.read_csv(TRAIN_BULKS)["sample_id"].tolist()
    for n in args.sizes:
        for r in args.replicates:
            print(f"{replicate_tag(n, r)}: {train(n, r, ids):.0f} s", flush=True)


if __name__ == "__main__":
    main()
