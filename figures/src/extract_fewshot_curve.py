"""Dump the few-shot fit-comparison curve to CSV.

Reshapes the recorded sweep at
`explorations/real_rock_fewshot/results/joint_heads_allrock_tuned_seed0/
curve_val.json` (fit_fewshot_curve.py's output; nothing is refit here) into two small
tables so the figure script never opens exploration results. Six fits: the
three recalibration tiers and the three heads trained on measured abundances.
The adapted arm is recorded in the sweep but stays out of the paper figure.

`fewshot_curve.csv` is one row per (fit, grid size, replicate) with the
realized rock count, the joint NLL and its presence/abundance branches, and
the abundance MAE.
`fewshot_curve_meta.csv` is the zero-shot reference, one row.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import DATADIR

CURVE = DATADIR / "curve_val.json"
ARMS = ("vector", "matrix", "tailed",
        "measured", "measured_raw", "measured_ablated")


def main():
    recorded = json.loads(CURVE.read_text())
    assert recorded["eval_split"] == "val"

    rows = [{
        "arm": r["arm"],
        "budget": r["budget"],
        "draw": r["draw"],
        "n_rocks": r["n_rocks"],
        "joint": round(r["joint"], 6),
        "bernoulli": round(r["bernoulli"], 6),
        "dirichlet": round(r["dirichlet"], 6),
        "share_mae": round(r["share_mae"], 6),
    } for r in recorded["curve"] if r["arm"] in ARMS]
    pd.DataFrame(rows).to_csv(DATADIR / "fewshot_curve.csv", index=False)

    pd.DataFrame([{
        "zero_shot_joint": round(recorded["zero_shot"]["joint"], 6),
        "zero_shot_share_mae": round(recorded["zero_shot"]["share_mae"], 6),
        "eval_rocks": recorded["splits"]["val"]["rocks"],
        "eval_units": recorded["splits"]["val"]["units"],
    }]).to_csv(DATADIR / "fewshot_curve_meta.csv", index=False)
    print(f"wrote {DATADIR}/fewshot_curve{{,_meta}}.csv ({len(rows)} curve rows)")


if __name__ == "__main__":
    main()
