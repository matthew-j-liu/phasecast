"""Retrain the standard joint-heads config on nested train-corpus subsets.

  python explorations/corpus_scaling/run_ladder.py
  python explorations/corpus_scaling/run_ladder.py --points 625,2500
  python explorations/corpus_scaling/run_ladder.py --draws 24,12

Run from the repo root. Each point takes `configs/heads/joint_heads.yaml`
verbatim except for the train-tier subset (`train_bulk_ids` for the bulk
ladder, `train_draw_limit` for the draws axis), trains to
`checkpoints/corpus_scaling_*/heads_best.pt`, then runs the ModeHead
interrogation so metrics land at `runs/mode_head/corpus_scaling_*/
metrics.json` for `aggregate.py`. A point whose checkpoint already exists is
not retrained (delete its checkpoint dir to force a rerun); its eval still
runs if the metrics file is missing.

Bulk subsets are nested by construction: sample ids sorted by their md5 hex
digest, prefix of length N. md5 is platform- and session-stable, unlike the
salted builtin `hash`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO / "configs" / "heads" / "joint_heads.yaml"
TRAIN_BULKS = REPO / "data" / "corpus" / "train_bulks.csv"
CHECKPOINTS = REPO / "checkpoints"
BULK_LADDER = (625, 1250, 2500, 5000, 10000)


def hash_order(sample_ids: list[int]) -> list[int]:
    return sorted(
        (int(s) for s in sample_ids),
        key=lambda s: hashlib.md5(str(s).encode()).hexdigest(),
    )


def point_config(base: dict, tag: str, roster_path: Path | None,
                 draw_limit: int | None) -> dict:
    config = dict(base)
    config["head_checkpoint_dir"] = str(CHECKPOINTS / tag)
    config.pop("train_bulk_ids", None)
    config.pop("train_draw_limit", None)
    if roster_path is not None:
        config["train_bulk_ids"] = str(roster_path)
    if draw_limit is not None:
        config["train_draw_limit"] = draw_limit
    return config


def run_point(base: dict, tag: str, ordered_ids: list[int],
              n_bulks: int | None, draw_limit: int | None) -> None:
    from cnmineral.eval.joint_heads import run_joint_eval
    from cnmineral.training import pick_device
    from cnmineral.training.joint_heads import train_joint_heads

    ckpt_dir = CHECKPOINTS / tag
    ckpt_path = ckpt_dir / base["head_checkpoint_filename"]
    metrics_path = REPO / "runs" / "mode_head" / tag / "metrics.json"

    roster_path = None
    if n_bulks is not None and n_bulks < len(ordered_ids):
        roster_path = ckpt_dir / "train_bulk_ids.csv"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample_id": sorted(ordered_ids[:n_bulks])}).to_csv(
            roster_path, index=False
        )

    if ckpt_path.exists():
        print(f"{tag}: checkpoint exists, skipping train")
    else:
        config = point_config(base, tag, roster_path, draw_limit)
        train_joint_heads(config, pick_device())

    if metrics_path.exists():
        print(f"{tag}: metrics exist, skipping eval")
    else:
        run_joint_eval(ckpt_path, heads=("mode",))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points", default=",".join(str(n) for n in BULK_LADDER),
        help="comma-separated bulk counts (nested subsets of the train tier)",
    )
    parser.add_argument(
        "--draws", default=None,
        help="comma-separated draw limits at full bulk count "
             "(e.g. 24,12; the bulks-vs-draws axis), instead of the bulk ladder",
    )
    args = parser.parse_args()

    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)
    ordered_ids = hash_order(pd.read_csv(TRAIN_BULKS)["sample_id"].tolist())

    if args.draws:
        points = [(f"corpus_scaling_d{int(k)}", None, int(k))
                  for k in args.draws.split(",")]
    else:
        points = [(f"corpus_scaling_n{int(n)}", int(n), None)
                  for n in args.points.split(",")]
    for tag, n_bulks, draw_limit in points:
        print(f"\n== {tag} ==")
        run_point(base, tag, ordered_ids, n_bulks, draw_limit)


if __name__ == "__main__":
    main()
