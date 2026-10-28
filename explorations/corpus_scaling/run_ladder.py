"""Retrain the standard joint-heads config on nested train-corpus subsets.

  python explorations/corpus_scaling/run_ladder.py
  python explorations/corpus_scaling/run_ladder.py --points 625,2500
  python explorations/corpus_scaling/run_ladder.py --points 2500 --seed 1
  python explorations/corpus_scaling/run_ladder.py --draws 24,12

Run from the repo root. Each point takes `configs/heads/joint_heads.yaml`
verbatim except for the train-tier subset (`train_bulk_ids` for the bulk
ladder, `train_draw_limit` for the draws axis) and the optional `--seed`
override, trains to `checkpoints/corpus_scaling_allrock_*/heads_best.pt`,
then runs the ModeHead interrogation so metrics land at
`runs/mode_head/corpus_scaling_allrock_*/metrics.json`. A point whose
checkpoint already exists is not retrained (delete its checkpoint dir to
force a rerun); its eval still runs if the metrics file is missing.

The `allrock` tag prefix marks the production stack these points train under
(all-rock encoder, tuned dials); the `corpus_scaling_n*` / `corpus_scaling_d*`
checkpoints are the #27 points under the earlier igneous-encoder production
config and must not be reused. The full-N point is not on the default ladder:
at 10000 bulks the config is exactly the production train, so scoring reuses
`joint_heads_allrock_tuned_seed0` (see `run_realrock.py`).

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
TAG_PREFIX = "corpus_scaling_allrock"
BULK_LADDER = (625, 1250, 2500, 5000)


def hash_order(sample_ids: list[int]) -> list[int]:
    return sorted(
        (int(s) for s in sample_ids),
        key=lambda s: hashlib.md5(str(s).encode()).hexdigest(),
    )


def write_roster(ckpt_dir: Path, ordered_ids: list[int], n_bulks: int) -> Path:
    """The first `n_bulks` ids of the md5 order, written sorted as the
    training config's `train_bulk_ids` file beside that point's checkpoint."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    roster_path = ckpt_dir / "train_bulk_ids.csv"
    pd.DataFrame({"sample_id": sorted(ordered_ids[:n_bulks])}).to_csv(roster_path, index=False)
    return roster_path


def point_config(base: dict, tag: str, roster_path: Path | None,
                 draw_limit: int | None, seed: int) -> dict:
    config = dict(base)
    config["head_checkpoint_dir"] = str(CHECKPOINTS / tag)
    config["seed"] = seed
    config.pop("train_bulk_ids", None)
    config.pop("train_draw_limit", None)
    if roster_path is not None:
        config["train_bulk_ids"] = str(roster_path)
    if draw_limit is not None:
        config["train_draw_limit"] = draw_limit
    return config


def run_point(base: dict, tag: str, ordered_ids: list[int],
              n_bulks: int | None, draw_limit: int | None, seed: int) -> None:
    from cnmineral.eval.joint_heads import run_joint_eval
    from cnmineral.training import pick_device
    from cnmineral.training.joint_heads import train_joint_heads

    ckpt_dir = CHECKPOINTS / tag
    ckpt_path = ckpt_dir / base["head_checkpoint_filename"]
    metrics_path = REPO / "runs" / "mode_head" / tag / "metrics.json"

    roster_path = None
    if n_bulks is not None and n_bulks < len(ordered_ids):
        roster_path = write_roster(ckpt_dir, ordered_ids, n_bulks)

    if ckpt_path.exists():
        print(f"{tag}: checkpoint exists, skipping train")
    else:
        config = point_config(base, tag, roster_path, draw_limit, seed)
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
    parser.add_argument(
        "--seed", type=int, default=0,
        help="training seed; nonzero appends _seed{N} to the point tags",
    )
    parser.add_argument(
        "--rosters-only", action="store_true",
        help="write the nested roster of each --points size and stop; "
             "run_variant.py trains fixed-depth cells on them",
    )
    args = parser.parse_args()

    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)
    ordered_ids = hash_order(pd.read_csv(TRAIN_BULKS)["sample_id"].tolist())

    if args.rosters_only:
        for n in args.points.split(","):
            path = write_roster(CHECKPOINTS / f"{TAG_PREFIX}_n{int(n)}", ordered_ids, int(n))
            print(f"{path}: {int(n)} compositions")
        return

    suffix = f"_seed{args.seed}" if args.seed else ""
    if args.draws:
        points = [(f"{TAG_PREFIX}_d{int(k)}{suffix}", None, int(k))
                  for k in args.draws.split(",")]
    else:
        points = [(f"{TAG_PREFIX}_n{int(n)}{suffix}", int(n), None)
                  for n in args.points.split(",")]
    for tag, n_bulks, draw_limit in points:
        print(f"\n== {tag} ==")
        run_point(base, tag, ordered_ids, n_bulks, draw_limit, args.seed)


if __name__ == "__main__":
    main()
