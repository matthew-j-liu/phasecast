"""Interrogate the CondHead of a joint checkpoint on the valtest val roster.

  python scripts/eval/cond_head.py --checkpoint checkpoints/joint_heads_seed0/heads_best.pt

`--checkpoint` is a `JointHeadCheckpoint` (three heads + encoder pointer +
label semantics). The val dataset is rebuilt from the checkpoint's stamped
`label_config` plus the committed valtest roster (`valtest_split.csv`);
scoring runs under both the teacher-forced true support and ModeHead-sampled
supports (the exposure-bias measurement). Figures + `metrics.json` land at
`results/cond_head/<checkpoint dir name>/`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

from cnmineral.eval.cond_head import run_cond_head_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="JointHeadCheckpoint produced by scripts/training/joint_heads.py",
    )
    args = parser.parse_args()
    run_cond_head_eval(args.checkpoint.resolve())


if __name__ == "__main__":
    main()
