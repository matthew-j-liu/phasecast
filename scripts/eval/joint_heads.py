"""Interrogate the heads of a joint checkpoint on the valtest val roster.

  python scripts/eval/joint_heads.py --checkpoint checkpoints/joint_heads_seed0/heads_best.pt
  python scripts/eval/joint_heads.py --checkpoint ... --heads mode,comp

`--checkpoint` is a `JointHeadCheckpoint` (three heads + encoder pointer +
label semantics). The val dataset is rebuilt once from the checkpoint's
stamped `label_config` plus the committed valtest roster
(`valtest_split.csv`) and shared across the requested heads. Figures +
`metrics.json` land at `results/{mode_head,comp_head,cond_head}/<checkpoint
dir name>/`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

from cnmineral.eval.joint_heads import HEAD_EVALS, run_joint_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="JointHeadCheckpoint produced by scripts/training/joint_heads.py",
    )
    parser.add_argument(
        "--heads", default=",".join(HEAD_EVALS),
        help=f"comma-separated subset of {','.join(HEAD_EVALS)} (default: all)",
    )
    args = parser.parse_args()
    heads = tuple(h.strip() for h in args.heads.split(",") if h.strip())
    run_joint_eval(args.checkpoint.resolve(), heads=heads)


if __name__ == "__main__":
    main()
