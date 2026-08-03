"""
Render the cross-version real-rock trajectory figure.

    python scripts/view/trajectory.py

Reads results/real_rock_ledger.yaml (plus the tracked checkpoint-contrast
JSON for the post-recalibration overlay, when present), writes
results/trajectory.png -- the private progress summary; regenerates from
tracked files alone.
"""

import argparse
from pathlib import Path

from cnmineral.eval.trajectory import (
    build_trajectory_figure,
    load_adapted_levels,
    load_ledger,
)

CONTRAST = Path("explorations/real_rock_fewshot/results/checkpoint_contrast_matrix.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path,
                        default=Path("results/real_rock_ledger.yaml"))
    parser.add_argument("--contrast", type=Path, default=CONTRAST)
    parser.add_argument("--out", type=Path, default=Path("results/trajectory.png"))
    args = parser.parse_args()
    benchmark, levels = load_adapted_levels(args.contrast)
    build_trajectory_figure(load_ledger(args.ledger), args.out,
                            adapted=levels, adapted_benchmark=benchmark)


if __name__ == "__main__":
    main()
