"""
Render the cross-version trajectory figure from the ledger.

    python scripts/view/trajectory.py

Reads results/ledger.yaml, writes results/trajectory.png -- the private
progress summary; regenerates from the ledger alone.
"""

import argparse
from pathlib import Path

from cnmineral.eval.trajectory import build_trajectory_figure, load_ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("results/ledger.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/trajectory.png"))
    args = parser.parse_args()
    build_trajectory_figure(load_ledger(args.ledger), args.out)


if __name__ == "__main__":
    main()
