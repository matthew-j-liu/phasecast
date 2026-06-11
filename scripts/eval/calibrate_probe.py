"""Cross-encoder probe-HPO calibration report CLI.

    python scripts/eval/calibrate_probe.py \\
        results/probe_hpo/*/summary.json \\
        --output-dir results/probe_calibration/

All business logic lives in `cnmineral.eval.probe.calibration`;
this file is the argparse shell.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

from cnmineral.eval.probe import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_paths", nargs="+", type=Path,
        help="probe-HPO summary JSONs, one per encoder",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/probe_calibration"),
    )
    args = parser.parse_args()
    run_calibration(args.summary_paths, args.output_dir.resolve())


if __name__ == "__main__":
    main()
