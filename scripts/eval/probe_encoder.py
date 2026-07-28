"""Probe a trained encoder -- single mode or Optuna HPO sweep.

  Single mode (fixed-protocol probes + UMAP figures):
      python scripts/eval/probe_encoder.py --checkpoint CKPT.pt

  HPO mode (Optuna sweep over MLP probe HPs):
      python scripts/eval/probe_encoder.py --hpo \\
          --checkpoint CKPT.pt --config probe_hpo.yaml

The dataset and train/test split are reconstructed from
`ckpt.config` (no separate pretrain YAML needed).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# silence libomp's informational stderr chatter (e.g. the
# omp_set_nested deprecation note umap/numba trips)
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path
from cnmineral.eval.probe import run_hpo, run_scoring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="encoder checkpoint produced by scripts/training/pretrain.py",
    )
    parser.add_argument(
        "--hpo", action="store_true",
        help="run an Optuna probe-HPO sweep instead of a single fixed-protocol pass",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="probe HPO sweep YAML (required when --hpo is set)",
    )
    parser.add_argument(
        "--eval-roster", type=Path, default=None,
        help="sample_id CSV restricting the scored test rows (shared-row "
             "protocol for cross-corpus encoder comparison)",
    )
    parser.add_argument(
        "--redox", choices=("populated", "padded"), default="populated",
        help="redox-token view for latent extraction on a redox corpus; "
             "padded is the head-to-head condition vs a 22-token encoder",
    )

    args = parser.parse_args()

    if args.hpo:
        if args.config is None:
            parser.error("--hpo requires --config")
        if args.eval_roster is not None or args.redox != "populated":
            parser.error("--eval-roster/--redox only apply in single mode")
        run_hpo(
            checkpoint_path=args.checkpoint.resolve(),
            hpo_config_path=args.config.resolve(),
        )
    else:
        if args.config is not None:
            parser.error("--config only applies in --hpo mode")
        run_scoring(
            checkpoint_path=args.checkpoint.resolve(),
            eval_roster=(
                args.eval_roster.resolve() if args.eval_roster else None
            ),
            redox=args.redox,
        )


if __name__ == "__main__":
    main()
