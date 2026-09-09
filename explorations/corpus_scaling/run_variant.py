"""Train one joint-heads variant off the standard config and evaluate it
zero-shot on the real-rock benchmark: one cell of the depth grid.

  python explorations/corpus_scaling/run_variant.py --tag steps56_allrock_n10000 \\
      --batches-per-epoch 8 --set epochs=7
  python explorations/corpus_scaling/run_variant.py --tag depth56_allrock_n78 \\
      --roster checkpoints/corpus_scaling_allrock_n78/train_bulk_ids.csv \\
      --batches-per-epoch 8 --set epochs=7
  python explorations/corpus_scaling/run_variant.py --tag final224_allrock_n78 \\
      --roster checkpoints/corpus_scaling_allrock_n78/train_bulk_ids.csv \\
      --batches-per-epoch 8 --set epochs=28 --mode-selection final
  python explorations/corpus_scaling/run_variant.py --tag stopmae_allrock_n10000_seed1 \\
      --batches-per-epoch 8 --mode-selection real_rock_train_share_mae --seed 1

Run from the repo root. `--set key=value` overrides a config key (value
parsed as YAML). `--batches-per-epoch N` truncates every training epoch to
its first N shuffled batches, so `--set epochs=E` fixes the head at N x E
gradient steps of 1,024 rows (per-head selection on corpus val each epoch);
the val loader is untouched. `--mode-selection` sets the ModeHead's selection
rule (`val`, `final`, or one of the real-rock train-split series the probe
supplies each epoch: hurdle NLL terms and share absolute error over the 372
evaluated train rocks). `run_depth_grid.py` finds cells by the tag grammar
in the README. An existing checkpoint is not retrained; an existing
zero-shot run is not re-evaluated.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO / "configs" / "heads" / "joint_heads.yaml"


def truncated_loader(n_batches: int):
    class TruncatedLoader(DataLoader):
        def __init__(self, *args, shuffle=False, **kwargs):
            super().__init__(*args, shuffle=shuffle, **kwargs)
            self.truncate = shuffle

        def __iter__(self):
            for i, batch in enumerate(super().__iter__()):
                if self.truncate and i >= n_batches:
                    return
                yield batch

    return TruncatedLoader


PROBE_SERIES = ("real_rock_train_joint", "real_rock_train_bernoulli",
                "real_rock_train_dirichlet", "real_rock_train_share_mae")


def real_rock_probe(config: dict, device):
    """Per-epoch hurdle NLL terms and share MAE of the mode head on the
    evaluated real-rock train split, computed from one cached set of latents."""
    from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
    from cnmineral.eval.mode_head import conditional_mean, magnitude_parity_stats
    from cnmineral.eval.real_rock import load_benchmark, predict, real_rock_nll, split_mask

    encoder, _ = load_encoder_from_checkpoint(config["encoder_checkpoint"])
    benchmark = load_benchmark()
    latents = bulk_latents(encoder.to(device), benchmark, device)
    mask = split_mask(benchmark, "train") & benchmark.clean & benchmark.has_magnitude
    numeric = benchmark.numeric.astype(np.float64)

    def probe(mode_head):
        pred = predict(mode_head, latents, benchmark)
        nll = real_rock_nll(pred["p7"], pred["alpha7"], benchmark.presence,
                            benchmark.magnitude, benchmark.numeric)
        cond_mean = conditional_mean(pred["alpha7"], numeric)
        parity = magnitude_parity_stats(cond_mean[mask], benchmark.magnitude[mask],
                                        numeric[mask])
        return {
            "real_rock_train_joint": nll["joint"][mask].mean(),
            "real_rock_train_bernoulli": nll["bernoulli"][mask].mean(),
            "real_rock_train_dirichlet": nll["dirichlet"][mask].mean(),
            "real_rock_train_share_mae": float(np.nanmean(parity["magnitude_mae"])),
        }

    return probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--roster", default=None,
                        help="train_bulk_ids CSV (omit for the full 10,000)")
    parser.add_argument("--set", action="append", default=[],
                        metavar="KEY=VALUE")
    parser.add_argument("--batches-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode-selection", default="val",
                        choices=("val", "final", *PROBE_SERIES))
    args = parser.parse_args()

    with open(BASE_CONFIG) as f:
        config = yaml.safe_load(f)
    config.pop("train_bulk_ids", None)
    config.pop("train_draw_limit", None)
    if args.roster:
        config["train_bulk_ids"] = str(Path(args.roster).resolve())
    for item in args.set:
        key, _, value = item.partition("=")
        config[key] = yaml.safe_load(value)
    config["seed"] = args.seed
    config["mode_selection"] = args.mode_selection
    config["head_checkpoint_dir"] = str(REPO / "checkpoints" / args.tag)
    ckpt_path = Path(config["head_checkpoint_dir"]) / config["head_checkpoint_filename"]

    from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
    from cnmineral.eval.real_rock import load_benchmark, run_real_rock_eval
    from cnmineral.training import pick_device
    from cnmineral.training import joint_heads as jh

    device = pick_device()
    if ckpt_path.exists():
        print(f"{args.tag}: checkpoint exists, skipping train")
    else:
        if args.batches_per_epoch:
            jh.DataLoader = truncated_loader(args.batches_per_epoch)
        overrides = {k: config[k] for k in
                     [i.partition("=")[0] for i in args.set]}
        print(f"{args.tag}: overrides {overrides} roster={args.roster} "
              f"batches_per_epoch={args.batches_per_epoch} "
              f"mode_selection={args.mode_selection}")
        probe = (real_rock_probe(config, device)
                 if args.mode_selection in PROBE_SERIES else None)
        jh.train_joint_heads(config, device, epoch_probe=probe)

    metrics_path = REPO / "runs" / "real_rock" / args.tag / "metrics.json"
    if metrics_path.exists():
        print(f"{args.tag}: zero-shot metrics exist, skipping eval")
        return
    ckpt = jh.JointHeadCheckpoint.load(ckpt_path)
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder, benchmark, device)
    run_real_rock_eval(ckpt, benchmark, latents, device=device, tag=args.tag)


if __name__ == "__main__":
    main()
