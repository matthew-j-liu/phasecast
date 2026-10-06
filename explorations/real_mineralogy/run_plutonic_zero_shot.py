"""The first plutonic number: a joint checkpoint zero-shot, beside eruption-v2.

Evaluates the checkpoint with nothing fit on the plutonic benchmark's train and
val rocks, per mineral category, and on eruption-v2's train and val rocks the
same way, so the two sit side by side; test stays sealed on both.  Calls
`score_real_rock` and `real_rock_metrics` directly rather than
`run_real_rock_eval`, which writes into `runs/real_rock/<tag>/`, where the
production eruption-v2 numbers live.  Writes
`results/<tag>/plutonic_zero_shot.json` and prints the per-category comparison.

    conda run -n mineral python explorations/real_mineralogy/run_plutonic_zero_shot.py \
        --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import (
    load_benchmark,
    real_rock_metrics,
    score_real_rock,
    split_mask,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from explorations.real_mineralogy.harvest import PLUTONIC_DIR
from explorations.real_mineralogy.plutonic import PLUTONIC_SPLIT_PATH, plutonic_benchmark

RESULTS = Path(__file__).resolve().parent / "results"
SUBSETS = ("train", "val")


def git_state() -> dict:
    def run(*args):
        return subprocess.run(args, cwd=REPO, capture_output=True, text=True).stdout
    return {"head": run("git", "rev-parse", "HEAD").strip(),
            "src_dirty": bool(run("git", "diff", "--", "src/").strip())}


@torch.no_grad()
def evaluate(head, encoder, benchmark, device, ckpt, tag) -> dict:
    """`real_rock_metrics` on the train and val rocks of `benchmark`."""
    benchmark = benchmark.subset(
        split_mask(benchmark, SUBSETS[0]) | split_mask(benchmark, SUBSETS[1])
    )
    latents = bulk_latents(encoder, benchmark, device)
    scored = score_real_rock(head, benchmark, latents)
    return real_rock_metrics(scored, benchmark, tag=tag,
                             encoder_checkpoint=ckpt.encoder_checkpoint)


def _fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    tag = args.checkpoint.parent.name
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder = encoder.to(device)
    head = ckpt.build_mode_head().to(device)

    eruption = evaluate(head, encoder, load_benchmark(), device, ckpt, tag)
    with plutonic_benchmark() as benchmark:
        # the numpy scoring path has no presence mask, so it reads the rocks
        # with every category measured: the 507 its committed results describe
        plutonic = evaluate(head, encoder, benchmark.subset(benchmark.measured.all(1)),
                            device, ckpt, tag)

    roster = pd.read_csv(PLUTONIC_SPLIT_PATH)
    in_subsets = int(roster["subset"].isin(SUBSETS).sum())
    # `real_rock_metrics` stamps the eruption benchmark's folder and freshness
    # grades; neither describes these rocks
    plutonic["benchmark"]["dir"] = str(PLUTONIC_DIR.relative_to(REPO))
    del plutonic["benchmark"]["clean_grades"]
    plutonic["benchmark"]["split"] = str(PLUTONIC_SPLIT_PATH.relative_to(REPO))
    plutonic["benchmark"]["n_left_out_unmeasured_category"] = (
        in_subsets - plutonic["benchmark"]["n_with_chemistry_and_modes"]
    )

    out = RESULTS / tag / "plutonic_zero_shot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "code": git_state(),
        "subsets": list(SUBSETS),
        "plutonic": plutonic,
        "eruption_v2": eruption,
    }, indent=2))
    print(f"wrote {out}")

    for name, m in (("plutonic", plutonic), ("eruption-v2", eruption)):
        print(f"{name}: {m['headline']['n']} rocks, joint nll "
              f"{m['headline']['joint_nll']:.3f} against climatology "
              f"{m['climatology']['joint_nll']:.3f}, "
              f"{m['headline']['mean_composition_size']:.2f} categories per rock")
    for p, e in zip(plutonic["per_group"], eruption["per_group"]):
        print(f"{p['group']}: share with it {_fmt(p['observed_rate'])} vs "
              f"{_fmt(e['observed_rate'])}, auroc {_fmt(p['auroc'])} vs "
              f"{_fmt(e['auroc'])}, share absolute error "
              f"{_fmt(p['magnitude_mae'], 3)} vs {_fmt(e['magnitude_mae'], 3)}")


if __name__ == "__main__":
    main()
