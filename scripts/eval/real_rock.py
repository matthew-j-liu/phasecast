"""Score a joint checkpoint on the real-rock benchmark, zero-shot.

  python scripts/eval/real_rock.py --checkpoint checkpoints/joint_heads_supportmag_seed0/heads_best.pt

Runs the frozen encoder + trained ModeHead over the eruption benchmark
(`data/real_mineralogy/`) with nothing fit: the 21-phase vocabulary is
coarsened onto the 7 groups real modal data resolves, and the same hurdle NLL
the corpus eval reports is assembled against measured phenocryst modes.
Figures + `metrics.json` land in `runs/real_rock/<checkpoint dir name>/`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, run_real_rock_eval
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="JointHeadCheckpoint produced by scripts/training/joint_heads.py",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    ckpt = JointHeadCheckpoint.load(checkpoint)
    device = pick_device()
    print(f"device:  {device}")
    print(f"heads:   {checkpoint}")
    print(f"encoder: {ckpt.encoder_checkpoint}")

    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    benchmark = load_benchmark()
    print(f"rocks:   {len(benchmark)} with chemistry + modes "
          f"({int(benchmark.clean.sum())} least-altered)")
    latents = bulk_latents(encoder, benchmark, device)
    run_real_rock_eval(ckpt, benchmark, latents, device=device,
                       tag=checkpoint.parent.name)


if __name__ == "__main__":
    main()
