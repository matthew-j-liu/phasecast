"""
Scenario view: an ambiguous rock's top-2 predicted assemblages + conditions.

Samples assemblages from ModeHead, groups them by exact phase set, and
shows the two dominant scenarios with their probabilities, plus CondHead's
(T, P) posterior conditioned on each scenario's support.

    python scripts/view/scenario_view.py \\
        --checkpoint checkpoints/joint_heads_seed0/heads_best.pt

Omitting `--bulk` auto-picks the val-roster bulk whose truth draws split
most evenly between two support patterns (the most measurably ambiguous
rock). The figure lands in `runs/summary/<checkpoint-dir-name>/` unless
`--out-dir` says otherwise.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import torch

from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.data.phase_vocab import SOLID_PHASES
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.summary_cards import (
    pick_ambiguous_bulk,
    sample_assemblages,
    scenario_figure,
    top_support_patterns,
)
from cnmineral.models.cond_head import COND_AXES
from cnmineral.training.joint_heads import JointHeadCheckpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to a joint-heads checkpoint (.pt)")
    parser.add_argument("--bulk", type=int, default=None,
                        help="val-roster bulk id (default: auto-pick most ambiguous)")
    parser.add_argument("--rung", type=int, default=50,
                        help="melt-fraction rung for the ambiguity scan")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="assemblage samples for scenario extraction")
    parser.add_argument("--n-cond-samples", type=int, default=400,
                        help="condition samples per scenario")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output dir (default runs/summary/<checkpoint dir>)")
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    mode_head = ckpt.build_mode_head()
    cond_head = ckpt.build_cond_head()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.eval()

    ds = JointHeadDataset(
        corpus_dir=ckpt.label_config["valtest_corpus_dir"],
        rungs=tuple(ckpt.label_config["rungs"]),
        presence_cutoff=ckpt.label_config["presence_cutoff"],
        comp_eps=ckpt.label_config["comp_eps"],
        bulk_ids=load_valtest_split("val"),
    )
    if args.bulk is None:
        bulk, score = pick_ambiguous_bulk(
            ds.presence.numpy(), ds.row_bulk_idx.numpy(), ds.rung.numpy(),
            ds.bulk_ids, args.rung,
        )
        if score == 0.0:
            print("warning: every val bulk is single-pattern at this rung; "
                  "using the first roster bulk")
        print(f"auto-picked bulk {bulk} "
              f"(second-pattern share of truth draws: {score:.2f})")
    else:
        bulk = args.bulk
    hits = np.flatnonzero(ds.bulk_ids == bulk)
    if len(hits) == 0:
        raise SystemExit(f"bulk {bulk} is not on the val roster")
    bpos = int(hits[0])

    latents = bulk_latents(encoder, ds, torch.device("cpu"))
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        logits, mode_alpha = mode_head(latents[bpos : bpos + 1])
    p = torch.sigmoid(logits)[0].numpy()
    sampled = sample_assemblages(p, mode_alpha[0].numpy(), args.n_samples, rng)
    scenarios = top_support_patterns(sampled, k=2)
    for i, sc in enumerate(scenarios):
        names = [n for n, keep in zip(SOLID_PHASES, sc["pattern"]) if keep]
        print(f"scenario {i + 1}: share {sc['share']:.2f}  phases {names}")

    t_idx, p_idx = COND_AXES.index("T"), COND_AXES.index("P_kbar")
    tp_samples = []
    with torch.no_grad():
        for sc in scenarios:
            support = torch.as_tensor(
                sc["pattern"].astype(np.float32)
            ).unsqueeze(0)
            log_w, mu, sigma = cond_head(latents[bpos : bpos + 1], support)
            w = np.exp(log_w[0].numpy())
            mu, sigma = mu[0].numpy(), sigma[0].numpy()
            tp = np.zeros((args.n_cond_samples, 2))
            for col, a in enumerate((t_idx, p_idx)):
                comp_idx = rng.choice(
                    w.shape[-1], size=args.n_cond_samples, p=w[a] / w[a].sum()
                )
                tp[:, col] = rng.normal(mu[a, comp_idx], sigma[a, comp_idx])
            tp_samples.append(tp)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("runs/summary") / args.checkpoint.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"{ds.bulk_qap_names[bpos]} (bulk {bulk}) -- top-2 predicted "
        f"assemblages from {args.n_samples} posterior samples"
    )
    scenario_figure(
        scenarios, tp_samples, title,
        out_dir / f"scenario_view_bulk{bulk}.png",
    )


if __name__ == "__main__":
    main()
