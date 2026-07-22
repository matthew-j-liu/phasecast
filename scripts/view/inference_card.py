"""
One-rock inference card: bulk chemistry in, full posterior out, truth overlaid.

Single canvas from a joint-heads checkpoint for one val-roster bulk: input
oxide bars, assemblage posterior vs the corpus draws, and T/P/H2O
posteriors over a faint histogram of all the bulk's truth draws.

    python scripts/view/inference_card.py \\
        --checkpoint checkpoints/joint_heads_seed0/heads_best.pt --bulk 871826

Omitting `--bulk` uses the first val-roster bulk. The figure lands in
`runs/summary/<checkpoint-dir-name>/` unless `--out-dir` says otherwise.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import torch

from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.summary_cards import inference_card, sample_assemblages
from cnmineral.models.cond_head import COND_AXES
from cnmineral.training.joint_heads import JointHeadCheckpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to a joint-heads checkpoint (.pt)")
    parser.add_argument("--bulk", type=int, default=None,
                        help="val-roster bulk id (default: first on the roster)")
    parser.add_argument("--rung", type=int, default=50,
                        help="melt-fraction rung for the truth rows")
    parser.add_argument("--n-samples", type=int, default=512,
                        help="posterior samples for the assemblage panel")
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
    bulk = int(ds.bulk_ids[0]) if args.bulk is None else args.bulk
    hits = np.flatnonzero(ds.bulk_ids == bulk)
    if len(hits) == 0:
        raise SystemExit(f"bulk {bulk} is not on the val roster")
    bpos = int(hits[0])
    rows = np.flatnonzero(
        (ds.row_bulk_idx.numpy() == bpos) & (ds.rung.numpy() == args.rung)
    )
    if len(rows) == 0:
        raise SystemExit(f"bulk {bulk} has no rows at rung {args.rung}")

    latents = bulk_latents(encoder, ds, torch.device("cpu"))
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        logits, mode_alpha = mode_head(latents[bpos : bpos + 1])
        log_w, mu, sigma = cond_head(
            latents[bpos].expand(len(rows), -1), ds.presence[rows]
        )
    p = torch.sigmoid(logits)[0].numpy()
    pred_modes = sample_assemblages(p, mode_alpha[0].numpy(), args.n_samples, rng)

    w = np.exp(log_w.numpy())
    mu, sigma = mu.numpy(), sigma.numpy()
    per_row = 4
    cond_samples = np.zeros((len(rows) * per_row, len(COND_AXES)))
    for a in range(len(COND_AXES)):
        for r in range(len(rows)):
            comp_idx = rng.choice(w.shape[-1], size=per_row,
                                  p=w[r, a] / w[r, a].sum())
            cond_samples[r * per_row : (r + 1) * per_row, a] = rng.normal(
                mu[r, a, comp_idx], sigma[r, a, comp_idx]
            )

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("runs/summary") / args.checkpoint.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"{ds.bulk_qap_names[bpos]} (bulk {bulk}), {args.rung}%-melt rung -- "
        f"truth ({len(rows)} draws) vs prediction ({args.n_samples} samples)"
    )
    inference_card(
        ds.bulk_11[rows].numpy(), ds.magnitude[rows].numpy(), pred_modes,
        cond_samples, ds.cond_target[rows].numpy(), ds.presence_cutoff,
        title, out_dir / f"inference_card_bulk{bulk}.png",
    )


if __name__ == "__main__":
    main()
