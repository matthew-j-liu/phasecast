"""
One-rock inference card: bulk chemistry in, full posterior out, truth overlaid.

Single canvas from a joint-heads checkpoint for one val-roster bulk, with no
teacher forcing anywhere: assemblages sample from the model's own presence
probabilities, and the condition posteriors condition on those sampled
mineral lists.

    python scripts/view/inference_card.py \\
        --checkpoint checkpoints/joint_heads_seed0/heads_best.pt

Omitting `--bulk` picks by per-bulk mode NLL: `--pick median` (default, the
representative rock) or `--pick best` (internal showcase). The figure lands
in `results/<checkpoint-dir-name>/` unless `--out-dir` says otherwise.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import torch

from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import hurdle_nll_rows
from cnmineral.eval.summary_cards import (
    inference_card,
    pick_bulk_by_nll,
    sample_assemblages,
)
from cnmineral.models.cond_head import COND_AXES
from cnmineral.training.joint_heads import JointHeadCheckpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to a joint-heads checkpoint (.pt)")
    parser.add_argument("--bulk", type=int, default=None,
                        help="val-roster bulk id (default: chosen by --pick)")
    parser.add_argument("--pick", choices=("median", "best"), default="median",
                        help="bulk selection by per-bulk mode NLL when --bulk absent")
    parser.add_argument("--rung", type=int, default=50,
                        help="melt-fraction rung for the truth rows")
    parser.add_argument("--n-samples", type=int, default=512,
                        help="posterior samples for the assemblage + condition panels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output dir (default results/<checkpoint dir>)")
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
    latents = bulk_latents(encoder, ds, torch.device("cpu"))
    row_bulk = ds.row_bulk_idx.numpy()
    rung = ds.rung.numpy()
    with torch.no_grad():
        logits_all, alpha_all = mode_head(latents)

    if args.bulk is None:
        nll_rows = hurdle_nll_rows(
            logits_all[row_bulk], alpha_all[row_bulk], ds.presence, ds.magnitude
        )["joint"]
        bpos = pick_bulk_by_nll(nll_rows, row_bulk, rung, args.rung, args.pick)
        bulk = int(ds.bulk_ids[bpos])
        picked = f"{args.pick}-NLL val bulk"
        print(f"picked bulk {bulk} ({args.pick} per-bulk mode NLL at "
              f"rung {args.rung})")
    else:
        bulk = args.bulk
        hits = np.flatnonzero(ds.bulk_ids == bulk)
        if len(hits) == 0:
            raise SystemExit(f"bulk {bulk} is not on the val roster")
        bpos = int(hits[0])
        picked = "bulk chosen by hand"
    rows = np.flatnonzero((row_bulk == bpos) & (rung == args.rung))
    if len(rows) == 0:
        raise SystemExit(f"bulk {bulk} has no rows at rung {args.rung}")

    rng = np.random.default_rng(args.seed)
    p = torch.sigmoid(logits_all[bpos]).numpy()
    pred_modes = sample_assemblages(p, alpha_all[bpos].numpy(), args.n_samples, rng)

    supports = torch.as_tensor((pred_modes > 0).astype(np.float32))
    with torch.no_grad():
        log_w, mu, sigma = cond_head(
            latents[bpos].expand(len(supports), -1), supports
        )
    w = np.exp(log_w.numpy())
    mu_n, sigma_n = mu.numpy(), sigma.numpy()
    cond_samples = np.zeros((len(supports), len(COND_AXES)))
    for a in range(len(COND_AXES)):
        for s in range(len(supports)):
            ci = rng.choice(w.shape[-1], p=w[s, a] / w[s, a].sum())
            cond_samples[s, a] = rng.normal(mu_n[s, a, ci], sigma_n[s, a, ci])

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("results") / args.checkpoint.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"{ds.bulk_qap_names[bpos]} (bulk {bulk}), {args.rung}%-melt rung -- "
        f"truth ({len(rows)} draws) vs prediction ({args.n_samples} samples); "
        f"{picked}; no teacher forcing anywhere"
    )
    inference_card(
        ds.bulk_11[rows].numpy(), ds.magnitude[rows].numpy(), pred_modes,
        cond_samples, ds.cond_target[rows].numpy(),
        title, out_dir / f"inference_card_bulk{bulk}.png",
    )


if __name__ == "__main__":
    main()
