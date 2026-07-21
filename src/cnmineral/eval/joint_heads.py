"""One-stop interrogation of a joint-heads checkpoint.

Loads a `JointHeadCheckpoint` once, rebuilds the valtest val-roster dataset
from its stamped `label_config`, computes one pooled latent per bulk, and
dispatches to the per-head eval engines (`eval.mode_head`, `eval.comp_head`,
`eval.cond_head`), each of which writes figures + `metrics.json` under
`runs/{mode_head,comp_head,cond_head}/<checkpoint dir name>/`. The
expensive setup (dataset rebuild + encoder forward) happens exactly once
regardless of how many heads are requested.
"""

from pathlib import Path

import torch

from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.eval.comp_head import run_comp_head_eval
from cnmineral.eval.cond_head import run_cond_head_eval
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import run_mode_head_eval
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint

HEAD_EVALS = {
    "mode": run_mode_head_eval,
    "comp": run_comp_head_eval,
    "cond": run_cond_head_eval,
}


def val_dataset_and_latents(
    ckpt: JointHeadCheckpoint, device: torch.device
) -> tuple[JointHeadDataset, torch.Tensor]:
    """Rebuild the valtest val-roster dataset + pooled latents for `ckpt`."""
    lc = ckpt.label_config
    if "valtest_corpus_dir" not in lc or "comp_eps" not in lc:
        raise ValueError(
            "checkpoint lacks valtest_corpus_dir/comp_eps in label_config; "
            "retrain with the current joint trainer"
        )
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    dataset = JointHeadDataset(
        corpus_dir=lc["valtest_corpus_dir"],
        rungs=tuple(lc["rungs"]),
        presence_cutoff=lc["presence_cutoff"],
        comp_eps=lc["comp_eps"],
        bulk_ids=load_valtest_split("val"),
    )
    latents = bulk_latents(encoder, dataset, device)
    return dataset, latents


def run_joint_eval(
    checkpoint_path: Path | str,
    heads: tuple[str, ...] = ("mode", "comp", "cond"),
    device: torch.device | None = None,
) -> dict[str, dict]:
    """Interrogate the requested heads of one joint checkpoint.

    Returns `{head_name: metrics}` for the requested heads; each engine also
    writes its figures + `metrics.json` under `runs/<head>_head/<tag>/`.
    """
    unknown = set(heads) - set(HEAD_EVALS)
    if unknown:
        raise ValueError(
            f"unknown heads {sorted(unknown)}; expected among {list(HEAD_EVALS)}"
        )
    checkpoint_path = Path(checkpoint_path)
    ckpt = JointHeadCheckpoint.load(checkpoint_path)
    device = device or pick_device()
    tag = checkpoint_path.parent.name
    print(f"device:  {device}")
    print(f"heads:   {checkpoint_path}")
    print(f"encoder: {ckpt.encoder_checkpoint}")
    dataset, latents = val_dataset_and_latents(ckpt, device)
    print(f"bulks={len(dataset.bulk_ids)} rows={len(dataset)} (valtest val roster)")

    return {
        name: engine(ckpt, dataset, latents, device=device, tag=tag)
        for name, engine in HEAD_EVALS.items()
        if name in heads
    }
