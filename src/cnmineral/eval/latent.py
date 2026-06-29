"""
Pooled-latent extraction for downstream probes + UMAP.

Runs an `Encoder` over a dataset in eval mode (no SSL masking, no dropout)
and returns the per-sample pooled latents stacked as `[N, d_model]`.  No
masking: probes see what the encoder produces from the unaltered input,
mirroring how inference-time consumers (k-NN lookup, classifier heads)
would call the encoder.

Three entry points:

- `extract_latents(encoder, dataset, ...)` for an in-memory encoder, used
  by tests and by the CLI after it has loaded the checkpoint.
- `load_encoder_and_head_from_checkpoint(path)` rebuilds both the `Encoder`
  and the `PretrainHead` from a checkpoint's stamped config and applies the
  state dicts.  It is the implementation base for the head-less loader below;
  callers that only need pooled latents go through that thinner entry point.
- `load_encoder_from_checkpoint(path)` is the head-less variant for probes
  and UMAP, which only read pooled latents.  Both return in eval mode on
  CPU; the caller `.to(device)`s.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead
from cnmineral.training.checkpoint import Checkpoint


# Throughput-only knob for the corpus forward pass that produces
# `[N, d_model]` pooled latents.  Output is identical regardless of
# value; only wall time and per-batch device memory change.
_EXTRACT_BATCH_SIZE = 512


def load_encoder_and_head_from_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[Encoder, PretrainHead, Checkpoint]:
    """Rebuild the `Encoder` + `PretrainHead` from a checkpoint's config.

    Returns both models (eval mode, CPU) plus the loaded `Checkpoint`
    so the caller can pull `epoch`, `val_loss`, `config`, etc.  The head is
    the SSL masked-frac predictor; this is the implementation base for the
    head-less `load_encoder_from_checkpoint`, which discards it.
    """
    ckpt = Checkpoint.load(checkpoint_path)
    encoder = Encoder(
        d_model=ckpt.config["d_model"],
        n_layers=ckpt.config["n_layers"],
        n_heads=ckpt.config["n_heads"],
        dropout=ckpt.config.get("dropout", 0.0),
        frac_resolution=ckpt.config.get("frac_resolution", 5000),
    )
    head = PretrainHead(d_model=ckpt.config["d_model"])
    ckpt.apply_to(encoder, head)
    encoder.eval()
    head.eval()
    return encoder, head, ckpt


def load_encoder_from_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[Encoder, Checkpoint]:
    """Head-less variant for probes / UMAP, which only read pooled latents."""
    encoder, _head, ckpt = load_encoder_and_head_from_checkpoint(checkpoint_path)
    return encoder, ckpt


@torch.no_grad()
def extract_latents(
    encoder: Encoder,
    dataset: Dataset,
    *,
    indices: list[int] | np.ndarray | None = None,
    batch_size: int = _EXTRACT_BATCH_SIZE,
    device: torch.device | str = "cpu",
    num_workers: int = 0,
) -> torch.Tensor:
    """Run `encoder` over `dataset` (or `Subset(dataset, indices)`) and stack
    pooled latents as `[N, d_model]` on CPU.

    Parameters
    ----------
    encoder
        An `Encoder` instance, already loaded with weights.  Will be moved
        to `device` and toggled to eval mode here.
    dataset
        Anything yielding `{"src": [T] long, "frac": [T] float}` dicts.
    indices
        Optional subset of dataset indices to encode.  None encodes the
        full dataset.
    batch_size
        Forward-pass batch size.  Larger reduces wall time but costs
        device memory.  The 512 default fits the canonical encoder
        comfortably on MPS.
    device
        Where to run the forward pass.  Output latents are returned on
        CPU regardless.
    num_workers
        DataLoader worker count.  Defaults to 0 because the dataset
        constructor preloads everything into tensors -- worker forks
        would just copy memory.

    Returns
    -------
    torch.Tensor
        Shape `[N, d_model]`, dtype float32, on CPU.  N matches
        `len(indices)` if provided, else `len(dataset)`.
    """
    encoder.eval()
    encoder.to(device)

    target = (
        Subset(dataset, list(indices)) if indices is not None else dataset
    )
    loader = DataLoader(
        target,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    chunks: list[torch.Tensor] = []
    for batch in loader:
        src = batch["src"].to(device)
        frac = batch["frac"].to(device)
        pooled = encoder(src=src, frac=frac)
        chunks.append(pooled.detach().cpu())
    return torch.cat(chunks, dim=0)
