"""
Pooled-latent extraction for downstream probes + UMAP.

Runs an `Encoder` over a dataset in eval mode (no SSL masking, no dropout)
and returns the per-sample pooled latents stacked as `[N, d_model]`.  No
masking: probes see what the encoder produces from the unaltered input,
mirroring how inference-time consumers (k-NN lookup, classifier heads)
would call the encoder.

Four entry points:

- `extract_latents(encoder, dataset, ...)` for an in-memory encoder, used
  by tests and by the CLI after it has loaded the checkpoint.
- `bulk_latents(encoder, dataset, ...)` for the mineralogy datasets, which
  carry one input vector per bulk (`bulk_src` / `bulk_frac`): one pooled
  latent per bulk, returned on the compute device (the table is tiny).
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
from cnmineral.data.tokens import REDOX_TOKEN_INDEX
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead
from cnmineral.training.build_from_config import create_encoder_and_head
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
    # single source of construction truth, incl. the token-width
    # derivation from data_kwargs.include_redox
    encoder, head = create_encoder_and_head(ckpt.config)
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


class PaddedRedoxView(Dataset):
    """Read-only view of a redox corpus with the redox token padded.

    The head-to-head condition for comparing a redox encoder against a
    22-token one: zeroing `(src, frac)` at the redox slot puts every row
    in the padded regime the encoder trained on for ~78% of the corpus.
    Wraps `__getitem__` only; consumers keep reading labels/attributes
    off the base dataset.
    """

    def __init__(self, base: Dataset) -> None:
        base_item = base[0]
        if base_item["src"].shape[0] <= REDOX_TOKEN_INDEX:
            raise ValueError(
                "PaddedRedoxView needs a redox (23-token) corpus; "
                f"got {base_item['src'].shape[0]} tokens"
            )
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        # BulkChemistryDataset.__getitem__ clones, so in-place zeroing
        # cannot touch the base tensors
        item = self.base[idx]
        item["src"][REDOX_TOKEN_INDEX] = 0
        item["frac"][REDOX_TOKEN_INDEX] = 0.0
        return item


@torch.no_grad()
def bulk_latents(
    encoder: Encoder,
    dataset,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """One pooled latent per dataset bulk, on `device`.

    `dataset` is a mineralogy dataset (`JointHeadDataset`) whose rows share
    per-bulk inputs: only `bulk_src` / `bulk_frac` are encoded, one forward
    per bulk rather than per row.
    """
    outs = []
    for start in range(0, len(dataset.bulk_ids), batch_size):
        src = dataset.bulk_src[start : start + batch_size].to(device)
        frac = dataset.bulk_frac[start : start + batch_size].to(device)
        outs.append(encoder(src=src, frac=frac))
    return torch.cat(outs)


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
