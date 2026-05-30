"""Masked-element modeling (MEM) loss + random-mask sampler for SSL pretrain.

Convention (revised after the kingcrab-padding-collision bug fix): to mask a
token we zero out BOTH `src` and `frac` so it is indistinguishable from an
absent token. kingcrab pads them out cleanly via its existing frac>0 logic.
Only two input states remain:

    absent / masked: src=0, frac=0
    present (real): src=atomic_num, frac>0

The encoder has no way to know whether a zeroed position was originally
populated. The PretrainHead (predict-all-from-pooled-latent) reconstructs the
full chemistry from the pooled latent of the surviving tokens; the loss
applies only at the positions we know were masked (caller tracks the mask).
"""

import torch
import torch.nn.functional as F


def sample_mask(
    src: torch.Tensor,
    frac: torch.Tensor,
    mask_p: float,
) -> torch.Tensor:
    """Sample a random Boolean mask over POPULATED tokens.

    Only tokens with src > 0 AND frac > 0 are eligible for masking; padding
    positions are never selected. Each eligible token is masked with
    probability `mask_p` independently.

    Uses the default per-device RNG (seed via `torch.manual_seed` for
    reproducibility — `torch.Generator` can't be passed across CPU/MPS).

    Parameters
    ----------
    src : LongTensor [B, T]
    frac : FloatTensor [B, T]
    mask_p : float
        Per-token mask probability among eligible positions. e.g., 0.25.

    Returns
    -------
    mask : BoolTensor [B, T]
        True at positions to mask.
    """
    populated = (src > 0) & (frac > 0)
    rand = torch.rand(src.shape, device=src.device)
    return populated & (rand < mask_p)


def apply_mask(frac: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out frac at masked positions; src is unchanged by the caller.

    Returns a NEW tensor; does not modify in-place.
    """
    return torch.where(mask, torch.zeros_like(frac), frac)


def masked_element_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    per_token_std: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """MSE on masked positions only, optionally normalized per token.

    Parameters
    ----------
    pred : FloatTensor [B, T]
        Per-token predicted frac from PretrainHead.
    target : FloatTensor [B, T]
        Original (pre-mask) frac values.
    mask : BoolTensor [B, T]
        True at positions that were masked.
    per_token_std : FloatTensor [T] | None
        If provided, divide (pred - target) by per-token std before squaring,
        so each token contributes proportionally regardless of its absolute
        magnitude. Critical for our setup where `frac` spans ~6 orders of
        magnitude across tokens (Si ~0.24 vs Th ~1e-6) -- unnormalized MSE
        is otherwise dominated by Si/Al/Ca/Mg/Fe reconstruction. Compute std
        once at corpus level and pass it here.

    Returns
    -------
    scalar tensor
        Mean (optionally per-token-normalized) squared error over the masked
        positions. Returns 0 if mask is empty.
    """
    n_masked = mask.sum()
    if n_masked == 0:
        return pred.sum() * 0.0
    diff = pred - target
    if per_token_std is not None:
        diff = diff / per_token_std.clamp(min=eps).to(diff.device)
    diff_sq = diff.pow(2)
    return (diff_sq * mask.to(diff_sq.dtype)).sum() / n_masked.clamp(min=1).to(diff_sq.dtype)
