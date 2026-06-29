"""
Log-space imputation read-out head.

A small head trained on the *frozen* SSL encoder's pooled latent `[B,
d_model]` to impute masked element values.  It is the application-slot
counterpart to `cnmineral.models.pretrain_head.PretrainHead`: same MLP shape
and the same per-element standardized log-space objective, but `PretrainHead`
is trained jointly with the encoder during SSL (it shapes the latent) while
this head trains alone on the already-frozen encoder.  Training the read-out
separately lets it size its capacity (hidden width, dropout) to the imputation
task without disturbing the latent the rest of the pipeline shares.

Outputs are in **standardized-log space** (per-element `log10(frac)`
centered/scaled by training-split statistics carried in the `target_mean`
/ `target_std` buffers).  `predict_log_frac` and `predict_frac`
de-standardize for inference and evaluation.  Training loss lives in
`cnmineral.losses.pretrain_loss.masked_element_log_mse`.
"""

import torch
from torch import nn

from cnmineral.data.tokens import N_TOKENS

_STD_FLOOR = 1e-6


class LogImputationHead(nn.Module):
    """Read-out head over the 22-token vocabulary in standardized-log space.

    Linear (`hidden=None`) or one-hidden-layer GELU MLP (`hidden=int`) from
    the pooled latent to one value per vocabulary position.  The head emits
    per-element standardized `log10(frac)`; `fit_standardization` writes the
    de-standardization buffers from the training split before training.

    Parameters
    ----------
    d_model
        Latent dim of the frozen encoder feeding this head.
    hidden
        Hidden width.  `None` builds a single linear layer; an int builds
        `d_model -> hidden -> N_TOKENS` with a GELU between.
    dropout
        Dropout after the GELU in the MLP path; ignored for the linear head.
        `0.0` disables it.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hidden: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.dropout = dropout

        if hidden is None:
            self.net = nn.Linear(d_model, N_TOKENS)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, N_TOKENS),
            )

        # Per-element log10(frac) center/scale, fit from the training split.
        # Identity until fit so an unfit head is a no-op transform.  Buffers
        # serialize in state_dict, so they travel with the checkpoint and are
        # encoder-version-agnostic.
        self.register_buffer("target_mean", torch.zeros(N_TOKENS))
        self.register_buffer("target_std", torch.ones(N_TOKENS))

    def forward(self, pooled_latent: torch.Tensor) -> torch.Tensor:
        """`[B, d_model]` -> `[B, N_TOKENS]` in standardized-log space.

        The loss reads this at `mask_positions` against the standardized-log
        target; the other positions are unsupervised side outputs.
        """
        return self.net(pooled_latent)

    def predict_log_frac(self, pooled_latent: torch.Tensor) -> torch.Tensor:
        """De-standardize the head output to `log10(frac)` per position."""
        return self.forward(pooled_latent) * self.target_std + self.target_mean

    def predict_frac(self, pooled_latent: torch.Tensor) -> torch.Tensor:
        """De-standardize and exponentiate to `frac` per position."""
        return torch.pow(10.0, self.predict_log_frac(pooled_latent))

    @torch.no_grad()
    def fit_standardization(
        self, target_mean: torch.Tensor, target_std: torch.Tensor,
    ) -> None:
        """Write the per-element `log10(frac)` center/scale buffers.

        Call once before training with statistics over the training split's
        populated rows.  `target_std` is clamped to a small floor so an
        element with near-constant log-frac cannot blow up the standardized
        target.
        """
        if target_mean.shape != (N_TOKENS,) or target_std.shape != (N_TOKENS,):
            raise ValueError(
                f"standardization stats must be [{N_TOKENS}]; got "
                f"{tuple(target_mean.shape)} / {tuple(target_std.shape)}"
            )
        self.target_mean.copy_(target_mean.to(self.target_mean))
        self.target_std.copy_(target_std.clamp_min(_STD_FLOOR).to(self.target_std))
