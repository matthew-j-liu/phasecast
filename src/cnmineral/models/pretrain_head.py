"""
SSL pretrain decoder head.

A small MLP that reads the encoder's pooled latent `[B, d_model]` and
predicts a `frac` value for every position in the token vocabulary
(22, or 23 when the corpus carries the redox token).  The pretrain loss
reads its targets at the originally-masked positions only; the other
predicted values are unsupervised side outputs that receive no gradient
through the loss.
"""

import torch
from torch import nn
from cnmineral.data.tokens import N_TOKENS


class PretrainHead(nn.Module):
    """
    Pooled-latent decoder over the 22-token vocabulary.

    Architecture: `d_model -> d_model // 2 -> N_TOKENS`, two linear
    layers with a single GELU between them; identity output.  Under the
    canonical standardized-log-space objective (ADR 007) the target is
    standardized `log10(frac)`, which is unbounded in R -- a squashing
    output activation would be wrong, so identity is the correct choice,
    not a deferred one.

    The `d_model // 2` bottleneck and the identity output are fixed, not
    hyperparameters: this head is a training-time scaffold that exists
    only to shape the encoder via the reconstruction gradient, and is
    discarded at inference (only the encoder ships).

    Parameters
    ----------
    d_model
        Latent dim of the encoder feeding this head.  Should match the
        encoder's d_model.
    n_tokens
        Vocabulary width to decode.  Default matches the 22-token
        corpus; redox corpora pass 23.
    """

    def __init__(self, d_model: int, n_tokens: int = N_TOKENS) -> None:
        super().__init__()
        hidden = d_model // 2
        self.n_tokens = n_tokens
        self.fc1 = nn.Linear(d_model, hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden, n_tokens)

    def forward(self, pooled_latent: torch.Tensor) -> torch.Tensor:
        """
        Predict `frac` for every vocabulary position.

        Parameters
        ----------
        pooled_latent
            `[B, d_model]` from the encoder's masked-mean pool.

        Returns
        -------
        predicted_frac
            `[B, N_TOKENS]` predicted frac per position.  The SSL loss
            indexes this at `mask_positions` to compute MSE against
            `target_frac`; the other positions are unsupervised side
            outputs.
        """
        x = self.fc1(pooled_latent)
        x = self.activation(x)
        return self.fc2(x)
