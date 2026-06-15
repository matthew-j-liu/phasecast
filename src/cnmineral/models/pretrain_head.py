"""
SSL pretrain decoder head.

A small MLP that reads the encoder's pooled latent `[B, d_model]` and
predicts a `frac` value for every position in the 22-token vocabulary.
The pretrain loss reads its targets at the originally-masked positions
only; the other 20 predicted values are unsupervised side outputs that
receive no gradient through the loss.
"""

import torch
from torch import nn
from cnmineral.data.tokens import N_TOKENS


class PretrainHead(nn.Module):
    """
    Pooled-latent decoder over the 22-token vocabulary.

    Architecture: `d_model -> d_model // 2 -> N_TOKENS`, two linear
    layers with a single GELU non-linearity between them.  Output is
    identity (unconstrained); MSE pulls predictions toward `frac` in
    [0, 1].

    Parameters
    ----------
    d_model
        Latent dim of the encoder feeding this head.  Should match the
        encoder's d_model.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = d_model // 2
        self.fc1 = nn.Linear(d_model, hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden, N_TOKENS)

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
