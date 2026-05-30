"""Pretrain head for SSL: vocab-decoder MLP over the POOLED encoder latent.

Reads `pooled ∈ R^{B × d_model}` from `encoder.Encoder` (default forward,
masked-mean pooled over UNMASKED tokens) and predicts a scalar `frac` per
position in the V2 cation vocabulary. The pretrain loss applies MSE only at
positions that were originally masked (zeroed before the encoder forward);
the head's predictions at unmasked positions are unsupervised side outputs.

This shape avoids the kingcrab padding-mask collision: masked positions are
zeroed before encoder forward and kingcrab pads them out cleanly. The head
reconstructs the full chemistry from the pooled latent of the survivors;
gradients flow back through the encoder via the masked positions' loss
terms naturally (the encoder's pooled output IS the only input to the head).

Discarded after pretraining — the encoder's weights are what's persisted.
"""

import torch
import torch.nn as nn


class PretrainHead(nn.Module):
    """Vocab-decoder MLP: d_model -> d_model//2 -> n_tokens.

    Parameters
    ----------
    d_model : int
        Encoder pooled-output dimension.
    n_tokens : int
        Vocabulary size (22 for V2: 9 majors + 13 traces).
    hidden : int | None
        Hidden width; defaults to d_model // 2.
    dropout : float
        Dropout between the two layers.
    """

    def __init__(self, d_model: int, n_tokens: int, hidden: int | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        hidden = hidden if hidden is not None else (d_model // 2)
        self.n_tokens = n_tokens
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tokens),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pooled : FloatTensor [B, d_model]
            Encoder output (masked-mean pooled over the unmasked / present tokens).

        Returns
        -------
        FloatTensor [B, n_tokens]
            Predicted frac per cation slot. Caller masks to the masked
            positions for loss computation.
        """
        return self.net(pooled)
