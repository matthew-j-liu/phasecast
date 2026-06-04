"""V3 SSL pretrain decoder head.

A small MLP that reads the encoder's pooled latent `[B, d_model]` and
predicts a `frac` value for every position in the 22-token vocabulary.
The pretrain loss reads its targets at the originally-masked positions
only; the other 20 predicted values are unsupervised side outputs that
receive no gradient through the loss.

This is the Option H decoder design from the v2 bug-fix.  The v2 original
read per-token outputs from the encoder (pre-pool), which collided with
the padding-mask zero-out and silently nuked encoder gradients.  The
pooled-latent design avoids that path entirely.

See `../crabnet-mineral-v2-legacy/src/cnmineral/models/pretrain_head.py`
for the v2 reference implementation.
"""

import torch
from torch import nn

from cnmineral.data.tokens import N_TOKENS


class PretrainHead(nn.Module):
    """Pooled-latent decoder over the 22-token vocabulary.

    Architecture: `d_model -> hidden -> N_TOKENS`, two linear layers
    with a single GELU non-linearity between them.

    Parameters
    ----------
    d_model
        Latent dim of the encoder feeding this head.  Should match the
        encoder's d_model.
    hidden
        Bottleneck dim.  Defaults to `d_model // 2`.
    output_activation
        Optional non-linearity applied to the output.  Default `None`
        keeps outputs unconstrained; `"sigmoid"` constrains to
        `[0, 1]` matching the `frac` target range exactly.

    Notes
    -----
    Output activation choice has subtle consequences for SSL training
    dynamics.  Without one, the model can predict `frac` outside
    [0, 1] and the MSE pulls it back; with sigmoid, outputs are
    physically meaningful at any training step but gradients saturate
    when the model predicts confidently wrong.  v2 used no output
    activation; start there in v3 and revisit if calibration is poor.
    """

    def __init__(
        self,
        d_model: int = 512,
        hidden: int | None = None,
        output_activation: str | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden if hidden is not None else d_model // 2
        self.hidden_dim = hidden
        self.fc1 = nn.Linear(d_model, hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden, N_TOKENS)

        if output_activation is None:
            self.output_activation: nn.Module = nn.Identity()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            raise ValueError(
                f"unsupported output_activation: {output_activation!r}"
            )

    def forward(self, pooled_latent: torch.Tensor) -> torch.Tensor:
        """Predict `frac` for every vocabulary position.

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
        x = self.fc2(x)
        return self.output_activation(x)
