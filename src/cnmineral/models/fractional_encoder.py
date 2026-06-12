"""
Fractional encoder for element abundances.

Copied directly from CrabNet's kingcrab.py


Notes
-----
- The encoding discretizes ``frac`` into ``resolution`` bins before a
  lookup into a precomputed sin/cos table (``self.pe``).  The lookup
  is differentiable with respect to encoder weights downstream, even
  though it is not differentiable w.r.t. ``frac`` itself.  For SSL
  training this is fine: gradients flow back into the encoder via the
  lookup output, not into the input ``frac``.
- ``d_model`` passed to ``__init__`` is the *full* per-token dim that
  downstream code expects; the encoder internally uses ``d_model // 2``
  channels (CrabNet's convention, so two FractionalEncoders -- one
  linear, one log -- concatenate to ``d_model``).
"""

import torch
from torch import nn


class FractionalEncoder(nn.Module):
    """Encode element fractional amount using a "fractional encoding".

    This is inspired by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """

    pe: torch.Tensor 

    def __init__(self, d_model, resolution=100, log10=False):
        """Instantiate the FractionalEncoder.

        Parameters
        ----------
        d_model : int
            Model size, see paper, by default 512
        resolution : int
            Number of discretizations for the fractional prevalence encoding, by default 100
        log10 : bool
            Whether to apply a log operation to fraction prevalence encoding, by default False
        """
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10

        x = torch.linspace(
            0, self.resolution - 1, self.resolution, requires_grad=False
        ).view(self.resolution, 1)
        fraction = (
            torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False)
            .view(1, self.d_model)
            .repeat(self.resolution, 1)
        )

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer("pe", pe)

    def forward(self, x):
        """Perform the forward pass of the fractional encoding.

        Parameters
        ----------
        x : torch.tensor
            Tensor of linear spaced values based on defined resolution

        Returns
        -------
        out
            Sinusoidal expansions of elemental fractions
        """
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x)) ** 2
            x[x > 1] = 1
        x[x < 1 / self.resolution] = 1 / self.resolution
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out
