"""Sim-to-real adaptation on the frozen ModeHead, in coarsened group space.

The heads are trained on MAGEMin labels and transfer with the ranking intact
but the scale wrong -- Pearson r 0.5-0.76 against negative R2 for four of
seven groups.  This module is the correction fit on real labels, and it acts
on the seven groups real modal data resolves rather than the model's 21
phases: that is where the labels live, where the residuals were measured, and
where the three tiers nest exactly.  Encoder and heads stay frozen; fitting
lives in `cnmineral.training.real_rock_recalibrator`.

The tiers are `vector` in `matrix`, so the ~21-parameter per-group affine is
the degenerate case of the coupled layer rather than a rival to it.  `matrix`
is the canonical layer.  Three extensions were built, measured and removed: an
MLP on top of `matrix` whose capacity never repaid itself at any labelled
budget (journal 2026-09-02), a pure-location silica slope on px/bi/q magnitudes (+0.007 [-0.004, +0.016]
nats, journal 2026-08-06) and a two-parameter precision contamination on top
of `matrix`, whose 0.3-nat margin turned out to be the cost of pinning
amphibole's correction at zero (journal 2026-09-03).
"""

import torch
from torch import nn

from ..data.phase_vocab import GROUPS

TIERS: tuple[str, ...] = ("vector", "matrix")


class RealRockRecalibrator(nn.Module):
    """Correction on `(p7 logits, alpha7)` fit against measured modes.

    Presence is corrected in logit space, which is where a Platt scaling
    lives; magnitude is corrected multiplicatively on the Dirichlet
    concentrations.  Multiplicative matters: coarsening leaves exact zeros
    wherever a group has no member phase on the support, `alpha * exp(c)`
    keeps those zeros, and an additive correction in log space would turn
    them into `-inf`.

    Every group's correction is fit; weight decay shrinks an unidentifiable
    one toward zero on its own.  Whether a fitted correction means anything
    is the mapping audit's question, not the layer's: amphibole's magnitude
    band straddles zero under plausible densities (-0.026..+0.050 on
    production, `explorations/real_rock_mapping`), so its fitted value reads
    the density assumption and is not interpretable.
    """

    def __init__(self, tier: str = "vector") -> None:
        super().__init__()
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
        n = len(GROUPS)
        self.tier = tier

        # identity at initialization, so an unfit recalibrator reproduces the
        # zero-shot number exactly and any gain reads as nats recovered.
        self.presence_scale = nn.Parameter(torch.ones(n))
        self.presence_offset = nn.Parameter(torch.zeros(n))
        self.magnitude_offset = nn.Parameter(torch.zeros(n))

        if tier == "matrix":
            self.presence_mix = nn.Parameter(torch.zeros(n, n))
            self.magnitude_mix = nn.Parameter(torch.zeros(n, n))

    def forward(
        self,
        logits7: torch.Tensor,
        alpha7: torch.Tensor,
        log_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """([B, 7] presence logits, [B, 7] alpha) -> corrected (logits, alpha, log_w).

        The returned log_w is the input log_w passed through unchanged.

        A mixture head passes alpha7 [B, K, 7] with gate log-weights: the one
        per-group correction is applied identically to every component and the
        gate is left alone, so recalibration capacity stays fixed across
        single-Dirichlet and mixture arms (the contrast measures the mixture,
        not extra correction parameters). The coupled tiers' `log_alpha`
        feature then reads the gate-weighted mean concentration -- a [B, 7]
        summary, not a per-component input.
        """
        if alpha7.dim() == 3:
            alpha_feat = (log_w.exp().unsqueeze(-1) * alpha7).sum(1)
        else:
            alpha_feat = alpha7
        # An off-support group has no concentration, so its feature is the
        # neutral 0 rather than log(tiny).  Clamping instead would hand the
        # coupled tiers a -27 spike on exactly the groups that carry no
        # information, and they diverge on it.
        log_alpha = torch.where(
            alpha_feat > 0, alpha_feat.clamp_min(1e-12).log(),
            torch.zeros_like(alpha_feat),
        )

        d_presence = self.presence_scale * logits7 + self.presence_offset
        d_magnitude = self.magnitude_offset.expand_as(alpha_feat)

        if self.tier == "matrix":
            d_presence = d_presence + logits7 @ self.presence_mix
            d_magnitude = d_magnitude + log_alpha @ self.magnitude_mix

        correction = d_magnitude.exp()
        if alpha7.dim() == 3:
            correction = correction.unsqueeze(1)
        alpha = alpha7 * correction
        return d_presence, alpha, log_w

    def extra_repr(self) -> str:
        return f"tier={self.tier}"
