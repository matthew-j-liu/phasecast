"""Sim-to-real adaptation on the frozen ModeHead, in coarsened group space.

The heads are trained on MAGEMin labels and transfer with the ranking intact
but the scale wrong -- Pearson r 0.5-0.76 against negative R2 for four of
seven groups.  This module is the correction fit on real labels, and it acts
on the seven groups real modal data resolves rather than the model's 21
phases: that is where the labels live, where the residuals were measured, and
where the three tiers nest exactly.  Encoder and heads stay frozen; fitting
lives in `cnmineral.training.real_rock_adapter`.

The tiers are `diagonal` in `linear` in `mlp`, so the ~23-parameter per-group
affine is the degenerate case of the MLP rather than a rival to it.
"""

import torch
from torch import nn

from ..data.phase_vocab import GROUP_INDEX, GROUPS

TIERS: tuple[str, ...] = ("diagonal", "linear", "mlp")

# The residual-vs-silica read gives a slope to q and amp presence and nothing
# else: magnitude residuals are flat against silica for every group, and so is
# presence outside these two.  Fitting the other twelve slopes would spend
# parameters on noise.
SIO2_PRESENCE_GROUPS: tuple[str, ...] = ("q", "amp")

# SiO2 enters centred and scaled so the slope is O(1) and the intercepts stay
# interpretable as the per-group offsets that were measured.
SIO2_CENTER: float = 65.0
SIO2_SCALE: float = 10.0


class RealRockAdapter(nn.Module):
    """Correction on `(p7 logits, alpha7)` fit against measured modes.

    Presence is corrected in logit space, which is where a Platt scaling
    lives; magnitude is corrected multiplicatively on the Dirichlet
    concentrations.  Multiplicative matters: coarsening leaves exact zeros
    wherever a group has no member phase on the support, `alpha * exp(c)`
    keeps those zeros, and an additive correction in log space would turn
    them into `-inf`.

    `pin_amp_magnitude` holds amphibole's magnitude correction at zero rather
    than fitting it.  The mapping audit's Monte Carlo band straddles zero for
    that one group and no density recovers the sign, so a fitted value there
    is reading the mapping's uncertainty, not the model's error.  The pin
    applies at every tier, not just to the constant: a coupled correction
    reached through the mixing matrix would be the same unidentifiable
    quantity in a different parameterization.
    """

    def __init__(
        self,
        tier: str = "diagonal",
        *,
        hidden: int = 32,
        pin_amp_magnitude: bool = True,
    ) -> None:
        super().__init__()
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
        n = len(GROUPS)
        self.tier = tier
        self.pin_amp_magnitude = pin_amp_magnitude

        # identity at initialization, so an unfit adapter reproduces the
        # zero-shot number exactly and any gain reads as nats recovered.
        self.presence_scale = nn.Parameter(torch.ones(n))
        self.presence_offset = nn.Parameter(torch.zeros(n))
        self.magnitude_offset = nn.Parameter(torch.zeros(n))
        self.presence_sio2 = nn.Parameter(torch.zeros(len(SIO2_PRESENCE_GROUPS)))
        self.register_buffer(
            "sio2_slots",
            torch.tensor([GROUP_INDEX[g] for g in SIO2_PRESENCE_GROUPS]),
        )
        self.register_buffer(
            "magnitude_free",
            torch.tensor([
                0.0 if (pin_amp_magnitude and g == "amp") else 1.0 for g in GROUPS
            ]),
        )

        if tier in ("linear", "mlp"):
            self.presence_mix = nn.Parameter(torch.zeros(n, n))
            self.magnitude_mix = nn.Parameter(torch.zeros(n, n))
        if tier == "mlp":
            self.trunk = nn.Sequential(
                nn.Linear(2 * n + 1, hidden), nn.GELU(), nn.Linear(hidden, 2 * n)
            )
            # zero the last layer so the MLP branch starts silent and the
            # module is exactly its `linear` tier at initialization.
            nn.init.zeros_(self.trunk[-1].weight)
            nn.init.zeros_(self.trunk[-1].bias)

    def forward(
        self, logits7: torch.Tensor, alpha7: torch.Tensor, sio2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """([B, 7] presence logits, [B, 7] alpha, [B] SiO2 wt%) -> corrected pair."""
        x = ((sio2 - SIO2_CENTER) / SIO2_SCALE).unsqueeze(-1)
        # An off-support group has no concentration, so its feature is the
        # neutral 0 rather than log(tiny).  Clamping instead would hand the
        # coupled tiers a -27 spike on exactly the groups that carry no
        # information, and they diverge on it.
        log_alpha = torch.where(
            alpha7 > 0, alpha7.clamp_min(1e-12).log(), torch.zeros_like(alpha7)
        )

        d_presence = self.presence_scale * logits7 + self.presence_offset
        d_presence = d_presence.index_add(
            -1, self.sio2_slots, x * self.presence_sio2
        )
        d_magnitude = self.magnitude_offset.expand_as(alpha7)

        if self.tier in ("linear", "mlp"):
            d_presence = d_presence + logits7 @ self.presence_mix
            d_magnitude = d_magnitude + log_alpha @ self.magnitude_mix
        if self.tier == "mlp":
            delta = self.trunk(torch.cat([logits7, log_alpha, x], dim=-1))
            n = len(GROUPS)
            d_presence = d_presence + delta[..., :n]
            d_magnitude = d_magnitude + delta[..., n:]

        return d_presence, alpha7 * (d_magnitude * self.magnitude_free).exp()

    def extra_repr(self) -> str:
        return f"tier={self.tier}, pin_amp_magnitude={self.pin_amp_magnitude}"
