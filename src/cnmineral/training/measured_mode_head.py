"""Fit a `MeasuredModeHead` on measured modal labels over the frozen latent.

The objective is `recalibrator_nll` -- the same hurdle NLL the recalibrator
minimizes and the real-rock ledger reports -- so the two arms' levels are
directly differenceable.  Deliberately not `losses.mode_loss.mode_head_loss`,
whose presence BCE is mean-reduced over phases while the benchmark's is summed
over groups: the same fit scored on the two would not land on one axis.

Fitting protocol matches `fit_recalibrator` exactly (full batch, to
convergence, no per-draw early stopping, one val-selected shrinkage constant),
because a protocol difference between the arms would be indistinguishable from
a label-source difference in the result.
"""

from dataclasses import replace

import torch

from ..models.measured_mode_head import MeasuredModeHead
from .real_rock_recalibrator import (
    GroupPredictions,
    recalibrator_nll,
    score_predictions,
)


def fit_measured_head(
    train: GroupPredictions,
    *,
    weight_decay: float = 1.0,
    hidden: int = 64,
    n_components: int = 2,
    epochs: int = 1500,
    lr: float = 0.03,
    seed: int = 0,
) -> MeasuredModeHead:
    """Full-batch fit on `train`, reading the latent and nothing the frozen head said.

    The magnitude branch is conditioned on the measured group presence, which is
    what the correction arm's Dirichlet also sits on (its support is built from
    the same truth by `truth_forced_support`).  Conditioning on `numeric`
    instead would hand this arm which groups happened to be *quantified* -- a
    reporting artifact of the source, not mineralogy, and one the correction arm
    never sees.

    Adam rather than AdamW, matching `fit_recalibrator`: its L2-in-the-gradient
    does saturate once the penalty dominates the adaptive normalization, but
    that only sets in above the shared grid's ceiling of 10, while decoupled
    decay at this learning rate diverges well inside the grid.  `lr` and
    `epochs` are fixed a priori at a train-loss plateau rather than tuned,
    which would hand this arm three val-selected knobs against the
    correction's one.
    """
    if train.latent is None:
        raise ValueError("fit_measured_head needs GroupPredictions.latent")
    torch.manual_seed(seed)
    dtype = train.logits7.dtype
    head = MeasuredModeHead(
        train.latent.shape[1], hidden=hidden, n_components=n_components,
    ).to(dtype)

    # Standardized on the drawn budget alone, never on the full roster: the
    # x-axis is only honest if undrawn rocks contribute nothing at all, and
    # that includes label-free input statistics.
    head.latent_mean = train.latent.mean(0)
    head.latent_std = train.latent.std(0).clamp_min(1e-6)

    # Decaying this head's weights toward zero means "ignore this rock's
    # latent", whose degenerate point is a constant forecast fit to the drawn
    # rocks -- and that forecast lives in the biases.  Decaying them too would
    # pull it toward p = 0.5 and alpha = softplus(0), which is not a forecast
    # anyone would make.  (The recalibrator needs no such carve-out: shrinking
    # every one of its parameters to zero is the identity correction, i.e. the
    # frozen simulator's own forecast.)
    decayed = [p for n, p in head.named_parameters() if not n.endswith("bias")]
    free = [p for n, p in head.named_parameters() if n.endswith("bias")]
    opt = torch.optim.Adam(
        [{"params": decayed, "weight_decay": weight_decay},
         {"params": free, "weight_decay": 0.0}],
        lr=lr,
    )
    for _ in range(epochs):
        opt.zero_grad()
        logits, alpha, log_w = head(train.latent, train.presence)
        loss = recalibrator_nll(
            logits, alpha, train.presence, train.magnitude, train.numeric,
            log_w=log_w, measured=train.measured,
        )["joint"].mean()
        loss.backward()
        opt.step()
    return head.eval()


@torch.no_grad()
def score_measured_head(
    head: MeasuredModeHead, data: GroupPredictions
) -> dict:
    """Score `data` under `head`, on the same scale as `score_recalibrator`."""
    return score_predictions(*head(data.latent, data.presence), data)


def ablate_latent(data: GroupPredictions) -> GroupPredictions:
    """`data` with the latent held constant: same rocks, no rock-specific input.

    A head fit on this can still learn the drawn cohort's label marginal, and
    the support it is conditioned on, but can read nothing about the individual
    rock -- so the gap to the unablated fit is how much of the measured arm is
    the representation rather than the marginal.

    Unlike the base-rate reference this resembles, it is fit on the drawn
    budget alone and scored on the same slice through the same likelihood, so
    its information access is matched to every other arm's.  That is the
    property whose absence made a rock-independent reference unusable.
    """
    return replace(data, latent=torch.zeros_like(data.latent))


def use_raw_chemistry(data: GroupPredictions) -> GroupPredictions:
    """`data` with the head reading the assay instead of the encoder's latent.

    The arm that asks whether the encoder was needed at all, rather than only
    whether the simulator supervision was: same head, same rocks, same fit, one
    input swapped for the thing that input was computed from.
    """
    return replace(data, latent=data.raw)
