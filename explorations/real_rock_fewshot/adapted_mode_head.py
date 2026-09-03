"""Adapt the frozen simulator-supervised ModeHead itself, on measured labels.

The correction arms fit at most 123 parameters on the frozen head's outputs, so
the honest reading of their loss to the measured-label head is that nothing was
ever adapted -- the head was rescaled.  This arm removes that reading: low-rank
updates inside the head's own linear layers, trained on the same measured modal
labels through `recalibrator_nll`, at a trainable-parameter count matched to the
measured head's.

Encoder, base head weights, and the 21-phase support the magnitude branch is
conditioned on all stay frozen.  Unlike every other arm this one runs the head
inside the training loop rather than on precomputed outputs, so the coarsening
onto the seven measured groups happens here in torch; the numpy twins in
`cnmineral.eval.real_rock` carry no gradient.
"""

import contextlib
import copy

import torch
import torch.nn.functional as F
from torch import nn

from cnmineral.data.phase_vocab import (
    GROUP_INDEX,
    GROUPS,
    MODEL_TO_GROUP,
    N_PHASES,
    PHASE_INDEX,
)
from cnmineral.eval.real_rock import truth_forced_support
from cnmineral.models.mode_head import ModeHead
from cnmineral.training.real_rock_recalibrator import (
    GroupPredictions,
    recalibrator_nll,
    score_predictions,
)

# 5 puts the adapters at 33,595 trainable parameters against `MeasuredModeHead`'s
# 34,839, so the two arms are compared at matched trainable budget.  That is a
# fairness gesture rather than a capacity claim: a low-rank perturbation of a
# trained 1.35M-parameter head is not the hypothesis class of a 35k-parameter
# head fit from scratch.
RANK: int = 5

# Fixed at a train-loss plateau by `run_plateau_probe.py`, never on val, so this
# arm carries the same single val-selected knob (weight decay) the others do
# rather than three.  `LR` is the largest rate that held its optimum at every
# budget and seed; `EPOCHS` is the slowest of those runs' plateaus, so the
# fastest-converging point on the curve is not what sets the budget.
LR: float = 1e-3
EPOCHS: int = 1636

# P below this reads as absent.  Only guards `log(0)` when an adapter drives
# every member phase of a group to a logit so negative that `log(1 - p)`
# underflows to exactly 0; it sits well past the 1e-7 the shipped eval clips at,
# so it never binds anywhere that clip is informative.
PRESENCE_FLOOR: float = 1e-12


def group_map(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The fixed [21, 7] 0/1 map from model phases onto measured groups.

    Phases no source reports have an all-zero row, which is how the off-vocabulary
    mass drops out of both coarsenings.
    """
    out = torch.zeros(N_PHASES, len(GROUPS), dtype=dtype)
    for phase, group in MODEL_TO_GROUP.items():
        out[PHASE_INDEX[phase], GROUP_INDEX[group]] = 1.0
    return out


def coarsen_presence_logits(
    logits21: torch.Tensor, phase_to_group: torch.Tensor
) -> torch.Tensor:
    """Group presence logits from the 21 per-phase logits, carried in log space.

    The same noisy-OR as `eval.real_rock.coarsen_presence` -- a group is present
    when any member phase is, exactly, since the head emits independent Bernoulli
    marginals -- but `log(1 - P) = sum_i log(1 - p_i)` is a sum of `logsigmoid`
    and the logit comes back as `log(-expm1(s)) - s`.

    Probability space is the wrong place to do this here.  For a group the model
    calls absent, `P` is recovered by subtracting a product very close to one
    from one, which discards most of the mantissa; on this benchmark that is
    worth up to 0.01 in the returned logit.  The shipped eval wears that error
    in a reported number, which is survivable.  Here it would be a gradient
    path.
    """
    s = (F.logsigmoid(-logits21) @ phase_to_group).clamp(max=-PRESENCE_FLOOR)
    return torch.log(-torch.expm1(s)) - s


def coarsen_alpha_on_support(
    alpha21: torch.Tensor, support21: torch.Tensor, phase_to_group: torch.Tensor
) -> torch.Tensor:
    """Group Dirichlet concentrations, summed over member phases on the support.

    Merging components of a Dirichlet sums their concentrations, so this is
    exact rather than an approximation, and it reproduces
    `eval.real_rock.coarsen_alpha` bit for bit.  A group with no member on the
    support comes back an exact zero; keeping those out of `lgamma` is
    `recalibrator_nll`'s guard, not this function's job.
    """
    return (alpha21 * support21) @ phase_to_group


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable rank-`r` update.

    `b` starts at zero, so the wrapped layer reproduces the frozen one exactly at
    initialization -- the same identity-at-init contract `RealRockRecalibrator`
    holds, and what lets a fitted gain read as nats recovered against the
    zero-shot forecast.

    It also fixes what weight decay means.  Shrinking every adapter to zero
    returns the frozen head, which is a forecast someone would actually make;
    `MeasuredModeHead` needs a bias carve-out to arrange the same thing, and
    this arm gets it for free.
    """

    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=base.weight.dtype)
        )
        self.b = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        return out + (x @ self.a.T) @ self.b.T

    def extra_repr(self) -> str:
        return f"rank={self.b.shape[1]}, enabled={self.enabled}"


class AdaptedModeHead(nn.Module):
    """The frozen ModeHead with a low-rank adapter on every linear layer.

    Emits the seven measured groups, so
    `training.real_rock_recalibrator.score_predictions` takes its forecast
    unchanged and this arm lands on the same axis as every other one.  The
    coarsening lives inside the module because the head now runs inside the
    training loop.

    The head is deep-copied rather than adapted in place: the caller holds one
    frozen head and shares it across every draw and every arm of the curve.

    It inherits that head's single Dirichlet, as `matrix` does; only
    `MeasuredModeHead` is two-component.  Widening `alpha_out` to reach a
    mixture would be new parameters rather than a low-rank update, and would
    forfeit the identity at initialization that makes the gain readable.
    """

    def __init__(self, head: ModeHead, *, rank: int = RANK) -> None:
        super().__init__()
        # `build_group_predictions` lands every array on the CPU, so the head
        # follows its data rather than making the caller match devices; the
        # whole fit is 380 rows through a 1.3M-parameter head.
        self.head = copy.deepcopy(head).to("cpu")
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.adapters = _wrap_linears(self.head, rank)
        self.register_buffer("phase_to_group", group_map())
        # the base head carries dropout, and waking it would hand this arm a
        # stochastic regularizer no other arm on the curve has
        self.eval()

    @property
    def head_dtype(self) -> torch.dtype:
        """What the base head computes in; the coarsening uses the map's instead."""
        return next(self.head.parameters()).dtype

    def lora_parameters(self) -> list[nn.Parameter]:
        """The only tensors that train; everything else is frozen."""
        return [p for lora in self.adapters for p in (lora.a, lora.b)]

    @contextlib.contextmanager
    def frozen_path(self):
        """Run the base head as if it had never been adapted."""
        for lora in self.adapters:
            lora.enabled = False
        try:
            yield
        finally:
            for lora in self.adapters:
                lora.enabled = True

    @torch.no_grad()
    def pinned_support(
        self, latent: torch.Tensor, presence: torch.Tensor
    ) -> torch.Tensor:
        """The frozen head's truth-forced support, which this arm never moves.

        `truth_forced_support` is a threshold on the per-phase probabilities plus
        an argmax, so it carries no gradient whatever the adapters do; letting it
        track them would only make the objective jump whenever a member phase
        crosses the threshold mid-fit.

        Pinning also keeps the invariant the magnitude term rests on: every
        truth-present group holds at least one member on the support, so a
        scored group's coarsened concentration is never the exact zero `lgamma`
        cannot take.  `recalibrator_nll`'s guard covers off-mask entries, not
        that.

        Reproduces `eval.real_rock.predict`'s `support21` exactly, which is what
        puts this arm's Dirichlet on the same support the correction arms'
        `alpha7` was built on.
        """
        with self.frozen_path():
            p21 = torch.sigmoid(
                self.head.presence_logits(latent.to(self.head_dtype))
            )
        return torch.as_tensor(
            truth_forced_support(presence.numpy(), p21.numpy()), dtype=latent.dtype
        )

    def forward(
        self, latent: torch.Tensor, support21: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """([B, d_model], [B, 21]) -> (logits [B, 7], alpha [B, 7], None).

        The head runs at its own dtype and the coarsening at the group map's, so
        the loss is assembled in the same float64 every other arm's is.  The
        trailing `None` is the single-Dirichlet gate, matching what
        `RealRockRecalibrator` returns.
        """
        out_dtype = self.phase_to_group.dtype
        z = latent.to(self.head_dtype)
        support = support21.to(self.head_dtype)
        logits21 = self.head.presence_logits(z)
        alpha21 = self.head.magnitude_alpha(z, support)
        return (
            coarsen_presence_logits(logits21.to(out_dtype), self.phase_to_group),
            coarsen_alpha_on_support(
                alpha21.to(out_dtype),
                support21.to(out_dtype),
                self.phase_to_group,
            ),
            None,
        )

    def extra_repr(self) -> str:
        n = sum(p.numel() for p in self.lora_parameters())
        return f"adapters={len(self.adapters)}, trainable={n}"


def _wrap_linears(head: nn.Module, rank: int) -> list[LoRALinear]:
    """Replace every `nn.Linear` in `head` with an adapted one, in place.

    Walks the tree rather than naming the eight layers, so a head built at a
    different `mode_layers` is adapted throughout instead of partially.  The
    snapshot is taken before any substitution: a `LoRALinear` holds an
    `nn.Linear`, and walking live would wrap the wrappers.
    """
    wrapped = []
    for module in [m for _, m in head.named_modules()]:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                lora = LoRALinear(child, rank)
                setattr(module, name, lora)
                wrapped.append(lora)
    return wrapped


def fit_adapted_head(
    train: GroupPredictions,
    *,
    head: ModeHead,
    weight_decay: float = 1.0,
    rank: int = RANK,
    epochs: int = EPOCHS,
    lr: float = LR,
    seed: int = 0,
) -> AdaptedModeHead:
    """Full-batch fit of the adapters on `train`, base head and encoder frozen.

    Protocol matches `fit_recalibrator` and `fit_measured_head` exactly -- full
    batch, to convergence, no per-draw early stopping, one val-selected
    shrinkage constant shared across every budget.  A protocol difference
    between the arms would be indistinguishable from the difference the curve is
    measuring.

    Every adapter parameter is decayed, with no carve-out: the degenerate point
    of shrinking them all to zero is the frozen head's own forecast.  That is
    the property `RealRockRecalibrator` has, and the reason `fit_measured_head`
    needs its bias exemption while this does not.

    Adam rather than AdamW, matching both, so the shared weight-decay grid means
    the same thing across arms.

    The returned head carries its per-epoch train loss on `train_loss`, which is
    what `run_plateau_probe.py` reads to fix `lr` and `epochs` without ever
    touching val.
    """
    if train.latent is None:
        raise ValueError("fit_adapted_head needs GroupPredictions.latent")
    torch.manual_seed(seed)
    adapted = AdaptedModeHead(head, rank=rank)
    support = adapted.pinned_support(train.latent, train.presence)
    opt = torch.optim.Adam(
        adapted.lora_parameters(), lr=lr, weight_decay=weight_decay
    )
    trace = []
    for _ in range(epochs):
        opt.zero_grad()
        logits, alpha, log_w = adapted(train.latent, support)
        loss = recalibrator_nll(
            logits, alpha, train.presence, train.magnitude, train.numeric,
            log_w=log_w,
        )["joint"].mean()
        loss.backward()
        opt.step()
        trace.append(loss.item())
    adapted.train_loss = trace
    return adapted


@torch.no_grad()
def score_adapted_head(adapted: AdaptedModeHead, data: GroupPredictions) -> dict:
    """Score `data` under `adapted`, on the same scale as `score_recalibrator`.

    The support is rebuilt from the frozen path rather than carried off the fit:
    scoring happens on a slice the fit never saw, and it must sit on the same
    support the correction arms' `alpha7` does there too.
    """
    support = adapted.pinned_support(data.latent, data.presence)
    return score_predictions(*adapted(data.latent, support), data)
