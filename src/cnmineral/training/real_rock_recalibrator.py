"""Fit the sim-to-real recalibrator, and compare recalibrators honestly.

The objective is `real_rock_nll` -- the same hurdle NLL the real-rock ledger
reports -- so a recalibrator's gain over the identity correction reads directly as
nats recovered on one scale.  Encoder and heads are frozen throughout: this
module only ever optimizes `RealRockRecalibrator` parameters against precomputed
group-space predictions.

Two measurement rules live here because getting them wrong is how this
benchmark manufactures a win.  Comparisons are paired on the same rocks and
bootstrapped over stratigraphic units, not rocks, since rocks inside a unit
are near-identical chemistry.  And the exposure probe scores every recalibrator on
a frozen row set, because the coverage restriction that makes it computable
depends on the presence probabilities the recalibrator modifies.
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..data.phase_vocab import GROUP_INDEX, GROUPS, MODEL_TO_GROUP, PHASE_INDEX
from ..eval.mode_head import (
    conditional_mean,
    magnitude_parity_stats,
    mixture_conditional_mean,
)
from ..eval.real_rock import coarsen_presence, raw_chemistry
from ..losses.mode_loss import masked_dirichlet_nll, mixture_dirichlet_nll
from ..models.real_rock_recalibrator import RealRockRecalibrator


@dataclass(frozen=True)
class GroupPredictions:
    """Frozen-model outputs in group space, plus the truth they are scored on.

    Everything a fitted arm sees, precomputed once: the encoder and heads never
    run again during fitting.  `logits7` is the coarsened presence in logit
    space (that is where a Platt correction acts), `alpha7` the coarsened
    Dirichlet concentrations on the truth-forced support.

    `latent` and `raw` are the two candidate inputs a re-supervised head can
    read -- the pooled encoder output, and the assay it was computed from.  The
    recalibrator reads neither, but carrying them here is what keeps every arm
    subset by the identical unit draw, which is what makes the arms
    differenceable rather than merely comparable.

    `measured` is the benchmark's mask of groups a source looked for; the
    presence term of an unmeasured group is left out of the NLL.  `None`
    scores every group.
    """

    logits7: torch.Tensor
    alpha7: torch.Tensor
    sio2: torch.Tensor
    presence: torch.Tensor
    magnitude: torch.Tensor
    numeric: torch.Tensor
    unit: np.ndarray
    log_w: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    raw: torch.Tensor | None = None
    measured: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.sio2)

    def subset(self, mask) -> "GroupPredictions":
        mask = np.asarray(mask, dtype=bool)
        t = torch.as_tensor(mask)
        return GroupPredictions(
            logits7=self.logits7[t], alpha7=self.alpha7[t], sio2=self.sio2[t],
            presence=self.presence[t], magnitude=self.magnitude[t],
            numeric=self.numeric[t], unit=self.unit[mask],
            log_w=None if self.log_w is None else self.log_w[t],
            latent=None if self.latent is None else self.latent[t],
            raw=None if self.raw is None else self.raw[t],
            measured=None if self.measured is None else self.measured[t],
        )


def build_group_predictions(
    head, benchmark, latents: torch.Tensor
) -> GroupPredictions:
    """Run the frozen head once and freeze its group-space outputs.

    Presence is carried as a logit rather than a probability because that is
    where the correction acts, and because `binary_cross_entropy_with_logits`
    is the numerically stable form.  The clip matches `real_rock_nll`'s, so
    this and the numpy eval agree at the identity correction -- which is the
    whole basis for reading a difference as nats recovered.
    """
    from ..eval.real_rock import predict

    pred = predict(head, latents, benchmark)
    p7 = np.clip(pred["p7"], 1e-7, 1 - 1e-7)
    log_w = pred.get("log_w")
    return GroupPredictions(
        logits7=torch.as_tensor(np.log(p7 / (1 - p7)), dtype=torch.float64),
        alpha7=torch.as_tensor(pred["alpha7"], dtype=torch.float64),
        sio2=torch.as_tensor(benchmark.sio2, dtype=torch.float64),
        presence=torch.as_tensor(benchmark.presence, dtype=torch.float64),
        magnitude=torch.as_tensor(benchmark.magnitude, dtype=torch.float64),
        numeric=torch.as_tensor(benchmark.numeric.astype(np.float64)),
        unit=benchmark.unit,
        log_w=None if log_w is None
        else torch.as_tensor(log_w, dtype=torch.float64),
        latent=latents.detach().cpu().to(torch.float64),
        raw=raw_chemistry(benchmark),
        measured=torch.as_tensor(benchmark.measured.astype(np.float64)),
    )


def recalibrator_nll(
    logits7: torch.Tensor,
    alpha7: torch.Tensor,
    presence: torch.Tensor,
    magnitude: torch.Tensor,
    numeric: torch.Tensor,
    log_w: torch.Tensor | None = None,
    measured: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-rock hurdle NLL in group space, differentiable.

    `measured` drops the presence term of a group the source did not look
    for; `real_rock_nll` has no such mask, so the twins agree only on a fully
    measured benchmark, the only kind `score_real_rock` accepts.

    The torch twin of `eval.real_rock.real_rock_nll`, and it must agree with it
    at the identity correction or a "recovered nats" number means nothing.

    The `torch.where` is the NaN guard, and it belongs here rather than in
    `masked_dirichlet_nll`: that loss forms `mask * lgamma(alpha)`, `lgamma(0)`
    is `+inf`, and coarsening produces exact zeros wherever a group has no
    member phase on the support.  Those groups are excluded by the mask anyway,
    so any positive filler works -- but `0 * inf` is NaN, not zero, and the
    gradient of a NaN is a dead fit.  `torch.where` keeps gradients intact on
    the entries that are actually scored.
    """
    bern = torch.nn.functional.binary_cross_entropy_with_logits(
        logits7, presence, reduction="none",
    )
    if measured is not None:
        bern = bern * measured
    bern = bern.sum(-1)
    mask = numeric if log_w is None else numeric.unsqueeze(1)
    safe_alpha = torch.where(mask > 0, alpha7, torch.ones_like(alpha7))
    if log_w is None:
        dirichlet = masked_dirichlet_nll(
            safe_alpha, magnitude, numeric, reduction="none",
        )
    else:
        dirichlet = mixture_dirichlet_nll(
            log_w, safe_alpha, magnitude, numeric, reduction="none",
        )
    return {"bernoulli": bern, "dirichlet": dirichlet, "joint": bern + dirichlet}


def fit_recalibrator(
    train: GroupPredictions,
    *,
    tier: str = "vector",
    weight_decay: float = 1e-2,
    epochs: int = 400,
    lr: float = 0.05,
    seed: int = 0,
) -> RealRockRecalibrator:
    """Full-batch fit of one recalibrator on `train`, to convergence.

    There is no per-draw early stopping by design.  `weight_decay` is chosen
    once on the fixed val slice and shared across every point of the learning
    curve, so the curve's x-axis stays the honest labelled budget: every drawn
    rock is a fitted rock, and the val slice's whole contribution to the curve
    is a single scalar rather than one decision per draw.

    Full-batch because the training sets are 25-380 rows of 7 numbers; a
    dataloader here would be ceremony.
    """
    torch.manual_seed(seed)
    recalibrator = RealRockRecalibrator(tier).to(train.logits7.dtype)
    opt = torch.optim.Adam(recalibrator.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        logits, alpha, log_w = recalibrator(
            train.logits7, train.alpha7, log_w=train.log_w
        )
        loss = recalibrator_nll(
            logits, alpha, train.presence, train.magnitude, train.numeric,
            log_w=log_w, measured=train.measured,
        )["joint"].mean()
        loss.backward()
        opt.step()
    return recalibrator


@torch.no_grad()
def score_predictions(
    logits: torch.Tensor,
    alpha: torch.Tensor,
    log_w: torch.Tensor | None,
    data: GroupPredictions,
) -> dict[str, np.ndarray]:
    """Per-rock NLL terms and per-group share error for one arm's forecast.

    Takes the forecast rather than the model that made it, so every arm on the
    curve -- the identity correction, a fitted `RealRockRecalibrator`, a
    `MeasuredModeHead` -- lands on one scale through one code path.  An arm
    scored by its own bespoke assembly would not be comparable to the others,
    which is the entire basis for reading a difference as nats.
    """
    nll = recalibrator_nll(
        logits, alpha, data.presence, data.magnitude, data.numeric,
        log_w=log_w, measured=data.measured,
    )
    numeric = data.numeric.numpy().astype(bool)
    if log_w is None:
        cond_mean = conditional_mean(alpha.numpy(), numeric.astype(np.float64))
    else:
        cond_mean = mixture_conditional_mean(
            alpha.numpy(), log_w.numpy(), numeric.astype(np.float64)
        )
    parity = magnitude_parity_stats(
        cond_mean, data.magnitude.numpy(), numeric.astype(np.float64),
    )
    return {
        "bernoulli": nll["bernoulli"].numpy(),
        "dirichlet": nll["dirichlet"].numpy(),
        "joint": nll["joint"].numpy(),
        "share_mae": parity["magnitude_mae"],
        "share_mae_mean": float(np.nanmean(parity["magnitude_mae"])),
        "mean_composition_size": float(numeric.sum(1).mean()),
    }


@torch.no_grad()
def score_recalibrator(
    recalibrator: RealRockRecalibrator | None, data: GroupPredictions
) -> dict[str, np.ndarray]:
    """Score `data` under `recalibrator`.

    `recalibrator=None` is the zero-shot baseline -- the same loss at the identity
    correction, which is what makes a difference read as nats recovered.
    """
    if recalibrator is None:
        forecast = data.logits7, data.alpha7, data.log_w
    else:
        forecast = recalibrator(
            data.logits7, data.alpha7, log_w=data.log_w
        )
    return score_predictions(*forecast, data)


def draw_units(
    data: GroupPredictions, target_rocks: int, rng: np.random.Generator,
    *, overshoot: float = 0.2,
) -> np.ndarray:
    """A mask over `data` holding about `target_rocks`, sampled as whole units.

    The curve's x-axis counts labelled rocks, but rocks are only harvestable an
    eruption at a time, so units are the sampling atom.  Unit sizes here run 1
    to 66, so a unit that would overshoot the target by more than `overshoot`
    is skipped rather than taken -- otherwise a single large unit decides the
    low end of the curve.  The realized count is what the caller should plot.
    """
    units = np.unique(data.unit)
    rng.shuffle(units)
    taken, total = [], 0
    for unit in units:
        size = int((data.unit == unit).sum())
        if total + size > target_rocks * (1 + overshoot):
            continue
        taken.append(unit)
        total += size
        if total >= target_rocks:
            break
    return np.isin(data.unit, taken)


def select_weight_decay(
    train: GroupPredictions,
    val: GroupPredictions,
    *,
    fit,
    score=score_recalibrator,
    grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    sizes: tuple[int, ...] = (25, 50, 100, 200),
    n_draws: int = 3,
    seed: int = 0,
) -> tuple[float, list[dict]]:
    """Pick one shrinkage constant per arm, once, on the fixed val slice.

    This is the curve's entire use of the val slice: one scalar per arm,
    fixed before the curve runs, rather than a per-draw early-stopping
    decision.  That keeps the x-axis honest -- every drawn rock is a fitted
    rock, so "N labelled rocks buys this much" is a claim someone could
    reproduce by harvesting N rocks.

    Chosen under the conditions the curve actually runs: candidates are scored
    across a range of training sizes spanning the curve's, not at full train
    size only, because the best shrinkage at 25 rocks is not the best at 380
    and the one constant has to serve both.

    `fit(drawn, weight_decay=..., seed=...)` and `score(model, val)` are
    injected rather than hardcoded so every arm's constant is chosen by the
    identical protocol.  A per-arm copy of this loop would let the protocols
    drift, and a drifted protocol is exactly what would make the arms
    incomparable.
    """
    records = []
    for weight_decay in grid:
        losses = []
        for size in sizes:
            rng = np.random.default_rng(seed)
            for draw in range(n_draws):
                mask = draw_units(train, size, rng)
                if not mask.any():
                    continue
                model = fit(
                    train.subset(mask), weight_decay=weight_decay, seed=draw,
                )
                losses.append(score(model, val)["joint"].mean())
        records.append({
            "weight_decay": weight_decay,
            "val_joint_nll": float(np.mean(losses)) if losses else float("inf"),
        })
    best = min(records, key=lambda r: r["val_joint_nll"])
    return best["weight_decay"], records


def paired_bootstrap(
    baseline: np.ndarray,
    adapted: np.ndarray,
    unit: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Mean paired improvement (baseline - adapted) with a cluster-bootstrap CI.

    Per-rock joint NLL has sd 13.8 against mean 10.4, so an absolute level on a
    ~160-rock slice carries roughly +/- 2 nats and comparing two levels is
    hopeless.  Pairing on the same rocks cancels almost all of that, which is
    why a recovered-nats number is measurable when the level is not.

    Resampling is over units rather than rocks: one eruption's rocks share
    nearly identical chemistry, so treating them as independent draws would
    understate the interval by roughly the square root of the cluster size.
    """
    rng = np.random.default_rng(seed)
    delta = baseline - adapted
    units, inverse = np.unique(unit, return_inverse=True)
    by_unit = [np.flatnonzero(inverse == i) for i in range(len(units))]

    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(by_unit), len(by_unit))
        draws[b] = delta[np.concatenate([by_unit[i] for i in pick])].mean()
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(delta.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_rocks": int(len(delta)),
        "n_units": int(len(units)),
    }


def lift_group_presence_to_phases(
    recalibrator: RealRockRecalibrator, logits7: torch.Tensor, p21: np.ndarray,
) -> np.ndarray:
    """The recalibrator's corrected group presence, pushed back down onto 21 phases.

    The recalibrator judges groups, but deployment stands on a 21-phase support --
    both the exposure probe and the scenario view need one -- so the group
    verdict has to be distributed over member phases.

    The transform is exact rather than approximate.  Raising each member's
    absence probability to a common power, `1 - p_i' = (1 - p_i)^gamma`, sends
    the group's noisy-OR to `1 - (1 - P)^gamma`, so
    `gamma = log(1 - P_target) / log(1 - P_now)` hits the corrected group
    probability on the nose while preserving the within-group ordering the
    model chose.

    Shifting each member's *logit* by the group's logit shift -- the obvious
    thing -- is badly wrong, because the noisy-OR is not linear in logits.  A
    group sitting at 0.9999 (logit 9.2) corrected to 0.84 (logit 1.7) gives a
    -7.5 shift, and applying that to both members drives each to ~0, so the
    group vanishes from every sampled support even though it was corrected
    *upward* of absent.  That produced scenarios with no feldspar in a rock
    whose feldspar is half the crystal cargo.
    """
    with torch.no_grad():
        corrected, _, _ = recalibrator(logits7, torch.ones_like(logits7))
    target = torch.sigmoid(corrected).numpy()
    now = coarsen_presence(p21)

    eps = 1e-9
    gamma = np.log(np.clip(1.0 - target, eps, 1.0)) / np.log(
        np.clip(1.0 - now, eps, 1.0 - eps)
    )
    out = p21.copy()
    for phase, group in MODEL_TO_GROUP.items():
        absent = np.clip(1.0 - p21[:, PHASE_INDEX[phase]], eps, 1.0)
        out[:, PHASE_INDEX[phase]] = 1.0 - absent ** gamma[:, GROUP_INDEX[group]]
    return out


def per_group_offsets(recalibrator: RealRockRecalibrator) -> dict[str, dict[str, float]]:
    """The fitted constants, per group, for reading against the measured ones.

    Only meaningful for the vector tier, where a group's correction is a
    constant; the higher tiers couple groups and their offsets are not
    separable this way.
    """
    with torch.no_grad():
        return {
            group: {
                "presence_scale": float(recalibrator.presence_scale[i]),
                "presence_offset": float(recalibrator.presence_offset[i]),
                "magnitude_offset": float(recalibrator.magnitude_offset[i]),
            }
            for i, group in enumerate(GROUPS)
        }
