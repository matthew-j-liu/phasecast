"""Fit the sim-to-real adapter, and compare adapters honestly.

The objective is `real_rock_nll` -- the same hurdle NLL the real-rock ledger
reports -- so an adapter's gain over the identity correction reads directly as
nats recovered on one scale.  Encoder and heads are frozen throughout: this
module only ever optimizes `RealRockAdapter` parameters against precomputed
group-space predictions.

Two measurement rules live here because getting them wrong is how this
benchmark manufactures a win.  Comparisons are paired on the same rocks and
bootstrapped over stratigraphic units, not rocks, since rocks inside a unit
are near-identical chemistry.  And the exposure probe scores every adapter on
a frozen row set, because the coverage restriction that makes it computable
depends on the presence probabilities the adapter modifies.
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..data.phase_vocab import GROUP_INDEX, GROUPS, MODEL_TO_GROUP, PHASE_INDEX
from ..eval.mode_head import conditional_mean, magnitude_parity_stats
from ..losses.mode_loss import masked_dirichlet_nll
from ..models.real_rock_adapter import RealRockAdapter


@dataclass(frozen=True)
class GroupPredictions:
    """Frozen-model outputs in group space, plus the truth they are scored on.

    Everything the adapter sees, precomputed once: the encoder and heads never
    run again during fitting.  `logits7` is the coarsened presence in logit
    space (that is where a Platt correction acts), `alpha7` the coarsened
    Dirichlet concentrations on the truth-forced support.
    """

    logits7: torch.Tensor
    alpha7: torch.Tensor
    sio2: torch.Tensor
    presence: torch.Tensor
    magnitude: torch.Tensor
    numeric: torch.Tensor
    unit: np.ndarray

    def __len__(self) -> int:
        return len(self.sio2)

    def subset(self, mask) -> "GroupPredictions":
        mask = np.asarray(mask, dtype=bool)
        t = torch.as_tensor(mask)
        return GroupPredictions(
            logits7=self.logits7[t], alpha7=self.alpha7[t], sio2=self.sio2[t],
            presence=self.presence[t], magnitude=self.magnitude[t],
            numeric=self.numeric[t], unit=self.unit[mask],
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
    return GroupPredictions(
        logits7=torch.as_tensor(np.log(p7 / (1 - p7)), dtype=torch.float64),
        alpha7=torch.as_tensor(pred["alpha7"], dtype=torch.float64),
        sio2=torch.as_tensor(benchmark.sio2, dtype=torch.float64),
        presence=torch.as_tensor(benchmark.presence, dtype=torch.float64),
        magnitude=torch.as_tensor(benchmark.magnitude, dtype=torch.float64),
        numeric=torch.as_tensor(benchmark.numeric.astype(np.float64)),
        unit=benchmark.unit,
    )


def adapter_nll(
    logits7: torch.Tensor,
    alpha7: torch.Tensor,
    presence: torch.Tensor,
    magnitude: torch.Tensor,
    numeric: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-rock hurdle NLL in group space, differentiable.

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
    ).sum(-1)
    safe_alpha = torch.where(numeric > 0, alpha7, torch.ones_like(alpha7))
    dirichlet = masked_dirichlet_nll(
        safe_alpha, magnitude, numeric, reduction="none",
    )
    return {"bernoulli": bern, "dirichlet": dirichlet, "joint": bern + dirichlet}


def fit_adapter(
    train: GroupPredictions,
    *,
    tier: str = "diagonal",
    weight_decay: float = 1e-2,
    epochs: int = 400,
    lr: float = 0.05,
    hidden: int = 32,
    pin_amp_magnitude: bool = True,
    seed: int = 0,
) -> RealRockAdapter:
    """Full-batch fit of one adapter on `train`, to convergence.

    There is no per-draw early stopping by design.  `weight_decay` is chosen
    once on the fixed val slice and shared across every point of the learning
    curve, so the curve's x-axis stays the honest labelled budget: every drawn
    rock is a fitted rock, and the val slice's whole contribution to the curve
    is a single scalar rather than one decision per draw.

    Full-batch because the training sets are 25-380 rows of 7 numbers; a
    dataloader here would be ceremony.
    """
    torch.manual_seed(seed)
    adapter = RealRockAdapter(
        tier, hidden=hidden, pin_amp_magnitude=pin_amp_magnitude,
    ).to(train.logits7.dtype)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        logits, alpha = adapter(train.logits7, train.alpha7, train.sio2)
        loss = adapter_nll(
            logits, alpha, train.presence, train.magnitude, train.numeric,
        )["joint"].mean()
        loss.backward()
        opt.step()
    return adapter


@torch.no_grad()
def score_adapter(
    adapter: RealRockAdapter | None, data: GroupPredictions
) -> dict[str, np.ndarray]:
    """Per-rock NLL terms and per-group share error under `adapter`.

    `adapter=None` is the zero-shot baseline -- the same loss at the identity
    correction, which is what makes a difference read as nats recovered.
    """
    if adapter is None:
        logits, alpha = data.logits7, data.alpha7
    else:
        logits, alpha = adapter(data.logits7, data.alpha7, data.sio2)
    nll = adapter_nll(
        logits, alpha, data.presence, data.magnitude, data.numeric,
    )
    numeric = data.numeric.numpy().astype(bool)
    cond_mean = conditional_mean(alpha.numpy(), numeric.astype(np.float64))
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
    tier: str,
    grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    sizes: tuple[int, ...] = (25, 50, 100, 200),
    n_draws: int = 3,
    seed: int = 0,
    **fit_kwargs,
) -> tuple[float, list[dict]]:
    """Pick one shrinkage constant per tier, once, on the fixed val slice.

    This is the curve's entire use of the val slice: one scalar per tier,
    fixed before the curve runs, rather than a per-draw early-stopping
    decision.  That keeps the x-axis honest -- every drawn rock is a fitted
    rock, so "N labelled rocks buys this much" is a claim someone could
    reproduce by harvesting N rocks.

    Chosen under the conditions the curve actually runs: candidates are scored
    across a range of training sizes spanning the curve's, not at full train
    size only, because the best shrinkage at 25 rocks is not the best at 380
    and the one constant has to serve both.
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
                adapter = fit_adapter(
                    train.subset(mask), tier=tier, weight_decay=weight_decay,
                    seed=draw, **fit_kwargs,
                )
                losses.append(score_adapter(adapter, val)["joint"].mean())
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


def lift_group_correction_to_phases(
    adapter: RealRockAdapter, logits7: torch.Tensor, logits21: np.ndarray,
    sio2: torch.Tensor,
) -> np.ndarray:
    """The adapter's presence correction pushed down onto the 21 phase logits.

    Needed only for the exposure probe.  Deployment stands on the model's own
    assemblage, and the adapter changes what that assemblage is -- but it acts
    on coarsened groups, so its verdict has to be pushed back down to decide a
    21-phase support.  Each group's logit shift is applied uniformly to its
    member phases, which leaves the within-group ranking the model chose
    untouched and moves only the group's overall call.
    """
    with torch.no_grad():
        corrected, _ = adapter(logits7, torch.ones_like(logits7), sio2)
    shift = (corrected - logits7).numpy()
    out = logits21.copy()
    for phase, group in MODEL_TO_GROUP.items():
        out[:, PHASE_INDEX[phase]] += shift[:, GROUP_INDEX[group]]
    return out


def per_group_offsets(adapter: RealRockAdapter) -> dict[str, dict[str, float]]:
    """The fitted constants, per group, for reading against the measured ones.

    Only meaningful for the diagonal tier, where a group's correction is a
    constant; the higher tiers couple groups and their offsets are not
    separable this way.
    """
    with torch.no_grad():
        return {
            group: {
                "presence_scale": float(adapter.presence_scale[i]),
                "presence_offset": float(adapter.presence_offset[i]),
                "magnitude_offset": float(
                    adapter.magnitude_offset[i] * adapter.magnitude_free[i]
                ),
            }
            for i, group in enumerate(GROUPS)
        }
