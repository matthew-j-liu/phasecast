# Decision 007: SSL pretrain objective is per-element standardized log-space masked MSE

**Date:** 2026-06-18
**Status:** Active

## Decision

The canonical SSL encoder is pretrained with a **per-element standardized
log-space masked-element MSE** loss (`loss: {type: log_mse, standardize:
true}`), replacing the linear-space masked MSE used by the prior canonical
encoder `pretrain_v3_seed0`.

Canonical checkpoint: `checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt`
(architecture unchanged: d_model 256 / 4 layers / 8 heads; 20 epochs on Gard
2019 igneous bulks). Optimization hyperparameters from the per-objective
Optuna sweep (`configs/pretrain/pretrain_hpo.yaml`).

Downstream consumers (corpus regen, supervised heads, imputation read-out
tune, all encoder-application probes) load this checkpoint.

## Why this decision

The linear-space MSE objective weights each element by its raw target
magnitude. Majors (wt%/100) and traces (ppm/1e6) differ by ~1e7-1e8x, so trace
residuals contribute almost nothing to the loss and the encoder is trained to
fit majors almost exclusively. Scoring the frozen latent on a leakage-guarded
XGBoost trace-imputation gate and on frozen-latent classification probes, the
standardized log-space objective measures higher on every axis than the
linear-space baseline, across two seeds:

| metric (frozen-latent) | linear MSE (baseline) | standardized log-MSE |
|---|---|---|
| gate median trace log-R2 | 0.839 | 0.877 (seed 1: 0.873) |
| gate worst-trace Yb log-R2 | 0.724 | 0.831 (seed 1: 0.841) |
| gate majors log-R2 | -- | +0.002..+0.032, all 8 |
| supergroup MLP probe | 0.806 | 0.821 (seed 1: 0.820) |
| qap MLP probe (fixed) | 0.762 | 0.769 (seed 1: 0.771) |
| qap MLP probe (HPO) | 0.776 | 0.790 (seed 1: 0.788) |

The major-driven classification probes do not regress; they improve. So the
objective is not trading major structure for trace recovery -- both move up.

## Alternatives rejected

- **Plain log-space MSE (`standardize: false`).** Same direction, uniformly
  weaker (gate median trace 0.871, supergroup MLP 0.812, qap MLP fixed 0.758
  -- below the baseline on qap). Per-element standardization in log space adds
  a consistent margin at no measured cost.

- **Inverse-variance-weighted linear-space MSE (`weighted_mse`).** Worse than
  the linear baseline on both instruments (gate median trace 0.706, supergroup
  MLP 0.549). Equalizing each element's loss contribution by a fixed weight
  does not address a dynamic-range problem carried in the residual magnitudes:
  the optimization still chases large-magnitude majors and learns a latent with
  less trace signal and less class structure. Rejected.

- **Re-searching the architecture per objective.** Held to the original
  256/4/8 winner so the objective is the only variable vs the linear-MSE
  encoder; re-searching architecture would reconfound the comparison.

## Consequences

- `pretrain_v3_seed0` (linear MSE) is retired as canonical; it remains on disk
  as a reference artifact and as the baseline in the gate/probe result trees.
- ~~The configurable-loss infrastructure
  (`cnmineral.training.pretrain_objective`) stays: it is how this comparison
  was run and how a future objective would be tested.~~ Superseded by the
  amendment below.
- The imputation read-out head re-scores against the new frozen latent; its
  earlier frozen-latent reference (0.839 median trace log-R2) becomes 0.877.

## Amendment (2026-06-29)

The configurable-loss infrastructure is removed. The linear-space objective
(`masked_element_mse`), the rejected plain-log variant (`standardize: false`),
and the `loss:` config token with its type-dispatch are deleted. The objective
is now singular and unconditional: per-element standardized log-space masked
MSE, defined in code and in this ADR rather than selected by config. The
present-tense decoupling seam is kept -- the train loop and `evaluate_on_test`
still consume an opaque `loss_fn` built by `make_loss_fn` -- but it builds the
one objective with no menu.

The original reason for keeping the linear code was reproducibility of the
linear-vs-log comparison. That is preserved by the existing tag
`v3.1-logmse-objective` (commit `6ba9695`, 2026-06-24), which points at a
pre-cleanup state containing both objectives and all linear configs:
`git checkout v3.1-logmse-objective` restores the full comparison harness. The
seed-1 confirmation run is reproducible from `pretrain_production.yaml` with
`seed: 1`.
