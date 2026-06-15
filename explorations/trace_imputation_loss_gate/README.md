# trace_imputation_loss_gate

## Question

The frozen SSL encoder + pretrain head imputes major elements well (R2 0.68-0.94)
but fails completely on traces (negative R2, predictions pinned to a ~1e-3 frac
floor). Root cause (docs/JOURNAL.md 2026-06-15): the SSL pretrain loss is raw-frac
MSE, major-dominated by ~1e7-1e8x, so traces are effectively unsupervised. That
loss played two roles -- it trained the head we reuse for imputation *and* it
shaped the encoder latent. So: which layer do we fix? A cheap log-space read-out
head (encoder untouched), or a rebalanced re-pretrain?

## Method

A controlled pair of predictors per element on the Gard test split, both in log
space -- only the input representation differs between them:

- **1a** input-space XGBoost: predict each element from the *other measured
  elements*. The practical ceiling / specialist baseline.
- **1b** XGBoost on the frozen latent `h`, with the target element masked before
  encoding (`src=MASK_TOKEN_ID, frac=0`) so `h` cannot leak the held-out element.

**Gate** on 1b vs 1a over the 13 traces:

- 1b ~ 1a -> **Scenario A**: the latent already carries the trace info. Fix the
  head (cheap log-space read-out on the frozen `h`); encoder untouched.
- 1b << 1a -> **Scenario B**: the latent is trace-blind. Rebalanced re-pretrain.

Tuning: light shared Optuna, two studies (one per feature set), objective = mean
val-MSE over the 13 traces, `n_trials=40`; the single best param set is reused
across all 22 final fits. XGBoost only. Read-only -- nothing here trains or
mutates the canonical encoder.

The current pretrain head's imputation (linear raw-frac MSE objective) is *not* a
controlled arm here -- scored in log space it only restates its known objective
mismatch. Its fair, linear-space numbers live in `results/imputation/`; the
genuinely controlled third arm is the future log-trained read-out head (see the
tracking issue), not the current head.

## Result

TODO -- run the gate and commit `results/<tag>/` (`diagnostic_metrics.json`,
`r2_by_element.png`, trace parity grids).

## Decision

TODO -- record the Scenario A/B verdict here and in docs/JOURNAL.md (and an ADR in
docs/decisions/ if the gate lands on B / changes the loss).

## Run

```
conda activate alignn_fix
python -u explorations/trace_imputation_loss_gate/run_gate.py \
  --checkpoint checkpoints/pretrain_v3_seed0/epoch_best.pt
```

## See also

- docs/JOURNAL.md 2026-06-15 -- the framing and the Phase 1 gate decision.
