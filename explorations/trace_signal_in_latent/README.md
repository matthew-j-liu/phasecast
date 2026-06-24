# trace_signal_in_latent

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

Full run (`--n-trials 40`, no subsample) on `pretrain_v3_seed0/epoch_best.pt`
(d_model 256, epoch 20, n_test 33,371). **Verdict: Scenario A.**

Over the 13 traces, median R2 **1a=0.892 vs 1b=0.839** (mean 0.899 vs 0.836) --
1b tracks the input-space ceiling within ~0.05. The latent is far from
trace-blind: XGBoost on the frozen masked latent recovers every trace at R2
0.72-0.90. The 1b deficit is largest on the HREE (Yb -0.14, Y -0.11; then
Nd/La ~-0.065) and smallest on Th/Rb/Ni/V (~-0.04). Majors: 1b within 0.01-0.04
of 1a. Artifacts in `results/pretrain_v3_seed0_epoch_best/`.

Note: the 1a and 1b studies converge to the same XGBoost HPs (`best_1a ==
best_1b`).  This is not under-tuning -- `audit_tuning_objective.py` (issue #6,
JOURNAL 2026-06-22) re-tuned under majors-only and all-22 objectives and all
land on the identical optimum: the HP preference ordering is the same across
element subsets, so the tuning objective doesn't move the gate's verdict or the
major R2s.

## Decision

**Scenario A** -- the trace-imputation failure is the *head's* objective (linear
raw-frac MSE), not encoder capacity. Fix = a cheap log-space read-out head on the
frozen latent; **do not** re-pretrain the encoder for imputation. The encoder
loss is unchanged, so no ADR. Read-out head tracked in issue #5; canonical
verdict also in docs/JOURNAL.md 2026-06-15.

## Run

```
conda activate alignn_fix
python -u explorations/trace_signal_in_latent/run_gate.py \
  --checkpoint checkpoints/pretrain_v3_seed0/epoch_best.pt
```

## See also

- docs/JOURNAL.md 2026-06-15 -- the framing and the Phase 1 gate decision.
