# presence_fraction_ceiling

## Question

The #41 measurement showed the ModeHead's accessory problem is not marginal
calibration (aggregate rates are on-diagonal within 1.6pp, isotonic remap has no
target) but per-bulk discrimination: on spl/ilm/ru/sph the head knows the base
rate yet can't tell which bulks are the high ones (spl per-bulk MAE 0.173, ilm
0.118, ru 0.075, sph 0.057; symmetric over/under tails). The per-bulk presence
fraction `f_bulk` is a deterministic function of chemistry -- the integral of
phase stability over the fixed P/H2O sampling distribution, estimated with only
~4pp binomial noise at 128 val draws -- so there is no in-principle ceiling
forcing the observed hedging. Where is the actual ceiling, and which component
is losing the signal?

## Method

Three predictors of `f_bulk`, per phase (all 21, spl/ilm/ru/sph headline),
scored on the val roster (per-bulk MAE + Pearson r, against the per-phase
binomial floor of the truth estimate itself):

1. **Head baseline** -- `sigmoid` of the production ModeHead's presence logits
   on the per-bulk pooled latents (`val_dataset_and_latents`).
2. **Raw-chemistry XGBoost** -- the input ceiling. Features `log10(bulk_frac)`
   with NaN where absent (same convention as `trace_signal_in_latent`); targets
   `within_bulk_frequency` on the train tier (48 draws/bulk, so targets carry
   ~7pp of their own noise at mid f). One regressor per phase, hyperparameters
   from one light shared Optuna study on the headline phases. Feature
   importances test the Ti/Cr hypothesis (ilm/ru/sph compete for the TiO2
   budget; spl rides Cr).
3. **Latent probe** -- same targets, inputs the frozen pooled encoder latents;
   1-hidden-layer MLP (house probe methodology) trained with BCE-with-logits on
   the soft f targets, plus a hyperparameter-free cosine-kNN read.

Routing (advisory verdict, tie margin 0.02 MAE ~ the binomial noise scale):
raw ~ head -> the corpus is the ceiling, enrichment is the lever; raw > head
and latent > head -> head-side fix (per-phase presence-BCE weighting /
capacity); raw > head but latent ~ head -> the encoder dropped the signal,
feed `bulk_frac` to the ModeHead via a skip connection.

    conda run -n alignn_fix python explorations/presence_fraction_ceiling/run_probe.py \
        --checkpoint checkpoints/joint_heads_seed0/heads_best.pt

Sanity anchor: the head arm must reproduce the #41 measurement's headline MAEs
above; if it doesn't, the instrument is mis-wired.

## Result

Run 2026-07-22 on `joint_heads_seed0_heads_best` (30 Optuna trials; full table
and figures in `results/joint_heads_seed0_heads_best/`). Headline per-bulk MAE
(mean over spl/ilm/ru/sph; binomial floors 0.016-0.024):

| arm | headline mean MAE |
|---|---|
| ModeHead | 0.106 |
| raw-chemistry XGBoost | 0.098 |
| latent MLP probe | 0.102 |
| latent kNN | 0.127 |

Both challengers land inside the 0.02 tie margin, on every headline phase
individually as well as the mean (largest single-phase gain: spl 0.173 ->
0.159). The chain loses nothing: the latent probe sits within 0.004 of the raw
ceiling (encoder preserves the signal) and the head within 0.004 of the probe
(head extracts it). But the ceiling itself sits 4-7x above the binomial floor
-- the missing discrimination is not extractable from the 22-token chemistry at
this corpus size and label noise (train targets carry ~4-7pp binomial noise of
their own at 48 draws/bulk, which caps every learner trained on them).

Feature importances: ilm leans on Ti (0.31) as hypothesized; ru/sph spread
over Ca/Ti/K/Na (partial support); spl leans Mg/Fe -- consistent with the
corpus's spinel actually being Cr-free. Side finding: the Cr token is populated
in 0.0% of train bulks (production tokenization runs `include_cr2o3=False`),
and the MAGEMin input was effectively Cr-free too (`bulk_11` Cr2O3 nonzero in
2.1% of rows, corr with f_spl +0.02), so the Cr omission is self-consistent for
this corpus and is NOT the spl gap. It would matter for real rocks, where
chromite rides Cr.

## Decision

The corpus is the ceiling: raw chemistry does not beat the head, so per-phase
presence-BCE weighting, head capacity, and a bulk_frac skip connection are all
dead ends for this failure -- stop architecture work on presence
discrimination. The lever is data: more draws per bulk (cuts the ~7pp target
noise the learners train against) and/or more bulks. Whether that enrichment is
worth scheduling, and the residual hypothesis for spl (fO2 assigned
class-conditionally injects variance the input cannot see), are owner calls
recorded in the journal when this investigation closes.

Follow-up (2026-07-22): the residual hypothesis was confirmed and priced by
`explorations/fe_redox_attribution` -- the binding constraint was the input
basis, not corpus size; the excess-oxygen column alone moves the spinel
ceiling 0.159 -> 0.059. Verdict revision recorded on #41; the redox input
token adopted on #45 supersedes the enrichment lever for this failure.

Reuse note (2026-07-27): every number in this file is stamped to the
pre-regen corpus (train at 48 draws/bulk, the 500-bulk val roster) -- do not
lift them for the traced corpus (24-draw train, 2000-bulk valtest). What
transfers is the code: `binomial_floor` and `bulk_frequency_targets` read
per-bulk draw counts from whatever dataset they are handed, and are the
designated pieces for measuring an instrument's noise floor. They were
promoted from this script into `cnmineral.eval.mode_head` (this script now
imports them), and every ModeHead eval stamps the floor into its
`metrics.json`. The 0.02 tie margin above was this analysis's advisory
routing margin only; no code reads it, and it is unrelated to the
instrument tie band in the results ledger.
