# fe_redox_attribution

## Question

The `presence_fraction_ceiling` exploration left spinel as the worst headline
phase at the raw-chemistry ceiling itself (per-bulk MAE 0.159 vs a 0.024
binomial floor), with fO2 as the residual hypothesis: the label generator saw
an Fe3+/Fe2+ split per bulk -- measured speciation when the source reports it,
else a Middlemost-1989 per-class lookup (`fe_redox.py`, provenance stamped by
`magemin_oxides.py`) -- while the model input carries only total Fe. Oxidation
state strongly controls Fe-oxide stability, so the labels hold variance no
input feature can see. Before the #45 regen spec locks the fO2 design (#46:
promote redox to an input token, to a sampled condition, or document the
status quo), two measurements attribute how much of the spinel gap redox
invisibility actually owns.

## Method

Both reuse the ceiling probe's machinery (targets, XGB protocol, scoring) via
`run_measurements.py`:

1. **Provenance split.** Per-bulk `fe_provenance` read from the corpus draws
   tables and joined to per-bulk errors of every arm (production head presence
   sigmoid + the XGB bases below). For assumed-class bulks the generator's
   redox is a deterministic function of `qap_name`, which chemistry largely
   determines, so those labels should be *more* predictable; measured bulks
   carry exogenous redox variance. Scored on the val roster and, for power, on
   the train-tier 20% holdout (the XGB early-stopping set -- mildly
   optimistic, but the contrast is between groups scored identically).
   Per group x arm: n, mean f, per-bulk MAE, Pearson r, binomial floor, plus a
   bootstrap 95% CI on the spl measured-minus-assumed MAE delta and a
   supergroup-composition table (provenance correlates with the historical
   wet-chemistry mafic bias, so the raw contrast could be a class artifact;
   within-supergroup contrasts are reported where both groups have n >= 20).

2. **True-input ceiling.** The XGB arm re-run under the generator's own input
   basis. Per-bulk features come from `bulk_11` with the per-draw H2O column
   dropped and the 10 anhydrous oxides renormalized (H2O is the sampled
   condition; the anhydrous vector is bulk-constant to float precision --
   asserted). Bulk-basis features stay raw fractions: trees are
   rank-invariant, and a log transform would map excess-O = 0 (fully reduced,
   informative) to missing. Four bases:

   | basis | features | role |
   |---|---|---|
   | `token22` | log10 cation fractions (production input) | sanity anchor: must reproduce spl MAE ~0.159 with the recorded hyperparameters |
   | `bulk10` | anhydrous generator basis incl. excess-O | the true-input ceiling |
   | `bulk9` | bulk10 minus excess-O, renormalized | bulk10 - bulk9 = the redox delta inside the oxide basis (bulk10 - token22 also confounds traces and the cation/oxide change) |
   | `token22_xo` | token22 + raw excess-O column | prices exactly what promoting redox to an input would add to the current tokenization |

   `token22` reuses the tuned hyperparameters recorded in
   `presence_fraction_ceiling/results/`; each new basis gets its own shared
   Optuna study (same protocol, headline-phase MSE objective).

```
conda run -n alignn_fix python explorations/fe_redox_attribution/run_measurements.py \
    --checkpoint checkpoints/joint_heads_supportmag_seed0/heads_best.pt
```

Reading the two measurements together: a markedly worse measured group AND a
bulk10-over-bulk9 gain confirm redox invisibility as the spinel driver and
size the recoverable share; matched groups with no basis gain route #46 to
"status quo, documented".

## Result

Run 2026-07-22 on `joint_heads_supportmag_seed0_heads_best` (30 trials per new
basis; full tables and figures in `results/`). Anchor reproduced exactly
(token22 spl MAE 0.159). Val-roster spl per-bulk MAE by basis:

| basis | spl MAE | spl r |
|---|---|---|
| token22 (current input) | 0.159 | +0.76 |
| bulk9 (generator basis minus excess-O) | 0.158 | +0.74 |
| bulk10 (generator basis) | 0.059 | +0.94 |
| token22_xo (current input + excess-O) | 0.072 | +0.92 |

The excess-O column is essentially the entire spinel gap: removing it from the
generator's own basis (bulk9) reproduces the 22-token ceiling to 0.001, and
appending that single column to the current input recovers ~87% of the distance
to the true-input ceiling. The gain is not spinel-only -- the same
bulk10-over-bulk9 pattern moves every Fe-bearing phase (ilm 0.118 -> 0.072,
ol 0.040 -> 0.023, g 0.041 -> 0.027, plus ru/ep/cpx/opx), while bulk9 matches
token22 throughout. excess-O is the rank-1 XGB feature for spl in both
redox-bearing bases.

Provenance split (train holdout, n = 786 assumed / 618 measured): under the
current input, measured-redox bulks run +0.109 MAE worse on spl than
class-assumed bulks (0.217 vs 0.109, CI [+0.094, +0.122]), and the penalty
persists inside every supergroup with n >= 20 on both sides (e.g.
mafic-volcanic 0.252 vs 0.113) -- not a class-composition artifact. Group
binomial floors are equal (0.038-0.041), so measured labels are not noisier;
the penalty is discrimination loss (r 0.54 vs 0.90). Handing the predictor the
excess-O column collapses the penalty to +0.023 (bulk10) / +0.022 (token22_xo)
on the identical bulks. Val roster shows the same picture at lower power.

Side finding: assumed-class bulks also improve under excess-O (0.109 -> 0.053)
-- the class lookup keys on `qap_name`, an external classification that
chemistry does not fully determine, so even rule-assigned redox is partly
input-invisible. Residual: the measured group still sits ~2x its floor under
bulk10 (0.076 vs 0.039); one scalar does not carry everything, but it carries
most of it.

## Decision

Both #46 predictions confirmed: errors concentrate exactly where redox is
exogenous, and the redox column recovers most of them -- ~87% of the spinel
gap plus broad Fe-bearing-phase gains from a single appended input feature.
The status-quo option is dead on these numbers.

Decision (2026-07-22, recorded on #45's ledger): promote Fe redox to an
optional input token at the regen, with uniform optional-token semantics --
fed where the generator saw a value, padded/masked otherwise, so unknown-redox
inference marginalizes implicitly through the probabilistic heads. No explicit
inference-time redox path: imputation-readout and pin-and-sweep already fall
out of the generic token machinery, and sampling redox at inference would
duplicate what calibrated heads carry in one pass. Spec knobs deferred to #45:
token parameterization, fed-vs-padded for class-assumed bulks, train-time mask
rate, and the eval layer's padded-redox calibration check + redox-availability
stratification.
