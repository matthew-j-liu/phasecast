# Decision 007: drop h2o_tot from model inputs (H2O target leak)

**Date:** 2026-05-22
**Status:** Active

## Decision

The v1 model takes **10 anhydrous oxide wt%** as input. `h2o_tot` is dropped.
The CondHead infers H2O genuinely from anhydrous chemistry.

## Context

In the FC corpus the input oxide `h2o_tot` equals the CondHead H2O target
`h2o_path_wt` (corr 1.0 -- the corpus set each bulk's water to the FC path's
sampled H2O). Feeding `h2o_tot` in target-leaked the H2O axis: the CondHead
was copying an input, not inferring it (fake H2O MAE 0.034). Dropping it makes
H2O a genuine prediction (honest test MAE 0.535 wt%). See JOURNAL 2026-05-22.

## Consequences

- Model input is 10 oxides / 9 cations (no H token). `V1_CATIONS_ORDER` in
  `dataset.py`; v0's `TrainMageminDataset` is untouched.
- Hydrous-phase modal R² regressed (amp/bi/mu/fl/ne, ~0.13-0.25 each) --
  `h2o_tot` was also a genuine predictor for water-dependent phases. This cost
  is accepted: the leaked feature was informative *because* it was the answer.

## Alternatives considered

- **Keep h2o_tot, CondHead predicts only T+P.** Rejected -- drops the H2O
  inverse-inference axis entirely.
- **Use complete.csv's measured `h2o_tot` as the input** (was "Option C").
  Rejected after checking the data: complete.csv `h2o_tot` is 1.1% populated
  for corpus bulks, and where present it does not track the FC `h2o_path_wt`
  (corr -0.35). Measured current water is decoupled from the sampled FC
  magmatic water that the modal labels were computed from, so it cannot
  recover the hydrous-phase regression. `h2o_path_wt` is a sampled
  experimental knob, not a rock property -- there is no honest input that
  substitutes for the leak.
- **Optional measured-water input token.** Rejected for the same reason -- no
  useful water signal exists to make optional.

## Revisit when

If a future corpus stops setting `h2o_tot := h2o_path_wt` and instead carries
a genuinely independent measured water column, reconsider adding it as input.
Not expected.
