# Decision 005: Production label generation via PetThermoTools fractional crystallization

**Date:** 2026-05-15
**Status:** Active

## Decision

Regenerate the training corpus by running **PetThermoTools fractional
crystallization** per bulk, sampled across a 2D grid of (P, H2O) per bulk,
with fO2 set class-conditionally by the existing `fe_redox.py` strategy.
The label for each (bulk, P, H2O) combination is the **in-situ modal vector
at a per-bulk T_FC_end**, where T_FC_end is the highest T at which the
bulk's own residual silicate melt fraction falls below 5 wt% (snapped to
nearest 25 °C).

This supersedes the v0 corpus generation procedure (Putirka 2008 (T, P)
sampling at single-snapshot MAGEMin equilibrium).

## Why this decision

ADR 004 resolved the question of *whether* to use FC versus near-solidus
equilibrium (answer: FC). This ADR locks *how* to use FC for production.

The design is anchored on the eq-vs-frac v2 experiment
(`docs/JOURNAL.md` 2026-05-15) which (a) showed FC modal vectors diverge
meaningfully from closed-system eq at the same T for 6 of 11 classes, and
(b) operationalized "subsolidus" as a class-aggregate liq < 5 wt% threshold.
Production lifts the design from class-aggregate to per-bulk to eliminate
the ~11% audit attrition seen in v2.

## Binding choices

### Per-bulk T_FC_end

For each bulk: run one FC sweep at the bulk's class-default (P, H2O),
inspect the per-step liq trajectory, set `T_FC_end_bulk` = highest T (snap
to 25 °C) at which liq < 5 wt%. Floor at MAGEMin's calibration limit per
class (~500 °C felsic, ~600 °C mafic).

**Reason:** within a class, H2O varies up to ~3x (e.g. andesite 2.8 - 4.8
wt% p25-p75). The wet solidus depression scales with sqrt(X_H2O), so one
T per class is wrong for a fifth of the bulks. v2 showed this directly:
basalt and gabbro had 30 % audit attrition at class-level T. Per-bulk T
costs one extra FC sweep per bulk; with the main run already doing ~16
sweeps per bulk this is a 6 % overhead and zero attrition.

**How to apply:** the per-bulk T discovery is a pre-pass that produces a
`T_FC_end_bulk` column in the bulks table. The main run reads it.

### (P, H2O) sampling per bulk

Cross-product of `n_P` × `n_H2O` paths per bulk:

- P axis: sampled across the per-class Putirka 2008 range from
  `src/cnmineral/data/tp_priors.py`. Reuse the existing range table; just
  drop the T axis.
- H2O axis: sampled across the per-class range from
  `src/cnmineral/data/h2o.py`. Reuse the strategy as-is.

`n_P` and `n_H2O` jointly set the path count per bulk. Total compute
scales linearly. **Specific path count is an implementation detail (see
"Open implementation questions"), not an ADR-level decision.**

### fO2 stays class-conditional

Use `fe_redox.py`'s class-default Fe3Fet ratio per bulk; do not sweep fO2
across paths in v1. fO2 sweep is deferred to v2.

**Reason:** v0 already worked with fixed class-conditional fO2 and 14 of
17 phases reached R² > 0.5. Adding an fO2 axis would double the compute
budget for unproven marginal gain. ADR 006 (when written) can revisit.

### Single endpoint per path

Each (bulk, P, H2O) path produces one training label: the in-situ modal
vector at `T_FC_end_bulk`. Do not emit per-step snapshots along the
cooling path.

**Reason:** path-along-cooling multimodality compounds with (P, H2O)
multimodality and inflates the corpus 5-10x for unproven gain. The (P, H2O)
sampling already gives the multimodal posterior we want. If v1 training
shows the posterior is too tight, add T-snapshot multimodality in a v1.5
iteration.

### Label format: in-situ modal vector

For each path, the label is the **accumulated cumsum solids + residual
liq + residual fl** at `T_FC_end_bulk`, normalized to 1.0. Same procedure
as the v2 experiment's FC leg. Includes `liq` and `fl` channels in the
modal vector.

**Reason:** the L1 comparison in v2 was over in-situ phases (including
liq + fl). The model's job mirrors that comparison — predict the in-situ
assemblage at the bulk's T_FC_end. Post-hoc "drop liq + fl and renormalize"
is a downstream choice the user can make at inference time; the training
labels keep all information.

### Vocabulary discovered fresh

The phase vocabulary for the production corpus is discovered from the FC
output, not inherited from v0's 25-phase Tier A. v2 already showed FC
produces `ep` and FC vs eq schemas differ slightly; the production schema
may differ further at scale.

## Open implementation questions

These are concrete choices the production code makes; they don't bind
architecturally and can shift based on compute headroom.

- **Path count per bulk.** 8 (e.g. 2×4 or 4×2) → ~175k labels, ~9 hr
  wall on 8 python workers. 16 (4×4) → ~350k labels matching v0
  density, ~18 hr. Pick once initial smoke run lands.
- **Parallelization.** ptt's intra-process multi_processing dispatches
  across Julia workers per parameter sweep (e.g. `label='P_bar'` with
  an array of P values). Python-level multiprocessing across bulks
  stacks badly with ptt's own workers (52+ processes on a 13-worker
  Julia config). Likely path: python-level worker pool (4-8 workers),
  ptt's multi_processing=False inside each. Measure on a smoke chunk.
- **Output format.** Extend the existing `train_magemin.h5` schema with
  a `path_idx` per (bulk, path), or use a new `train_fc_v1.h5`. New
  file is cleaner and preserves v0 baseline. Use a new file.
- **Chunked orchestrator structure.** Mirror `scripts/orchestrate_magemin.py`
  pattern: chunk by bulk, write per-chunk parquets to `data/fc_cache/`,
  retry crashed chunks via subdivision (same recovery story as v0).
- **Per-bulk T_FC_end pre-pass storage.** Write to
  `data/bulks_v1_with_T.parquet` or similar; the main orchestrator
  reads it.
- **Audit policy for the pre-pass.** Some bulks may not satisfy liq <
  5 wt% within the calibration floor (the v2 11 % tail). For
  production: use the floor T for those bulks anyway, tag with a
  `T_FC_end_attrition` flag in the corpus, and let evaluation
  optionally drop them.

## Revisit when

- The v1 corpus is built and BaseModel retrains. Check per-class R²
  and mean corpus liq fraction. If liq mean > 5 wt% or per-class R²
  collapses below v0 for any class with > 1 % presence, reassess one
  of the binding choices.
- fO2 sweep adoption (currently class-conditional). Open question: does
  doubling compute for fO2 multimodality move per-phase R²? Plausible
  v2 ADR.
- Per-step (T-snapshot) labels. Same question: does it help volcanic vs
  plutonic disambiguation? Plausible v1.5 ADR.

## Cross-reference

- Predecessor: ADR 004 (PetThermoTools adopted, Phase B resolved)
- Experiment: `docs/JOURNAL.md` 2026-05-13 (v1, dropped) and 2026-05-15
  (v2, decision-locking)
- Diagnostics: `docs/eq_vs_frac_diagnostics.md`
- Existing pipeline patterns to mirror: `scripts/orchestrate_magemin.py`,
  `scripts/retry_failed_chunks.py`, `scripts/parse_magemin.py`
- Strategy modules to reuse: `src/cnmineral/data/tp_priors.py` (P axis
  only), `src/cnmineral/data/h2o.py` (H2O axis), `src/cnmineral/data/fe_redox.py`
  (fO2)
