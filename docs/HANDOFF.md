# Handoff: v1 trained, what's next

Orientation for the next agent picking up CrabNet-Mineral after the
`v1-trained` tag. Prioritized roadmap; reflects decisions made 2026-05-22.

For *how the model works* see `docs/MODEL.md`. For *what happened* see
`docs/JOURNAL.md` (2026-05-22 entry). For *why* decisions were made see the
ADRs, especially 006 (architecture) and 007 (the H2O leak fix).

## Where v1 stands

`v1-trained` tag: K=5 deep ensemble, honest 10-oxide model, at
`checkpoints/base_v1_ensemble/`. Test: T MAE 15.4 C, P 1.76 kbar, H2O 0.535
wt%; 7/18 phases R^2 > 0.5 (anhydrous majors fsp/q/cpx/ol 0.91-0.97; hydrous
and rare phases weak). Two corpus-handling bugs were caught and fixed during
the build (row-level split leak, H2O target leak) -- see JOURNAL.

v1 is a preliminary model. It predicts modal *abundance* + formation
*conditions*; it does not yet predict per-mineral *composition*, which is the
main thing standing between it and the project's stated contribution.

## Priority tiers

### Tier 1 -- top priority

| id | item | depends on |
|---|---|---|
| B1 | **CompHead** -- per-mineral solid-solution chemistry (`[n_minerals, 11]` oxide wt%) + mineralML empirical-grain fusion. The step that makes the ModeHead irreducible to MAGEMin. (Does **not** fix amp abundance -- B2 below.) | corpus regen |
| B8 | **GsysHead** -- regenerate corpus with MAGEMin `G_sys` (Gibbs energy of the system), add a prediction head. Elevated to Tier 1 by the user. | corpus regen |
| B2 | ~~**`amp` diagnostic**~~ -- **done 2026-05-22** (`scripts/b2_amp_diagnostic.py`, JOURNAL). amp per-path R^2 = 0.22 = ceiling 0.67 x R^2_bulk 0.32. Ceiling is amp-specific (amp/fl/liq are the only P,H2O-sensitive majors with large within-bulk path spread). h2o_tot leak bought +0.13. Bulk-level R^2 negative for amp-rich classes (gabbro, diorite, andesite). CompHead does not address either term -- amp abundance is largely stuck. | -- |
| B3 | **modal haze / sparsity** -- the Dirichlet ModeHead smears ~1% mass onto phases that are exact-zero in truth. A sparsity-inducing or zero-inflated head. ModeHead-only change. | none |

### Tier 2 -- quick win, do anytime

| id | item | notes |
|---|---|---|
| A2 | ~~**calibration**~~ -- **done 2026-05-22** (`scripts/07_calibrate.py`, `checkpoints/base_v1_ensemble/calibration_v1.json`, JOURNAL). Guo et al. 2017 sigma-scaling, NLL-fit per axis. Adopted T=0.655, P=0.943, H2O=0.932. PIT diagnostic reframed the problem: T was genuinely loose (KS 0.17 -> 0.12, fixed), P and H2O were already near-calibrated (their PIT lumpiness is largely discrete-target artifact -- 4 P values / 2 H2O values per bulk). Inference wiring (one flag in 06_eval / the notebook reading the JSON) still to do. |

### Tier 3 -- after Tier 1

- B5 polymorph head (Ky/Sil/And, Qz/Crs/Trd disambiguation) -- v2 territory
- B6 ALIGNN structural stream -- v2
- B7 fO2 sweep -- add oxygen fugacity as a sampled FC path axis (currently
  frozen class-conditional via `fe_redox.py`). v2.
- A5 finish/extend the HPO sweep; B4 (K=10 ensemble) folds in here. Not
  urgent -- v1 is preliminary.

## The corpus-regeneration bundle (B1 + B8)

B1 and B8 both need extra columns pulled from the MAGEMin / PetThermoTools FC
output: per-mineral `Comp_wt` (B1) and `G_sys` (B8). **Do this in one corpus
regeneration, not two.**

**Gating resolved (2026-05-22): full FC re-run required.** All 731 chunks in
`data/fc_corpus_cache/` (and 40 in `_legacy/`) carry meta + 11 oxides +
per-phase modal wt% only.
`scripts/production/02_generate_corpus.py:final_modal_from_res` extracts
`res["mass_g"]` and discards the rest of the PetThermoTools FC return dict;
the per-phase compositions and `G_sys` were computed in-memory and never
persisted. So B1 + B8 must re-run FC -- expect the 31-hour multi-crash path
plus the `maxtasksperchild=1` + `chunk_size=25` + per-path Julia GC +
`caffeinate -i` workaround (JOURNAL 2026-05-17). Needs explicit
time/infra buy-in before starting.

Once the corpus carries `Comp_wt` + `G_sys`: add CompHead + GsysHead (both
additive heads off the shared latent, same pattern as ModeHead/CondHead),
re-HPO, retrain the K=5 ensemble. Fold in C1 (config dedup) at that point.

## A1 -- investigated and dropped, do not redo

A1 was "make H2O an optional model input to recover the hydrous-phase modal
regression." It was investigated on 2026-05-22 and **dropped**:

- complete.csv's measured `h2o_tot` is 1.1% populated for corpus bulks.
- Where present, it does **not** track the FC `h2o_path_wt` (corr -0.35).
- The modal labels were computed by FC *at* `h2o_path_wt`; a water input
  decoupled from `h2o_path_wt` cannot predict them. The old `h2o_tot` input
  "worked" only because it *was* `h2o_path_wt` (the leak).
- `h2o_path_wt` is a sampled experimental knob, not a recoverable rock
  property. No honest input substitutes for the leak.

The hydrous-phase modal weakness is therefore largely **irreducible**, and
partly a **metric artifact**: within a bulk the 8 FC paths have different
hydrous-phase content but identical model input, so the Dirichlet correctly
predicts the bulk-average and per-path R^2 eats the spread. Reframe the
evaluation (judge the Dirichlet by NLL/calibration, not R^2 against
individual path labels) rather than trying to "fix" it. See ADR 007.

## Dropped / deferred (do not spend time on)

- A3 two-stage MAGEMin baseline -- was an observation, not needed.
- A4 EDS-map cross-modal eval -- far down the line.
- A6 v0 vs v1 comparison -- not meaningful (different corpora, different
  cation count), dropped.

## Repo hygiene

- C1 -- `base_v1.yaml` and `hpo_v1.yaml` share a duplicated `data:` block;
  factor it out. Do this during the Tier-1 corpus work.
- C3 -- HPO is slow on MPS (~30+ min/trial at d=512). An HPC/CUDA box would
  make wider sweeps practical; overnight MPS runs are adequate for now.

## Blue-sky (not scoped)

Trace-element / REE partitioning per phase; forward+inverse unification with
a physics-informed consistency loss; active-learning loop (use ensemble
epistemic sigma to choose where MAGEMin samples next); fine-tuning to natural
samples; generative P-T-t-X petrogenetic histories.

## Key files

- `docs/MODEL.md` -- how the v1 model works
- `docs/JOURNAL.md` -- chronological log; 2026-05-22 is the v1 build
- `docs/decisions/006*, 007*` -- architecture + H2O-leak decisions
- `src/cnmineral/models/` -- encoder, heads, model, losses, dataset, training, objective
- `scripts/03..06_*.py` -- train / HPO / ensemble / eval
- `checkpoints/base_v1_ensemble/` -- the v1 K=5 ensemble + eval_v1.json
- `data/splits_v1.json` -- frozen bulk-level train/val/test split
- `notebooks/v1_eval.ipynb` -- eval dashboard + simulated-user demo
