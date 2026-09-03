# magmadb_analogs

## Question

The Track B output layer wants to show a user real rocks with *measured*
(P, T, fO2) near their query, as transparent evidence, plus a coarse out-of-
distribution flag (ADR 004 "Output stance", "Latent-neighbor
analogs"). Our corpus has no measured conditions. MagmaDB (Candioti et al. 2026)
is a compilation of arc-magma crystallization experiments that does. Two
questions: can MagmaDB starting bulks be embedded in the frozen SSL encoder
latent at all (it is majors-only), and does the latent neighborhood of a test
andesite look petrogenetically coherent? Then: is 484 experiments enough, or is
a larger natural-sample thermobarometry compilation worth assembling?

## Method

Tokenize MagmaDB starting bulks exactly like the production gard pipeline (8
majors; Cr padded to match `include_cr2o3=False`; all 13 traces padded), embed
through `checkpoints/pretrain_v3_logmse_std_seed0` alongside the full gard2019
igneous corpus, and unit-normalize. Per gard bulk, take the cosine to its
nearest MagmaDB experiment as a coverage / OOD score. Retrieve k=5 latent
neighbors for a few andesites and read off their measured conditions +
assemblages. Figure: TAS + SiO2-Mg# composition overlays, a latent PCA, and the
nearest-analog cosine distribution by rock type.

The MagmaDB CSV ships outside the repo, in the ArcThermoML download
(`~/Downloads/lcandiot-ArcThermoML-676d770/data/lab_experiments/`); pass its
path with `--magmadb`.

## Result

- **Embeddable, no imputation.** Every row maps to 8 populated major tokens.
  Majors-only is in distribution: ~24% of the gard pretrain corpus is itself
  majors-only.
- **Coherent neighborhoods on the arc slice.** Andesite neighbors track query
  SiO2 and return plag+cpx(+opx/amph) assemblages with real conditions.
- **Sparse and biased.** 510 rows but only **78 distinct starting bulks**
  (median 3 conditions each, max 51), so k=5 neighbors often collapse onto 1-2
  compositions. Coverage (frac of gard with nearest-analog cos >= 0.7): all
  igneous 0.29, andesite 0.41, basalt 0.42, rhyolite/granite 0.08, ultramafic
  0.16. Within SiO2 52-58, subalkaline median cos 0.66 vs alkaline 0.52. MagmaDB
  is a dense island in arc mafic-intermediate space, sparse-to-absent for
  felsic / alkaline / ultramafic.

## Decision

Use MagmaDB as-is for the shown-not-reweighted analog panel + OOD flag on the
arc basalt-andesite slice; no larger compilation yet (gated on Track B heads
existing + a demonstrated case where a panel would have corrected a scenario).
Verdict recorded in [ADR 004](../../docs/decisions/004_petthermotools_for_fractional_crystallization.md) "Output stance"
(Output stance).

Artifacts: `results/magmadb_vs_gard.png`, `results/coverage_metrics.json`.
Rerun: `conda run -n mineral python explorations/magmadb_analogs/run_analogs.py --magmadb <path>`.
