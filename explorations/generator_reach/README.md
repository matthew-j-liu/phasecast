# generator_reach

## Question

Can equilibrium crystallization on a benchmark rock's own chemistry reach
its measured mineralogy at all, and where it cannot, which failure explains
it per mineral category: the right conditions buried under the draw prior,
the wrong stage along the cooling path, or the wrong recipe altogether? The
generator run on the 640 evaluated benchmark rocks, model-free (#78, epic
#74 step 2). The bins decide which lever, if any, a candidate corpus is
worth: buried conditions send water and pressure to the prior search (#79),
wrong stage sends the rung, and wrong recipe says equilibrium is the wrong
generator for that category (#22's question).

## Method

The rulings (owner, 2026-09-09/10 and 09-15), recorded here because the
earlier record sat in a gitignored `tmp/README.md`:

- **Rocks.** The evaluated set, `benchmark.clean & benchmark.has_magnitude`
  from `cnmineral.eval.real_rock.load_benchmark` (grade 1 and 2 with a
  quantified mode, 640 rocks), in benchmark order. `compositions.py` builds
  `results/compositions.csv` from `data/real_mineralogy/samples.csv`: an
  integer `bulk` index (benchmark ids are strings and many collide
  numerically with gard2019 ids), the reported majors (no Cr2O3, no water
  column), the split subset, the crystallinity, and the iron class. Nothing
  reads `data/bulks/gard2019.parquet`.
- **Iron split.** Total iron (`feo_tot`, or `fe2o3_tot` on the five Pinatubo
  rows) splits by `ClassRatioStrategy` (Middlemost 1989 Fe3+/sum Fe by rock
  name, `cnmineral.data.fe_redox`). The benchmark carries no rock name and
  the repo has no classifier, so each rock is classified on the total alkali
  versus silica diagram (Le Bas et al. 1986; polygons transcribed from
  pyrolite) on its ten anhydrous majors renormalized to 100. TAS names are
  the ratio table's keys except two: basaltic trachyandesite takes basaltic
  andesite's 0.25 (the silica-matched neighbour between trachybasalt 0.20 and
  trachyandesite 0.35; 36 rocks) and picrobasalt takes picrite's. A
  composition outside every field (one rock at 90 wt% SiO2) takes the nearest
  field and is flagged `tas_inside = False`. Census on the 640: trachyandesite
  180, rhyolite 176, trachydacite 147, dacite 46, basaltic trachyandesite 36,
  andesite 34, basaltic andesite 16, basalt 4, trachybasalt 1.
- **Draws.** 128 per rock from the broad uniform box the production corpus
  used, `draw_conditions` in `cnmineral.data.condition_proposal` (P 0.5 to
  12 kbar, H2O 0 to min(8, S(P)) wt%, Latin hypercube, density `q_density`),
  rng `default_rng(seed + bulk)` with production's seed 20260714. HEAD's
  `magemin_corpus.build_inputs` hard-codes the trace-conditional proposal, so
  `run_generator.py` carries its own expansion; H2O is injected by
  renormalizing the anhydrous oxides to 100 - H2O, as production does.
- **Ladder.** 1200 to 600 C in 20 C steps (`T_LADDER`, 31 points), one
  Julia call per 25 rocks through `magemin/run_magemin.jl` with the
  production script's timeout and crash bisection (`run_group`, imported
  from `scripts/build/generate_mineralogy_corpus.py`). The exploration's
  parser keeps every converged point with its melt fraction; production's
  truncates at 30% melt. Rungs are read as production reads them: the
  highest-T converged point with `liq` below 0.50 / 0.30 (`fl` is not melt);
  a draw that never gets there is attrition at that rung.
- **Normalization, as the paper.** Measured share: the crystals-renormalized
  mass share over the categories the rock quantifies (`_truth_arrays`;
  volume to mass by per-mineral density, groundmass dropped, trace-only
  categories excluded). Simulated share: MAGEMin's raw phases onto the seven
  categories (`fsp`; `cpx` + `opx`; `amp`; `bi`; `ol`; `spl` + `ilm`; `q` +
  `trd` + `crst`), melt and fluid dropped, every other solid counted as
  `other`, then renormalized over the same quantified categories, as
  `conditional_mean(alpha7, numeric)` scores the head. A point with no
  crystal mass in those categories has share 0 everywhere, as the head's
  conditional mean does on an empty mask. Groundmass never enters either
  side.
- **Three distances per (rock, quantified category)**, absolute error from
  the measured share: `d_marginal`, the mean share over the pooled rung-50
  and rung-30 rows of all draws (the marginal the training corpus presents);
  `d_best_draw`, the minimum over (draw, rung) rows, chosen per category
  independently (owner, 2026-09-15); `d_best_step`, the minimum over every
  converged step of that category's best draw. No melt-fraction rule in any
  of the three. `d_best_any_step` (the minimum over all draws and steps) and
  `d_q10_draw` (the 10th percentile over rung rows, so a buried verdict can
  be read as a tail or a mass of draws) are diagnostic columns.
- **Bins at tau** (owner, 2026-09-15: 0.05 absolute share, printed also at
  0.02 and 0.10): `reproduced` if `d_marginal <= tau`, else `buried` if
  `d_best_draw <= tau`, else `wrong_stage` if `d_best_step <= tau`, else
  `wrong_recipe`. A rock with no draw at either rung has no marginal and
  cannot be reproduced. Fractions are reported over all rocks and over rocks
  quantifying at least two categories (a single-category rock's share is 1
  wherever any simulated mass exists, so it is reached trivially; 17 of the
  640). The CDFs of the three distances are drawn regardless of tau, with
  tau and the production head's zero-shot per-category error
  (`explorations/corpus_scaling/results/depth_ladder.csv`, tag
  `joint_heads_allrock_tuned_seed0`) as reference lines.
- **Diagnostic only, afterwards.** The melt fraction at the best step
  against `1 - total_crystals_pct / 100` (a volume fraction against a mass
  fraction, not converted).
- **No model in the loop.** Production and the paper's numbers are untouched;
  the agent reports the bins and stops. Which way to move pressure and
  water, and whether #79's search is warranted, is the owner's decision.

Commands, from the repo root with Julia and MAGEMin installed:

```
python explorations/generator_reach/run_generator.py --smoke        # rock 0, two draws, ~10 s
python explorations/generator_reach/run_generator.py                # 640 x 128 x 31, about 6 h, resumable
python explorations/generator_reach/run_read.py                     # distances, bins, figures
python explorations/generator_reach/run_read.py --figures-only
```

The run writes under `work/broad128/` (gitignored): `chunk_XXXX_draws.parquet`
(one row per draw: `P_kbar, H2O, q`, the input oxides, `n_failed, n_points,
oversaturated, T_end_50, T_end_30, attrition_50, attrition_30`),
`chunk_XXXX_ladder.parquet` (one row per converged point and phase instance:
`bulk, draw, T_celsius, liq_frac, phase, instance, mode_wt`) and
`manifest.jsonl`; the read caches `points.parquet` beside them. Committed
under `results/`: `compositions.csv`, `distances.csv` (one row per rock and
quantified category with the three distances and the draw, rung, step and
melt fraction behind each minimum), `summary.json` (per-category bin
fractions at each tau, quantiles, the reference errors, the run facts) and
`reach_cdf.png`, `reach_bins.png`, `reach_melt.png`. The smoke run passed
2026-09-15 under Julia 1.12.6: 62 points, all converged, 9.6 s wall.

## Result

Run 2026-09-15/16: 640 rocks x 128 draws x 31 steps, 2,539,520 points,
6.1 h wall, no solver crashes, 4% of points failed, every draw reached both
rungs, 3.5% of draws oversaturated at the liquidus. Fractions of rocks per
category at tau 0.05 (`results/summary.json` holds 0.02 and 0.10 and the
rocks quantifying at least two categories, which read the same):

| category | n | reproduced | buried | wrong stage | wrong recipe | median error, marginal / best draw / best step |
|---|---|---|---|---|---|---|
| fsp | 601 | 0.16 | 0.71 | 0.05 | 0.08 | 0.147 / 0.002 / 0.002 |
| px | 292 | 0.45 | 0.50 | 0.03 | 0.02 | 0.056 / 0.001 / 0.001 |
| amp | 312 | 0.23 | 0.69 | 0.03 | 0.05 | 0.124 / 0.003 / 0.003 |
| bi | 322 | 0.41 | 0.39 | 0.04 | 0.16 | 0.064 / 0.002 / 0.001 |
| ol | 79 | 0.22 | 0.23 | 0.13 | 0.43 | 0.141 / 0.061 / 0.030 |
| opq | 559 | 0.31 | 0.25 | 0.10 | 0.34 | 0.076 / 0.043 / 0.030 |
| q | 148 | 0.11 | 0.59 | 0.17 | 0.14 | 0.171 / 0.006 / 0.005 |

- Buried conditions dominate the silicates: some draw reaches within 0.05
  for 80 to 95% of feldspar, pyroxene, amphibole and biotite rocks while the
  marginal reaches for 16 to 45%. The hits are a mass for pyroxene and
  biotite (a tenth of a rock's draws reach for 80% and 66% of rocks) and a
  tail for feldspar and amphibole (42% and 21%).
- The rung is not the lever: scanning the best draw's whole path adds 3 to
  17 points, and the melt fraction at the best step barely tracks the
  rock's crystallinity (Spearman 0.04 to 0.27, `results/reach_melt.png`).
- Olivine and Fe-Ti oxides are recipe-limited: no draw at either rung comes
  within 0.05 for 43% and 34% of their rocks, and the best step on the best
  draw still misses by about 0.1.
- The categories disagree on where the hits sit. Feldspar, pyroxene,
  amphibole and biotite hits are at high pressure and moderate water
  (median 8 to 10 kbar, 3.6 to 4.7 wt%). Oxide, olivine and quartz hits are
  low and dry (median 1 to 3 kbar, 1.3 to 2.4 wt%).
- Read jointly (scratch, not in `run_read.py`): one (draw, rung) reaches
  every quantified category of a rock within 0.05 for 28% of rocks and
  within 0.10 for 59%; the joint best row halves the typical rock's mean
  error against the marginal (0.058 vs 0.114); its conditions spread over
  the whole box with the box's own median pressure and water; the oxides
  block the joint hit most often.
- Not measured here: reading each rock at its own melt fraction (the
  rulings deferred it), and joint reach over the four silicates alone.

## Decision

Owner, 2026-09-16. Pursue the coherent subset, feldspar, pyroxene, amphibole
and biotite (about 90% of the crystal cargo on the typical rock, hits in the
same direction), at high pressure and relatively wet. #79 becomes a screen
of candidate (P, H2O) priors ranked by a scalability score written before
any corpus is generated: the per-doubling slope of the recalibrated val
share error on the four categories across a short ladder at fixed fitting
depth, with a level check against the production-prior cells at matched
size and seeds on the endpoints of any winner. Stage 0 ranks priors by
reweighting this run's broad-box draws (no MAGEMin); stage 1 generates the
top schemes at 312 and 1,250 compositions in the production table shape;
stage 2 extends survivors to 5,000 with three seeds. Lawrencium readiness
for stage 2 is the next session's discussion. Direction for the oxides,
olivine and quartz is not chosen; they are reported beside the score.

Superseded 2026-09-18 by `explorations/prior_search/`: the prior screen
and the predictability read, both on this run's draws, found no condition
prior that moves the silicate labels, and no single condition that
reaches all four silicates for 38% of rocks. The scaling pair changes
ruler first (journal 2026-09-18).

## Does the equilibrium label track rocks? (2026-09-29, #85)

#79 read how far the simulator's average share sits from the measured share
(its level), never whether its rock-to-rock variation follows the measured
share (its signal), which is what a recalibration layer can use and what
scaling depends on. `run_signal.py` reads the signal on this run's rows,
model-free, beside the 5,000-composition head at 224 steps. Seconds; no
MAGEMin, no training. No gate: the owner writes it after the read.

- **Sources.** Per-rock shares over the rock's measured mineral list:
  - the simulator's average over the pooled rung rows (the broad box);
  - the #79 rows rebuilt per rock with the prior_search functions: the
    best rectangle, the forest window, and the two oracles, own hit
    window and hit rows. They reproduce `prior_search/results` and
    `results/distances.csv` to 1e-16;
  - a constant, the train-mean measured share renormalized over the
    mineral list;
  - the head (`depth224_allrock_n5000`) zero-shot, recalibrated (matrix,
    weight decay 1.0, fit on train), and the head given no chemistry.
- **Metrics,** per category over the val rocks quantifying it:
  - `mae`;
  - `pearson_r` across rocks;
  - `mae_linear_fit`, the error after a per-category least-squares line
    from the source's share to the measured share, fit on the train
    rocks, then clipped and renormalized over the mineral list. Sources
    already fit to train labels get none.

  Bands are a unit bootstrap (95%, 2,000 resamples) over the 39 val
  units, with the fits held fixed, and every source's difference from the
  constant is computed on the same resample.
- **Ruler hint.** The 63 paired DS 1099 rocks outside test (61 train, 2
  val, two tuff eruptions) are read under both bases with pyroxene and
  olivine dropped. Two numbers come out:
  - `ruler_disagreement`, the mean absolute difference between the visual
    and point-count share over the rocks both rulers quantify;
  - the simulator's and a constant's error against each ruler (the
    constant is fit on the other train rocks' visual estimates).

  This is a hint at the ruler's own error, not a floor.

```
python explorations/generator_reach/run_signal.py
```

Committed under `results/`:
- `signal.csv`: one row per source and category or mean, every metric
  with its band.
- `signal.json`: settings, the reproduction check, the constant, the fit
  coefficients and the ruler read.
- `signal.png`: measured against simulated and head share per category,
  plus the four-silicate errors per source.

### Result

Four silicates on the val split (106 rocks, 39 units), with 95% bands:

| source | `mae` | `pearson_r` | `mae_linear_fit` | linear fit minus constant |
|---|---|---|---|---|
| simulator | 0.136 [0.113, 0.166] | 0.27 [0.04, 0.43] | 0.102 [0.087, 0.120] | -0.004 [-0.011, +0.002] |
| best rectangle | 0.130 [0.107, 0.159] | 0.32 [0.10, 0.47] | 0.102 [0.087, 0.121] | -0.004 [-0.010, +0.001] |
| forest window | 0.131 [0.108, 0.159] | 0.28 [0.08, 0.42] | 0.104 [0.089, 0.123] | -0.002 [-0.011, +0.007] |
| own hit window (oracle) | 0.107 [0.087, 0.132] | 0.51 [0.31, 0.64] | 0.091 [0.077, 0.108] | -0.015 [-0.024, -0.008] |
| hit rows (oracle) | 0.088 [0.070, 0.110] | 0.66 [0.49, 0.78] | 0.077 [0.063, 0.094] | -0.029 [-0.039, -0.022] |
| constant | 0.106 [0.091, 0.126] | 0.48 [0.20, 0.56] | | |
| head zero-shot | 0.123 [0.103, 0.148] | 0.35 [0.13, 0.50] | 0.100 [0.085, 0.119] | -0.006 [-0.013, -0.000] |
| head recalibrated | 0.101 [0.086, 0.121] | 0.45 [0.22, 0.60] | | (`mae` minus constant -0.005 [-0.013, +0.002]) |
| head given no chemistry | 0.106 [0.091, 0.126] | 0.48 [0.21, 0.55] | | |

- **Most of #79's error is level.** A straight line fit on train takes the
  simulator from 0.136 to 0.102 on the four silicates. That is the
  constant's level (0.106), not reliably below it.
- **The signal is per category.** After the line, the simulator beats the
  constant on:
  - feldspar, -0.015 [-0.030, -0.003];
  - quartz, -0.043 [-0.064, -0.023].

  It ties on pyroxene (-0.006 [-0.016, +0.003]) and opaques (0.000) and
  carries nothing on amphibole (+0.005 [-0.014, +0.022]). The simulator's
  biotite share runs against the measured one (r -0.27, fit slope -0.44).
  Even the oracle hit rows leave biotite at the constant.
- **The mineral list carries more of the rank information than the
  simulator.** The constant, which varies only through which categories a
  rock reports, correlates 0.48 with the measured share on the four
  silicates, against 0.27 for the simulator.
- **The heads carry about the simulator's signal.** The zero-shot head
  after the line reads 0.100 and the recalibrated head 0.101, both within
  0.006 of the constant. The head given no chemistry equals the constant.
  #79's "the heads already beat every marginal" held against the
  simulator's average, not against a constant.
- **Ruler hint.** The visual estimate and the point count of the same thin
  sections differ in share by:

  | category | disagreement | notes |
  |---|---|---|
  | feldspar | 0.127 | visual lower by 0.120; counting floor about 0.015 |
  | amphibole | 0.021 | 10 rocks |
  | biotite | 0.046 | |
  | opaques | 0.067 | |
  | quartz | 0.072 | |

  Against the point count, the simulator's error is about half its error
  against the visual estimate for amphibole (0.038 against 0.061),
  biotite (0.042 against 0.083) and opaques (0.041 against 0.110). It is
  unchanged for feldspar (0.142 against 0.138) and quartz (0.257 against
  0.237). Two eruptions, 10 to 63 rocks per category.
