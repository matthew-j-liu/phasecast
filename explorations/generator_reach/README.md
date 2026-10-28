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

## Plutonic signal read (2026-10-05, #83 step 0)

The first read of #83, on the plutonic whole-rock benchmark (#82), before any
generator run: does the production head track the plutonic rocks better than
baselines that use no simulator? Owner rulings 2026-10-02 set the readout;
the volcanic read above is re-run under them in a later change.

**Settings.** Every source is read the two ways the scaling review figures
read a head: with no correction, and under the matrix recalibrator (weight
decay 1.0). Since 2026-10-05 every fitted source is cross-fit: grouped
5-fold over the train and val rocks, intrusions kept whole, each rock read
once out of fold (`cross_fit`), and the 68 rocks with oxides lumped into the
accessories are read with the oxides unmeasured (`RealRockBenchmark.measured`).
Test stays sealed. Metrics, with 95% unit-bootstrap bands over the
intrusions read:
- share absolute error over each rock's measured mineral list;
- Spearman rank correlation across rocks, with its sign;
- presence AUROC where a source has probabilities;
- joint NLL where a source has a probability model;
- each source minus each baseline on the same resample.

No straight line.

**Sources.**
- the production head (`joint_heads_allrock_tuned_seed0`), zero-shot and
  recalibrated;
- four baselines that use no simulator:
  - the constant over the measured mineral list (train mean);
  - the head given no chemistry (`fit_measured_head` on `ablate_latent`,
    weight decay 10);
  - the CIPW norm, read like any source;
  - the head given raw geochemistry (`use_raw_chemistry`, weight decay 1.0).

  The two weight decays are `run_curve.WEIGHT_DECAY`'s, not re-selected on
  the split being read.

**The norm.** pyrolite's `CIPW_norm` on the majors. Iron is all FeO in the
sources, so pyrolite applies its Le Maitre split. Its summary minerals map to
the groups as `CIPW_TO_GROUP`, and it has no amphibole or biotite. For the
recalibrator, `packaged` gives a share source a forecast:
- presence 1 − 1e-3 where its share is above 0;
- concentrations kappa times its shares, floored at 1e-3 so that the
  multiplicative correction can still build a group the norm lacks;
- kappa is the production head's median train concentration, 50.7.

A packaged source gets no zero-shot NLL, since its concentration is assumed
rather than forecast.

```
python explorations/generator_reach/run_signal.py --benchmark plutonic
```

Written under the `plutonic` key of `results/signal.json`: settings, the NLL
terms with bands, and the table (one record per source and category or
mean). About 20 s.

**First result (superseded).** Fit on train, read on the val split (61
rocks, 19 intrusions; 8 of them carry pyroxene and 1 olivine), the 68
oxide-lumped rocks left out:

| source | share error | Spearman | joint NLL |
|---|---|---|---|
| head zero-shot | 0.080 [0.067, 0.122] | 0.36 [-0.08, 0.51] | 0.60 [-2.21, 5.93] |
| head recalibrated | 0.046 [0.032, 0.094] | 0.38 [-0.11, 0.55] | -3.69 [-6.16, -1.31] |
| CIPW norm | 0.069 [0.040, 0.084] | 0.06 [-0.29, 0.34] | |
| CIPW norm recalibrated | 0.038 [0.026, 0.062] | 0.35 [-0.07, 0.57] | -4.22 [-6.57, -1.81] |
| constant | 0.050 [0.033, 0.096] | 0.09 [-0.23, 0.20] | |
| head given no chemistry | 0.047 [0.030, 0.098] | 0.08 [-0.24, 0.17] | -3.89 [-5.98, -1.66] |
| head given raw geochemistry | 0.051 [0.035, 0.092] | 0.23 [-0.15, 0.38] | -3.94 [-6.06, -1.71] |

Differences on the four silicates, on the same resamples:
- **Recalibrated head.** Its share error is within 0.005 of every
  chemistry-free baseline, with every band crossing zero. Its NLL sits
  +0.20 [-0.69, +1.68] above the head given no chemistry.
- **Recalibrated norm.** It is below the constant (-0.011 [-0.037,
  -0.005]), the head given no chemistry (-0.008 [-0.039, -0.002]) and the
  head given raw geochemistry (-0.012 [-0.033, -0.007]). Against the
  recalibrated head it is -0.007 [-0.036, +0.002], crossing zero.

**Result under cross-fitting (2026-10-05).** Four silicates, 404 train and
val rocks from 61 intrusions read out of fold (Rader Creek carries 89 of
them and forms a fold of its own):

| source | share error | Spearman | joint NLL |
|---|---|---|---|
| head zero-shot | 0.080 [0.067, 0.092] | 0.45 [0.33, 0.60] | -0.69 [-1.62, 0.61] |
| head recalibrated | 0.042 [0.038, 0.048] | 0.42 [0.30, 0.61] | -4.79 [-5.37, -3.87] |
| CIPW norm | 0.082 [0.065, 0.094] | 0.35 [0.19, 0.56] | |
| CIPW norm recalibrated | 0.042 [0.035, 0.053] | 0.32 [0.19, 0.58] | -4.69 [-5.25, -3.81] |
| constant | 0.054 [0.045, 0.079] | -0.32 [-0.37, -0.05] | |
| head given no chemistry | 0.053 [0.042, 0.075] | -0.12 [-0.21, 0.06] | -4.18 [-4.76, -3.18] |
| head given raw geochemistry | 0.052 [0.045, 0.077] | -0.14 [-0.27, 0.05] | -4.17 [-4.78, -3.11] |

- **Recalibrated head.** It ties the recalibrated norm (-0.0003 [-0.0095,
  +0.0072]; NLL -0.10 [-0.37, +0.23]) and is below the constant (-0.012
  [-0.033, -0.005]), the head given no chemistry (-0.012 [-0.029, -0.002])
  and the head given raw geochemistry (-0.010 [-0.032, -0.004]).
- **Recalibrated norm.** It is below the same three (-0.012 [-0.028,
  -0.007], -0.011 [-0.024, -0.005], -0.010 [-0.027, -0.005]).

The first result's reading, that the head ties the simple baselines, does
not survive: the 19 val intrusions were unfavourable to the head. The bar
for Stage A is 0.042, held jointly by the recalibrated head and the
recalibrated norm.

**Which change moved the head (2026-10-06).** Step 0's sources were re-run
in all four corners (scratch, no generator sources). Recalibrated head /
recalibrated norm, four silicates:

| | fit on train, read on val | grouped 5-fold |
|---|---|---|
| 68 rocks excluded | 0.046 / 0.038 | 0.042 / 0.044 |
| 68 rocks included | 0.049 / 0.035 | 0.042 / 0.042 |

The protocol moved the head, not the rocks. Under cross-validation the head
reads 0.042 with or without the 68 rocks, and head minus norm crosses zero
in both corners. Under train/val the 68 rocks push the head up and the norm
down (head minus norm +0.014 [+0.002, +0.052]). A plutonic comparison read
on val alone is suspect until it is read out of fold.

## Stage A: the simulator near the solidus (2026-10-06, #83)

Run 2026-10-05: 578 rocks x 128 draws x 31 steps, 5.5 h wall, no solver
crashes, 0.09 failed points per draw. 130 and 222 of 73,600 draws never
reach 10% and 5% melt, and no train or val rock falls back to its last
step. The three near-solidus rungs read within 0.001 of each other: on wet
draws the 10% and 5% reads often land on the same fully solid step. The
read at 50 and 30% melt, the production head's own rungs, was added
2026-10-06 from the same ladder. Four silicates, 404 train and val rocks
out of fold, bands over 61 intrusions:

| source | share error | Spearman | joint NLL |
|---|---|---|---|
| head zero-shot, teacher-forced on the measured list (step 0) | 0.080 [0.067, 0.092] | 0.45 | -0.69 |
| head zero-shot, marginal over sampled assemblages | 0.058 [0.051, 0.077] | 0.48 | |
| simulator at 50% melt, zero-shot | 0.059 [0.049, 0.083] | 0.48 | |
| simulator at 50% melt, recalibrated | 0.042 [0.034, 0.067] | 0.46 | -4.89 [-5.62, -3.78] |
| simulator at 30% melt, zero-shot | 0.056 [0.047, 0.064] | 0.49 | |
| simulator at 30% melt, recalibrated | 0.041 [0.035, 0.060] | 0.44 | -4.84 [-5.51, -3.76] |
| simulator at 10% melt, zero-shot | 0.060 [0.049, 0.068] | 0.49 | |
| simulator at 10% melt, recalibrated | 0.044 [0.036, 0.067] | 0.41 | -4.71 [-5.33, -3.63] |
| joint fit B, norm plus simulator at 10% | 0.042 [0.036, 0.056] | 0.36 | -4.87 [-5.43, -3.95] |
| joint fit A, head given norm and simulator | 0.049 [0.041, 0.074] | -0.12 | -4.29 [-4.87, -3.30] |
| head recalibrated (step 0) | 0.042 [0.038, 0.048] | 0.42 | -4.79 [-5.37, -3.87] |
| CIPW norm recalibrated (step 0) | 0.042 [0.035, 0.053] | 0.32 | -4.69 [-5.25, -3.81] |

- **The simulator ties the bar after recalibration.** Zero-shot it is the
  best source at 0.060, against 0.080 for the head and 0.082 for the norm,
  with the highest Spearman of any source. Recalibrated it reads 0.044
  against the bar's 0.042, bands overlapping.
- **Joint fit B: no share-error gain, a small NLL gain.** Against the
  recalibrated norm, share error +0.0002 [-0.0015, +0.0045]; joint NLL
  -0.18 [-0.30, -0.02] at 10% melt, -0.15 [-0.25, -0.01] at 5% and -0.19
  [-0.27, -0.08] at the last step. What the simulator adds beyond the norm
  is probabilistic, mostly presence of the minerals the norm lacks.
- **Joint fit A ranks nothing.** The head given the norm, the simulator or
  both reads 0.049 to 0.051, above the bar, as the head given raw
  geochemistry did at step 0: the small measured head is a weaker fitter
  than the recalibrator on 320 rocks.
- **Amphibole and biotite track.** Spearman, simulator zero-shot /
  recalibrated, then the recalibrated norm and head: amphibole (226 rocks)
  0.58 / 0.52, norm 0.29, head 0.44; biotite (391 rocks) 0.52 / 0.52, norm
  0.42, head 0.58. On the eruption benchmark amphibole carried nothing and
  biotite ran backwards, so #85's hypothesis holds: whole-rock point counts
  track the simulator on these two, and the recipe and database are not
  the cause there. Pyroxene tracks at 0.57 on 40 rocks. Feldspar is weak
  for every source (0.19 to 0.31).
- **Reach.** One (draw, rung) row lands within 0.05 on all four silicates
  for 90% of the read rocks (92% of the 574 with them quantified), against
  62% on the eruption rocks; per category 0.88 (quartz) to 0.98, olivine
  0.67 on few rocks.

No gate (owner). The reading: three unrelated sources, thermodynamics, the
norm's arithmetic and the head, meet at 0.042 share error after a linear
correction, so share error is saturated by what major-element chemistry
plus a 7x7 correction fit on about 320 rocks can say; the simulator's edge
over the norm is in NLL and ranking.

**The head's zero-shot gap is the estimator, not the rung and not copying
error (2026-10-06; a first version of this paragraph read it as copying
error and is superseded).** The head emulates MAGEMin at 50 and 30% melt,
so its zero-shot 0.080 against the simulator's 0.060 near the solidus
could have been the rung. It is not: at the head's own rungs the simulator
reads 0.059 and 0.056 zero-shot, and recalibrated it ties the bar there
too with the best NLL of any single source. Nor has the head failed to
copy the simulator. The read scores the head teacher-forced on the rock's
measured mineral list, the conditional mean given every listed mineral
present, and the simulator as its draw mean, which includes the draws that
lack a mineral. For a mineral the simulator produces in only some draws
(amphibole on 18% of the typical granitoid's draws) the two estimators
differ by construction. Read the way the simulator is read, its support
sampled from its own presence probabilities (32 samples), the head scores
0.058 zero-shot with Spearman 0.47 against the simulator's 0.059 and 0.48
at 50% melt, and matches per category: feldspar 0.076 against 0.074,
amphibole 0.052 against 0.051, biotite 0.034 both, pyroxene 0.070 against
0.076. Its marginal share differs from the simulator's draw mean by 0.019
on the four silicates over the seven groups, and its presence
probabilities match the simulator's presence fractions (amphibole 0.14
against 0.18, biotite 0.73 against 0.67). On its own simulated val bulks
the same marginal read differs by 0.030, and blanking the trace tokens
moves nothing (0.061 to 0.063 under the teacher-forced construction), so
the plutonic rocks' sparse traces are not a shift the head feels. The
read now carries the head's marginal as a source (`head_marginal`,
`marginal_shares`, 32 support samples) and `head_copying_error` in
`signal.json` is that marginal against the generator's draw mean over the
measured list: 0.026 at 50% melt, 0.028 at 30%, 0.036 at 10%; the
simulated-val reads are scratch, journal 2026-10-06. So the production head has
converged to the simulator on these rocks, and the only zero-shot
difference between them is which estimator is read.

**The rung is a per-mineral lever that cancels in the mean (2026-10-06).**
The four-silicate error is the same at every rung, but the assemblage is
not: between 50% melt (median 720 C) and 10% (640 C) the model grows
quartz late (+0.058 share), consumes most of its amphibole (-0.063; carried
by 18% of draws at 50% and 6% at 10%), loses pyroxene (-0.042) and gains
biotite (67% to 85% of draws). The shifts push the error in opposite
directions and cancel: at 50% melt quartz and amphibole are almost
unbiased against the measured rock (+0.005, +0.006) and at 10% they are
not (+0.063, -0.056), while pyroxene improves (+0.074 to +0.033). So the
measured granitoid keeps its amphibole and carries less quartz than the
model's near-solidus assemblage; whether that is a recipe signal is a
hypothesis, journal 2026-10-06. The rung has to be read per mineral, and
no single rung serves every mineral on the same rock. The reach read
keeps the near-solidus rows (`SOLIDUS_RUNGS`), so its numbers above are
unchanged.

## Seven-mineral ladder read (2026-10-09, step 1 before Stage B)

The rung finding above says the four-silicate error is flat across rungs
while every mineral in it moves. Before the recipe is changed, the kept
equilibrium rows say per mineral whether the late-stage error is a
conditions problem (some corner of the (P, H2O) box keeps the mineral,
so a prior could) or a reaction problem (every draw consumes it, so only
the recipe can), and what the fractional run should then show.
`run_ladder.py`, model-free, nine seconds, both benchmarks.

**Method.** The evaluated train and val rocks of each run (404 plutonic
rocks, 51,712 draws; 478 eruption rocks, 61,184 draws; test sealed), every
converged step with each category's crystal mass on the whole-system basis.
Per (draw, category): the peak mass and its T down the ladder, whether the
draw carries the category (seven-category crystal share at the peak above
`PRESENCE_CUTOFF`), the mass at the 50, 30, 10 and 5% rungs (highest-T step
with melt below, as the Stage A read) as a fraction of the peak, and, when
less than 0.9 of the peak is left at 10%, the gain budget: per unit of the
category's mass lost between its peak and the 10% step, the mass every other
category gained over the same steps (the sum exceeds one because the melt
keeps crystallizing). Splits at the kept draws' medians, 3.57 wt% H2O and
6.25 kbar. The prediction for the fractional run is the no-resorption pile:
per draw the running sum of each category's positive mass increments down
the path, what the equilibrium path grew had nothing it grew been consumed;
read at each rung as a draw-mean share over the measured mineral list beside
the equilibrium share, both scored against the measured share. It keeps
each step's growth but not fractionation's changed liquid line of descent,
so it is a proxy, not the run's result.

```
python explorations/generator_reach/run_ladder.py
```

Committed under `results/`: `ladder.csv` (one row per benchmark, category
and split), `ladder.json` (settings, medians, counts, the prediction) and
`ladder.png` (mean category mass against T by split).

**Result.** Carrying draws, both benchmarks read the same way:

| category | plutonic: carrying, declining, median T at peak, remaining at 50 / 30 / 10% | gain per unit lost (plutonic, 10% step) |
|---|---|---|
| fsp | 51,654 draws, 6%, 620 C, 0.64 / 0.99 / 1.00 | grows to the end |
| px | 43,147, 69%, 700 C, 0.38 / 0.58 / 0.61 | fsp +5.6, q +3.5, bi +2.0, amp +0.2 |
| amp | 23,697, 98%, 760 C, 0.00 / 0.00 / 0.00 | fsp +2.4, q +1.4, bi +0.75, other +0.7, px +0.4 |
| bi | 50,843, 44%, 600 C, 0.61 / 0.86 / 0.93 | px +1.1, other +1.0, fsp +0.85 |
| ol | 518, 98%, 960 C, 0.00 / 0.00 / 0.00 | fsp +7.5, q +3.9, bi +3.6, amp +3.0 |
| opq | 51,294, 61%, 800 C, 0.87 / 0.88 / 0.82 | small losses; fsp +20, q +14 |
| q | 51,701, 12%, 600 C, 0.53 / 0.94 / 1.00 | grows to the end |

- **Amphibole is a reaction problem.** It peaks at 760 C on 46% of the
  draws and is gone by the 10% rung on 98% of them, with the median draw
  already at zero by 50% melt; wet, dry, deep and shallow halves all read
  0.00 remaining (declining 1.00 / 0.93 / 0.97 / 0.99). No corner of the
  box keeps it, so a prior cannot; per unit lost the path grows 0.75
  biotite and 1.4 quartz beside feldspar, the amphibole-to-biotite-plus-
  quartz reaction. Olivine, on 1% of draws, is consumed the same way.
- **Pyroxene is half and half.** It declines on 69% of draws to 0.61 of
  its peak, wet draws keeping 0.86 and dry ones 0.47; its loss grows
  biotite (+2.0) and quartz (+3.5). A conditions lever (water) exists but
  the reaction runs on both halves.
- **Biotite, quartz and feldspar grow to the end** (remaining 0.93 to
  1.00 at 10%). Biotite declines on 44% of draws by a tenth at most, into
  pyroxene and other phases. The oxides lose little except at depth
  (deep draws keep 0.06 at 10%, shallow 0.94), a conditions signal.
- The eruption rows read the same per mineral (amphibole 97% declining
  to 0.00, pyroxene 68% to 0.68, biotite 94% kept, quartz 99%), so the
  ladder's shape is the recipe's, not the benchmark's.

**Prediction for the fractional run** (absolute error against the measured
share, equilibrium rung / no-resorption proxy):

| rung | plutonic fsp | px | amp | bi | q | four silicates | eruption four silicates |
|---|---|---|---|---|---|---|---|
| 50% | 0.074 / 0.081 | 0.076 / 0.113 | 0.051 / 0.068 | 0.034 / 0.032 | 0.069 / 0.069 | 0.053 / 0.058 | 0.139 / 0.129 |
| 30% | 0.068 / 0.082 | 0.059 / 0.085 | 0.058 / 0.045 | 0.039 / 0.033 | 0.069 / 0.057 | 0.053 / 0.053 | 0.158 / 0.130 |
| 10% | 0.070 / 0.088 | 0.063 / 0.075 | 0.066 / 0.042 | 0.041 / 0.036 | 0.080 / 0.059 | 0.056 / 0.055 | 0.167 / 0.130 |

- On the plutonic rocks, keeping what the path grew improves amphibole
  (0.066 to 0.042 at 10%), quartz (0.080 to 0.059) and biotite, and
  worsens pyroxene (0.063 to 0.075) and feldspar (0.070 to 0.088): the
  pile keeps the pyroxene the rock does not have and shifts the feldspar
  share. The four-silicate mean moves by 0.001, the cancellation again.
  So the fractional run should be read per mineral: amphibole and quartz
  are the minerals it is predicted to fix, pyroxene and feldspar the
  ones it is predicted to break, and a recipe that fractionates perfectly
  may overshoot on pyroxene where partial removal would not.
- On the eruption rocks the proxy helps every category and the four
  silicates by 0.01 at 50% and 0.04 at 10%, mostly feldspar and
  amphibole. The phenocryst counts sit closer to what the path grew than
  to what it keeps at any rung.

The fractional run tests these numbers directly; the proxy is retired once
it has run.

## Fractional crystallization generator (2026-10-09, #83 and #84 step 2)

`run_fractional.py` and `run_fractional.jl` run fractional crystallization
of each benchmark rock's own chemistry, paired draw for draw with the
equilibrium run: the same composition tables, the same 128 (P, H2O) draws
from `draw_conditions` with `seed + bulk`, the same 1200 to 600 C ladder in
20 C steps, the same production Julia environment (Julia 1.12.6, MAGEMin_C
2.1.9, database `ig`), the same subprocess, timeout and crash bisection as
`run_generator.py`, here at path boundaries.

**Recipe.** A path starts from the rock's 11-oxide bulk (FeO and Fe2O3
separate, as the equilibrium run) with the system mass at 1. After each
converged step the melt's own composition, its oxides including the excess
oxygen `O`, becomes the next step's bulk and the system mass is scaled by
the melt's weight fraction: everything else is removed, exsolved fluid
included, and iron speciation follows the oxygen mass balance of the melt
with no buffer. A failed step (status not 0, G non-finite or not negative)
is written and skipped with the bulk kept; a path ends when no melt is
stable, when the melt falls below `--melt-floor` (1e-4 of the starting
mass) or at the ladder's end, and its rows below the end are not run. This
is MAGEMinCalc's fractionation scheme (PetThermoTools' Julia half, read
2026-10-06: minimize, record, next bulk the melt, mass times the melt
fraction, no liquidus bracketing, no adaptive step, no stop at melt-out)
driven directly, with the status and G of every step kept and raw MAGEMin
phase names; ADR 004's amendment of 2026-10-09 records why.

**Check against PetThermoTools.** Its environment was aligned to the same
MAGEMin_C 2.1.9 (from 2.0.5) and plutonic rock 0's first 12 draws run
through `ptt.isobaric_crystallisation(Model="Green2025", Frac_solid=True)`
on the same ladder. On 11 of the 12 draws every step agrees within 0.002 of
the starting mass on the melt and 0.004 on any phase (PetThermoTools passes
iron as a ratio with 159.59 for Fe2O3's molar mass). The twelfth, wet and
deep (5.0 wt%, 11.6 kbar), agrees to the same tolerance down to its last
ladder step at 600 C with a fifth of the melt left, where the two found
different minima (nepheline plus quartz against feldspar, 0.13 of the
starting mass). PetThermoTools stays installed as that reference, never as
the driver.

**Smoke.** Plutonic rock 0 (a granite): the dry draw (0.56 wt% H2O, 6.05
kbar) crystallizes from 1040 C, feldspar then quartz, pyroxene and oxides,
biotite late, and runs out of melt at 640 C; the wet draw (6.45 wt%, 7.47
kbar) carries 98% melt to 800 C, then quartz, biotite and feldspar, and
runs out at 620 C with 5% fluid exsolved at the end. The melt's Fe3+/sum Fe
falls from 0.45 to 0.2 while magnetite is removed and climbs to 0.8 in the
last few percent of melt. Twelve paths: 350 steps in 4.9 s, 14 ms a step,
none failed; a path that runs out of melt stops there.

```
python explorations/generator_reach/run_fractional.py --smoke                      # rock 0, two draws, prints every step
python explorations/generator_reach/run_fractional.py --smoke --benchmark plutonic
python explorations/generator_reach/run_fractional.py --benchmark plutonic         # 578 x 128, work/plutonic128_fc
python explorations/generator_reach/run_fractional.py                              # 640 x 128, work/broad128_fc
python explorations/generator_reach/run_signal.py --benchmark plutonic --work explorations/generator_reach/work/plutonic128_fc --table pile --recipe fractional
```

Per chunk, under the gitignored work dir: `chunk_XXXX_draws.parquet` (the
equilibrium columns plus `end`, `T_liquid_out`, `n_steps_run`, and rung
read-outs at 50, 30, 10 and 5% melt on the starting mass),
`chunk_XXXX_step.parquet` (each step's phases as they form, the cargo),
`chunk_XXXX_pile.parquet` (the solids and fluid accumulated down the path,
the residual melt as its `liq` rows: the plutonic label) and
`chunk_XXXX_melt.parquet` (per converged step the melt's Fe3+/sum Fe and
H2O, the status and G). Both ladder tables are in `LADDER_COLS` with
`mode_wt` and `liq_frac` on the starting mass, so `run_read.build_points
(work, table)` and the Stage A read work on either; `run_signal.py
--recipe fractional` writes the plutonic read under `plutonic_fractional`
in `signal.json` and `plutonic_stage_a_fractional.png`, leaving the
equilibrium record in place. The eruption read of a fractional run waits on
the eruption benchmark's re-run under the cross-fitted protocol (#85).

## Stage A on the fractional run (2026-10-13, #83 and #84)

Run 2026-10-09 to 10-13 under the recipe above: plutonic 578 rocks x 128
draws, 5.6 h, no solver crashes, 2.16 M of 2.29 M laddered steps run (the
rest below melt-out); eruption 640 x 128, 6.1 h, 2.39 M of 2.54 M. The melt
runs out at 640 C on the median draw (620 to 680 C from the 10th to the 90th
percentile) and 16% of draws reach the ladder's 600 C end with melt left
(median a fifth of the starting mass; wet and deep, median 11 kbar). Failed
steps, 0.3 per draw, sit at the top of the ladder above the liquidus, where
they are skipped with the bulk kept, and at the 600 C step on the draws
whose melt persists; 0.2% of draws lose their tail above 700 C. So 12% of
draws never reach 10% melt (4% never reach 30%, 0.4% 50%), against 0.2%
under equilibrium, and the near-solidus rungs read the other 88%, the
drier and shallower draws. The melt's Fe3+/sum Fe climbs from 0.4 to a
median 0.64 below 10% melt, oxygen mass balance with no buffer.

The same cross-fitted read as the equilibrium Stage A (404 train and val
rocks, 61 intrusions, `--table pile --recipe fractional`), four silicates:

| source | equilibrium | fractional |
|---|---|---|
| simulator at 50% melt, zero-shot | 0.059 [0.049, 0.083] | 0.073 [0.060, 0.107] |
| simulator at 30% melt, zero-shot | 0.056 [0.047, 0.064] | 0.062 [0.051, 0.077] |
| simulator at 10% melt, zero-shot | 0.060 [0.049, 0.068] | 0.056 [0.046, 0.067] |
| simulator at 30% melt, recalibrated | 0.041 [0.035, 0.060], NLL -4.84 | 0.038 [0.034, 0.046], NLL -4.84 |
| simulator at 10% melt, recalibrated | 0.044 [0.036, 0.067], NLL -4.71 | 0.039 [0.034, 0.053], NLL -4.88 |
| joint fit B, norm plus simulator at 30% | 0.041 [0.035, 0.055] | 0.039 [0.033, 0.050], NLL -4.98 |
| recalibrated norm (step 0) | 0.042 [0.035, 0.053], NLL -4.69 | same |
| recalibrated head (step 0) | 0.042 [0.038, 0.048], NLL -4.79 | same |

- **Under the bar, by a hair.** Recalibrated at 30% melt the fractional
  simulator reads 0.038 against the norm's 0.042 on the same resamples,
  -0.0044 [-0.0092, +0.0005]; at 10% -0.0030 [-0.0060, +0.0030]. The joint
  fit with the norm at 30% clears zero, -0.0036 [-0.0054, -0.0006] in share
  and -0.29 [-0.44, -0.08] nats, the first source to do so on share error.
  The equilibrium simulator's best was -0.0011 [-0.0049, +0.0088].
- **Zero-shot the near-solidus rungs improve and the 50% rung worsens**
  (0.060 to 0.056 at 10%, 0.059 to 0.073 at 50%): the early pile is
  pyroxene-heavy (bias +0.125 in share at 50% melt against +0.074 under
  equilibrium).
- **Spearman falls** on the four silicates, 0.49 to 0.42 zero-shot, carried
  by feldspar (0.31 to 0.20) and biotite (0.52 to 0.35); amphibole holds at
  0.58 and quartz rises 0.48 to 0.62.

Per mineral at 10% melt, zero-shot, the ladder read's prediction beside the
result (absolute error; the signed bias is simulated minus measured share):

| category | n | equilibrium | predicted | fractional | bias eq -> fc | draws carrying eq -> fc |
|---|---|---|---|---|---|---|
| amphibole | 226 | 0.066 | 0.042 | 0.038 | -0.056 -> -0.027 | 0.09 -> 0.46 |
| quartz | 403 | 0.080 | 0.059 | 0.054 | +0.063 -> +0.018 | 1.00 -> 1.00 |
| pyroxene | 40 | 0.063 | 0.075 | 0.066 | +0.033 -> +0.062 | 0.86 -> 0.95 |
| feldspar | 404 | 0.070 | 0.088 | 0.065 | -0.035 -> +0.038 | 1.00 -> 1.00 |
| biotite | 391 | 0.041 | 0.036 | 0.054 | +0.008 -> -0.053 | 0.85 -> 0.93 |

- **Amphibole and quartz as predicted, better than predicted.** The
  amphibole the equilibrium path consumed is kept: it is in the pile on 46%
  of draws against 9% at the equilibrium rung, and its bias halves. The
  quartz overshoot drops from +0.063 to +0.018.
- **Pyroxene worse, as predicted.** Perfect removal keeps every early
  pyroxene; the bias doubles.
- **Feldspar and biotite go the other way from the prediction.** Feldspar
  improves slightly while its bias flips sign (the pile is feldspar-rich
  where the equilibrium assemblage was feldspar-poor); biotite is the
  casualty, 0.041 to 0.054 with a bias of -0.053, its rank correlation
  down to 0.35 and its reach from 0.98 to 0.75 of rocks. The two misses
  are one mechanism: the proxy credited the biotite the equilibrium path
  grew by reacting amphibole and pyroxene with the melt, and under perfect
  removal those crystals never react, so the pile is biotite-poor. The
  measured granitoids keep their amphibole and still carry their biotite,
  which neither recipe produces: the equilibrium path converts the first
  into the second, perfect fractionation keeps the first and starves the
  second. Partial removal sits between them; named as the next recipe
  question, not run.
- The production head's marginal sits 0.043 from the fractional draw mean
  at 10% melt against 0.036 from the equilibrium one, as a head trained on
  equilibrium labels should.
- Reach: one (draw, rung) row within 0.05 on all four silicates for 85% of
  the read rocks against 90% under equilibrium; biotite 0.75 against 0.98,
  amphibole 0.95 against 0.93.

The eruption run is read model-free, beside the plutonic run, in the next
section.

## Recipe and read point across melt fractions (2026-10-28, #84)

**Question.** Read without any model or fit, does fractional
crystallization of each benchmark rock's own composition land closer to
its measured mineralogy than equilibrium does, and at which melt fraction
should a label be read?

**Method.** `run_recipe_rungs.py` reads both runs of each benchmark
(`broad128` / `broad128_fc`, `plutonic128` / `plutonic128_fc`). The two
runs of a benchmark drew identical (P, H2O) for every (rock, draw), checked
at run time, so the recipe is the only difference.

- Reads: rungs 90 to 10% and 5% melt (each draw's highest-temperature
  point below the rung, the rule the runs themselves use), each draw's last
  converged step, the pooled 50 + 30 rungs the corpus label uses, and on
  the volcanic rocks each rock's own melt fraction (1 -
  `total_crystals_pct` / 100, a volume fraction against a mass fraction;
  the plutonic rocks are fully crystalline and have none).
- At every read both recipes use only the draws both reach there. Every
  rock's simulated share is the equal-weight draw mean of each draw's
  share over the rock's measured mineral list.
- Rocks: train and val, test sealed, the same rocks at every read.
  Volcanic 477 rocks in 85 eruptive units (one rock at 100% crystals has
  no own-melt read and leaves); plutonic 404 rocks in 61 intrusions.
- Metric: raw simulator error (absolute share error, no fit) per mineral
  category and the mean of seven, with Spearman and presence AUROC beside
  it. Unit bootstrap (2,000 resamples) for the bands, the fractional minus
  equilibrium differences, and the fraction of resamples in which each read
  has the lowest mean-of-seven error.
- Fit-free baselines: the production head read as its marginal (shares
  averaged over assemblages sampled from its own presence probabilities,
  the basis of the simulator's draw mean) and the CIPW norm. No
  recalibration anywhere, and no folds, since nothing is fit.

```
python explorations/generator_reach/run_recipe_rungs.py --benchmark volcanic
python explorations/generator_reach/run_recipe_rungs.py --benchmark plutonic
```

Writes `results/recipe_rungs_{benchmark}.csv` (one row per source, read and
category: error with its 95% band and interquartile range, fractional minus
equilibrium with its band, Spearman, AUROC), `.json` (settings, draw counts
per read, dropped rocks, lowest-error fractions, run facts) and `.png` /
`.pdf` (the locked data-scaling frame of
`explorations/corpus_scaling/fig_zero_shot_scaling.py`; bars are the
interquartile range). About 15 seconds each once the points are cached.

**Result.** Mean-of-seven raw simulator error, percent:

| read | volcanic eq | volcanic fc | plutonic eq | plutonic fc |
|---|---|---|---|---|
| 90 | 15.28 | 15.17 | 12.83 | 12.91 |
| 80 | 14.00 | 13.90 | 9.96 | 10.45 |
| 70 | 13.52 | 13.38 | 7.84 | 8.67 |
| 60 | **13.28** | **13.26** | 6.21 | 7.33 |
| 50 | 13.50 | 13.37 | 5.16 | 6.34 |
| 40 | 14.14 | 13.62 | **4.74** | 5.65 |
| 30 | 14.76 | 13.82 | 4.87 | 5.24 |
| 20 | 15.24 | 13.95 | 4.95 | 4.98 |
| 10 | 15.59 | 13.96 | 5.02 | 4.81 |
| 5 | 15.71 | 13.96 | 5.09 | **4.77** |
| last step | 15.66 | 13.99 | 5.84 | 4.79 |
| own melt | 14.60 | 14.46 | | |
| 50+30 | 13.97 | 13.53 | 4.66 | 5.74 |
| production head (marginal) | 14.10 | | 5.17 | |
| CIPW norm | 15.59 | | 6.36 | |

Bold marks each recipe's best single rung.

- **At its best read the recipe ties on both benchmarks.** Volcanic: both
  lowest at 60% melt, 13.28 against 13.26. Plutonic: equilibrium at 40%
  4.74 against fractional at 5% 4.77. Fractional changes less with depth
  (its early crystals are frozen in the pile), so it pulls ahead only where
  equilibrium has drifted: volcanic at 10% melt -1.6 points [-2.1, -1.1],
  plutonic at the last step -1.0 [-1.7, -0.2]. Read at 70 to 40% on the
  plutonic rocks it is worse (+0.8 to +1.2), its early pile rich in
  pyroxene (+3.8 and +5.1 points on pyroxene at 70 and 50%).
- **The same per-mineral signature on both benchmarks.** Fractional reduces
  amphibole error (volcanic -0.9 to -5.1 points from 70% down, plutonic
  -1.6 to -3.2 from 70% down) and quartz error when read deep (volcanic
  -0.9 to -1.6, plutonic -1.0 to -3.5), and raises biotite error from 70%
  down (volcanic +1.0 to +2.0, plutonic +0.4 to +1.8; at 90% the recipes
  nearly coincide). Equilibrium converts
  amphibole to biotite as it cools; perfect removal never lets that
  reaction run. For amphibole and biotite together the measured rocks sit
  between the two recipes, the gap #90 (partial crystal removal) targets.
  Volcanic olivine and Fe-Ti oxides are also lower under fractional
  (olivine -0.6 to -3.2 on 60 rocks; oxides -0.2 to -1.2, with the run's
  missing redox buffer as a caveat); plutonic oxides are slightly lower
  under equilibrium, on shares of 1 to 2 percent.
- **Biotite's ranking.** Volcanic equilibrium biotite tracks the rocks when
  read shallow (Spearman +0.33 at 90%, +0.29 at 70%) and runs backwards at
  or below 50% (-0.20 at 50, -0.34 at 30, -0.32 for the 50+30 label); the
  production head reads -0.27, carrying its label's ordering. Fractional
  biotite never runs backwards (+0.13 to +0.17 at or below 50%). On the plutonic
  rocks equilibrium biotite tracks at every read from 50% down (+0.51 to
  +0.53) and fractional less (+0.33 to +0.38): the backwards biotite is a
  property of the volcanic phenocrysts, not of equilibrium.
- **Shape of the curves.** Plutonic error falls with depth and flattens
  near the solidus, as whole-rock modes of a fully crystallized rock
  should. Volcanic error is U-shaped, as phenocrysts (the crystals grown
  before eruption, groundmass excluded) should be, but its minimum sits at
  60% melt while the rocks erupted near 82% (median 18% crystals), and each
  rock's own melt fraction reads worse than a fixed 60 to 70%.

**One rung for one corpus.** A single training corpus serving both
benchmarks wants one label read. Equilibrium, mean of seven, percent:

| read | volcanic | plutonic | mean of the two | mean ratio to each benchmark's best |
|---|---|---|---|---|
| 60 | 13.28 | 6.21 | 9.75 | 1.17 |
| 50 | 13.50 | 5.16 | 9.33 | 1.06 |
| 50+30 | 13.97 | 4.66 | 9.32 | 1.03 |
| 40 | 14.14 | 4.74 | 9.44 | 1.04 |
| 30 | 14.76 | 4.87 | 9.82 | 1.08 |

The volcanic optimum (60) costs the plutonic rocks a third more error; the
corpus label, 50+30, is the best compromise on either weighting, tied
with 50 in points. Fractional's best global reads (5 and 10%) average 9.37
and 9.39.

**Decision.** Equilibrium at 50+30 stays the corpus label; neither the
recipe nor a benchmark-specific rung is adopted. Fractional
crystallization is not a better label on the mean of seven for either
benchmark; its value is per mineral (amphibole, quartz read deep) and in
biotite's ordering, which points at partial removal (#90) rather than a
fractional corpus. The plutonic zero-shot scaling line of the existing
50+30 heads is the next read.
