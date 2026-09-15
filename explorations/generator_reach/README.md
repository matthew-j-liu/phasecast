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

Pending the run.

## Decision

Pending the debrief.
