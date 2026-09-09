# corpus_scaling

## Question

How does joint-heads quality scale with training-corpus size -- and does the
curve justify the ~10x NERSC regen (#27, feeding the #45 "draws/bulks
sizing" ledger line)? A phase whose metrics are still improving at N=10k
bulks is data-limited (more corpus will help); one that went flat is
architecture/loss-limited (more corpus is wasted on it). The prior two-point
line (47-chunk vs 100-chunk, 2026-07-17) moved garnet and ilmenite but not
spinel or the amphibole magnitude; this ladder replaces it with a proper
curve on the committed val roster.

## Method

- Nested train-bulk subsets 625 / 1250 / 2500 / 5000 / 10000 (full): sample
  ids sorted by md5 hex digest, prefix of length N -- platform-stable, and
  every smaller set is contained in the larger by construction. Class mix is
  approximately proportional at every N; rare classes thin out first, as
  they would in a genuinely smaller corpus.
- Each point retrains the standard config (`configs/heads/joint_heads.yaml`:
  joint heads + support-conditioned magnitude, 120 epochs, seed 0) with only
  the train tier subset changed, then runs the ModeHead interrogation on the
  fixed 222-bulk val roster.
- Optional second axis (`--draws 24,12`): all bulks at reduced
  draws-per-bulk. Matched-row comparisons (10k bulks x 24 draws vs 5k x 48;
  10k x 12 vs 2.5k x 48) separate more-bulks from more-paths-per-bulk -- the
  actual NERSC design question.
- Single seed per point; the full-N point doubles as a plumbing check
  against the production `joint_heads_supportmag_seed0` numbers.

Run from the repo root (env active):

```
python explorations/corpus_scaling/run_ladder.py                # ~1.5-2h streaming
python explorations/corpus_scaling/run_ladder.py --draws 24,12  # optional, ~+35 min
python explorations/corpus_scaling/aggregate.py
```

## Result

Run 2026-07-22/23, 7 points (5 bulk ladder + draws 24/12 at full bulks);
`results/scaling.json`, `scaling.png`, `scaling_draws.png`.

- Plumbing check: the full-N point reproduces the production
  `joint_heads_supportmag_seed0` metrics to 4 decimals on all three NLL
  components.
- Bulk axis, no saturation: val joint NLL -1.178 -> -2.456 across
  625 -> 10000, still gaining 0.24 nats on the last doubling. A saturating
  power law `L_inf + a*N^-b` fits within 0.02 nats everywhere
  (b = 0.225, L_inf ~ -3.9, floor poorly constrained); a pure log-line
  misses by 0.06 -- the per-doubling gain is slowly shrinking
  (0.42 -> 0.30 -> 0.31 -> 0.24).
- Draws axis, flat: 12 -> 24 -> 48 draws/bulk at full bulks spans 0.08
  nats non-monotonically (24 beats 48 by 0.035 -- the single-seed noise
  scale). At matched training rows, all-bulks-fewer-draws wins by huge
  margins (240k rows: -2.41 vs -1.90; 480k rows: -2.49 vs -2.21). The one
  monotone draws signal is CondHead (~0.06 nats/doubling), consistent
  with draws being the (P, H2O) variation itself; mode/comp and every
  highlighted phase are flat in draws.
- Per-phase verdicts: garnet data-limited (pf_r 0.66 -> 0.83 and still
  climbing); spinel weakly data-limited but structurally capped at ~0.70
  pf_r (the #46 redox input token is the lever, data secondary);
  amphibole not data-limited (pf_r flat ~0.93 across the ladder,
  magnitude MAE flat since 5000 -- the #25 mixture question, not data);
  ilmenite saturated since ~1250 bulks.

The mechanism behind the bulks/draws asymmetry: the model never sees
(P, H2O), so draws are unbiased Monte-Carlo samples of each bulk's outcome
distribution -- draw count only adds variance that pooling across
compositionally similar bulks averages out, while bulk count controls
coverage of input space, a bias no draw budget can fix. Most phases are
near-deterministic given the bulk (within-bulk path sensitivity
concentrates in amphibole and accessories), so extra draws mostly
re-observe the same assemblage.

## Decision

Regen sizing (the #45 "draws/bulks sizing" line): allocate to bulks, not
draws -- 24 draws/bulk is the knee (48 -> 24 costs nothing measurable on
mode/comp, ~0.06 nats on cond, and buys a bulk doubling worth ~0.25-0.3).
Recommended weekend shape (NERSC down, NeurIPS deadline, ~60 h laptop
budget at the measured ~4 draws/s): ~30k bulks x 24 draws, keeping the
existing 10k train bulks and the sealed 500-bulk valtest roster
(regenerated under the new proposal; valtest at 256 draws/bulk), rare-class
quotas on the new bulks. Caveat carried forward: the 24-draw knee was
measured under the current broad P/H2O proposal; the #48 dry run checks
whether the trace-conditional proposal changes per-draw redundancy.
Final bulk count flexes with actual weekend hours; locked by the owner on
#45, not here.

## The same ladder on the real-rock axis (#60)

The corpus-side curve above is measured on held-out MAGEMin labels, and the
traced-corpus regen showed corpus-side and real-rock performance can move in
opposite directions (sealed three-way contrast 2026-07-31). So the "no
saturation" conclusion is unknown on the axis that matters. This re-run
retrains the same nested subsets on the frozen production stack (all-rock
encoder, `joint_heads_allrock_hpo` trial-32 dials, old broad-draw corpus,
120 epochs) and scores every point on the eruption benchmark. The curve's
shape is the decision input for whether a larger broad-draw regen is worth
generating; the owner adjudicates from the numbers.

### Method

- Same md5-nested bulk subsets 625 / 1250 / 2500 / 5000 / 10000, retrained
  by `run_ladder.py` under the `corpus_scaling_allrock_` tag prefix (the
  untagged `corpus_scaling_n*` checkpoints are the #27 points under the
  earlier production stack -- different encoder and dials, not comparable).
  The 10000 point reuses `joint_heads_allrock_tuned_seed0` directly: at
  full N the point config is exactly the production train.
- One seed repeat at 2500 bulks (`--seed 1`) prices training stochasticity
  on this axis, which #27 never measured.
- `run_realrock.py` scores each point three ways: corpus-side val
  (valtest128-broad, teacher-forced hurdle -- the #27 axis at the new
  dials), zero-shot eruption-v2 (headline 640), and matrix-recalibrated
  val (tier fit on the full 368-rock train roster, weight decay 1.0,
  identical protocol per point; val slice only -- the test slice was spent
  2026-07-31).
- Levels on a ~110-rock / 39-unit slice carry more unit noise than
  per-doubling effects, so the readable statistics are paired: all pairwise
  unit-clustered bootstrap contrasts on adapted val joint NLL, plus
  per-rock slopes of adapted NLL against log2(bulks) -- full-ladder (the
  best-powered average per-doubling gain) and top-of-ladder 2500/5000/10000
  (the extrapolation-relevant one, since the corpus-side curve is concave).

### Result

Run 2026-08-06; `results/scaling_realrock.json` / `.png` (matrix tier) and
`results/scaling_realrock_tailed.json` / `.png` (tailed tier, scored the
same day it became canonical; zero-shot numbers are tier-independent and
shared). Plumbing anchors: the n10000 point reproduces the ledger zero-shot
(11.638) and both recalibrated production numbers (1.277 matrix / 0.972
tailed) digit-for-digit.

**Corpus-side replicates #27 at the tuned dials; the real-rock axis is
flat.** Corpus val NLL falls -1.119 -> -2.671 across 625 -> 10000, monotone,
no saturation. Zero-shot 640 *worsens* with corpus size (7.999 at n625 vs
11.638 at n10000), +3.1 nats carried almost entirely by the Dirichlet width
term while presence barely moves: more data of the same kind sharpens
confidence that real rocks do not support. Recalibrated val under the matrix
tier leans harmful (0.910 -> 1.277 across the ladder; per-rock slope +0.114
[-0.043, +0.333] nats/doubling; all ten pairwise contrasts include zero;
seed repeat at n2500 prices training noise at +0.053 [-0.244, +0.375]).

**The harmful lean is recoverable overconfidence, shown by intervention.**
Re-scored under the tailed tier (matrix + 2-param heavy-tail precision
term; deleted 2026-09-03 after its margin was traced to the amphibole pin,
so this read is not reproducible from HEAD and the unpinned matrix tier is
the layer to re-read it with), the corpus-size penalty mostly vanishes: slope +0.031 [-0.081,
+0.176], top-of-ladder -0.013 [-0.236, +0.203], point spread 0.54 -> 0.21
nats. Accuracy is insensitive to corpus size everywhere: share MAE
0.093-0.099 and adapted presence NLL 2.87-2.91 across the full 16x range.
The width mechanism matches the traced-regen finding (sharper labels, worse
zero-shot width) produced by a different knob -- corpus growth instead of
draw narrowing -- which is the mechanism's cross-validation.

### Decision

No larger broad-draw regen: the real-rock axis is flat within ~0.1-0.2
nats/doubling under the canonical instrument, against 0.25-0.30 corpus-side
-- the sim-to-real gap is not a data-quantity problem. The freed background
thread pivots to real-data expansion and a provenance audit of the existing
benchmark (owner decision 2026-08-06; journal).

Correction (2026-09-09): every point on this ladder was stopped where corpus
val was best, about 800 gradient steps at 625 compositions and 5,600 at
10,000, so it varied fitting depth with corpus size. The next section holds
corpus size at fixed depth; the verdict stands.

## Corpus size at fixed fitting depth (#69, #74 step 1)

The zero-shot re-read of the ladder above on absolute error (#69, journal
2026-09-03) found a hump: error rising from 78 to about 625 compositions
and flattening, in every mineral category, with a 78-composition head
beating production on all seven. Each point had been stopped where corpus
val was best, 56 gradient steps at 78 compositions and about 5,600 at
10,000, so the ladder varied fitting depth with corpus size. This section
holds the grid that separated the two, and the two figures every scaling
experiment is reviewed on.

### Method

- `run_variant.py` trains one cell: the standard config with every
  training epoch cut to its first 8 shuffled batches of 1,024 rows, so
  `--set epochs=E` fixes the head at 8E gradient steps and only the
  composition pool differs between cells. The ModeHead is selected on
  corpus val within the epoch budget (`grid` cells), kept at the last
  epoch (`final`), or selected on the real-rock train split's per-epoch
  probe (`stopnll`, `stopmae`; the trainer's `mode_selection` key and
  `epoch_probe` hook, patience 10).
- Tag grammar, which `run_depth_grid.py` enumerates cells by:
  `steps{D}_allrock_n10000` (the full corpus at D steps);
  `depth{D}_allrock_n{N}` (the md5-nested N-composition roster at D
  steps); `final{D}_allrock_n{N}` (same, last-epoch state);
  `stopnll_allrock_n{N}` / `stopmae_allrock_n{N}` (selected on the train
  split's hurdle NLL or share error); `depth{D}_traced_n10000` (the
  traced-regen corpus at D steps); each with an optional `_seed{K}`
  suffix. D is 24 / 56 / 112 / 224 / 448, N is 78 / 312 / 1250 / 5000.
  Rosters are `checkpoints/corpus_scaling_allrock_n{N}/train_bulk_ids.csv`;
  the traced cells pass `--roster data/corpus/train_bulks.csv --set
  corpus_dir=data/corpus/train_traced --set
  valtest_corpus_dir=data/corpus/valtest_traced`. The checkpoint does not
  stamp the seed, the batches per epoch, the selection rule or the roster,
  so the tag is their record.
- `run_depth_grid.py` reads every cell whose checkpoint and zero-shot run
  both exist: per-category absolute error and headline NLL from
  `runs/real_rock/<tag>/metrics.json`, then the matrix recalibrator fit on
  the 372-rock train split and read on the 106-rock / 39-unit val split,
  with per-doubling slopes (least squares on log2 compositions; the
  recalibrated slope is a unit bootstrap over per-rock slopes) and the
  production-minus-cell contrasts. Test stays sealed. The corpus-val ladder
  above is read alongside as its own row. Writes
  `results/depth_ladder.{csv,json}`; `--figures-only` redraws.

```
python explorations/corpus_scaling/run_variant.py --tag depth56_allrock_n78 \
    --roster checkpoints/corpus_scaling_allrock_n78/train_bulk_ids.csv \
    --batches-per-epoch 8 --set epochs=7
python explorations/corpus_scaling/run_depth_grid.py
```

### The two review figures (owner ruling 2026-09-04, amended 09-09)

`results/depth_ladder.png` and `results/depth_ladder_groups.png` are the
figures every scaling experiment is reviewed on, in their current style.
Corpus size on the x axis. The first has four panels for four rulers:
zero-shot share error (mean of seven categories), zero-shot headline NLL,
matrix-recalibrated val NLL, recalibrated val share error. The second has
one panel per mineral category plus the mean, on the zero-shot share-error
ruler. Four curves on every panel: heads at fixed 56, 224 and 448 gradient
steps (last-epoch state where corpus-val selection would stop short; a
hollow marker is a cell it did stop short on), and the head stopped where
share error on the real-rock train split is lowest. One dashed
no-information reference per ruler: the constant forecast (train-split mean
share) for the zero-shot rulers, the head given no chemistry for the
recalibrated ones. Production as a star; bars span the seeds.

A **scaling pair** is a simulated label convention and a measurement
convention whose error falls with corpus size at fixed depth, after
recalibration, on seeds: on these figures, a line that falls in the
zero-shot and recalibrated share-error panels to below the shallow heads.
A candidate corpus gets its cells trained under the grammar above and its
own pair of figures, single seed; if a line looks like a pair, its two
endpoints get three seeds before the word is used. Paper styling (the
ratified frame, fonts, cividis) waits until a pair exists.

### Result

Run 2026-09-04, 78 rows in the csv (journal 2026-09-03 and 09-04 hold the
full read).

- Zero-shot mean-of-seven absolute error at 24 or 56 steps is flat in
  corpus size, 0.101 to 0.104 from 78 to 10,000 compositions. Every depth
  line rises with depth at every size. The hump of #69 was the selection
  set.
- After recalibration corpus size helps modestly at 56 to 224 steps, 0.02
  to 0.05 nats per doubling (0.85 at 78 down to 0.64 to 0.69 at 5,000 to
  10,000, against production's 0.83), all of it width: recalibrated share
  error stays 0.101 to 0.104. The one corpus-size slope whose interval
  excludes zero is the 224-step line, -0.046 [-0.087, -0.010] nats per
  doubling, once the small corpora are held at true fixed depth (a
  78-composition head pushed to 224 steps overfits its own compositions).
- Selection on the real-rock train split picks 16 to 24 steps at every
  corpus size and is flat zero-shot (0.098 to 0.103); those heads
  recalibrate among the worst (0.82 to 0.95). The zero-shot optimum and the
  recalibrated optimum are different heads.
- The traced corpus is 0.003 to 0.007 better zero-shot at 24 and 56 steps
  (three seeds, range under 0.001) and identical after recalibration.
- Seeds: three at 10,000 x 56 / 112 / 224 span 0.002 zero-shot and 0.03 to
  0.05 nats recalibrated; 78 x 56 spans 0.004 and 0.12.

### Decision

Owner, 2026-09-04: fitting depth is the large lever and the generator the
small one so far. No stopping rule is chosen, no head is promoted,
production is untouched. The target of #74 is a scaling pair, searched on
the simulator side (rung, water and pressure prior, redox, recipe, the
phenocryst/groundmass split of the label) through #78 and then #79; the
plutonic branch (#63, #22) is parked until every volcanic pair tried fails
to scale.
