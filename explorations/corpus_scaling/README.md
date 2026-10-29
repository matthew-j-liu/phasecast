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
  suffix (K 1 or 2; seed 0 carries none). D is 24 / 56 / 112 / 224 / 448,
  N is 78 / 312 / 1250 / 5000, and 10000 for `final` (no `--roster`).
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
  above is read alongside as its own row. Each cell also carries its
  val-only share error and Pearson r per category, zero-shot
  (`val_mae_{g}`, `val_r_{g}`) and recalibrated (`recal_mae_{g}`,
  `recal_r_{g}`); the zero-shot `mae_{g}` columns the figures draw are
  read over all 640 rocks from `metrics.json`. Each fixed-depth line
  carries per-doubling slopes of those, per category and on the
  four-silicate, other-three and seven-category means, with a unit
  bootstrap over the val rocks (#85). Writes
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
  Rebuilt 2026-10-02 from last-epoch heads on three seeds
  (`final224_allrock_n{N}[_seed{K}]`), it reads -0.032 [-0.069, +0.0005],
  and recalibrated share error is flat on the four silicates (journal
  2026-10-05).
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

## The zero-shot line per mineral, filled below 78 compositions (2026-10-27)

The #85 read above was the recalibrated one. Its zero-shot columns
(`val_mae_{g}`, the 106 val rocks) are the signal for a scaling pair, since
the recalibrator is fit on hundreds of measured rocks and does much of the
work on its own, so a head trained on very little simulated data looks
mature after it. Read zero-shot, the 224-step line falls from 78 to about
1,250 compositions on quartz and pyroxene and is flat after, which left the
rising regime sampled by two points. This section fills it and draws the
line per mineral.

### Method

- Five sizes added to the `final224_allrock_n{N}[_seed{K}]` family: 10, 20
  and 40 (new md5-prefix rosters, `run_ladder.py --rosters-only`) and 156
  and 625 (rosters already on disk). Three seeds each, last-epoch state,
  48 draws and the two rungs held.
- Below 78 compositions an epoch holds fewer than the 8 truncated batches
  of 1,024 rows (one batch at 10, two at 20, four at 40), so the epoch
  count scales to hold 224 gradient steps: `--set epochs=224` / `112` /
  `56`, against 28 from 78 up. A step below 78 compositions therefore sees
  a partial last batch (960, 896 or 768 rows). `selected_steps` in
  `run_depth_grid.py` records the step count with the epoch's own batch
  count, and `SIZES` carries the ten sizes.
- `fig_zero_shot_scaling.py` draws the line in the paper's style
  (`paper/figures/src/style.py`, imported at draw time): eight panels
  grouped by family (feldspar, quartz; pyroxene, olivine; amphibole,
  biotite; Fe-Ti oxides, mean of seven), seed mean with the seed range as
  bars, every corpus size labelled. No reference lines and no fitted
  slopes: the curves are read by eye first and any slope convention is
  chosen after.

```
python explorations/corpus_scaling/run_ladder.py --rosters-only --points 10,20,40
python explorations/corpus_scaling/run_variant.py --tag final224_allrock_n10 \
    --roster checkpoints/corpus_scaling_allrock_n10/train_bulk_ids.csv \
    --batches-per-epoch 8 --set epochs=224 --mode-selection final   # and seeds 1, 2; 20/112, 40/56, 156/28, 625/28
python explorations/corpus_scaling/run_depth_grid.py
python explorations/corpus_scaling/fig_zero_shot_scaling.py --depth 224   # and 56, 896, 3584
```

### Result

The figures at `results/gradient_steps/zero_shot_scaling_{D}.png` now show
train and val rocks and the section below supersedes this one; the numbers
here are the val-only read, kept in `results/depth_ladder.csv`.

`results/gradient_steps/zero_shot_scaling_{D}.png`, one per fitting depth; the fifteen new cells all report 224
steps, the earlier rows reproduce. Zero-shot absolute error on the val
rocks, in percent, seed means; the constant forecast (train-split mean
share over each rock's measured list) in parentheses where it matters:

- Mean of seven: 13.6 at 10 compositions, 12.9 at 40, 12.5 at 625, 12.3 at
  10,000. The rise runs from 10 to about 625 and the plateau starts there,
  not at 1,250.
- Feldspar falls hard from 10 to 40 (20.2 to 17.1) and wanders after
  (16.6 at 10,000). Amphibole drifts down the whole way (14.5 to 12.1),
  with the widest seed range at the small end.
- Quartz and pyroxene are not monotone: both worsen from 10 to 40 (quartz
  17.0 to 18.8, pyroxene 7.5 to 8.8 at 20) and then fall to their plateau
  by 625 to 1,250 (13.6 and 6.8). These two carry the mean's fall past 78.
- Biotite worsens monotonically from 10 to 625 (10.0 to 12.1) and stays
  there, all three seeds agreeing at every size; the constant forecast
  (9.4) beats every head. The mechanism is on record: the equilibrium paths
  convert amphibole into biotite as they cool (the seven-mineral ladder
  read, `generator_reach`), so the simulated labels carry more biotite
  than du Bray's phenocryst counts. A head with little data predicts
  something near an average share; each doubling teaches it more of the
  simulator's biotite, which the phenocryst counts do not carry, until it has learned
  it fully. Biotite is a category where learning the simulator better
  hurts the zero-shot read.
- Olivine is a U: 17.8 at 10, 13.7 at 40 and 78, above 16 from 1,250. The
  shape holds across seeds but 13 val rocks quantify olivine (seed range
  four points at 10 compositions), and #78 found olivine recipe-limited,
  so it is not built on. The per-rock predictions on the 13 rocks across
  sizes would say whether it is biotite's story with a dip first.
- Fe-Ti oxides never leave their seed range (7.8 to 8.4) and never
  approach the constant forecast (6.0): a redox and recipe problem (#46,
  #78), not a data problem. Their panel autoscales to under one point, so
  the seed bars look large there and are not.

### Open

The line is compute-matched, not convergence-matched: 224 steps is 24
passes over a 10-composition corpus and 2% of one pass over 10,000. How
much of the shape belongs to that choice, across fixed steps, train to
convergence and fixed epochs, is #92.

### The 56-step line, and why shallower reads better zero-shot

`results/gradient_steps/zero_shot_scaling_56.png`, same ten sizes and three
seeds. Mean of seven 12.2 at 10 compositions, 11.3 by 40, flat to 10,000,
against 13.6 to 12.3 at 224 steps: lower at every size, and the rise ends
by 40 compositions rather than 625. Amphibole and olivine fall from 10 or
20 to 40 and are flat after; olivine sits near 12.3 against 16 to 17 at
224 steps, so the U at 224 is a depth effect. Biotite rises again, 9.0 to
11.4 by 156. Quartz and pyroxene settle by 40 to 312. The oxides are best
at 10 compositions (6.3, near the constant forecast's 6.0) and flat at 7.3
after.

The reading, with what is measured and what is not. The head is trained to
match the simulator, and the simulator is biased against the phenocryst
measurements in a consistent direction per mineral (the ladder read: biotite made from
amphibole, olivine overstated). Early in training the head predicts
something near the corpus mean and moves toward the simulator's per-rock
labels with depth; for the minerals the simulator gets wrong, the halfway
point beats the destination. Measured: zero-shot error rises with depth at
every size (24 to 448 steps, now 56 against 224 on ten sizes); the
simulator's own draw mean on the benchmark chemistry is worse zero-shot
than every head on every category (`generator_reach/results/signal.csv`),
and it is the limit a perfect head reaches; the constant forecast beats
every head on the oxides and, on biotite, every head but the 10-composition
one at 56 steps (journal 2026-10-27, the correction entry); seed ranges do
not widen with depth; the recalibrated read reverses the depth order. Not measured: that the
heads' predictions move toward the simulator's draw mean as depth grows.
Until that distance is measured (#92) the sentence "shallower heads are
worse models of the simulator that sit closer to the rocks" is a reading,
not a result.

Refinement of the ceiling argument behind #83: "a head cannot beat the
simulator" holds for information (after recalibration the head and the
simulator tie), not for raw zero-shot error, where under-training acts as
shrinkage toward an average that happens to sit closer to the rocks than
the simulator's bias. The plutonic order of operations stands; the bound
is not to be stated on zero-shot numbers.

Getting under the constant forecast zero-shot on biotite and the oxides
is a label-side question, since both move away from the constant as the
head learns more: the fractional recipe on the volcanic rocks (#84, on
disk, read after #85), a per-mineral read point (#88), the redox input for
the oxides (#46), or supervision on measured rocks, which is calibration
rather than zero-shot. More simulated data or another depth will not do
it.

Depth sweep (2026-10-27): 56 and 896 steps on three seeds beside
224; the 3,584 row dropped as not worth its compute for this argument. The
other two protocols and the compute view are #92.

### The 896-step line

`results/gradient_steps/zero_shot_scaling_896.png`. Mean of seven 14.0 at
10 compositions, 13.2 to 13.4 from 20 to 1,250, 12.6 at 5,000, 13.0 at
10,000: higher than the 224-step line at every size, and the size effect
is a single step from 10 to 20 followed by wobble within the seed range.
So the depth ordering holds at every corpus size (56 below 224 below 896
on the mean), and the deeper the head, the less corpus size does. Per
mineral: quartz still falls (17.6 at 10, 20.0 at 40, 14.4 by 1,250);
pyroxene drifts 7.9 to 6.7; feldspar drops 19 to 17.5 by 40 and is flat;
amphibole is flat near 14 to 312 and falls to 12 by 5,000; olivine's U is
deeper (20.8, 14 at 40 to 78, 18.5 at 10,000); biotite rises 9.5 to 12 by
625; the oxides are 9.2 at 10 and near 8 to 8.8 after, their worst row.
The three depth rows together are the sensitivity read: the shape of the
corpus-size line depends on depth mainly through where its fall ends (40,
625, 20 compositions at 56, 224, 896 steps) and its level, while the
per-mineral directions (quartz and pyroxene down, biotite up, oxides flat
or up) hold at every depth.

### Head against the simulator

`fig_head_simulator_distance.py` -> `results/gradient_steps/head_simulator_distance.{csv,png,pdf}`.
Every cell of the depth sweep (56, 224, 896 steps x ten sizes x three
seeds) read on the 106 val rocks against two targets. The simulator target
is its draw mean over the pooled 50 and 30% rung rows under the broad box,
exactly `signal.csv`'s `simulator` row; the head is read on the same basis,
a marginal over assemblages sampled from its own presence probabilities
(`run_signal.marginal_shares`), because a teacher-forced head against a
marginal simulator differs on every mineral whose presence varies across
draws (a first draft did that and amphibole and olivine read backwards).
The distance to the rocks keeps the teacher-forced read, the per-depth
figures' number, and reproduces `depth_ladder.csv`'s `val_mae_*` to 1e-16;
the simulator's own distance to the rocks reproduces `signal.csv` to 1e-16.
Colour encodes depth on this figure (224 heavy in the dark cividis, 56 and
896 lighter), the one departure from the locked single-line style.

Mean of seven, val rocks, percent, seed means:

| compositions | to the simulator, 56 / 224 / 896 steps | to the rocks (forced), 56 / 224 / 896 | to the rocks (marginal), 56 / 224 / 896 |
|---|---|---|---|
| 10 | 9.4 / 7.7 / 8.1 | 12.2 / 13.6 / 14.0 | 14.0 / 15.1 / 15.4 |
| 40 | 4.3 / 4.1 / 4.4 | 11.5 / 12.9 / 13.3 | 15.6 / 16.6 / 16.5 |
| 625 | 2.9 / 2.5 / 2.3 | 11.3 / 12.5 / 13.3 | 15.2 / 15.7 / 15.9 |
| 10,000 | 3.3 / 2.5 / 2.3 | 11.3 / 12.3 / 13.0 | 15.2 / 15.6 / 16.2 |

The simulator's own distance to the rocks on the same basis is 15.9.

**Result.** The distance to the simulator falls with corpus size at every
depth (7.7 to 9.4 at 10 compositions, 2.3 to 3.3 at 10,000, most of it by
312) and, from a few hundred compositions on, with depth (896 below 224
below 56 at 625 and 10,000). Per mineral at 10,000 compositions the deeper
head is the closer copy on biotite (2.9 / 1.5 / 1.2), the oxides (3.2 /
2.7 / 1.9), quartz (4.3 / 3.4 / 2.7) and feldspar (5.1 / 3.4 / 3.0);
amphibole and olivine sit near the simulator at every depth (about 3 and
1.5). Over the same cells the teacher-forced distance to the rocks rises
with depth, and the marginal distance to the rocks climbs toward the
simulator's own 15.9 (15.2, 15.6, 16.2 at 10,000). So the heads converge
on the simulator as they see more compositions and train longer, and
inherit its error against the phenocryst measurements as they do: the
sentence "shallower heads are worse copies of the simulator that sit
closer to the rocks" is now a measurement for biotite, the oxides, quartz
and feldspar, and the depth ordering of the per-depth figures is the
simulator's bias arriving, not noise. The one qualification: the shallow
head's advantage on the rocks is also partly a basis effect, since on the
marginal basis every head sits within a point of the simulator's 15.9 from
625 compositions up.

## The zero-shot line by training subset, both benchmarks (2026-10-28/29)

**Question.** Read on every measured rock zero-shot can use, and averaged
over the choice of which compositions a head trains on, does zero-shot
abundance error fall with corpus size, on the volcanic and on the plutonic
benchmark?

**Setup.**
- The simulated corpus holds 10,000 gard2019 compositions (9,946 with
  labels). Each contributes 96 rows: 48 (P, H2O) draws read at the 50 and
  30% rungs. A row is one composition, one draw, one rung.
- A training subset is N compositions drawn at random from the corpus. At
  each N from 10 to 5,000, 50 training subsets each train one head at seed
  0, so heads at one N differ only in their compositions. At 10,000 there
  is one subset, the whole corpus, trained at seeds 0, 1 and 2.
- Every head trains for 224 gradient steps. A gradient step is one
  parameter update on a batch of 1,024 rows, so compute is fixed at about
  229,000 rows at every N. At N=10 the corpus is 960 rows, less than a
  batch, so each step is one pass over all of it.
- Parameters change only through the training loss, a negative
  log-likelihood (Bernoulli presence and Dirichlet abundance). The
  validation set is simulated (held-out corpus compositions) and does not
  choose the abundance head, which is kept at its last epoch.
- Evaluation is zero-shot: the heads never see a measured rock. Every head
  is read on the same rocks, each benchmark's train and val splits (test
  sealed): volcanic 478 rocks in 86 eruptive units (phenocryst modes),
  plutonic 404 rocks in 61 intrusions (whole-rock modes). Shares are read
  teacher-forced over each rock's reported minerals; the absolute error is
  averaged over rocks per mineral category, then over the seven categories.

**Method.**
- `run_zero_shot_line.py --benchmark {volcanic,plutonic}` reads every head
  (the nested-subset `final{56,224,896}_allrock_n{N}` heads with their
  seeds, and the random-subset `final224_allrock_n{N}_subset{r}` heads) and
  writes one row per head and rock group. Restricted to volcanic val rocks
  it reproduces `depth_ladder.csv` exactly.
- A unit bootstrap (2,000 resamples of whole units, the same resample at
  every size) bands the nested-subset line and its rise (78 - 10), fall
  (10,000 - 78) and net (10,000 - 10). The val rocks' rise, fall and net
  are placed among random draws of as many train and val units.
- Plutonic rocks are also read by pretraining exposure. Each rock's eight
  majors, renormalized to 100 with total iron as FeO, are matched (every
  oxide within 0.02 wt%) against the rebuilt 787,721-row pretraining set; a
  rock is seen if any copy sat in pretraining train or val, held out if
  every copy sat in pretraining test. DS 942: 281 seen, 48 held out; PP
  1793: 243 absent, 1 seen, 2 held out
  (`results/plutonic_pretraining_membership.csv`).
- `run_training_subsets.py` trains the random subsets through
  `run_variant.py` with the `final224` protocol. Heads take about 216 s at
  N=10 (224 epochs, each with a validation pass), 120 s at 20, 70 s at 40
  and 44 s from 78 up; the 450 took about 8.1 h on a laptop.

```
python explorations/corpus_scaling/run_training_subsets.py
python explorations/corpus_scaling/run_zero_shot_line.py --benchmark volcanic   # and plutonic
python explorations/corpus_scaling/fig_zero_shot_scaling.py --benchmark volcanic --depth 224   # 56, 896; plutonic
python explorations/corpus_scaling/fig_zero_shot_scaling.py --benchmark plutonic --depth 224 --mean-vs-median
```

Outputs under `results/`: `zero_shot_line_{benchmark}.csv` (one row per
head and rock group), `zero_shot_line_{benchmark}_bands.json`,
`plutonic_pretraining_membership.csv`, and under `gradient_steps/` per
benchmark the nested-subset line per depth, the `_224_subsets` line, the
`_224_variation` figure (seed, rock and training-subset variation on one
axis), the plutonic `_pretraining` figures and `mean_vs_median_{benchmark}`.

**Result.**
- **The nested-subset line was one draw.** On train and val rocks the
  volcanic nested line rises from 11.0 at 10 to 12.1 at 40 to 78 and falls
  back to 11.0 at 10,000; on val alone it had fallen from 13.6 to 12.3.
  The rise is not established (+1.13 [-0.62, +2.78] points, unit
  bootstrap), the val rocks' net change sits at the 7th percentile of
  random 39-unit draws, and the nested point at 78 lies above all random
  subsets. Below about 78 compositions the training subset dominates: at
  volcanic N=10 the 50 subsets span 10.3 to 15.6 points against a seed
  range of 0.35.
- **Averaged over 50 training subsets, error falls with corpus size on both
  benchmarks.** Mean of seven, percent (median over subsets in brackets,
  standard error of the mean below 0.3 everywhere):

| N | 10 | 20 | 40 | 78 | 156 | 312 | 625 | 1,250 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| volcanic | 12.17 (11.99) | 11.74 (11.57) | 11.44 (11.40) | 11.30 (11.33) | 11.35 (11.32) | 11.38 (11.31) | 11.18 (11.18) | 11.08 (11.05) | 11.03 (11.06) | 11.02 |
| plutonic | 7.36 (7.01) | 6.53 (6.53) | 6.34 (6.14) | 6.26 (6.35) | 6.14 (6.15) | 5.99 (6.04) | 5.82 (5.82) | 5.78 (5.79) | 5.66 (5.66) | 5.95 |

  The 10,000 column is the full corpus at three seeds (plutonic 5.70 to
  6.42, inside the 5,000 subsets' 5.17 to 6.16). Spread across subsets
  shrinks with N: the middle half spans 1.4 points at volcanic N=10 and
  0.2 at 5,000, 2.8 and 0.3 plutonic.
- **Per mineral, 10 to 5,000.** Volcanic: feldspar 15.8 to 14.0, pyroxene
  10.0 to 7.0, olivine 12.5 to 9.7 and amphibole 13.6 to 11.6 fall; biotite
  (8.1 to 8.6) and quartz (17.6 to 18.4) rise; oxides are flat. Plutonic:
  feldspar, quartz, pyroxene and amphibole fall; biotite and oxides are
  about flat; olivine rises on very few rocks.
- **Saturating power law.** error(N) = E + A N^-alpha fit to the subset
  means and medians, intervals from resampling the 50 subsets: alpha 0.37
  to 0.66 with intervals spanning about 0.15 to 2.7; E about 11.1
  (volcanic) and 5.6 to 5.7 (plutonic), intervals not overlapping; the
  reducible part A N^-alpha is 0.9 to 1.6 points at N=10 and nearly gone
  by 1,000. Typical deep-learning data exponents are about 0.07 to 0.35;
  ours sit above them at face value but every interval reaches into them,
  and the comparison is not like for like (fixed compute, zero-shot on
  real rocks, under three decades of N). Whether E is a floor is not
  established: within 10 to 5,000 a floor and a slow continued decline
  fit equally, and the fixed 224 steps may be what flattens the curve.
- **Pretraining exposure.** Within DS 942 the held-out rocks read 0.5 to
  1 point worse than the seen rocks at most sizes with a similar shape (31
  held-out rocks in train and val): consistent with exposure flattering
  the level, not the slope, and not established.

**Protocol this section settles for reading a scaling line.**
1. Fix compute (gradient steps) across N and state the depth.
2. Average over many random training subsets (50 here), above all at small
   N; a single subset there is a draw, not a point.
3. Hold the seed fixed across subsets so their spread is the subset alone;
   vary seeds where there is one subset.
4. Read zero-shot on the train and val rocks (nothing is fit to them); test
   stays sealed.
5. Read every head on the same rocks with the basis stated.
6. Summarize with the median and quartiles; back a trend with an interval
   over subsets; say which source of variation a bar shows.
7. Band rock sampling by unit bootstrap, and check a subset of rocks is
   ordinary before reading it alone.
8. Read every result per mineral, not only as the mean of seven.
9. Fit E + A N^-alpha with intervals, and state that under three decades of
   N, alpha and E trade off.
10. Split plutonic reads by pretraining exposure.
11. When the reading code changes, reproduce the previous numbers first.
12. Compare matched: same rocks, same draws, across what is compared.

**Open.** The full curve needs larger runs: more training subsets, more
seeds, more compute per head, and corpus sizes beyond 10,000, which means
new simulation. Whether more compute at large N lowers the flat end is #92.
