# real_rock_fewshot

**Question.** How many labelled real rocks bridge the sim-to-real gap while
still generalizing to eruptions the recalibrator never saw? (#52)

The zero-shot gap is measured -- frozen MAGEMin-trained heads score joint NLL
10.406 over 640 least-altered volcanic rocks, ranking transferring but absolute
scale not. This asks what closing it costs in labelled real rocks. The
deliverable is the curve's shape, not a single recovered-nats number: a point
estimate on a ~160-rock slice carries roughly +/- 2 nats, and the shape is what
says whether harvesting more eruptions is worth the transcription.

## Method

Frozen encoder, frozen heads. A small `RealRockRecalibrator` corrects the coarsened
seven-group outputs and is fit by minimizing `real_rock_nll` -- the same loss
the ledger reports -- so a gain reads directly as nats recovered. Three nested
tiers: `vector` (23 params: per-group presence scale + offset, per-group
magnitude log-offset, SiO2 slopes on q and amp presence) in `matrix` (121, 7x7
coupling) in `mlp` (1095, with a linear skip). Amphibole's magnitude correction
is pinned at zero at every tier. (Tier names follow the calibration
literature -- vector scaling, matrix scaling. Before 2026-07-27 they were
`diagonal` and `linear`; the journal's #52 entries and the shipped figure
legends carry those older names.)

Split is fixed and grouped by stratigraphic unit (`real_rock_unit_split.csv`,
built by `scripts/build/build_real_rock_split.py`): 368 train rocks / 51 units,
111 val, 161 test / 45 units. Units are assigned by rock count rather than
count of units, or two large units would decide what the test slice measures,
and the test slice's silica range is constrained to sit inside train's so the
SiO2 terms are never extrapolated.

The x-axis is labelled rocks drawn as whole units, ten random draws per budget,
plotted with a 10-90 spread band. Weight decay is chosen once per tier on the
fixed val slice *before* the curve runs and shared across every budget and
draw, so every drawn rock is a fitted rock and "N rocks buys this much" is a
claim someone could reproduce by harvesting N rocks. Comparisons are paired on
the same test rocks and bootstrapped over units, not rocks.

## Result

Zero-shot on the test slice: Dirichlet 2.701, share MAE 0.1068.

| tier | 25 rocks | 50 | 100 | 200 | 368 (all) |
|---|---|---|---|---|---|
| vector | -0.69 | -0.93 | -0.95 | -1.32 | -1.33 |
| matrix | -0.81 | -1.48 | -1.69 | -2.22 | **-2.33** |
| mlp | **-1.11** | -1.56 | -1.65 | -2.01 | -2.08 |

(median test Dirichlet NLL; lower is better)

**Adaptation is cheap and the curve is steeply diminishing.** About 25 labelled
rocks -- eight eruptions -- recovers 3.4-3.8 of the 5.0 nats that the full 368
rocks recover. Fourteen times more data buys the remaining third.

**Coupling pays; nonlinearity does not.** `matrix` wins at every budget from 50
rocks up and reaches 5.03 nats recovered; the MLP's extra capacity never repays
itself. Both beat the per-group `vector` baseline, so the correction the
benchmark needs is genuinely group-coupled rather than seven independent
constants.

At 25 rocks the MLP has the better median, but **that lead is not
established**. The tiers see identical unit draws, so the honest read is the
paired one: the MLP wins 6 of 10 draws by a mean 0.22 nats, against a
draw-to-draw spread of 2.13 nats, and the mean is carried by one unlucky draw
where every tier does badly and the MLP's heavier shrinkage limits the damage
(+0.41 against matrix's +1.73). That is a robustness story about
regularization, not evidence that capacity helps at small n. From 200 rocks up
the sign is unambiguous: matrix beats the MLP on 10 of 10 draws.

**Much of the gap is over-confidence, not wrong means.** The fitted vector tier
drops alpha0 on the scored support from 46.2 to 25.1 -- it widens the Dirichlet
by roughly half. At 25 rocks the vector tier *worsens* share MAE (0.107 ->
0.114) while still improving the Dirichlet term: at that budget it is buying
calibration, not accuracy. Share MAE only improves once there is enough data to
move the means too (0.0876 at full data, matrix).

**At 25 rocks, which units you drew dominates**, exactly as #52 predicted: the
10-90 band spans -1.24 to +0.24 for the vector tier, crossing zero. The band
collapses to a point at the largest budget by construction -- 368 rocks is the
whole train pool, so every draw is the same.

## Decision

Adaptation works and is cheap, so the sim-to-real gap is not a barrier: a
handful of well-characterized eruptions is enough to recalibrate frozen heads.
Carry `matrix` as the default tier. HPO comes next, on train/val.

Two things this does *not* settle. The benchmark's noise floor is unknown --
most of the set is visual estimates at +/- 5-10 vol% and only ~11 rocks are
point-counted -- so the spread band is not a noise floor. And amphibole
detection is untouched by design: a monotone per-group map cannot move AUROC,
and that hands off to corpus regen and #49.

The same curve re-run on post-regen heads is the regen's transfer evidence,
since under the segmented-ledger convention this benchmark is the designated
cross-instrument judge (#45).

## The same result at the level a user meets it

`run_cards.py` renders one held-out rock at a time instead of a curve: chemistry
in, mineralogy out, measurement beside it, zero-shot over adapted on the same
rock. Conditioned on 25 rocks / 10 eruptions, rendered on test-slice rocks only.
The card teacher-forces the measured assemblage ("given these minerals are here,
how much of each?"); the scenario view drops that and samples supports from the
model's own corrected presence, which is the deployment question. Predicted
shares carry their Beta 10-90% interval -- the Dirichlet marginal, closed-form --
because widening that interval is most of what the correction does, and a
mean-only card cannot show it.

Two things the cards make concrete that the curve cannot. On the best-scoring
rock the correction pushes quartz 0.90 -> 0.41 while quartz *is* measured
present: one per-group correction applies to every rock identically, so it buys
aggregate accuracy at the cost of rocks running against the trend. And the two
top scenarios for the median rock differ precisely by amphibole present vs
absent -- the ambiguity this project has been chasing, surfaced in a
user-facing figure rather than a diagnostic.

**Reading "best" needs care.** Ranking is by per-rock adapted joint NLL, and
that score favours rocks spanning MORE mineral groups: a one-group composition
is a degenerate point mass contributing exactly 0, while a five-group one has a
higher-dimensional simplex on which a sharp Dirichlet earns strongly negative
log-density. Measured on the test slice, mean adapted NLL runs 6.9 / 6.9 / 7.3 /
1.4 / -2.0 at 1/2/3/4/5 groups, and every rock in the top 20 has at least four.
Number of reported *elements* is unrelated (r = +0.006). Composition size is
stamped on every card for this reason.

## Gotcha found here

Two, both silent.

The coupled tiers read `log(alpha)` as an input feature, and coarsening leaves
exact zeros wherever a group has no member phase on the support. Clamping
those to `1e-12` before the log hands the mixing matrix a -27 spike on exactly
the groups carrying no information, and `matrix` / `mlp` diverge on it -- val
NLL reaching 1e20, with an erratic non-monotone weight-decay curve that reads
like "the coupled tiers do not help." They do; an off-support group's feature
is the neutral 0. Worth knowing before adding any further input to this module.

And the recalibrator judges groups while deployment needs a 21-phase support, so the
group verdict has to be pushed back down. Shifting each member's *logit* by the
group's logit shift is wrong, because the noisy-OR is not linear in logits: a
group at 0.9999 corrected to 0.84 gives a -7.5 shift that drives both members
to ~0, so the group vanishes from every sampled support. Caught only because a
scenario view came back with no feldspar in a rock that is half feldspar. The
exact transform raises each member's absence probability to a common power,
`gamma = log(1 - P_target) / log(1 - P_now)`. That function also feeds the
exposure probe, so this was never only a plotting bug.

## Residual decomposition: width vs location

`run_residual_decomposition.py` asks what the recalibrator's functional form can
reach at all, before any hyperparameter sweep on it: fit vector and matrix on
the train roster, then split each group's remaining miss into width (interval
too narrow) and location (share systematically off). The oracle reparameterizes
each group's Beta marginal into mean and precision -- `k` scales precision at
fixed mean, `delta` shifts the mean at fixed precision -- and grid-searches the
coverage each knob can buy. Those two directions are exactly what the
recalibrator's own alpha offsets entangle: a per-group `magnitude_offset` moves
mean and concentration together, and the only mean-preserving direction in the
parameterization is an offset shared by all seven groups.

Measured on the regen arms (2026-07-31, val slice, fit on train, test sealed):

- The fit's constant offsets absorb most of the location error. Zero-shot
  medians run to 2.8x off (bi, opq under; fsp, q over); after the matrix fit
  every group's median ratio is within 1.4x and mean PIT sits near 0.5. The
  zero-shot suspicion that opq/bi were location-limited does not survive the
  fit -- opq lands nearly centered.
- What remains is under-dispersion everywhere: empirical coverage of the
  nominal 80% interval is 0.45-0.75 (matrix) against 0.12-0.42 zero-shot.
  Reaching nominal by pure widening needs precision cuts of 6x to 1000x, and
  for most groups the interval at that optimum spans half the simplex or more
  -- coverage bought vacuously. The two groups whose optimum stays sharp (px,
  bi) are also the two whose width ceiling falls short of nominal. So the
  coverage gap is not a missing-knob problem: the residual is heavier-tailed
  than any Beta at the fitted mean, which points at mixture / per-rock
  precision structure, not more constant knobs.
- The location residual that does survive is SiO2-structured (Spearman of
  signed residual vs SiO2: bi +0.6, q and px +0.5, stable across arms and
  tiers). The recalibrator carries SiO2 slopes on presence only; a
  magnitude-side SiO2 term is the one targeted parametric extension the data
  argues for.
- Recalibration nearly closes the tuned-vs-baseline zero-shot gap: val joint
  NLL 31.7 vs 25.4 zero-shot collapses to 2.0 vs 1.7 after the matrix fit.

Numbers and per-group detail in `results/<tag>/residual_decomposition.json`;
the figure annotates each width ceiling with the `k` that buys it, so a bar at
1.0 reached only by a 100x precision cut reads as the vacuity it is.

## The q/amp presence slopes, measured at last

`run_presence_slope.py` supplies the contrast the presence-side SiO2 slopes
never got. They entered the parameterization at design time from the
residual-vs-SiO2 read above, while the magnitude location term that read also
argued for was built, measured at +0.007 [-0.004, +0.016] nats, and excised
(journal 2026-08-06). The presence slopes stayed, unmeasured, as 2 of the
canonical tier's 123 parameters.

The tailed tier is fit twice on the train roster, once with `presence_sio2`
free and once held at its zero initialization, and scored paired on val.
Measured 2026-08-18 on `joint_heads_allrock_tuned_seed0` (368 train rocks / 51
units, 111 val rocks / 39 units):

- **The slopes are worth +0.002 [+0.001, +0.002] nats**, all of it in the
  presence term. For scale, the precision contamination in the same tier is
  worth +0.304 [+0.049, +0.659], and the excised magnitude term measured
  +0.007. Val joint goes 0.974 pinned to 0.972 free; share MAE is 0.0954 in
  both arms to four decimals.
- The interval excludes zero only because the contrast is deterministic: both
  tiers initialize at zero/one, so the arms differ by the slopes alone with no
  seed noise to widen it. A reliably positive 0.002 nats is still nothing.
- The fitted slopes are small in their own units: +0.048 logit per 10 wt% SiO2
  for quartz and +0.003 for amphibole, against a benchmark spanning roughly
  50-78 wt%.
- Re-selecting the pinned arm's shrinkage constant on val returns the same 1.0
  the free arm uses, so the contrast is not an artifact of sharing one.

The abundance term is bit-identical across the arms, as it must be: presence
and abundance parameters draw gradients from separate terms of the loss, which
is what makes this contrast the slopes' attribution rather than a two-change
comparison.

## Re-supervising the head instead of correcting it (#53)

The three correction tiers all start from the simulation-trained head's
forecast.  Three more arms ignore it entirely and train a 7-group hurdle head
(`MeasuredModeHead`) on measured proportions alone, on the same rocks, the same
draws, the same objective, differing only in what they are given about a rock:
the frozen encoder's latent representation, only the assay that latent was
computed from, or neither.  All six arms fit the identical drawn rocks at each
(budget, draw), so every comparison is paired within the draw.

Read the paired per-draw difference, never the difference of medians.  At the
26-rock budget the two differ by almost a nat (+1.38 against +2.28) because the
arms' worst draws are different draws.  For the same reason the 10-90 bands on
the figures overlap almost completely while the paired sign is 10/10 -- the
bands understate every contrast here, and a paired-difference panel is what
the figures owe.

Three things separate cleanly, all on the val slice at the full 368-rock roster:

- **The encoder earns its place, on presence.** A head given the latent beats
  one given only the assay by +0.50 to +0.67 nats at every budget swept (9-10
  of 10 draws), and beats one given neither by +0.77.  The ordering on the
  presence term alone is 2.77 / 3.36 / 3.55.  Below ~50 rocks the assay-given
  head is beaten outright by the one given neither: twenty-two raw numbers
  overfit where the latent does not.
- **Nothing the head reads changes the amounts.**  Given the measured mineral
  list, the amount term is 2.50 / 2.50 / 2.51 nats across the three -- the head
  that cannot tell one rock from another matches the one reading full
  chemistry.  Whether that is petrology or the benchmark's 7-group resolution
  is not resolved here.
- **The simulation-trained head is the only source of per-rock amount
  information.** It wins absolute share error (0.0954 against ~0.1035 for all
  three re-supervised heads, 10/10 draws from 210 rocks) while losing the
  likelihood badly (0.97 against 0.27).  Better centre, wrong confidence.  Its
  share error only clears the zero-shot line at ~106 rocks; at 26 it is worse
  than no fit at all.

`measured_ablated` is not blind to the rock -- every arm receives the true
mineral list as input to its amount branch.  It knows the assemblage and no
chemistry.

## Adapting the head instead of correcting its outputs

The correction tiers fit at most 123 parameters on the frozen head's outputs,
so the obvious objection to their loss above is that the head was never
adapted -- it was rescaled.  A seventh arm (`adapted`) removes that reading:
rank-5 low-rank updates inside each of the head's eight linear layers, trained
on the same measured labels through the same objective, on the same draws.  At
33,595 trainable parameters against `MeasuredModeHead`'s 34,839 the two arms
are compared at a matched trainable budget -- a fairness gesture rather than a
capacity claim, since a low-rank perturbation of a trained 1.35M-parameter head
is not the hypothesis class of a 35k-parameter head fit from scratch.

Everything else stays frozen, including the 21-phase support the amount branch
is conditioned on: it is a threshold plus an argmax, so it carries no gradient
in any case, and letting it track the adapters would only make the objective
jump whenever a member phase crossed 0.5 mid-fit.  Pinning it also keeps this
arm's Dirichlet on the same support the correction arms' `alpha7` was built on.

Two asymmetries to read the result against.  This arm inherits the frozen
head's single Dirichlet, while `tailed` and every measured arm are
two-component and `tailed`'s whole margin over `matrix` was its precision
contamination -- so the same-K comparison here is `matrix`.  And it runs the
head inside the training loop, which is why the coarsening onto the seven
measured groups is reimplemented in torch (`adapted_mode_head.py`); that form
carries the noisy-OR in logs rather than probabilities, because the shipped
numpy path recovers a nearly-absent group's probability by subtracting a
near-one product from one, which is survivable in a reported number and not in
a gradient.

`lr` and `epochs` come from `run_plateau_probe.py`, which reads the train loss
and never val, so this arm spends the same single val-selected scalar every
other one does.  Probing at zero shrinkage measures nothing: 33k adapters
against at most 368 rocks drive the Dirichlet density down without bound, and
the train loss is still descending 0.4 nats per 500 epochs at epoch 2000 with
no plateau to find.  The probe therefore runs at the shrinkage the measured
arms and both surviving correction tiers already selected, and this arm's own
constant is chosen on val afterwards with `lr` and `epochs` already fixed.  That
lands at lr 1e-3 over 1636 epochs, and a shrinkage of 10 -- an order of
magnitude above what any other arm wanted, which is what 33k adapters against at
most 368 rocks cost.  Over-shrinking past it returns the frozen head exactly
(val joint 5.66 -> 9.35 -> 13.44, and 13.44 is the zero-shot number), which is
the identity-at-initialization contract showing up in a fit rather than a test.

### The result, and the one place the protocol pinches

Median val joint NLL against zero-shot 13.44:

| budget (rocks) | 26 | 54 | 106 | 210 | 368 |
|---|---|---|---|---|---|
| adapted | 7.19 | 5.75 | 5.86 | 4.97 | 4.86 |
| adapted, per-budget oracle shrinkage | 7.62 | 3.11 | 3.97 | 1.74 | 1.08 |
| matrix | 3.97 | 2.62 | 2.36 | 1.37 | 1.28 |
| tailed | 3.97 | 2.59 | 2.08 | 1.11 | 0.97 |
| measured | 1.69 | 1.25 | 1.12 | 0.38 | 0.27 |

**Adapting the head from the inside does not close the gap, and mostly does not
even match correcting it from the outside.**  That holds under the shared
protocol and it still holds after handing this arm an advantage no other arm
gets: the oracle row picks shrinkage per budget on val and is then scored on
val, and it loses to `tailed` at all five budgets and to `measured` by a wide
margin throughout.  So the reading that the correction arms lost only because
123 parameters cannot carry a real adaptation does not survive.  (Oracle row:
3 draws rather than 10, and different draws from the sweep's, so the 26-rock
entry is noisy -- it is an upper bound, not a curve.)

**The loss is almost entirely presence.**  At the swept shrinkage the arm's
Bernoulli term is 5.813 against `matrix`'s 2.896 -- essentially unmoved from
zero-shot's 5.813 -- and that accounts for 82% of the gap.  The amount branch
goes the other way: at a shrinkage of 1 its Dirichlet reaches -2.05 against
`matrix`'s -1.62, and its share MAE at 26 rocks is 0.099, the lowest of any arm
on the curve (`tailed` 0.114, `measured` 0.111).

The mechanism is the arm's own parameterization.  A per-group presence
correction is an intercept, and a low-rank update with no bias term cannot
express one cheaply -- it has to route a constant through `B @ A @ latent`.
That intercept is exactly what the correction arms get from seven
`presence_offset` parameters, and what `MeasuredModeHead` gets from biases its
fit deliberately exempts from shrinkage.  Adding zero-initialized bias deltas
(~3.1k parameters, identity at init preserved) is the targeted test, and it was
not run: the oracle bound already loses to `tailed` everywhere, so closing the
presence gap is unlikely to reverse the ordering.

**One protocol caveat, which this arm is the first to expose.**  Its best
shrinkage slides 30x across the swept range -- 10, 3, 3, 1, 0.3 from 26 rocks to
368 -- while `select_weight_decay` searches sizes 25 to 200 and the curve sweeps
to 400.  For a 23- to 123-parameter correction the optimum is flat and the
mismatch is invisible; for 33k adapters a near-tie at small budgets (5.76
against 5.66) picks a constant that costs 3.7 nats at the largest.  The single
val-selected constant is kept anyway, because it is what makes every arm's
x-axis mean the same thing, and the conclusion does not depend on the
handicap.

## Files

- `run_curve.py` -- the sweep, the figures, `results/<tag>/curve.json`.
  `--replot` redraws from the recorded curve without refitting, so figure
  wording does not cost an eleven-minute sweep
- `adapted_mode_head.py` -- the low-rank adapters, the torch coarsening they
  train through, and the fit/score pair the `adapted` arm plugs into
- `run_plateau_probe.py` -- fixes that arm's `lr` and `epochs` off the train
  loss alone; writes to a scratch directory, not `results/`
- `run_residual_decomposition.py` -- width-vs-location residual decomposition
  after the fit, `results/<tag>/residual_decomposition.{json,png}`
- `run_tailed_acceptance.py` -- the tailed-vs-matrix acceptance read and the
  three-arm coverage/sharpness figure, `results/<tag>/tailed_acceptance.json`
- `run_presence_slope.py` -- the q/amp presence-slope ablation,
  `results/<tag>/presence_slope.json`
- `results/joint_heads_supportmag_seed0/learning_curve.png` -- the deliverable
- `results/joint_heads_supportmag_seed0/curve_by_units.png` -- priced in eruptions
- `results/joint_heads_allrock_tuned_seed0/learning_curve_val.png` -- the
  production re-run with the tailed tier
- `results/joint_heads_allrock_tuned_seed0/share_mae_val.png` -- absolute share
  error against budget, the four arms that communicate
- `results/joint_heads_allrock_tuned_seed0/arm_contrast_val.png` -- paired
  per-draw differences against the head given the latent

Backing: journal 2026-07-25 (curve), 2026-07-31 (decomposition), 2026-08-05/06
(tailed tier + production curve re-run),
`results/real_rock_ledger.yaml` (`eruption-v2`),
`explorations/real_rock_mapping/` (the truth mapping this rests on).
