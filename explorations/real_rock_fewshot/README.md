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

## Files

- `run_curve.py` -- the sweep, the figures, `results/<tag>/curve.json`
- `run_residual_decomposition.py` -- width-vs-location residual decomposition
  after the fit, `results/<tag>/residual_decomposition.{json,png}`
- `results/joint_heads_supportmag_seed0/learning_curve.png` -- the deliverable
- `results/joint_heads_supportmag_seed0/curve_by_units.png` -- priced in eruptions

Backing: journal 2026-07-25 (curve), 2026-07-31 (decomposition),
`results/real_rock_ledger.yaml` (`eruption-v2`),
`explorations/real_rock_mapping/` (the truth mapping this rests on).
