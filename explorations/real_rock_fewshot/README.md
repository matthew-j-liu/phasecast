# real_rock_fewshot

**Question.** How many labelled real rocks bridge the sim-to-real gap while
still generalizing to eruptions the adapter never saw? (#52)

The zero-shot gap is measured -- frozen MAGEMin-trained heads score joint NLL
10.406 over 640 least-altered volcanic rocks, ranking transferring but absolute
scale not. This asks what closing it costs in labelled real rocks. The
deliverable is the curve's shape, not a single recovered-nats number: a point
estimate on a ~160-rock slice carries roughly +/- 2 nats, and the shape is what
says whether harvesting more eruptions is worth the transcription.

## Method

Frozen encoder, frozen heads. A small `RealRockAdapter` corrects the coarsened
seven-group outputs and is fit by minimizing `real_rock_nll` -- the same loss
the ledger reports -- so a gain reads directly as nats recovered. Three nested
tiers: `diagonal` (23 params: per-group presence scale + offset, per-group
magnitude log-offset, SiO2 slopes on q and amp presence) in `linear` (121, 7x7
coupling) in `mlp` (1095, with a linear skip). Amphibole's magnitude correction
is pinned at zero at every tier.

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
| diagonal | -0.69 | -0.93 | -0.95 | -1.32 | -1.33 |
| linear | -0.81 | -1.48 | -1.69 | -2.22 | **-2.33** |
| mlp | **-1.11** | -1.56 | -1.65 | -2.01 | -2.08 |

(median test Dirichlet NLL; lower is better)

**Adaptation is cheap and the curve is steeply diminishing.** About 25 labelled
rocks -- eight eruptions -- recovers 3.4-3.8 of the 5.0 nats that the full 368
rocks recover. Fourteen times more data buys the remaining third.

**Coupling pays; nonlinearity does not.** `linear` wins at every budget from 50
rocks up and reaches 5.03 nats recovered; the MLP's extra capacity never repays
itself. Both beat the per-group `diagonal` baseline, so the correction the
benchmark needs is genuinely group-coupled rather than seven independent
constants.

At 25 rocks the MLP has the better median, but **that lead is not
established**. The tiers see identical unit draws, so the honest read is the
paired one: the MLP wins 6 of 10 draws by a mean 0.22 nats, against a
draw-to-draw spread of 2.13 nats, and the mean is carried by one unlucky draw
where every tier does badly and the MLP's heavier shrinkage limits the damage
(+0.41 against linear's +1.73). That is a robustness story about
regularization, not evidence that capacity helps at small n. From 200 rocks up
the sign is unambiguous: linear beats the MLP on 10 of 10 draws.

**Much of the gap is over-confidence, not wrong means.** The fitted diagonal
drops alpha0 on the scored support from 46.2 to 25.1 -- it widens the Dirichlet
by roughly half. At 25 rocks the diagonal tier *worsens* share MAE (0.107 ->
0.114) while still improving the Dirichlet term: at that budget it is buying
calibration, not accuracy. Share MAE only improves once there is enough data to
move the means too (0.0876 at full data, linear).

**At 25 rocks, which units you drew dominates**, exactly as #52 predicted: the
10-90 band spans -1.24 to +0.24 for the diagonal tier, crossing zero. The band
collapses to a point at the largest budget by construction -- 368 rocks is the
whole train pool, so every draw is the same.

## Decision

Adaptation works and is cheap, so the sim-to-real gap is not a barrier: a
handful of well-characterized eruptions is enough to recalibrate frozen heads.
Carry `linear` as the default tier. HPO comes next, on train/val.

Two things this does *not* settle. The benchmark's noise floor is unknown --
most of the set is visual estimates at +/- 5-10 vol% and only ~11 rocks are
point-counted -- so the spread band is not a noise floor. And amphibole
detection is untouched by design: a monotone per-group map cannot move AUROC,
and that hands off to corpus regen and #49.

The same curve re-run on post-regen heads is the regen's transfer evidence,
since under the segmented-ledger convention this benchmark is the designated
cross-instrument judge (#45).

## Gotcha found here

The coupled tiers read `log(alpha)` as an input feature, and coarsening leaves
exact zeros wherever a group has no member phase on the support. Clamping
those to `1e-12` before the log hands the mixing matrix a -27 spike on exactly
the groups carrying no information, and `linear` / `mlp` diverge on it -- val
NLL reaching 1e20, with an erratic non-monotone weight-decay curve that reads
like "the coupled tiers do not help." They do; an off-support group's feature
is the neutral 0. Worth knowing before adding any further input to this module.

## Files

- `run_curve.py` -- the sweep, the figures, `results/<tag>/curve.json`
- `results/joint_heads_supportmag_seed0/learning_curve.png` -- the deliverable
- `results/joint_heads_supportmag_seed0/curve_by_units.png` -- priced in eruptions

Backing: journal 2026-07-25, `results/real_rock_ledger.yaml` (`eruption-v2`),
`explorations/real_rock_mapping/` (the truth mapping this rests on).
