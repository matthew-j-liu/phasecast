# real_rock_mapping

**Question.** The zero-shot real-rock eval reports 10.404 nats and a per-group
residual shape that the thin sim-to-real recalibrator will be designed against. Both
pass through a truth mapping nobody measured: mineral densities are mid-series
stand-ins for solid solutions, the 0.5 vol% trace bound is read off a reporting
grid, the freshness cut is a leading-digit parse of a 1-4 grade, and the support
forcing invents a rule for splitting "pyroxene is present" into cpx/opx. Does
any defensible reading of those choices change a decision?

Three decisions are at stake, and only the third is about the nats:

| | Decision | Robust iff |
|---|---|---|
| D1 | recalibrator = per-group constants + SiO2 slopes on q and amp presence | offset signs and rough magnitudes survive |
| D2 | amphibole is a discrimination failure, not a calibration miss | AUROC stays ~0.58 |
| D3 | 10.404 is quotable as the zero-shot gap | the spread is small next to what the recalibrator must recover |

**Method.** `run_sensitivity.py` re-scores one frozen checkpoint under ~30
mapping variants across six knobs, going through the shipped
`cnmineral.eval.real_rock` loader and loss every time -- an audit of a
reimplementation audits nothing. Densities get eight named corners plus a
200-draw Monte Carlo over petrologically plausible per-mineral ranges; the trace
bound gets four values plus the structural alternative of imputing traces at
their bound instead of left-censoring them; the alteration rule gets five grade
cutoffs plus a reading where `+` means "worse than the digit"; the support
forcing gets three member thresholds plus all-members and single-member; and two
knobs the original plan did not name get audited too (13 uncertain-identification
mode rows, and the reporting-basis conversion that only touches the
point-counted rocks). Nothing is fit, so the recalibrator's held-out estimate is
untouched.

Reading rules the audit enforces on itself: variants are compared like-for-like
only when the row count *and* the mean composition size match, because the
Dirichlet term is a density over however many groups a rock spans. A sign flip
counts only when both sides clear a materiality floor (0.05 of presence
probability, 0.10 log units of magnitude) on a subset of at least 100 rocks --
otherwise amphibole's magnitude offset of -0.002 "flips" under every
perturbation while changing nothing a recalibrator would do.

## Result

**Two invariances are exact, not approximate.** The Bernoulli term is
bit-identical (5.708414604) across every density variant, and the headline is
bit-identical across trace bounds from 0.1 to 1.0 vol%. Density enters only the
mass conversion and the bound sizes only a censored ceiling, so presence-side
conclusions -- every AUROC, every presence offset, the amphibole verdict --
cannot move with either. Both are asserted as tests, then demonstrated.

**D1 holds.** No presence-offset sign flips at all. One magnitude flip: px,
under `all_members` only, which is mechanical rather than a mapping question
(forcing both cpx and opx on splits the group's concentration). The Monte Carlo
pins every recalibrator constant tightly -- opq +0.942..+1.027 (shipped +0.991), bi
+0.445..+0.568, q -0.541..-0.519, fsp -0.138..-0.114. Amphibole's magnitude band
straddles zero (-0.033..+0.045), so the recalibrator should learn ~0 there and the
sign is not recoverable from this benchmark at any density.

**D2 holds outright.** Amphibole AUROC spans 0.497-0.592 across every variant --
never better than trivial under any mapping, and below chance under one. The
discrimination failure is not an artifact of the truth mapping.

**D3 holds for measurement assumptions, and fails for the invented one.**
Plausible density uncertainty is worth ±0.3 nats (Monte Carlo 5-95%:
10.08-10.70); the alteration cutoff ±0.3 (10.15-10.77); uncertain
identifications and xenocrysts ±0.02. But the **support forcing rule spans
9.86-13.38** -- a 3.5-nat range, an order of magnitude more than any physical
assumption, entirely from a rule the mapping made up. `resolve_at_0.3` costs 3
nats; `resolve_at_0.7` gains 0.5.

**The vol%-to-mass conversion does real work, in the conservative direction.** Skipping it
(`no_conversion`) improves the headline by 2.4 nats -- more than the entire
plausible density band -- because it is what exposes the opq under-sizing: the
oxide magnitude offset falls from +0.991 to +0.583 when truth is left in volume
units. MAGEMin modes are mass fractions, so converting is the correct basis and
not converting would have flattered the model on a basis it does not predict.
The choice of *which* density values matters ~8x less than the choice to convert
at all.

**Imputing traces instead of censoring them costs 2.2 nats (10.40 -> 12.65)**,
but the composition size moves 3.61 -> 4.23, so that number is not a
like-for-like read; it is reported with the size stamped for exactly that
reason. Left-censoring stands.

**The groundmass drop is what makes the two sources commensurable.** The 13
groundmass rows are exactly the 13 point-counted samples, so this knob only ever
touches the precision tier. Dropping those rocks moves the headline by +0.04
nats (10.404 -> 10.444); on their own they score 6.26 at composition size 4.50,
which is a different measurement regime rather than a better one. The literal
"keep groundmass as an unscored denominator component" is not scoreable: the
label is crystals-renormalized so there is no `liq` in the 21-phase vocabulary
to pair it with, and a Dirichlet needs its components to renormalize.

## Decision

The mapping is robust where it encodes a measurement and fragile where it
encodes an invention. Densities, the trace bound, the freshness cut, uncertain
identifications and the reporting basis all leave the recalibrator's design intact,
so **build 2 proceeds on the shipped mapping and on the residual shape as
measured**. The two things to carry forward: the support-forcing rule deserves
to be treated as a modelling choice with a 3.5-nat lever rather than a mapping
detail, and amphibole's magnitude constant should be pinned at zero rather than
fit, since no density makes its sign recoverable.

Canonical verdict in [docs/journal.md](../../docs/journal.md).

    conda run -n alignn_fix python explorations/real_rock_mapping/run_sensitivity.py \
        --checkpoint checkpoints/joint_heads_supportmag_seed0/heads_best.pt
