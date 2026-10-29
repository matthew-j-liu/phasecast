# sediment_transfer

## Question

Does the frozen stack transfer to sedimentary rocks, measured a different way,
and is the sim-to-real gap a representation problem or a supervision one?

Every real-rock verdict rests on eruption-v2: volcanic rocks whose modal
mineralogy was mostly estimated by eye. This points the same instrument at 605
sedimentary rocks quantified by XRD -- an independent rock class *and* an
independent measurement modality -- and asks three things: what the model does
there, whether an encoder that saw sedimentary chemistry at pretraining does
better, and how much of any failure a correction layer can recover.

## Data

`dp_crab_Andrews.pkl` from the old CrabNet-Mineral repo
(`~/git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn/datasets/`), reached from
issue #24. 605 rocks; nine elements as weight fractions; 22 minerals as XRD
weight fractions. **Provenance is unrecovered** -- samples are anonymised
`Rock_N` with no locality, formation, freshness grading or identified source
publication -- so nothing here is a reportable benchmark number. #63
workstream 2 owns finding the source.

Compatibility, measured before anything was run:

- **7 of 22 input tokens populate.** No Ti, no Cr, no traces. Cl and S are
  reported but have no token, and they are what predict the halite and pyrite
  the truth contains.
- **Truth resolves 5 of the 7 groups.** Olivine and biotite appear in zero
  rocks, so their presence branch is scoreable (a call is a false positive) and
  their magnitude branch is undefined.
- **43% of measured modal mass has no slot** in the 21-phase vocabulary
  (carbonates, evaporite, sulfide, clays; p5 0.151, median 0.426, p95 0.641).
  Chlorite, dropped at the <0.1% presence floor on igneous statistics, is in
  100% of Andrews rocks.
- **Inputs sit on a different scale.** Implied oxide total median 59.3 wt%
  against ~98 for gard and ~99.7 for eruption-v2, because carbonate, water and
  chloride mass is untokenised. That is a 40% offset, not the 2% #56 was
  written around, and this exploration does not separate it from rock class.

## Method

`load_andrews.py` reshapes the pickle into the shipped `RealRockBenchmark`, so
every production eval and plot function runs unmodified. The oxide conversion
inverts `OXIDE_TO_CATION_FACTOR` exactly, so each token equals the source
number to float32 precision; truth goes through the shipped `_truth_arrays`
with density 1.0 (XRD reports mass, so eruption-v2's volume-to-mass step and
its mid-series solid-solution assumption are simply absent here).

| script | what it measures |
|---|---|
| `run_card.py` | the five canonical headline figures + `manifest.json` |
| `run_encoder_contrast.py` | igneous-only vs all-rock encoder, paired per rock |
| `run_recalibrate.py` | the recalibration layer fit on train, scored on the untouched test slice, all three tiers |
| `run_bootstrap_width.py` | what a per-rock bootstrap costs, priced on eruption-v2 |
| `run_dubray_reference.py` | the eruption-v2 figures the above are read against, same checkpoint |

All on `joint_heads_allrock_tuned_seed0`. The shipped eruption-v2 figures were
re-run afterwards and came out byte-identical, so nothing shared moved.

## Result

**Zero-shot, no transfer.** 605 rocks, +13.19 nats worse than a forecast that
ignores the rock. Presence Brier 0.34 against a 0.027 null; 80% intervals cover
50%. Quartz is called in 23% of rocks that all contain it; olivine in 46% that
contain none.

The raw score inverts without the null and this is the trap: joint NLL is
*lower* on Andrews than on eruption-v2 (7.893 vs 11.638), because sediments are
an easier forecast and climatology scores -5.301 there against +1.527. Never
quote the raw number alone.

**The encoder is not what is missing.** Paired on the same 605 rocks, all-rock
minus igneous: presence **-1.156** [-1.470, -0.842], composition **+3.583**
[+3.134, +4.021], joint **+2.427** [+1.813, +3.047]. Pretraining on sedimentary
chemistry made the model better at knowing which minerals are present and worse
at sizing them. Representation breadth improves recognition and cannot touch
quantification. (An arm contrast, not an encoder-only ablation: each arm
carries its own tuned head dials.)

**A correction layer recovers almost all of it.** On the never-spent test slice
(152 rocks), the tailed tier recovers **13.595 nats** [-14.706, -12.483] and
ends up **1.12 nats better** than climatology, not worse. Against eruption-v2's
val slice, where the same layer sits 1.20 nats ahead of a climatology of 2.175,
the margins are near-identical at comparable fit budgets (363 vs 368 rocks).
So a few hundred real labels put sedimentary prediction on the same footing as
volcanic -- which is the most direct evidence yet that the missing ingredient
is target-domain head supervision, not representation.

**The tailed tier is not a du Bray artifact.** (Measured while the tier
existed; it was deleted 2026-09-03 after its du Bray margin was traced to the
amphibole pin, and this read was not repeated with the pin gone.) It won on
Andrews too, by 1.08 nats over matrix. If its heavy tail were mostly modelling visual-estimate
sloppiness it should not help on XRD Rietveld data. This agrees with the du
Bray-side result (`explorations/real_mineralogy`, journal 2026-08-06): the
contamination weight correlates with ruler disagreement at only Spearman +0.22.

**Per-group bias is benchmark-specific.** Median predicted/measured share,
eruption-v2 then Andrews, same checkpoint: fsp x1.14 / x0.39, px x0.76 / x2.42,
amp x0.92 / x8.77, opq x0.34 / x3.39, q **x2.01 / x1.01**. A correction layer
fit on one rock class does not port to another. The quartz entry supports #57
from a test that could have refuted it: the over-call vanishes where the
proposed mechanism cannot operate, since a sandstone has no glass and its
quartz is detrital. (Read with the du Bray ruler caveat: point counts move
measured quartz and opaques, so part of that column is the ruler.)

**Error bars.** Andrews has no units to cluster on, so its intervals are the
per-rock kind, which on eruption-v2 runs **2.08x narrower** than the
unit-clustered one. That factor is printed on the two figures that draw bars.

## Decision

The sediment failure becomes structure in the paper rather than an aside:
the canonical encoder already saw sedimentary chemistry, so the representation
is not the limitation, and sedimentary prediction still fails zero-shot and
recovers only when given real sedimentary labels. That localises the gap to
head supervision and motivates simulating sediments rather than harvesting
them.

Standing caps, not fixable by more labels: 43% of truth mass is outside the
vocabulary, so every number here is a within-vocabulary result. Any real
sedimentary extension needs a vocabulary extension *and* a different generator
-- clays, carbonates and evaporites are low-temperature diagenetic products
outside MAGEMin's regime.

Open: the encoder-is-not-the-limitation claim is inferred, not controlled;
#53 is the experiment that would underwrite it. And the 40% input-scale offset
is confounded with rock class throughout (#56).

Canonical verdict in `docs/journal.md`, 2026-08-06 entries.
