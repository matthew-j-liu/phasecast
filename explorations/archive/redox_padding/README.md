# redox_padding

## Question

The 23rd input token carries the Fe3+/total-Fe ratio, populated where a lab
measured both iron flavours and padded otherwise. The docs describe the padded
state as marginalizing implicitly over an unknown redox ratio. Is that what it
actually does?

The question matters because padded is the deployment regime and not by
choice: the eruption-v2 benchmark carries only `feo_tot` / `fe2o3_tot`, which
are total-iron reportings rather than a speciation pair, so 0 of its 640 rocks
can populate the slot. The production heads also train at 22 tokens
(`configs/heads/joint_heads.yaml` carries no `data_kwargs`), so they have
never seen the slot populated.

## Method

Three readouts, none of which touch a label or the sealed test slice.

**Geometry** (`run_geometry.py`). Every corpus bulk whose source reported a
real FeO + Fe2O3 pair (`fe_provenance == measured`, 3070 bulks across both
tiers) is encoded with the slot padded and once per ratio on a 0.05-0.95 grid,
tracing the curve the token sweeps in latent space. The padded latent is then
located against that curve. `derived_partial` is excluded deliberately: only
81 of its 3095 bulks derive a real ratio, the rest take the same `qap_name`
lookup `assumed_class` gets (journal 2026-08-06). Because padding also drops a
token from the pool, part of any off-curve displacement is arithmetic rather
than semantics, so the statistics are reported twice -- raw, and after
stripping the displacement component shared by every bulk.

**Missingness** (`run_missingness.py`). If the rest of the composition
predicts whether the pair was reported, a blank slot is partly a statement
about the analysis rather than a neutral unknown. Two XGBoost arms on the SSL
pretrain corpus (787,721 rows, 24.1% populated), split on its pinned test
roster: from the raw inputs (22 element values plus presence flags, redox
excluded) as the ceiling, and from the frozen pooled latent with the slot
padded, which is what the encoder actually hands downstream.

**Recoverability**. `scripts/eval/impute_elements.py` on
`imputation_head_allrock_redox_seed0` already scores the redox slot, giving
how well the rest of the chemistry determines the ratio at all. That readout
masks the slot (`src = MASK`), which is a different input state from padding;
it bounds what any conditional distribution over redox could be worth, not
what the padded state does.

## Result

**Padding is exactly a no-op.** A padded 23-token input reproduces the
22-token latent to 0.0 absolute difference on the real weights, confirmed
against the production encoder at the top of every geometry run and pinned by
`TestRedoxTokenWidth::test_padded_redox_slot_reproduces_the_22_token_latent`.
Nothing in the token path is position-aware, `src == 0` masks the slot out of
attention, and the masked-mean pool divides by the non-padding count. The
production config's "22 tokens and padded-23 are one regime" is correct.

**The token matters, moderately.** Sweeping the ratio across its plausible
range moves a bulk's latent by 26% of the median distance between two
different rocks (p10-p90 0.20-0.33). Not negligible, not dominant. The path is
not straight: median arc length runs about five times the end-to-end span, so
the encoder's response to the ratio is jagged rather than smooth, which
already makes "average over plausible ratios" and "assume a typical ratio"
different objects.

**The padded state is not a marginalization.** It sits off the populated
family entirely -- 0.52 of the sweep's own span from the nearest point on the
curve (0.31 after removing the shared offset). Where it does land resembles a
low fixed ratio: nearest grid value has median 0.15 and mode 0.15 raw, median
0.25 after the offset correction, against a measured-tier true median of 0.33.
And it barely tracks the rock: Pearson correlation between the nearest ratio
and the bulk's own measured ratio is 0.23 (Spearman 0.18), essentially
unchanged by the offset correction. A state that marginalized over p(ratio |
chemistry) would track the rock; a state that assumed one global default would
not. This is much closer to the second. The displacement direction is
substantially shared across bulks (median pairwise cosine 0.62), consistent
with a common "one fewer token" component plus a rock-specific remainder.

**There is little conditional to marginalize over anyway.** The imputation
head recovers the redox slot at log-space R2 0.307, against 0.766-0.944 for
every one of the 22 elements. Bulk chemistry does not determine oxidation
state, which is the petrologically expected answer -- the same composition
crystallizes under different fO2 -- and it means any honest conditional over
redox would be wide.

**A blank slot is informative about the analysis.** Whether the pair was
reported is predictable from the rest of the composition at AUROC 0.907 from
the raw inputs, and the frozen padded latent carries essentially all of it at
0.898. The top features are trace presence flags (Th, Yb, Nd), not
compositions: measured-pair analyses are overwhelmingly older, and older
analyses report different trace panels. Read PR-AUC (0.849 / 0.832) against
the test slice's 0.374 base rate rather than the corpus 0.241 -- the pinned
roster over-represents measured-pair rows by construction.

## Decision

The padded state is a distinct learned "no speciation reported" state, not an
implicit average over redox. It behaves like a low default, it does not adapt
to the rock, and the encoder can read it as a signal about where the analysis
came from. Describe it that way.

Nothing here argues against the production configuration: training and
deployment share one regime, that regime is forced by what real rocks report,
and the equivalence that makes it coherent is now pinned by a test. What it
removes is the mechanism story that was attached to it.

Verdict in `docs/journal.md`, 2026-08-06.
