# Decision 004: Use PetThermoTools for fractional-crystallization experiments and (potential) production label generation

**Date:** 2026-05-13
**Status:** Resolved → Phase B, **amended 2026-06-30 (production scoped to volcanic; equilibrium, not FC)**; the volcanic generation record folded in 2026-09-03

## Amendment 2026-06-30 -- v1 production is volcanic + equilibrium; FC deferred to the plutonic branch

The v1 corpus is scoped to the **volcanic + has-iron** subset of gard2019.
For that scope the production generator is **MAGEMin equilibrium on a
cooling ladder, read at melt-fraction rungs**, not fractional
crystallization. This *scopes* the Phase-B decision below; it does not
overturn the eq-vs-frac measurement.

The eq-vs-frac experiment (Resolution below) measured how closely near-solidus
equilibrium *reproduces FC endpoint labels* and found a large divergence
(median L1 0.35). That result stands. What changes is the target physics, not
the measurement. A volcanic rock crystallizes closed-system, in place: the melt
keeps its components (water included) and the assemblage that forms **is** the
equilibrium assemblage at those conditions. FC models the open-system case --
crystals settle out and the residual melt evolves -- which is the
plutonic/cumulate story. So for volcanic targets we are not approximating FC
with equilibrium; equilibrium is the correct generator, and the eq-vs-FC
divergence is the real difference between two petrogenetic regimes, not error.
No eq-vs-FC A/B was run for the volcanic scope; the conceptual argument was
accepted.

Consequences:
- v1 production generator = `magemin/run_magemin.jl` in a Julia subprocess,
  driven by `scripts/build/generate_mineralogy_corpus.py`; `(P, H2O)` are
  sampled and T is derived by cooling. `NearSolidusTPStrategy` stays
  unimplemented.
- FC (and PetThermoTools as its driver -- the tooling choice below is
  unchanged) moves to the deferred **plutonic/cumulate branch**, where it
  models FC of parent magmas (issue #22). ADR 005's FC production recipe
  applies there, not to the volcanic v1 corpus.

### Generation record (landed 2026-07-27; the planning spec is retired)

The spec that planned this corpus (`docs/corpus_regen_spec.md`, tracker #9)
was deleted 2026-09-03 once every step had landed. This section is the
generation-parameter record it left behind. Evidence for each choice lives
in the archived explorations named at the end.

**Which corpus is production.** Two generations exist.

- First generation (2026-07-14/17): the 10k stratified train bulks x 48
  draws from the broad proposal, plus 500 valtest bulks x 128 draws
  (`data/corpus/train`, `data/corpus/valtest`).
- Traced regen (2026-07-23/27, #45): 35k train bulks (the 10k plus a
  rare-class-quota expansion, `data/corpus/train_bulks_expansion.csv`,
  alpha=0.5 temperature smoothing emitted nested-block) x 24 draws from the
  trace-conditional proposal, plus 2000 valtest bulks x 128 draws
  (`data/corpus/train_traced`, `data/corpus/valtest_traced`; roster 887 val
  / 1112 test, grown from the first 500 by extension so earlier assignments
  never moved).

Heads trained on the traced corpus lost to the first-generation corpus on
the real-rock benchmark after identical recalibration (journal 2026-08-03),
so **production heads train on the first generation** (all 48 draws, rungs
50 + 30, early stopping on the first-generation valtest roster;
`configs/heads/joint_heads.yaml`). The traced tier is a measured
non-production arm; its wins are quartz presence and assemblage ranking.

**Subset + QC.** Volcanic core = the 25 aphanitic QAPF names in
`cnmineral.data.volcanic_subset.VOLCANIC_QAP` (~335k bulks); plutonic-named
`rock_origin` rows stay out; the foidite family is excluded (`ig` does not
converge). has-iron = total Fe sufficient, since the locked Fe path imposes
the class-conditional split regardless; rows with measured FeO + Fe2O3 carry
a provenance tier so a stricter cut stays available at training time. QC
flags (oxide-sum window, supersilicic, ultramafic) are tags, not drops, but
flagged bulks are weighted out of the stratified draw. Bulk tables:
`scripts/build/build_corpus_bulk_tables.py`; health gate:
`scripts/build/report_corpus_health.py`.

**Conditions (the bounded sampler).** `(P, H2O)` are sampled per draw and T
is derived by cooling to the endpoint, never a sampled axis. The broad
proposal is `q = Uniform(P [0.5, 12] kbar x H2O [0, min(8, S(P))] wt%)`,
Latin-Hypercube, implemented in `cnmineral.data.condition_proposal`. The
traced regen (#48) drew instead from the per-bulk trace-conditional proposal
(`condition_proposal.bulk_proposal`): immobile-element discriminants route
each bulk to a posterior-weighted mixture of (maficity, tectonic-setting)
literature boxes, with a per-class fallback and 8% mass always on the broad
proposal. Every draw stores its exact proposal density `q`, so either
proposal reweights the same way at inference.

**Water bound `S(P)`.** The Newman & Lowenstern 2002 (VolatileCalc)
rhyolite-curve H2O solubility, the more water-generous of their two curves
since a composition-independent bound must cover every composition, capped at
8 wt% (the melt-inclusion supply ceiling; Wallace 2005; Plank et al. 2013).
The cap binds above ~4-5 kbar, so the curve is only evaluated inside its
calibration range. Accepted gap: alkaline melts dissolve somewhat more at low
P, so very wet low-P alkaline draws are under-covered. Oversaturated draws
exsolve excess H2O as `fl` inside MAGEMin and are tagged. The Wallace/Plank
per-class ranges of ADR 003 belong to the inference prior, never the sampler.

**Endpoint and rungs.** Cool each draw until residual melt < 30 wt%, no
near-solidus leg. The assemblage is read at four rungs, 60/50/40/30%
residual melt, re-picked from the stored ladder at no extra MAGEMin cost.
Training rungs **50 + 30** were picked 2026-07-14 by a label-level
pre-screen against the real benchmark (per-silica-band presence match first,
rank-level crystals-renormalized proportions second): 50 is the best
aggregate match, 30 the only rung where intermediate biotite exists to be
learned. A composition-conditional endpoint (deeper stop for felsic bulks)
was rejected: quartz and K-feldspar saturate early in felsic bulks, real
felsic lavas are the crystal-poorest, and near-solidus invents felsic
phenocrysts in mafic bulks. The amphibole/orthopyroxene multimodality holds
across the whole 70-1% melt range.

**Label.** Crystals-renormalized solid assemblage: `liq` and `fl` are dropped
and the solids renormalized to sum to 1, matching the benchmark's granularity
(itemized phenocrysts plus one lumped groundmass bucket). There is no
melt-fraction head, since at a fixed rung the melt fraction is a constant and
real groundmass fraction is set by ascent, not chemistry; the liq/fl
composition and fraction are stored per row so the mass-balance closure can
teacher-force the label melt fraction. Presence targets are built at mode >
0.1 wt%; raw modes are stored, so a different cutoff is a label rebuild, not
a regen. The pre-screen ran at 0.1 and 1 wt% and the rung choice was
cutoff-insensitive.

**Fe / fO2.** Class-conditional Middlemost Fe3+/total-Fe split
(`cnmineral.data.fe_redox`), fO2 fixed, not swept. Per-draw `in_feo` /
`in_fe2o3` (ratio-invariant under the H2O renormalization) plus the
`fe_provenance` stamp are the complete source for the optional redox input
token (#46); no extra column is stamped.

**CompHead target.** Per-phase oxide wt% (`Comp_wt`) for all phases; the
endmember vector (`emFrac_wt`) is stored as a diagnostic only, because it
leaves the positive simplex on real bulks. See CLAUDE.md settled decisions.

**Storage shape.** Three tables per chunk in `cnmineral.data.magemin_corpus`:
`draws` (one row per bulk x draw: passthrough bulk columns and traces, the
sampled `(P, H2O)`, `q`, Fe inputs and provenance), `ladder` (modes only per
cooling step, so rungs can be re-picked), `rungs` (full labels at the four
rungs: modes, `Comp_wt`, `emFrac_wt`, liq/fl composition and fraction,
`G_sys`, `T_end`). Roughly 5x less disk than full-label ladders.

**Orchestration.** Chunked Python orchestrator wrapping `run_magemin.jl`
unchanged; ~100-bulk chunks, each flushed atomically to its own parquet and
appended to a completion manifest, workers recycled per chunk. Resume is
rerunning the same command with at most one in-flight chunk redone. Solver
crashes (deterministic MAGEMin segfaults on rare inputs) are bisected at draw
boundaries and the isolated draw dropped into ordinary attrition accounting;
NaN-bearing records are serialized as failed rows; a complete output file
overrides a nonzero exit.

**Encoder / supervised split.** The encoder is pretrained on all of gard2019
(all-rock, relaxed Fe); the heads are supervised on the volcanic + has-iron
subset, so OOD and abstention still work on a user's non-volcanic input.

**Output stance (design, not implemented).** Because every draw stores its
proposal density, an inference-time weight `p(c) / q(c)` can cancel the
sampling proposal and substitute an evidence-based prior: ADR 003's per-class
H2O ranges, trace-ratio depth proxies (Sr/Y, La/Yb; soft nudges, large
single-sample sigma), or user-supplied constraints (#21). Scenario weights
are labelled prior-weighted plausibility, never nature's rock-specific
probability. Latent-neighbour analogs against MagmaDB were assessed and
shelved (too sparse: 78 distinct bulks, arc basalt-andesite only).

**Evidence.** `explorations/archive/volcanic_proxy_definition/` (endpoint
re-pick, felsic saturation), `explorations/archive/equilibrium_multimodality/`
(multimodality verdict, water A/B: the amp/opx switch holds at every rung,
biotite did not move), `explorations/archive/condition_sampling/` (LHS,
effective sample size), `explorations/archive/trace_conditional_proposal/`
(the #48 proposal), `explorations/archive/rare_class_quota/` (the expansion
table), `explorations/archive/magmadb_analogs/`.

**Deferred.** The plutonic/cumulate branch, FC of parent magmas (#22).
Itemized groundmass, blocked on real itemized data and a kinetic generator;
a future branch, not a toggle on this corpus.

## Resolution

Experiment ran 2026-05-13. 547/550 bulks across 11 classes converged on FC;
526/550 on eq. Overall median L1 = 0.350 (mean 0.482) — far above the 0.10
"adopt eq" threshold. Dacite at L1 = 1.14, 9 of 11 classes above 0.25.
Decision: adopt fractional crystallization for production label generation.
Full results, plots, and discussion: `docs/JOURNAL.md` 2026-05-13 entry
"PetThermoTools integration + eq-vs-frac experiment results".

`NearSolidusTPStrategy` (the Phase A code path) is NOT implemented and should
not be added.

## Decision

Adopt **PetThermoTools** (https://petthermotools.readthedocs.io) as the
fractional-crystallization driver for the equilibrium-vs-fractional
comparison experiment, and — if the experiment indicates fractional
crystallization is needed for v1.5/v2 production label generation — for that
work too.

PetThermoTools is a Python wrapper around MAGEMin_C.jl that exposes
fractional-crystallization simulation via `Frac_solid=True` plus auxiliary
features (solidus/liquidus bracketing, configurable cooling step sizes,
proper handling of solid-removal accumulation, plotting utilities).

## Why this decision

The immediate need is the experiment described in journal entry
`docs/JOURNAL.md` ("PetThermoTools experiment design: equilibrium vs
fractional crystallization"). The experiment requires reliable, defensible
fractional-crystallization simulation across many bulks for comparison
against single-snapshot equilibrium labels.

Two candidates were considered:

### Alternative 1: Custom Julia loop

Write `scripts/julia/fractional_crystallization.jl` (~50 lines) that wraps
`MAGEMin_C.point_wise_minimization` in a cooling loop, removes some fraction
of solids per step, accumulates the removed mass, and returns final modal
composition.

Pros:
- No new dependency
- Transparent — we control exactly what's happening
- Integrates with our existing chunked-orchestration infrastructure
  (`scripts/orchestrate_magemin.py`)
- Faster to implement initially

Cons:
- Reinvents the wheel
- Misses subtle correctness details that PetThermoTools handles (liquidus
  bracketing, adaptive step sizing near solidus, terminal-solid edge cases)
- Doesn't reuse if we later need Phase B at production scale
- Less defensible in the paper — reviewers may ask why we wrote our own
  rather than using the community tool

### Alternative 2 (chosen): PetThermoTools

Pros:
- Community-vetted, peer-reviewed implementation. Wadsworth et al. 2022,
  Petrology, doi:10.30909/vol.05.02.215231
- Built-in liquidus/solidus bracketing — we don't have to implement the
  numerical bisection ourselves
- Sensible defaults for cooling step size, solid removal fraction, output
  format. PetThermoTools' authors are petrologists; their defaults reflect
  what the field considers correct.
- Reusable if the experiment indicates Phase B is needed for production:
  same tool, scaled up
- Defensible in the paper: "labels generated using PetThermoTools v[X.Y.Z]
  with default fractional crystallization parameters"

Cons:
- New Python dependency (PetThermoTools + its own MAGEMin_C.jl integration)
- Possible version-pin issues with our existing MAGEMin_C.jl + Project.toml
  setup; need to verify compatibility
- Initial integration cost is somewhat higher than the custom Julia loop
  (~3-4 hours vs ~1-2 hours)
- More opaque — we don't directly see what's happening inside the
  fractional-crystallization step

The defensibility, reusability, and community-vetting considerations
outweigh the implementation-cost difference. The experiment we're running
is meant to *decide* whether to adopt fractional crystallization for
production — if we used a custom loop, we'd have two different
implementations to reconcile when scaling up.

## Open implementation questions (for the integration session)

These are not blockers but should be resolved during the integration plan:

- **Compatibility with our existing MAGEMin_C.jl pin.** PetThermoTools
  may pin a different MAGEMin_C version. Need to test that both can
  coexist or accept upgrading our pin. (Our existing Project.toml /
  Manifest.toml may need to be updated.)
- **Output format adaptation.** PetThermoTools' output structure may
  differ from our `phases_json`/`ss_json` schema in
  `scripts/julia/run_magemin.jl`. May need a small bridge module to map
  PetThermoTools output to our existing parser format.
- **Default parameters to use.** Cooling step size, % solids removed per
  step, liquidus offset — use PetThermoTools defaults unless the experiment
  shows them to be inappropriate.
- **Where to plug in.** For the experiment: a new `scripts/experiments/eq_vs_frac/`
  subdirectory. For production (if Phase B selected): would replace
  `scripts/julia/run_magemin.jl` or extend it with a `--mode frac` flag.

## Revisit when

- The equilibrium-vs-fractional experiment completes and we know whether
  fractional crystallization is needed for production. Three outcomes:
  - Median L1 distance < 0.10 → near-solidus equilibrium is fine; this
    ADR becomes "we evaluated PetThermoTools for v1.5+ work but production
    uses subsolidus equilibrium."
  - Median L1 distance > 0.25 for some classes → adopt PetThermoTools
    for production label generation.
  - In-between → hybrid; PetThermoTools for felsic classes only.
- Or: PetThermoTools' development stalls / breaks compatibility with newer
  MAGEMin_C versions, at which point we'd reassess.

## Cross-reference

- Journal entry: `docs/JOURNAL.md` 2026-05-13 "PetThermoTools experiment
  design: equilibrium vs fractional crystallization"
- Implementation will live in: `scripts/experiments/eq_vs_frac/` (TBC during
  integration plan)
- Existing MAGEMin integration: `scripts/julia/run_magemin.jl`,
  `scripts/julia/Project.toml`
- Existing handoff brief (PetThermoTools not mentioned there;
  this is new infrastructure): `docs/magemin_handoff.md`
