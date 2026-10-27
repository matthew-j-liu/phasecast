# Decisions

Binding architectural choices, one numbered record each (ADR = architecture
decision record): the decision, the alternatives considered, and when to
revisit. A new decision is appended with the next number; an existing one
changes only by a dated amendment inside its own section. Status lines say
which records still bind and which are historical. The historical records
name files of the pre-v3 codebase; one missing from `main` is on tag
`v2.0-traces-ssl` (`git show v2.0-traces-ssl:<path>`).

| ADR | Decision | Date |
|---|---|---|
| [001](#adr-001) | v1 paper scope | 2026-05-12 |
| [002](#adr-002) | Paths explicitly rejected | 2026-05-12 |
| [003](#adr-003) | Tiered H2O handling strategy | 2026-05-12 |
| [004](#adr-004) | Use PetThermoTools for fractional-crystallization experiments and (potential) production label generation | 2026-05-13 |
| [005](#adr-005) | Production label generation via PetThermoTools fractional crystallization | 2026-05-15 |
| [006](#adr-006) | v1 model architecture — shared encoder + Dirichlet ModeHead + GMM CondHead + joint Optuna HPO | 2026-05-19 |
| [007](#adr-007) | SSL pretrain objective is per-element standardized log-space masked MSE | 2026-06-18 |
| [008](#adr-008) | drop h2o_tot from model inputs (H2O target leak) | 2026-05-22 |

<a id="adr-001"></a>

## ADR 001: v1 paper scope

**Date:** 2026-05-12
**Status:** Historical -- the scope held (Position C corpus, unimodal CrabNet-style model, mineralML classifiers as post-processor), with drift recorded elsewhere: the contribution is now framed around the SSL encoder + multimodal posterior (`docs/encoder.md`, `docs/heads.md`), and the deep ensemble is still unbuilt.

### Decision

v1 paper = Position C training corpus + unimodal CrabNet with 4 heads (mode, comp, gsys-as-regularizer, class) + Dirichlet output + deep ensemble for uncertainty + mineralML classifiers as deterministic post-processor for canonical naming.

### Alternatives considered

- **ALIGNN structural stream in v1.** Rejected: Position C training data is already structured (real-rock chemistry + MAGEMin equilibrium + empirical compositions). Structure's marginal value is low. Reserve it for the v2 polymorph branch where ALIGNN distinguishes Ky/Sil/And by structure (chemistry is identical).
- **Multi-output G_sys per polymorph in v1.** Rejected: requires polymorph-forced MAGEMin runs (3-5× cost), and the disambiguation story is more compelling when paired with ALIGNN. Defer the whole polymorph branch to v2.
- **Single-output paper without composition head.** Rejected: mode-only CrabNet ≈ faster CIPW. The composition head is what justifies the contribution and enables SS endmember prediction.

### Why this scope

Four-axis differentiation table is the contribution: bulk-only input + no spot analyses + multimodal output + hydrous/sedimentary coverage. v1 delivers all four. Adding ALIGNN or multi-output G_sys doesn't widen the differentiation table — it only enables the polymorph axis, which is a separate (worthy) v2 story.

### Revisit when

- v1 trained and benchmarked. If composition predictions are good but ambiguous on polymorph pairs (Qz/Crs/Trd in evolved rocks; rare in our likely test set), v2 priority confirmed.

<a id="adr-002"></a>

## ADR 002: Paths explicitly rejected

**Date:** 2026-05-12
**Status:** Active

Things the team has already considered and rejected, with reasons. Don't re-pitch these without new evidence.

### CIPW norm as training target or auxiliary output
**Rejected.** CIPW is a deterministic calculator (pyrolite implements it). Training a model to predict CIPW = building a worse surrogate for a free function. Only residual use: optional sanity check on synthetic rocks during training-data generation (not at inference).

### Bulk-chemistry realism filter as assemblage realism
**Rejected.** complete.csv tells you the bulk is plausible. It does NOT tell you which of many possible assemblages corresponds to that bulk. By the inverse problem's definition, bulk → assemblage is 1-to-many; the bulk-realism filter only addresses bulk → bulk.

### Empirical assemblage mining from mineralML's Sample Name co-occurrence
**Rejected.** mineralML's training corpus is 98.3% single-mineral samples (35,062 of 35,677 by Sample Name). Only 615 multi-mineral samples. Useful as a smoke test, not as a primary signal.

### Validation_Data/PetDB_allminerals/ as a training source
**Rejected.** It's mineralML's test set (partial coverage, single source). Use the mineralML grain corpus from Zenodo (`data/grains/mineralml.csv`) instead — multi-source, curated, 88k rows.

### Using mineralML as a runtime dependency
**Rejected.** Vendor only the specific classifiers needed (FeldsparClassifier, PyroxeneClassifier, etc.) into `src/cnmineral/stoichiometry/`. Avoids version coupling and lets us hot-fix boundary rules without upstream PRs.

### Position B alone: T,P sweep emulator
**Rejected as primary framing.** Position B (CrabNet as MAGEMin-marginalized-over-T,P) is a Bayesian wrapper that adds amortization but not new information. Position C (MAGEMin × mineralML fusion) is the stronger contribution because it adds empirical composition realism that neither tool produces alone. Position B is the methodological scaffolding inside Position C, not a standalone story.

### Single-scalar G_sys for polymorph disambiguation
**Rejected.** A single G_sys output cannot distinguish Ky/Sil/And (they have different G but same bulk). For real polymorph disambiguation, need either (a) multi-output G_sys head, one scalar per polymorph candidate, or (b) conditional G_sys head taking candidate assemblage as input. Both are v2. In v1, G_sys is an encoder regularizer only.

### Brute-force enumeration of C(31, k) combinatorial space
**Rejected.** Real rocks live on a low-dimensional manifold defined by petrologic processes. Condition on rock class first (~15 IUGS buckets) → within-class assemblage is canonical (3-6 phases). Combinatorial size collapses by ~3 orders of magnitude.

<a id="adr-003"></a>

## ADR 003: Tiered H2O handling strategy

**Date:** 2026-05-12
**Status:** Active

### Decision

Apply a three-tier H2O strategy at the bulk → MAGEMin schema-conversion step,
mirroring `fe_redox.py`:

- **T1 (measured):** if `complete.csv` reports `h2o_tot` for a row, use it
  directly. No assumption. Affects ~1% of rows.
- **T2 (class-conditional sample):** for the other ~99%, draw an H2O value
  uniformly from a per-qap_name range. Defaults from Wallace 2005 (MORB/OIB),
  Plank et al. 2013 (arc compositions), Burnham 1979 + Holtz et al. 2001
  (granitic systems), Best 2003 (general).
- **T3 (does not exist):** the `default` range is always defined as a fallback
  for unknown classes; no row drops on H2O grounds.

Each training tuple carries an `h2o_provenance` label (`measured` or
`assumed_class`) parallel to `fe_provenance`, so we can stratify evaluation
by H2O-tier.

### Why per-class **ranges**, not point estimates

Petrologists agree on central tendencies per class but the within-class
variance is genuinely large:
- MORB: 0.1-0.5 wt% (modal ~0.2)
- Arc basalt: 2-6 wt% (sometimes up to 8 in wet arcs)
- Granite I-type: 3-5; S-type: 4-7; A-type: 1-3
- Mantle peridotite: 0.005-0.1 (essentially dry)

Using point estimates would (a) commit to a precision the literature doesn't
support and (b) draw skeptical reviewer responses. Ranges + uniform sampling
mirrors the `TPRangeStrategy` pattern and stays consistent with how we treat
T/P priors.

### Alternatives considered

**(A) Skip H2O fix; document hydrous-phase gap as known limitation.**
Rejected. The paper's differentiation table claims hydrous-phase coverage as
one of four axes. Amp 0.3% / bi 0.2% in training labels would force an
asterisk that materially weakens the contribution. Also: training the ModeHead
on degenerate amp/bi labels collapses those gates to permanent zero — retraining
to recover them later means starting from scratch, not incrementally improving.

**(B) Per-class point estimates (e.g., MORB = 0.3, granite = 4).**
Rejected. Point estimates ignore within-class variance and are less defensible
in the methods section. Geologist reviewers would push back on any single
number for "the" H2O content of a rock class.

**(C) Fit H2O empirically from major-element composition.**
Considered briefly — papers like Plank et al. (2013) fit fO2 / H2O from
correlations with K2O or alkalis. Rejected for v1 as overkill: adds another
calibration to validate, and the per-class uniform-range approach is already
defensible. Could revisit for v2 if class-conditional sampling produces visible
artifacts in the trained model.

### Why this scope

- Symmetric to existing `fe_redox.py` pattern → reviewers can see we treat
  oxidation state and water content with the same methodological rigor
- Toggleable via strategy injection — sensitivity analysis is one config change
- Provenance tracking carries through to H5 → stratified eval is possible
- Cheap (~half a day) and orthogonal to model architecture work

### Revisit when

- Training shows amp/bi predictions remain degenerate despite the H2O fix —
  could indicate ranges are still too narrow for arc / dacite / granite classes
- Real-rock evaluation on EDS-map-derived bulks shows systematic
  underprediction of hydrous phases at conditions where geologists would expect
  them — could indicate per-class central tendencies are off
- Reviewers push back on the per-class table → swap to Plank/Wallace
  literature-fit composition-conditional model (option C above)

### Cross-reference

- Journal entry: `docs/JOURNAL.md` 2026-05-12 "Hydrous-phase gap forces a
  v0-blocking design decision"
- Implementation: `src/cnmineral/data/h2o.py` (TBW); integration in
  `src/cnmineral/data/schema.py`
- Symmetric pattern: `src/cnmineral/data/fe_redox.py`, `src/cnmineral/data/tp_priors.py`

<a id="adr-004"></a>

## ADR 004: Use PetThermoTools for fractional-crystallization experiments and (potential) production label generation

**Date:** 2026-05-13
**Status:** Resolved → Phase B, **amended 2026-06-30 (production scoped to volcanic; equilibrium, not FC)**; the volcanic generation record folded in 2026-09-03; **amended 2026-10-09 (the benchmark fractional runs drive MAGEMin_C directly; PetThermoTools kept as the reference check)**

### Amendment 2026-06-30 -- v1 production is volcanic + equilibrium; FC deferred to the plutonic branch

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

Measured 2026-09-18 (`explorations/generator_reach/`,
`explorations/prior_search/`, journal 2026-09-16 and 09-18). The argument
holds for the whole rock but not for the benchmark's ruler: du Bray counts
phenocrysts, and no single (P, H2O) equilibrium of the rock's own bulk
reproduces all four silicate phenocryst shares for 38% of the evaluated
rocks, while no prior over conditions, global or chemistry-conditioned,
moves those labels by more than 0.03 in share. The production recipe is
unchanged by this; it sequences the next generator (fractional
crystallization of the rock's own bulk with the accumulated crystals as
the label, against the phenocryst ruler) behind a plutonic whole-rock
ruler read at a solidus rung (journal 2026-09-18).

Consequences:
- v1 production generator = `magemin/run_magemin.jl` in a Julia subprocess,
  driven by `scripts/build/generate_mineralogy_corpus.py`; `(P, H2O)` are
  sampled and T is derived by cooling. `NearSolidusTPStrategy` stays
  unimplemented.
- FC (and PetThermoTools as its driver -- the tooling choice below is
  unchanged) moves to the deferred **plutonic/cumulate branch**, where it
  models FC of parent magmas (issue #22). ADR 005's FC production recipe
  applies there, not to the volcanic v1 corpus.

### Amendment 2026-10-09 -- the benchmark fractional runs drive MAGEMin_C directly

The fractional crystallization of the benchmark rocks' own chemistry (#83
Stage B's recipe arm, #84) runs through
`explorations/generator_reach/run_fractional.jl`, a loop in the production
Julia environment, not through PetThermoTools. PetThermoTools stays installed
(0.4.5 from the owner's clone, its Julia environment aligned to MAGEMin_C
2.1.9 on 2026-10-06) as the reference the loop is checked against, never as
the driver. The choice below was made on what PetThermoTools' fractionation
was expected to handle: liquidus bracketing, adaptive step sizing near the
solidus and terminal-solid edge cases. Reading MAGEMinCalc 0.6.1, its Julia
half, on 2026-10-06: the fractionation is a fixed-step loop (minimize at each
T, record every phase, set the next bulk to the melt's composition, scale the
mass by the melt fraction) with none of the three; the liquidus search exists
but cannot be used, since pairing with the equilibrium run needs the shared
1200 to 600 C ladder. Driving that loop from Python costs what the May run
paid: the juliacall memory leak (65 MB per path measured, so workers recycled
every 32 draws), no timeout reaching the Julia call in this version,
convergence status not exposed (G rebuilt from H and S), a two-layer phase
renaming to undo, and iron passed as a ratio. The loop is the same scheme with
status and G per step, a stop at melt-out, raw MAGEMin phase names, FeO and
Fe2O3 passed as the equilibrium run passes them, and the equilibrium runner's
subprocess, timeout and crash bisection; on plutonic rock 0's first 12 draws
it agrees with PetThermoTools within 0.002 of the starting mass on the melt
and 0.004 on any phase at every step but one (`explorations/generator_reach/
README.md`). The defensibility argument below ("labels generated using
PetThermoTools") is replaced by "MAGEMinCalc's fractionation scheme, verified
against PetThermoTools path for path". ADR 005's production recipe for the
plutonic branch (#22) is untouched by this; if that branch is built, the same
loop serves it.

#### Generation record (landed 2026-07-27; the planning spec is retired)

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

### Resolution

Experiment ran 2026-05-13. 547/550 bulks across 11 classes converged on FC;
526/550 on eq. Overall median L1 = 0.350 (mean 0.482) — far above the 0.10
"adopt eq" threshold. Dacite at L1 = 1.14, 9 of 11 classes above 0.25.
Decision: adopt fractional crystallization for production label generation.
Full results, plots, and discussion: `docs/JOURNAL.md` 2026-05-13 entry
"PetThermoTools integration + eq-vs-frac experiment results".

`NearSolidusTPStrategy` (the Phase A code path) is NOT implemented and should
not be added.

### Decision

Adopt **PetThermoTools** (https://petthermotools.readthedocs.io) as the
fractional-crystallization driver for the equilibrium-vs-fractional
comparison experiment, and — if the experiment indicates fractional
crystallization is needed for v1.5/v2 production label generation — for that
work too.

PetThermoTools is a Python wrapper around MAGEMin_C.jl that exposes
fractional-crystallization simulation via `Frac_solid=True` plus auxiliary
features (solidus/liquidus bracketing, configurable cooling step sizes,
proper handling of solid-removal accumulation, plotting utilities).

### Why this decision

The immediate need is the experiment described in journal entry
`docs/JOURNAL.md` ("PetThermoTools experiment design: equilibrium vs
fractional crystallization"). The experiment requires reliable, defensible
fractional-crystallization simulation across many bulks for comparison
against single-snapshot equilibrium labels.

Two candidates were considered:

#### Alternative 1: Custom Julia loop

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

#### Alternative 2 (chosen): PetThermoTools

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

### Open implementation questions (for the integration session)

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

### Revisit when

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

### Cross-reference

- Journal entry: `docs/JOURNAL.md` 2026-05-13 "PetThermoTools experiment
  design: equilibrium vs fractional crystallization"
- Implementation will live in: `scripts/experiments/eq_vs_frac/` (TBC during
  integration plan)
- Existing MAGEMin integration: `scripts/julia/run_magemin.jl`,
  `scripts/julia/Project.toml`
- Existing handoff brief (PetThermoTools not mentioned there;
  this is new infrastructure): `docs/magemin_handoff.md`

<a id="adr-005"></a>

## ADR 005: Production label generation via PetThermoTools fractional crystallization

**Date:** 2026-05-15
**Status:** Resolved → Executed (2026-05-19) → **Superseded for the volcanic
v1 corpus** by ADR 004's 2026-06-30 amendment (equilibrium generator;
generation record in ADR 004). The FC recipe below is deferred with the
plutonic/cumulate branch (issue #22).

### Execution outcome

Corpus generated at `data/train_fc_v1.parquet` (gitignored). 177,976 path-rows
across 22,245 bulks × 8 paths. Mean `liq` mass fraction = 0.049 (vs v0
baseline 0.70 — 14× reduction). 99.96% FC convergence. Memory leak in
juliacall+MAGEMin required workarounds (maxtasksperchild=1, chunk_size=25,
explicit Julia GC between paths) — full diagnosis in JOURNAL.md 2026-05-17
entry. ADR 006 picks up at model architecture.

### Decision

Regenerate the training corpus by running **PetThermoTools fractional
crystallization** per bulk, sampled across a 2D grid of (P, H2O) per bulk,
with fO2 set class-conditionally by the existing `fe_redox.py` strategy.
The label for each (bulk, P, H2O) combination is the **in-situ modal vector
at a per-bulk T_FC_end**, where T_FC_end is the highest T at which the
bulk's own residual silicate melt fraction falls below 5 wt% (snapped to
nearest 25 °C).

This supersedes the v0 corpus generation procedure (Putirka 2008 (T, P)
sampling at single-snapshot MAGEMin equilibrium).

### Why this decision

ADR 004 resolved the question of *whether* to use FC versus near-solidus
equilibrium (answer: FC). This ADR locks *how* to use FC for production.

The design is anchored on the eq-vs-frac v2 experiment
(`docs/JOURNAL.md` 2026-05-15) which (a) showed FC modal vectors diverge
meaningfully from closed-system eq at the same T for 6 of 11 classes, and
(b) operationalized "subsolidus" as a class-aggregate liq < 5 wt% threshold.
Production lifts the design from class-aggregate to per-bulk to eliminate
the ~11% audit attrition seen in v2.

### Binding choices

#### Per-bulk T_FC_end

For each bulk: run one FC sweep at the bulk's class-default (P, H2O),
inspect the per-step liq trajectory, set `T_FC_end_bulk` = highest T (snap
to 25 °C) at which liq < 5 wt%. Floor at MAGEMin's calibration limit per
class (~500 °C felsic, ~600 °C mafic).

**Reason:** within a class, H2O varies up to ~3x (e.g. andesite 2.8 - 4.8
wt% p25-p75). The wet solidus depression scales with sqrt(X_H2O), so one
T per class is wrong for a fifth of the bulks. v2 showed this directly:
basalt and gabbro had 30 % audit attrition at class-level T. Per-bulk T
costs one extra FC sweep per bulk; with the main run already doing ~16
sweeps per bulk this is a 6 % overhead and zero attrition.

**How to apply:** the per-bulk T discovery is a pre-pass that produces a
`T_FC_end_bulk` column in the bulks table. The main run reads it.

#### (P, H2O) sampling per bulk

Cross-product of `n_P` × `n_H2O` paths per bulk:

- P axis: sampled across the per-class Putirka 2008 range from
  `src/cnmineral/data/tp_priors.py`. Reuse the existing range table; just
  drop the T axis.
- H2O axis: sampled across the per-class range from
  `src/cnmineral/data/h2o.py`. Reuse the strategy as-is.

`n_P` and `n_H2O` jointly set the path count per bulk. Total compute
scales linearly. **Specific path count is an implementation detail (see
"Open implementation questions"), not an ADR-level decision.**

#### fO2 stays class-conditional

Use `fe_redox.py`'s class-default Fe3Fet ratio per bulk; do not sweep fO2
across paths in v1. fO2 sweep is deferred to v2.

**Reason:** v0 already worked with fixed class-conditional fO2 and 14 of
17 phases reached R² > 0.5. Adding an fO2 axis would double the compute
budget for unproven marginal gain. ADR 006 (when written) can revisit.

#### Single endpoint per path

Each (bulk, P, H2O) path produces one training label: the in-situ modal
vector at `T_FC_end_bulk`. Do not emit per-step snapshots along the
cooling path.

**Reason:** path-along-cooling multimodality compounds with (P, H2O)
multimodality and inflates the corpus 5-10x for unproven gain. The (P, H2O)
sampling already gives the multimodal posterior we want. If v1 training
shows the posterior is too tight, add T-snapshot multimodality in a v1.5
iteration.

#### Label format: in-situ modal vector

For each path, the label is the **accumulated cumsum solids + residual
liq + residual fl** at `T_FC_end_bulk`, normalized to 1.0. Same procedure
as the v2 experiment's FC leg. Includes `liq` and `fl` channels in the
modal vector.

**Reason:** the L1 comparison in v2 was over in-situ phases (including
liq + fl). The model's job mirrors that comparison — predict the in-situ
assemblage at the bulk's T_FC_end. Post-hoc "drop liq + fl and renormalize"
is a downstream choice the user can make at inference time; the training
labels keep all information.

#### Vocabulary discovered fresh

The phase vocabulary for the production corpus is discovered from the FC
output, not inherited from v0's 25-phase Tier A. v2 already showed FC
produces `ep` and FC vs eq schemas differ slightly; the production schema
may differ further at scale.

### Open implementation questions

These are concrete choices the production code makes; they don't bind
architecturally and can shift based on compute headroom.

- **Path count per bulk.** 8 (e.g. 2×4 or 4×2) → ~175k labels, ~9 hr
  wall on 8 python workers. 16 (4×4) → ~350k labels matching v0
  density, ~18 hr. Pick once initial smoke run lands.
- **Parallelization.** ptt's intra-process multi_processing dispatches
  across Julia workers per parameter sweep (e.g. `label='P_bar'` with
  an array of P values). Python-level multiprocessing across bulks
  stacks badly with ptt's own workers (52+ processes on a 13-worker
  Julia config). Likely path: python-level worker pool (4-8 workers),
  ptt's multi_processing=False inside each. Measure on a smoke chunk.
- **Output format.** Extend the existing `train_magemin.h5` schema with
  a `path_idx` per (bulk, path), or use a new `train_fc_v1.h5`. New
  file is cleaner and preserves v0 baseline. Use a new file.
- **Chunked orchestrator structure.** Mirror `scripts/orchestrate_magemin.py`
  pattern: chunk by bulk, write per-chunk parquets to `data/fc_cache/`,
  retry crashed chunks via subdivision (same recovery story as v0).
- **Per-bulk T_FC_end pre-pass storage.** Write to
  `data/bulks_v1_with_T.parquet` or similar; the main orchestrator
  reads it.
- **Audit policy for the pre-pass.** Some bulks may not satisfy liq <
  5 wt% within the calibration floor (the v2 11 % tail). For
  production: use the floor T for those bulks anyway, tag with a
  `T_FC_end_attrition` flag in the corpus, and let evaluation
  optionally drop them.

### Revisit when

- The v1 corpus is built and BaseModel retrains. Check per-class R²
  and mean corpus liq fraction. If liq mean > 5 wt% or per-class R²
  collapses below v0 for any class with > 1 % presence, reassess one
  of the binding choices.
- fO2 sweep adoption (currently class-conditional). Open question: does
  doubling compute for fO2 multimodality move per-phase R²? Plausible
  v2 ADR.
- Per-step (T-snapshot) labels. Same question: does it help volcanic vs
  plutonic disambiguation? Plausible v1.5 ADR.

### Cross-reference

- Predecessor: ADR 004 (PetThermoTools adopted, Phase B resolved)
- Experiment: `docs/JOURNAL.md` 2026-05-13 (v1, dropped) and 2026-05-15
  (v2, decision-locking)
- Diagnostics: `docs/eq_vs_frac_diagnostics.md`
- Existing pipeline patterns to mirror: `scripts/orchestrate_magemin.py`,
  `scripts/retry_failed_chunks.py`, `scripts/parse_magemin.py`
- Strategy modules to reuse: `src/cnmineral/data/tp_priors.py` (P axis
  only), `src/cnmineral/data/h2o.py` (H2O axis), `src/cnmineral/data/fe_redox.py`
  (fO2)

<a id="adr-006"></a>

## ADR 006: v1 model architecture — shared encoder + Dirichlet ModeHead + GMM CondHead + joint Optuna HPO

**Date:** 2026-05-19
**Status:** Historical -- the decision principles hold (shared encoder, probabilistic heads, NLL losses, joint single-stage HPO), but the concrete v1 shapes below are superseded: current input is the 22-token cation+trace vector (23 with the redox token), ModeHead is the hurdle presence+magnitude head, CondHead is support-conditioned, and no K-seed ensemble is built yet. Current architecture in `docs/heads.md` / `docs/encoder.md`.

### Decision

The v1 model is a **single shared encoder feeding two probabilistic heads**:

```
Input: 11 oxide wt% (bulk only — no T, P, or H2O at inference)
   ↓
Encoder (SubCrab wrapper) → pooled d_model latent
   ├─→ ModeHead: Dirichlet(α) over n_minerals modal vector
   └─→ CondHead: 3 × K_gmm GMM over (T, P, H2O) conditions
```

Trained with joint NLL loss: `L = 0.5·NLL_dirichlet + 0.5·NLL_gmm` (after
z-scoring the GMM contribution for scale parity). K=5 deep ensemble for
σ_epistemic. Hyperparameters chosen via joint single-stage Optuna TPE search.

This is the **inverse-inference architecture**. From bulk chemistry alone the
model predicts the joint posterior `P(T, P, H2O, modal | bulk)` — the
deployed model never sees (T, P, H2O) at inference, but the training corpus
supplies them as labels.

### Why this decision

#### Project differentiation argument (from 2026-05-19 journal entry)

The two existing tools live on different sides of the problem:
- **MAGEMin**: forward map (bulk + conditions → assemblage). Needs conditions
  the geologist doesn't have.
- **mineralML**: per-grain classification from spot analyses. Wrong granularity
  for whole-rock analysis.

Geologist's workflow today: run MAGEMin over a (T, P, H2O) sweep, stare at an
unweighted range of possible mineralogies. Our model **collapses that range
into the conditioned posterior** by learning, from a corpus of realistic
(bulk, conditions, assemblage) triples, which conditions are likely for a
given bulk.

A pure modal-prediction model (deterministic OR probabilistic) is Position B —
"MAGEMin emulator marginalized over T, P" — which was explicitly rejected at
project inception. Our architecture must do something MAGEMin can't, and
**condition inference is exactly that.**

#### Per-component rationale

**Shared encoder.** Multiple heads benefit from a common bulk representation.
This also sets up future CompHead (mineralML compositional fusion) and
GsysHead (thermodynamic regularizer) without further architecture work.

**Dirichlet ModeHead.** Mineral wt% labels sum to exactly 1 (MAGEMin output
is normalized). Dirichlet is the natural distribution over the simplex,
parameterized by `α` (concentration parameters). Provides:
- Sum-to-1 enforced by construction
- Aleatoric uncertainty per phase (Dirichlet variance)
- Multimodal-friendly via ensemble aggregation of K=5 `α` vectors
- Label smoothing ε=1e-3 handles exact zeros in sparse targets

**GMM CondHead.** Conditions (T, P, H2O) are continuous, and the corpus has
many paths per bulk → the marginal posterior over conditions for a given
bulk is genuinely multimodal (a basalt could plausibly be either shallow
volcanic OR deeper plutonic). A unimodal Gaussian would average over those
modes and miss the multimodality. GMM with K_gmm components captures it.
Linear-space (no log transform) — predictions outside the physical range
are diagnostic information, not problems to hide.

**Joint single-stage HPO.** A two-stage HPO (e.g., tune model size first,
loss weights second) assumes separability that doesn't hold — `d_model` and
`λ_cond` interact (a bigger encoder can carry condition signal at lower λ;
a smaller encoder needs higher λ). Joint search over all dims with TPE
sampler.

**NLL objective (not R²).** NLL is a proper scoring rule that automatically
yields calibrated uncertainty estimates. R² and calibration coverage are
reported *separately* in the evaluation, but they don't drive the
optimization. (Optimizing R² alone would reward overconfidence; optimizing
NLL doesn't.)

### Alternatives considered

#### Alt 1: Deterministic centroid head (rejected — Position B)

A single deterministic head predicting the mean modal vector. Marginalizes
over the 8 paths per bulk into one "average" assemblage. Was rejected
because:
- It IS Position B in different clothing. Bayesian-amortization of MAGEMin
  over the rock-class-conditional (T, P, H2O) prior.
- Loses the "calibrated multimodal" differentiation row of the 4-axis table
  in CLAUDE.md.

#### Alt 2: Probabilistic ModeHead only (no CondHead) — rejected

Just the Dirichlet head over modal. Captures uncertainty but doesn't predict
conditions. Reviewer's natural rejoinder: "what does the model do that
MAGEMin doesn't?" Without CondHead the answer is just "marginalizes over T,
P, H2O with calibrated uncertainty" — which is amortization, not new
science.

#### Alt 3: CondHead only, then pipe to MAGEMin — partially adopted

A model that predicts only (T, P, H2O), then the user calls MAGEMin with
those conditions to get modal. This is a valid simpler architecture and we
should evaluate against it in Stage 3 (two-step pipeline as upper bound for
modal prediction accuracy, since MAGEMin generated the labels).

But the joint architecture has two advantages:
- Single-forward-pass inference (microsecond claim)
- ModeHead can later be trained against non-MAGEMin labels (mineralML
  fusion) to deviate from pure MAGEMin output

Stage 3 will validate whether the ModeHead is materially better than the
two-step baseline. If not, the project pivots toward "CondHead + MAGEMin" as
the deployed product and ModeHead becomes optional.

#### Alt 4: ALIGNN structural embeddings as v1 head — rejected, deferred to v2

Per ALIGNN-related decisions from project inception. Marginal value for
modal prediction; only meaningful for polymorph disambiguation in v2.

#### Alt 5: DFT-derived ΔG catalog as physics-constraint loss — rejected, deferred

User direction: the "we have better ΔG than HP" argument is debatable on
its own merit (HP is itself partly DFT-informed). Not pursuing in v1.

#### Alt 6: mineralML CompHead in v1 — deferred to v1.1

The most natural escape from "MAGEMin surrogate" framing is empirical
per-mineral composition fusion. But it requires real data engineering work
on mineralML's grain database, and the v1 architecture works regardless of
which corpus deviations get added later. So mineralML fusion is v1.1 and v1
ships with MAGEMin-only labels.

#### Alt 7: Two-stage HPO — rejected

Fixing some hyperparameters in stage 1 and tuning others in stage 2 misses
interactions. Joint single-stage TPE search is the principled choice even
though it explores a higher-dimensional space.

#### Alt 8: MSE on Dirichlet's mean instead of NLL — rejected

Loses the calibration signal. The architecture is probabilistic by design;
the loss should match.

#### Alt 9: Log-space transformation for T and P in CondHead — rejected

Adds complexity (transform target on input, exp on output) to enforce
non-negativity that the data already enforces (T, P >> 0). A model predicting
negative T is a diagnostic signal, not a problem to hide via transform.

#### Alt 10: Multivariate normal CondHead (correlated T, P) — viable, deferred

(T, P) are anti-correlated for plutonic-vs-volcanic. Independent GMMs per
axis don't model this. A joint MVN-GMM would. But independent is cheaper
and we don't have evidence yet that the correlation matters. Revisit if
Stage 2 results show smearing across the volcanic/plutonic divide.

### Open implementation questions

These don't affect the architectural decision but need to be settled during
implementation:

- **Stratification key for k-fold CV**: `qap_name`, matching v0. 64 distinct
  classes, some very rare — may need `StratifiedShuffleSplit` with
  smallest-class fallback.
- **`λ_cond` z-scoring reference**: against the first trial's NLL_cond value,
  cached in the Optuna study.
- **`K_gmm` discrete search range**: 2, 3, 4. Probably converges to 3.
- **Batch size search range**: must be compatible with MPS memory.
- **Train epoch budget per HPO trial**: 30 (matches upstream CV_EPOCHS).
  Final non-HPO run uses 200+ epochs.

### Revisit when

- Stage 3 evaluation completes. If ModeHead is no better than two-step
  baseline, architecture is overweight — consider stripping back to
  CondHead-only.
- mineralML fusion is built (v1.1). CompHead becomes a third head, joint
  loss adds a fourth term, HPO search space gains another dimension.
- DFT-derived ΔG or G_sys is captured (v2). GsysHead becomes a fourth head.

### Cross-reference

- Project differentiation framing: JOURNAL.md 2026-05-11 "Position C
  crystallized as the design"
- Inverse-inference reframing: JOURNAL.md 2026-05-19 entry
- Corpus generation: JOURNAL.md 2026-05-17 to 05-19 entry, ADR 005
- Upstream Optuna reference:
  - `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn/src/objective.py`
  - `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn/scripts/run_scaling.py`
- Stage-1 implementation will live in:
  - `src/cnmineral/models/{encoder,heads,model,losses,dataset,training,objective}.py`
  - `scripts/{03_train_base.py, 04_hpo.py}`
  - `configs/{base_v1.yaml, hpo_v1.yaml}`

<a id="adr-007"></a>

## ADR 007: SSL pretrain objective is per-element standardized log-space masked MSE

**Date:** 2026-06-18
**Status:** Active

### Decision

The canonical SSL encoder is pretrained with a **per-element standardized
log-space masked-element MSE** loss (`loss: {type: log_mse, standardize:
true}`), replacing the linear-space masked MSE used by the prior canonical
encoder `pretrain_v3_seed0`.

First canonical checkpoint under this objective:
`checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt`
(architecture unchanged: d_model 256 / 4 layers / 8 heads; 20 epochs on Gard
2019 igneous bulks). Optimization hyperparameters from the per-objective
Optuna sweep (`configs/pretrain/pretrain_hpo.yaml`).

The objective carried unchanged into the all-rock re-pretrain: the canonical
accepted encoder is now `checkpoints/pretrain_allrock_redox_seed0` (accepted
2026-07-28, the scorecard in `docs/encoder.md`), riding these same ADR-007 hyperparameters.
The production heads still load the igneous checkpoint until cascade 2
retrains them on the new latent.

### Why this decision

The linear-space MSE objective weights each element by its raw target
magnitude. Majors (wt%/100) and traces (ppm/1e6) differ by ~1e7-1e8x, so trace
residuals contribute almost nothing to the loss and the encoder is trained to
fit majors almost exclusively. Scoring the frozen latent on a leakage-guarded
XGBoost trace-imputation gate and on frozen-latent classification probes, the
standardized log-space objective measures higher on every axis than the
linear-space baseline, across two seeds:

| metric (frozen-latent) | linear MSE (baseline) | standardized log-MSE |
|---|---|---|
| gate median trace log-R2 | 0.839 | 0.877 (seed 1: 0.873) |
| gate worst-trace Yb log-R2 | 0.724 | 0.831 (seed 1: 0.841) |
| gate majors log-R2 | -- | +0.002..+0.032, all 8 |
| supergroup MLP probe | 0.806 | 0.821 (seed 1: 0.820) |
| qap MLP probe (fixed) | 0.762 | 0.769 (seed 1: 0.771) |
| qap MLP probe (HPO) | 0.776 | 0.790 (seed 1: 0.788) |

The major-driven classification probes do not regress; they improve. So the
objective is not trading major structure for trace recovery -- both move up.

### Alternatives rejected

- **Plain log-space MSE (`standardize: false`).** Same direction, uniformly
  weaker (gate median trace 0.871, supergroup MLP 0.812, qap MLP fixed 0.758
  -- below the baseline on qap). Per-element standardization in log space adds
  a consistent margin at no measured cost.

- **Inverse-variance-weighted linear-space MSE (`weighted_mse`).** Worse than
  the linear baseline on both instruments (gate median trace 0.706, supergroup
  MLP 0.549). Equalizing each element's loss contribution by a fixed weight
  does not address a dynamic-range problem carried in the residual magnitudes:
  the optimization still chases large-magnitude majors and learns a latent with
  less trace signal and less class structure. Rejected.

- **Re-searching the architecture per objective.** Held to the original
  256/4/8 winner so the objective is the only variable vs the linear-MSE
  encoder; re-searching architecture would reconfound the comparison.

### Consequences

- `pretrain_v3_seed0` (linear MSE) is retired as canonical; it remains on disk
  as a reference artifact and as the baseline in the gate/probe result trees.
- ~~The configurable-loss infrastructure
  (`cnmineral.training.pretrain_objective`) stays: it is how this comparison
  was run and how a future objective would be tested.~~ Superseded by the
  amendment below.
- The imputation read-out head re-scores against the new frozen latent; its
  earlier frozen-latent reference (0.839 median trace log-R2) becomes 0.877.

### Amendment (2026-06-29)

The configurable-loss infrastructure is removed. The linear-space objective
(`masked_element_mse`), the rejected plain-log variant (`standardize: false`),
and the `loss:` config token with its type-dispatch are deleted. The objective
is now singular and unconditional: per-element standardized log-space masked
MSE, defined in code and in this ADR rather than selected by config. The
present-tense decoupling seam is kept -- the train loop and `evaluate_on_test`
still consume an opaque `loss_fn` built by `make_loss_fn` -- but it builds the
one objective with no menu.

The original reason for keeping the linear code was reproducibility of the
linear-vs-log comparison. That is preserved by the existing tag
`v3.1-logmse-objective` (commit `6ba9695`, 2026-06-24), which points at a
pre-cleanup state containing both objectives and all linear configs:
`git checkout v3.1-logmse-objective` restores the full comparison harness. The
seed-1 confirmation run is reproducible from `pretrain_production.yaml` with
`seed: 1`.

<a id="adr-008"></a>

## ADR 008: drop h2o_tot from model inputs (H2O target leak)

**Date:** 2026-05-22
**Status:** Active

### Decision

The model takes anhydrous inputs only -- `h2o_tot` is dropped (the v1 shape was 10 anhydrous oxide wt%; the current 22/23-token cation+trace vector keeps the same anhydrous rule).
The CondHead infers H2O genuinely from anhydrous chemistry.

### Context

In the FC corpus the input oxide `h2o_tot` equals the CondHead H2O target
`h2o_path_wt` (corr 1.0 -- the corpus set each bulk's water to the FC path's
sampled H2O). Feeding `h2o_tot` in target-leaked the H2O axis: the CondHead
was copying an input, not inferring it (fake H2O MAE 0.034). Dropping it makes
H2O a genuine prediction (honest test MAE 0.535 wt%). See JOURNAL 2026-05-22.

### Consequences

- Model input is 10 oxides / 9 cations (no H token). `V1_CATIONS_ORDER` in
  `dataset.py`; v0's `TrainMageminDataset` is untouched.
- Hydrous-phase modal R² regressed (amp/bi/mu/fl/ne, ~0.13-0.25 each) --
  `h2o_tot` was also a genuine predictor for water-dependent phases. This cost
  is accepted: the leaked feature was informative *because* it was the answer.

### Alternatives considered

- **Keep h2o_tot, CondHead predicts only T+P.** Rejected -- drops the H2O
  inverse-inference axis entirely.
- **Use complete.csv's measured `h2o_tot` as the input** (was "Option C").
  Rejected after checking the data: complete.csv `h2o_tot` is 1.1% populated
  for corpus bulks, and where present it does not track the FC `h2o_path_wt`
  (corr -0.35). Measured current water is decoupled from the sampled FC
  magmatic water that the modal labels were computed from, so it cannot
  recover the hydrous-phase regression. `h2o_path_wt` is a sampled
  experimental knob, not a rock property -- there is no honest input that
  substitutes for the leak.
- **Optional measured-water input token.** Rejected for the same reason -- no
  useful water signal exists to make optional.

### Revisit when

If a future corpus stops setting `h2o_tot := h2o_path_wt` and instead carries
a genuinely independent measured water column, reconsider adding it as input.
Not expected.
