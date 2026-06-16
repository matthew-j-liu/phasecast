# crabnet-mineral

## House Rules

**Use the `alignn_fix` conda env when running any code, tests, or
scripts in this repo** (`conda run -n alignn_fix ...` or activate
first) — unless an untracked `CLAUDE.local.md` at repo root
overrides the env name.  That override exists so sibling clones of
this repo can install their own editable copy without colliding on
`pip install -e .`.  

**Run the unit tests with** `pytest tests/`.
Single-threaded — cleaner stdout / tracebacks beats any speed win
from parallel workers at this suite size.

**After making code changes, follow this checklist:**

1. Add new tests if the change warrants them (new behavior,
   regression fix, non-trivial edge case).  Check first whether
   existing tests cover it implicitly — when a code change forces
   old fixtures to exercise the new contract (a new invariant
   they now satisfy or fail on), that IS the coverage.  Each new
   test should fail on something that would otherwise go
   unnoticed; flag redundant or outdated tests along the way.
   **Whenever a test is added, deleted, renamed, or repurposed,
   update [tests/TESTS.md](tests/TESTS.md) in the same change** —
   it's the per-test inventory (one sentence each, organized by
   directory) the owner uses to track suite coverage.
2. **Profile new or touched tests before checkin.**  Run
   `pytest <touched_tests>`. We will dial in specificity on what
   test durations we should shoot for, though as a general principle, code 
   should be written to be as low run time as possible (e.g., use
   vectorized operations whenever possible).
3. Use discretion on code review: invoke `/review` for non-trivial
   changes; skip for small / mechanical edits.  When uncertain,
   err on the side of reviewing.
4. Address easy fixes from the review; surface major issues to
   the user before acting on them.
5. If the change touches multiple responsibility areas and the
   coupling seems too tight, flag it for potential further
   review — don't fold a refactor into the current diff.

Before declaring a change done, read the diff as a reviewer would —
fresh skepticism, not the author's confidence.

**When renaming a function, class, or module, update every
reference in the same change**: comments, docstrings,
cross-references in other modules' prose, test names/docstrings,
and docs.  Grep for the old name repo-wide before declaring the
rename done; a rename that leaves stale prose references is
incomplete.

Don't run the full test suite by default — it's slow enough to be
friction on every change.  Assume tests are green; run them when
explicitly asked or when debugging a specific failure.

**When debugging test failures, iterate on the failed subset only**
— `pytest tests/test_foo.py::test_bar` for a single regression, or
`pytest --lf` (last-failed) after a round of fixes.  Re-run the
full suite once at the end.  Re-running ~200 tests between each
fix burns minutes per iteration with no added signal.

**Name files, tests, docs, and comments by what they cover, not
when they landed.**  References to PR numbers, plan steps, issue
numbers, or migration phases are anti-patterns — a reader six
months later has to do archaeology to figure out what they meant.
Names should reflect the *responsibility* being exercised: what
subsystem, which contract, which invariant.  Avoid jargon the
reader has to decode before they can guess what's covered.  If
new content fits the scope of an existing file, merge it in;
don't create a new file just to keep the diff visually tidy.

**Don't leak internal planning code-names into code or
user-facing output.**  Labels like `Phase 1`, `Scenario A/B`,
`cell-c`, or `step 2b` are fine in planning docs, JOURNAL, issues,
and conversation -- they organize the work.  But code, log lines,
CLI output, metric keys, and docstrings must name the actual
assumption and conclusion, not the code-name: a reader of the
terminal output or the source must understand what was assumed and
what was concluded without opening the planner or decoding a label.
When output summarizes a decision, spell out the criterion, the
measured values, and the resulting action -- not `-> Scenario A`
but e.g. `latent keeps 94% of the input ceiling -> recover traces
with a read-out head, no re-pretrain`.  Before committing decision
terminology to code (verdict strings, metric keys, the
assumption/conclusion wording), confirm the exact phrasing with the
owner.

**Reference code by symbol, not line number.**  Markdown links
like `file.py#L112` rot — line numbers shift on every edit.  Use
the function / class / variable name in the link text and let
the reader grep.

**Comments in production code and configs must outlive the
experiments that motivated them.**  A comment that reads
correctly today but is meaningless in six months is bloat, not
neutral.  The corpus changes, the architecture changes, the
benchmark numbers change; comments that anchor to a particular
run's outcome rot the moment that run is no longer the latest.

In production source files (`.py`, `.yaml`, etc.), the following
are anti-patterns:

- **Multi-line storytelling blocks about a specific experimental
  result on a specific day.**  Whole paragraphs explaining "we
  compared X vs Y and chose Z because of Q" -- belongs in
  JOURNAL, not next to the parameter.
- **Anchoring to specific numbers** that only mean something if
  the reader ran that exact comparison (e.g., "supergroup MLP
  +0.003, qap_name MLP -0.003").  Those numbers are valid for
  one snapshot of the codebase; they decay silently.
- **Treating one experiment as authoritative** ("20 epochs is
  the canonical budget because the 40-epoch run at seed 0
  showed...").  One experiment on one corpus with one
  architecture is not a permanent fact.  If the value is the
  right default, write `epochs: 20`.  If it needs context, the
  context belongs in JOURNAL.
- **Forcing the reader into archaeology.**  Any comment whose
  meaning requires the reader to look up a journal entry, a PR,
  a study journal file, or a memory file to validate is doing
  the wrong job.  Comments should be self-contained.

Where session-specific evidence DOES belong:

- `docs/JOURNAL.md` -- chronologically anchored by design; "on
  2026-06-11 we compared 20 vs 40 epochs and got X" reads
  correctly here forever because it's explicitly a log of what
  happened when.
- `docs/decisions/NNN_*.md` -- ADRs for binding architectural
  choices with rationale.
- `docs/V3_WORKFLOW.md` Track-status sections -- explicitly
  current-state; updates as gates close.
- PR descriptions and commit messages.

Production-side comments instead aim for **durable structural
content**: what a less-intuitive parameter represents, shared
constraints (`d_model % n_heads == 0`), cross-file consistency
reminders ("should match what was used in `probe_hpo.yaml`"),
action items if a value is changed ("update the checkpoint_dir
with this as well!").  Group related parameters by logical
bundle (architecture / optimization / data / I/O) with a one-line
header so the reader can navigate by skim.  Keep individual
comments terse; single line, inline preferred.

**Module docstrings: model them on
`src/cnmineral/training/reconstruction_mse.py`.**  A short
paragraph that says what the module does and where it sits in the
pipeline (who calls it, when, what its neighbors own) -- no
changelog, no structured `Run:`/`Outputs:` sections, no
function-by-function inventory.  If a responsibility lives
elsewhere, one pointer line ("checkpoint persistence lives in
`cnmineral.training.checkpoint`") is enough.

**Never add `from __future__ import annotations` to any file in this
repo.**  Quote individual forward references as strings; disabling
runtime evaluation for a whole module causes silent type-hint failures
in anything that calls `get_type_hints()` (Pydantic, FastAPI,
dataclass introspection).

**Schema-shape changes go through Schema Design Discipline first.**
Before adding a stamped metadata field, a new index, a new
discriminator, or a new schema dataclass, score the proposed
change against the the following rules:

1. **Every field earns its place against reconstruction.** Can this field be reconstructed from inputs + dependency outputs at query time? If yes, don't stamp. The same test applies to existing fields: if a stamped field is
reconstructible with no information loss and no current consumer needs it, drop it.  Carrying speculative metadata is unbounded; rebuilding when a real consumer surfaces is bounded.

2. **Schema evolution is asymmetric: adding is cheap, removing is expensive.** Adding a field is one line, but removing it requires updating every consumer, every test fixture, every doc, plus (with production data) a migration script. *Parsimony on the way in; aggressive deletion on the way out.*

3. **Strong domain models displace defensive coding.** When ``if X is None or X == "default" or ...`` shows up in multiple places, the data model wants to be tightened. *The data model owns invariants; the code reads them.*

4. **Beware speculative future-proofing.** Designs that solve future problems carry real maintenance cost in
the present. Defaults should be parsimony; future-proofing earns its keep only when the future arrives.

5. **Discipline pays compound interest.** Tests, types, responsibility-named symbols, and clear docs are
cheap relative to scary changes; expensive relative to first-write deadlines. Pay the cost early. 

6. **Naming reveals (or hides) confusion.** When a name feels awkward, the schema is telling you something. Whenever a name feels off, lean into the discomfort: good naming forces clarity about what each value means at each layer; bad naming masks confusion that compounds across boundaries.

**Production class design serves production callers — never tests.**
Don't add a field, constructor parameter, "override" knob, mutator
method, or validation relaxation to a production class because a test
fixture needs it.  That bakes test setup into the schema as permanent
public API and the cost compounds.  Tests adapt to production, not the other way around.  If a
test cannot be expressed using the existing API, the right move is one
of: (1) construct a minimal real instance that satisfies the production
shape; (2) use the same constructor / factory production uses; (3) propose a clean seam with
explicit rationale — don't smuggle one in as a "convenience override."
Before adding anything to a production class, run the test: "is the
only caller a test?".  If yes, rework the test or rework the API
surface explicitly — never both bend the schema *and* call it a feature.


**Default the schema dataclasses to `frozen=True, slots=True`; drop
`slots=True` only when a derived `@cached_property` makes it pay for
itself.**  `slots=True` blocks `cached_property` because the cache
writes through `__dict__`.  Frozen alone still blocks post-construction
field reassignment, so `slots=True` is buying *attribute-layout
compactness* — irrelevant at our schema-instance counts.

### ⚠️ DANGEROUS: a frozen schema property MUST NOT hand callers a shared mutable container

This is the single most-load-bearing rule in this section.  The bug
class it prevents is silent, latent, and non-local: the cache returns
a mutable object → some caller mutates it for legitimate local work →
**every other caller reading the same property afterwards sees the
corrupted value, with no exception raised**.  The corruption surfaces
far from its origin, and tests that exercise
each call site in isolation pass.  

**Frozen advertises an immutable identity.** Returning a mutable data structure
through a `@property` or `@cached_property` breaks that contract.  This
applies whether or not the property is currently cached — once the
class is frozen, the property's return shape is part of its public
contract, and "this caller doesn't mutate today" is not a guarantee.

**Choose one of:**

1. **Return an inherently-immutable value** (`tuple`, `frozenset`,
   `str`, scalar).  Always preferred when the underlying data is
   simple.
2. **Cache a private template under `_name` and copy on access** —
   the public accessor is a thin `@property` returning
   `self._name.copy()`.  Use this when the underlying type has no
   immutable variant.

**NEVER** offload "you must copy before mutating" onto the call
sites.  The only-call-site-that-mutates argument fails the moment a
second caller appears (or a refactor moves logic), and the failure
mode is silent corruption, not a clear exception.  Treat any
`@property` or `@cached_property` on a frozen class that returns a
mutable container as a structural smell to fix at the property, not a
documentation problem to solve at the readers.


**Imports go at the top of the file.**  Inline imports only for (1)
per-environment optional dependencies or (2) genuine circular imports 
(verify by moving to the top first).

**Don't alias cheap property or dataclass-field reads to locals.**
`x = obj.cheap_property` used fewer than ~5 times in close succession
is clutter, not caching.

**Audits update the codebase as they go.**  Deliverable is a
codebase that doesn't need the audit document to be understood.

**Consult the design docs before proposing changes.**  Two triggers:

1. **Before proposing an architectural change** in an area with a
   `docs/*.md`, skim the existing design sections.
2. **When something looks wrong or off** — "that doesn't look
   right," "this is redundant," "why isn't this simpler?" — check
   the docs before proposing a fix.  The "smells off" reaction is
   often the signal that a decision has hidden load-bearing
   justification.  Frequently the thing that looks like a bug is
   the fix for a subtler bug, or the thing that looks redundant is
   a deliberate defensive layer.

If your recommendation contradicts a recorded rationale, address
it specifically — don't re-litigate decisions without engaging with
why they were made.

**Planning documents stay prospective.**  A plan answers "what
does someone need to know to finish implementing these steps?" —
nothing else.  As a step lands, descriptions of "how the system
now works" migrate to permanent docs ([docs/*.md](docs/) or
module docstrings); the plan's landed-step section collapses to a
terse pointer ("Landed; see X.md").  Don't keep retrospective
enumerations of call sites, refinements, or pre-implementation
spec in the plan once the code ships.  Once every step has
landed, delete the plan.

**Don't leave dead code "because it's harmless."**  Unused fields,
functions, or branches are hand grenades — a future refactor
changes their inputs, or someone wires them up without
understanding why they were abandoned.  Git has it if you need it
back.

**Task tracking lives in GitHub Issues**
(https://github.com/matthew-j-liu/CrabNet-Mineral-v2), not in a static
doc.  Create tasks with `gh issue create`, applying one label from
each of `P{1,2,3}-*`, `type: *`, `area: *` (plus `status: *` or
`epic` where it applies).  `gh label list` enumerates the valid
values.  Link commits and PRs to issues via `Fixes #N` / `Closes #N`
in the message — GitHub auto-closes on merge.

## What This Is

"If I can generate the labels with MAGEMin, why do I need ML?" assumes
the model's job is to reproduce the generator's outputs.  Drop that
framing.

**The model's job is to learn a representation of rock chemistry,
supervised through samples from the generator.**  The representation
is the contribution.  Empirically: the v3 SSL-pretrained encoder
reaches 80.6% supergroup MLP-probe accuracy on a held-out test
split without ever seeing a supergroup label — pure structure
learned from masked-element correlations across 474k real bulks
(v2 cell-c was 76%).

The supervised heads are application slots: ModeHead (modal abundance),
CompHead (per-phase composition), CondHead (formation conditions),
ClassHead (supergroup, auxiliary), optional G_sys head.  They happen
to use MAGEMin labels now, but they could use real-rock labels, or no
labels at all — the encoder is useful on its own (see
[docs/PRETRAIN_APPLICATIONS.md](docs/PRETRAIN_APPLICATIONS.md) for 11
standalone capabilities).  MAGEMin's role isn't "labels you couldn't
get otherwise" — it's a way to shape the latent space toward
petrologically meaningful axes by giving the supervised loss real work
to do.

What the heads realize jointly is a probabilistic emulator of MAGEMin
integrated over geologically plausible (T, P, H2O) — the user gives a
bulk and gets the distribution over mineralogy and conditions
consistent with it.  MAGEMin doesn't natively do that integral.

As a product question: a geologist with MAGEMin doesn't need this for
assemblages, but does need it for the things MAGEMin can't compute —
per-mineral compositions calibrated against real grains, similarity
retrieval, partial-analysis imputation, measurement-error flagging,
tectonic-setting classification — and to do all of it at scale on
millions of rocks without managing Julia.

Concretely the architecture supports two job families through shared
infrastructure:

1. **Supervised heads against MAGEMin-derived labels.** ModeHead +
   CompHead + CondHead trained on a per-bulk × N FC paths corpus
   (PetThermoTools at per-bulk T_FC_end where residual liq < 5%).
   Inputs: anhydrous oxide wt% + trace ppm.  MAGEMin's job here is to
   shape the latent — the supervised loss pushes the encoder to
   organize rocks along thermodynamically meaningful axes, and the
   heads emit probability distributions (Dirichlet, GMM) that integrate
   over plausible (P, H2O) per bulk.  Bulk in, distribution over
   assemblage + conditions out, microsecond inference.
2. **Encoder applications without supervised heads.**  The same
   encoder, no extra training, supports element imputation, k-NN
   similarity lookup, OOD flagging, tectonic-setting classification,
   and petrogenetic-ratio regression — see
   [docs/PRETRAIN_APPLICATIONS.md](docs/PRETRAIN_APPLICATIONS.md) for
   the 11 capabilities and prioritization.  Improvements to the encoder
   (better SSL pretrain, broader pretrain corpus, joint multi-task aux
   losses) propagate to both job families.

### Four-axis differentiation table

| Tool | bulk-only input | no spot analyses | multimodal output | hydrous + sedimentary phases |
|---|---|---|---|---|
| MAGEMin | needs T,P | yes | one assemblage | yes |
| mineralML | yes | needs spots/pixels | no | partial |
| CIPW (pyrolite) | yes | yes | single answer | no |
| **CrabNet-Mineral v3** | **yes** | **yes** | **yes** | **yes** |

The contribution is covering all four axes simultaneously — not any
single one.

### Status snapshot

v3 fresh-codebase rewrite in progress.  v1 (`v1-trained` tag) and v2
(`v2-traces-ssl-final` tag) ship as reference artifacts; v3 keeps the
SSL-pretrained encoder design but rewrites the supervised heads to
add CompHead (per-phase composition) + optional ClassHead + optional
G_sys head.  Implementation plan in [docs/V3_WORKFLOW.md](docs/V3_WORKFLOW.md);
crystallized architecture and value-prop in
[docs/FRESH_START.md](docs/FRESH_START.md).

**Track A is closed** (2026-06-11): SSL encoder trained at the
Optuna-tuned hyperparameters on Gard 2019 igneous bulks for 20
epochs (a 40-epoch comparison showed no measurable downstream
gain, so 20 is the canonical compute budget).  Canonical
checkpoint: `checkpoints/pretrain_v3_seed0/epoch_best.pt`.  Probe
accuracy beats v2 cell-c by ~4-5 points on both targets
(supergroup MLP 0.806 vs 0.76; qap_name MLP 0.762 vs 0.72).  Next:
**Track A part 2** (PRETRAIN_APPLICATIONS, unblocked, starting
with element imputation) runs in parallel with **Track B** prep
(still gated on FC corpus pilots in
`~/.claude/plans/sunny-weaving-nebula.md`).

v2 numbers (cell-c) for benchmarking v3 against: T 14.8 °C MAE,
P 1.73 kbar, H2O 0.51 wt%; per-phase `R²_bulk` improvements over v1
on hydrous + foid-bearing phases (bi +0.18, ne +0.22, mu +0.23, amp
+0.08).

### Data flow

```
INPUTS (per sample)
  src  : [22]  atomic numbers, 0 = absent (padding)
  frac : [22]  values in [0, 1] (majors: wt%/100, traces: ppm/1e6)

  cation order (22 tokens; fixed):
    majors (9): Si Al Ca Mg Fe K Na Ti Cr
    traces (13): Sr Zr Ba Y Rb Ni Nb V La Ce Nd Yb Th

ENCODER (shared across all heads, frozen after SSL pretrain)
  element embedding (mat2vec) + frac projection -> per-token vector
  -> Transformer encoder (d_model / N / heads set per pretrain config, stamped in ckpt.config)
  -> masked-mean pool over non-padding tokens
  -> latent h : [B, d_model]

HEADS (parallel, all read the same h)
  ModeHead    : h -> Dirichlet alpha over ~18 phases       [B, 18]
  CompHead    : h -> per-phase composition                 [B, 18, K]
                K = 11 (oxide-wt%) OR per-phase endmember count
                (parameterization pending pilot 2b decision)
  CondHead    : h -> GMM per axis (T, P, H2O)              [B, 3, K_gmm, 3]
  ClassHead   : h -> petrogenetic supergroup logits (aux)  [B, 9]
  G_sys head  : h -> scalar regression (aux, optional)     [B, 1]
```

### Three corpora

1. **SSL pretrain**: `data/bulks/gard2019.csv` (Gard 2019, ~1M rows; ~474k after
   igneous filter).  Inputs only.  Drives the masked-element pretraining.
2. **Supervised (v3)**: per-bulk × N FC paths, generated via PetThermoTools.
   Each path produces (T_FC_end, P, H2O, modal-18, per-phase composition,
   optional G_sys).  ~22k bulks × 8 paths ≈ 178k rows.  Lands at
   `data/MAGEMin/fc_corpus.parquet` once the v3 regen runs.
3. **Held-out eval**: mineralML grain corpus (`data/grains/mineralml.csv`,
   88k grains) for CompHead cross-modal validation against real-rock
   chemistry.  Never used in training.

### Three injectable strategy modules

The corpus-gen pipeline encodes petrologic priors as toggle-able
strategies, not hardcoded constants.  Mirror this pattern for any
new prior.  All three survive into v3:

- `fe_redox.py` — Fe³⁺/ΣFe ratio (Middlemost 1989 per-class defaults;
  ClassRatioStrategy + ConstantRatioStrategy)
- `tp_priors.py` — (T, P) sampling (Putirka 2008 per-class ranges).
  The P axis carries forward into FC's (P, H2O) sampling; T sampling
  is replaced by per-bulk T_FC_end at which residual liq < 5%.
- `h2o.py` — H2O wt% (Wallace 2005 + Plank 2013 + Burnham 1979
  per-class ranges; ClassRangeH2OStrategy + ConstantH2OStrategy)

Tier-1 (measured) → use directly.  Tier-2 (assumed) → strategy
applies.  Provenance label carried through the pipeline so evaluation
can stratify by signal quality.

### The single most load-bearing code invariant

```
padding mask is derived from `src == 0` ONLY.
masked-for-SSL sets `src = MASK_TOKEN_ID` (= 119) and `frac = 0`.
masked-for-SSL is therefore DISTINGUISHABLE from padding by `src`,
and the encoder attends to MASK positions (it must, to predict).
```

This is **convention C** from the 2026-06-02 design discussion
(BERT-style mask sentinel).  The padding-mask line is the v2 bug
fix; the original v2 convention derived padding from `frac > 0`,
which collided with any masking that zeroed `frac` and silently
zeroed encoder gradients at masked positions for 8 epochs.  Own this
convention explicitly in the
v3 encoder — don't rely on a vendored padding heuristic.

## Architectural Decisions (don't re-litigate)

These are settled.  If you find yourself recommending a change to any
of them, address the rationale specifically rather than re-opening
the question:

- **CIPW norm dropped.** Deterministic calculator; ML surrogate adds
  no value.  pyrolite already does it.
- **ALIGNN structural stream deferred** (originally to v2, now to v4+).
  Position C data is already structured; ALIGNN's marginal value is
  low.  Reserve for the polymorph branch.
- **Single-scalar G_sys is NOT a polymorph disambiguator.**  Same
  bulk can have different polymorph identities with very similar
  G_sys.  G_sys is an encoder regularizer / auxiliary head only.
- **Hand-written stoichiometric rules kept** (from mineralML): used
  as (a) training-time endmember label generators, (b) inference-time
  post-processors for canonical naming, (c) evaluation-time ground
  truth.
- **Many-to-one in training data is intentional.**  Different (P, H2O)
  FC paths per bulk → different assemblages → multimodal posterior.
  The within-bulk variance IS the signal CondHead/ModeHead learn to
  represent, not noise to average out.
- **EDS-map data used for evaluation only, not training.**  Cross-modal
  validation anchor; preserves differentiation from mineralML.
- **qap_name locked as stratification axis** (IUGS Streckeisen QAPF).
  70 unique values, 83% non-null in `data/bulks/gard2019.csv`.
- **PetThermoTools (not custom Julia loop) for FC.**  See ADR 004.
- **Production label generation: FC via PetThermoTools at per-bulk
  T_FC_end.**  ADR 005 Resolved → Executed.  ~16 (P, H2O) paths per
  bulk, single endpoint per path.  fO2 stays class-conditional via
  `fe_redox.py`.
- **Bulk-level data splits, not row-level.**  Each bulk's FC paths
  share nearly identical inputs; all paths from one bulk must go to
  the same split or the model memorizes across train/val.
  64/16/20 stratified by `qap_name`; rare classes (<25 bulks) dropped.
- **SSL pretrain via masked-element modeling, Option H pooled-latent
  decoder.**  See [docs/FRESH_START.md](docs/FRESH_START.md).
- **Encoder is the durable contribution.**  Heads are application
  slots that can be re-supervised against any label source (MAGEMin,
  mineralML grains, EDS) without retraining the encoder.

## What NOT to Do

- Don't train a model to predict CIPW outputs (pyrolite already does it).
- Don't add ALIGNN to the supervised heads — deferred to v4+.
- Don't claim single-scalar G_sys is a polymorph disambiguator.
- Don't use mineralML as a runtime dependency at inference time —
  vendor the specific classifiers needed.
- Don't use mineralML's `Validation_Data/PetDB_allminerals/` as a
  training source (it's their test set; partial coverage).
- Don't enumerate the full C(31, k) combinatorial space; condition on
  rock class first.
- **Don't implement `NearSolidusTPStrategy`.**  Decided against by the
  eq-vs-frac v2 experiment (ADR 004).  Production uses FC, not
  near-solidus equilibrium.
- Don't propose a custom Julia loop for FC — we chose PetThermoTools
  (ADR 004).
- Don't use the v2 experiment's per-class T_FC_end_class as the
  production T directly.  Pick T per bulk based on the bulk's own
  liq trajectory; the class-constant approach causes ~11% audit
  attrition (JOURNAL 2026-05-15).
- **Don't pitch a centroid (deterministic) head as the model.**  That
  is Position B in different clothing (amortized MAGEMin marginalized
  over T, P, H2O).  Specifically rejected at project inception.  The
  model must include the CondHead for the inverse-inference
  contribution.
- **Don't use MSE as the head loss.**  Heads are probabilistic; use
  NLL (Dirichlet for ModeHead, GMM for CondHead).  MSE optimizes
  point estimates and ignores calibration.
- Don't log-transform T or P inputs to the CondHead.  Linear-space
  heteroscedastic Gaussian / GMM is sufficient; a negative T
  prediction is informative diagnostic, not a problem to hide.
- Don't do two-stage HPO (e.g., structural params first, loss
  weights second).  They interact non-separably; joint single-stage
  Optuna search across all dims (per ADR 006).
- **Don't include T, P, or H2O as model inputs.**  They're per-row
  supervision targets for CondHead; the model's forward graph only
  takes the 22-token cation+trace vector.

## Codebase Conventions

- **src layout** under `src/cnmineral/`.
- **Configs as YAML** in `configs/`; sole CLI flag is `--config PATH`.
- **Architectural decisions** (binding) → ADR in
  `docs/decisions/NNN_topic.md`.
- **Chronological research journal** → append to `docs/JOURNAL.md`
  (forward-chronological, Q / Context / Considered / Decision / Next
  template).
- **Living architecture docs**:
  - [docs/FRESH_START.md](docs/FRESH_START.md) — v3 architecture +
    value proposition (current).
  - [docs/V3_WORKFLOW.md](docs/V3_WORKFLOW.md) — v3 execution plan
    (current).
  - [docs/PRETRAIN_APPLICATIONS.md](docs/PRETRAIN_APPLICATIONS.md) —
    SSL encoder capability list (current).
  - [docs/methodology/](docs/methodology/) — meta tracking
    (technique index source + auto-generated index + design doc).
- **Element-properties CSVs** at `src/cnmineral/data/element_properties/`
  (the v3 encoder loads `mat2vec.csv` from here).
- **Generated artifacts gitignored**; regenerable via `scripts/`.

## macOS Gotcha

PyTorch and MAGEMin both link `libomp.dylib`.  Concurrent execution
aborts with "OpenMP already initialized" unless `KMP_DUPLICATE_LIB_OK=TRUE`
is set.  Set this at the top of any script that imports both torch and
PetThermoTools-using code:

```python
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

Symptom if it breaks: SIGABRT immediately after torch import.

## Sanity-Check References

External sources to consult before re-opening a design question:

- mineralML paper (Methods Section 2): `~/Downloads/mineralml_tex_eartharxiv_withsi.pdf`
- mineralML stoichiometry source: `~/git_clones/mineralML/src/mineralML/stoichiometry.py`
- Original CrabNet-Mineral kingcrab (v3 replaces this with own thin
  transformer, but the mat2vec embeddings and tokenization pattern
  come from here): `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/kingcrab.py`
- ALIGNN structure model (for v4+ reference):
  `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn/src/structure_model.py`
- PetThermoTools docs: https://petthermotools.readthedocs.io
- Optuna HPO reference (port when needed):
  `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn/src/objective.py`
  and `.../alignn/scripts/run_scaling.py`.  Storage convention:
  `JournalStorage` (filesystem), not SQLite.

## Documentation

- [docs/JOURNAL.md](docs/JOURNAL.md) — append-only chronological log
  of design discussions, observations, hypotheses tested, and decisions
  made.  The lab-notebook analog for the research process; v0/v1/v2
  history through 2026-06-01.
- [docs/FRESH_START.md](docs/FRESH_START.md) — v3 architecture +
  value proposition + the kingcrab decision + where G_sys fits.  Read
  this before any v3 design discussion.
- [docs/V3_WORKFLOW.md](docs/V3_WORKFLOW.md) — v3 execution plan:
  Track A (foundation encoder) and Track B (mineralogy heads), gate
  tables, ownership split for what you write vs what Claude writes,
  agentic-session patterns.
- [docs/PRETRAIN_APPLICATIONS.md](docs/PRETRAIN_APPLICATIONS.md) — 11
  capabilities the SSL-pretrained encoder enables beyond mineralogy
  (element imputation, k-NN retrieval, OOD detection, tectonic-class
  inference, etc.).
- [docs/decisions/](docs/decisions/) — ADRs.  001 scope, 002 rejected
  paths, 003 H2O strategy, 004 PetThermoTools for FC, 005 production
  FC label generation, 006 v1 model architecture.