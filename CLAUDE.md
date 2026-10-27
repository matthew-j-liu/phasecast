# phasecast

## What this is

"If I can generate the labels with MAGEMin, why do I need ML?" assumes the
model's job is to reproduce the generator's outputs.  Drop that framing.

**The model's job is to learn a representation of rock chemistry, supervised
through samples from the generator.**  The representation is the contribution:
the SSL-pretrained encoder organizes rock chemistry by petrogenetic structure
without ever seeing a label, and the supervised heads (ModeHead, CompHead,
CondHead, auxiliary ClassHead / G_sys) are application slots that could be
re-supervised against real-rock labels, or none at all.  What the heads realize
jointly is a probabilistic emulator of MAGEMin integrated over geologically
plausible (T, P, H2O) — bulk in, distribution over mineralogy and conditions
out, microsecond inference — which MAGEMin doesn't natively compute.

See [docs/encoder.md](docs/encoder.md) for the value proposition, the four-axis
differentiation, the data flow, and the corpora; [docs/heads.md](docs/heads.md)
for the supervised posterior.

## House Rules

**Use the `mineral` conda env when running any code, tests, or scripts in
this repo** (`conda run -n mineral ...` or activate first).

**Run the unit tests with** `pytest tests/`.  Single-threaded — cleaner stdout /
tracebacks beats any speed win from parallel workers at this suite size.

## Test execution

- Don't run the full suite by default — it's friction on every change.  Assume
  tests are green; run them when explicitly asked or when debugging a specific
  failure.
- **When debugging, iterate on the failed subset only** — `pytest
  tests/test_foo.py::test_bar` for a single regression, or `pytest --lf`
  (last-failed) after a round of fixes.  Re-run the full suite once at the end.
- **Profile new or touched tests before checkin** (`pytest <touched_tests>`).
  Code should run as fast as is reasonable — vectorize where possible.

## After code changes

1. Add tests when the change warrants them (new behavior, regression, non-trivial
   edge case).  Check first whether existing tests cover it implicitly — when a
   change forces old fixtures to exercise the new contract, that IS the coverage.
   **Whenever a test is added, deleted, renamed, or repurposed, update
   [tests/TESTS.md](tests/TESTS.md) in the same change** (the per-test inventory).
2. Use discretion on code review: `/review` for non-trivial changes, skip for
   mechanical edits; when uncertain, review.  Address easy fixes; surface major
   issues before acting on them.
3. If the change touches multiple responsibility areas and the coupling seems too
   tight, flag it — don't fold a refactor into the current diff.
4. **Methodology index:** any change that introduces, renames, or removes an ML
   technique (loss, head, training strategy, eval, preprocessing step) updates
   [docs/methodology/techniques.yaml](docs/methodology/techniques.yaml) with
   distinctive symbol patterns and regenerates in the same change: `conda run -n
   mineral python scripts/build/build_technique_index.py` (verify `zero hits:
   none`).
5. **Overview deck:** after a major architecture change — anything that would
   make an existing slide *wrong* (a new/changed model component, loss, training
   stage, tokenization, data corpus, or how the pieces compose) — update and
   regenerate [slides/overview.py](slides/overview.py) in the same change
   (`conda run -n mineral python slides/overview.py`).  A **new ML
   component** (head, encoder variant) additionally gets the standard
   two-slide report — anatomy (`head_anatomy_slide` template) + eval-semantics
   (worked example) — covering the 8-item checklist in the
   `slides/overview.py` docstring; param counts pulled from the checkpoint,
   never estimated.

## How to work

- **Consult the design docs first** — before proposing an architectural change in
  an area with a `docs/*.md`, and whenever something "smells off."  That reaction
  is often the signal that a decision has hidden load-bearing justification (the
  thing that looks like a bug is the fix for a subtler one).  If your
  recommendation contradicts a recorded rationale, engage with it specifically;
  don't re-litigate.
- **Audits update the codebase as they go.**  The deliverable is a codebase that
  doesn't need the audit document to be understood.

## Writing code

**Read [docs/code_style.md](docs/code_style.md) before writing or modifying
code** — it owns the comment/docstring durability rules, Schema Design
Discipline, the frozen-dataclass conventions, imports, and the "production
classes serve callers, not tests" rule.  The non-negotiables, inline:

**Never add `from __future__ import annotations` to any file in this repo.**
Quote individual forward references as strings; disabling runtime evaluation for
a whole module causes silent type-hint failures in anything that calls
`get_type_hints()` (Pydantic, FastAPI, dataclass introspection).

**A frozen schema property MUST NOT hand callers a shared mutable container.**
Returning a mutable object through a `@property` / `@cached_property` on a frozen
class lets one caller's local mutation silently corrupt every later reader, with
no exception raised.  Return an immutable value (`tuple`, `frozenset`, scalar) or
cache a private `_name` template and return `self._name.copy()`.  Never offload
"copy before mutating" onto call sites.  Full rationale in
[docs/code_style.md](docs/code_style.md).

**Schema-shape changes go through Schema Design Discipline first** (the six
principles in [docs/code_style.md](docs/code_style.md)): before adding a stamped
field, index, discriminator, or schema dataclass, score it — parsimony on the
way in, aggressive deletion on the way out.

**When renaming a function, class, or module, update every reference in the same
change**: comments, docstrings, cross-references in other modules' prose, test
names/docstrings, and docs.  Grep for the old name repo-wide before declaring the
rename done.

**Name files, tests, docs, and comments by what they cover, not when they
landed.**  PR numbers, plan steps, issue numbers, migration phases are
anti-patterns — names should reflect the *responsibility*: what subsystem, which
contract, which invariant.  If new content fits an existing file's scope, merge it
in; don't create a new file to keep the diff tidy.

**Don't leak internal planning code-names into code or user-facing output.**
`Phase 1`, `Scenario A/B`, `cell-c`, `step 2b` are fine in planning docs,
journal, issues, and conversation — but code, log lines, CLI output, metric keys,
and docstrings must name the actual assumption and conclusion.  Spell out the
criterion, measured values, and resulting action — not `-> Scenario A` but
`latent keeps 94% of the input ceiling -> recover traces with a read-out head, no
re-pretrain`.  Agree decision terminology (verdict strings, metric keys) before
committing it to code.

**Reference code by symbol, not line number.**  Links like `file.py#L112` rot;
use the function / class / variable name and let the reader grep.

**Don't leave dead code "because it's harmless."**  Unused fields, functions, or
branches are hand grenades — git has it if you need it back.

**Production code is not an experiment's configuration surface.**  An
experiment that needs a variant of a production step (a different sampler, a
truncated loader, an extra output, a selection rule) writes the variant inside
its exploration, duplicating a few dozen lines if it must; it never adds a
flag, branch, optional parameter, hook or config key to `src/` or `scripts/`
whose only caller is that experiment.  A variant that becomes the production
choice replaces the step it beat, in the same change, rather than sitting
beside it behind a switch.  Every branch in production serves a caller that
exists today.  `explorations/README.md` has the operational version and
`docs/code_style.md` the sibling rule for tests.

**Planning documents stay prospective.**  A plan answers "what does someone need
to know to finish these steps?" — nothing else.  As a step lands, "how it works
now" migrates to permanent docs; the landed step collapses to a terse pointer.
Delete the plan once every step has landed.

## The load-bearing code invariant

```
padding mask is derived from `src == 0` ONLY.
masked-for-SSL sets `src = MASK_TOKEN_ID` (= 119) and `frac = 0`,
so it is DISTINGUISHABLE from padding by `src` and the encoder attends
to MASK positions (it must, to predict).
```

This is convention C (BERT-style mask sentinel).  The v2 bug was deriving padding
from `frac > 0`, which collided with any masking that zeroed `frac` and silently
zeroed encoder gradients at masked positions.  Own this explicitly; don't rely on
a vendored padding heuristic.  Full story in [docs/encoder.md](docs/encoder.md).

## Terminology (locked 2026-07-16)

- **bulk** — one gard2019 composition; the MAGEMin input; the unit of
  splitting.  "Rock" is its informal synonym in slides/prose, never the name
  for an output.
- **draw** — one sampled (P, H2O) path = one MAGEMin cooling run (24/bulk
  train, 128/bulk valtest).
- **rung** — the melt-fraction depth a draw is read out at (50/30).
- **row** — one (bulk, draw, rung): the atomic training example / label.
- **assemblage** — a row's mineral content (phases + modes).
- **scenario** — RESERVED for the model's posterior clusters at inference
  (heads.md View 2); never a corpus row.

## Settled decisions (don't re-litigate)

If you find yourself recommending a change, address the rationale specifically —
don't re-open the question.

- **CIPW norm dropped.** Deterministic; pyrolite already does it.  Don't train a
  model to predict CIPW outputs.
- **ALIGNN structural stream deferred to v4+.**  Position C data is already
  structured; marginal value is low.  Don't add ALIGNN to the supervised heads;
  reserve it for the polymorph branch.
- **Single-scalar G_sys is NOT a polymorph disambiguator** (same bulk, different
  polymorphs, near-identical G_sys).  It's an encoder regularizer / auxiliary head
  only — don't pitch it as a disambiguator.
- **Heads are probabilistic — don't use MSE as the head loss.**  NLL (Dirichlet
  for ModeHead, GMM for CondHead); MSE optimizes point estimates and ignores
  calibration.
- **Don't include T, P, or H2O as model inputs.**  They're per-row supervision
  targets for CondHead; the forward graph takes only the 22-token cation+trace
  vector.  And don't log-transform T or P — linear-space heteroscedastic
  Gaussian/GMM is sufficient; a negative prediction is informative diagnostic, not
  a problem to hide.
- **Don't pitch a centroid (deterministic) head as the model.**  That's amortized
  MAGEMin marginalized over T, P, H2O — rejected at inception.  The model must
  include CondHead for the inverse-inference contribution.
- **Fractional crystallization drives MAGEMin_C directly through
  `explorations/generator_reach/run_fractional.jl`** (ADR 004, amended
  2026-10-09): MAGEMinCalc's scheme reimplemented in the production Julia
  environment with status per step, a stop at melt-out and the equilibrium
  runner's timeout and bisection; PetThermoTools stays installed as the
  reference the loop is checked against, never as the driver.  Production
  labels: FC at per-bulk T_FC_end (ADR 005), ~16 (P, H2O) paths per bulk, single
  endpoint each; fO2 stays class-conditional via `fe_redox.py`.  Pick T per bulk
  from its own liq trajectory, not a per-class T_FC_end_class (that caused ~11%
  audit attrition).  Don't implement `NearSolidusTPStrategy` — rejected by the
  eq-vs-frac experiment (ADR 004); production uses FC, not near-solidus eq.
  The volcanic v1 corpus does not use this recipe: ADR 004's 2026-06-30
  amendment (generation record in the ADR) generates it by equilibrium on a
  cooling ladder read at melt-fraction rungs, run directly through
  `magemin/run_magemin.jl` in a Julia subprocess with no PetThermoTools in the
  stack; the FC recipe is deferred to the plutonic branch (#22).
- **Many-to-one training data is intentional.**  Different (P, H2O) FC paths per
  bulk → different assemblages → multimodal posterior; the within-bulk variance IS
  the signal CondHead/ModeHead learn, not noise to average out.
- **ModeHead vocabulary locked: 21 solid phases** (2026-07-15,
  `cnmineral.data.phase_vocab`). trd/crst merge into q (SiO2 polymorphs,
  large mass when present); chl + fper dropped (<0.1% presence floor);
  and/ky/sill + cor kept; fsp split into plag/afs via the vendored mineralML
  feldspar rule (miscibility-gap compositions route by dominant component).
  liq/fl are not vocabulary -- the label is the crystals-renormalized solid
  assemblage. Other multi-instance phases sum modes.
- **CondHead conditions on the presence support** (chain rule, locked
  2026-07-15): teacher-forced true mask at train, ModeHead-sampled at
  inference. Independent heads assume a conditional independence the corpus
  violates by design; the scenario view requires the coupling. Don't feed the
  support into the encoder -- it's a CondHead input only.
- **CompHead is a per-phase Dirichlet over oxide wt%** (locked 2026-07-15;
  eps-clipped targets). Per-oxide heteroscedastic Gaussians rejected: samples
  that violate the simplex are parameterization artifacts, not diagnostics --
  the OOD diagnostic lives in the cross-head closure residual, which survives
  either choice.
- **Bulk-level splits, not row-level.**  A bulk's FC paths share nearly identical
  inputs; all paths go to the same split or the model memorizes across train/val.
  64/16/20 stratified by `qap_name`; rare classes (<25 bulks) dropped.  Two-tier
  mapping (locked 2026-07-16): all train-tier bulks train; the valtest bulks
  divide 16:20 into val (early stopping, tuning, interrogation figures) / test
  (sealed until final numbers).  The valtest roster grows by extension only
  (`build_valtest_split.py --extend`) so committed assignments never move: 500
  bulks -> 2000 on 2026-07-26, 1999 assigned 887 val / 1112 test.
- **qap_name locked as stratification axis** (IUGS Streckeisen QAPF; 70 values,
  83% non-null in `data/bulks/gard2019.csv`).
- **Input chemistry is not renormalized to a common analytical total** (#56,
  closed 2026-09-02).  `tokenize_bulk_frame` passes each row through on the
  total its source reports: gard2019 rows are raw analyses, du Bray's
  benchmark tables are recalculated volatile-free to 100, and the encoder can
  see the ~2% scale difference.  Undoing it measured second-order (zero-shot
  interval not clearing zero) and cannot be done for every rock (MSH and
  Pinatubo carry no raw total), so the benchmark stays as published.  That
  was acceptable here; every new data source needs its total convention
  checked against the corpus before it is scored, and any renormalization
  applied to every row or none.
- **The condition prior is not the lever for the phenocryst ruler** (#79,
  closed 2026-09-18).  For single-condition equilibrium scored against du
  Bray's phenocryst counts, no (P, H2O) prior, global or conditioned on the
  major oxides, moves the four-silicate labels by more than 0.03 in share;
  38% of the evaluated rocks have no single condition that reproduces all
  four silicates together; and the heads already score within 0.01 of a
  per-rock oracle prior.  Don't re-open pressure or water as the fix for
  the volcanic pair.  A scaling pair changes the ruler (plutonic whole-rock
  modes at a solidus rung) or the recipe (fractional crystallization);
  `explorations/prior_search/README.md`, journal 2026-09-18.
- **Hand-written stoichiometric rules kept** (from mineralML): training-time
  endmember label generators, inference-time canonical naming, eval-time ground
  truth.  But don't depend on mineralML at runtime — vendor the specific
  classifiers; and don't train on its `Validation_Data/PetDB_allminerals/` (their
  test set, partial coverage).
- **EDS-map data is evaluation-only, not training.**  Cross-modal validation
  anchor; preserves differentiation from mineralML.
- **Don't enumerate the full C(31, k) combinatorial space** — condition on rock
  class first.
- **Joint single-stage Optuna HPO**, not two-stage (structural then loss weights);
  they interact non-separably (ADR 006).
- **SSL pretrain via masked-element modeling, Option H pooled-latent decoder**
  (see [docs/encoder.md](docs/encoder.md)).
- **Encoder is the durable contribution.**  Heads are application slots,
  re-supervisable against any label source (MAGEMin, mineralML grains, EDS)
  without retraining the encoder.

## Task tracking

**Task tracking lives in GitHub Issues**
(https://github.com/matthew-j-liu/phasecast), not a static doc.  Create
with `gh issue create`, applying one label from each of `P{1,2,3}-*`, `type: *`,
`area: *` (plus `status: *` or `epic` where it applies; `gh label list`
enumerates valid values).

## Living docs

- **src layout** under `src/cnmineral/`; **configs as YAML** in `configs/`, sole
  CLI flag `--config PATH`; **element-properties CSVs** at
  `src/cnmineral/data/element_properties/` (encoder loads `mat2vec.csv`);
  **generated artifacts** split by role, regenerable via `scripts/`:
  `runs/` = transient instrument snapshots, gitignored, overwritten as
  iteration continues; `results/` = per-version headline summaries
  (model-version ledger, trajectory, per-model headline figures; see
  `results/README.md`), tracked.
  (`explorations/*/results/` stay tracked -- exploration conclusions.)
- [literature/](literature/) — manuscript bibliography: `references.bib` (the
  BibTeX the paper compiles) and [claims.md](literature/claims.md) (claim →
  citation map, roles, do-not-cite warnings, verification status).  The PDFs in
  that directory stay gitignored; the two text files are tracked.
- [docs/encoder.md](docs/encoder.md) — foundation encoder, value proposition,
  four-axis differentiation, tokenization/data-flow, the three corpora, SSL
  pretrain mechanic, and the load-bearing code invariant; then the
  representation scorecard (living, provisional rubric for latent strength,
  one anchored axis per eval, re-scored each encoder cascade) and the 11
  capabilities beyond mineralogy (imputation, k-NN, OOD, etc.).
- [docs/heads.md](docs/heads.md) — the supervised heads + multimodal posterior
  (hurdle ModeHead, CompHead, CondHead, mass-balance, output views, eval).
- [docs/code_style.md](docs/code_style.md) — code conventions and schema
  discipline.
- [docs/journal.md](docs/journal.md) — append-only chronological research log
  (Q / Context / Considered / Decision / Next); v0/v1/v2 history.  Written
  impersonally: decisions and the evidence for them, no attribution, schedule
  or deliberation narrative (the entries before 2026-10-27 were rewritten once
  to this standard; otherwise the log is append-only).
- [docs/decisions.md](docs/decisions.md) — ADRs as numbered sections (001
  scope, 002 rejected paths, 003 H2O strategy, 004 PetThermoTools for FC, 005
  production FC labels, 006 v1 model architecture, 007 SSL pretrain objective,
  008 H2O target leak).
- [docs/methodology/](docs/methodology/) — technique index source +
  auto-generated index + design doc.
- [tests/TESTS.md](tests/TESTS.md) — per-test inventory.

## Sanity-Check References

External sources to consult before re-opening a design question:

- mineralML paper (Methods Section 2): https://eartharxiv.org/repository/view/12356/
- mineralML stoichiometry source: https://github.com/sarahshi/mineralML
  (`src/mineralML/stoichiometry.py`)
- CrabNet (v3 replaces its transformer with our own thin one, but the mat2vec
  embeddings and tokenization pattern come from here):
  https://github.com/sparks-baird/CrabNet
- ALIGNN structure model (for v4+ reference): https://github.com/usnistgov/alignn
- PetThermoTools docs: https://petthermotools.readthedocs.io
- Optuna HPO storage convention: `JournalStorage` (filesystem), not SQLite.

## macOS Gotcha

Several packages in the env ship their own OpenMP runtime (torch bundles a
`libomp.dylib`, conda's `llvm-openmp` provides another, xgboost links its own).
Loading two in one process aborts with "OpenMP already initialized" unless
`KMP_DUPLICATE_LIB_OK=TRUE` is set.  `cnmineral/__init__.py` and
`tests/conftest.py` set it on import; set it at the top of any standalone
script that imports torch before `cnmineral`:

```python
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

Symptom if it breaks: SIGABRT immediately after torch import.  The collision
was first hit through PetThermoTools' in-process MAGEMin, which is no longer in
the stack; the corpus generator runs MAGEMin in a Julia subprocess and is
unaffected.
