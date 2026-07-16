# crabnet-mineral

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

**Use the `alignn_fix` conda env when running any code, tests, or scripts in
this repo** (`conda run -n alignn_fix ...` or activate first) — unless an
untracked `CLAUDE.local.md` at repo root overrides the env name.  That override
exists so sibling clones of this repo can install their own editable copy
without colliding on `pip install -e .`.

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
   alignn_fix python scripts/build/build_technique_index.py` (verify `zero hits:
   none`).
5. **Overview deck:** after a major architecture change — anything that would
   make an existing slide *wrong* (a new/changed model component, loss, training
   stage, tokenization, data corpus, or how the pieces compose) — update and
   regenerate [slides/overview.py](slides/overview.py) in the same change
   (`conda run -n alignn_fix python slides/overview.py`).  A **new ML
   component** (head, encoder variant) additionally gets the standard
   two-slide report — anatomy (`head_anatomy_slide` template) + eval-semantics
   (worked example) — covering the 8-item checklist in the
   `slides/overview.py` docstring; param counts pulled from the checkpoint,
   never estimated.

Before declaring a change done, read the diff as a reviewer would — fresh
skepticism, not the author's confidence.

## How to work

- **Ask, don't assume.**  When intent, architecture, or requirements are unclear,
  ask before writing code.  A silent assumption that turns out wrong costs more
  than the question did.
- **Flag uncertainty explicitly.**  If you're not confident in an approach or a
  technical detail, say so before proceeding — confidence without certainty does
  more damage than naming the gap.
- **Leave unrelated code alone.**  If a file or function isn't part of the task,
  don't modify it even if you spot an improvement — note it separately.
- **Propose better approaches.**  Raise a cleaner or more durable way of doing
  something (especially one with lasting impact) rather than silently taking the
  obvious path.
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
re-pretrain`.  Confirm decision terminology (verdict strings, metric keys) with
the owner before committing it to code.

**Reference code by symbol, not line number.**  Links like `file.py#L112` rot;
use the function / class / variable name and let the reader grep.

**Don't leave dead code "because it's harmless."**  Unused fields, functions, or
branches are hand grenades — git has it if you need it back.

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
- **PetThermoTools (not a custom Julia loop) for FC** (ADR 004).  Production
  labels: FC at per-bulk T_FC_end (ADR 005), ~16 (P, H2O) paths per bulk, single
  endpoint each; fO2 stays class-conditional via `fe_redox.py`.  Pick T per bulk
  from its own liq trajectory, not a per-class T_FC_end_class (that caused ~11%
  audit attrition).  Don't implement `NearSolidusTPStrategy` — rejected by the
  eq-vs-frac experiment (ADR 004); production uses FC, not near-solidus eq.
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
  mapping (locked 2026-07-16): all 10k train-tier bulks train; the 500 valtest
  bulks divide 16:20 into ~222 val (early stopping, tuning, interrogation
  figures) / ~278 test (sealed until final numbers).
- **qap_name locked as stratification axis** (IUGS Streckeisen QAPF; 70 values,
  83% non-null in `data/bulks/gard2019.csv`).
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
(https://github.com/matthew-j-liu/CrabNet-Mineral-v2), not a static doc.  Create
with `gh issue create`, applying one label from each of `P{1,2,3}-*`, `type: *`,
`area: *` (plus `status: *` or `epic` where it applies; `gh label list`
enumerates valid values).  Link commits and PRs via `Fixes #N` / `Closes #N` —
GitHub auto-closes on merge.

## Living docs

- **src layout** under `src/cnmineral/`; **configs as YAML** in `configs/`, sole
  CLI flag `--config PATH`; **element-properties CSVs** at
  `src/cnmineral/data/element_properties/` (encoder loads `mat2vec.csv`);
  **generated artifacts gitignored**, regenerable via `scripts/`.
- [docs/encoder.md](docs/encoder.md) — foundation encoder, value proposition,
  four-axis differentiation, tokenization/data-flow, the three corpora, SSL
  pretrain mechanic, and the load-bearing code invariant.
- [docs/heads.md](docs/heads.md) — the supervised heads + multimodal posterior
  (hurdle ModeHead, CompHead, CondHead, mass-balance, output views, eval).
- [docs/roadmap.md](docs/roadmap.md) — execution plan: Track A (foundation
  encoder, closed) / Track B (mineralogy heads) / Track C, gate tables, ownership
  split, current track-status.
- [docs/pretrain_applications.md](docs/pretrain_applications.md) — the 11
  SSL-encoder capabilities beyond mineralogy (imputation, k-NN, OOD, etc.).
- [docs/scorecard.md](docs/scorecard.md) — living, provisional rubric for latent
  representation strength; one anchored axis per eval, status candidate/
  provisional/locked, re-scored each encoder cascade.
- [docs/code_style.md](docs/code_style.md) — code conventions and schema
  discipline.
- [docs/journal.md](docs/journal.md) — append-only chronological research log
  (Q / Context / Considered / Decision / Next); v0/v1/v2 history.
- [docs/decisions/](docs/decisions/) — ADRs (001 scope, 002 rejected paths, 003
  H2O strategy, 004 PetThermoTools for FC, 005 production FC labels, 006 v1 model
  architecture, 007 SSL pretrain objective, 008 H2O target leak).
- [docs/methodology/](docs/methodology/) — technique index source +
  auto-generated index + design doc.
- [tests/TESTS.md](tests/TESTS.md) — per-test inventory.

## Sanity-Check References

External sources to consult before re-opening a design question:

- mineralML paper (Methods Section 2): `~/Downloads/mineralml_tex_eartharxiv_withsi.pdf`
- mineralML stoichiometry source: `~/git_clones/mineralML/src/mineralML/stoichiometry.py`
- Original CrabNet-Mineral kingcrab (v3 replaces this with own thin transformer,
  but the mat2vec embeddings and tokenization pattern come from here):
  `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/kingcrab.py`
- ALIGNN structure model (for v4+ reference):
  `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn/src/structure_model.py`
- PetThermoTools docs: https://petthermotools.readthedocs.io
- Optuna HPO reference (port when needed):
  `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn/src/objective.py` and
  `.../alignn/scripts/run_scaling.py`.  Storage convention: `JournalStorage`
  (filesystem), not SQLite.

## macOS Gotcha

PyTorch and MAGEMin both link `libomp.dylib`.  Concurrent execution aborts with
"OpenMP already initialized" unless `KMP_DUPLICATE_LIB_OK=TRUE` is set.  Set this
at the top of any script that imports both torch and PetThermoTools-using code:

```python
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

Symptom if it breaks: SIGABRT immediately after torch import.
