# Code style and schema discipline

Project-specific conventions for writing and modifying code in this repo. The
root `CLAUDE.md` points here before any code change; the two sharpest rules
(the `from __future__ import annotations` ban and a trimmed version of the
frozen-mutable-container rule) also live inline in `CLAUDE.md`, with the full
rationale below.

## Comments and docstrings

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

- `docs/journal.md` -- chronologically anchored by design; "on
  2026-06-11 we compared 20 vs 40 epochs and got X" reads
  correctly here forever because it's explicitly a log of what
  happened when.
- `docs/decisions/NNN_*.md` -- ADRs for binding architectural
  choices with rationale.
- `docs/roadmap.md` Track-status sections -- explicitly
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

## Imports

**Imports go at the top of the file.**  Inline imports only for (1)
per-environment optional dependencies or (2) genuine circular imports
(verify by moving to the top first).

## Schema Design Discipline

**Schema-shape changes go through Schema Design Discipline first.**
Before adding a stamped metadata field, a new index, a new
discriminator, or a new schema dataclass, score the proposed
change against the following rules:

1. **Every field earns its place against reconstruction.** Can this field be reconstructed from inputs + dependency outputs at query time? If yes, don't stamp. The same test applies to existing fields: if a stamped field is
reconstructible with no information loss and no current consumer needs it, drop it.  Carrying speculative metadata is unbounded; rebuilding when a real consumer surfaces is bounded.

2. **Schema evolution is asymmetric: adding is cheap, removing is expensive.** Adding a field is one line, but removing it requires updating every consumer, every test fixture, every doc, plus (with production data) a migration script. *Parsimony on the way in; aggressive deletion on the way out.*

3. **Strong domain models displace defensive coding.** When ``if X is None or X == "default" or ...`` shows up in multiple places, the data model wants to be tightened. *The data model owns invariants; the code reads them.*

4. **Beware speculative future-proofing.** Designs that solve future problems carry real maintenance cost in
the present. Defaults should be parsimony; future-proofing earns its keep only when the future arrives.

5. **Discipline pays compound interest.** Tests, types, responsibility-named symbols, and clear docs are
cheap relative to scary changes; expensive relative to first-write deadlines. Pay the cost early.

6. **Naming reveals (or hides) confusion.** When a name feels awkward, the schema is telling you something. Whenever a name feels off, lean into the discomfort: good naming forces clarity about what each value means at each layer; bad naming masks confusion that compounds across boundaries.

## Production classes serve callers, not tests

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

## Frozen dataclasses

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

## Don't alias cheap reads

**Don't alias cheap property or dataclass-field reads to locals.**
`x = obj.cheap_property` used fewer than ~5 times in close succession
is clutter, not caching.
