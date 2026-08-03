# Decision 004: Use PetThermoTools for fractional-crystallization experiments and (potential) production label generation

**Date:** 2026-05-13
**Status:** Resolved → Phase B, **amended 2026-06-30 (production scoped to volcanic; equilibrium, not FC)**

## Amendment 2026-06-30 — v1 production is volcanic + equilibrium; FC deferred to the plutonic branch

The v1 corpus is scoped to the **volcanic + has-iron** subset of gard2019
(`docs/corpus_regen_spec.md`). For that scope the production generator is
**MAGEMin equilibrium cooled to near-solidus**, not fractional crystallization.
This *scopes* the Phase-B decision below; it does not overturn the eq-vs-frac
measurement.

The eq-vs-frac experiment (Resolution below) measured how closely near-solidus
equilibrium *reproduces FC endpoint labels* and found a large divergence
(median L1 0.35). That result stands. What changes is the target physics, not
the measurement. A volcanic rock crystallizes closed-system, in place: the melt
keeps its components (water included) and the assemblage that forms **is** the
equilibrium assemblage at its near-solidus conditions. FC models the
open-system case -- crystals settle out and the residual melt evolves -- which
is the plutonic/cumulate story. So for volcanic targets we are not approximating
FC with equilibrium; equilibrium is the correct generator, and the eq-vs-FC
divergence is the real difference between two petrogenetic regimes, not error.

Consequences:
- v1 production generator = the equilibrium near-solidus runner
  (`magemin/run_magemin.jl`), sampling `(P, H2O)` with T derived at a
  near-solidus liq threshold; `NearSolidusTPStrategy` stays unimplemented.
- FC (and PetThermoTools as its driver -- the tooling choice below is unchanged)
  moves to the deferred **plutonic/cumulate branch**, where it models FC of
  parent magmas (issue #22). ADR 005's FC production recipe applies there, not
  to the volcanic v1 corpus.

Rationale, locked decisions, and the composition-stratified equilibrium test
that validates the near-solidus assemblage behavior live in
`docs/corpus_regen_spec.md` and `explorations/archive/equilibrium_multimodality/`.

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
