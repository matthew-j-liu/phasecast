# volcanic_proxy_definition

**Status: open design synthesis (no experiment run yet).** This is the foundational
question under the whole Track B corpus: *what physical state should a MAGEMin
simulation represent, so that it is a faithful, invertible proxy for the erupted
volcanic rock a geologist actually measures?* Every corpus knob (pressure, water,
where we stop cooling, equilibrium vs fractional, bulk vs melt) is a facet of it.

This doc synthesizes two independent high-reasoning expert reviews (ChatGPT and
Gemini, on the brief in this repo's history) against the project's own context.
It is discussion feeding `docs/corpus_regen_spec.md`; it does not itself lock
anything.

## The question in one breath

A volcanic rock's real history: magma equilibrates in the crust at some
(P, T, H2O) -> ascends -> decompresses (P -> ~1 bar) -> degasses (dissolved H2O
exsolves) -> crystallizes groundmass -> quenches, freezing residual melt to glass.
The measured rock is a *mixture of records*: phenocrysts (deep, hydrous) +
groundmass (shallow, degassed) + glass. Our current generator does one thing:
sample (P, H2O), cool in equilibrium at that single pressure to a near-solidus
endpoint (residual melt < 1-5 wt%), read the assemblage. So: what is that
single-pressure, near-solidus, equilibrium snapshot actually a good proxy for, and
where does it lie?

## What both expert reviews converged on (treat as well-grounded)

1. **A flat 0-8 wt% H2O box is physically impossible.** It must be bounded by
   pressure-dependent water solubility (a melt cannot hold 8 wt% dissolved H2O at
   shallow depth). Both reviews call this the single most important fix.
2. **Sampled water = deep pre-eruptive dissolved H2O**, not erupted water. Erupted
   subaerial lavas have degassed to <0.5 wt% in the matrix glass; the deep water is
   physically lost, but its *signature* is preserved in the bulk major/trace
   variations (delayed plagioclase saturation, early Fe-Ti oxide / amphibole).
3. **Target storage (deep) pressure.** That is where the recoverable petrogenetic
   signal lives; all thermobarometry targets storage conditions, not eruptive ones.
4. **Whole-rock bulk approximate to liquid breaks down for crystal-rich rocks**
   (porphyritic dacites/rhyolites; accumulative picrites/ankaramites). Confidence
   should be stratified by crystallinity: aphyric lavas high, porphyritic lower,
   cumulates lowest.

Both also flag that the **residual-liquid = glass** equivalence is weakest exactly
at our near-solidus, silicic end: thermodynamic melt models get unreliable at low
melt fraction in polymerized (high-SiO2) systems (unphysical alkali enrichment,
oversaturation). So "residual liq = glass" is convenient but shakiest where we cut.

## Where they conflict: the near-solidus endpoint (the crux)

- **ChatGPT: near-solidus is defensible.** It is an idealized *equilibrium
  reference state*; cooling at fixed P is not "plutonic," it is a reference
  abstraction like every thermobarometer, not a literal history. Single-stage
  equilibrium is the *identifiable* choice, because fractional crystallization
  introduces extraction efficiency, recharge, mixing, settling, residence time --
  none observable from bulk chemistry. Its only major fix is the water prior.
- **Gemini: near-solidus is a paradox -- "you are generating plutonic rocks to
  train on lavas."** Two physical objections: (a) real lavas erupt near the
  rheological lock-up window (~45-60 vol% crystals; Marsh 1981, Scaillet 1998),
  not at 1-5% melt, so near-solidus over-crystallizes and invents low-T phases the
  lava never grew; (b) cooling to near-solidus runs *peritectic reactions* that
  alter or destroy the phenocrysts that record deep conditions (real rocks preserve
  them precisely because rapid ascent is a kinetic quench; Blundy & Cashman 2005).
  Its fix is a **two-stage** model: equilibrium at depth to ~10-40% crystallinity,
  freeze the phenocrysts, then fractionally crystallize / quench the residual at
  1 bar.

### Adjudication

Gemini is more right on the *physics*, ChatGPT on what is *learnable*, and the
resolution depends on what ModeHead's label is validated against:

- If the label must match **what a geologist counts** (a real lava is ~40%
  phenocrysts + ~60% groundmass/glass), near-solidus (95% crystalline) does not
  look like that rock -- Gemini's critique has teeth, and `heads.md` does intend
  eventual validation against real mineralML grains + EDS maps.
- If the label is an abstract **reference state for inverting conditions**, the
  assemblage *identity* carries the signal and near-solidus is a defensible,
  identifiable idealization -- ChatGPT.

Gemini overstates by calling it "granites and gabbros" (that conflates mineral
identity with rock classification), and its two-stage fix reintroduces exactly the
unconstrainable eruption-dynamics parameters (ascent rate, freeze point, degassing
path) ChatGPT says bulk chemistry cannot pin -- and reintroduces fractional
crystallization, which ADR 004 deferred.

**The sweet spot neither fully states:** stop at the **eruptible-crystallinity
window (~50% melt)** -- Gemini's physically honest endpoint -- but keep it
**single-stage equilibrium** -- ChatGPT's identifiable kinetics. Gemini's endpoint
with ChatGPT's simplicity: realistic lava modes without the unconstrainable
two-stage parameters.

## What our project context changes (neither review knew this)

1. **We already stored the entire cooling ladder for every draw** (raw JSONL has
   the assemblage at every T from 1200 C down, not just the endpoint). So switching
   the endpoint criterion from "1-5% melt" to "~50% melt (eruptible)" is a **cheap
   re-analysis of existing data, not a re-run.** We can test Gemini's claim
   directly: re-pick every draw at a crystallinity ladder and see whether the
   amp/opx multimodality survives and how the modes change. Philosophical dispute
   -> one-afternoon experiment. This is the proposed next step (below).
2. **The encoder is trained separately and frozen; only the heads consume these
   labels.** The label proxy is cheap to iterate -- regenerate labels, retrain
   heads, without touching the representation. Lower stakes on getting the proxy
   perfect now; argues for shipping a defensible v1 with a *revisable* endpoint.
3. **The water fix reconciles with locked item 1 (prior-agnostic sampling), it
   does not overturn it.** A solubility bound is a physical-possibility constraint,
   not a geological prior. Reconciliation: sample broadly/uninformatively *within
   the physically possible (P, H2O) region*, not a rectangle with impossible
   corners. And this makes the deep-wet-amphibole / shallow-dry-opx result *more*
   defensible (water becomes physically structured), not less -- the opposite of
   the F4 worry.
4. **Our amp/opx result probably survives an eruptible-crystallinity endpoint**,
   because the wet -> amphibole / dry -> pyroxene split is a real *phenocryst*
   phenomenon (amphibole is a legitimate wet-arc phenocryst), not only a
   near-solidus one -- but verify it on the existing data rather than assume.

## Key considerations, per axis

| Axis | Current | Candioti 2026 | ChatGPT | Gemini | Proposed v1 | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| Pressure | single deep P, to solidus | free 0.1-10 kbar (forward) | storage P (reference state) | two-stage deep->1 bar | **storage P, single-stage** | high |
| Endpoint / T | near-solidus (1-5% melt) | free T 650-1300 C | near-solidus OK as reference | eruptible ~50% crystallinity | **OPEN -- test the re-pick** | low |
| Water | flat 0-8 wt% | fixed 4 wt% | P-conditioned, solubility-bounded, deep | P-bounded deep, ->degassed | **solubility-bounded, deep** | high |
| Kinetics | equilibrium | equilibrium | equilibrium (identifiable) | hybrid EC+FC | **equilibrium, single-stage** | med-high |
| Input comp | whole-rock bulk | whole-rock volcanic | bulk, confidence-tiered | partition porphyritic | **bulk + crystallinity flag** | med |
| Glass | residual liq | validated vs expt glass | liq approx glass, weak near-solidus | unreliable in silicic near-solidus | **liq-as-glass, flag silicic** | med |

## Conclusions (what I would propose for a defensible v1)

- **Adopt now (both reviews agree, cheap, high confidence):**
  1. Replace the flat water box with **pressure-dependent solubility-bounded
     sampling of deep pre-eruptive H2O** (Newman & Lowenstern 2002 / Plank 2013).
     This is the highest-value single change and reconciles with item 1.
  2. Keep **equilibrium, single-stage** (identifiability; do not reopen FC / ADR 004).
  3. Keep **storage pressure** as the target (do not model surface groundmass).
  4. Keep **whole-rock bulk** input but **carry a crystallinity/porphyricity flag**
     so downstream can down-weight or exclude crystal-rich rocks; the bulk-as-liquid
     assumption is strongest for aphyric mafic-intermediate.
  5. **Frame the labels honestly as an equilibrium reference state** (ChatGPT's
     Methods wording), not as a literal simulation of eruption.
- **The one genuinely open architectural decision: the endpoint.** near-solidus
  (reference state, identifiable, but modes do not match real lavas + liq-as-glass
  weakest here) vs eruptible-crystallinity (~50% melt, realistic lava modes, but T
  becomes a bounded/derived-at-higher-melt quantity). Resolve it **empirically**
  with the re-pick experiment before locking.

## Things we need to discuss / decide (open)

1. **Endpoint definition** -- run the re-pick experiment first, then decide.
2. **Water prior mechanics** -- which solubility model; condition on composition or
   only pressure; how to keep it "uninformative within the possible region" so item
   1's prior-agnostic principle survives.
3. **Bulk vs melt** -- do we only *flag* crystal-rich rocks, or actually partition
   porphyritic bulks into liquid + phenocryst cargo? We do not currently measure
   crystallinity; what proxy (petrographic metadata? a chemistry-based heuristic?).
4. **Does deep-water-bounded sampling change the multimodality magnitude (F4)?**
   Re-measure after the water fix.
5. **Scope** -- none of this reopens volcanic-vs-plutonic or the equilibrium choice;
   confirm the owner agrees before it drifts into a rearchitecture.

## Proposed next experiment (cheap, on existing data)

Re-pick every draw's endpoint from the stored cooling ladders at a **crystallinity
ladder** (e.g. melt = 50%, 30%, 10%, near-solidus), and per composition report:
(a) does the amp/opx identity switch survive, (b) how do the modal proportions and
the phase list change, (c) where does liq-as-glass become unreliable (silicic
near-solidus). This directly adjudicates ChatGPT vs Gemini on the endpoint using
the `equilibrium_multimodality` raw output -- no new MAGEMin run.

## Consolidated ledger + order of operations (the durable tracker)

The corpus open items collapse into four tasks (from the earlier F1-F7 triage;
this section supersedes that list):

- **A -- Endpoint decision** (near-solidus vs eruptible-crystallinity ~50% melt).
  Absorbs F6 (trace-phase / muscovite min-mode cutoff -- likely a near-solidus
  artifact that shrinks if the endpoint moves) and F7 (cooling floor vs melt
  threshold). Settled by the re-pick experiment above -- cheap, existing data.
- **B -- Water redesign** (flat 0-8 box -> pressure-dependent-solubility-bounded,
  deep pre-eruptive H2O). Absorbs F1 (Wallace/Plank per-class ranges as an input)
  and F4 (re-measure multimodality magnitude after the fix).
- **C -- Subset lock / productionize.** Absorbs F2 (reuse `bulk_chemistry`, persist
  the volcanic-subset parquet), F3 (literature alkalinity boundary + QC windows),
  F5 (foidite / silica-undersaturated non-convergence: exclude or route to the
  alkaline database).
- **D -- Real-data role + sourcing** (new). Validation-only vs. training the
  mineralogy heads on real data; the requirements on a real set; PANGAEA / LEPR /
  reconsidering the 484 MagmaDB experiments as a validation set. Constraint:
  CondHead (conditions) can only be supervised by synthetic labels or experiments,
  so real data can lighten/validate the mineralogy heads but not replace the
  synthetic corpus.

**Order of operations** (the real-data decision gates how much proxy polish is
worth; the two Phase-0 items are cheap and high-information, and independent):

- **Phase 0 (parallel, cheap):** D-scope -- does a usable real bulk+mineralogy
  volcanic set exist? AND Task A -- endpoint re-pick on the existing cooling
  ladders (no new MAGEMin run).
- **Phase 1 (decide):** real-data role from the search; endpoint from the re-pick.
- **Phase 2 (productionize, only after Phase 1):** Task B (water) + Task C (subset)
  -> generate the corpus -> train heads -> validate against the real set.

Do not start B or C until Phase 1 lands -- the endpoint and real-data answers
change what "productionize" means. Tasks graduate to GitHub issues when work
begins on them.

## Literature (from both reviews)

- Rheological lock-up / eruptibility: Marsh (1981); Scaillet et al. (1998).
- Kinetic quench, phenocryst preservation: Blundy & Cashman (2005).
- Eruption-T windows / thermobarometry: Putirka (2008); Blundy; also Winter and
  Wilson textbooks for phase equilibria.
- Silicic melt-model limits: Ghiorso et al. (2002); Gualda et al. (2012,
  rhyolite-MELTS).
- Water solubility / degassing: Newman & Lowenstern (2002, VolatileCalc); Plank et
  al. (2013); Wallace (2005).
- Cryptic hydrous / amphibole "sponge": Davidson et al. (2007).
- Crystal mush, bulk not equal to liquid in silicic systems: Hildreth (1981).
- Precedent: Candioti et al. (2026, G-cubed) -- volcanic-as-liquid rationale,
  forward surrogate target, fixed 4 wt% H2O, liquid validated vs experimental glass.
