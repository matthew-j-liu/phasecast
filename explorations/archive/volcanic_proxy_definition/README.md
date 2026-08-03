# volcanic_proxy_definition

**Status: RESOLVED 2026-07-14 — decisions locked into
`docs/corpus_regen_spec.md`, which supersedes this doc's proposals.** The
endpoint is the phenocryst regime (stop cooling at residual melt < 30%; label
rungs 60/50/40/30, single composition-independent rung picked by label-level
pre-screen), labels are crystals-renormalized solids, water is
solubility-bounded (N&L 2002 rhyolite curve capped at 8 wt%). The two
experiments that settled it live here: `endpoint_repick.py` (does the amp/opx
multimodality survive shallower endpoints? yes, across the whole 70-1% melt
range) and `felsic_saturation.py` (do felsic diagnostics need a deeper stop?
no — see below). The synthesis below is kept as the reasoning record.

This is the foundational
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

## Anchor: think about this problem as transfer learning

Owner call 2026-07-01, alongside the mineralogy-first priority. The full
framing lives in [docs/heads.md](../../docs/heads.md) ("The transfer-learning
lens"): train where physics-rooted labels are simulable (equilibrium cargo of a
stored melt), hypothesize transfer to the downstream task (real-rock
mineralogy, label-poor), measure the hypothesis on a validation-only real set.
Mineralogy is the transfer target; CondHead is a bonus rider, never a design
driver.

## The two experiments (what resolved Task A)

**`endpoint_repick.py`** — re-picks every stored draw's endpoint on a
residual-melt ladder (70/60/50/40/30/20/10/5/1%). The amp<->opx switch holds at
51-60% of bulks (64-73% of intermediates) at EVERY rung, so the multimodality
headline constrains nothing about the endpoint. Near-solidus grows the phase
list ~50% and carries a 10% wet-bulk attrition tail; the phenocryst regime has
~zero attrition. Results: `results/endpoint_ladder_{summary,by_bulk}.csv`,
`endpoint_ladder.png`.

**`felsic_saturation.py`** — the composition-conditional-endpoint test. Tracks
quartz, biotite, and K-dominant feldspar (sanidine proxy, K2O > Na2O in the fsp
instances) presence at each rung, per SiO2 band, against real du Bray
phenocryst presence (subalkaline sim bulks only, matching the arc suite).
Verdict: composition-conditional endpoint REJECTED. Quartz saturates EARLY in
felsic bulks (median first appearance at 75% melt remaining; 0.95 presence at
the 50% rung vs 0.53 real), K-feldspar matches real at 50-60% melt (0.50 vs
0.49), and real felsic lavas are the crystal-poorest — a deeper felsic stop
moves away from the real rocks, not toward them. Near-solidus invents felsic
phenocrysts in mafic bulks (quartz 61%, biotite 91%, vs 0% real) — the
strongest quantitative argument for the phenocryst regime. The one tension:
biotite under-shoots real arc frequencies at every phenocryst-regime rung and
matches only near-solidus; it is water-hungry and the flat H2O box includes
dry draws real wet-arc magmas don't sample, so biotite is the SENTINEL PHASE
to re-check after the solubility-bounded water redesign. Results:
`results/felsic_saturation_by_band.csv`, `felsic_saturation.png`.

Both read the stored cooling ladders in `../equilibrium_multimodality/work/`
(gitignored, regenerable via that exploration's `driver.py`).

## Consolidated ledger + order of operations (the durable tracker)

The corpus open items collapse into four tasks (from the earlier F1-F7 triage;
this section supersedes that list). All four are now decided; statuses below,
detail in `docs/corpus_regen_spec.md`:

- **A -- Endpoint decision.** RESOLVED (2026-07-14): phenocryst regime,
  crystals-renormalized labels, rungs 60/50/40/30 from one stored ladder,
  single composition-independent rung picked by label-level pre-screen. F6
  (min-mode cutoff) demoted: the muscovite artifact largely dies with the
  endpoint move; start at 0.1 wt%, pre-screen at both 0.1 and 1. F7 (cooling
  floor) mooted: stop at melt < 30%, ~zero attrition.
- **B -- Water redesign.** DESIGN LOCKED: N&L 2002 rhyolite-curve solubility
  bound capped at 8 wt% (melt-inclusion supply ceiling); Wallace/Plank ranges
  (F1) migrate to the inference prior. F4 (re-measure multimodality magnitude)
  is the regen agent's 36-bulk A/B, still to run.
- **C -- Subset lock / productionize.** DECIDED: volcanic core 25 aphanitic
  names, foidite excluded (F5), total-Fe sufficient with provenance tier, QC
  flags weight bulks out of the stratified draw (F3); implementation with the
  regen agent (F2).
- **D -- Real-data role.** RESOLVED: validation-only. The benchmark is built
  (`explorations/real_mineralogy/`, 730 samples) plus the experimental tier
  (LEPR 2,368 + MagmaDB 510). Fine-tuning on it would spend the only
  measurement of the sim-to-real gap.
  *Superseded 2026-07-24*: that measurement was taken -- 10.404 nats over 635
  rocks, zero-shot -- which released the condition this verdict rested on. The
  set is split by stratigraphic unit and adapted on from there.

Order of operations for what remains (regen agent): bounded sampler ->
36-bulk water A/B -> pilot -> label-level pre-screen picks the rung ->
production launch. Detail in the spec's "Remaining work."

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
