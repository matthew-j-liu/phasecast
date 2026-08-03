# Trace-conditional (P, H2O) proposal: sampler-level dry run (#48)

**Question.** Before the regen commits ~60 h of MAGEMin to the trace-conditional
proposal (#45), how drastically do the per-bulk condition draws change vs the
current broad sampler -- tier coverage, box shrink, proxy-vs-class
contradictions, retained ambiguity, and the 24-draw envelope?

**Method.** `dry_run.py` runs both proposals side by side on the committed
rosters (10k train + 500 valtest), sampler only, no MAGEMin. The candidate is
`bulk_proposal` in `cnmineral.data.condition_proposal`: immobile-element
discriminant cascade (Pearce 2008 Th/Yb-Nb/Yb -> Shervais 1982 Ti-V ->
Winchester & Floyd Nb/Y) -> soft 3-setting posterior (logistic, sigma 0.15
dex, doubled for felsic) -> posterior-weighted (maficity, setting) literature
boxes; class fallback = recovered ADR 003 Wallace/Plank H2O x Putirka 2008 P
ranges; 8% broad floor on every bulk. Per bulk: 512-draw clouds for both
proposals (5-95% widths, joint area), 24-draw envelopes, class-range overlap,
posterior entropy. 512 draws per proposal per bulk, seed 0; full run ~25 s.

**Result.**
- **Tier occupancy** (`occupancy.png`, `tier_occupancy.csv`): 63% of bulks hit
  a discriminant rung (Th/Yb-Nb/Yb 30%, Ti-V 26%, Nb/Y 7%), 37% class
  fallback; 58.5% are majority-trace after the Nb/Y rung's subalkaline mass
  falls through. Uniform across supergroups (56-70%).
- **Narrowing** (`narrowing.png`, `narrowing_stats.json`): median joint-area
  shrink 0.26 (p10 0.13, p90 0.44); H2O and P width ratios both ~0.52. No
  bulk fails to shrink (floor at 8% guarantees none can widen).
- **Proxy contradicts class** (`overlap_flags.csv`): 354 bulks (3.4%) put
  <10% of their new draws inside their ADR 003 class H2O range -- 324 of them
  `basalt` bulks reassigned wet-arc by the discriminant. This is the known
  arc-basalt/MORB conflation inside the `basalt` qap class, i.e. the gap the
  revision exists to fix, not a scheme error; the list is the manual-review
  deliverable.
- **Retained ambiguity** (`setting_entropy.png`): bimodal -- a confident
  spike at 0 nats and a large shoulder near uniform; 56% of trace-tier bulks
  above 0.5 nats keep genuine multi-setting mixtures (the deep-wet vs
  shallow-dry mass, by design).
- **24-draw envelope** (`envelope_contraction.png`): median contraction 0.25,
  so per-draw coverage density roughly quadruples; the #27 24-draw knee holds
  with margin under the new proposal (contraction can only shrink by
  construction -- the floor is the widest component).
- **Box atlas** (`box_atlas.png/.json`): every box actually sampled with its
  share of total corpus draws; headline shares: basalt fallback 16.9%,
  intermediate-arc 11.3%, andesite fallback 8.2%, floor 8.0%, the rest <=7.2%.
- **Knobs** (`knob_sensitivity.csv`): tier split is insensitive to sigma/eps
  (only felsic-out moves it, 0.585 -> 0.443); median shrink spans 0.21-0.44
  across sigma x0.5-x2 and eps 0.04-0.16 -- eps is the strongest lever.

**Decision.** Adopted for the regen (owner, 2026-07-23) with default knobs:
sigma 0.15 (intermediate softness), eps 0.08 floor, felsic in the trace tier.
The 354 flagged conflicts are accepted -- trace evidence overrules the
hand-assembled class table's single dry-basalt range, which is the arc/MORB
conflation the revision targets. Knob effects on holdout and real-rock
transfer are speculative pre-regen; revisit only at a future regen, keeping
every knob away from extremes. Boundary-constant
provenance: Pearce array edges anchored on Sun & McDonough 1989 reference
values (chondritic Th/Nb 0.118; N-MORB/E-MORB Nb/Yb split 1.64) verified
against the pyrolite implementation; Shervais Ti/V 20/50 and Nb/Y 0.67 are the
standard published thresholds, taken from the literature record without an
independent web check (search tooling unavailable at run time). The 10%
overlap flag threshold is a reporting choice, not a scheme knob.
