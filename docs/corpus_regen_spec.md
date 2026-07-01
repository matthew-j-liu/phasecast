# Track B corpus regen — spec and open items

Prospective spec and the shared source of truth across the parallel work
sessions. This is a planning doc: items collapse to pointers as they land, and
the doc is deleted once regen ships. Reasoning is not repeated here — see the
cross-references. Issue #9 is the tracker; this doc is the detail.

## Locked decisions

- **Generator:** MAGEMin equilibrium (not PetThermoTools/FC) on the volcanic +
  has-iron subset of `data/bulks/gard2019`. For each sampled condition, cool the
  bulk to near-solidus and read the equilibrium crystalline assemblage there.
  Rationale: a volcanic rock crystallizes closed-system in place, which *is*
  equilibrium; FC models open-system cumulate-shedding (plutonic). Amends ADR
  004, scoped to volcanic; no eq-vs-FC A/B (conceptual argument accepted).
- **Conditions:** (P, H2O) are sampled; **T is derived** per sample by cooling to
  near-solidus (liq ~ 1-2%), not a sampled axis. The corpus is **prior-agnostic**:
  sample (P, H2O) for broad *coverage* only, and defer all prior/weighting to
  inference (see Output stance). Sampling *method* (Monte Carlo from a broad
  proposal vs grid) is open — item 1.
- **Fe/fO2:** class-conditional Middlemost Fe3+/SigmaFe split, fO2 fixed (not
  swept). Candioti's all-ferrous shortcut would distort the oxide phases we report.
- **CompHead target:** per-phase oxide wt% (`Comp_wt`) for ALL phases. The
  endmember vector (`emFrac_wt`) is infeasible as a simplex target — recon on real
  volcanic bulks gave amphibole per-mineral endmember-sum median 2.3, worst
  component -110% (signed affine combination, leaves the positive simplex).
  Endmember heads are an optional bonus on clean phases (fsp/ilm/ol/opx) via
  per-instance loss-masking; amphibole oxide-only; mineral names from the vendored
  stoichiometry classifiers applied to oxide/apfu. See
  [[comphead-parameterization-options]].
- **Labels saved per row:** modes (incl. liq/fl), per-phase oxide wt% (`Comp_wt`),
  `emFrac_wt` (diagnostic), liq/fl oxide comp (mass-balance pinning), renormalized
  bulk, `G_sys`, the sampled (P, H2O) + derived T, provenance/tier tag.
- **Encoder/supervised split:** encoder pretrained on full igneous gard; heads
  supervised on the volcanic + has-iron subset (so OOD/abstention still works on a
  user's non-volcanic input).

## Output / uncertainty stance (post-corpus, does NOT affect regen)

- The contribution is the **marginal posterior under an explicit empirical prior**
  — the integral MAGEMin doesn't compute. This is the spine and needs no external
  dataset.
- Because the corpus is prior-agnostic (coverage `q` only), the inference-time
  weight reweights each posterior sample by `p(c)/q(c)`: `q` (the corpus proposal)
  cancels the arbitrary sampling prior, `p` substitutes an evidence-based one.
  Sources of `p`, weakest-to-strongest, used in combination:
  - **Trace-ratio priors** (Sr/Y, La/Yb): volcanic-calibrated crustal-depth proxy,
    but a regional/median proxy doing single-sample duty — composition-gated to
    intermediates, large single-sample sigma, so a soft directional nudge, not a
    collapse.
  - **Latent-neighbor analogs (SHELVED):** the idea -- show real rocks with measured
    conditions as transparent evidence + a coarse OOD flag -- was assessed against
    MagmaDB and shelved: too sparse/narrow (78 distinct bulks, ~29% igneous coverage,
    arc basalt-andesite only). Revisit only with a better reference set. See
    `explorations/magmadb_analogs/`.
  - **User-supplied constraints** (observed assemblage, thermometry): strongest
    de-circularizer; issue #21.
- Scenario weights are labeled "prior-weighted plausibility," never nature's
  rock-specific probability. Headline framing keeps the physics-linked conditional
  hypotheses + thin-section tests; the weight is a reweightable annotation.

## Open items — Level 1 (gate a small-scale test, ~hundreds of bulks)

1. **Conditions sampling** — LOCKED (MC agent; see `explorations/condition_sampling/`). Per bulk,
   draw N `(P, H2O)` by Latin-Hypercube/Sobol from a fixed broad proposal
   `q = Uniform(P[0.5,12] kbar x H2O[0,8] wt%)`; apply NO prior at generation, store
   `(P_i, H2O_i, q(C_i))` per row; defer every prior to inference via importance
   weights `w = p(C)/q(C)` (change prior -> recompute targets -> retrain heads,
   never re-run MAGEMin). P capped at 12 (volcanic scope; deep-arc out). Grid-density
   artifact confirmed real (16x4 vs 4x16 grids disagree; MC matches a balanced grid).
   · status: LOCKED · blocks: items 2-6 lock, extractor
2. **Near-solidus rule** — liq ~1-2% target, per-class calibration floor, attrition
   tag for bulks that never reach it. MC agent confirmed the tail is real and
   H2O-driven: 17% of bulk-6875's (wettest) draws never reached liq<2% above a
   640 C floor. Consider extending the cooling ladder below 640 C for wet bulks
   before tagging. · owner: integrator+owner · status: open
3. **Subset + QC** — exact volcanic `qap_name` list, has-iron filter, oxide-sum
   sanity, optional Candioti-style 89% KDE OOD screen; komatiite/picrite flagged not
   filtered. · owner: integrator+owner · status: open
4. **Extraction code** — productionize the prototype to emit the full label set.
   Durable prototype in `explorations/condition_sampling/` (`extract.jl` = item-4
   seed, `gen_inputs.py` = LHS draws, `analyze.py` = reweighting/marginals). The
   recovered `run_magemin.jl` + Project/Manifest are in `/tmp/mgm_sanity`
   (ephemeral, recoverable from git `2fd5559^`). · owner: integrator · depends: 1
5. **Phase vocabulary** — discover fresh from output + rare-phase policy (g, cd, ol
   appeared rarely in recon). · owner: integrator · status: open
6. **Output schema** — parquet column layout. · owner: integrator · status: open

## Open items — Level 2 (gate the production launch)

7. **Validate small-test corpus** — liq-fraction distribution, per-phase coverage,
   emFrac stats confirm recon (amphibole ~2.3), mass-balance closure holds. AND the
   now-PIVOTAL question: **is the assemblage posterior actually multimodal in phase
   identity under equilibrium?** MC agent's N=1 test (bulk 6875) found amphibole on
   every path, opx absent. This is genuine closed-system equilibrium behavior, NOT
   an FC-approximation artifact: the system keeps its water so amphibole is stable
   across (P,H2O); the heads.md amp-in/opx-in switch was an FC (open-system,
   melt-drying) phenomenon. Surviving within-bulk variation is degree-of-melting +
   fluid, not silicate identity. If this holds across the subset, the "multimodal
   posterior over assemblages" headline shifts toward "degree-of-melting +
   composition," OR reopens whether eq is the right generator for the desired
   output. Caveat: N=1, alkalic/wet andesite (amph-prone). Test design: run a
   COMPOSITION-STRATIFIED subset (basalt->rhyolite, dry->wet, alk->subalk), classify
   each bulk phase-identity-multimodal vs proportion/melt-fraction-only, report the
   fraction and where it lives. At near-solidus the major framework (plag+px) is
   ~stoichiometry-pinned, so multimodality shows in secondary phases (amph/bt/grt
   in-out); 6875 sat deep in the amph field. Cheap pre-regen diagnostic with the
   existing runner. · depends: small test · status: OPEN (headline risk)
8. **Sample count finalized** — MC agent: run the small test at N=128/bulk (presence
   SE ~0.043; ESS table in `explorations/condition_sampling/README.md`), finalize production N ~ 48-64 by
   subsampling to the tolerated SE/ESS, paired with a class-informed proposal
   density (ESS ~ N for typical priors; sharp-prior ESS ~ N/6). · depends: 1, 7
9. **Compute + orchestration** — N bulks x N samples x near-solidus search; chunked
   orchestrator (adapt old pattern) with juliacall+MAGEMin leak workarounds
   (maxtasksperchild=1, GC between chunks). · status: open
10. **Provenance/tier schema** — incl. optional Fe-imputation tier-2 (v1.5, not v1).
    · status: open

## Parallel exploration (feeds the OUTPUT layer, not corpus-gen)

- **MagmaDB assessment** (MagmaDB agent, done) — SHELVED: MagmaDB is too sparse/narrow
  (78 distinct bulks, arc basalt-andesite only) to carry the analog panel; the
  latent-neighbor-analogs output idea waits for a better reference set. Output-layer
  only, never blocked regen. Full: `explorations/magmadb_analogs/`.

## Critical path

MC agent (item 1) -> lock items 2-6 -> build extractor (item 4) -> small test
(item 7) -> production (items 8-10). MagmaDB agent runs in parallel, feeds the
output/inference layer (Sr/Y + latent-neighbor reweighting), not corpus-gen.

## Deferred (post-volcanic-v1)

- **Plutonic/cumulate branch (FC-of-parents)** — cumulates require FC of the parent
  magma (equilibrium can't produce them faithfully); a real coverage gap the
  volcanic-only corpus leaves. High roadmap priority, starts after v1 ships. Issue #22.

## Cross-references

- ADR 004/005 (FC decision being amended), docs/encoder.md, docs/heads.md
- CLAUDE.md "Settled decisions"
- issues: #9 (this corpus, tracker), #13 (inverse model), #21 (optional conditioning)
- memory: comphead-parameterization-options (CompHead + emFrac recon stats)
- agent explorations: `explorations/{condition_sampling,magmadb_analogs}/` (README
  + committed results)
- working MAGEMin runner: `/tmp/mgm_sanity` (recoverable from `2fd5559^`)
