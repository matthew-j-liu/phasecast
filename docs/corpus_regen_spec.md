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
  - **Latent-neighbor analogs:** shown to the user as transparent evidence, NOT
    reweighted (owner preference). Coarse OOD flag for unsupported scenarios.
  - **User-supplied constraints** (observed assemblage, thermometry): strongest
    de-circularizer; issue #21.
- Scenario weights are labeled "prior-weighted plausibility," never nature's
  rock-specific probability. Headline framing keeps the physics-linked conditional
  hypotheses + thin-section tests; the weight is a reweightable annotation.

## Open items — Level 1 (gate a small-scale test, ~hundreds of bulks)

1. **Conditions sampling** — Monte Carlo from a broad proposal `q` (coverage),
   prior deferred to inference; `q`, sample count N, procedure, and an illustration
   of the generated per-bulk data.
   - owner: **MC agent** (done, integrate) · blocks: items 2-6 lock, extractor
2. **Near-solidus rule** — liq ~1-2% target, per-class calibration floor, attrition
   tag for bulks that never reach it (alkalic andesite held 20-33% melt at floor —
   real tail). · owner: integrator+owner · status: open
3. **Subset + QC** — exact volcanic `qap_name` list, has-iron filter, oxide-sum
   sanity, optional Candioti-style 89% KDE OOD screen; komatiite/picrite flagged not
   filtered. · owner: integrator+owner · status: open
4. **Extraction code** — productionize the prototype to emit the full label set.
   Prototype `extract.jl` (modes + `Comp_wt` + `G_system`) and the recovered
   `run_magemin.jl` + Project/Manifest live in `/tmp/mgm_sanity`; `/tmp` is
   ephemeral, all recoverable from git `2fd5559^`. · owner: integrator · depends: 1
5. **Phase vocabulary** — discover fresh from output + rare-phase policy (g, cd, ol
   appeared rarely in recon). · owner: integrator · status: open
6. **Output schema** — parquet column layout. · owner: integrator · status: open

## Open items — Level 2 (gate the production launch)

7. **Validate small-test corpus** — liq-fraction distribution, per-phase coverage,
   emFrac stats confirm recon (amphibole ~2.3), mass-balance closure holds,
   multimodality present. · depends: small test
8. **Sample count finalized** — after the small test shows whether N resolves the
   multimodal structure. · depends: 1, 7
9. **Compute + orchestration** — N bulks x N samples x near-solidus search; chunked
   orchestrator (adapt old pattern) with juliacall+MAGEMin leak workarounds
   (maxtasksperchild=1, GC between chunks). · status: open
10. **Provenance/tier schema** — incl. optional Fe-imputation tier-2 (v1.5, not v1).
    · status: open

## Parallel exploration (feeds the OUTPUT layer, not corpus-gen)

- **MagmaDB assessment** (MagmaDB agent, done) — what real analogs-as-evidence +
  coarse OOD flagging buy; go/no-go on a larger natural-sample thermobarometry
  compilation. Output in `/tmp` (to be summarized + integrated). Does NOT block
  regen.

## Critical path

MC agent (item 1) -> lock items 2-6 -> build extractor (item 4) -> small test
(item 7) -> production (items 8-10). MagmaDB agent runs in parallel, feeds the
output/inference layer (Sr/Y + latent-neighbor reweighting), not corpus-gen.

## Cross-references

- ADR 004/005 (FC decision being amended), docs/encoder.md, docs/heads.md
- CLAUDE.md "Settled decisions"
- issues: #9 (this corpus, tracker), #13 (inverse model), #21 (optional conditioning)
- memory: comphead-parameterization-options (CompHead + emFrac recon stats)
- working MAGEMin runner: `/tmp/mgm_sanity` (recoverable from `2fd5559^`)
