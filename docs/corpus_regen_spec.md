# Track B corpus regen — spec and open items

Prospective spec and the shared source of truth across the parallel work
sessions. This is a planning doc: items collapse to pointers as they land, and
the doc is deleted once regen ships. Reasoning is not repeated here — see the
cross-references. Issue #9 is the tracker; this doc is the detail.

All design decisions are now locked (2026-07-14); what remains is
implementation, the pilot, and two re-measurements. See "Remaining work."

## Locked decisions

- **Generator:** MAGEMin equilibrium (not PetThermoTools/FC) on the volcanic +
  has-iron subset of `data/bulks/gard2019`. For each sampled condition, cool the
  bulk on the equilibrium ladder and read the assemblage in the phenocryst
  regime (see Endpoint). Rationale: a volcanic rock crystallizes closed-system
  in place, which *is* equilibrium; FC models open-system cumulate-shedding
  (plutonic). Amends ADR 004, scoped to volcanic; no eq-vs-FC A/B (conceptual
  argument accepted).
- **Conditions:** (P, H2O) are sampled; **T is derived** per sample by cooling
  to the endpoint, not a sampled axis. The corpus is **prior-agnostic**: sample
  (P, H2O) for broad *coverage* only, and defer all prior/weighting to
  inference (see Output stance). Per bulk, Latin-Hypercube draws from the fixed
  proposal `q = Uniform(P [0.5, 12] kbar x H2O [0, min(8, S(P))] wt%)`; store
  `(P_i, H2O_i, q(C_i))` per row. Grid-density artifact confirmed real; MC
  matches a balanced grid (`explorations/condition_sampling/`).
- **Water bound `S(P)`:** the Newman & Lowenstern 2002 (VolatileCalc)
  rhyolite-curve H2O solubility — the more water-generous of their two
  published curves, since a composition-independent bound must cover all
  compositions — capped at 8 wt%, the melt-inclusion supply ceiling (Wallace
  2005; Plank et al. 2013, arc magmas 4 +/- 2 wt%, extremes ~6-7). The cap
  binds above ~4-5 kbar, so the solubility curve is only evaluated inside its
  calibration range. Known, accepted gap: alkaline melts (phonolite, trachyte)
  can dissolve somewhat more than the rhyolite curve at low P, so very wet
  low-P alkaline draws are under-covered; documented, not patched (a
  composition-aware published model, e.g. Iacono-Marziano, is the upgrade path
  and costs a regen). Oversaturated borderline draws exsolve excess H2O as
  `fl` inside MAGEMin and are tagged. Wallace/Plank per-class ranges (ADR 003)
  migrate to the **inference prior**, never the sampler.
- **Endpoint: phenocryst regime, single and composition-independent.** Cool
  each draw until residual melt < 30 wt%, then stop (no near-solidus leg).
  Read the assemblage at four rungs — 60/50/40/30% residual melt — re-picked
  from the same stored ladder, no extra MAGEMin cost. The training rungs were
  chosen by a **label-level pre-screen** against the real benchmark
  (per-SiO2-band presence match first, rank-level crystals-renormalized
  proportions second, exact magnitudes only on the 13 point-counted samples):
  **rungs 50 + 30**, picked 2026-07-14 (see "Remaining work" item 4). A composition-conditional
  endpoint (deeper stop for felsic) is REJECTED: measured on the stored
  ladders, quartz/K-feldspar saturate early in felsic bulks (present at the
  50%-melt rung at 0.95/0.50 vs real 0.53/0.49), real felsic lavas are the
  crystal-poorest, and near-solidus invents felsic phenocrysts in mafic bulks
  (quartz 61%, biotite 91% vs 0% real). Evidence:
  `explorations/volcanic_proxy_definition/` (`endpoint_repick.py`,
  `felsic_saturation.py`). The amp/opx multimodality headline survives the
  entire 70-1% melt range.
- **Primary label: crystals-renormalized solid assemblage.** liq and fl are
  dropped from the label and the solids renormalized to sum to 1 — matching
  the real data's granularity (itemized phenocrysts + one lumped groundmass
  bucket) and the validation view (drop real groundmass, renormalize
  phenocrysts). Groundmass stays lumped; itemized groundmass is deferred
  (unvalidatable with current real data AND unfaithfully modeled by
  equilibrium — quench microlites need a kinetic generator). **No
  melt-fraction head**: at a fixed melt-fraction endpoint the label is a
  constant, so there is nothing to learn; real groundmass % is set by ascent
  dynamics, not chemistry. Consequence to carry into heads work: mass-balance
  closure teacher-forces the LABEL melt fraction at training (liq/fl comp and
  fraction are stored per row for exactly this); the inference-time closure
  residual as an OOD signal degrades and needs a rethink when heads land.
- **Presence cutoff:** presence targets built at mode > 0.1 wt% to start. Raw
  modes are stored, so the cutoff is a label-rebuild + head-retrain, never a
  regen. The pre-screen runs at both 0.1 and 1 wt% once to confirm the rung
  choice is cutoff-insensitive (the near-solidus muscovite/cd trace-phase
  artifact largely dies with the endpoint move, so stakes are low).
- **Scale, two tiers:** production **10k bulks x 48 draws** (stratified by
  `qap_name` from the volcanic core, rare-class floors, ~480k rows per rung ~
  v2 corpus scale) + a **high-precision tier ~500 bulks x 128 draws** assigned
  to val/test (presence-frequency SE 0.04 vs 0.07 — the calibration/ablation
  reference; training tolerates noise, measurement doesn't). A ~500-bulk
  pilot (the first 5 chunks of the shuffled train table) gates production
  launch. ~13M MAGEMin points total, ~2 days
  accumulated laptop runtime at the measured ~4-6k points/min.
- **Subset + QC:** volcanic core = the 25 aphanitic QAPF names in
  `cnmineral.data.volcanic_subset::VOLCANIC_QAP` (~335k bulks);
  plutonic-named rock_origin superset stays OUT. Foidite family excluded (`ig`
  non-convergent). **has-iron = total Fe sufficient** (the locked Fe path
  imposes the class-conditional Middlemost split regardless); rows with
  measured FeO+Fe2O3 get a provenance tier tag so a stricter cut stays
  available at training time. QC flags (oxide-sum window, supersilicic,
  ultramafic) are tags, not drops — but flagged bulks are weighted OUT of the
  stratified 10k draw (clean bulks are abundant; the flags/rules stay for
  inference-time OOD use).
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
- **Storage shape:** per ladder point, modes only; full labels at the four
  rungs — modes, per-phase oxide wt% (`Comp_wt`), `emFrac_wt` (diagnostic),
  liq/fl oxide comp + fractions (mass-balance pinning), renormalized bulk,
  `G_sys`, the sampled (P, H2O) + q + derived T_end, provenance/tier tag, and
  the bulk traces (kept for the future trace-retrieval / trace-partition
  mechanisms — nothing this regen closes off). Keeps mode re-picks possible at
  ~5x less disk than full-label ladders (~20 GB raw otherwise).
- **Orchestration:** chunked Python orchestrator wrapping `run_magemin.jl`
  unchanged; ~100-bulk chunks, each flushed atomically to its own parquet +
  appended to a completion manifest; workers recycled per chunk (the juliacall
  leak mitigation and the checkpoint want the same boundary). Hard
  requirement: **resume = rerun the same command; at most one in-flight chunk
  redone** (~20-30 min), so generation is freely pausable. **Solver-crash
  containment** (added during production, 2026-07-14): MAGEMin segfaults
  deterministically on rare pathological inputs and the overflow corrupts the
  process heap, so a dead Julia call group is bisected at draw boundaries and
  the isolated draw dropped wholesale (lands in ordinary n_failed/attrition
  accounting; dropped draws recorded per chunk in the manifest as
  `solver_crashes`); NaN-bearing records are serialized as failed rows instead
  of killing the batch; a complete output file overrides a nonzero exit
  (cleanup-phase crashes).
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
  - **Wallace/Plank per-class H2O ranges** (ADR 003 migration) and
    **trace-ratio priors** (Sr/Y, La/Yb): volcanic-calibrated crustal-depth
    proxy, but a regional/median proxy doing single-sample duty —
    composition-gated to intermediates, large single-sample sigma, so a soft
    directional nudge, not a collapse.
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

## Remaining work (implementation, no open decisions)

1. **Bounded sampler** — LANDED: `cnmineral.data.condition_proposal`
   (VolatileCalc rhyolite pure-H2O port, 850 C curve; LHS warped by
   `w_max(P)`; per-row closed-form `q`), tests in
   `tests/data/test_condition_proposal.py`.
2. **Orchestrator + schema** — LANDED:
   `scripts/build/generate_mineralogy_corpus.py` (chunked, manifest-resumable,
   fresh Julia process per call group) over `cnmineral.data.magemin_corpus`
   (three-table draws/ladder/rungs layout per the Storage shape; note MAGEMin
   outputs total-Fe-as-FeO + excess O, not Fe2O3). Bulk tables:
   `scripts/build/build_corpus_bulk_tables.py`; health gate:
   `scripts/build/report_corpus_health.py`.
3. **Water A/B** — DONE (journal 2026-07-14, second entry): amp/opx switch
   holds at every rung (0.57–0.60, intermediates unchanged); biotite did NOT
   move — the gap is an inference-prior question (dry draws kept by design),
   not corpus coverage. Results:
   `explorations/equilibrium_multimodality/results/water_ab_*.csv`.
4. **Pilot (~500 bulks, production config)** — RUN 2026-07-14, health gates
   passed (attrition ≤1.4% per rung, fully-failed draws 0.8%, closure p95
   3.5e-5 with the >1e-3 tail 0.1% and concentrated in tagged oversaturated
   draws). Pre-screen run at both cutoffs
   (`explorations/volcanic_proxy_definition/results/rung_prescreen_*.csv`):
   rung 50 best on aggregate presence MAE and rank Spearman at both cutoffs;
   intermediate-band biotite still zero at 60/50 and matching real only at 30.
   **Rung pick made (owner, 2026-07-14): train on rungs 50 + 30** — 50 as the
   primary (best aggregate match), 30 because it is the only rung where
   intermediate biotite exists to be learned.
5. **Production launch** — IN FLIGHT (launched 2026-07-14): 10k x 48 + 500 x
   128 tiers, chunked, pausable (rerun the same orchestrator command to
   resume). Train tier ~37/100 chunks as of 2026-07-15 morning; crash
   containment added after one pathological bulk (976755, six draws dropped;
   see Orchestration). After each tier: `report_corpus_health.py --corpus
   <dir> --figures`.

Only 5 remains (1-4 landed).

## Follow-ups owned outside regen

- **heads.md alignment:** ModeHead vocabulary (currently "~18 phases including
  liq, fl"), the mass-balance section, and the worked scenario example must be
  updated to the crystals-renormalized label + teacher-forced closure when
  heads work starts. Not a regen blocker.
- **Real-benchmark label adapter** (`data/real_mineralogy/` README "Model
  compatibility"): vol%->wt% densities, unsplit-phase rules, hand-sample view.
  Needed before scoring, not before generating.

## Deferred (post-volcanic-v1)

- **Plutonic/cumulate branch (FC-of-parents)** — cumulates require FC of the parent
  magma (equilibrium can't produce them faithfully); a real coverage gap the
  volcanic-only corpus leaves. High roadmap priority, starts after v1 ships. Issue #22.
- **Itemized groundmass** — blocked on real itemized data (Rietveld XRD /
  QEMSCAN maps) AND a kinetic/decompression generator; a future branch, not a
  granularity toggle on this corpus.

## Cross-references

- ADR 004/005 (FC decision amended), ADR 003 (H2O -> inference prior),
  docs/encoder.md, docs/heads.md
- CLAUDE.md "Settled decisions"
- issues: #9 (this corpus, tracker), #13 (inverse model), #21 (optional conditioning)
- memory: comphead-parameterization-options (CompHead + emFrac recon stats)
- endpoint evidence: `explorations/volcanic_proxy_definition/` (ladder re-pick +
  felsic saturation probe), `explorations/equilibrium_multimodality/`
  (multimodality verdict), `explorations/real_mineralogy/` (benchmark + EDA)
- condition sampling: `explorations/condition_sampling/` (LHS, ESS table)
