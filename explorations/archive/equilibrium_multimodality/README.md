# equilibrium_multimodality

**Question.** Does the mineralogy posterior stay **multimodal in phase identity**
under the LOCKED equilibrium generator, or does it collapse to "same phases,
different proportions"? `docs/heads.md` sells a top-K scenario headline built on
an andesite whose amphibole-in (deep/wet) vs orthopyroxene-in (shallow/dry)
silicate switch is the showcase. The MC agent's N=1 test (bulk 6875) found
amphibole on every near-solidus path and read that as an FC-only phenomenon,
putting the headline at risk. This resolves the risk by measuring it on a
composition-stratified subset. It also exercises spec items 2-6 in passing.

**Method.** 36 volcanic bulks stratified across SiO2 (basalt -> rhyolite) x
alkalinity (subalkaline vs alkaline), drawn class-representatively from the
volcanic + has-iron subset of gard2019 (`selection.py`). Per bulk: class-
conditional Middlemost Fe split (recovered `cnmineral.data.fe_redox` +
`magemin_oxides`, provenance tagged), N=128 `(P, H2O)` Latin-Hypercube draws from
the LOCKED proposal `q = Uniform(P[0.5,12] kbar x H2O[0,8] wt%)`, each cooled on a
1200->600 C ladder (H2O injected as an exact wt fraction). MAGEMin `ig`
equilibrium at every point via the productionized runner (`magemin/run_magemin.jl`).
Per draw, take the near-solidus endpoint (highest-T converged point with
`liq < tau`), at tau = 1% and 5%. Classify each bulk phase-identity-multimodal
(a silicate identity -- amp/opx/bi/g/ol/ne/mu -- sits at intermediate presence
frequency across draws) vs proportion/melt-only. 142,848 MAGEMin points total.

```
conda activate mineral && export KMP_DUPLICATE_LIB_OK=TRUE
python explorations/equilibrium_multimodality/selection.py   # subset + 36 test bulks
python explorations/equilibrium_multimodality/driver.py --n 128   # ~25 min, writes work/ (gitignored)
python explorations/equilibrium_multimodality/analyze.py     # verdict + results/ + figures
```

**Result.** The multimodality survives, decisively, and it is the exact
heads.md structure. 33/35 converged bulks (94%) show at least one silicate-
identity flip; **89% excluding suspect near-solidus muscovite, 54% on hard
silicate switches (amp/opx/ol/g/ne), and 51% show the specific amp<->opx
switch**, peaking at 73% in the intermediate 52-63% SiO2 band where the heads.md
andesite lives. The switch IS deep-wet-amphibole / shallow-dry-orthopyroxene:
for andesite 52.4, amphibole appears when H2O~4.3 wt% (P~8.7 kbar) and opx when
H2O~0.18 (amp-only 69 draws, opx-only 4, neither 53) -- textbook anti-correlation.
The verdict is threshold-robust (tau 1% vs 5% give 94% / 51% vs 54%). The 6875
N=1 result was unrepresentative: a wet andesite sitting in the amphibole field.
Under a broad H2O proposal the amp-in/amp-out and amp/opx structure reappears.
The degree-of-melting axis (T_end / liq spread) survives too -- it is an
additional axis, not a replacement. Evidence: `results/multimodality.png`,
`results/melting_axis.png`, `results/bulk_verdicts_tau0{1,5}.csv`,
`results/summary.csv`.

**Decision.** The heads.md multimodal-scenario headline stands (do not reframe
to proportion-only). The full handback below proposes the item-2..7 spec values;
the owner integrates them into the corpus regen spec (since folded into ADR 004).

## Water A/B (regen spec, now ADR 004, "Water A/B")

The results above were generated under the since-superseded flat proposal box
(`q` uniform on the full P x H2O rectangle). `driver.py` now draws from the
locked solubility-bounded proposal (`cnmineral.data.condition_proposal`,
N&L 2002 rhyolite curve capped at 8 wt%) and writes to `work/bounded/`; the
flat-box run stays at `work/` root as the A/B baseline. `water_ab.py` compares
the two on the same bulks at the phenocryst-regime rungs: amp/opx switch
magnitude, and quartz/biotite/K-feldspar presence vs the real du Bray
frequencies (biotite is the sentinel -- it under-shoots everywhere under the
flat box).

```
conda activate mineral && export KMP_DUPLICATE_LIB_OK=TRUE
python explorations/equilibrium_multimodality/driver.py --n 128   # ~25 min, writes work/bounded/
python explorations/equilibrium_multimodality/water_ab.py         # A/B tables + results/water_ab_*.csv
```

---

## Handback

### VERDICT

**Phase-identity multimodality survives the equilibrium generator. Keep the
heads.md top-K scenario framing.** The amphibole-in / orthopyroxene-in switch is
real, is the dominant intermediate-composition structure (51% of bulks, 73% at
52-63% SiO2), and is driven by the water axis exactly as heads.md describes.
The corpus is multimodal in BOTH phase identity (amp/opx, plus biotite / garnet /
nepheline in-out) AND continuously in degree-of-melting (T_end, liq fraction).

One honesty rider for the headline (not a reframe): the amphibole axis is
**exposed by the breadth of the H2O proposal** `[0, 8] wt%`. This is the intended
prior-agnostic coverage design -- so the *displayed* scenario weights are
prior-dependent: a tight wet prior collapses the amp axis back toward the 6875
picture, a broad prior shows the full switch. The corpus is multimodal; how much
a user sees depends on `p(H2O)`. State this where the headline is sold.

### EVIDENCE

- `summary.csv`: tau=1% -> 94% phase-identity-multimodal, 51% amp<->opx switch,
  10.4% mean attrition; tau=5% -> 94% / 54% / 9.6%. Verdict threshold-robust.
- Muscovite-discounted bounds (35 bulks): all-silicate 94%, excl mu 89%, hard
  switches (amp/opx/ol/g/ne) 54%, amp<->opx 51%.
- amp<->opx by SiO2 band: mafic<52 40%, intermediate 52-63 **73%**, felsic>63 44%.
- Mechanism (deep-wet amp / dry opx), near-solidus endpoints:
  andesite 52.4 amp@H2O 4.3/P 8.7 vs opx@H2O 0.18; andesite 57.5 amp@H2O 4.1/P 7.4
  vs opx@H2O 3.1/P 2.9 (shallower). Anti-correlated (amp-only >> both).
- Convergence 35/36 bulks 100% at N=1 preflight; foidite (SiO2 36.5) 0%.

### SPEC IMPACT (proposed values, keyed to the regen spec's items; the spec is folded into ADR 004)

- **Item 2 (near-solidus rule).** Adopt **tau = 1%** liq (the verdict is
  threshold-insensitive; 1% vs 5% move nothing material). The real lever is the
  ladder floor, not tau: attrition is ~0% for all subalkaline bulks but 30-84%
  for wet alkaline/undersaturated bulks (basanite 39, phonolite 47, phonolitic
  tephrite) that never reach liq<tau by 600 C. Proposal: extend the cooling floor
  to ~550 C (or 500) for wet bulks, OR tag+accept attrition and drop those draws.
  Do NOT raise tau to hide it -- tau=5% barely helps (9.6% vs 10.4%).
- **Item 3 (subset + QC).** Volcanic core = the 25 aphanitic QAPF names in
  `cnmineral.data.volcanic_subset::VOLCANIC_QAP` (built here as
  `selection.py::VOLCANIC_QAP`, promoted to `src/` when the corpus bulk tables
  began importing it), ~335k bulks. **rock_origin tension (owner call):**
  `gabbro`/`norite`/`anorthosite`/`orthopyroxene gabbro` are majority-tagged
  `volcanic` in the full-CSV `rock_origin` despite plutonic names -- left OUT of
  the core here; recommend keeping them out (trust the aphanitic-name definition;
  they are plutonic textures) unless the owner wants the rock_origin superset.
  QC counts (informational flags, not hard drops): oxide-sum out-of-window 7628,
  supersilicic (SiO2>80) 1207, ultramafic (MgO>18) 309. **Convergence QC:**
  extreme silica-undersaturated foidite/nephelinite (SiO2 ~36) does NOT converge
  in the `ig` database (positive G) -- exclude it, or route to the alkaline DB.
- **Item 4 (extraction).** Done: `magemin/run_magemin.jl` (durable Julia project,
  pinned Manifest) emits per-point JSONL with modes + per-phase `Comp_wt`
  (SS + PP, duplicates preserved) + `emFrac_wt` + `G` + `status`. Fe utilities
  recovered to `src/cnmineral/data/fe_redox.py` (Middlemost table extended for the
  full volcanic vocab) and `magemin_oxides.py` (split_fe + provenance + injected
  H2O). Propose merging these back as the production Fe path.
- **Item 5 (phase vocabulary).** Observed: liq, fl, fsp, cpx, opx, amp, ol, spl,
  ilm, bi, mu, g, cd, q, ne (+ fper only in the failing foidite). Rare/suspect:
  `mu` (muscovite) appears at 0.1-0.7 presence even in mafic bulks (basalt/
  andesite) where it is petrologically implausible -- likely a near-solidus ig
  white-mica artifact. Propose a **min-mode presence threshold (~1 wt%, not
  0.1%)** for the ModeHead presence target so trace near-solidus phases (mu, cd)
  do not inflate the vocabulary or the presence labels.
- **Item 6 (output schema).** `results/endpoints_tau01.csv` is the concrete
  per-draw schema seed: `bulk, draw, qap_name, series, sio2_bulk, P, H2O, T_end,
  attrition, liq_end` + the assemblage. Production per-row parquet adds `q`,
  `fe_provenance`, per-phase `Comp_wt`, `emFrac_wt` (diagnostic), liq/fl comp,
  renormalized bulk, `G_sys`.
- **Item 7 (multimodality headline).** Resolved above: keep multimodal scenarios;
  add the H2O-prior rider.
- **Item 8 (sample count).** N=128 presence SE ~ sqrt(p(1-p)/128) ~ 0.04; the
  verdict is stable, consistent with the `condition_sampling` N-table. Production
  N ~ 48-64 remains defensible for the marginals.

### OPEN-RISKS

- **Muscovite / near-solidus trace phases** inflate the broad 94% number; the
  robust claim is the 51-54% hard-switch band. Needs the item-5 min-mode decision
  before the presence labels are trusted.
- **Multimodality magnitude is H2O-proposal-dependent** by construction. Fine for
  the corpus (coverage), but the headline and any scenario-weight display must
  name the prior.
- **Attrition floor** for wet alkaline/undersaturated bulks (item 2).
- **Foidite convergence failure** in the `ig` DB -- a real scope edge (item 3).
- **Fe split mole-vs-mass approximation** (recovered original treats Fe3+/SigmaFe
  as a mass ratio; ~few % at r=0.3-0.45). Minor; flag for the production Fe path.
- **35 stratified bulks vs 335k production.** Representative but small; the
  intermediate-band amp/opx result is strongest, felsic/alkaline thinner.

### ARTIFACTS

- Code: `selection.py`, `driver.py`, `analyze.py`; runner `magemin/run_magemin.jl`
  (+ `Project.toml`, `Manifest.toml`); `src/cnmineral/data/{fe_redox,magemin_oxides}.py`.
- Committed results: `results/{subset_class_counts,selected_bulks,summary}.csv`,
  `results/{endpoints,bulk_verdicts}_tau0{1,5}.csv`,
  `results/{multimodality,melting_axis}.png`.
- Tests: `tests/data/test_magemin_oxides.py`,
  `tests/explorations/test_equilibrium_multimodality.py`.
- Raw MAGEMin JSONL (142,848 points) in `work/` -- gitignored, regenerable via
  `driver.py`.
- ADR: `docs/decisions.md` ADR 004 amended 2026-06-30 (volcanic/equilibrium scope).
