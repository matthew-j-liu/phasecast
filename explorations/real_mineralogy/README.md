# real_mineralogy

The sim-to-real benchmark for the mineralogy heads: real volcanic rocks with
whole-rock chemistry + measured modal mineralogy. Tests the sim-to-real transfer
claim (docs/heads.md, "The transfer-learning lens") -- does a head trained on
MAGEMin reference-state labels predict the minerals a geologist counts?

The zero-shot answer is measured: **joint NLL 10.404** over 635 least-altered
rocks, on heads that had never seen one (journal 2026-07-24). The set is a
benchmark like any other from there -- split it, adapt on it, evaluate on
held-out rocks, and compare models and strategies on the same test slice
(`results/real_rock_ledger.yaml`, issue #52). Splits group by **stratigraphic
unit**: 132 units over the 635 rocks, median 2 rocks each, largest 66, and the
same eruption means near-identical chemistry, so a rock-level split leaks.

Data lands in `data/real_mineralogy/` (gitignored: raw source files + the
generated `samples.csv` / `modes.csv`). `harvest.py` regenerates the unified
tables from the raw downloads.

## The screen (why these sources, per the owner's cut)

A sample qualifies on TWO requirements, not four:

1. **Whole-rock major-element chemistry** on the same sample/unit as the mineralogy.
2. **Quantitative modal mineralogy** (vol% or wt%) of igneous phases.

Freshness is OUR QC filter (LOI, oxide totals, secondary phases in the mode
list), not a required field. A phenocryst/groundmass split is not required
either -- it only decides which comparison a sample supports: **presence** needs
no split (and is where the amp/opx headline lives); **magnitude** needs a split
or a glass% / texture note, because groundmass microlites are a later crystal
generation than the reference-state cargo we predict. See docs/heads.md.

## Sources

Two tiers, by modal-measurement precision:

**Precision tier -- formal point counts / image analysis, phenocryst-groundmass split:**
- **Pinatubo 1991** (Pallister et al. 1996, USGS *Fire and Mud*). 6 samples,
  dacite -> andesite -> basalt inclusion, 400x point counts (1.6-3.2k points),
  explicit groundmass. Modes from `table1.gif`, chemistry from `table7p*.gif`
  (`chem_status=table7`). 5 of the 6 carry chemistry: Table 1's andesite lava
  `CN6791-h` has no Table 7 row, and the nearest is `CN6791d`, a different
  split of the same locality and date, so it keeps none rather than borrow one
  (`chem_status=no_table7_row`). Y, Nb and V are absent from Table 7, so those
  traces pad on these rows.
  https://pubs.usgs.gov/pinatubo/pallister/
- **Mount St. Helens 2004-06** (Pallister et al. 2008, USGS PP1750 ch.30).
  7 dome samples, dacite, 1,000-1,300-pt optical modes, matrix=groundmass
  explicit. Modes transcribed from the chapter PDF Table 2; chemistry joined from
  the Thornber OFR 2008-1130 catalog (`of2008-1130_v1.1.xls`). All show coexisting
  hornblende + hypersthene -- a real amp+opx dacite.
  https://pubs.usgs.gov/pp/1750/ , https://pubs.usgs.gov/of/2008/1130/

**Coverage tier -- visual phenocryst estimates (coarser, +/-5-10 vol%), no split:**
- **du Bray et al. USGS DS 764** (Bodie Hills) and **DS 1099** (Tonopah/Divide/
  Goldfield). ~717 samples, basalt -> rhyolite, ancestral Cascades arc. XRF +
  major/trace joined to visual phenocryst abundance estimates on `Field_ID`.
  Machine-readable Excel; the bulk of the benchmark. Serves presence + coarse
  magnitude only. `AltExtnt` 1-2 = least-altered (freshness QC).
  https://pubs.usgs.gov/ds/764/ , https://pubs.usgs.gov/ds/1099/
- **DS 1099 Appendix 10** -- microscope point counts of the Fraction and Heller
  Tuff samples, 95 rocks at 389-867 points each, of which 70 are already in the
  benchmark under a visual estimate. Machine-readable, so unlike the two
  eruptions above it needed no transcription. These rows are emitted *beside*
  the visual ones rather than replacing them, and `load_benchmark`'s
  `prefer_basis` picks which a scoring run reads; the default is the visual
  estimate every number on record was measured against. Appendix 10 counts five
  of the seven scored groups and never looks for pyroxene or olivine.

## Schema

`samples.csv` (one row per sample): `source, sample_id, unit, rock_class`,
oxides (`sio2 tio2 al2o3 feo_tot mno mgo cao na2o k2o p2o5 loi`, wt%), traces
(`sr_ppm zr_ppm ba_ppm y_ppm rb_ppm ni_ppm nb_ppm v_ppm la_ppm ce_ppm nd_ppm
yb_ppm th_ppm`, du Bray only), `total_crystals_pct`, `alteration_extent`,
`texture`, `mode_basis`, `n_points`, `chem_status`. Column names match the
encoder tokenizer (`cnmineral.data.bulk_chemistry`): the majors + `<sym>_ppm`
traces feed it directly; `feo_tot` resolves via `_resolve_iron`; `cr2o3` absent
so Cr pads (matches `include_cr2o3=False`); `mno/p2o5/loi` are ignored (no token).
A du Bray sample tokenizes to 21/22 populated tokens; MSH majors-only (traces
pad, in-distribution); Pinatubo carries Table 7 majors + 10 of 13 traces.
`n_points` is the section total for the point-counted rocks (Pinatubo and the
Appendix 10 rocks); `mode_basis` names a rock's primary measurement, but
`modes.csv`'s per-row `basis` is what decides which rows a load reads.

`modes.csv` (one row per sample x present phase x measurement): `source,
sample_id, phase` (as-reported name), `magemin_phase` (mapped),
`abundance_pct`, `qualifier` (tr / <1 / X when not numeric), `basis`. Absent
phases are omitted, not zero-rowed. A rock measured twice carries a row set per
basis, and `load_benchmark` reads one of them -- summing them would double the
same thin section.

Phase -> MAGEMin crosswalk: plagioclase & alkali feldspar -> `fsp`; hornblende &
cummingtonite -> `amp`; augite -> `cpx`; hypersthene & bronzite -> `opx`;
olivine -> `ol`; biotite -> `bi`; quartz & silica phases -> `q`; Fe-Ti oxides ->
`opq_unsplit`; groundmass -> `groundmass` (the real-rock counterpart of predicted
`M_liq` + microlites, used only when a split exists). `opq_unsplit` and du Bray's
`px_unsplit` need a downstream rule to compare against MAGEMin's split
spl/ilm/mag and cpx/opx.

## Current contents

730 samples: du Bray 717, MSH 7, Pinatubo 6. SiO2 49-93 wt%, 322 in the 52-63%
intermediate band. 724 carry both chemistry and modes; 70 of the du Bray rocks
carry a point count as well as a visual estimate. Regenerate: `conda run -n
mineral python explorations/real_mineralogy/harvest.py`. The scored-set
breakdown lives in `results/benchmark_composition.md`, generated rather than
hand-counted.

## Model compatibility

- **Input (samples.csv -> encoder): ready.** Column names match the tokenizer;
  values in the training units (oxide wt%, trace ppm). Feed real samples through
  the SAME `bulk_chemistry` path so Fe split / normalization match the corpus.
- **Label (modes.csv -> ModeHead): needs an adapter, not drop-in.** Three gaps:
  (1) units -- modes are volume percent (point counts) / percent, MAGEMin modes
  are weight fraction, so convert vol% -> wt% (mineral densities) and /100;
  (2) unsplit phases -- `opq_unsplit` compares against MAGEMin `spl+ilm(+mag)`
  summed, du Bray `px_unsplit` against `cpx+opx` summed;
  (3) melt/groundmass -- use the hand-sample view (drop model liq/fl, renormalize
  solids; drop real `groundmass`, renormalize) for magnitude, or compare model
  liq-fraction vs real groundmass-fraction. Presence comparison needs none of
  this. See docs/heads.md output views.

## Scoring this benchmark

`cnmineral.eval.real_rock` is the instrument (CLI `scripts/eval/real_rock.py`,
output `runs/real_rock/<tag>/`, scores in `results/real_rock_ledger.yaml`). It
owns everything the "model compatibility" gaps above called for: the 21-phase
vocabulary coarsens onto the 7 groups this data resolves (`MODEL_TO_GROUP` /
`TRUTH_TO_GROUP`, so `opq_unsplit` meets spl+ilm and `px_unsplit` meets
cpx+opx), vol% converts to wt% by per-reported-name density
(`MINERAL_DENSITY`) before coarsening, and groundmass drops so the truth
vector renormalizes to the crystal cargo.

Four data facts that surface only here, all encoded in the loader:

- **The abundances are volume percent, verified against the publications, not
  the files.** du Bray's workbooks never label the abundance block; DS 764's
  publication text states the units, DS 1099's Appendix 10 point counts are
  volume by construction (points / total), and MSH's ch30 says "Optical modes,
  as volume percent". Settled -- do not re-open. It matters: skipping the
  vol% -> wt% conversion moves the headline 2.4 nats, more than the entire
  plausible density band, because the conversion is what exposes the Fe-Ti
  oxide under-sizing (journal 2026-07-24).
- **`TR` / `<1` rows carry no number** (~500 of 3245, a quarter of amphibole's
  appearances). They are present, and left-censored: the abundance is known
  only to sit below du Bray's 0.5 vol% reporting grid. Scored by that bound,
  never imputed -- phenocryst totals are small enough that any imputed
  constant lands inside the range of genuinely-reported values.
- **A reported `0.0` is an absence**, not a presence (3 MSH `silica phases`
  rows): the phase was looked for and not found.
- **`alteration_extent` is a string with modifiers** (`1+`, `2-`), so the
  freshness cut parses the leading digit; `fresh` is grade 1.

## Generated summaries

- `run_composition_tables.py` -> `results/benchmark_composition.{md,json}`: the
  funnel from what USGS published down to the scored set, plus the survey /
  ruler / unit-structure / split / silica / mineral breakdowns. The paper's
  benchmark appendix reads from these rather than from hand-copied numbers.
- `run_pointcount_contrast.py` -> `results/<tag>/pointcount_contrast.{json,png}`:
  the frozen stack scored against both rulers on the 64 rocks that carry both.
- `run_unit_leakage.py` -> `results/<tag>/unit_leakage.json`: whether the
  fragmented unit key leaks eruptions into val, measured rather than repaired
  (the whitespace / case fragmentation has since been repaired in the loader;
  the `?` qualifiers and sub-facies suffixes this rule merges have not).
  An over-permissive merge rule flags every val rock that might share an
  eruption with a train rock, and val is scored with and without them under the
  same fitted layer.
- `run_scale_convention.py` -> `results/<tag>/scale_convention.json`: the frozen
  stack scored on both analytical-total conventions (issue #56). du Bray's
  majors are published recalculated to 100 volatile-free, which the workbooks'
  `Total_I_pct` / `Volatile_pct` pair makes invertible per rock, so the
  benchmark can be put back on the raw convention gard2019 carries.

## TODOs before this is a locked benchmark

- freshness QC beyond `AltExtnt` (oxide totals / LOI thresholds).
- the harvest reads 8 of 42 petrography columns and 24 of 68 chemistry columns,
  and never opens Appendix 1 (the sample catalog) in either source. What the
  DS 764 / DS 1099 pamphlets' "Data Fields" sections define the unread ones to
  be, checked against the files:
  - `Cpx_Opx` is presence plus a dominance ordering (`C`, `O`, dominant
    uppercase, subordinate lowercase, both uppercase = roughly equal), **not a
    numeric ratio**. 357 benchmark rocks carry it; 299 name a single pyroxene,
    so the `px` group the eval lumps splits exactly on those, and the other 58
    would need an invented dominance ratio to split by magnitude.
  - `ClrIndx` is *defined* as the sum of the hbl/bt/px/ol/opq abundances, and
    is exactly that sum in the files (745 of 747 DS 764 rows, 416 of 416
    DS 1099, the two exceptions being source typos). It is arithmetic on
    columns already read -- **not** an independent cross-check on the
    abundances.
  - `Total_I_pct` is the pre-recalculation analytical total and `Volatile_pct`
    its volatile part; together they reconstruct the un-recalculated majors
    that issue #56 needs. The published majors are recalculated to 100
    volatile-free (confirmed: they sum to 100.00 at p5, median and p95), and
    `(Total_I_pct - Volatile_pct)/100` is the per-rock factor that
    recalculation applied -- median 1.7% on DS 764, 3.2% on DS 1099.
  - provenance is `Chem_Src` in DS 764's **Appendix 1** and `chem_src` in
    DS 1099's Appendix 3, covering 713 of 717 rocks -- but constant at source 4
    (du Bray and others, unpublished USGS data) on all 509 DS 764 benchmark
    rocks that carry it, so the variance is confined to the 204 DS 1099 rocks.
  - Appendix 1 also carries `Strat_Name` (a second unit labeling, below),
    `Alt_Desc` (the alteration descriptors the freshness TODO above wants),
    `Ign_Form` (lava / ash-flow tuff / plug) and coordinates.
- unit labels are the raw source strings, so one eruption can appear under
  several spellings (`Fraction Tuff` / `FractionTuff`, trailing spaces, `?`
  qualifiers, `, pumice` suffixes): 148 raw spellings collapse to 141 on case
  and whitespace alone. That fragmented the clusters the split and the bootstrap
  group by, and one rock reached val whose 36 same-unit siblings were in train.
  Since 2026-09-02 `load_benchmark` canonicalizes the whitespace / case
  variants (`canonical_units`) and the roster was rebuilt on the canonical
  key; the `?` qualifiers and sub-facies suffixes are left distinct.
  Appendix 1's `Strat_Name` is a second labeling at the same
  granularity (144 distinct either way) that disagrees with the petrography
  column on 91 rows, some substantively rather than by spelling -- four rocks
  logged as `Heller Tuff` there are `Fraction Tuff` in Appendix 1 -- and
  regrouping on it would straddle four keys rather than one. So it is not a
  drop-in authority, and it stays unrepaired.
- more precision-tier eruptions (Unzen, Soufriere Hills, Colima, Mazama) if the
  benchmark needs weight beyond the coverage tier.
- nothing pins which revision of the USGS workbooks is on disk: the raw files
  and the generated tables are gitignored, with no checksum or download
  manifest. Appendix 10's derived-percent check is the only such guard.
