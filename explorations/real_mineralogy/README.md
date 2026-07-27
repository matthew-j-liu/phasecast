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
pad, in-distribution); Pinatubo has no chemistry yet.

`modes.csv` (one row per sample x present phase): `source, sample_id, phase`
(as-reported name), `magemin_phase` (mapped), `abundance_pct`, `qualifier`
(tr / <1 / X when not numeric), `basis`. Absent phases are omitted, not zero-rowed.

Phase -> MAGEMin crosswalk: plagioclase & alkali feldspar -> `fsp`; hornblende &
cummingtonite -> `amp`; augite -> `cpx`; hypersthene & bronzite -> `opx`;
olivine -> `ol`; biotite -> `bi`; quartz & silica phases -> `q`; Fe-Ti oxides ->
`opq_unsplit`; groundmass -> `groundmass` (the real-rock counterpart of predicted
`M_liq` + microlites, used only when a split exists). `opq_unsplit` and du Bray's
`px_unsplit` need a downstream rule to compare against MAGEMin's split
spl/ilm/mag and cpx/opx.

## Current contents

730 samples: du Bray 717, MSH 7, Pinatubo 6. SiO2 49-93 wt%, 321 in the 52-63%
intermediate band. Regenerate: `conda run -n alignn_fix python
explorations/real_mineralogy/harvest.py`.

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

## TODOs before this is a locked benchmark

- Pinatubo whole-rock chemistry (Table 7 transcription) -- 6 rows. Until they
  land the precision tier is 6 scoreable rocks (MSH), not 13.
- freshness QC beyond `AltExtnt` (oxide totals / LOI thresholds).
- a noise floor: most of the set is visual estimates at +/-5-10 vol%, so part
  of any measured gap is the measurement. 6 point-counted rocks cannot pin it.
- more precision-tier eruptions (Unzen, Soufriere Hills, Colima, Mazama) if the
  benchmark needs weight beyond the coverage tier.
