# Decision 003: Tiered H2O handling strategy

**Date:** 2026-05-12
**Status:** Active

## Decision

Apply a three-tier H2O strategy at the bulk → MAGEMin schema-conversion step,
mirroring `fe_redox.py`:

- **T1 (measured):** if `complete.csv` reports `h2o_tot` for a row, use it
  directly. No assumption. Affects ~1% of rows.
- **T2 (class-conditional sample):** for the other ~99%, draw an H2O value
  uniformly from a per-qap_name range. Defaults from Wallace 2005 (MORB/OIB),
  Plank et al. 2013 (arc compositions), Burnham 1979 + Holtz et al. 2001
  (granitic systems), Best 2003 (general).
- **T3 (does not exist):** the `default` range is always defined as a fallback
  for unknown classes; no row drops on H2O grounds.

Each training tuple carries an `h2o_provenance` label (`measured` or
`assumed_class`) parallel to `fe_provenance`, so we can stratify evaluation
by H2O-tier.

## Why per-class **ranges**, not point estimates

Petrologists agree on central tendencies per class but the within-class
variance is genuinely large:
- MORB: 0.1-0.5 wt% (modal ~0.2)
- Arc basalt: 2-6 wt% (sometimes up to 8 in wet arcs)
- Granite I-type: 3-5; S-type: 4-7; A-type: 1-3
- Mantle peridotite: 0.005-0.1 (essentially dry)

Using point estimates would (a) commit to a precision the literature doesn't
support and (b) draw skeptical reviewer responses. Ranges + uniform sampling
mirrors the `TPRangeStrategy` pattern and stays consistent with how we treat
T/P priors.

## Alternatives considered

**(A) Skip H2O fix; document hydrous-phase gap as known limitation.**
Rejected. The paper's differentiation table claims hydrous-phase coverage as
one of four axes. Amp 0.3% / bi 0.2% in training labels would force an
asterisk that materially weakens the contribution. Also: training the ModeHead
on degenerate amp/bi labels collapses those gates to permanent zero — retraining
to recover them later means starting from scratch, not incrementally improving.

**(B) Per-class point estimates (e.g., MORB = 0.3, granite = 4).**
Rejected. Point estimates ignore within-class variance and are less defensible
in the methods section. Geologist reviewers would push back on any single
number for "the" H2O content of a rock class.

**(C) Fit H2O empirically from major-element composition.**
Considered briefly — papers like Plank et al. (2013) fit fO2 / H2O from
correlations with K2O or alkalis. Rejected for v1 as overkill: adds another
calibration to validate, and the per-class uniform-range approach is already
defensible. Could revisit for v2 if class-conditional sampling produces visible
artifacts in the trained model.

## Why this scope

- Symmetric to existing `fe_redox.py` pattern → reviewers can see we treat
  oxidation state and water content with the same methodological rigor
- Toggleable via strategy injection — sensitivity analysis is one config change
- Provenance tracking carries through to H5 → stratified eval is possible
- Cheap (~half a day) and orthogonal to model architecture work

## Revisit when

- Training shows amp/bi predictions remain degenerate despite the H2O fix —
  could indicate ranges are still too narrow for arc / dacite / granite classes
- Real-rock evaluation on EDS-map-derived bulks shows systematic
  underprediction of hydrous phases at conditions where geologists would expect
  them — could indicate per-class central tendencies are off
- Reviewers push back on the per-class table → swap to Plank/Wallace
  literature-fit composition-conditional model (option C above)

## Cross-reference

- Journal entry: `docs/JOURNAL.md` 2026-05-12 "Hydrous-phase gap forces a
  v0-blocking design decision"
- Implementation: `src/cnmineral/data/h2o.py` (TBW); integration in
  `src/cnmineral/data/schema.py`
- Symmetric pattern: `src/cnmineral/data/fe_redox.py`, `src/cnmineral/data/tp_priors.py`
