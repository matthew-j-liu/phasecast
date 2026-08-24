# EDA on the real-mineralogy benchmark

Plots for the endpoint/label-granularity decision. Regenerate:
`conda run -n mineral python explorations/real_mineralogy/eda.py`.

## Plots
1. `1_coverage.png` -- SiO2 histogram + TAS-style scatter by source. du Bray spans
   basalt->rhyolite (arc, intermediate-heavy); MSH/Pinatubo are the dacite point-count anchors.
2. `2_granularity.png` -- **the crux.** Phenocryst presence frequency vs SiO2 for the three
   "granularity" minerals (quartz, biotite, alkali feldspar) and, for contrast, the
   mafic/intermediate phases.
3. `3_crystallinity.png` -- total crystals % vs SiO2 and its spread (the axis chemistry can't set).
4. `4_amp_vs_px.png` -- amphibole vs pyroxene footprint vs SiO2 (the headline switch in real rocks).
5. `5_richness_cooccurrence.png` -- phenocryst-phase count vs SiO2 + co-occurrence matrix.
6. `6_visual_vs_pointcount.png` -- visual estimate against point count on the 70 rocks
   measured both ways (DS 1099 Appendix 10), paired per rock so the ruler is not confounded
   with the suite. The model-side version of this contrast is
   `run_pointcount_contrast.py`.

## The granularity question, answered

Concern: a phenocryst-regime synthetic label predicts phenocrysts + a lumped groundmass, so it
would "not predict" quartz / biotite / alkali feldspar -- phases mineralogists routinely label.

The data says those three are **felsic-restricted phenocrysts** (du Bray, presence includes trace):

| mineral | n | median SiO2 | frac occurring <63% SiO2 | frac <52% (mafic) |
| --- | --- | --- | --- | --- |
| quartz | 242 | 73.6 | 0.14 | 0.00 |
| alkali feldspar | 203 | 74.4 | 0.06 | 0.00 |
| biotite | 443 | 65.3 | 0.29 | 0.00 |
| olivine (contrast) | 110 | 57.1 | 0.90 | 0.04 |

None of the three appears as a phenocryst in a mafic rock (0% below 52% SiO2); quartz and alkali
feldspar are essentially rhyolite-only (median ~74), biotite is dacite-up. So a phenocryst-regime
label does not categorically lose them -- it captures them in felsic rocks (where they ARE
phenocrysts) and correctly omits them in mafic rocks (where they are groundmass-only or absent).

**The residual open item** (not resolved by this EDA): whether a single endpoint crystallinity
saturates quartz/sanidine/biotite in felsic bulks, since these are late (low-T) phases. That is the
composition-vs-endpoint interaction the corpus design must handle -- felsic bulks likely need a
deeper crystallinity stop than mafic ones to reveal their diagnostic phenocrysts.

## Caveats
- du Bray modes are visual estimates (717 of 730 samples); presence is trustworthy, magnitudes coarse
  (see plot 6). MSH/Pinatubo are point counts but only 13 samples, dacite.
- The EDA predates the Pinatubo chemistry transcription, so its SiO2-axis plots are du Bray + the
  6 MSH dacites; 5 Pinatubo rocks now carry chemistry and would join them on a rerun.
- "present" counts trace occurrences (`tr` / `<1`); ~490 mode rows are trace-level.
