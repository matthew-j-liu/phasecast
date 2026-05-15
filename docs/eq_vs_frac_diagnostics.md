# eq-vs-frac T/H2O diagnostics (run 2026-05-13, written 2026-05-15)

Diagnostic tables for the `data/experiments/eq_vs_frac/` run. Purpose:
expose the T configuration of each leg, the H2O distribution per class,
and the empirical residual-melt fractions, so the redesign of the
experiment can pick `T_eq` and `T_FC_end` more carefully.

## 1. Procedure recap

- **eq leg**: one `MAGEMin Initialize_MAGEMin("ig")` call per bulk at
  `T_eq = (T_lo + T_hi)/2`, `P_kbar` per class. Closed system. Reports
  the full equilibrium assemblage including any residual `liq`.
- **FC leg**: `ptt.isobaric_crystallisation(Model="Green2025",
  Frac_solid=True, dt_C=25)`. Cools from `T_start_C` (~50-100 C above
  the class Putirka upper-bound liquidus) to `T_FC_end = T_lo`. At each
  step the solids that crystallized are removed and accumulated; the
  remaining melt enters the next step. Reports the accumulated solids
  + residual `liq` at `T_FC_end`.

Both legs use the same bulk composition (with the H2O strategy already
applied upstream in `02_filter_complete.py`) and the same `P_kbar`.

## 2. Per-class T configuration as run

`gap` is `T_eq - T_FC_end`. **Both legs should terminate at the same T
for a clean redo**; this column quantifies the artifact size.

| class        | N  | P (kbar) | T_start | T_eq | T_FC_end | gap (C) |
|--------------|----|----------|---------|------|----------|---------|
| basalt       | 50 | 3        | 1350    | 925  | 850      | 75      |
| andesite     | 50 | 3        | 1200    | 825  | 750      | 75      |
| dacite       | 50 | 2        | 1100    | 725  | 650      | 75      |
| rhyolite     | 50 | 2        | 1000    | 625  | 550      | 75      |
| monzogranite | 50 | 3        | 1000    | 600  | 550      | 50      |
| granodiorite | 50 | 3        | 1050    | 650  | 600      | 50      |
| diorite      | 50 | 3        | 1150    | 825  | 750      | 75      |
| gabbro       | 50 | 5        | 1300    | 925  | 850      | 75      |
| dunite       | 50 | 10       | 1500    | 1000 | 900      | 100     |
| foidite      | 50 | 3        | 1300    | 950  | 850      | 100     |
| syenite      | 50 | 3        | 1050    | 725  | 650      | 75      |

## 3. Wet/dry solidus reference values

From Best 2003, Putirka 2008, Wallace 2005, Burnham 1979. Wet solidus
values assume H2O-saturation at the given P; under-saturated bulks
have higher solidi. Used here as a sanity benchmark, not a per-bulk
prediction.

| class        | dry solidus (~C) | wet solidus at ~3-5 kbar (~C) |
|--------------|------------------|-------------------------------|
| basalt       | 1100             | 950-1000                      |
| andesite     | 1000             | 800-850                       |
| dacite       | 950              | 700-750                       |
| rhyolite     | 900              | 650-700                       |
| monzogranite | 950              | 650                           |
| granodiorite | 950              | 670-700                       |
| diorite      | 1000             | 800-850                       |
| gabbro       | 1100             | 950-1000                      |
| dunite       | 1500             | 1100                          |
| foidite      | 1200             | 1000-1100                     |
| syenite      | 900-950          | 700-750                       |

## 4. Bulk H2O distribution per class (in the 50 bulks we ran)

Wt% on the input bulk. Comes from the class-conditional H2O strategy
in `src/cnmineral/data/h2o.py` plus the small fraction (~1%) with
measured h2o_tot.

| class        | h2o p25 | h2o median | h2o p75 | h2o max |
|--------------|---------|------------|---------|---------|
| basalt       | 0.38    | 0.54       | 0.78    | 0.99    |
| andesite     | 2.81    | 3.89       | 4.83    | 5.61    |
| dacite       | 4.03    | 4.97       | 6.00    | 6.55    |
| rhyolite     | 3.24    | 4.74       | 5.42    | 6.51    |
| monzogranite | 3.59    | 4.39       | 5.29    | 5.69    |
| granodiorite | 3.58    | 4.56       | 5.12    | 5.73    |
| diorite      | 2.82    | 3.28       | 4.28    | 5.00    |
| gabbro       | 0.89    | 1.26       | 1.73    | 2.19    |
| dunite       | 0.02    | 0.05       | 0.08    | 0.12    |
| foidite      | 2.22    | 3.26       | 4.14    | 5.69    |
| syenite      | 1.83    | 2.56       | 3.34    | 3.87    |

## 5. Empirical melt fractions: what actually happened at T_eq and T_FC_end

This is the diagnostic that matters. `liq_eq` is the residual silicate
melt mass fraction predicted by the equilibrium calculation at `T_eq`.
If it is ≫ 0, the eq snapshot is sampling melt-plus-crystals — i.e.
`T_eq` is sitting in the wet-melting interval for that class's H2O.
`liq_fc` is the residual melt mass fraction at `T_FC_end` after the
full cooling path.

| class        | T_eq | T_FC_end | mean liq_eq | median liq_eq | bulks w/ liq_eq>0.5 | mean liq_fc | median liq_fc |
|--------------|------|----------|-------------|----------------|---------------------|--------------|----------------|
| basalt       | 925  | 850      | 0.124       | 0.118          | 0%                  | 0.124        | 0.127          |
| andesite     | 825  | 750      | 0.450       | 0.465          | 36%                 | 0.396        | 0.409          |
| dacite       | 725  | 650      | 0.453       | 0.463          | 42%                 | 0.000        | 0.000          |
| rhyolite     | 625  | 550      | 0.000       | 0.000          | 0%                  | 0.002        | 0.000          |
| monzogranite | 600  | 550      | 0.001       | 0.000          | 0%                  | 0.002        | 0.000          |
| granodiorite | 650  | 600      | 0.000       | 0.000          | 0%                  | 0.000        | 0.000          |
| diorite      | 825  | 750      | 0.181       | 0.184          | 0%                  | 0.183        | 0.179          |
| gabbro       | 925  | 850      | 0.065       | 0.042          | 0%                  | 0.187        | 0.190          |
| dunite       | 1000 | 900      | 0.030       | 0.007          | 0%                  | 0.082        | 0.054          |
| foidite      | 950  | 850      | 0.281       | 0.272          | 16%                 | 0.334        | 0.356          |
| syenite      | 725  | 650      | 0.168       | 0.168          | 0%                  | 0.009        | 0.000          |

## 6. Diagnosis: which classes were actually subsolidus at T_eq

Three regimes emerge.

### A. Truly subsolidus at both T_eq and T_FC_end (clean comparison)

`liq_eq ≈ 0` and `liq_fc ≈ 0`. The L1 here measures path-dependence
only — fractional accumulation of early-crystallized solids vs the
single closed-system snapshot.

| class        | liq_eq mean | liq_fc mean | observed L1 (median) |
|--------------|-------------|-------------|----------------------|
| rhyolite     | 0.000       | 0.002       | 0.131                |
| monzogranite | 0.001       | 0.002       | 0.157                |
| granodiorite | 0.000       | 0.000       | 0.214                |

These are the only classes where the experiment's comparison is
mechanically clean. The 0.13-0.21 L1 here is the **true
path-dependence signal**.

### B. eq retains substantial liq at T_eq — eq is in the wet-melting interval

`liq_eq` is large (>0.15) AND much larger than `liq_fc`. The eq leg is
"magma in progress" while the FC leg has cooled into the crystallized
endpoint. L1 here is **dominated by terminal-T asymmetry plus
wet-solidus-above-T_eq**, not by path-dependence.

| class    | liq_eq mean | liq_fc mean | h2o median | observed L1 (median) |
|----------|-------------|-------------|------------|----------------------|
| dacite   | 0.453       | 0.000       | 4.97       | 1.144                |
| syenite  | 0.168       | 0.009       | 2.56       | 0.411                |
| andesite | 0.450       | 0.396       | 3.89       | 0.376                |

Note dacite vs andesite: both have ~0.45 liq at T_eq, but andesite FC
also retains ~0.40 liq while dacite FC clears to ~0.00. The
catastrophic dacite L1 is driven by the **75 C terminal-T gap pushing
dacite FC just below its wet solidus while leaving dacite eq still
above it**. Andesite's smaller L1 reflects that both legs are above the
wet solidus.

### C. FC retains MORE liq than eq — fractionation-trapped residual melt

`liq_fc` > `liq_eq`. Counter-intuitive but real: FC removes solids as it
cools, enriching the residual melt in incompatibles (Si, K, H2O) until
a small evolved-melt fraction is stable down to `T_FC_end`. The closed-
system eq snapshot doesn't go through that enrichment.

| class   | liq_eq mean | liq_fc mean | h2o median | observed L1 (median) |
|---------|-------------|-------------|------------|----------------------|
| gabbro  | 0.065       | 0.187       | 1.26       | 0.674                |
| dunite  | 0.030       | 0.082       | 0.05       | 0.486                |
| foidite | 0.281       | 0.334       | 3.26       | 0.413                |

This is a **real path-dependence signal**: gabbro's L1 of 0.67 mostly
comes from FC accumulating early-Mg-rich olivine + pyroxene that the eq
snapshot at 925 C doesn't preserve, plus the trapped evolved melt.
Petrologically meaningful, not an artifact.

### D. Roughly matched

| class  | liq_eq mean | liq_fc mean | h2o median | observed L1 (median) |
|--------|-------------|-------------|------------|----------------------|
| basalt | 0.124       | 0.124       | 0.54       | 0.292                |
| diorite| 0.181       | 0.183       | 3.28       | 0.547                |

Diorite's high L1 with matched liq fractions is interesting — it's
~pure path-dependence (no T or solidus confound). 0.55 is a strong
path-dependence signal for an intermediate plutonic.

## 7. Effects of water — summary

- **Class h2o ranking** (median): dunite (0.05) ≪ basalt (0.54) <
  gabbro (1.26) < syenite (2.56) < foidite (3.26) ≈ diorite (3.28) <
  andesite (3.89) < monzogranite/granodiorite/rhyolite (4.4-5.0) <
  dacite (4.97).
- **The wet-solidus depression is the dominant water effect we're
  hitting.** For dacite at 5 wt% H2O, the wet solidus drops well below
  the canonical 750 C — our `T_eq=725 C` ends up at or above the
  actual solidus. Same story for andesite (3.9 wt%, T_eq=825 C vs wet
  solidus ~750-800 at this h2o) and syenite (2.6 wt%, T_eq=725 C vs wet
  solidus ~700-750).
- **Felsic classes (rhyolite, granite, granodiorite) have high H2O but
  also have low T_eq** (600-650 C). Their T_eq sits *below* their wet
  solidus, so eq is subsolidus and liq_eq=0. These are the classes
  that worked.
- **The class redo strategy is not "lower T_eq uniformly."** Lowering
  T_eq below 550 C for granite would push MAGEMin into metamorphic
  drift (greenschist assemblages). The redo needs to pick T per
  bulk based on the bulk's actual wet solidus, then anchor both legs
  to the same final T.

## 8. Implications for the redo

Three design corrections, in priority order:

1. **Anchor both legs at the same terminal T.** Currently eq runs at
   `T_eq = midpoint`, FC ends at `T_FC_end = T_lo`. Drop the midpoint
   and run both at `T_lo` (or at a per-bulk-chosen solidus-minus-50).
   This alone removes the 50-100 C gap.
2. **Pick T per bulk based on its actual wet solidus**, not class-
   level constants. For each bulk, bracket the wet solidus with a
   ~3-call MAGEMin search (e.g. binary search between T_dry-150 and
   T_wet-50), then run eq + FC anchored 25-50 C below that. This
   eliminates the "T_eq sits above wet solidus for hydrous bulks"
   problem that drove the dacite blowup.
3. **Renormalize the comparison.** Decide whether `liq` is part of
   the modal vector or excluded. If both legs are below the wet
   solidus, residual liq ≈ 0 for both and the question is moot. If
   they aren't, the residual liq channel dominates L1 in a way that
   isn't physically the same as path-dependence and the comparison
   conflates "trapped FC melt" with "eq mid-melt snapshot." Suggest:
   re-run with anchored terminal T, then compute L1 both with and
   without the liq channel; the with-liq version is the
   "what users will see" metric, the without-liq version is the
   "true crystalline assemblage divergence" metric.

The clean-class L1s (rhyolite 0.13, monzogranite 0.16, granodiorite
0.21) and the path-dependence-only diorite L1 (0.55) suggest the
true path-dependence signal is in the 0.1-0.6 range — still above
the 0.10 "adopt eq" threshold for most plutonics. So the **final
decision to adopt FC is likely robust to fixing these design issues**,
but the per-class numbers will compress and dacite will drop from
1.14 to something probably 0.3-0.5.
