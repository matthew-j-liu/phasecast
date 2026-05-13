# MAGEMin pipeline handoff

Self-contained brief covering everything you need to invoke our MAGEMin pipeline
or extend it. The source code lives in the sibling repo at `CrabNet-Mineral` on
the `structure-edm` branch under `alignn/magemin/`. Stage 1 has been run end to
end (3,707 / 4,508 points converged); Stage 2 (production scale) is wired up
but never executed.

---

## 1. How MAGEMin is invoked from Python

**Not directly.** Python and Julia communicate via files, not in-process bindings.

The pipeline is three scripts, run sequentially:

```
complete.csv ──> filter_complete_csv.py ──> inputs_stage{N}.csv
                                                    │
                                                    ▼  julia subprocess (CLI)
                                              run_magemin.jl
                                                    │
                                                    ▼
                                              results_stage{N}.csv
                                                    │
                                                    ▼  python
                                       parse_magemin_output.py
                                                    │
                                                    ▼
                                          train_magemin_stage{N}.h5
```

The actual MAGEMin call lives entirely inside `run_magemin.jl`. From Python, you
just spawn julia as a subprocess:

```bash
julia --project=alignn/magemin -t auto alignn/magemin/run_magemin.jl --stage 1
```

Inside Julia, we use the **`MAGEMin_C.jl`** package directly (the official Julia
wrapper around the MAGEMin C library):

```julia
using MAGEMin_C
data = Initialize_MAGEMin("ig", verbose=false)
results = multi_point_minimization(
    P_vec, T_vec, data;
    X = X_vec,          # Vector{Vector{Float64}}, one composition per point
    Xoxides = OXIDES,   # oxide order, see §2
    sys_in = "wt",      # weight-percent input
    progressbar = true,
)
Finalize_MAGEMin(data)
```

**Why file-based IPC and not PyJulia / juliacall?** PyJulia is fragile when the
Julia process needs threading (MAGEMin parallelizes via `-t auto`), and the Julia
C-extension env (`MAGEMin_C` links a shared library) sometimes conflicts with
Python's loaded libstdc++/libgomp. The CSV-in / CSV-out hand-off is dumb but
reliable. Nested data (phase lists, SS compositions) is serialized as JSON
strings inside the CSV columns; the parser deserializes on the Python side.

**Install requirements:**
```bash
# Julia
brew install julia                  # or juliaup
# Inside the alignn/magemin/ project dir:
julia -e 'using Pkg; Pkg.add(["MAGEMin_C", "CSV", "DataFrames", "JSON", "ArgParse"])'
```

**Smoke test that MAGEMin itself works:**
```bash
julia -e 'using MAGEMin_C
data = Initialize_MAGEMin("ig", verbose=false)
out = single_point_minimization(10.0, 1100.0, data,
       X=[48.43,15.19,11.57,10.13,6.65,1.64,0.59,1.87,0.68,0.0,3.0],
       Xoxides=["SiO2","Al2O3","CaO","MgO","FeO","Fe2O3","K2O","Na2O","TiO2","Cr2O3","H2O"],
       sys_in="wt"); println(out.ph)'
```
Should print something like `["liq", "cpx", "ol", "spn", "amp", ...]`.

---

## 2. Input shape MAGEMin expects

**Database**: `"ig"` (igneous) — Holland et al. 2018 thermodynamic dataset.
Other databases exist (`"mp"` metapelite, etc.) but we haven't tested them.

**Oxide system**: 11-oxide igneous system, fixed order **must match `Xoxides`**:

```
["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3",
 "K2O", "Na2O", "TiO2", "Cr2O3", "H2O"]
```

Note ordering does not follow alphabetical or molecular weight — it matches
MAGEMin's internal expectation and `Xoxides`.

**Normalization**: oxides must **sum to 100 wt%** (LOI-free). The filter script
LOI-normalizes by scaling each row so its 11 oxides sum to 100. `sys_in="wt"`
tells MAGEMin to interpret the input as weight percent (the alternative is
`sys_in="mol"`).

**Composition CSV column names** (what `run_magemin.jl` reads from
`inputs_stage{N}.csv`):

```
sample_id, sio2, al2o3, cao, mgo, feo, fe2o3, k2o, na2o, tio2, cr2o3, h2o_tot
```

Notice the CSV columns are **lowercase**; the Julia code maps them to the
mixed-case `Xoxides` order before calling MAGEMin.

**T**: degrees Celsius, float. Our grid is `[800, 900, 1000, 1100, 1200, 1300, 1400]`.

**P**: kilobars, float. Our grid is `[1, 5, 10, 20]` kbar (= 0.1, 0.5, 1.0, 2.0 GPa).

**Sparse oxides allowed?** Cr2O3 and H2O often missing from real bulk analyses.
Stage 2 fills these with 0; MAGEMin handles `Cr2O3 = 0` and `H2O = 0` cleanly
(per docs). Don't pass NaN — pass zero.

**Multi-point call shape**: for N compositions × M = 28 (T, P) points, you pass

- `P_vec :: Vector{Float64}` length `N*M`
- `T_vec :: Vector{Float64}` length `N*M`
- `X_vec :: Vector{Vector{Float64}}` length `N*M`, each inner vector length 11

Order doesn't matter to MAGEMin, but we order as (composition, P, T) so
`out[i]` corresponds to a single (rock, P, T) cell. See `build_grid` in
`run_magemin.jl`.

---

## 3. Output schema MAGEMin returns

`multi_point_minimization` returns a `Vector{...}` of result structs, one per
(P, T, X) point. The fields we extract (see `extract_result` in
`run_magemin.jl`):

| Field | Type | Meaning |
|---|---|---|
| `out.status` | Int | 0 = converged. Non-zero = failure (codes documented in MAGEMin source) |
| `out.G_system` | Float64 | Total Gibbs energy of the equilibrium assemblage, kJ/mol |
| `out.ph` | Vector{String} | Phase abbreviations in the stable assemblage (e.g. `["ol", "cpx", "fsp"]`) |
| `out.ph_frac_wt` | Vector{Float64} | Wt fractions for each phase in `out.ph`, sums to ~1.0 |
| `out.SS_vec[i]` | SS struct | Per-phase SS detail (only meaningful for solution phases) |
| `out.SS_vec[i].emNames` | Vector{String} | Endmember names for solid solution phase `i` (e.g. `["fo", "fa", "mont", "cfm"]`) |
| `out.SS_vec[i].emFrac_wt` | Vector{Float64} | Endmember mass fractions; **can be negative** (basis projections) |
| `out.SS_vec[i].Comp_wt` | Vector{Float64} | Phase's bulk composition in the 11-oxide system (also available) |

After our Julia layer flattens everything to a CSV, the schema written to
`results_stage{N}.csv` is:

```
sample_id        int
P_kbar           float
T_celsius        float
status           int
G_system         float (kJ/mol)
phases_json      str — JSON of {"ph": [...], "ph_frac_wt": [...]}
ss_json          str — JSON of {"phase": {"emNames": [...], "emFrac_wt": [...]}, ...}
```

Failed points get `status != 0`, `G_system = NaN`, and `phases_json = ss_json = "{}"`.

**After `parse_magemin_output.py`** (the H5 schema for training):

```
/X         [N, 13]  bulk_comp (11 oxides) + T_celsius + P_kbar
/X_cols    str array, 13 names
/Y_wt      [N, n_phases]   mineral wt fractions; phases auto-discovered from data
/Y_cols    str array, n_phases phase names
/Y_ss      [N, n_ss_em]    per-mineral endmember fractions; NaN where phase absent
/Y_ss_cols str array, format "phase_em" e.g. "ol_fo", "fsp_ab"
/Y_energy  [N]             G_system
```

Stage 1 numbers: `n_phases = 20`, `n_ss_em = 103`.

---

## 4. Known failure modes / edge cases worth handling up front

**Convergence failures (~18% on our Stage 1).** Some (T, P, composition) cells
fail to converge. Check `status != 0` and drop them. Failures cluster at:
- Extreme cool / wet corners (e.g. 800 °C + 20 kbar + high H2O)
- Compositions far from the igneous training space of MAGEMin's database

If your data is sedimentary or strongly oxidized/aqueous, expect more failures.

**Phase abbreviations have drifted from the 2022 paper.** The paper lists `hb`,
`pl4T`, `spn`. Current MAGEMin returns `amp`, `fsp`, `spl`. **Do not hardcode
the phase set** — discover it from `out.ph` across your dataset. Our parser
does this; copy that pattern.

**Olivine has 4 endmembers, not 2.** The current database returns
`["fo", "fa", "mont", "cfm"]` for olivine, not just `["fo", "fa"]`. Don't
assume binary SS topology when computing things like Fo#. Same for other SS
phases — endmember count varies.

**Endmember fractions can be negative.** `emFrac_wt` values are basis
projections, not physical mole fractions; mathematically valid even when
negative. **Preserve as-is**; do not clamp to `[0, 1]` or you'll corrupt the
data. Treat them as features, not probabilities.

**`G_system` is at finite (T, P), not 0 K formation energy.** Range is roughly
`-1062` to `-631` kJ/mol on our grid. This is the **total** Gibbs energy of the
assemblage including thermal entropy and PV work — **not** the same physical
quantity as `ΔG_f` from JANAF or ALIGNN/DFT 0 K formation energies (which are
typically `-1` to `-3` eV/atom, or `-200` to `-500` kJ/mol formula unit). Don't
compare directly. If you use it as an ML target, **z-score standardize** so it
doesn't blow up your loss scale.

**Wt fractions don't sum to exactly 1.** Numerical tolerance puts them at
~0.997 ± 0.005 in our Stage 1. If something downstream needs strict sums-to-1,
re-normalize per row.

**`H2O` and `Cr2O3` missing in real data (~98% and ~97% null in
complete.csv).** Per MAGEMin docs, setting them to 0 is safe and that's what
Stage 2 does. Don't pass NaN.

**ArgParse.jl `parse_args` shadows the package function.** If you write more
Julia, name your CLI parser something other than `parse_args` (ours is
`parse_cli`).

**One Julia process per shell invocation.** Each `julia` invocation re-JITs
MAGEMin (a few seconds). If you batch many small runs from Python, prefer one
long Julia process over many short ones. The `multi_point_minimization`
function is designed for this: pass thousands of (P, T, X) tuples in one call,
not one tuple per call.

**Threading.** `julia -t auto` uses all available cores. `multi_point_minimization`
auto-parallelizes across points. Our Stage 1 throughput was 10.8 ms/point on
a single 8-core node. Don't over-thread; one Julia process with `-t auto` is
usually optimal.

---

## 5. How `complete.csv` was filtered to the 161-row Stage 1 set

Concrete answers to filter-logic questions, with the relevant source linked.
The full filter script is ~80 lines:
`CrabNet-Mineral/alignn/magemin/filter_complete_csv.py` (branch
`structure-edm`).

### 5.1 Stage 1 logic (1M rows → 161 compositions)

The filter is **deterministic and very simple** — no random sampling, no
stratification, no geochemical bounds. Three steps:

1. **Load and restrict to igneous rows.** `load_igneous()` reads
   `complete.csv` keeping only `MAGEMIN_OXIDES + [sample_id, rock_group,
   fe2o3_tot, feo_tot]` columns, then filters
   `df[df["rock_group"] == "igneous"]`. The brief states this gave **739,727
   igneous rows** out of ~1.02M total.

2. **Require all 11 MAGEMin oxides explicitly present.** `filter_stage1()` is
   one line:
   ```python
   mask = igneous[MAGEMIN_OXIDES].notna().all(axis=1)
   ```
   where
   ```python
   MAGEMIN_OXIDES = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
                     "k2o", "na2o", "tio2", "cr2o3", "h2o_tot"]
   ```
   This is the brutal step. H2O is missing in ~98.8% of rows and Cr2O3 in
   ~97.2%, so requiring both knocks the count from 740k → 161.

3. **LOI-normalize.** `loi_normalize()` scales each row so the 11 oxides sum
   to exactly 100. Rows whose oxides sum to zero are dropped. No outlier
   filter, no chemical-validity check.

**No** stratified sampling, **no** rock_subgroup balance, **no** SiO2 or MgO
bounds, **no** lab/region/time-window exclusions. The 161 rows are whatever
samples in Gard et al. 2019 happen to have all 11 oxides explicitly
measured. Geochemical diversity is "whatever falls out of that," not designed.

### 5.2 Did Stage 1 already filter to `rock_group == "igneous"`?

Yes. See `load_igneous()`:
```python
return df[df["rock_group"] == "igneous"].copy()
```
Sedimentary and metamorphic rows are dropped before any other filtering.
MAGEMin's `"ig"` database is calibrated for igneous compositions, so this
matches the database scope.

Verified by joining the 161 `sample_id`s back to `complete.csv`:
**161/161 are `rock_group == "igneous"`** — no leakage.

### 5.3 Fe column handling, pre-LOI normalization, oxide minimum coverage

**Fe handling**: Stage 1 requires `feo` AND `fe2o3` columns both
explicitly present. Stage 1 does **not** use `feo_tot` or `fe2o3_tot` (those
columns are loaded but ignored by `filter_stage1`). So Stage 1 implicitly
restricts to samples where the analytical lab partitioned Fe into FeO and
Fe2O3 — typically wet-chemistry titrations or Mössbauer, not modern XRF
"FeO_total" reports.

**Pre-LOI normalization**: none. The filter assumes Gard et al.'s columns are
already in LOI-free wt%. `loi_normalize()` is just a final renormalize to
exactly 100 wt% to correct minor rounding (typical input sums are ~98-102).
No subtraction of LOI, no closure correction beyond scalar renormalization.

**Oxide minimum coverage**: stage 1 requires **all 11** oxides to be
non-NaN. There is no "minimum 8 of 11" rule, no per-oxide minimum value, no
"must report sum ≥ 95 wt%" check. Pure `.notna().all(axis=1)`.

**Fe-availability requirement**: implicit via the 11-oxide rule — both `feo`
and `fe2o3` must be reported. **There is no T1/T2/T3 Fe-tiering**; the
filter is binary. If the team you're handing off to is proposing
Fe-tiering (e.g. T1 = both reported, T2 = only `fe2o3_tot` reported, T3 = only `feo_tot`), that is **new and not what
Stage 1 did**. Stage 1 = T1 only.

### 5.4 Was Stage 2's filter the same logic just scaled up?

**No, the filter is different.** `filter_stage2()` relaxes the requirement
to **8 majors + explicit FeO + Fe2O3 + TiO2** (10 columns required), and
**fills missing Cr2O3 and H2O with 0**:

```python
required = MAJOR_OXIDES + ["fe2o3"]   # 8 majors includes feo, tio2; +Fe2O3 = 10
mask = igneous[required].notna().all(axis=1)
out = igneous.loc[mask, ...]
out["cr2o3"] = out["cr2o3"].fillna(0.0)
out["h2o_tot"] = out["h2o_tot"].fillna(0.0)
```

This is consistent with MAGEMin docs that `Cr2O3=0` and `H2O=0` are valid
"dry, Cr-free" assumptions. Target was ~169k rows (per the brief), though
Stage 2 has never actually been executed end to end.

Note: Stage 2 still requires BOTH `feo` AND `fe2o3` reported. It does
**not** fall back to `feo_tot` / `fe2o3_tot`. Same Fe-partitioning
restriction as Stage 1 — just more permissive on Cr and H2O.

### 5.5 Rock-class diversity of the 161 Stage 1 samples

**Strongly mafic and basalt-dominated.** Joining sample_ids back to
`complete.csv` reveals:

| Field | Value | Count |
|---|---|---|
| `rock_group` | igneous | 161 (100%) |
| `rock_origin` | volcanic | 151; plutonic 5; NaN 5 |
| `frost_class1` | magnesian | 161 (100%) — **single Frost class** |
| `rock_type` | subalkalic gabbro | 130 |
| | picrite | 11 |
| | mantle peridotite | 4 |
| | alkalic gabbro | 3 |
| | intrusive komatiite | 3 |
| | gabbroic diorite / komatiite, foid monzosyenite, diorite, foid gabbro, mafic foidolite, mantle pyroxenite | 1-2 each |
| `rock_name` | basalt | 141 |
| | tholeiite | 5 |
| | harzburgite | 4 |
| | andesite, wyomingite, orendite, madupite, wehrlite | 1 each |
| `qap_name` | basalt | 116 |
| | clinopyroxene norite | 23 |
| | harzburgite | 9 |
| | other (gabbro, dacite, phonolite, foidite, etc.) | 1-2 each |

**Compositional spread:**

| Oxide | mean | std | min | max |
|---|---|---|---|---|
| SiO2 | 48.5 | 3.5 | 33.3 | 60.2 |
| MgO | 10.5 | 7.6 | 4.9 | 37.9 |
| K2O+Na2O | 2.6 | 1.5 | 0.1 | 12.2 |

**Implications for using Stage 1 H5 as v0 training data:**

- Heavy basalt/picrite skew: ~141 basalts + ~11 picrites covers ~95% of
  samples. Other rock types are 1-3 samples each — essentially anecdotal.
- SiO2 capped at 60% (one outlier) — **no felsic representation at all**
  (no granites, rhyolites, dacites beyond a single sample). Useless for
  felsic mineral assemblages.
- All "magnesian" Frost class — **no ferroan suites**, no
  metaluminous/peraluminous variety in the alkali-aluminum plane.
- 151/161 volcanic — **near-absent plutonic / slow-cooled signatures**.

**Verdict**: Stage 1 is a smoke test of the pipeline, not a training set.
Treating it as v0 training data would teach the model "the world is
basalts and picrites." Stage 2 (when run) will be much more diverse
(169k rows across all the missing igneous types), but as written it
inherits the same `rock_group == "igneous"` restriction — so sedimentary
phases in tier A (calcite, dolomite, halite, anhydrite, etc.) will still
be unrepresented in MAGEMin labels.

### 5.6 Hidden exclusions in the filter

**None encoded.** No exclusions by:
- Lab / institution / submitter (`source` and `reference` columns are not even loaded)
- Geographic region (`country`, `tectonic_setting`, `location` columns ignored)
- Time period (`age`, `era` columns ignored)
- Quality flags (any `quality`, `notes`, `flag` columns ignored)

The only implicit exclusions are:
1. **`rock_group != "igneous"`** (explicit)
2. **Any oxide NaN in the required set** (explicit; this is the dominant filter)
3. **Sum-to-zero rows** dropped during LOI normalize (extreme edge case)
4. **Implicit Fe-partitioning bias**: only samples with both `feo` and
   `fe2o3` measured (excludes XRF-only labs that report `feo_tot`)

The Fe-partitioning bias is the subtle one — it likely under-represents
modern (post-1990s) XRF datasets and over-represents older wet-chemistry
samples. If you care about temporal balance or lab representativeness,
you'll need to add explicit handling.

---

## What's verified and what isn't

**Verified working (Stage 1):**
- 161 compositions × 28 (T, P) = 4,508 attempted, **3,707 converged (82.2%)**
- Wall time: 49 s on auto-threaded local node
- 20 phases discovered, 103 SS endmember columns
- G_sys range: -1062 to -631 kJ/mol
- Phase wt sums: 0.997 mean (passes sanity)
- H5 schema validated end-to-end

**Not yet done:**
- Stage 2 production run (~169k compositions × 28 ≈ 4.7M points; ~2 h on 8 threads)
- MORB / peridotite literature spot-check (does a MORB bulk give ol+cpx+fsp at high T?)
- Decomposition of `G_sys` into G_0K + ΔG(T, P) for cleaner ML targets
- Polymorph / SS-composition probes (data is in the H5 but no downstream eval)

## Pointers

- Code: `CrabNet-Mineral`, branch `structure-edm`, under `alignn/magemin/`
- Source of bulks: `complete.csv` at repo root (Gard et al. 2019, ~1M rows,
  latin-1 encoded, gitignored — must be downloaded separately)
- MAGEMin docs: https://computationalthermodynamics.github.io/MAGEMin/
- MAGEMin paper: Riel et al. 2022, *Geochemistry, Geophysics, Geosystems*
- Why we paused this direction (for context): the same X at different (T, P)
  gives different assemblages, which would force a probabilistic head. The
  bimodal-energy direction sidesteps this by using ΔG side-info per mineral
  rather than full MAGEMin labels. Stage 1 outputs remain useful for
  diagnostic work.
