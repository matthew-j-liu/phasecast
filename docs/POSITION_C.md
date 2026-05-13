# Position C: MAGEMin × mineralML fusion

The training-data generation pipeline. See `CLAUDE.md` for high-level architecture.

## Why this fusion (not either alone)

| Tool | What it gives | What it lacks |
|---|---|---|
| MAGEMin | Stable assemblage + modes + G_sys for any (bulk, T, P) | Idealized phase compositions from solution models; needs T, P |
| mineralML grain DB | Empirical per-mineral chemistry from natural samples | No assemblage info; per-grain, not bulk |
| CIPW | Deterministic normative assemblage from bulk | Idealized endmembers; no hydrous phases; single answer |

Position C: MAGEMin provides the **assemblage label** (which phases, what fractions). mineralML provides the **composition realism** (each phase's actual oxide chemistry as seen in nature). The fusion produces training data neither tool offers alone.

## Pipeline (one rock walkthrough)

Step 1 — Sample bulk from `complete.csv`:
```
SiO2=50.5, TiO2=1.6, Al2O3=15.3, FeOt=10.4, MnO=0.20, MgO=7.6,
CaO=11.4, Na2O=2.7, K2O=0.16, P2O5=0.18, Cr2O3=0.04
→ classified as MORB
```

Step 2 — Sample (T, P) from rock-class-conditional prior:
```
MORB prior: T ~ U(1150, 1250) °C, P ~ U(1, 3) kbar
sampled:    T = 1175 °C, P = 2 kbar
```

Step 3 — Run MAGEMin:
```
Phase         Mode    Idealized composition
Plagioclase   52.3%   An70 (Si 51.1, Al 30.4, Ca 14.0, Na 3.6)
Clinopyroxene 24.8%   Augite (Si 51.5, Al 4.1, Ca 20.0, Mg 15.2, Fe 7.5)
Olivine       12.1%   Fo80 (Si 39.6, Mg 42.1, Fe 18.6)
Spinel         2.6%   Cr-Al spinel
Liquid         8.2%   MORB residual
G_sys = -2.847e6 J/mol
```

Step 4 — Replace each phase's composition with an empirical grain from `min_df_v1.csv`:
- Filter pool: `Mineral=="Olivine" AND Tectonic Setting in {MORB, Iceland, Extensional}`
- Match Fo# to MAGEMin's prediction (±2)
- Sample one grain → real olivine with minor MnO, CaO, Cr2O3 that MAGEMin's basic solution model lacks
- Repeat for plagioclase (match An#), cpx (match En-Fs-Wo), spinel (match Cr#/Al#)

Step 5 — Recompute bulk:
```
bulk_actual = Σ_i mode_i × empirical_composition_i
```
This drifts ~0.1-0.5 wt% from x_real because empirical compositions deviate from MAGEMin ideals. Accept the drift; the new bulk is the training input.

Step 6 — Training tuple:
```python
{
    "bulk":   [11 oxide wt% values],          # input
    "phases": ["Plagioclase", "Clinopyroxene", "Olivine", "Spinel", "Liquid"],
    "modes":  [0.523, 0.248, 0.121, 0.026, 0.082],
    "comp":   [5, 11] matrix of empirical oxide wt% per phase,
    "G_sys":  -2.847e6,
    "class":  "MORB",
    "T_P":    (1175, 2),    # kept for analysis, not used as model input
}
```

## Edge cases (handle later)

- **MAGEMin predicts a phase mineralML doesn't have data for** (melilite, leucite, kalsilite). Fall back to mineralML's `SolidSolutionGenerator` or MAGEMin's ideal composition. Flag in tuple metadata.
- **MAGEMin predicts a composition outside min_df_v1's range** (e.g., very Fe-rich Fo20). Pool empty after filtering. Widen context filter, fall back to synthetic, or use MAGEMin ideal.
- **MAGEMin name ↔ mineralML class mapping.** Build a small lookup table (`src/cnmineral/data/iugs.py`). MAGEMin uses `pl`/`cpx`/`ol`/`sp`; mineralML uses full names.
- **Glass/liquid phase.** Keep MAGEMin's output composition (min_df_v1's glass class is sparse when filtered by context). Optionally validate against TAS field.

## Scale targets

- ~1M bulks × ~5-10 (T,P) samples per bulk → ~5-10M training tuples
- MAGEMin call is the bottleneck (~1-10s per call). Parallelize across nodes.
- Cache MAGEMin outputs by `(bulk_hash, T, P)` → never recompute.
