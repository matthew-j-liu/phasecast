# condition_sampling

**Question.** How should the Track B corpus sample formation conditions
`(P, H2O)` per bulk so the corpus commits only to *coverage* and every
prior/weighting is deferred to inference -- i.e. off the uniform product grid,
whose point density is a hidden prior? What proposal `q`, what sample count `N`,
and what does the generated per-bulk data actually look like?

**Method.** One worked bulk, gard `sample_id` 6875 (andesite). Draw 256
Latin-Hypercube `(P, H2O)` from a broad box, plus three product grids (balanced
8x8, P-dense 16x4, H2O-dense 4x16) over the same support. Cool each draw to the
near-solidus endpoint (liq < 2 wt%) with single-point MAGEMin `ig` on a
1180->640 C ladder; read the assemblage there. Then: compare MC vs grid
marginals, importance-reweight the one cloud to several priors (`w = p/q`),
bootstrap the error-bar-vs-N curve, and cluster the cloud into scenarios.

- `gen_inputs.py` -- builds the MAGEMin T-ladder input (MC + 3 grids) for 6875.
- `extract.jl` -- per-point MAGEMin, dumps modes + per-phase `Comp_wt` + `G`
  (also item 4's prototype extractor).
- `analyze.py` -- endpoint pick, marginals, IS reweighting + ESS, N-scaling,
  clustering, plots.

Run from a dir with the MAGEMin `Project.toml` (recover from git `2fd5559^` or
`/tmp/mgm_sanity`), env `mineral`, `KMP_DUPLICATE_LIB_OK=TRUE`:

```
python gen_inputs.py
julia --project=. -t auto extract.jl mc_in.csv mc_out.jsonl
python analyze.py
```

**Result.** Procedure is in `analyze.py`; committed evidence in `results/`.
Headlines:

- Grid density is a hidden prior: same support, same N=64, the P-dense vs
  H2O-dense grids disagree on residual-melt presence (0.20 vs 0.30); MC matches
  the *balanced* grid and has no aspect-ratio knob to get wrong.
- Reweighting the one 256-path cloud to different priors moves the marginals with
  no new MAGEMin (ilm presence 0.61 -> 0.99 dry / 0.46 wet), at ESS 256 -> 38.
- Error bar `SE(f) ~ sqrt(p(1-p)/N)`; a sharp prior costs ~2.4x SE and ESS ~ N/6.
- For this bulk under the *equilibrium* generator, amphibole is major on every
  path (median 37 wt%) and opx is absent -- the amp/opx scenario switch in
  `heads.md` is an FC narrative and does not appear here (multimodality risk,
  N=1 bulk).

Sample count (SE vs N, bootstrap on 6875; sharp prior = literature andesite box):

| N   | SE uniform-q | SE sharp prior | median ESS (sharp) | empty-support bulks |
|-----|--------------|----------------|--------------------|---------------------|
| 8   | 0.169        | 0.366          | 1                  | 28%                 |
| 16  | 0.119        | 0.289          | 2                  | 7%                  |
| 32  | 0.081        | 0.198          | 5                  | 0%                  |
| 64  | 0.058        | 0.138          | 10                 | 0%                  |
| 128 | 0.043        | 0.095          | 20                 | 0%                  |
| 256 | 0.031        | 0.065          | 39                 | 0%                  |

Committed evidence in `results/`: `cloud.png`, `marginals.png`, `nscaling.png`,
`endpoints.csv`.

**Decision.** Locks open item 1 of
[docs/corpus_regen_spec.md](../../docs/corpus_regen_spec.md):
`q = Uniform(P[0.5,12] kbar x H2O[0,8] wt%)`, LHS draws with `(P, H2O, q)` stored
per row, prior deferred to inference via `w = p/q`; small test at N=128, production
N ~= 48-64. Spec impact on items 2 (near-solidus attrition) and 8 (sample count)
is recorded in [docs/corpus_regen_spec.md](../../docs/corpus_regen_spec.md).
