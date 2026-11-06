# phasecast data release

This record contains two datasets used in ["Simulated petrological labels
enable probabilistic mineralogy prediction from whole-rock geochemistry"](https://arxiv.org/abs/2610.36429)
(NeurIPS 2026 Sim2Science workshop, arXiv 2610.36429) and in the phasecast
code at [github.com/matthew-j-liu/phasecast](https://github.com/matthew-j-liu/phasecast): (1) the simulated
label corpus, on which the prediction heads are trained and evaluated, and
(2) the pretraining set used by the composition encoder. Every rock
composition in both datasets comes from the global whole-rock compilation of
[Gard, Hasterok and Halpin (2019)](https://doi.org/10.5194/essd-11-1553-2019), and the simulated label corpus
was computed with the Gibbs energy minimizer MAGEMin of [Riel, Kaus, Green
and Berlie (2022)](https://doi.org/10.1029/2022GC010427) - please cite both alongside this record (details
at the end).

The DOI of this record is https://doi.org/10.5281/zenodo.23992951.

To train or evaluate the prediction heads with the phasecast code, unzip
`simulated_corpus.zip` (stored uncompressed, since parquet is already
compressed) at the root of a phasecast clone. Unzipping produces
`data/corpus/train` and `data/corpus/valtest`, the two paths used by the
configurations under `configs/heads`. After that, you can retrain the
production heads with `python scripts/training/joint_heads.py --config
configs/heads/joint_heads.yaml` and evaluate a head checkpoint on the valtest
split with `scripts/eval/joint_heads.py`.

## The simulated label corpus

**Simulation methodology.** Each composition was simulated with MAGEMin
([Riel et al. 2022](https://doi.org/10.1029/2022GC010427), MAGEMin_C 2.1.9 under Julia 1.12, igneous
database),
which calculates the equilibrium mineral assemblage at a given temperature,
pressure and water content. A simulation is comprised of three parts.
The first part is a cooling ladder from 1200 to 600 C in 20 C steps, stopped once
residual melt falls below 30 wt%. The second part is a sampled pressure and water
content (i.e., a draw), with pressure uniform on 0.5 to 12 kbar and water
uniform on 0 to min(8 wt%, S(P)). The latter is informed by the rhyolite water
solubility curve S(P) of VolatileCalc ([Newman and Lowenstern 2002](https://doi.org/10.1016/S0098-3004%2801%2900081-4))
and the 8 wt% cap of the upper end of melt-inclusion water contents in arc
magmas ([Wallace 2005](https://doi.org/10.1016/j.jvolgeores.2004.07.023), [Plank et al. 2013](https://doi.org/10.1016/j.epsl.2012.11.044)). Lastly, the third part is an iron split, because MAGEMin requires the
iron in a composition to be divided between FeO and Fe2O3. Where the
compilation reports both oxides we used the measured ratio, and where it
reports only total iron we assigned a ferric fraction by rock type
([Le Maitre 1976](https://doi.org/10.1007/BF00399603), [Middlemost 1989](https://doi.org/10.1016/0009-2541%2889%2990011-9)). The compositions are the volcanic rock names
in the compilation that (1) report iron, (2) have SiO2 at most 80 wt%, (3)
have MgO at most 18 wt%, and (4) sum to at least 90 wt% across the oxides,
minus the foidite family, on which the solver would not converge. The generator is
`scripts/build/generate_mineralogy_corpus.py` driving
`magemin/run_magemin.jl`, and the design history is ADR 004 in
`docs/decisions.md`.

**Corpus size.**

| split | compositions | draws each | draws rows | rungs rows | ladder rows |
|---|---|---|---|---|---|
| train | 10,000 | 48 | 480,000 | 11,330,942 | 35,288,867 |
| valtest | 500 | 128 | 64,000 | 1,492,897 | 4,654,554 |

In the paper we count labels as (composition, draw, rung) triples at rungs
50 and 30 that converged, 949,228 from 9,946 train compositions and 125,105
from 493 valtest compositions, which is why those counts are slightly smaller
than the row counts above. There are 2,000 compositions in `valtest_bulks.csv`
because we later extended that roster for a second corpus, but only the
first 500 are simulated here. Their division into validation and test is in
the repository at `src/cnmineral/data/valtest_split.csv`.

**What is in each folder.** Each split is written as a hundred compositions
per chunk, in `chunk_NNNN_draws.parquet`, `chunk_NNNN_rungs.parquet` and
`chunk_NNNN_ladder.parquet`. Alongside the chunks you'll find
`manifest.jsonl` with one record per chunk (composition range, row counts,
attrition, wall time), `health_phase_coverage.csv`, and
the corpus health figures under `figures/`. Compositions are keyed by `bulk`,
which is the Gard 2019 sample id, and draws by `(bulk, draw)`.

The **draws** table has one row per (composition, draw).

| column | meaning |
|---|---|
| `bulk`, `draw` | composition id and draw index |
| `P_kbar`, `H2O` | the sampled pressure (kbar) and dissolved water (wt%) |
| `q` | the proposal density of the draw, kept so you can reweight the corpus under a different prior over conditions |
| `fe_provenance` | how the ferric fraction was set, one of `measured`, `derived_partial` or `assumed_class` |
| `in_sio2` ... `in_cr2o3` | the oxide composition actually handed to MAGEMin, after the iron split and renormalization (wt%) |
| `ba_ppm` ... `zr_ppm` | the 13 trace elements of the source composition (ppm), passed through untouched |
| `qap_name`, `series`, `sio2`, `fe_measured` | roster columns passed through, namely the IUGS QAPF name, alkaline or subalkaline, the source SiO2, and whether both iron oxides were measured |
| `n_points`, `n_failed` | ladder points attempted and points where the solver failed |
| `oversaturated` | whether the draw exsolved a fluid at the very first ladder point |
| `T_stop` | the temperature (C) at which residual melt fell below 30 wt% |
| `attrition_60`, `attrition_50`, `attrition_40` | true if the draw never reached that rung |

The **rungs** table has one row per (composition, draw, rung, phase
instance). We read each draw out at four residual-melt rungs, 60, 50, 40 and
30 wt%, even though training only uses 50 and 30.

| column | meaning |
|---|---|
| `bulk`, `draw`, `rung` | keys, with `rung` being the residual melt in wt% |
| `T_end`, `liq_frac`, `G_sys` | temperature (C), melt fraction and system Gibbs energy at the readout |
| `phase`, `instance` | the MAGEMin phase name and its position in the phase list at that point, so two feldspars stay distinguishable |
| `mode_wt` | the mass fraction of that phase in the whole system, melt and fluid included |
| `wt_SiO2` ... `wt_H2O` | the composition of that phase in oxide wt% |
| `em_json` | endmember names and fractions for solution phases, as JSON |

The **ladder** table has one row per (composition, draw, temperature step,
phase instance) with just `bulk`, `draw`, `T_celsius`, `phase`, `instance`
and `mode_wt`. Modes are the only quantity in this table, which is enough to
re-pick a different rung without re-running MAGEMin.

**Stored rows are not yet training labels.** The tables keep raw MAGEMin
output. The 21-phase vocabulary the model predicts is built when the corpus is loaded, by
`cnmineral.data.phase_vocab` and `cnmineral.data.mineralogy_labels`. In
that step we

- keep rungs 50 and 30 only
- drop `liq` and `fl` and renormalize the solids to sum to one
- merge tridymite and cristobalite into quartz
- drop chlorite and ferropericlase
- split the single MAGEMin feldspar into plagioclase and alkali feldspar by
  composition
- sum repeated instances of a phase
- count a phase present above 0.1 wt% of the crystals

Any of these vocabulary choices can be changed from the stored tables without
regenerating the corpus.

## The pretraining set

`pretraining_set.parquet` has one row for each of the 787,721 compositions
the production encoder was pretrained on, along with the split it sat in
(504,141 train, 126,036 validation, 157,544 test). Pretraining is
self-supervised masked-element reconstruction on chemistry alone, and no
simulated label is used in pretraining. We rebuilt the same rows from the compilation
with `configs/pretrain/pretrain_allrock_redox.yaml` and checked the rebuilt
split against the indices stamped in the encoder checkpoint
(`scripts/build/build_zenodo_release.py`). With this file you avoid
downloading the 800 MB compilation and know exactly which rows the encoder
saw.

| column | meaning |
|---|---|
| `sample_id` | Gard 2019 sample id |
| `split` | `train`, `val` or `test` |
| `qap_name`, `rock_group` | the IUGS QAPF name (may be empty) and whether the rock is igneous, sedimentary or metamorphic |
| `sio2`, `al2o3`, `cao`, `mgo`, `k2o`, `na2o`, `tio2`, `cr2o3`, `feo`, `fe2o3`, `feo_tot`, `fe2o3_tot` | oxides exactly as reported in the compilation (wt%) |
| `sr_ppm`, `zr_ppm`, `ba_ppm`, `y_ppm`, `rb_ppm`, `ni_ppm`, `nb_ppm`, `v_ppm`, `la_ppm`, `ce_ppm`, `nd_ppm`, `yb_ppm`, `th_ppm` | trace elements exactly as reported (ppm) |

The values in this file are the values reported in the compilation, before
any tokenization. In tokenization we

- treat an exact zero as missing
- treat an oxide above 100 wt% or a trace above 1e6 ppm as missing
- take iron from `feo` and `fe2o3` where both are present and from `feo_tot`
  and `fe2o3_tot` otherwise

The rows were selected by taking every rock group, keeping rows with at least
four populated tokens after those rules, and keeping one row per exact
tokenized chemistry. We pinned the test rows of the previous encoder and the
2,000 valtest compositions of the corpus into the test split (1,976 of the
valtest compositions survive the filters, all in test). Of the 10,000 corpus
train compositions, 9,256 are in the pretraining set, with 5,836 in train,
1,496 in validation and 1,924 in test. The encoder saw the chemistry of those
9,256 compositions during pretraining and never a label for any of them.

## License and citation

This record is released under CC BY 4.0. The compositions are redistributed
from the Gard, Hasterok and Halpin (2019) compilation, which is also CC BY
4.0. If you use this data, please cite the paper and the sources below that
apply.

- Liu, M. J., Kaplan, A. D., Whittaker, M. L. and Persson, K. A. (2026),
  Simulated petrological labels enable probabilistic mineralogy prediction
  from whole-rock geochemistry, NeurIPS 2026 Workshop on Sim2Science,
  [arXiv 2610.36429](https://arxiv.org/abs/2610.36429)
- Gard, M., Hasterok, D. and Halpin, J. A. (2019), Global whole-rock
  geochemical database compilation, Earth System Science Data 11, 1553-1566,
  [https://doi.org/10.5194/essd-11-1553-2019](https://doi.org/10.5194/essd-11-1553-2019)
- Riel, N., Kaus, B. J. P., Green, E. C. R. and Berlie, N. (2022), MAGEMin,
  an efficient Gibbs energy minimizer, application to igneous systems,
  Geochemistry, Geophysics, Geosystems 23, e2022GC010427,
  [https://doi.org/10.1029/2022GC010427](https://doi.org/10.1029/2022GC010427)
- Newman, S. and Lowenstern, J. B. (2002), VolatileCalc, a silicate
  melt-H2O-CO2 solution model written in Visual Basic for Excel, Computers
  and Geosciences 28, 597-604, [https://doi.org/10.1016/S0098-3004(01)00081-4](https://doi.org/10.1016/S0098-3004%2801%2900081-4)
- Wallace, P. J. (2005), Volatiles in subduction zone magmas, concentrations
  and fluxes based on melt inclusion and volcanic gas data, Journal of
  Volcanology and Geothermal Research 140, 217-240,
  [https://doi.org/10.1016/j.jvolgeores.2004.07.023](https://doi.org/10.1016/j.jvolgeores.2004.07.023)
- Plank, T., Kelley, K. A., Zimmer, M. M., Hauri, E. H. and Wallace, P. J.
  (2013), Why do mafic arc magmas contain 4 wt% water on average?, Earth and
  Planetary Science Letters 364, 168-179, [https://doi.org/10.1016/j.epsl.2012.11.044](https://doi.org/10.1016/j.epsl.2012.11.044)
- Le Maitre, R. W. (1976), Some problems of the projection of chemical data
  into mineralogical classifications, Contributions to Mineralogy and
  Petrology 56, 181-189, [https://doi.org/10.1007/BF00399603](https://doi.org/10.1007/BF00399603)
- Middlemost, E. A. K. (1989), Iron oxidation ratios, norms and the
  classification of volcanic rocks, Chemical Geology 77, 19-26,
  [https://doi.org/10.1016/0009-2541(89)90011-9](https://doi.org/10.1016/0009-2541%2889%2990011-9)
