# archive

Closed explorations, retired here once their question was settled and the
verdict journaled. Directories keep their committed code, README, and
`results/` evidence; their mirrored tests were removed with them (git history
keeps both). One line each: what was asked, what was concluded, where the
journal records it.

| Study | Asked | Concluded | Journal |
|---|---|---|---|
| [condition_sampling](condition_sampling/README.md) | How to sample `(P, H2O)` per bulk for the Track B corpus? | LHS from `Uniform(P[0.5,12] x H2O[0,8])`, prior deferred to inference | 2026-06-30 |
| [equilibrium_multimodality](equilibrium_multimodality/README.md) | Does phase-identity multimodality survive the equilibrium generator? | Yes, decisively: 94% of bulks flip a silicate, 51% amp<->opx; heads.md scenario headline stands. `work/` (532MB MAGEMin ladders) deleted 2026-08-03; in the desktop backup, regenerable ~25 min | 2026-07-14, 2026-07-17 |
| [fe_redox_attribution](fe_redox_attribution/README.md) | Is Fe redox invisible to the model inputs? | Confirmed; one excess-O column recovers most of it; optional redox input token decided for the regen | 2026-07-22 (#46) |
| [imputation_head_design](imputation_head_design/README.md) | How to build the trace-imputation head on the frozen latent? | MLP default; ablation behind the shipped `LogImputationHead` | 2026-06-16 |
| [input_knn_ceiling](input_knn_ceiling/README.md) | Is qap_name kNN-separable from raw chemistry? | ~Not; the latent's weak kNN purity is a data ceiling, not a representation gap | 2026-06-25 |
| [latent_anisotropy](latent_anisotropy/README.md) | How anisotropic is the canonical latent? | Encoder removes the input cone; latent uses ~12.5% of its dim budget (re-measured for the all-rock encoder at acceptance: eff rank 49 vs 32) | 2026-06-25 |
| [lepr_validation](lepr_validation/README.md) | Can LEPR lab experiments supply validation pairs with measured conditions? | 2,368 presence-only pairs harvested; fine-tuning rejected (no modes, no melt fraction, imposed lab conditions); surviving role is presence-level generator validation, unscheduled | 2026-07-14, closure 2026-08-03 |
| [lineup_coherence](lineup_coherence/README.md) | How much sampler mass falls off the corpus's observed assemblage menu? | Material (mean 11%, rhyolites worst 48%); restricting to the observed-pattern library costs 0.6% truth mass -- adopted for the #47 build | 2026-07-22 (#47) |
| [magmadb_analogs](magmadb_analogs/README.md) | Can MagmaDB anchor latent-neighbor analogs with measured conditions? | Assessed and shelved: 78 distinct bulks, ~29% igneous coverage -- too sparse for single-sample duty | 2026-06-30 |
| [mode_mixture](mode_mixture/README.md) | Why doesn't the K=2 mixture's corpus win transfer to real rocks, and what do K>1 canonical figures look like? | Instrument passes 61% of the gap; nature at matched chemistry has the corpus's width but ONE hump (the broad (P, H2O) prior's artifact); one gate-offset parameter recovers ~60%, to a tie, never a win -- kept as a dial, K=1 canonical | 2026-08-04 (#49, #62) |
| [presence_fraction_ceiling](presence_fraction_ceiling/README.md) | Is the presence ceiling the head or the corpus? | The corpus: head, raw-XGB, and latent-MLP all tie at it; regen became the goalpost (the v3.2 lesson) | 2026-07-22 (#41) |
| [rare_class_quota](rare_class_quota/README.md) | How should rare classes be quotaed in the regen bulk roster? | alpha=0.5 temperature smoothing locked; hybrid rejected | 2026-07-23 |
| [trace_conditional_proposal](trace_conditional_proposal/README.md) | Does a trace-conditional `(P, H2O)` proposal shift draws materially, and does the 24-draw knee hold? | Material shift, knee holds; adopted at default knobs and wired into the regen generator | 2026-07-23 (#48) |
| [trace_signal_in_latent](trace_signal_in_latent/README.md) | Why does imputation fail on traces -- encoder or head? | The latent carries the trace signal; the head was the fix, encoder untouched. `results/` still read by the imputation eval's gate reference | 2026-06-15, 2026-06-22 |
| [volcanic_proxy_definition](volcanic_proxy_definition/README.md) | Which moment of the simulated cooling story is the training label? | Phenocryst regime; rungs pre-screened against the real benchmark (later locked 50+30) | 2026-07-14 |
