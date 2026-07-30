# Test inventory

One-sentence summary per test: what it looks for and how.  Organized
to mirror the `tests/` directory; numbering restarts per file.

**Maintenance contract: any change to the test suite (add, delete,
rename, or repurpose a test) updates this file in the same change.**

## Shared fixtures

- `tests/conftest.py` -- sets `KMP_DUPLICATE_LIB_OK` and
  `OMP_NUM_THREADS` before imports; `pretrain_runs_root` (autouse)
  redirects `run_single` artifacts into tmp dirs; `corpus_subset_path`
  points at the checked-in 2000-row parquet fixture
  (`tests/fixtures/complete_subset.parquet`, rebuilt by
  `scripts/build/build_test_fixture.py`); `SyntheticTokenDataset` +
  `synthetic_token_dataset` (tiny token datasets) and
  `tiny_pretrain_checkpoint` (a real on-disk encoder+head checkpoint),
  shared by the eval and explorations suites;
  `synthetic_joint_corpus` (the `write_synthetic_rungs` +
  `write_synthetic_draws` builder: one rungs parquet with two draws per
  bulk, mixed assemblage with full 11-oxide comps and T_end, one plag +
  one afs feldspar instance, a fluid row, and a rung-30 attrition case,
  plus the matching draws parquet whose input oxides equal each draw's
  rung-50 reconstruction so rung-50 groups close under the mass-balance
  identity), shared by the mineralogy-dataset, eval, and joint-heads
  suites.
- `tests/training/conftest.py` -- `_tiny_config()` (minimal config on
  the corpus fixture; `min_class_count=5` so both stratified splits
  survive) and `_write_yaml()`.

## data/

### test_bulk_chemistry.py

1. `test_sample_shape_and_dtype` -- `__getitem__` returns `src` (long) and `frac` (float32) tensors of length 22.
2. `test_obeys_padding_convention` -- every constructed sample satisfies `src == 0` iff `frac == 0` via `assert_padding_convention`.
3. `test_no_mask_token_in_raw_output` -- the raw dataset never emits `MASK_TOKEN_ID`; masking is the training loop's job.
4. `test_sample_id_stable_across_construction` -- constructing the dataset twice yields the same `sample_id` for the same index.
5. `test_oxide_to_cation_conversion` -- sio2=50 wt% converts to Si frac 50 * 0.4675 / 100.
6. `test_iron_aggregation_feo_plus_fe2o3` -- Fe frac = (feo * 0.7773 + fe2o3 * 0.6994) / 100 when both oxides are present.
7. `test_iron_fallback_to_feo_tot` -- with feo and fe2o3 both NaN under `require_all_majors=False`, Fe falls back to feo_tot.
8. `test_zero_is_nan` -- an exact-zero major becomes a padded slot under `require_all_majors=False` (zero-as-NaN).
9. `test_zero_drops_row_under_strict` -- under the strict default, a zero major NaNs out and the whole row is dropped.
10. `test_negative_drops_row` -- a negative major oxide discards the entire row as corrupt.
11. `test_negative_trace_nans_out_only_that_slot` -- a negative trace pads only that trace slot; the row survives.
11a. `test_overrange_magnitudes_nan_the_slot` -- under `drop_overrange_as_nan`, an oxide above 100 wt% or a trace above 1e6 ppm pads its slot; with the knob off (historical-corpus reproduction) the row drops via the unconditional frac-overflow guard.
11b. `test_frac_overflow_row_dropped` -- a joint pathology (feo + fe2o3 summing past 100 wt% in the Fe slot) that survives per-column bounds drops the row unconditionally.
12. `test_qap_name_attribute` -- `qap_names` is parallel-indexed to `__getitem__` and preserves the label string.
13. `test_rock_group_filter_excludes_sedimentary` -- `rock_group="igneous"` drops rows labeled with other rock groups.
14. `test_include_cr2o3_toggle` -- Cr stays padded by default and populates when `include_cr2o3=True`.
15. `test_trace_conversion_ppm_to_frac` -- sr_ppm=500 converts to Sr frac 500 / 1e6.
16. `test_missing_csv_raises` -- a nonexistent `csv_path` raises FileNotFoundError instead of silently proceeding.
17. `test_real_corpus_default_filter_size` -- the strict default filter on the real 1M-row CSV lands near ~167k rows (wide bounds; pins filter drift early).
18. `test_require_qap_name_drops_nulls` -- under `require_qap_name=True` no surviving row has a null qap label.
19. `test_min_class_count_drops_below_threshold` -- classes under `min_class_count` vanish from the dataset; classes at/above survive.
20. `test_filtered_dataset_passes_padding_convention` -- the corpus fixture under production-shape filters passes the padding convention end to end.
21. `test_fixture_size_regression_pin` -- pins `len(dataset)` for the strict default filter on the fixture, catching fixture/filter/pandas drift.
22. `test_petrological_axes_returns_finite_values` -- SiO2 / Mg# / alkalis reconstructed from `frac` land in sane wt% ranges.
23. `test_petrological_axes_indices_subset_shape` -- passing `indices` returns arrays sized to the subset.
24. `test_redox_token_populated_iff_measured_pair` -- the redox slot populates only where feo AND fe2o3 are measured (ratio = fe3/(fe2+fe3) in cation wt); fallback iron sources pad it.
25. `test_redox_dataset_obeys_padding_convention` -- a redox corpus (populated and padded redox rows) passes `assert_padding_convention`.
26. `test_redox_default_off_keeps_22_tokens` -- without `include_redox` the tensors stay 22-wide.
27. `test_min_populated_tokens_drops_thin_rows` -- rows below the populated-token floor are dropped; full rows survive.
28. `test_dedup_chemistry_keeps_first` -- exact tokenized duplicates collapse to the earliest row with attributes staying index-parallel.
28a. `test_dedup_prefer_path_wins_duplicate_group` -- a row named in `dedup_prefer_path` survives its duplicate group even when not earliest (pinned-to-test eval chemistry cannot leak into training through a duplicate); unnamed groups keep first.
29. `test_strat_labels_composite_and_rare_relabel` -- `strat_labels` is qap-else-rock_group; a rare qap class relabels to its rock_group instead of dropping rows.
30. `test_strat_labels_still_rare_collapse_to_unknown` -- a label still rare after the rock_group collapse lands in the "unknown" bucket.

### test_condition_proposal.py

1. `test_standard_state_anchor` -- the ported N&L 2002 rhyolite solubility reproduces the model's own standard state (3.45 wt% at 799 bar / 850 C).
2. `test_matches_reference_volatilecalc_across_live_range` -- pins `S(P)` at six pressures spanning the live range to values round-tripped through the third-party VolatileCalc implementation (transcription-error tripwire).
3. `test_solubility_monotonic_in_pressure` -- `S(P)` increases with pressure below the cap crossover.
4. `test_cap_binds_above_crossover` -- `w_max` equals the 8 wt% cap above ~3 kbar and sits below it at 2.5 kbar.
5. `test_draws_inside_support` -- every LHS draw lands inside the bounded (P, H2O) region with positive density.
6. `test_q_matches_closed_form` -- the returned per-row `q` equals `1 / ((P_HI - P_LO) * w_max(P))` and agrees with `q_density`.
7. `test_q_zero_outside_support` -- `q_density` is 0 above the solubility bound, above the cap, and outside the P box.
8. `test_lhs_stratifies_p_marginal` -- with n draws, every one of the n equal P strata contains exactly one draw.
9. `test_importance_weights_recover_support_area` -- `E_q[1/q]` recovers the area of the bounded region, tying sampler and density together.
10. `test_draws_deterministic_under_seed` -- same seed, same draws.
11. `test_setting_posterior_sums_to_one_and_orders` -- arc-like traces give arc-dominant setting weights, depleted give ridge-dominant, both summing to 1.
12. `test_cascade_rung_selection` -- missing Th routes to the Ti-V rung, Nb/Y-only leaves subalkaline mass on fallback, no usable ratio yields `rung=None` all-fallback.
13. `test_traced_draws_inside_support` -- every trace-conditional draw lands inside the broad support with positive mixture density.
14. `test_traced_q_matches_closed_form` -- the q returned by `BulkProposal.draw` equals `BulkProposal.q` at the drawn points.
15. `test_floor_keeps_old_support` -- the traced density is at least `eps * q_density` (and positive) everywhere the broad proposal has support.
16. `test_traced_importance_weights_recover_area` -- `E_q[1/q]` under the mixture recovers the broad-support area (the floor makes the supports equal).
17. `test_wet_box_p_floor` -- a wet box (`w_lo` = 3) never draws above the solubility bound even with the floor off.
18. `test_fallback_covers_rosters` -- every qap_name in both corpus rosters resolves to a single fallback box.
19. `test_traced_deterministic_under_seed` -- same seed, same traced draws.

### test_feldspar_rule.py

1. `test_ternary_coordinates_roundtrip` -- oxide mass fractions constructed from a target (An, Or) recover exactly that ternary coordinate.
2. `test_labradorite_is_plag` -- An60 classifies plagioclase.
3. `test_albite_is_plag` -- near-pure albite falls on the plagioclase side of the DHZ boundary.
4. `test_anorthite_is_plag` -- pure anorthite classifies plagioclase.
5. `test_sanidine_is_afs` -- Or60 classifies alkali feldspar.
6. `test_anorthoclase_is_afs` -- an Or20 anorthoclase composition routes to afs (mineralML puts anorthoclase under Alkali_Feldspar).
7. `test_miscibility_gap_routes_by_dominant_component` -- inside-the-gap ternary compositions route plag when An >= Or and afs otherwise (the documented fallback).
8. `test_degenerate_zero_alkalis_is_plag` -- an all-zero composition degrades to plag instead of erroring.
9. `test_nan_treated_as_zero` -- NaN oxide inputs are treated as 0, not propagated.
10. `test_vectorized_input` -- array inputs classify elementwise.

### test_magemin_corpus.py

1. `test_ladder_truncates_at_stop_melt` -- stored ladder points end at the first point below 30% residual melt (inclusive) and `T_stop` records it.
2. `test_rung_pick_is_highest_T_below_target` -- each 60/50/40/30 rung label comes from the highest-T converged point below that melt fraction.
3. `test_attrition_when_melt_never_drops` -- a draw that never reaches a rung gets that rung's attrition flag, no rung rows, and a full-floor ladder.
4. `test_failed_points_skipped_and_counted` -- non-converged points are excluded from ladder/rung picking and counted in `n_failed`.
5. `test_oversaturated_tag_from_hottest_point` -- free fluid at the hottest converged point sets the draw's `oversaturated` tag.
6. `test_em_json_matches_duplicate_instances_in_order` -- endmember records attach to duplicate solution-phase instances in `out.ph` order; pure phases get null.
7. `test_build_inputs_h2o_injection_sums_to_100` -- injected H2O + renormalized anhydrous majors sum to exactly 100 wt% on every input row.
8. `test_build_inputs_q_and_bounds` -- the draws table carries the trace-conditional mixture density (recomputed via `bulk_proposal`), respects the solubility bound, and copies bulk traces.
9. `test_build_inputs_draws_stable_across_chunk_boundaries` -- per-bulk seeding makes a bulk's draws identical whether built alone or inside a larger table (resume-stable chunking).
10. `test_build_inputs_stamps_proposal_routing` -- arc-signature traces stamp `proposal_tier="trace"` / the Th-Nb-Yb rung on the draws table; a traces-free bulk stamps fallback with a null rung.

### test_magemin_oxides.py

1. `test_split_fe_measured_tier_passes_values_through` -- both `feo` and `fe2o3` reported -> `measured` tier, values untouched.
2. `test_split_fe_assumed_class_applies_ratio` -- only a `_tot` column -> `assumed_class`, split by the class Fe3+/SigmaFe ratio.
3. `test_split_fe_derived_partial_from_measured_feo_plus_total` -- measured `feo` + a `_tot` -> `derived_partial`, remainder derived.
4. `test_split_fe_no_iron_returns_zeros` -- no Fe column -> `no_iron` tier and zero FeO/Fe2O3.
5. `test_bulk_to_magemin_injects_h2o_and_drops_mno_p2o5` -- H2O is the caller-injected value, MnO/P2O5 are dropped, missing Cr2O3 -> 0.
6. `test_loi_normalize_is_exact_and_passes_strings` -- numeric oxides renormalize to exactly 100, non-numeric keys pass through.
7. `test_class_ratio_lookup_and_default` -- Middlemost lookup returns the class value and falls back to `default` for unknown classes.

### test_mineralogy_dataset.py

1. `test_rows_and_target_shapes` -- one item per (bulk, draw, rung) label row with 21-phase presence/magnitude targets; present magnitudes sum to 1, absent are exactly 0; `draws_per_bulk` reports the fixture's 2 draws.
2. `test_input_tokens_match_bulk_chemistry` -- the item's src/frac equal the `BulkChemistryDataset` row for the same sample id (inputs come from the original bulk chemistry, never per-draw oxides).
3. `test_unresolvable_bulk_dropped` -- a corpus bulk id absent from the chemistry table is dropped with a printed count, not an error.
4. `test_bulk_ids_filter_restricts_rows` -- the `bulk_ids` filter keeps only the requested bulks' label rows and raises when the filter empties the corpus.
5. `test_draw_limit_filter_restricts_rows` -- `draw_limit` keeps only rows with draw id below it (per-bulk draw subsample), leaves the bulk set intact, and raises when it empties the corpus.
6. `test_joint_dataset_shapes_and_dtypes` -- `JointHeadDataset` items add comp/cond targets and closure fixings with the right shapes, float32 dtypes, and simplex comp rows.
7. `test_joint_dataset_cond_join_matches_draws` -- per-row (T_end, P_kbar, H2O) targets join the correct draws-table row for each (draw, rung).
8. `test_joint_dataset_rung50_closure_identity` -- true modes x comp targets plus teacher-forced melt/fluid fixings reconstruct `bulk_11` on rung-50 rows to eps-clip precision.

### test_valtest_split.py

1. `test_roster_partitions_and_excludes_dropped_bulk` -- the committed roster holds 1999 unique bulks assigned val/test only, with the dropped bulk absent.
2. `test_roster_counts_and_stratification` -- 887 val / 1112 test, and every qap class appears in both subsets.
3. `test_load_valtest_split_sorted_disjoint` -- both subsets load as sorted int64 arrays, disjoint, union 1999.
4. `test_load_valtest_split_unknown_subset_raises` -- a subset name other than val/test raises.

### test_mineralogy_labels.py

1. `test_one_row_per_reached_rung` -- one label row per (bulk, draw, rung) actually present in the rungs table; the modes matrix spans `SOLID_PHASES`.
2. `test_crystals_renormalization_and_fsp_routing` -- liq is dropped, solids renormalize to 1, fsp instances route to plag/afs, and trd merges into q.
3. `test_dropped_phase_mass_renormalizes_away` -- chl mass disappears and the surviving solids still sum to 1.
4. `test_instances_sum` -- two instances of one phase sum into a single mode.
5. `test_rung_filter` -- only requested rungs produce rows.
6. `test_unknown_phase_raises` -- a phase outside the vocabulary raises instead of being silently dropped.
7. `test_missing_corpus_raises` -- an empty corpus dir raises FileNotFoundError.
8. `test_hurdle_targets_cutoff_and_renorm` -- sub-cutoff phases are marked absent and the present modes renormalize.
9. `test_hurdle_targets_single_present` -- a single-phase row yields magnitude exactly 1 on that phase.
10. `test_joint_comp_mass_weighted_instance_merge` -- two same-token instances merge to the mass-weighted mean oxide composition.
11. `test_joint_feldspar_pair_not_averaged` -- plag and afs keep separate comps (the vocabulary split precedes averaging).
12. `test_joint_trd_comp_merges_into_q` -- trd's comp folds into q by the same mass weighting as its mode.
13. `test_joint_comp_targets_are_eps_clipped_simplexes` -- comp targets are strictly positive and sum to 1 after the eps-clip renorm.
14. `test_joint_fixings_and_absent_fluid` -- melt/fluid fractions and comps extract per group; groups without a liq/fl row get zeros.
15. `test_joint_cond_targets` -- (T_end, P_kbar, H2O) assemble from the rung row and the draws join.
16. `test_joint_bulk_11_fe_split` -- the draws in_* oxides convert to the 11-oxide output basis (Fe2O3 -> FeO-equivalent + excess O) against a hand computation, normalized to sum 1.
17. `test_joint_labels_close_on_real_corpus` -- (skipped without `data/corpus/valtest`) the mass-balance identity holds through the builder's own outputs at quantile thresholds (a small corpus tail does not self-close: melt immiscibility, dropped-phase mass).

### test_volcanic_subset.py

1. `test_measured_feo_and_fe2o3_take_precedence` -- measured FeO + Fe2O3 win over `_tot` columns in the total-Fe-as-FeO precedence.
2. `test_single_measured_species_counts_alone` -- one measured Fe species alone yields its own FeO-equivalent contribution.
3. `test_feo_tot_fallback_then_fe2o3_tot` -- with no measured species, `feo_tot` is used first, then `fe2o3_tot` converted.
4. `test_no_iron_is_nan` -- no Fe column populated returns NaN, not 0.
5. `test_qap_vocabulary_invariants` -- VOLCANIC_QAP holds exactly 25 distinct names and ALKALINE_QAP is a proper subset.
6. `test_expansion_quotas_budget_caps_and_tail_saturation` -- temperature-smoothed quotas meet the budget exactly, respect availability, saturate the rare class at alpha=0.5, and shift head-ward at alpha=1.
7. `test_expansion_quotas_budget_beyond_availability` -- a budget above total availability returns every class at its cap.

## build/

### test_build_corpus_bulk_tables.py

1. `test_nested_block_prefix_realizes_inner_quotas` -- `--expand-block` emits the expansion as two shuffled blocks: full-table class counts equal the launch-budget quotas and the block-1 prefix equals the smaller-budget quotas.
2. `test_block_reorders_but_never_changes_the_selection` -- nested-block ordering permutes rows only; the selected sample_id set is identical to the single-shuffle emit at the same seed.
3. `test_expansion_never_draws_a_bulk_another_tier_holds` -- the drawn bulks miss every sample_id in the train, expansion, and valtest tables (the contamination guard).
4. `test_exhausted_class_keeps_its_held_count_and_budget_respills` -- a class with no bulks left gets quota 0 and the freed budget respills, so the table still totals the requested size.
5. `test_valtest_expand_leads_with_the_held_roster` -- `--valtest-expand` rewrites valtest_bulks.csv with the held roster as block one, uncontaminated and unique, in KEEP_COLS order.
6. `test_valtest_expand_carries_the_alpha_shape_for_the_whole_tier` -- the grown table's per-class counts equal held + `expansion_quotas`, i.e. the alpha-shape target for the expanded tier.

### test_build_valtest_split.py

1. `test_extend_freezes_committed_assignments` -- every bulk in the pre-extension roster keeps its original val/test subset.
2. `test_extend_assigns_every_new_bulk_at_the_split_ratio` -- the added bulks all get assigned, at `TEST_FRACTION` (sklearn ceils the test count).
3. `test_extend_refuses_a_table_the_roster_already_covers` -- `--extend` with nothing new to assign exits instead of rewriting the roster.
4. `test_extend_rejects_a_class_too_thin_to_stratify` -- a single-member class exits with a readable message rather than a sklearn traceback.

### test_generate_mineralogy_corpus.py

1. `test_clean_group_runs_once` -- a call group whose Julia run succeeds is invoked exactly once, no bisection.
2. `test_bisection_drops_whole_poison_draws` -- a crash bisects at draw boundaries and drops the poison draw wholesale, leaving all other draws in the merged output (row order preserved) and writing the crash stderr beside the input.
3. `test_complete_output_salvaged_from_cleanup_crash` -- a nonzero exit with a complete output file (cleanup-phase crash) counts as success, no bisection.
4. `test_crash_cap_aborts` -- more than CRASH_CAP dropped draws in one call group raises instead of silently dropping a systemic failure.

## losses/

### test_mode_loss.py

1. `test_full_support_matches_torch_dirichlet` -- with an all-present mask the masked NLL equals `torch.distributions.Dirichlet` exactly.
2. `test_subcomposition_matches_sliced_dirichlet` -- with random supports the masked NLL equals the Dirichlet built on the sliced (present-only) alphas and targets, row by row.
3. `test_absent_phases_get_zero_gradient` -- alphas at masked-out phases receive exactly zero gradient; present phases receive nonzero.
4. `test_single_phase_support_contributes_zero` -- a one-phase support is a degenerate point mass and contributes NLL 0.
5. `test_mode_head_loss_composition` -- `total` is BCE + `magnitude_weight` x Dirichlet NLL, and both unweighted terms are returned for logging.
6. `test_gradient_reaches_unfrozen_encoder` -- backprop through ModeHead reaches a live encoder's parameters (the Track C end-to-end readiness gate).

### test_comp_loss.py

1. `test_matches_torch_dirichlet_on_present` -- with an all-present mask the masked comp NLL equals `torch.distributions.Dirichlet` on the oxide simplex.
2. `test_absent_phases_get_zero_loss_and_gradient` -- absent phases contribute zero loss and exactly zero alpha gradient.
3. `test_mean_reduction_divides_by_present_pairs` -- the mean reduction normalizes by the number of present (row, phase) pairs, not the batch size.

### test_cond_loss.py

1. `test_single_component_matches_normal` -- K=1 reduces to the summed per-axis `Normal.log_prob`.
2. `test_multi_component_matches_mixture_same_family` -- multi-K matches a hand-assembled `MixtureSameFamily` per axis.
3. `test_mean_reduction` -- mean reduction averages the per-row NLL.

### test_closure_loss.py

1. `test_consistent_inputs_give_zero_residual` -- modes/comps/fixings constructed to reconstruct the bulk give residual ~0 and loss ~0.
2. `test_gradient_reaches_modes_and_comp_not_fixings` -- the closure loss backpropagates into predicted modes and comps; teacher-forced fixings stay gradient-free.
3. `test_loss_is_mean_squared_l2` -- the loss equals the row-mean squared L2 norm of the per-oxide residual.

### test_pretrain_loss.py

1. `test_log_mse_zero_when_no_mask` -- `masked_element_log_mse` returns 0 (not NaN) on an all-False mask.
2. `test_log_mse_finite_with_zero_unmasked_targets` -- frac=0 at unmasked positions (log10 -> -inf) stays finite because the target is floored before the log and those positions are masked out.
3. `test_log_mse_zero_on_exact_standardized_prediction` -- predicting the standardized-log target exactly gives loss 0, confirming the per-element standardization the head emits is what the loss expects.

## models/

### test_encoder.py

1. `TestPaddingConvention::test_valid_batch_passes` -- a convention-obeying batch passes `assert_padding_convention` without raising.
2. `TestPaddingConvention::test_src_populated_but_frac_zero_fails` -- populated src with frac=0 (the v2 bug shape) raises ValueError.
3. `TestPaddingConvention::test_src_zero_but_frac_populated_fails` -- src=0 with frac>0 raises ValueError.
4. `TestMat2vecLoading::test_shape` -- the loaded matrix is [ELEMENT_EMBEDDING_ROWS, MAT2VEC_DIM].
5. `TestMat2vecLoading::test_dtype_is_float32` -- matrix is float32 so `nn.Embedding.from_pretrained` doesn't upcast to float64.
6. `TestMat2vecLoading::test_padding_row_is_zero` -- row 0 (padding) is all zeros.
7. `TestMat2vecLoading::test_mask_row_is_zero_before_encoder_rebinds` -- row MASK_TOKEN_ID is zero at load time (Encoder re-inits it later).
8. `TestMat2vecLoading::test_real_elements_have_nonzero_embeddings` -- Si/Fe/Th rows actually carry data (catches symbol-to-row mapping bugs).
9. `TestMat2vecLoading::test_known_element_distinct_from_padding` -- Si's embedding differs from the padding row (catches wrong-row writes).
10. `TestEmbeddingTable::test_mask_row_reinitialized_nonzero` -- `Encoder.__init__` re-initializes the MASK embedding row to non-zero values (a silently-skipped rebind would leave the mask token degenerate).
11. `TestEmbeddingTable::test_padding_row_zero_after_init` -- the padding embedding row is all zeros after construction.
12. `TestEmbeddingTable::test_padding_row_stays_zero_after_optimizer_step` -- one real AdamW step (with weight decay) leaves the padding row exactly zero, pinning the `padding_idx=0` contract.
13. `test_odd_d_model_raises` -- odd `d_model` fails loudly at construction (the two FractionalEncoder halves must concatenate to d_model).
14. `test_shape_dims_required` -- `d_model`/`n_layers`/`n_heads` carry no defaults; omitting one raises TypeError rather than silently building a phantom model from a stale default.
15. `TestEncoderForward::test_output_shape` -- forward pass pools to [B, d_model].
16. `TestEncoderForward::test_all_padding_does_not_nan` -- an all-padding row produces finite output (no divide-by-zero in masked-mean pool).
17. `TestEncoderForward::test_padding_invariance` -- permuting populated-vs-padded slot positions leaves the pooled latent unchanged within tolerance.
18. `TestSSLGradientFlow::test_encoder_receives_gradient` -- THE v2-bug regression test: after one SSL loss backward, every encoder parameter has non-zero gradient.
19. `TestSSLGradientFlow::test_loss_is_finite` -- the masked-element MSE on a masked batch is finite.
20. `TestMaskingConvention::test_masked_src_is_mask_token_id` -- masked positions carry src=MASK_TOKEN_ID and frac=0 (convention C).
21. `TestMaskingConvention::test_mask_token_distinguishable_from_padding` -- MASK_TOKEN_ID appears at exactly the masked positions, never colliding with padding.
22. `TestMaskingConvention::test_masking_preserves_target` -- `target_frac` equals the original (uncorrupted) frac.
23. `TestMaskingConvention::test_only_populated_positions_get_masked` -- absent tokens (src=0) are never selected for masking.
24. `TestRedoxTokenWidth::test_default_table_is_120_rows` -- the default embedding table keeps the pre-redox 120 rows so old checkpoints load strictly.
25. `TestRedoxTokenWidth::test_redox_table_has_one_extra_row` -- `include_redox_token=True` widens the table by exactly one row.
26. `TestRedoxTokenWidth::test_redox_row_reinitialized_nonzero` -- the REDOX_TOKEN_ID row is re-initialized non-zero (learnable, distinct from padding).
27. `TestRedoxTokenWidth::test_redox_forward_pools_23_tokens` -- a 23-token batch with a populated redox slot pools to a finite [B, d_model] latent.
28. `TestRedoxTokenWidth::test_width_mismatch_fails_loudly` -- loading a 22-token state dict into a redox-width model raises (no silent wrong-width scoring).

### test_fractional_encoder.py

1. `test_output_shape_is_half_d_model` -- output is [B, T, d_model // 2] (two encoders concatenate to d_model downstream).
2. `test_extreme_fracs_stay_finite` (x2, parametrized over linear/log10) -- encodings stay finite at frac 0 / 1e-6 / 1 bin / 0.5 / 1.0; the log10 branch survives `log2(0) = -inf` only via its clamp chain.
3. `test_deterministic_and_distinct_at_production_resolution` -- same input encodes identically twice, and 0.1 vs 0.2 land in different bins at resolution 5000.

### test_imputation_head.py

1. `test_linear_build_forward_shape` -- `hidden=None` builds a single linear read-out emitting one value per token.
2. `test_mlp_build_forward_shape` -- an int `hidden` builds a GELU MLP with the same per-token output shape.
3. `test_mlp_dropout_inserted_after_gelu` -- a nonzero `dropout` adds one `nn.Dropout` in the MLP path; the linear path ignores it.
4. `test_destandardize_roundtrip` -- `predict_log_frac` undoes the standardization (`out * std + mean`) and `predict_frac` exponentiates it.
5. `test_fit_standardization_floors_std` -- a zero-std element is clamped to the std floor so the standardized target can't blow up.
6. `test_fit_standardization_wrong_shape_raises` -- stats not shaped `[N_TOKENS]` raise rather than silently broadcasting.
7. `test_buffers_serialize_and_reload` -- the standardization buffers are part of `state_dict` and reload exactly (they must travel with the checkpoint).

### test_mode_head.py

1. `test_output_shapes` -- forward on (latent, support) returns presence logits and alphas, both `[B, N_PHASES]`.
2. `test_alpha_strictly_positive` -- alphas respect the softplus + `alpha_eps` floor even on extreme inputs.
3. `test_default_hidden_is_half_d_model` -- both trunk widths default to `d_model // 2` and honor an explicit `hidden`.
4. `test_layers_stack_the_trunk` -- `layers` stacks blocks in both trunks (shapes hold, params grow) and `layers < 1` raises.
5. `test_support_changes_alpha` -- flipping one support bit moves the magnitude alphas (the support conditioning is live).
6. `test_presence_logits_ignore_support` -- presence logits are identical across support inputs and match `presence_logits` (the presence branch never sees the support it predicts).
7. `test_gradients_flow_to_all_parameters` -- every head parameter receives gradient from a combined logits+alpha objective.

### test_comp_head.py

1. `test_output_shape_and_positivity` -- forward returns `[B, N_PHASES, N_OXIDES]` alphas, strictly positive.
2. `test_mean_is_simplex` -- the implied Dirichlet mean sums to 1 per phase.
3. `test_shared_concentration_scales_alpha_not_mean` -- rescaling the concentration branch changes alphas but leaves the mean fixed (the shared-concentration factorization).
4. `test_default_hidden_is_half_d_model` -- trunk width default and override.
5. `test_layers_stack_the_trunk` -- `layers` stacks trunk blocks; shapes and positivity hold, params grow.
6. `test_gradients_flow_to_all_parameters` -- every parameter receives gradient.

### test_cond_head.py

1. `test_output_shapes` -- forward returns (log_w, mu, sigma), each `[B, 3, K]`.
2. `test_log_weights_normalize` -- mixture log-weights logsumexp to 0 per axis.
3. `test_sigma_respects_floor` -- sigmas respect the softplus + `sigma_eps` floor.
4. `test_presence_changes_output` -- flipping one presence bit moves the output (the support conditioning is live).
5. `test_init_output_bias_seeds_mixture_at_target_scale` -- `init_output_bias` puts component means/sigmas at the per-axis target scale (the raw-unit cold-start fix).
6. `test_layers_stack_the_trunk` -- `layers` stacks trunk blocks; output shapes hold, params grow.
7. `test_gradients_flow_to_all_parameters` -- every parameter receives gradient.

### test_real_rock_recalibrator.py

1. `test_recalibrator_is_the_identity_at_initialization` -- an unfit recalibrator reproduces the zero-shot outputs exactly, which is what lets a fitted one's gain read as nats recovered on the ledger's scale.
2. `test_structural_zeros_survive_the_magnitude_correction` -- coarsening leaves alpha 0 where a group has no member on the support; the correction is multiplicative so those stay 0 rather than becoming -inf.
3. `test_richer_tiers_contain_the_vector_tier_exactly` -- copying a fitted vector tier's parameters into `matrix` / `mlp` reproduces it, so the 23-param affine is the degenerate case rather than a rival.
4. `test_amphibole_magnitude_is_pinned_at_every_tier` -- the pin holds through the mixing matrix and the MLP, not only the constant, since a coupled correction would be the same unidentifiable quantity reparameterized.
5. `test_off_support_groups_contribute_a_neutral_log_feature` -- scaling the mixing weights that read an off-support column is a no-op, which only holds if that column enters as 0 rather than log(tiny).
6. `test_unknown_tier_is_rejected` -- a mistyped tier raises rather than silently falling back.

## training/

### test_splits.py

1. `test_reserve_test_split_sizes` -- `reserve_test` produces ceil(n * test_split) test rows, disjoint and covering all indices, identical across same-seed calls.
2. `test_reserve_test_stratifies_when_qap_names_present` -- on the corpus fixture every qap class appears in both splits with a test ratio near test_split.
3. `test_reserve_test_rejects_null_labels` -- a no-label corpus (None entries in qap_names) raises a clear `ValueError` pointing to the inference path, not a deep sklearn failure.
4. `test_reserve_test_prefers_strat_labels` -- a dataset carrying `strat_labels` splits fine even with null qap_names (the composite label is the stratification source).
5. `test_reserve_test_pins_roster_rows` -- surviving `test_roster_path` sample_ids land in test, the stratified draw tops test up to the test_split fraction, and the split stays deterministic; ids absent from the corpus are expected attrition.
6. `test_reserve_test_roster_matching_nothing_raises` -- a roster matching zero corpus rows raises (wrong roster or wrong corpus), not a silent no-op.
7. `test_reserve_test_roster_with_zero_test_split_raises` -- a roster combined with `test_split=0` is rejected as contradictory.

### test_ssl_pretrain.py

1. `test_one_step_runs` -- one mask/forward/backward/step on the fixture dataset changes at least one encoder parameter (training is not a silent no-op).
2. `test_val_split_zero_raises` -- `val_split <= 0` fails loudly at entry (HPO pruning and the best-checkpoint policy both need a val curve).
3. `test_run_pretraining_returns_result_dict` -- `run_pretraining` returns the expected keys with finite losses.
4. `test_run_pretraining_monitor_shape` -- the monitor carries train+val lists of length epochs, in memory and on disk.
5. `test_run_single_returns_test_loss` -- `run_single` returns finite val/train/test losses.
6. `test_run_single_writes_results_artifacts` -- `run_single` lands run_summary.json + loss_curve.png with headline fields matching the stamped best checkpoint and a full-run monitor.
7. `test_test_split_zero_records_none` -- `test_split: 0` surfaces `test_loss=None` at the API boundary, not NaN.
8. `test_best_and_last_are_the_only_files_after_run` -- a multi-epoch run leaves exactly the best + last snapshots in the store directory.
9. `test_epoch_best_holds_minimum_val_loss` -- the best snapshot's val_loss and epoch match the run's val-curve minimum.
10. `test_best_and_last_have_distinct_model_states` -- at a deliberately-overshooting lr (0.1) the best and last snapshots hold genuinely different encoder weights.
11. `test_run_single_stamps_test_results_on_store_best` -- after `run_single`, `store.load_best().test_loss` is populated and matches the returned value (the stamp travels through CheckpointStore).
12. `test_last_carries_test_loss_none_after_run_single` -- the last snapshot stays unstamped; only best carries the headline test_loss.
13. `test_run_single_writes_to_configured_filenames` -- YAML-configured checkpoint filenames land on disk and the default names do not (two runs can share a dir safely).
14. `test_log_loss_one_step_updates_encoder` -- the standardized log-space objective yields nonzero encoder gradients in one step (the trace-supervision fix trains the encoder, not just the head).
15. `test_run_pretraining_returns_loss_stats` -- the objective runs end-to-end and returns matching-shaped `loss_stats` (mean+std over the train split).
16. `test_early_stop_patience_halts_before_cap` -- with min_delta above any plausible improvement, patience stops the run at epoch 2 with the monitor truncated to the epochs actually run; best-val tracking runs storeless (the HPO path the hoist protects).
17. `test_early_stop_still_writes_best_and_last` -- an early-stopped run leaves the store with a best snapshot and a last snapshot at the stop epoch.
18. `test_run_single_stamps_test_sample_ids` -- the best checkpoint carries `test_sample_ids` mapping index-for-index onto `test_indices` (the durable row identity a later corpus pins on).

### test_pretrain_objective.py

1. `test_log_frac_stats_hand_computed` -- per-element log10(frac) mean/std matches a hand case; absent elements keep the identity (mean 0, std 1).
2. `test_compute_loss_stats_returns_log_center_scale` -- `compute_loss_stats` returns the per-element log center/scale over the train split.
3. `test_make_loss_fn_matches_underlying_loss` -- the built callable's value equals `masked_element_log_mse` on the same batch.
4. `test_make_loss_fn_requires_stats` -- `make_loss_fn(None)` raises (the objective needs its corpus stats).
5. `test_log_frac_stats_width_follows_frac` -- stats size from the batch width, not a global constant: a 23-wide (redox) corpus gets 23-element stats.
6. `test_log_frac_stats_std_floor` -- a near-constant populated column keeps a floored std (1e-6) so the standardized target cannot blow up.

### test_hpo.py

1. `test_sample_hyperparameter_dispatches_each_type` -- float/int/categorical search-space specs route to the matching `trial.suggest_*`, unknown types raise.
2. `test_objective_end_to_end` -- a 1-trial Optuna study over a tiny search space on the corpus fixture returns a finite val loss.
3. `test_run_hpo_summary_carries_test_loss` -- `run_hpo`'s summary JSON includes the post-sweep production run + test loss, with study figures and per-run artifacts alongside.
4. `test_hpo_trial_does_not_write_checkpoints` -- trials run with `store=None`, so nothing lands in `checkpoint_dir` during the sweep.
5. `test_production_run_restores_base_epochs` -- with `base.epochs=2` and `hpo_run_epochs=1`, the trial reports 1 intermediate value while the production checkpoint trains for 2 epochs (the epochs restore isn't silently dropped).
6. `test_study_resumes` -- a JournalStorage study recreated with `load_if_exists=True` retains its completed trials.

### test_reconstruction_mse.py

1. `test_checkpoint_class_roundtrip` -- `from_models` + save + load + `apply_to` recovers weights bit-exact and carries metadata through the disk round-trip.
2. `test_checkpoint_with_test_results_creates_new_instance` -- `with_test_results` returns a fresh Checkpoint with test fields set, leaving the original unchanged.
3. `test_checkpoint_load_ignores_unknown_payload_keys` -- a payload carrying a dropped legacy key (e.g. `rng_state`) still loads; real on-disk checkpoints depend on tolerant reads.
4. `test_summary_renders_headline_fields` -- `summary()` renders epoch, losses, architecture knobs, and provenance, and says "not stamped" instead of inventing a test_loss.
5. `test_compact_curve_truncates_long_runs` -- short val curves print in full; long ones keep head + tail around an ellipsis with epoch numbering intact.
6. `test_checkpoint_tag_stable_and_filesystem_safe` -- `checkpoint_tag` derives a deterministic, sanitized `<parent>_<stem>` string.
7. `test_evaluate_on_test_returns_finite` -- training a tiny model then calling `evaluate_on_test` on the held-out indices yields a finite float.

### test_imputation_head_train.py

1. `test_encoder_receives_no_gradient` -- `_head_forward_loss` freezes the encoder: backward leaves every encoder parameter grad-free while the head's parameters get gradients.
2. `test_train_imputation_head_end_to_end` -- training on the tiny encoder checkpoint fills the monitor, fits the standardization buffers off the identity, and yields a head-only checkpoint that rebuilds a working head.
3. `test_early_stop_returns_best_val_weights` -- with `early_stop_patience` set, the returned losses are the best-val epoch's (best-weight restore) and no more than `epochs` are recorded.
4. `test_dropout_roundtrips_through_checkpoint` -- `dropout` is stamped into head_config and rebuilds via `build_head`; a pre-dropout checkpoint (key absent) rebuilds at the 0.0 default.
5. `test_imputation_head_checkpoint_save_load` -- `ImputationHeadCheckpoint` survives a disk round-trip with its head_config, encoder reference, and buffers intact.
6. `test_run_single_trains_then_scores_test` -- the single entry trains, saves the head, and scores the test split in one pass: returns the train/val losses plus a `trace_summary` and writes the readout scorecard.
7. `test_n_tokens_roundtrips_through_checkpoint` -- a redox-width (23-token) head stamps its width and rebuilds at it; a pre-redox head_config without the key rebuilds at 22.

### test_imputation_head_hpo.py

1. `test_run_imputation_hpo_end_to_end` -- a 2-trial sweep against the tiny encoder produces a well-formed summary (best_params, finite best_value, baseline reference) plus a resumable journal and history figure; exercises split + per-trial head train + val LOO scoring.

### test_joint_heads_train.py

1. `test_train_smoke_and_checkpoint_roundtrip` -- two epochs of the joint trainer on synthetic rungs+draws corpora (roster monkeypatched): the monitor tracks every loss term per epoch, `comp_eps`, the loss weights, and the per-head architecture keys are stamped, the saved file round-trips through `JointHeadCheckpoint.load`, and all three `build_*_head` reconstructors return working heads.
2. `test_per_head_selection_restores_each_heads_best` -- under a forced selection where mode peaks at epoch 1 while comp/cond keep improving, the stamped mode state bitwise-matches a same-seed 1-epoch run and comp/cond states do not (each head keeps its own best epoch).
3. `test_early_stop_waits_for_slowest_head` -- with mode/comp flat after epoch 1 and cond improving through epoch 4, patience 1 stops at epoch 5, not when the fast heads plateau.

### test_joint_heads_hpo.py

1. `test_tiebreak_mode_ties_break_to_comp_then_cond` -- mode-objective read-out: trials within the tie band are ties broken by comp NLL then cond NLL; outside-band trials never win.
2. `test_tiebreak_minisweep_requires_mode_within_band` -- comp-objective read-out admits only trials whose mode best stays within the band of `mode_reference`, returning None when none qualify.
3. `test_ledger_tie_band_reads_the_instrument_stanza` -- the tie band resolves from the ledger's instruments stanza, the single source of truth.
4. `test_ledger_tie_band_refuses_unmeasured_instrument` -- a null `tie_band_nats` (unmeasured instrument) and an undefined instrument both raise instead of falling back to a default.
5. `test_run_joint_heads_hpo_end_to_end` -- a 2-trial sweep on the synthetic joint corpus: tie band resolved from a ledger fixture (not a config copy), enqueued baseline as trial 0, per-head bests + epochs + predicted-modes closure in user attrs, tiebreak winner in the summary, journal + history + scatter on disk (importances skipped below 4 trials).

### test_real_rock_recalibrator_train.py

1. `test_off_mask_zero_concentrations_do_not_make_the_loss_nan` -- coarsening's exact zeros would make `mask * lgamma(0)` evaluate `0 * inf`; the guard at this call site keeps every term finite.
2. `test_gradients_survive_the_nan_guard` -- the guard does not detach the scored entries, and an off-mask group draws no gradient.
3. `test_torch_objective_matches_the_shipped_numpy_eval` -- the fit's loss equals `real_rock_nll`, so "nats recovered" is not measured against a baseline on a different scale.
4. `test_fitting_reduces_the_loss_it_optimizes` -- a fit lowers the joint NLL it minimizes.
5. `test_pinned_amphibole_offset_reports_as_zero` -- `per_group_offsets` reports the effective correction rather than the raw parameter.
6. `test_bootstrap_resamples_units_not_rocks` -- declaring the same rocks to belong to fewer units widens the interval, since rocks within a unit are not independent draws.
7. `test_lifting_the_group_correction_hits_the_target_noisy_or` -- pushing the corrected group presence down onto member phases reproduces the group probability exactly; shifting member logits by the group's logit shift does not, since the noisy-OR is not linear in logits, and it erases near-certain groups from every sampled support.
8. `test_draws_sample_whole_units_without_overshooting` -- draws take whole units, never split one, and stay within the overshoot tolerance.
9. `test_a_unit_larger_than_the_budget_is_skipped_not_taken` -- otherwise one 66-rock unit would decide the low end of the curve.

## eval/

### test_probes.py

1. `test_qap_to_supergroup_known_pins` -- representative qap names map to their expected supergroups (sanity on the v2-ported lookup).
2. `test_qap_to_supergroup_handles_none_and_empty` -- None / empty / non-string inputs collapse to "other".
3. `test_map_qap_array_returns_object_ndarray` -- array mapping returns an object-dtype ndarray of supergroup strings.
4. `test_supergroups_tuple_has_nine_entries` -- the SUPERGROUPS palette has exactly nine entries including "other".
5. `test_extract_latents_full_dataset_shape` -- extracting over the fixture dataset yields a finite [N, d_model] CPU tensor.
6. `test_extract_latents_indices_subset_shape` -- passing `indices` returns latents sized to the subset.
7. `test_extract_latents_deterministic_in_eval_mode` -- two extractions agree exactly (guards a forgotten `encoder.eval()` with dropout active).
7a. `test_padded_redox_view_zeroes_the_slot` -- `PaddedRedoxView` pads exactly the redox slot, leaves the base dataset and other slots untouched, and refuses a 22-token corpus.
8. `test_logreg_probe_runs_and_beats_random` -- the LR probe trains on separable latents and beats the random baseline.
9. `test_mlp_probe_runs_and_beats_random` -- same contract for the MLP probe.
10. `test_probe_metrics_to_dict_is_json_friendly` -- `ProbeMetrics.to_dict()` survives `json.dumps`.
11. `test_probe_metrics_carries_hyperparameters` -- `to_dict()` records the exact kwargs that produced the metrics.
12. `test_logreg_probe_standardize_helps_at_skewed_scales` -- `standardize=True` converges at least as accurately on a badly-scaled latent.
13. `test_mlp_probe_l2_normalize_runs_and_records_flag` -- the l2-normalize path runs without NaN and the flag lands on the metrics.
14. `test_mlp_probe_two_layer_runs_end_to_end` -- `n_hidden_layers=2` builds the deeper topology and still trains.
15. `test_mlp_probe_invalid_n_hidden_layers_raises` -- 0 or 3 hidden layers raises rather than silently building something else.
16. `test_mlp_probe_class_weighting_lifts_rare_class` -- `class_weight='balanced'` raises the rare class's per-class accuracy (the trade-off, not overall accuracy, is asserted).
17. `test_knn_label_purity_above_random` -- k-NN purity on separable latents beats the class-balanced random baseline.
18. `test_random_label_purity_balanced_classes` -- balanced 5-class labels give baseline 0.2.
19. `test_run_probes_end_to_end_on_fixture` -- smoke-run of the full scoring pipeline (latents + both probes + UMAP 2D/3D + every plot) on the corpus fixture; guards wiring and finiteness, not accuracy.

### test_probe_hpo.py

1. `test_run_probe_hpo_end_to_end_on_fixture` -- a 2-trial probe-HPO sweep against a 1-epoch fixture checkpoint emits a well-formed summary JSON (best_params, final_mlp, baseline_lr).
2. `test_compare_probe_hpos_flags_converged_vs_scattered` -- `build_report`'s convergence flags fire correctly on hand-crafted agreeing/scattering encoder summaries.
3. `test_compare_probe_hpos_writes_report_and_yaml` -- `run_calibration` writes both calibration_report.md and calibrated_probe.yaml.
4. `test_run_probe_hpo_zero_trials_returns_baseline_only` -- `n_trials=0` skips the sweep but still writes the LR baseline.

### test_imputation.py

1. `test_run_logspace_end_to_end_on_fixture` -- `run_imputation` scores a trained head over the encoder's test split in log space, writing the metrics JSON + R2 chart under a `readout/` subdir.
2. `test_back_transform_inverts_normalization` -- majors invert to oxide wt%, Fe to cation wt%, traces to ppm, with the matching axis labels.
3. `test_load_gate_references_reads_run_1a_1b` -- the gate-reference loader maps the gate's run_1a / run_1b R2 onto the input-ceiling / frozen-latent reference keys.
4. `test_load_gate_references_missing_returns_empty` -- a missing gate file yields empty refs so the eval still runs without reference lines.

### test_mode_head_eval.py

1. `test_reliability_table_hand_case` -- fixed-width bins land counts/mean-p/empirical-frequency where a 4-point hand case says, and ECE equals the count-weighted |p - freq|.
2. `test_reliability_table_edge_probabilities_bin` -- p=0.0 and p=1.0 fall into the first and last bins instead of overflowing.
3. `test_within_bulk_frequency` -- per-bulk presence frequency over that bulk's rows; a bulk contributing no rows yields NaN, not zero.
4. `test_bulk_frequency_targets_derives_draws_from_rungs` -- the noise-floor targets read per-bulk draw counts from the dataset itself (rows / n_rungs) alongside the presence fractions.
5. `test_bulk_frequency_targets_raises_on_rowless_bulk` -- a bulk contributing no rows makes the targets undefined and raises instead of propagating NaN.
6. `test_binomial_floor_hand_case` -- per-phase expected MAE of the finite-draw truth estimator matches the sqrt(2/pi)*sd hand computation; degenerate frequencies carry zero noise.
7. `test_within_bulk_conditional_sigma` -- pooled within-bulk std of conditional magnitude per phase (the parity-band summary width) matches a hand case; a never-present phase yields NaN.
8. `test_within_bulk_sigma_profile` -- the level-binned band profile puts a low-scatter bulk and a high-scatter bulk in separate quantile bins with their own sd.
9. `test_conditional_mean` -- Dirichlet mean restricted to a mask renormalizes over it; an empty mask returns all zeros instead of dividing by zero.
10. `test_beta_pit_matches_scipy_and_masks_degenerates` -- PIT values equal `scipy.stats.beta.cdf` under the Beta marginal of the support Dirichlet; absent phases and single-phase supports are NaN.
11. `test_fp_fn_mass_hand_case` -- FP-mass is the predicted-support conditional mean on truly-absent phases and FN-mass the true mass below threshold, per a hand-computed 3-phase row.
12. `test_sample_supports_deterministic_extremes` -- p=0/1 presence probabilities sample deterministically.
13. `test_magnitude_parity_stats_hand_case` -- per-phase conditional-magnitude MAE/r on truly-present rows match a hand case; never-present phases are NaN.
14. `test_presence_scores_nan_when_undefined` -- per-phase recall/precision at 0.5, NaN for never-present / never-predicted phases.
15. `test_recall_by_band_separates_intermediate_frequencies` -- recall stratified by the bulk's own presence-frequency band splits the intermediate (path-sensitive) misses from the high-frequency hits.
16. `test_hurdle_nll_rows_matches_reference_assembly` -- per-row joint hurdle NLL equals summed Bernoulli log-likelihood plus `torch.distributions.Dirichlet` NLL on the sliced support, and the per-phase BCE decomposition sums back to the Bernoulli term.
17. `test_stratified_mean` -- values grouped by label report the right per-group mean and count.
18. `test_support_group_pit_stats_classifies_hand_groups` -- per-(bulk, support)-group PIT stats split groups within one bulk by support, classify a piled-near-0 group as `shifted`, a both-extremes group as `two_sided`, a uniform grid as `flat`, drop groups below `min_rows` from coverage, and return the empty result for an all-NaN phase.
19. `test_support_group_pit_stats_uniform_false_positive_rate` -- seeded calibrated-uniform groups at the minimum group size keep the `two_sided` misclassification rate below 0.2 and `shifted` below 0.1, pinning the tail/mean thresholds.
20. `test_beta_overlay_exemplar_selection` -- the overlay exemplar picker returns the constructed support-separated bulk (dominant vs second support means apart) and the constructed within-support bimodal bulk (2-means split with minority-share floor) as unique winners.
21. `test_run_mode_head_eval_end_to_end` -- full harness smoke via `run_joint_eval(heads=("mode",))` on a test-trained joint checkpoint against a synthetic valtest corpus (roster and results root monkeypatched): metrics.json carries the roster split block, val counts, and the instrument binomial-floor block, and figures land under `runs/mode_head/<tag>/`.

### test_comp_head_eval.py

1. `test_comp_nll_matrix_masks_absent` -- the per-(row, phase) NLL matrix is NaN at absent phases and finite at present ones.
2. `test_comp_beta_pit_uniform_alpha_analytic` -- with alpha = ones the oxide PIT equals the analytic Beta(1, O-1) CDF.
3. `test_comp_beta_pit_calibrated_samples_are_uniform` -- PIT of true Dirichlet samples under their own alphas is uniform (mean and tail mass).
4. `test_comp_parity_stats_perfect_prediction` -- a perfect predictor scores MAE 0 and r 1 per (phase, oxide).
5. `test_an_number_hand_case` -- equal molar CaO and Na2O with no K2O gives An# 0.5.
6. `test_run_comp_head_eval_end_to_end` -- full harness smoke via `run_joint_eval(heads=("comp",))` on a test-trained joint checkpoint (roster and results root monkeypatched): metrics.json + parity/closure figures land, and `run_predicted_closure_eval` on the shared `(ckpt, dataset, latents)` returns a finite predicted-modes closure summary over the same rows.

### test_cond_head_eval.py

1. `test_gmm_nll_per_axis_matches_loss_module` -- the numpy per-axis NLL sums to the torch `gmm_nll` per row.
2. `test_gmm_nll_per_axis_float32_outlier_stays_finite` -- a ~19-sigma float32 outlier row gets a finite NLL matching the float64 value (regression: the pre-logsumexp path underflowed to inf).
3. `test_gmm_pit_single_component_is_normal_cdf` -- K=1 PIT equals the normal CDF.
4. `test_gmm_pit_calibrated_coverage` -- PIT of samples drawn from the scored mixture yields nominal central coverage at 50/90%.
5. `test_central_coverage_hand_case` -- a 3-value hand case lands the right central-interval fraction.
6. `test_effective_components_limits` -- exp-entropy of one-hot weights is 1 and of uniform weights is K.
7. `test_mass_below_zero` -- mixture mass below zero is 0.5 for a zero-centered unit normal and ~0 far above zero.
8. `test_run_cond_head_eval_end_to_end` -- full harness smoke via `run_joint_eval(heads=("cond",))` on a test-trained joint checkpoint (roster and results root monkeypatched): per-axis metrics incl. the exposure-bias delta are finite and PIT/component figures land.

### test_joint_heads_eval.py

1. `test_run_joint_eval_shares_one_dataset_build` -- a two-head run (`mode`, `comp`) on a test-trained joint checkpoint builds the val dataset exactly once (counted via a monkeypatched constructor), returns both metrics dicts, and lands both heads' metrics.json files.
2. `test_run_joint_eval_rejects_unknown_head` -- an unrecognized head name raises before any checkpoint loading.
3. `test_stale_label_config_raises` -- `val_dataset_and_latents` raises on a checkpoint whose `label_config` lacks `comp_eps` (predates the current joint trainer).

### test_real_rock.py

1. `test_coarsen_presence_is_the_noisy_or` -- a group's presence probability is `1 - prod(1 - p_member)`; a group with no member maps to 0.
2. `test_coarsen_alpha_sums_members_on_the_support_only` -- group concentrations sum only on-support members, and the phases no source reports come back as the separate off-vocabulary total.
3. `test_coarsen_alpha_matches_dirichlet_aggregation` -- the coarsened feldspar share matches the mean of summed member shares over 200k Dirichlet draws, confirming the merge is exact rather than approximate.
4. `test_parse_alteration_grade_reads_modifiers_and_fresh` -- `1+`/`2-` take their leading digit, `fresh` is grade 1, blanks are -1.
5. `test_trace_row_is_present_and_censored_not_zero` -- a `TR` row counts as present, carries a censoring bound, and is excluded from the magnitude vector rather than treated as absent.
6. `test_reported_zero_is_absence_not_presence` -- a phase reported at 0.0 was looked for and not found, so it is neither present nor censored.
7. `test_groundmass_is_dropped_before_renormalizing` -- groundmass leaves the truth vector entirely, so the composition renormalizes to the crystal cargo.
8. `test_volume_converts_to_mass_before_coarsening` -- equal volumes of feldspar and Fe-Ti oxide give unequal masses, in the density ratio.
9. `test_group_densities_apply_per_reported_name` -- augite and hypersthene coarsening into one pyroxene group each convert with their own density.
10. `test_quantified_and_trace_in_one_group_counts_as_quantified` -- a numeric hornblende beside a trace cummingtonite leaves amphibole observed, with its bound discarded, and the composition still sums to 1.
11. `test_rock_without_chemistry_or_modes_is_not_scoreable` -- rocks missing either side drop out at load.
12. `test_unknown_mineral_name_is_rejected` -- a mineral with no density entry raises rather than being silently skipped.
13. `test_alteration_grade_masks_rather_than_filters` -- altered rocks stay loaded and are selected by the `clean` mask, so they remain available as a diagnostic.
14. `test_truth_forced_support_respects_the_group_level_only` -- a truth-present group lets the model pick its members, a truth-absent group forces all members off, and phases no source reports are never forced absent.
15. `test_uniform_densities_reproduce_volume_shares` -- an all-equal density table makes the conversion the identity, so truth shares are the reported volume fractions.
16. `test_presence_truth_is_invariant_to_the_density_table` -- density enters only the mass conversion, so presence / quantified / censored / has-magnitude are unchanged by it while the magnitude vector moves.
17. `test_headline_arrays_are_invariant_to_the_trace_bound` -- the bound sizes a censored ceiling and nothing else, so the arrays the headline NLL reads are identical across bounds and only `censor_bound` moves.
18. `test_member_threshold_controls_which_members_join_the_support` -- a lenient threshold admits a second member of a truth-present group, and a threshold nothing clears still falls back to the likeliest one.
19. `test_truth_forced_support_turns_on_the_likeliest_member_when_none_clears` -- a truth-present group whose members all sit below even odds still gets its single most likely member.
20. `test_nll_finite_when_coarsening_leaves_off_support_zeros` -- off-support groups coarsen to alpha 0, which would make the masked Dirichlet NLL evaluate `lgamma(0)` and return NaN; the assembled NLL stays finite and the Bernoulli term matches its closed form.
21. `test_single_group_composition_contributes_no_dirichlet_term` -- a one-group composition is a degenerate point mass and scores exactly 0.
22. `test_censored_consistency_rewards_predicting_a_small_share` -- a model keeping amphibole tiny scores above 0.9 on a trace report while one sizing it large scores below 0.1; non-censored groups stay NaN.
23. `test_tokenize_bulk_frame_pads_rather_than_populating_a_zero` -- a reported-zero oxide pads its slot, preserving `src == 0 iff frac == 0`.
24. `test_tokenize_bulk_frame_agrees_with_the_dataset` -- a frame carrying one corpus row tokenizes to that row's dataset tensors.
25. `test_tokenize_bulk_frame_pads_when_iron_columns_are_absent` -- a table with only `feo_tot` resolves iron through the fallback and pads the oxides it never reports.
26. `test_load_benchmark_invariants` -- on the shipped benchmark: magnitude is nonzero exactly where quantified, quantified and censored are disjoint, presence covers both, compositions sum to 1, and every array is length-aligned.
27. `test_no_stratigraphic_unit_straddles_a_split_boundary` -- one eruption's rocks share nearly identical chemistry, so every unit lives in exactly one subset and the roster covers all of them.
28. `test_committed_roster_covers_every_benchmark_rock_exactly_once` -- the three subsets partition the benchmark, with no rock dropped or double-counted.
29. `test_test_silica_range_sits_inside_train` -- the recalibrator carries SiO2 terms, so a test rock more extreme than anything it was fit on would measure extrapolation rather than adaptation.
30. `test_unit_grouped_split_is_deterministic_given_a_seed` -- the carve-out is the board strategies get compared on, so the same seed reproduces it and a different seed does not.
31. `test_split_balance_prefers_slices_that_match_the_population` -- the criterion ranks an even split above a lopsided one, and reads truth marginals only so selecting on it cannot leak a model output.
32. `test_unknown_split_subset_is_rejected` -- a mistyped subset name raises rather than silently returning nothing.
33. `test_wilson_interval_brackets_and_narrows_with_count` -- the interval contains the observed frequency (including at 0 and 1, where the normal approximation collapses), stays inside [0, 1], and narrows as the bin count grows.
34. `test_unit_bootstrap_resamples_whole_units` -- every resample carries each drawn unit's rows together, so the clustered intervals never mix partial units.

### test_summary_cards.py

1. `test_sample_assemblages_rows_on_simplex` -- every sampled assemblage sums to 1 on its support, zero-probability phases never enter, and sampling is seed-deterministic.
2. `test_sample_assemblages_alpha_fn_sees_unique_supports` -- `alpha_fn` is called once with deduplicated support patterns, and each sample draws from its own support's alphas (a support-dependent alpha shows up in the drawn shares).
3. `test_pick_ambiguous_bulk_prefers_even_split` -- a 50/50 two-pattern bulk beats 90/10 and unimodal bulks; the score is the second-pattern share.
4. `test_pick_ambiguous_bulk_unimodal_fallback` -- an all-unimodal roster returns the first bulk with score 0.
5. `test_pick_bulk_by_nll_median_and_best` -- per-bulk mean NLL selection respects the rung filter; best = lowest mean, median = middle rank; bad `pick` and empty rung raise.
6. `test_top_support_patterns_groups_and_shares` -- hand-built samples with two patterns return correct shares, patterns, and per-pattern mean modes.

### test_trajectory.py

1. `test_load_ledger_validates_schema` -- a well-formed two-instrument ledger loads in order; missing keys (including the instrument stamp), unknown head names, a head lacking its headline NLL, and an instrument absent from the `instruments:` block each raise.
2. `test_build_trajectory_figure_smoke` -- a 3-entry ledger spanning two instruments (one version missing comp/cond) renders a nonempty PNG.

### test_training_curves.py

1. `test_plot_run_loss_returns_axes_with_two_lines` -- plotting a train+val monitor draws exactly two lines on the axes.
2. `test_plot_run_loss_handles_train_only` -- a monitor without a val key still renders cleanly.
3. `test_plot_run_loss_custom_ylabel` -- a custom ylabel lands on the axes.
4. `test_plot_joint_run_loss_panel_per_term` -- a joint-heads monitor renders one panel per loss term plus the total, each with train and val lines.

## explorations/

### test_trace_signal_in_latent.py

1. `test_log_feature_matrix_absent_is_nan` -- populated entries map to log10(frac) and absent entries become NaN (XGBoost native-missing input feature).
2. `test_masked_encode_cannot_leak_target` -- masking token e makes the encoded latent independent of e's input value, the invariant that keeps the 1b latent baseline from reading the held-out element off `h`.
3. `test_run_diagnostic_end_to_end_on_fixture` -- `run_imputation_diagnostic` produces both controlled arms (1a/1b) per element, a populated gate block, and the metrics JSON + R2 chart.

### test_input_knn_ceiling.py

1. `test_build_raw_representations_shapes_and_absent_fill` -- the raw-frac and std-log test matrices are `[n_test, N_TOKENS]` and finite, and absent elements are filled with the neutral standardized 0.
2. `test_build_raw_representations_standardization_matches_train_stats` -- std-log on populated positions equals `(log10(frac) - mean) / std` from the train-split stats, and raw-frac is the unmodified frac.

### test_fe_redox_attribution.py

1. `test_anhydrous_basis_drops_h2o_and_renormalizes` -- per-bulk features from `bulk_11` drop the per-draw H2O column, renormalize to sum 1, and recover the bulk-constant anhydrous vector.
2. `test_anhydrous_basis_rejects_within_bulk_variation` -- anhydrous chemistry that varies across a bulk's rows raises instead of silently averaging (the feature basis contract).
3. `test_without_excess_o_renormalizes_away_the_deficit` -- dropping the excess-O column renormalizes the remaining oxides so a tree can't read the redox column back off the row-sum deficit.
4. `test_provenance_by_bulk_join_contract` -- the draws-table join returns one `fe_provenance` per bulk and raises on conflicting stamps.
5. `test_group_slices_partitions_and_scores` -- group slicing partitions bulks by label and scores each group's MAE independently.
6. `test_bootstrap_delta_ci_sign_and_null` -- the bootstrap CI on the MAE delta collapses to (0, 0) under zero error, hits the exact delta for constant errors, and stays off zero for a real gap.

### test_presence_fraction_ceiling.py

1. `test_log_feature_matrix_nan_exactly_where_absent` -- populated entries map to log10(frac) and absent entries become NaN (XGBoost native-missing input feature).
2. `test_binomial_floor_matches_halfnormal_mean` -- the per-phase floor equals the half-normal mean `sqrt(2/pi) * sqrt(f(1-f)/n)` and vanishes at degenerate rates.
3. `test_score_arm_perfect_and_antiperfect` -- a perfect predictor scores MAE 0 / r 1 per phase; a constant predictor degrades to NaN r instead of crashing.
4. `test_knn_frequency_recovers_cluster_means` -- on two separated latent clusters the cosine-kNN read returns each cluster's own mean f.
5. `test_fraction_probe_output_shape` -- the 1-hidden-layer latent probe emits `[B, N_PHASES]` logits.
6. `test_build_verdict_routes` -- the advisory verdict routes to corpus-ceiling / head-side-fix / skip-connection under the agreed MAE-margin criterion.

### test_magmadb_analogs.py

1. `test_eight_majors_populated_cr_and_traces_padded` -- MagmaDB starting-bulk tokenization populates 8 major slots, padding Cr (production `include_cr2o3=False`) and all 13 traces.
2. `test_si_frac_is_cation_wt_pct_over_100` -- the Si slot is oxide wt% times the cation factor over 100, matching the production gard conversion.
3. `test_fe_blends_feo_and_fe2o3` -- the Fe slot blends FeO + Fe2O3 cation contributions like `_resolve_iron`.

### test_condition_sampling.py

1. `test_lhs_draws_inside_envelope_and_stratified` -- Latin-Hypercube `(P, H2O)` draws stay inside the declared proposal envelope and place exactly one point per stratum on each axis (unbiased coverage).
2. `test_grid_cell_centered_inside_envelope` -- the product-grid helper returns cell-centered points strictly inside the envelope.

### test_equilibrium_multimodality.py

1. `test_endpoint_picks_highest_T_below_tau` -- the near-solidus pick returns the highest-T converged point below the liq threshold, and the pick moves with tau (1% vs 5%).
2. `test_endpoint_attrition_when_never_below_tau` -- when no point crosses the threshold the pick falls to the coldest point and flags attrition.
3. `test_endpoint_ignores_nonconverged_points` -- non-converged points are excluded before the endpoint search.
4. `test_classifier_flags_amp_opx_identity_switch` -- an anti-correlated amp/opx cloud is classified phase-identity-multimodal with the amp<->opx switch set.
5. `test_classifier_proportion_only_is_not_multimodal` -- a fixed identity set with only modal variation is NOT multimodal (amphibole present on every path is not a flip).

(The driver's former flat-box LHS test moved to `tests/data/test_condition_proposal.py` with the sampler itself. The volcanic-core subset builder it once owned lives at `tests/data/test_volcanic_subset.py` since its promotion to `src/`.)

### test_lepr_validation.py

1. `test_coarse_map_and_drop_codes_disjoint` -- no LEPR phase code is both mapped to the coarse vocabulary and dropped.
2. `test_parse_phases_splits_and_strips` -- comma-separated phase strings split and strip; NaN degrades to a harmless unmapped code.
3. `test_dedup_experiments_drops_conflicting_keys` -- (Citation, Experiment) keys with conflicting T/P/Phases are dropped; exact duplicates collapse to one row.
4. `test_dedup_bulk_flags_disagreeing_duplicates` -- duplicate bulks disagreeing by >0.01 wt% on any oxide are flagged `bulk_conflict`; within-tolerance duplicates are clean.

### test_latent_anisotropy.py

1. `test_isotropic_reference_is_nearly_isotropic` -- an isotropic Gaussian scores effective-rank fraction near 1 and mean random-pair cosine near 0 (both anisotropy metrics read "isotropic").
2. `test_rank_one_matrix_collapses_to_effective_rank_one` -- a rank-1 cloud scores effective rank ~1 with >99% variance in the top component (second-order collapse detected).
3. `test_mean_offset_creates_first_order_cone_removed_by_centering` -- a shared mean offset reads as first-order cone (high pair cosine) that centering removes, while the covariance spectrum stays full-rank.

### test_rare_class_quota.py

1. `test_hybrid_takes_all_tail_then_smooths_the_rest` -- the rejected hybrid candidate exhausts availability-bound classes and allocates the remaining budget by alpha=0.82 smoothing, meeting the budget exactly.
2. `test_powerlaw_figure_fits_exact_law` -- an exact rank^-2 series recovers slope -2 with R2 ~ 1 and writes a nonempty PNG.

### test_corpus_scaling.py

1. `test_hash_order_deterministic_and_input_order_independent` -- the md5 roster order is a fixed permutation of the ids regardless of input order.
2. `test_hash_order_prefixes_nest` -- a shorter prefix of the hash order is a strict subset of a longer one (the nested-subset ladder contract).
3. `test_point_config_overrides_only_the_subset_keys` -- per-point configs change only checkpoint dir + subset knobs, drop stale subset keys from the base, and leave the base dict untouched.
4. `test_point_record_parses_both_axes` -- `corpus_scaling_n*` / `corpus_scaling_d*` tags parse into bulk/draw axis coordinates with the right nominal row counts and per-phase metrics.
5. `test_last_doubling_deltas` -- the slope table compares the two largest bulk points only (draw points excluded) and is empty with fewer than two.

### test_real_rock_mapping.py

1. `test_density_ranges_bracket_the_shipped_table` -- the Monte Carlo ranges cover exactly the shipped minerals and each admits its shipped value, so the sampling is centered on a table the eval actually ships.
2. `test_monte_carlo_and_endpoint_tables_stay_in_range` -- drawn and endpoint density tables never leave their declared per-mineral ranges.
3. `test_endpoint_tables_pull_mafics_and_felsics_apart` -- the contrast corners move ferromagnesians and felsics in opposite directions, which is the mafic/felsic mass ratio the magnitude comparison turns on.
4. `test_imputed_composition_matches_a_direct_recomputation` -- imputing traces at their bound gives compositions summing to 1, puts each censored group exactly on its bound, preserves the quantified groups' ratios, and leaves rocks with no quantified phase unscored so the row set stays the headline's.
5. `test_coarse_support_rules_span_and_narrow_the_group` -- `all_members` turns on every member of a truth-present group and `single_member` only its likeliest; both force truth-absent groups off, leave unreported phases to the model, and reject an unknown rule.
6. `test_score_reproduces_the_hurdle_nll_it_reports` -- the audit's scorer on the shipped mapping equals `real_rock_nll` assembled directly, so variants are measured against the number the eval reported.
7. `test_alteration_masks_nest_from_strict_to_everything` -- the freshness cuts nest from grade-1-only up to all rocks, `grades_1_2` reproduces the shipped `clean` mask, and reading `+` as "worse" can only shrink the clean set.
8. `test_row_index_maps_a_shortened_benchmark_back_onto_the_latents` -- a variant that drops a whole rock re-indexes onto the base latents, and an unknown id raises rather than silently misaligning.
9. `test_sign_flips_ignores_offsets_too_small_to_correct` -- a near-zero shipped offset has no correction to reverse and small subsets are excluded, so only a materially reversed offset on enough rocks counts as a flip.
10. `test_run_variants_covers_every_knob_and_reproduces_the_shipped_row` -- the tripwire: the whole audit runs over all six knobs, its `shipped` row equals the eval's own headline, and the trace-bound rows are identical to each other.

### test_real_rock_fewshot.py

1. `test_weight_decay_is_recorded_for_every_tier` -- the curve's only use of the val slice is one fixed scalar per tier; a missing one would silently take a default.
2. `test_run_curve_produces_one_row_per_tier_budget_draw` -- the tripwire: the sweep runs end to end, every row carries a realized rock and unit count, and each paired interval brackets its own point estimate.
3. `test_budgets_are_ascending_so_the_curve_reads_left_to_right` -- the budget grid is sorted.
