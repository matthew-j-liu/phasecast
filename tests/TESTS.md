# Test inventory

One-sentence summary per test: what it looks for and how.  Organized
to mirror the `tests/` directory; numbering restarts per file.

**Maintenance contract: any change to the test suite (add, delete,
rename, or repurpose a test) updates this file in the same change.**

## Shared fixtures

- `tests/conftest.py` -- sets `KMP_DUPLICATE_LIB_OK` before imports;
  `pretrain_results_root` (autouse) redirects `run_single` artifacts
  into tmp dirs; `corpus_subset_path` points at the checked-in
  2000-row parquet fixture (`tests/fixtures/complete_subset.parquet`,
  rebuilt by `scripts/build/build_test_fixture.py`).
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

## losses/

### test_pretrain_loss.py

1. `test_reductions_match_hand_computed_values` -- each of the three reductions matches a hand-computed value on a small case with uneven per-sample mask counts (where the reductions genuinely differ).
2. `test_unmasked_positions_carry_no_gradient` -- a wildly-wrong prediction at an unmasked position contributes zero gradient; masked positions contribute non-zero.
3. `test_zero_masked_positions_returns_zero` -- an all-False mask yields loss 0 (not NaN) under every reduction.
4. `test_validation_raises` -- shape mismatches, non-bool masks, and unknown reduction names raise instead of silently broadcasting.

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

### test_splittional_encoder.py

1. `test_output_shape_is_half_d_model` -- output is [B, T, d_model // 2] (two encoders concatenate to d_model downstream).
2. `test_extreme_fracs_stay_finite` (x2, parametrized over linear/log10) -- encodings stay finite at frac 0 / 1e-6 / 1 bin / 0.5 / 1.0; the log10 branch survives `log2(0) = -inf` only via its clamp chain.
3. `test_deterministic_and_distinct_at_production_resolution` -- same input encodes identically twice, and 0.1 vs 0.2 land in different bins at resolution 5000.

## training/

### test_splits.py

1. `test_reserve_test_split_sizes` -- `reserve_test` produces ceil(n * test_split) test rows, disjoint and covering all indices, identical across same-seed calls.
2. `test_reserve_test_stratifies_when_qap_names_present` -- on the corpus fixture every qap class appears in both splits with a test ratio near test_split.

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

## eval/

### test_probes.py

1. `test_qap_to_supergroup_known_pins` -- representative qap names map to their expected supergroups (sanity on the v2-ported lookup).
2. `test_qap_to_supergroup_handles_none_and_empty` -- None / empty / non-string inputs collapse to "other".
3. `test_map_qap_array_returns_object_ndarray` -- array mapping returns an object-dtype ndarray of supergroup strings.
4. `test_supergroups_tuple_has_nine_entries` -- the SUPERGROUPS palette has exactly nine entries including "other".
5. `test_extract_latents_full_dataset_shape` -- extracting over the fixture dataset yields a finite [N, d_model] CPU tensor.
6. `test_extract_latents_indices_subset_shape` -- passing `indices` returns latents sized to the subset.
7. `test_extract_latents_deterministic_in_eval_mode` -- two extractions agree exactly (guards a forgotten `encoder.eval()` with dropout active).
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

### test_training_curves.py

1. `test_plot_run_loss_returns_axes_with_two_lines` -- plotting a train+val monitor draws exactly two lines on the axes.
2. `test_plot_run_loss_handles_train_only` -- a monitor without a val key still renders cleanly.
3. `test_plot_run_loss_custom_ylabel` -- a custom ylabel lands on the axes.
