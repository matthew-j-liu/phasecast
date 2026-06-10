# Techniques: implementation vs activation

`docs/techniques.yaml` lists the ML techniques implemented in this codebase.
`scripts/build_technique_index.py` scans `src/cnmineral/`, `scripts/`, and
`configs/` for each technique's substring patterns and writes the matching
file paths to `docs/technique_index.json`, grouped by technique type
(`architecture`, `optimization`, `hpo`, etc.) and then by technique name.
No line numbers, since those churn on every edit.

The index is a **static** view -- "where is X implemented?"  It does not
answer "did this run actually use X?"  This file documents that gap and
proposes a hook map for closing it when an analysis consumer surfaces.

## The static-vs-runtime gap

Several techniques are unconditionally present in the source but only fire
under specific config values:

- `cosine_scheduler` appears in `src/cnmineral/eval/probes.py` regardless of
  whether `use_cosine_schedule: true` is set in `configs/probe/probe_hpo.yaml`.
- `label_smoothing` always appears; only active when its value > 0.
- `l2_normalize_inputs` always appears; only active when toggled true.
- `masked_element_modeling` always appears; logically inactive at
  `mask_prob = 0`.
- `optuna_tpe_sampler`, `optuna_median_pruner`, `optuna_journal_storage`
  only activate in HPO mode (`scripts/pretrain.py --hpo`), not in
  single-config production runs.

The static index marks these as implemented every time.  A reader cannot
tell from `technique_index.json` alone which technique combinations were
load-bearing in any given experiment.

## Closing the gap: per-run activation JSON

Proposal -- **not implemented**.  The design below is documented here so
the work is unambiguous when a concrete consumer appears (ablation table
for the paper, scaling-experiment cross-run join, etc.).

### Hook insertion points

| file | symbol | would register |
|---|---|---|
| `src/cnmineral/models/encoder.py` | `Encoder.__init__` | `mat2vec_embedding`, `transformer_encoder`, `fractional_encoding`, `masked_mean_pool`, `dropout` (if > 0) |
| `src/cnmineral/data/tokens.py` | `mask_for_ssl` | `masked_element_modeling` (with `mask_prob`) |
| `src/cnmineral/losses/pretrain_loss.py` | `masked_element_mse` | `masked_element_mse` (with reduction mode) |
| `src/cnmineral/eval/probes.py` | `train_mlp_probe` | `mlp_probe`, `adamw`, conditionally `cosine_scheduler`, `label_smoothing`, `l2_normalize_inputs` |
| `src/cnmineral/eval/probes.py` | `train_logreg_probe` | `logreg_probe`, `standard_scaler` |
| `scripts/pretrain.py` | end of `run_single` or `run_hpo` | `optuna_tpe_sampler`, `optuna_median_pruner`, `optuna_journal_storage`, `stratified_split`, `grad_clip` (HPO-only entries gated) |

### Flush sites (one JSON per surviving run, not per Optuna trial)

- Single-run (`run_single` in `scripts/pretrain.py`): one JSON written next
  to the checkpoint after the final test eval.
- HPO sweep (`run_hpo` in `scripts/pretrain.py`): one JSON written for the
  post-sweep production run on `study.best_params`.  Per-trial writes are
  explicitly excluded -- Optuna's `JournalStorage` already records every
  trial's params and val loss, and per-trial technique JSONs would be
  redundant noise.

### Example output (single-run pretrain)

`data/pretrain/production/run_techniques.json`:

```json
{
  "run": {
    "config_path": "configs/pretrain/pretrain_production.yaml",
    "checkpoint": "data/pretrain/production/encoder_best.pt",
    "completed_at": "2026-06-09T14:32:11Z",
    "epochs_run": 10,
    "stop_reason": "epoch_budget_exhausted"
  },
  "active_techniques": {
    "mat2vec_embedding":       {"dim": 200, "frozen_rows": "1..118"},
    "transformer_encoder":     {"d_model": 512, "n_layers": 4, "n_heads": 16, "norm_first": true, "activation": "gelu"},
    "fractional_encoding":     {"resolution": 5000, "linear_and_log": true},
    "masked_mean_pool":        {},
    "masked_element_modeling": {"strategy": "bert-style", "mask_prob": 0.25, "mask_token_id": 119},
    "masked_element_mse":      {"reduction": "mean_per_masked_position"},
    "adamw":                   {"lr": 3e-4, "weight_decay": 0.01},
    "grad_clip":               {"max_norm": 1.0},
    "stratified_split":        {"axis": "qap_name", "ratios": [0.64, 0.16, 0.20], "level": "bulk"}
  },
  "metrics": {
    "val_loss": 0.0142,
    "test_loss": 0.0151
  }
}
```

`optuna_*` entries are absent because this was a single-config run.
`dropout` is absent because `pretrain_production.yaml` has `dropout: 0.0`.

### What the JSON unlocks

```python
import glob, json
rows = []
for f in glob.glob("data/**/run_techniques.json", recursive=True):
    payload = json.load(open(f))
    rows.append({**flatten(payload["active_techniques"]), **payload["metrics"]})
```

Joining `docs/technique_index.json` (implemented set) against the
collection of `run_techniques.json` files (activated set across runs)
distinguishes:

- **dead code**: in the index, never in any run JSON.
- **config-conditional techniques**: in the index, present in some run
  JSONs but absent in others (e.g. `cosine_scheduler` toggled per
  probe-HPO trial).
- **always-on techniques**: in the index, present in every run JSON.

### Caveat: reconstruction vs explicit stamping

Most of `active_techniques` is reconstructible from the stamped config in
the checkpoint payload (see `save_checkpoint` in `scripts/pretrain.py`,
which already stores the full `config` dict).  Per the project's
schema-design rule ("every field earns its place against reconstruction"),
the runtime registry earns its keep only for the parts that are NOT
reconstructible from config alone:

- conditional firing (config sets `use_cosine_schedule: true` -- did the
  scheduler actually step?),
- cross-module aggregation (encoder + head + scheduler + sampler in one
  dict, queryable across runs without re-parsing each YAML),
- multi-run analysis (glob over `run_techniques.json` files is
  one-liner; glob over checkpoints + load each is not).

Defer implementation until that consumer exists.  Naming the hook points
now is cheap; building the registry without a concrete user is speculative.
