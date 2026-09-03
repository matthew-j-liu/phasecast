# Techniques: where each is implemented, and which ones a run used

`docs/methodology/techniques.yaml` is the canonical list of ML techniques in
this codebase, each with distinctive symbol patterns.
`scripts/build/build_technique_index.py` scans `src/cnmineral/`, `scripts/`
and `configs/` for those patterns and writes the matching file paths to
`docs/methodology/technique_index.json`, grouped by technique type
(`architecture`, `loss`, `hpo`, ...) and then by name. No line numbers: they
churn on every edit, and the project rule is to reference code by symbol.
Regenerate after any change that adds, renames or removes a technique
(CLAUDE.md "After code changes"); the build prints `zero hits: none` when
every pattern still matches something.

## What the index answers, and what it does not

The index is a static view: "where is X implemented?" It does not answer
"did this run use X?", because several techniques are unconditionally
present in source and only fire under specific config values:

- `use_cosine_schedule`, `label_smoothing` and `l2_normalize_inputs` in the
  probe trainer (`cnmineral.eval.probe.primitives`) are toggled per config
  (`configs/probe/probe_hpo.yaml`); the index lists them either way.
- masked-element modeling (`mask_prob`, `cnmineral.data.tokens`) is
  logically inactive at `mask_prob: 0`.
- the Optuna pieces (`TPESampler`, `MedianPruner`, `JournalStorage` in
  `cnmineral.training.hpo` and `cnmineral.eval.probe.hpo`) only run under the
  `--hpo` flag of the training scripts, never in a single-config run.

## Which techniques a run used: read its checkpoint

Every checkpoint stamps the config it was trained under, so activation is
reconstructed from the artifact rather than tracked by a registry:

- the encoder checkpoint (`cnmineral.training.checkpoint.Checkpoint`)
  carries `config`, `git_sha` and `saved_at`;
- the head checkpoint (`cnmineral.training.joint_heads.JointHeadCheckpoint`)
  carries `head_config`, `loss_config` and `label_config` (the corpus, rungs,
  presence cutoff and any train-tier subset the labels were built from) plus
  `git_sha`;
- an HPO sweep keeps every trial's parameters and objective in its
  `JournalStorage` file, so per-trial stamping would be redundant.

Joining `technique_index.json` (the implemented set) against a checkpoint's
stamped config (the activated set) separates dead code (indexed, active in
no run) from config-conditional techniques (active in some runs) from
always-on ones. A per-run activation registry with hooks at each technique's
call site was designed in 2026-06 and not built: the config stamp already
answers the question, and the one thing it cannot show, whether a
conditional path actually executed, has never had a consumer.
