# explorations

Decision-guiding *exploratory* analyses: code that runs an investigation to
settle a design question, as distinct from production `src/cnmineral`. An
exploration is committed and pytest-covered, but it is not production code and
isn't held to production standards.

## Rules

- **Not in `src/`.** Exploration code lives in `explorations/<name>/`. It is not
  installed.
- **Reuse flows one direction.** Explorations import production `cnmineral`
  (encoder, `extract_latents`, the scoring helpers in `cnmineral/eval/`).
  Exploration-specific scaffolding stays in the exploration dir; it never goes
  into `src/`.
- **Mirrored, tripwire test.** Each exploration's test lives at
  `tests/explorations/test_<name>.py` (mirroring how `tests/` mirrors `src/`).
  Its smoke test doubles as a tripwire: a `src/` change that breaks a downstream
  exploration fails here. Tests import the exploration's code via the repo-root
  `pythonpath` (`pyproject.toml` `[tool.pytest.ini_options]`) -- e.g.
  `from explorations.<name>.run_gate import ...`; each exploration carries an
  empty `__init__.py` so it imports as a package.
- **Commit the evidence.** README, code, and small result artifacts (PNG, JSON)
  are committed so the finding persists without a rerun (`results/` under each
  exploration is exempt from the repo-root `results/` ignore). Only
  large/regenerable files (cached latents, study DBs) are gitignored.
- **Canonical verdict lands in `docs/JOURNAL.md`** (chronological) and an ADR in
  `docs/decisions/` if it changes a production choice. The exploration dir is the
  backing exhibit + repro; cross-link both ways.
- **Name by responsibility**, not date/number/phase.

## Layout

```
explorations/<name>/
  __init__.py        empty; makes the exploration importable
  run_*.py           entry code (imports cnmineral; has an argparse __main__)
  README.md          Question / Method / Result / Decision
  results/           committed small artifacts (JSON, PNG)
tests/explorations/test_<name>.py   mirrored tripwire test
```

## Index

| Exploration | Question |
|---|---|
| [condition_sampling](condition_sampling/README.md) | How should the Track B corpus sample `(P, H2O)` per bulk so it commits only to coverage -- what proposal `q`, what N, off the density-biased product grid? |
| [equilibrium_multimodality](equilibrium_multimodality/README.md) | Does the mineralogy posterior stay multimodal in phase identity under the locked equilibrium generator, or collapse to "same phases, different proportions"? |
| [imputation_head_design](imputation_head_design/README.md) | How should the trace-imputation head on the frozen latent be built (the ablation behind the shipped `LogImputationHead`; code retired, verdict kept)? |
| [input_knn_ceiling](input_knn_ceiling/README.md) | How separable is qap_name by raw chemistry at all -- is the latent's weak kNN purity a data ceiling or a fixable representation gap? |
| [latent_anisotropy](latent_anisotropy/README.md) | How anisotropic is the canonical latent, in both the mean/cone (first-order) and covariance-shape (second-order) senses? |
| [lepr_validation](lepr_validation/README.md) | Can LEPR lab experiments supply (bulk -> assemblage) validation pairs with *measured* (P, T, H2O)? Harvested, 2368 pairs; use not yet decided. |
| [magmadb_analogs](magmadb_analogs/README.md) | Can MagmaDB anchor latent-neighbor analogs with measured conditions? Shelved: too sparse/narrow for single-sample duty. |
| [real_mineralogy](real_mineralogy/README.md) | The validation-only benchmark: real volcanic rocks with measured modal mineralogy, harvested + EDA'd. Never train on it. |
| [real_rock_mapping](real_rock_mapping/README.md) | Does any defensible reading of the benchmark's truth mapping (densities, trace bound, freshness cut, support forcing) change a sim-to-real conclusion? Robust where it encodes a measurement; the invented support rule is a 3.5-nat lever. |
| [trace_signal_in_latent](trace_signal_in_latent/README.md) | Why does the frozen encoder+head impute majors well but fail on traces, and which layer should be fixed? |
| [volcanic_proxy_definition](volcanic_proxy_definition/README.md) | Which moment of the simulated cooling story is the training label? Resolved: phenocryst regime, rungs 60/50/40/30, pre-screened against the real benchmark. |
