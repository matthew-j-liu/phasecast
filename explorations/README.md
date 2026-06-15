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
| [trace_imputation_loss_gate](trace_imputation_loss_gate/README.md) | Why does the frozen encoder+head impute majors well but fail on traces, and which layer should be fixed? |
