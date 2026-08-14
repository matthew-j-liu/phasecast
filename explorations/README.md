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
- **Closed studies retire to `archive/`.** Once the question is settled and the
  verdict journaled, `git mv explorations/<name> explorations/archive/<name>`
  (history follows), remove its mirrored test (git keeps it), and move its
  index row to [archive/README.md](archive/README.md). Active top level stays
  minimal.

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

Active studies. Closed ones live in [archive/](archive/README.md) with a
one-line verdict each.

| Exploration | Question |
|---|---|
| [corpus_scaling](corpus_scaling/README.md) | How do the heads scale with training bulks and draws? Corpus-axis answer measured (#27); the real-rock-axis rerun is #60. |
| [input_probe_ceiling](input_probe_ceiling/README.md) | Does the qap probe measure the latent's organization, or just that gard2019's `qap_name` is computed from the same majors the encoder reads? The same probe on the raw assay, paired on identical rows -- the missing ceiling for the scorecard's two decodability rows (#19). |
| [real_mineralogy](real_mineralogy/README.md) | The sim-to-real benchmark: real volcanic rocks with measured modal mineralogy, harvested + EDA'd. Zero-shot gap measured (10.406 nats over 640 rocks, `eruption-v2`); split by stratigraphic unit and adapted on from there. Stays active: `harvest.py` is the sole builder of the gitignored benchmark inputs. |
| [real_rock_fewshot](real_rock_fewshot/README.md) | How many labelled real rocks close the sim-to-real gap while still generalizing to unseen eruptions? ~25 rocks (8 eruptions) recovers 3.4-3.8 of the 5.0 nats the full 368 recover; coupling pays, nonlinearity does not. |
| [real_rock_mapping](real_rock_mapping/README.md) | Does any defensible reading of the benchmark's truth mapping (densities, trace bound, freshness cut, support forcing) change a sim-to-real conclusion? Robust where it encodes a measurement; the invented support rule is a 3.5-nat lever. Stays active: the benchmark's error-bar instrument. |
| [sediment_transfer](sediment_transfer/README.md) | Does the stack transfer to sedimentary rocks (Andrews, 605 rocks, XRD modes), and is the sim-to-real gap representation or supervision? Zero-shot fails by 13.2 nats; the all-rock encoder improves presence and worsens sizing; a correction layer fit on ~360 real rocks recovers 13.6 nats and reaches the same margin over baseline as eruption-v2 -- so the gap is head supervision. Provenance unidentified, so not reportable. |
| [redox_padding](redox_padding/README.md) | Does a padded Fe-redox token marginalize over the unknown ratio? No: padding is exactly a no-op on the pooled latent, but the padded state sits off the populated family, imitates a low fixed ratio, barely tracks the rock (r 0.23), and is itself predictable from the analysis (AUROC 0.91). |
