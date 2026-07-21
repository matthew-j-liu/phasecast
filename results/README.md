# results/ -- curated, tracked history

This directory is version-controlled. It holds the durable record of model
improvement -- the artifacts presentations and papers pull from -- as opposed
to `runs/` (gitignored), where instruments write their regenerable output.

Repo-wide invariant: **results = tracked conclusions** (here and in
`explorations/*/results/`); **runs = untracked machine output**.

## Contents

- `ledger.yaml` -- one entry per shipped model version: version name, date,
  checkpoint tag, per-head skill scores and headline metrics, one line on
  what changed. Append-only; appended when a version ships. The skill
  trajectory figure regenerates from this file alone, so history survives
  checkpoint and runs/ cleaning.
- `reports/<version>/` -- the frozen headline report for each shipped
  version: the four canonical figures (skill trajectory, mode truth-vs-pred,
  inference card, scenario view) plus a metrics.json snapshot, copied in at
  ship time.

## What does NOT belong here

Anything an eval or training script regenerates on demand -- that goes to
`runs/<instrument>/<checkpoint-tag>/`. This directory is appended by hand (or
by the ship-a-version workflow), never written by instruments directly.

## Unregenerable record warning

`runs/mode_head/mode_head_seed0/` is the standalone-era ModeHead evaluation
record. Its pipeline (checkpoint format, trainer, eval scripts) was deleted
2026-07-21; the checkpoint survives at `checkpoints/mode_head_seed0/` but
current code cannot read it. Do not delete that directory when cleaning
`runs/` -- it is the only copy.
