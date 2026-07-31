# results/ -- headline summaries (tracked)

`results/` is tracked; `runs/` is gitignored machine output. The split is
by role. `results/` holds the per-version headline summaries -- the small
set of artifacts we look at to judge a model version and track progress.
`runs/` holds the transient instrument snapshots (full eval dirs, HPO
state, probe output) that get overwritten or cleaned as iteration
continues.

## Contents

Cross-model, at the root:

- `ledger.yaml` -- one entry per shipped model version: version name, ship
  date, checkpoint tag, instrument stamp, per-head headline NLLs, one line
  on what changed. Append-only. Each entry records which instrument
  (valtest roster + draws/bulk + proposal, defined in the file's
  `instruments:` block with its noise floor and tie band -- the single
  source of truth for those constants: the heads-HPO tiebreak reads
  `tie_band_nats` from the stanza via `ledger_tie_band`) measured it, and
  is never re-measured on a later instrument -- when the valtest rows
  change, a new instrument starts a new trajectory segment, and numbers
  compare only within a segment. Cross-instrument model comparison is the
  real-rock benchmark's job.
- `trajectory.png` -- raw headline NLL per head across shipped versions,
  one disconnected segment per instrument
  (`scripts/view/trajectory.py`; regenerates from the ledger alone).

Per-model, one subdirectory per checkpoint tag (e.g. `joint_heads_seed0/`),
written directly by the view scripts and overwritten on regeneration:

- `inference_card_bulk<id>.png` -- one rock end to end, no teacher forcing
  (`scripts/view/inference_card.py`; default bulk = median per-bulk mode
  NLL, `--pick best` for the internal showcase).
- `scenario_view_bulk<id>.png` -- an ambiguous rock's top-2 predicted
  assemblages + per-scenario conditions (`scripts/view/scenario_view.py`).
- `cohort_card.png`, `magnitude_ratio.png`, `nll_distribution.png`,
  `presence_brier.png` -- the real-rock headline set, zero-shot on the
  least-altered eruption rocks (`scripts/view/real_rock_card.py`). The cohort
  card is the single-rock card generalized to the whole benchmark (inputs,
  which minerals the model calls, how much of each on the measured and on its
  own assemblage); the ratio plot is the paired per-rock magnitude error the
  card's marginals cannot see. The other two are scored against a climatology
  null -- base rates plus the mean measured composition, fit on the train/val
  carve-out -- because a cost in nats has no scale until you know what
  ignoring the rock would have cost. Cross-instrument comparison lives here
  rather than in the ledger, which is why these are per-model rather than a
  single trajectory.

The fuller diagnostic sweeps (per-head interrogation figures,
truth-vs-pred panels) stay in `runs/` -- this directory holds only the
pithy per-version headline.

## Unregenerable record warning

`runs/mode_head/mode_head_seed0/` is the standalone-era ModeHead evaluation
record. Its pipeline (checkpoint format, trainer, eval scripts) was deleted
2026-07-21; the checkpoint survives at `checkpoints/mode_head_seed0/` but
current code cannot read it. Do not delete that directory when cleaning
`runs/` -- it is the only copy. Its headline numbers are preserved in
`ledger.yaml` (standalone-modehead entry).
