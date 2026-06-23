# v3 execution roadmap

Practical execution plan for the v3 rewrite. Distinct from
`docs/encoder.md` (which crystallizes the architecture and value
proposition); this doc covers git layout, track division, gate tables,
and the ownership split between you and agentic work.

## Git layout: new branch in this repo, not a fresh repo

Reasons to stay in this repo:

- `data/` contains ~1 GB of files (`bulks/gard2019.csv`,
  `grains/mineralml.csv`, `optuna/` journals, diagnostics outputs,
  and Track B's eventual `MAGEMin/` corpus).  All gitignored, all
  already where you want them.  Migrating to a fresh repo means
  symlinks or copies.
- Memory files in `~/.claude/projects/-Users-mliu-git-clones-crabnet-mineral-v2/memory/`
  are keyed by repo path. New repo = orphaned memories = re-deriving
  what you already learned.
- `docs/journal.md`, `docs/decisions/*.md`, `CLAUDE.md` are valuable
  historical context. You want them quietly accessible while you ignore
  them, not gone.
- `git checkout main -- path/to/file` lets you pull v1+v2 code in
  on-demand as a reference without polluting the v3 working tree.

Concrete setup:

```bash
git checkout main
git pull
git checkout -b v3
mkdir src_v3 scripts_v3 configs_v3
git rm -r src/ scripts/ configs/ checkpoints/  # checkpoints is gitignored; rm just removes the empty dirs
git tag v3-empty
git commit -m "v3: start of fresh rewrite; v1+v2 code removed (available via git checkout main -- path)"
```

You can rename `src_v3 -> src` later. The v3 tree stays clean; v1+v2 is
one git checkout away when you want it as reference.

`CLAUDE.md` is gitignored per project convention so it doesn't carry to
v3 automatically. Copy the still-relevant parts (project-in-one-paragraph,
Position C framing, four-axis differentiation table) into a fresh
v3-scoped `CLAUDE.md` once that branch's structure is clear.

`data/diagnostics/latent_v2/` and `data/diagnostics/degeneracy_viz/`
should stay on disk -- they're gitignored, won't move, and you'll want
to reference them during track A's encoder validation.

## Track division: foundation model A, then mineralogy heads B

The v2 cell-c result already validated this pattern. SSL-pretrained
encoder + supervised mineralogy is what won; the encoder doesn't need
to keep changing during supervised training. So: build A fully, save
the checkpoint, freeze it, then build B on top.

End-to-end fine-tuning (Track C) is optional and last. Probably small
gains given how strong the SSL encoder is. Add only if Track B's eval
suggests headroom.

### Track A: foundation-model encoder (closed 2026-06-11)

Wrapped up: SSL encoder pretrained on Gard 2019 (~474k igneous
bulks) at the post-Optuna hyperparameters in
`configs/pretrain/pretrain_production.yaml`.  Canonical budget is
20 epochs (a 40-epoch comparison landed within stochastic noise:
supergroup MLP +0.003, qap_name MLP -0.003 -- SSL loss kept
descending but downstream metric did not).

The canonical SSL objective is **per-element standardized log-space
masked MSE** (ADR 007, adopted 2026-06-18); it measures higher on
every probe and the trace-imputation gate than the original
linear-space MSE, which is retired to reference.  Canonical
checkpoint lives at
`checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt`.

Probe results on the 20-epoch encoder (`scripts/eval/probe_encoder.py`,
library-default probe HPs; figures + JSON under
`results/probe/pretrain_v3_logmse_std_seed0_epoch_best/`):

| probe | test accuracy | v2 cell-c bar |
|---|---|---|
| supergroup MLP | 0.821 | 0.76 (+6.1) |
| qap_name MLP   | 0.769 | 0.72 (+4.9) |
| qap_name MLP (probe-HPO-tuned) | 0.790 | -- |
| supergroup LR  | 0.749 | -- |
| qap_name LR    | 0.722 | -- |

The tuned row is the same encoder probed at the
probe-HPO best params (`results/probe_hpo/`); the
library-default rows are the apples-to-apples comparison against
v2 cell-c.

k-NN qap purity at k=5/10/50: 0.582 / 0.554 / 0.487 against a
random baseline of 0.100 -- the encoder organizes rock chemistry
into qap-coherent neighborhoods without ever seeing a label.

Stops here: the encoder is the durable contribution.  Two parallel
follow-up tracks: **Track A part 2** (`pretrain_applications.md`) and
**Track B** (supervised mineralogy heads).  Track A part 2 is
unblocked now; Track B is still gated on the FC corpus regen
pilots in `~/.claude/plans/sunny-weaving-nebula.md`.

### Track B: supervised mineralogy heads (~4-6 weeks)

| step | output | gate |
|---|---|---|
| port the v3 corpus regen + run pilots 1, 2, 2b, 3, 4 | `results/v3_pilot/` + an explicit CompHead parameterization decision | pilot decision gate passes (see `~/.claude/plans/sunny-weaving-nebula.md` section "Pilot decision gate") |
| run the full v3 corpus regen | `data/MAGEMin/fc_corpus.parquet` (per-phase composition + optional G_sys) | 95%+ FC convergence; per-bulk audit clean |
| build ModeHead (Dirichlet) | head module | predicted alpha positive; mean sums to 1 |
| build CompHead (using pilot-2b parameterization) | head module | physically valid outputs (within stoichiometric bounds for endmember version; within oxide-sum constraints for oxide version) |
| build CondHead (GMM per axis) | head module | NLL computable; sigmas positive; weights sum to 1 |
| build optional ClassHead + G_sys head | head modules | each can be turned off without breaking joint loss |
| write joint loss + mass-balance term | `losses.py` | each term has correct sign; backward works on a random batch; gradient flows to encoder |
| train single seed on full v3 corpus | a checkpoint per ablation cell | matches or beats v2 cell-c per-phase R²_bulk |
| K=5 ensemble + calibration | the deployment artifact | 1-sigma coverage closer to nominal 0.68 than v1's 0.88 / 0.64 / 0.76 |

Acceptance: per-phase R²_bulk improves over v2 cell-c on the hydrous
phases (amp/bi/fl/mu/ne) that the corpus regen + CompHead were
designed to help; CondHead calibration is honest.

v2 cell-c baseline to beat: CondHead conditions T 14.8 °C MAE,
P 1.73 kbar, H2O 0.51 wt%; the per-phase R²_bulk gains cell-c booked
over v1 on the hydrous + foid-bearing phases were bi +0.18, ne +0.22,
mu +0.23, amp +0.08 -- the same phases the v3 corpus regen + CompHead
target.

### Track C: end-to-end fine-tune (optional, last)

If Track B leaves measurable headroom, unfreeze the encoder and
fine-tune with all heads end-to-end at a low encoder LR
(`encoder_lr = 0.1 * head_lr`, same recipe as v2 cell-c). Run
ablation against the Track-B frozen-encoder version. Ship whichever
calibrates better.

## Ownership split: what you own vs what Claude brute-forces

The split that respects "you have ML ownership" and "Claude eats the
boring stuff." A practical rule of thumb at the end.

### You own (conceptual mastery, decisions, math)

- **Loss function design.** Dirichlet NLL, GMM NLL, mass-balance term,
  joint weights, label smoothing. This is the intellectual core. Design
  on paper; have Claude write the numerically-stable code from your spec.
- **Padding/masking semantics.** Got bitten once. You should be able
  to draw the data-flow diagram for `src==0` vs `frac==0` vs
  masked-for-SSL on a whiteboard from memory.
- **Head parameterization choices.** Dirichlet alpha clamp, GMM
  mixture count, CompHead oxide-vs-endmember. These determine model
  behavior; pick them deliberately.
- **The encoder forward pass.** ~50 lines of pytorch. Type it
  yourself; understand every line. This is the foundation model.
- **Evaluation methodology.** R²_bulk vs per-path R², PIT calibration,
  bulk-level vs row-level splits. Two corpus-handling bugs in v1 were
  methodology bugs, not modeling bugs.
- **The mass-balance constraint.** How CompHead and ModeHead link via
  `bulk = Σ M × C`. Chemistry-meets-ML core.
- **Whether to include G_sys and ClassHead.** Decision yours; the
  corpus regen cost for G_sys is real, ClassHead is free.

### Claude brute-forces (boring/mechanical)

- **Dataset classes and dataloaders.** Parquet reading, batch
  collation, igneous filter, splits handling, supergroup labeling.
  Patterns already in v2; can be ported and adapted.
- **Visualization scaffolding.** UMAP, violin plots, calibration plots,
  ablation tables. Established patterns; just say what plots against what.
- **Configuration handling.** YAML loaders, argparse plumbing, config
  validation, manifest serialization.
- **Memory-leak workarounds.** `maxtasksperchild=1` + `chunk_size=25`
  + per-path Julia GC + `caffeinate`. Just port from v1.
- **Phase-name mapping tables.** The `_phase_map.py` + the
  `_ALL_SHORT_TO_SHORT` table added in `visualize_degeneracy.py`.
  Tedious lookup work; copy verbatim.
- **mat2vec CSV loading + element-properties bookkeeping.** Boilerplate.
- **HPO loop scaffolding.** Optuna study, storage, objective wrapping.
  You specify the search space and metric; Claude writes the loop.
- **Checkpoint save/load.** Boilerplate that needs care with state-dict
  naming but is mechanical.
- **Training loop boilerplate.** Epoch iteration, early stopping,
  logging. Once the loss is specified, the loop writes itself.
- **Test/smoke scripts.** "Encoder produces non-zero output", "head
  produces valid Dirichlet alpha", "loss gradient flows to all
  parameters". Important to have, mechanical to write.
- **Corpus regen orchestration.** The chunked-parquet + retry-failed
  pipeline. Largely lifted from `scripts/production/`.

### Blurred middle (you understand, Claude types)

- **Head MLPs.** Architecture is straightforward (`d_model ->
  d_model//2 -> n_outputs` residual block). You read the code and
  understand why each layer is there; Claude types it once you say
  "Dirichlet head with softplus + clamp 1e-6, 2-layer residual MLP."
- **Optuna objective.** You decide search bounds and metric; Claude
  wraps the train-eval loop into a callable.
- **The cnmineral package skeleton.** You decide the module structure;
  Claude generates the empty files and the `__init__.py` exports.
- **The corpus regen schema additions.** You decide what new columns
  are persisted (per-phase composition, G_sys, etc.); Claude modifies
  `02_generate_corpus.py`'s extraction logic.

### Practical rule of thumb

If a code chunk encodes a *decision*, you write it. If it encodes the
*consequence* of an already-made decision, Claude writes it.

## Agentic session pattern

Instead of long sessions where Claude writes large chunks (lots of
room to drift from intent), prefer short focused sessions with a
bounded spec. Pattern:

```
"Port the unlabeled dataset class from v2 to v3 with these schema
 changes: [list]. Keep the igneous filter logic exactly as-is.
 Acceptance: length matches ~474k; supergroup counts are within 1%
 of the v2 numbers in memory file mineralml_grain_data.md."
```

```
"Write the SSL training loop. Loss is MSE on masked positions.
 Decoder is d_model -> d_model//2 -> 22, on pooled latent (NOT
 per-token; see [[kingcrab_padding_convention]] for why). Mask 25%
 of populated tokens by zeroing BOTH src AND frac. Save checkpoint
 and a manifest each epoch. Acceptance: held-out reconstruction MSE
 drops monotonically over 8 epochs."
```

```
"Add a smoke test: random batch through encoder + each head, assert
 all outputs have correct shape and at least one non-zero gradient
 on the encoder. Save as tests/test_smoke.py. Acceptance: the test
 passes on a CPU machine in under 10 seconds."
```

Each is bounded, verifiable, and doesn't require Claude to make
design decisions you'd want to make. The encoder forward, the loss
math, the head parameterizations -- type those yourself.

## Suggested first-week sequence

A concrete week-one to get unstuck on the blank-page problem.

Day 1 (you): set up the branch, draft the package skeleton (top-level
modules: `data/`, `models/`, `losses/`, `training/`, `eval/`), write
the encoder forward (~50 lines), draft the SSL loss math on paper.

Day 2 (claude): port mat2vec loader, element-properties CSV, the
phase-mapping tables. Mechanical.

Day 3 (you): write the encoder unit test; write the SSL pretrain
script's `forward + loss + backward` core. Verify on a random batch
that encoder grads are non-zero (the v2 sanity gate).

Day 4 (claude): port the BulkChemistryDataset class from v2 with the
schema changes you specify (e.g., supergroup label addition, trace
column list, scale config).

Day 5 (you): write the pretrain loop (epoch iteration, optimizer step,
checkpoint save). Claude can do the logging plumbing.

Day 6 (claude): port the UMAP + probe diagnostics from
`scripts/09_visualize_latent.py` + the recoloring from
`scripts/09b_latent_chem_axes_and_3d.py`. These are established
patterns.

Day 7 (you): kick off the SSL pretrain on the full 474k corpus
(~16 min on MPS per v2 numbers). Verify the encoder reproduces v2
cell-c probe numbers.

End of week 1: Track A is roughly half done. Encoder works, SSL loop
runs, first diagnostic plots land. Two more weeks of probe / OOD /
mineralML-style figures get Track A across the gate.

## What survives from v2 unchanged

The strategy modules and the corpus-gen orchestration are well-tested
and shouldn't be rewritten:

- `src/cnmineral/data/fe_redox.py`, `tp_priors.py`, `h2o.py` -- per-class
  sampling priors as injectable strategies. The ADRs justifying these
  (003, 004, 005) still apply.
- `scripts/production/_phase_map.py` + the `_ALL_SHORT_TO_SHORT` table
  from `scripts/experiments/visualize_degeneracy.py` -- the PTT
  phase-name maps. Just copy.
- `scripts/production/01_find_T_FC_end_per_bulk.py` and
  `02_generate_corpus.py` -- the FC corpus pipeline with memory-leak
  workarounds. Adapt the extraction to also persist per-phase
  composition (~10 line addition) and optionally G_sys.
- `data/splits_v1.json` if you keep the v1 split; rebuild if you
  change input vocab or filter rules. Bulk-level split is non-negotiable.

## What survives from v2 changed but referenced

- The v2 4-cell ablation methodology (random/SSL × majors/majors+traces).
  v3 expands this with cell (e) auxiliary ClassHead and possibly cell
  (f) G_sys head, but the ablation table format stays.
- The K=5 ensemble pattern. Same idea, applied to v3 architecture once
  the single-seed result is favorable.
- The Optuna HPO setup. Search space updates with new hyperparams
  (lambda_balance, K_endmember if endmember CompHead, etc.) but the
  TPE-with-JournalStorage harness is unchanged.

## What v2 had that v3 starts without

- ALIGNN structural stream (deferred again to v4+)
- Polymorph head (still no polymorph labels in the corpus)
- fO2 sweep (still class-conditional)
- TracePartitionHead (Kd-database lookup work not done)

These are not gates for v3 shipping. They're v4 candidates.

## The decision points you'll hit during the rewrite

In rough chronological order, decisions where you should NOT delegate
to agentic work:

1. **Encoder dim / depth / heads.** v1 won at d=512, N=4, heads=16.
   Start there. Change only if HPO suggests it.  (v3 HPO re-searched
   the space and landed at d=256, N=4, heads=8.)
2. **Mask probability for SSL.** v2 used 25%. Sweep {15, 25, 40} if
   pretraining is sensitive.
3. **K_gmm for CondHead.** v1 won at K_gmm=3. Bigger gives more
   multimodality, more parameters to fit.
4. **CompHead parameterization (oxide-wt% vs endmember-fraction).**
   Pilot 2b. Critical; see plan + `[[comphead_parameterization_options]]`.
5. **lambda_balance for the mass-balance loss term.** Pilot 3.
   Sweep 0 / 0.1 / 1.0 / 10.0.
6. **Whether to add G_sys head.** Costs corpus regen rerun if not
   persisted. Skip unless you can persist it in the same regen.
7. **Whether to add ClassHead aux supervision.** Free; just adds a
   tiny MLP. Include in cell (e) ablation.
8. **K (ensemble size).** v1 used K=5. Stay there unless inference
   latency matters.
9. **Whether to add Track C (end-to-end fine-tune).** Decide after
   Track B's eval lands.

Each of these is a coin flip you want to make consciously. Don't let
an agentic session "default" you into one.

## Memory hygiene during the rewrite

The memory files in `~/.claude/projects/-Users-mliu-git-clones-crabnet-mineral-v2/memory/`
will accumulate v3 facts. Keep using them. Worth a one-time pass at
the start of v3 to mark stale v1/v2 facts (the kingcrab padding bug
becomes "historical context"; the SSL findings become "validated
prior result"). Do this in `MEMORY.md` by editing the one-line
descriptions, not by deleting files -- the content is still useful
context.
