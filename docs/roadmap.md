# v3 execution roadmap

Practical execution plan for the v3 rewrite. Distinct from
`docs/encoder.md` (which crystallizes the architecture and value
proposition); this doc covers track division, gate tables, and the
ownership split between you and agentic work.

## Track division: foundation model A, then mineralogy heads B

The v2 cell-c result already validated this pattern. SSL-pretrained
encoder + supervised mineralogy is what won; the encoder doesn't need
to keep changing during supervised training. So: build A fully, save
the checkpoint, freeze it, then build B on top.

End-to-end fine-tuning (Track C) is optional and last. Probably small
gains given how strong the SSL encoder is. Add only if Track B's eval
suggests headroom.

### Track A: foundation-model encoder (closed 2026-06-11)

Wrapped up: SSL encoder pretrained on Gard 2019 (~167k igneous
bulks with clean majors) at the post-Optuna hyperparameters in
`configs/pretrain/pretrain_production.yaml`.  Canonical budget is
20 epochs (a 40-epoch comparison landed within stochastic noise:
supergroup MLP +0.003, qap_name MLP -0.003 -- SSL loss kept
descending but downstream metric did not).  That 40-epoch
comparison was on the retired linear-space encoder, so it holds for
the canonical log encoder only by analogy -- never tested at 40
epochs directly (JOURNAL 2026-06-24, #15).

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
unblocked now; Track B is gated on the equilibrium corpus regen
(`docs/corpus_regen_spec.md`; issue #9 tracks it).  The ~500-bulk
pilot passed its health gate 2026-07-14; the training-rung pick and
production launch are pending.

A third thread, **Track A part 3** (epic #15, on the
`encoder-representation` branch), pursues a *stronger* encoder via new
SSL pretext objectives.  Motivation: the masked-element objective
decodes rock type well (supervised probe 0.769) but does not arrange
the latent so same-class rocks are neighbors (kNN purity 0.582) -- a
representation-geometry gap, not an epoch-budget one (JOURNAL
2026-06-24).  #15 targets it with pull-together (non-contrastive),
forgery-detection (ELECTRA-style), and group-masking pretexts, judged
on raw-latent probe + distance-weighted kNN rather than UMAP-2D.

### Track B: supervised mineralogy heads (~4-6 weeks)

| step | output | gate |
|---|---|---|
| pilot the equilibrium corpus regen (~500 bulks, done 2026-07-14) | pilot chunks in `data/corpus/train/` + health report + rung pre-screen | attrition/convergence/closure/coverage pass (they did); owner picks the training rung(s) |
| run the production regen | `data/corpus/{train,valtest}/` (10k x 48 + 500 x 128 draws; modes, `Comp_wt`, `G_sys` per rung) | health report clean at scale |
| build ModeHead (Dirichlet) | head module | predicted alpha positive; mean sums to 1 |
| build CompHead (per-phase oxide wt% `Comp_wt`, resolved 2026-06-30) | head module | physically valid outputs (within oxide-sum constraints; endmember bonus heads loss-masked to clean phases) |
| build CondHead (GMM per axis) | head module | NLL computable; sigmas positive; weights sum to 1 |
| build optional ClassHead + G_sys head | head modules | each can be turned off without breaking joint loss |
| write joint loss + mass-balance term | `losses.py` | each term has correct sign; backward works on a random batch; gradient flows to encoder |
| train single seed on full v3 corpus | a checkpoint per ablation cell | matches or beats v2 cell-c per-phase R²_bulk |
| K=5 ensemble + calibration | the deployment artifact | 1-sigma coverage closer to nominal 0.68 than v1's 0.88 / 0.64 / 0.76 |

Acceptance: per-phase R²_bulk improves over v2 cell-c on the hydrous
phases (amp/bi/fl/mu/ne) that the corpus regen + CompHead were
designed to help; CondHead calibration is honest.

Status (2026-07-21): heads-study epic (#32) COMPLETE -- per-head best
checkpointing (#33, gate #34 passed), main sweep (#35: 42 trials,
trial-21 winner adopted), architecture mini-sweeps (#36: comp keeps
128x1, cond adopts 512x2), production retrain + readout (#37: replicates
the sweep cell exactly; alpha0 compression and the closure signed bias
CURED, cond teacher-forced calibration honest). Per-head evals run
behind the single `scripts/eval/joint_heads.py` entry point (standalone
ModeHead pipeline deleted 2026-07-21). Confirmed structural, next in
order: accessory over-prediction (#41, post-hoc hurdle recalibration on
the new reliability instrument) then the staged amp magnitude fixes
(#25, mixture branch); cond exposure bias measured wider post-tuning
(#31, stays P3 per mineralogy-first). #38/#39 closed on the readout
evidence; journal 2026-07-21 has the full record.

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
- **Corpus regen orchestration.** Landed:
  `scripts/build/generate_mineralogy_corpus.py` (chunked, atomic
  per-chunk parquet, manifest-resumable).

### Blurred middle (you understand, Claude types)

- **Head MLPs.** Architecture is straightforward (`d_model ->
  d_model//2 -> n_outputs` residual block). You read the code and
  understand why each layer is there; Claude types it once you say
  "Dirichlet head with softplus + clamp 1e-6, 2-layer residual MLP."
- **Optuna objective.** You decide search bounds and metric; Claude
  wraps the train-eval loop into a callable.
- **The cnmineral package skeleton.** You decide the module structure;
  Claude generates the empty files and the `__init__.py` exports.
- **The corpus regen schema.** You decided the persisted columns
  (`corpus_regen_spec.md` "Storage shape"); Claude implemented the
  three-table layout in `cnmineral.data.magemin_corpus`.

### Practical rule of thumb

If a code chunk encodes a *decision*, you write it. If it encodes the
*consequence* of an already-made decision, Claude writes it.

## What survives from v2 unchanged

What actually carried over, and what the volcanic pivot re-scoped:

- `src/cnmineral/data/fe_redox.py` -- ported, in production use (the
  class-conditional Middlemost Fe split). v2's `tp_priors.py` / `h2o.py`
  sampling strategies were NOT ported: ADR 003's per-class H2O ranges
  migrated to the inference prior, and (P, H2O) sampling is the bounded
  proposal in `cnmineral.data.condition_proposal`.
- The v2 PTT phase-name maps (`_phase_map.py`, `_ALL_SHORT_TO_SHORT`)
  and the FC pipeline (`01_find_T_FC_end_per_bulk.py`,
  `02_generate_corpus.py`) are FC-side assets -- deferred with the
  plutonic/cumulate branch (ADR 004 amendment, issue #22).
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
   RESOLVED 2026-06-30: per-phase oxide wt% for all phases (endmember
   recon on real bulks leaves the simplex); endmember bonus heads only
   on clean phases, loss-masked. See `corpus_regen_spec.md`.
5. **lambda_balance for the mass-balance loss term.**
   Sweep 0 / 0.1 / 1.0 / 10.0 when the joint loss lands.
6. **Whether to add G_sys head.** `G_sys` is persisted per rung in the
   regen (spec "Storage shape"), so the head is now a free decision --
   no corpus cost either way.
7. **Whether to add ClassHead aux supervision.** Free; just adds a
   tiny MLP. Include in cell (e) ablation.
8. **K (ensemble size).** v1 used K=5. Stay there unless inference
   latency matters.
9. **Whether to add Track C (end-to-end fine-tune).** Decide after
   Track B's eval lands.

Each of these is a coin flip you want to make consciously. Don't let
an agentic session "default" you into one.
