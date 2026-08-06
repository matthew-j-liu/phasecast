# v3 execution roadmap

Practical execution plan for the v3 rewrite. Distinct from
`docs/encoder.md` (which crystallizes the architecture and value
proposition); this doc covers track division, track status, and the
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
linear-space MSE, which is retired to reference.  Two encoder
generations exist: `pretrain_v3_logmse_std_seed0` (igneous-only,
22-token -- the previous canonical) and the canonical encoder
`pretrain_allrock_redox_seed0` (all-rock, 23-token with the Fe-redox
ratio; accepted ride-as-is 2026-07-28, adopted as canonical 2026-08-05
on the cascade-2 tuned-vs-tuned encoder verdict; scored in
`docs/scorecard.md`).  The production heads sit on the canonical
encoder.

Probe results on the 20-epoch igneous encoder
(`scripts/eval/probe_encoder.py`, library-default probe HPs; probe
snapshots are transient `runs/` artifacts, regenerable; current
numbers live in `docs/scorecard.md`):

| probe | test accuracy | v2 cell-c bar |
|---|---|---|
| supergroup MLP | 0.821 | 0.76 (+6.1) |
| qap_name MLP   | 0.769 | 0.72 (+4.9) |
| qap_name MLP (probe-HPO-tuned) | 0.790 | -- |
| supergroup LR  | 0.749 | -- |
| qap_name LR    | 0.722 | -- |

The tuned row is the same encoder probed at the
probe-HPO best params; the
library-default rows are the apples-to-apples comparison against
v2 cell-c.

k-NN qap purity at k=5/10/50: 0.582 / 0.554 / 0.487 against a
random baseline of 0.100 -- the encoder organizes rock chemistry
into qap-coherent neighborhoods without ever seeing a label.

Stops here: the encoder is the durable contribution.  Two follow-up
tracks came out of it: **Track A part 2** (`pretrain_applications.md`)
and **Track B** (below; its gating corpus regen shipped via
`docs/corpus_regen_spec.md` / #9).  The all-rock relaxed-Fe
re-pretrain (#24) LANDED 2026-07-28 as `pretrain_allrock_redox_seed0`
-- see "Current status" at the end of this doc.

A third thread, **Track A part 3** (epic #15, on the
`encoder-representation` branch), pursues a *stronger* encoder via new
SSL pretext objectives.  Motivation: the masked-element objective
decodes rock type well (supervised probe 0.769) but does not arrange
the latent so same-class rocks are neighbors (kNN purity 0.582) -- a
representation-geometry gap, not an epoch-budget one (JOURNAL
2026-06-24).  #15 targets it with pull-together (non-contrastive),
forgery-detection (ELECTRA-style), and group-masking pretexts, judged
on raw-latent probe + distance-weighted kNN rather than UMAP-2D.

### Track B: supervised mineralogy heads (built; first full version shipped 2026-07-22)

Every step of the original gate table landed; the architecture and
posterior live in `docs/heads.md`, per-version numbers and their
instruments in `results/ledger.yaml` (which supersedes the v2 cell-c
acceptance bars written here as the running comparison). Terse trail,
with the journal holding each record:

- Joint ModeHead + CompHead + CondHead on the frozen encoder:
  heads-study epic #32 (main sweep, architecture mini-sweeps,
  production retrain; journal 2026-07-21/22), then the
  support-conditioned magnitude branch (#25: joint hurdle NLL
  -1.16 -> -2.46, amp magnitude r 0.715 -> 0.876). Shipped as
  `joint_heads_supportmag_seed0` (production until the 2026-08-05
  cascade-2 promotion); evals behind the single
  `scripts/eval/joint_heads.py` entry point.
- Presence-side architecture work stopped on evidence: the ceiling
  probe measured the corpus, not the architecture, as the
  discrimination limit (#41); redox input-invisibility attributed and
  routed to the regen as an optional input token (#46).
- Real rocks became a normal benchmark (eruption-v2: 724 scoreable /
  640 headline; `results/real_rock_ledger.yaml`). Zero-shot joint NLL
  10.406 -- ranking transfers, absolute scale does not (journal
  2026-07-24) -- and the #52 few-shot curve: ~25 labelled rocks from
  eight eruptions recover 3.4-3.8 of the 5.0 available nats via
  post-hoc recalibration on the frozen heads (journal 2026-07-25).
  Under the segmented-ledger convention this benchmark is the
  cross-instrument judge.

Still open in this area, sequenced behind cascade 2: the ModeHead
Dirichlet mixture (the scheduled next architecture change, motivated
by amp bimodality + the heavier-than-Beta real-rock residual + the
value of broad-draw hedging; #49 is its amp-specific ancestor), G_sys
head (#43), prediction-set output (#47), K-seed ensemble + calibration
(the one original-table row not yet built).

### Track C: end-to-end fine-tune (optional, last)

If Track B leaves measurable headroom, unfreeze the encoder and
fine-tune with all heads end-to-end at a low encoder LR
(`encoder_lr = 0.1 * head_lr`, same recipe as v2 cell-c). Run
ablation against the Track-B frozen-encoder version. Ship whichever
calibrates better.

## Current status and next steps (2026-08-05)

Full decision record in `docs/2026_08_03_recap.md`; journal entries
2026-07-31 through 2026-08-05 hold the measurements.

**Cascade 1 (heads on the traced corpus) is complete and adjudicated
against.** Both regen checkpoints (`joint_heads_regen_seed0`,
`joint_heads_regen_tuned_seed0`) were trained, measured, and lost to
the old-corpus production model on the real-rock benchmark under
identical recalibration (sealed test joint NLL 0.478 / 0.769 / 1.371;
all pairwise unit-clustered contrasts exclude zero; few-shot curves
show no crossover at any labelled budget). Production stayed
`joint_heads_supportmag_seed0` through the cascades (the 2026-07-29
trial-22 adoption is reversed; the regen checkpoints are measured
non-production arms) until the cascade-2 verdict promoted
`joint_heads_allrock_tuned_seed0` on 2026-08-05 (step 1 below).
The traced corpus's real wins (quartz presence, assemblage ranking)
and the sharpening lesson are the paper story, not a production
upgrade. The real-rock **test slice is spent**; recalibrator work
validates on train/val only. The all-rock encoder re-pretrain (#24)
landed and was accepted 2026-07-28 (`pretrain_allrock_redox_seed0`).

Next, one change at a time, each judged on the real-rock benchmark:

1. **Cascade 2 -- encoder swap on the old corpus. COMPLETE, verdict
   rendered 2026-08-05.** All-rock arm (journal 2026-08-03/04): the
   baseline at production dials tied production on recalibrated val,
   and the corrected-criterion sweep winner
   (`joint_heads_allrock_tuned_seed0`, effective lambda 0.59, ModeHead
   512x3) posted the best corpus numbers on both axes and tied with
   the best point estimate. The matched igneous sweep independently
   picked near-identical dials (trial 43 -- effective lambda 0.609,
   512x3); its 120-epoch retrain (`joint_heads_igneous_tuned_seed0`)
   trails on both corpus axes and the paired recalibrated val contrast
   leans all-rock without separating (-0.209 [-0.460, +0.090]). Per
   the pre-registered rule the tie licenses adoption on independent
   merits: **the all-rock encoder is canonical and
   `joint_heads_allrock_tuned_seed0` is production** (journal
   2026-08-05). #54 closed -- comp/cond mini-sweeps waived, superseded
   by planned re-tunes that re-open the full search. The #53
   fresh-head control still decomposes encoder-vs-head contribution
   when scheduled.
2. **ModeHead dirichlet layer (mixture)** -- VERDICT RENDERED 2026-08-04
   (journal, two entries): kept as a dial, K=1 stays the default.
   Trained K=2 wins the corpus on both axes and collapses the amp
   two-sided PIT residue, but leans against the K=1 control on
   recalibrated real-rock val (-0.365 [-0.864, +0.118], all in the
   magnitude term). Attribution measured: the 7-group instrument passes
   61% of the corpus advantage (not the bottleneck); the real amp
   population at matched chemistry has the corpus's total width but ONE
   hump (the second hump is a property of the broad hidden-(P, H2O)
   prior); a single gate-offset recalibration parameter recovers ~60%
   of the penalty, to a tie, never a win. `joint_heads_allrock_mixture_
   seed0` stays on disk as the K=2 reference; no ledger entries.
3. **Recalibrator upgrades -- COMPLETE, verdict 2026-08-06** (journal
   2026-08-05/06). The `tailed` tier -- matrix plus a two-parameter
   mean-preserving precision contamination -- is the canonical
   recalibration layer and headline transfer instrument: val joint
   1.277 -> 0.972, +0.304 [+0.049, +0.659] over matrix, the gain
   concentrated on interval-miss rocks (score repair, not
   stated-interval repair). The SiO2 magnitude-side term was built,
   measured at +0.007 [-0.004, +0.016], and excised; the gate-offset
   term stays contingent on K>1 ever fielding; rendered verdicts stay
   stamped to the matrix tier. The #52 few-shot curve re-ran on
   production: matrix and tailed indistinguishable at ~26 rocks,
   tailed wins every paired draw from 50 rocks up, and ~26 rocks
   still recover ~77% of the magnitude gain.
4. **#60 real-rock-axis corpus scaling** (now unblocked) -- decides
   whether a larger broad-draw regen is generated. Prioritized over
   #55 encoder re-HPO for the NeurIPS window: the corpus axis owns
   every large measured real-rock effect, and the venue is the
   simulator side; #55 stays open, parked past the deadline.
5. **Paper progression**: the real-rock benchmark is the cross-era
   spine, plus a one-time table re-scoring archived era checkpoints
   on the final valtest instrument. Era production checkpoints
   (encoder + heads) therefore stay loadable and archived until that
   table is measured.

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
