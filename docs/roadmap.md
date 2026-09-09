# Roadmap

Execution plan after the 2026-08-27 submission. `docs/encoder.md` and
`docs/heads.md` say what the system is; this doc says what happens next
and in what order. Tasks live in GitHub issues; this doc names them and
the rulings that sequence them. Status stamped 2026-09-03.

## Where the system stands

- **Encoder** (Track A, closed 2026-06-11): `pretrain_allrock_redox_seed0`,
  all-rock, 23 tokens with the Fe-redox ratio, ADR 007 objective; canonical
  since 2026-08-05. Value proposition and data flow in `encoder.md`;
  representation rubric in `scorecard.md` (dormant while encoder work is
  deprioritized).
- **Heads** (Track B, shipped 2026-07-22): `joint_heads_allrock_tuned_seed0`
  is production (cascade-2 verdict, 2026-08-05), trained on the
  first-generation corpus (10k compositions x 48 draws, rungs 50 + 30; ADR
  004 "Generation record"). Architecture and posterior in `heads.md`;
  per-version numbers in `results/ledger.yaml`.
- **Real-rock benchmark**: eruption-v2, 640 evaluated rocks, split 372
  train / 106 val / 162 test by canonical unit key (#67, 2026-09-03). The
  matrix recalibrator (119 parameters) is canonical; the tailed tier was
  deleted once its margin bisected to the amphibole pin. Restamped val:
  zero-shot 11.29, vector 1.42, matrix 0.83 nats. The rebuilt test split
  has never been evaluated.
- **Paper**: submitted 2026-08-27, notification 2026-09-29. The stack is
  frozen for it; `docs/camera_ready_reconcile.md` lists the macros that
  moved between the submitted numbers and HEAD, and is deleted at
  camera-ready. Refactors must not move stamped numbers before then.
- **Corpus scaling** (#60 2026-08-06, #69 2026-09-03, grid 2026-09-04):
  the #69 hump was the stopping rule, not the corpus. Heads were selected
  where corpus val was best, which is 56 gradient steps at 78 compositions
  and about 5,600 at 10,000; at matched depth, or stopped on the real-rock
  train split, zero-shot error is flat in corpus size. After
  recalibration corpus size helps modestly at 56 to 224 steps, all of it
  in the width of the amount distributions, none in the predicted amounts.
  Fitting depth is the large lever, the generator the small one so far.
  The target is a **scaling pair**: a simulated label convention and a
  measurement convention whose error falls with corpus size on the two
  review figures (`explorations/corpus_scaling/README.md`, "Corpus size at
  fixed fitting depth").

## Three thrusts, in order

Ruled 2026-09-02, amended 2026-09-03. All on `main` until the third.

### 1. Research week (now)

- **Epic #74, the search for a scaling pair.** #75, #76, #77 measured
  2026-09-03/04 (journal). Next, in order: #78 (the generator on the 640
  benchmark compositions at all four rungs, model-free: buried conditions,
  wrong rung, or wrong recipe, per category; also the only cheap test of
  whether a scaling pair can exist under equilibrium at all); then each
  generator setting #78 points at becomes one candidate corpus drawn on
  the canon figure, single seed, three seeds on the endpoints of any line
  that falls (the traced-regen corpus on disk is the first candidate);
  #79 (the generator-prior search, the one candidate cluster job) only if
  those figures show a direction, with its objective written before any
  corpus is generated. The stopping rule (fixed step budget, and on which
  ruler) is deliberately undecided until those results coalesce; no head
  is promoted and production is untouched.
- **#63 and #22, the plutonic branch**, are parked: a different recipe
  and a different counting convention, so a volcanic failure to scale
  says nothing about them. They open only if every volcanic pair tried
  fails to scale.
- **Lawrencium**: the corpus-size use is dropped (#69). The one candidate
  workload is #79.

### 2. Software thrust

- #71 architecture audit (abstractions, duplication, coupling, config
  brittleness) and #70 stable inference API. #73, the terminology sweep,
  rides this thrust since it touches every identifier.
- Product definition (settled 2026-09-02, recorded on #70): the user is
  anyone with elemental data, by spreadsheet upload or API call, dozens of
  rocks per batch; the output is the full inference card (presence, ranked
  assemblages, compositions, conditions, plot and numbers); recalibration
  is optional and only on the caller's own labelled rocks; corpus
  generation, training and cascades stay offline. Hosting and front end
  are abstracted away.
- Deliverable is refactors plus issues, not a document. Natural milestone:
  public release after the 09-29 notification, under a freeze discipline
  so nothing moves stamped numbers before camera-ready.

### 3. Crystal-structure pretraining

Polymorphs and solid solutions: the largest open research question, on
its own era branch with dedicated time. Not before thrust 2.

## Deprioritized and parked

- Encoder-representation issues: #65, #64, #16, #17, #19, #55, #28, #14.
- Heads and output issues: #47, #40, #21.
- #72, comparison-arm follow-ups after the split rebuild: housekeeping,
  unsequenced.
- Track C (end-to-end fine-tune of the encoder with the heads): never
  started; no headroom evidence has asked for it.

## Closed tracks and where their records live

- Track A: `encoder.md`, `scorecard.md`, ADR 007; the #15 pretext epic on
  the `encoder-representation` branch is closed.
- Track B: `heads.md`, `results/ledger.yaml`, journal 2026-07-21 through
  2026-07-25 (heads study #32, support-conditioned magnitude #25, the
  benchmark and the #52 few-shot curve).
- Cascades and the recalibrator: journal 2026-07-31 through 2026-08-06
  (cascade 1 against the traced corpus, cascade 2's encoder verdict, the
  mixture kept as a dial, the tailed tier) and 2026-09-02/03 (#67, the
  tailed deletion, the restamp).
