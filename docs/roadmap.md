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
- **Corpus scaling** (#60 2026-08-06, #69 2026-09-03): more simulated
  compositions under the current generator do not help real rocks. The
  #69 read found a hump, not a plateau: zero-shot error rises from 78 to
  about 625 compositions and flattens, in every mineral category, while
  corpus-side emulation keeps improving. The mechanism is label mismatch
  (rung, water prior, redox), not composition coverage.

## Three thrusts, in order

Ruled 2026-09-02, amended 2026-09-03. All on `main` until the third.

### 1. Research week (now)

- **Epic #74, generator calibration against measured mineralogy.** Run in
  order: #75 (rung-60 labels vs the 50 + 30 ladder), #76 (regularised
  10,000-composition head vs the 78 head), #77 (matrix recalibrator on the
  78 head), then #78 (the generator on the 640 benchmark compositions at
  all four rungs, model-free: buried conditions, wrong rung, or wrong
  recipe, per category). #78's verdict closes the week. #79 (the
  generator-prior search) is deferred past thrust 2 and the camera-ready.
- **#63, plutonic harvest**: du Bray DS 942 and PP 1793, then the current
  heads evaluated zero-shot on plutonic point counts to measure the gap
  #22 must close. Inside #74 this is the second ruler: the prior selected
  on volcanic phenocryst counts should not be the one plutonics select.
- **#22, the plutonic recipe**: equilibrium-to-solidus of the rock's own
  composition (cheap, current generator) versus fractional crystallisation
  of parents (PetThermoTools back). #78 is this crux read on the volcanic
  side, so the two decide together.
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
