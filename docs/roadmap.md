# Roadmap

Execution plan after the 2026-08-27 submission. `docs/encoder.md` and
`docs/heads.md` say what the system is; this doc says what happens next
and in what order. Tasks live in GitHub issues; this doc names them and
the rulings that sequence them. Status stamped 2026-09-18.

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
  the #69 hump was the stopping rule, not the corpus. At matched fitting
  depth, or stopped on the real-rock train split, zero-shot error is flat
  in corpus size; after recalibration corpus size helps modestly, all of
  it width. The target is a **scaling pair**: a simulated label convention
  and a measurement convention whose error falls with corpus size on the
  two review figures (`explorations/corpus_scaling/README.md`, "Corpus
  size at fixed fitting depth").
- **Generator calibration against the phenocryst ruler, closed** (#78
  2026-09-16, #79 2026-09-18; `explorations/generator_reach/`,
  `explorations/prior_search/`; journal both dates). Equilibrium on each
  benchmark rock's own chemistry reaches the measured share of each
  silicate category at some draw, but no single (P, H2O) reaches all four
  silicates together for 38% of rocks; no prior over conditions, global or
  conditioned on the majors, moves the silicate labels by more than 0.03
  in share; and the heads already score within 0.01 of a per-rock oracle
  prior. For the pair (single-condition equilibrium, du Bray phenocryst
  counts) the generator's conditions are not the lever. The condition
  prior is a settled decision (CLAUDE.md).
- **The label's signal, re-read** (#85, 2026-09-29;
  `explorations/generator_reach/`, journal that date). #79 measured how
  far the label sits from the measured share, not whether it tracks
  rocks. After a per-category line fit on train, the simulator reaches a
  constant forecast over each rock's mineral list on the four silicates
  (0.102 against 0.106); feldspar and quartz beat it, amphibole and
  biotite carry nothing. #79's narrow claim stands, its broad one does
  not. The 224-step line with last-epoch heads on three seeds (journal
  2026-10-05): recalibrated share error does not fall with corpus size
  on the silicates; only quartz falls, and it stops by 1,250
  compositions. Open: #86, which part of the falling joint NLL carries it.

## Three thrusts, in order

Ruled 2026-09-02, amended 2026-09-03 and 2026-09-18. All on `main` until
the third.

### 1. Research thrust: the scaling pair, second attempt (now)

- **What the first attempt settled (epic #74, steps 1 to 3, 2026-09-03
  to 09-18).** The hump was the stopping rule (#75 to #77, the depth
  grid); the generator reaches each silicate category but not all four at
  one condition (#78); no condition prior, global or chemistry-conditioned,
  moves the silicate labels, and the heads already sit at the labels'
  ceiling (#79, both screens run on the #78 draws without MAGEMin). Owner
  ruling 2026-09-18: for the current (generator, ruler) pair there is no
  way to change how the generator runs to attain scaling. The pair has to
  change, on the ruler side or the recipe side.
- **Path 2 first, the ruler.** du Bray's plutonic whole-rock modes read
  on the current generator at a solidus rung. The benchmark is built
  (#82, journal 2026-10-02; `explorations/real_mineralogy/README.md`,
  "The plutonic benchmark"): 507 evaluated rocks from DS 942 and PP
  1793, split by whole intrusion. Next the model-free reach on each
  plutonic rock's own bulk as #78 did, opening with the #85 signal read
  per category under the two settings the review figures use (no
  correction, the production recalibrator) against a ladder of baselines
  (the constant over the mineral list, the head given no chemistry, the
  CIPW norm, the head given raw geochemistry), heads first and before
  any training (the heads' read is done and re-read under grouped 5-fold
  cross-validation with the 68 oxide-lumped rocks, journal 2026-10-05: the
  recalibrated production head and the recalibrated CIPW norm tie at 0.042
  and both beat the simple baselines, so 0.042 is the bar the generator
  must clear; Stage A is read, journal 2026-10-06: the simulator near the
  solidus ties the bar after recalibration at 0.044, beats the norm only
  on joint NLL, -0.18 with the band clear of zero, and tracks amphibole
  and biotite where the volcanic read did not; at the head's own rungs
  the simulator still reads 0.056 to 0.059 against the head's 0.080, and
  read as a marginal the way the simulator is read the head scores 0.058,
  so the production head has converged to the simulator on these rocks
  and the zero-shot gap was the estimator, not the rung, not copying
  error and not the sparse trace tokens (journal 2026-10-06). What a
  corpus-size ladder can show here is smaller heads approaching that
  ceiling, which is the simulator's own score, the norm's after
  recalibration on share error with a small NLL edge. Next, owner's call:
  the on-disk head ladder read on the 404 rocks with both estimators as a
  screen; #87; one regeneration under its policy with the whole cooling
  path kept; then the ladder as the full test), and
  the ladder on the two review figures (#83) with the
  `explorations/prior_search/rosters/` sizes and training seeds on the
  endpoints. FC-of-parents (#22) serves cumulates only, and the plutonic
  benchmark shows almost none, so #83 builds no cumulate rule (journal
  2026-10-05). Its outcome sorts path 1: whole-rock equilibrium scaling
  on whole-rock modes says the volcanic failure was the phenocryst
  ruler; failing there too says the problem is deeper than the recipe.
- **Path 1 after it, the recipe.** Fractional crystallization of the
  rock's own bulk with the accumulated crystals as the label, the
  phenocryst cargo as it forms, against the du Bray ruler (#84). Run on
  both benchmarks 2026-10-09 through the exploration's own loop over the
  production MAGEMin_C, checked against PetThermoTools (ADR 004 amendment
  2026-10-09; PetThermoTools stays the reference, not the driver). Read on
  the plutonic rocks 2026-10-13: recalibrated 0.038 against the bar's
  0.042 with the paired band on zero, amphibole kept and biotite starved,
  so partial removal is the recipe question between the two (journal
  2026-10-13; the eruption read waits on #85).
- **Held throughout.** The stopping rule stays fixed depth on the review
  figures; no head is promoted; production and the paper's numbers are
  untouched. The condition prior for the volcanic pair is settled
  (CLAUDE.md); oxides, olivine and quartz carry #78's recipe verdict.
- **Lawrencium**: no cluster use until a generation workload exists. A
  plutonic solidus corpus or an FC corpus would be the case; the port
  (Julia plus the pinned Manifest, a pandas-only env, the bulk table and
  rosters, a sharded driver) was scoped 2026-09-16 and is delayed by the
  owner.

### 2. Software thrust

- #81, the first sweep under the rule that production code is not an
  experiment's configuration surface (CLAUDE.md, 2026-09-10): eleven
  verified cases in `src/` and `scripts/`, one commit each, before or as
  the opening step of #71.
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
