# Structure proposal -- Sim2Science workshop paper

Status: PROPOSAL, awaiting owner approval before any further drafting.
Prose protocol: the abstract and introduction below are full-prose DRAFTS for
the owner to rewrite or accept; every other section is bullets by design.
All numbers in tex prose are macros from `numbers.tex` (current values shown
here in brackets for orientation only, with their source). `[slot: ...]`
marks where cascade-2 results swap in if they land before the deadline.

Venue: Sim2Science (NeurIPS 2026 workshop), 5-page track excluding
references, unlimited appendix, non-archival, double-blind, checklist
mandatory after references. Deadline 2026-08-29.

## Title candidates (owner picks or rewrites)

1. Emulating an imperfect thermodynamic simulator: mineralogy transfer
   fails in confidence, not accuracy
2. Broad simulator supervision transfers, sharp supervision does not:
   sim-to-real mineralogy from a rock-chemistry foundation encoder
3. A probabilistic emulator of MAGEMin under uncertain conditions,
   evaluated on measured volcanic mineralogy

## Abstract (draft prose, ~170 words)

> Thermodynamic equilibrium codes such as MAGEMin compute the mineral
> assemblage of a rock composition at one specified pressure, temperature,
> and water content. Real rocks form at unknown conditions, and the codes'
> equilibrium labels are an imperfect proxy for the mineralogy geologists
> measure. We present a system that emulates the simulator integrated over
> uncertain conditions: a transformer encoder pretrained self-supervised on
> \sysPretrainBulksAllrock measured rock compositions, probabilistic heads
> trained on simulator assemblages sampled over many plausible
> (pressure, water) paths per rock, and a \sysRecalibParams-parameter
> recalibrator fit on measured volcanic mineralogy. Scored zero-shot on
> \rrBenchN erupted rocks, the system ranks minerals and rocks correctly
> but overstates its confidence; the small recalibrator recovers most of
> the gap, and 25 labeled rocks recover most of the rest. Two findings
> bear on learning from imperfect scientific models: sharper, more
> realistic simulator sampling made real-rock transfer worse, because
> broad sampling had been supplying condition uncertainty the model needs;
> and model selection on validation likelihood reliably chose
> overconfident models, which we trace to a hidden 21x loss-weight
> mismatch and correct.

## 1. Introduction (draft prose, ~0.75 page)

Paragraph plan with draft text; macros stand where numbers go.

> **P1 -- the problem.** A thermodynamic simulator answers a conditional
> question: given a bulk composition and one set of formation conditions,
> which minerals form. The scientific question is usually marginal and
> inverse: given only the composition, what mineralogy is plausible, and
> under what conditions. Integrating the simulator over geologically
> plausible conditions is what a petrologist does mentally and what the
> simulator does not natively compute. Beyond that, the simulator's answer
> is imperfect for the rocks we care about: equilibrium crystal cargo at a
> sampled condition is a proxy for the phenocryst modes measured on an
> erupted rock, and no public compilation pairs bulk chemistry with
> measured modal mineralogy at training scale.
>
> **P2 -- the system.** We train a system in three layers with different
> exposure to the simulator. A transformer encoder is pretrained
> self-supervised on \sysPretrainBulksAllrock measured rock compositions
> (masked-element prediction), so the representation never sees anything
> synthetic. Probabilistic heads read the frozen latent and are trained on
> simulator labels: for each of \sysCorpusTrainBulks training compositions
> we run MAGEMin fractional crystallization along \sysCorpusTrainDraws
> sampled (pressure, water) paths, so the within-rock spread of assemblages
> across paths becomes the trained uncertainty rather than noise to average
> out. A hurdle head factorizes presence from abundance-given-presence; a
> conditions head learns the inverse posterior over (T, P, H2O)
> conditioned on the assemblage. Finally a \sysRecalibParams-parameter
> recalibrator -- per-mineral constant corrections -- is fit on measured
> volcanic mineralogy, confining sim-to-real correction to a module the
> size of a linear probe.
>
> **P3 -- what we measure.** We evaluate on \rrBenchN erupted rocks with
> point-counted or estimated phenocryst modes, split by stratigraphic unit.
> Zero-shot, ranking and accuracy transfer while stated confidence does
> not: the failure is predictive width, not location. The recalibrator
> absorbs most of the penalty; a few-shot curve over labeled-rock budgets
> shows 25 rocks buy most of the remaining gain. What no constant
> correction absorbs is rock-dependent width: residuals are heavier-tailed
> than the head's distributional family at the fitted mean.
>
> **P4 -- the two workshop findings + contributions.** First, a corpus
> regeneration that made simulator sampling sharper and more realistic per
> rock -- conditioning the sampled paths on trace-element chemistry --
> improved every simulator-side validation metric and made real-rock
> transfer worse, surviving identical recalibration. Broad condition
> sampling had been functioning as a transfer asset: implicit hedging over
> conditions the sharper corpus deleted. Second, hyperparameter selection
> on validation likelihood chose an overconfident model even
> in-distribution; a hidden averaging mismatch meant the tuned loss weight
> sat at \sysEffLambdaTuned{}x the properly-scored joint density, and the
> literature's prescription under misspecification points the opposite
> direction (temper the suspect factor down). We contribute: (i) the
> system and its measured zero-/few-shot transfer, (ii) the
> sharper-supervision result, (iii) the corrected selection criterion.

- Related-work option A: short standalone section 6 (below). Option B:
  fold citation families into P1/P4 and drop section 6 for space. Owner
  picks; I recommend A while the page budget allows.

## 2. System: a probabilistic emulator of an imperfect simulator (bullets, ~1 page incl. Fig 1)

- Three-layer exposure table (the paper's organizing device, from
  heads.md's transfer lens): encoder = real chemistry only; heads =
  simulator labels on real inputs (gap confined to label semantics);
  recalibrator = real labels, \sysRecalibParams params.
- Encoder: \sysInputTokens-token composition (9 majors + \sysTraceTokens
  traces), mat2vec element embeddings + fractional encoding, masked-mean
  pool; SSL = masked-element modeling, per-element standardized log-space
  MSE. [slot: cascade-2 winner -- whether the paper's production stack
  carries the igneous 22-token or all-rock 23-token encoder]
- Supervised corpus: per-bulk MAGEMin fractional crystallization,
  \sysCorpusTrainDraws draws over literature-bounded (P, H2O) boxes,
  crystals-renormalized solid assemblage labels over \sysPhaseVocab
  phases; many-to-one by design (the within-bulk variance IS the target
  distribution). valtest at \sysCorpusValtestDraws draws; bulk-level
  splits.
- Heads (one compact paragraph each, math in appendix):
  - ModeHead hurdle: presence Bernoullis x support-conditioned Dirichlet
    (chain rule; teacher-forced at train, sampled at inference). Why a
    single Dirichlet fails (shared concentration).
  - CondHead: per-axis GMM over (T, P, H2O) conditioned on the support --
    the inverse-inference output.
  - CompHead: per-phase oxide Dirichlet (one line; mostly appendix).
  - Joint loss in effective-lambda units: state \sysEffLambdaFactor{}x
    nominal-to-effective conversion ONCE here; all reported weights in
    effective lambda thereafter.
- Output views: marginal table vs ranked scenarios (one worked scenario
  line, maybe in Fig 1).

## 3. Zero-shot and few-shot transfer to measured mineralogy (bullets, ~1.25 pages incl. Figs 2-3)

- Benchmark: \rrBenchTotal rocks with chemistry + modes from published
  eruption records; \rrBenchN least-altered headline cohort;
  \rrBenchGroups coarsened mineral groups (coarsening exact on both
  hurdle factors); split \sysBenchTrainRocks/\rrValNRocks/\rrTestNRocks
  by stratigraphic unit. Censored trace-only records reported beside the
  headline, never inside it.
- Zero-shot headline: joint NLL \rrZsJointProd (production checkpoint,
  eruption-v2, n=\rrBenchN), decomposed presence vs magnitude. Accuracy
  transfers (per-group ranking/AUROC flat across versions), confidence
  does not -- with claim A's two qualifications (amphibole
  discrimination; accessory drift in log-ratio geometry) stated, not
  hidden.
- Recalibrator: tiers, fit on \sysBenchTrainRocks train rocks; val
  \rrValAdProd after correction. The one sealed test read: \rrTestAdProd
  / \rrTestAdRegen / \rrTestAdRegenTuned, disclosed as spent
  (single disclosed read, all later selection on train/val).
- Scope limit (claim F): affine/concentration rescaling recovers NLL and
  location, not coverage -- residuals heavier-tailed than any Beta at
  the fitted mean; SiO2-structured residual survives on three groups.
  (Coverage FIGURE is #59-gated; the coverage NUMBERS in text are part
  of the calibration claim -- owner decides whether a calibration claim
  ships, which gates both.)
- Few-shot: matrix tier at 25 rocks \fsTwentyFiveMatrixProd vs zero-shot
  -- most of the recoverable gap at trivial label cost. Framed as
  demonstrated capability, not deployment.
- [slot: cascade-2 winner -- headline table gains a row when the two
  matched sweeps report; text slots for "which encoder wins at tuned
  best".]

## 4. Sharper simulator supervision transferred worse (bullets, ~0.6 page)

- The regen: trace-conditional (P, H2O) sampling, more bulks, fewer
  draws -- better on every simulator-side metric; worse zero-shot on
  rocks (\rrZsJointRegen vs \rrZsJointProd), nearly all in the Dirichlet
  width term.
- Identical matrix recalibration does not save it: sealed test
  \rrTestAdProd vs \rrTestAdRegen (paired unit-clustered contrast
  \rrTestConProdRegenMean\ [\rrTestConProdRegenLo,
  \rrTestConProdRegenHi]); few-shot curves never cross.
- Mechanism stated as hypothesis, not isolated (sampler, draw count, and
  roster changed together): narrowed per-bulk condition sampling deleted
  rock-dependent hedging that broad marginalization supplied implicitly;
  constant corrections cannot restore per-rock width.
- Convergent evidence for the same missing structure: measured amphibole
  bimodality, heavier-than-Beta residuals, the broad-draw advantage.
- What the traced corpus DID buy, stated fairly: assemblage ranking; the
  quartz-presence improvement only as #57 permits (verify the issue's
  guard before this bullet ships).
- Workshop moral: for transfer, how you integrate over the simulator's
  uncertain inputs can matter more than per-sample label fidelity.

## 5. Model selection under misspecification (bullets, ~0.6 page)

- The hurdle likelihood factorizes exactly (Cragg/Mullahy): the
  presence/magnitude weight is not a likelihood parameter; at effective
  lambda 1 the loss is the joint hurdle log-density; other values =
  Bissiri-Holmes-Walker general-Bayes loss (NEVER "improper scoring
  rule").
- The hidden 21x: phase-averaged BCE vs summed Dirichlet NLL; nominal
  weight understates effective lambda by \sysEffLambdaFactor{}x; the
  sweep's winner ran effective lambda \sysEffLambdaTuned.
- Under misspecification the literature tempers the suspect factor DOWN
  (Grunwald; semi-modular inference); our measured misspecification
  (amphibole bimodality a single Dirichlet cannot represent) says the
  Dirichlet is the suspect factor -- the sweep moved it up.
- Validation-NLL selection prefers overconfident models even
  in-distribution (Mukhoti et al.): trial 22 won teacher-forced val NLL
  by ~4x the tie band while regressing every real-rock and
  sampled-support metric.
- The corrected criterion (shipped): search effective lambda at or below
  1; sampled-support NLL as a tiebreak veto. Live confirmation: on the
  encoder-swap baseline the veto axis agreed with the rocks while
  teacher-forced NLL pointed the other way.
- [slot: cascade-2 winner -- first sweep run fully under the corrected
  criterion; outcome sentence lands here.]

## 6. Related work (bullets, ~0.25 page; drop to intro if space forces)

- Simulator/domain: MAGEMin; mineralML (spot analyses, the complementary
  input regime); CrabNet-family composition encoders (mat2vec,
  fractional tokenization, with attribution).
- Calibration under proper losses and shift: Guo et al.; Ovadia et al.
  (shift caveat on post-hoc recalibration); Gneiting et al. sharpness-
  subject-to-calibration.
- Scoring-rule geometry for the NLL/CRPS magnitude gap: Bolin & Wallin
  (claim B's frame).
- Hurdle/two-part joint models in ecology (Kong et al. closest
  architectural precedent); general-Bayes/tempering (Bissiri; Grunwald;
  semi-modular inference).
- Gaps we state as gaps (claim H): no hurdle-Dirichlet for mineral
  assemblages; no named mean-preserving Dirichlet concentration
  recalibration.

## 7. Limitations + conclusion (bullets, ~0.4 page)

- Single simulator, single seed per production checkpoint (seed
  variability bounded only by the tie-band convention); mechanism in
  section 4 hypothesized, not isolated.
- Benchmark: visual estimates + point counts, volume-to-mass conversion,
  censored traces, thin classes caveated; one spent test read.
- Recalibrator guarantee weakens under shift (Ovadia); coverage not
  recovered (claim F scope limit).
- Conditions head evaluated against simulator draws only (no measured
  (P, H2O) truth at benchmark scale).
- Capability, not product: no deployment claims.
- 2-3 sentence conclusion: the encoder is the durable layer; heads are
  re-supervisable application slots; the recalibrator is where the
  sim-to-real correction concentrates.

## Appendix inventory (unlimited, "optional reading")

- A. Architecture + training detail: tokenization, encoder config, SSL
  objective, head parameterizations + math, param counts
  (\sysHeadParams heads on \sysEncoderParams frozen encoder), dials in
  effective-lambda units.
- B. Corpus generation: FC protocol, (P, H2O) proposal + bounds, rungs,
  renormalization, vocabulary rules, splits + stratification.
- C. Benchmark construction: sources, freshness rule, exact coarsening
  (noisy-OR presence; Dirichlet aggregation), censoring treatment,
  thin-class list, unit-clustered bootstrap procedure.
- D. Recalibrator: tier definitions, exact param count, fit protocol,
  residual decomposition (location absorbed / width not), SiO2
  structure.
- E. HPO + selection: search spaces, corrected criterion mechanics,
  trial-22 record.
- F. Extended results: per-group tables, per-tier few-shot, exposure
  bias, benchmark-version cross-checks (v1 vs v2 stability of the
  zero-shot headline).
- G. Compute + reproducibility statement (owner supplies hardware
  facts).

## Candidate figures (build after structure approval; PNG mode)

| # | figure | sources | placement |
|---|---|---|---|
| F1 | system overview schematic: three layers, what each trained on, output views | drawn (conventions from slides/overview.py) | sec 2 |
| F2 | transfer headline: per-version zero-shot joint NLL split into presence/magnitude + recalibrated val with paired CIs | real_rock_ledger.yaml; both contrast JSONs | sec 3 |
| F3 | few-shot curve, 3 checkpoints, matrix tier, val | curve_val.json x3 | sec 3/4 |
| F4 | selection divergence: teacher-forced val NLL vs sampled-support / real-rock direction across versions | ledger.yaml; runs/mode_head/*/metrics.json; real_rock_ledger.yaml | sec 5 |
| A1 | residual decomposition (width vs location) | explorations/real_rock_fewshot residual decomposition outputs | appendix D |
| A2 | cohort card (production) -- NOT the climatology-null panels | results/joint_heads_supportmag_seed0/cohort_card.png regenerated into paper/figures/ | appendix F |
| (gated) | coverage panel | -- | #59-gated |

## Checklist item -> section map (born compliant)

| checklist item | answered by | note |
|---|---|---|
| 1 Claims | abstract + intro contributions == secs 3-5 | |
| 2 Limitations | sec 7 | |
| 3 Theory | NA -- no new theoretical results (factorization cited, not proven here) | |
| 4 Reproducibility | secs 2-3 + appendices A-E | |
| 5 Open access | OWNER DECISION -- weights policy says no releases until submission; benchmark sources public | open q |
| 6 Experimental details | secs 2-3 + appendices A-E | |
| 7 Statistical significance | paired unit-clustered contrasts + tie band, secs 3-4 + appendix C | a strength |
| 8 Compute | appendix G | owner supplies hardware facts |
| 9 Code of ethics | yes, standard | |
| 10 Broader impacts | short appendix statement | low-risk geoscience emulation |
| 11 Safeguards | NA -- no high-misuse-risk assets | |
| 12 Licenses | appendix B/C citations; verify licenses during citation pass | |
| 13 New assets | benchmark harvest, IF released -- owner decision | open q |
| 14 Crowdsourcing | NA | |
| 15 IRB | NA | |
| 16 LLM usage | NA per policy (writing/engineering assistance, not core method) -- owner confirms | open q |

## Submission logistics (standing)

- Reciprocal reviewing: OWNER must nominate one qualified reviewer at
  submission. Owner's task, flagged every session.
- Anonymization: dblblindworkshop suppresses the author block, but the
  \author macro in neurips_2026.tex carries real names -- sweep prose,
  links, acks before any submission-bound render.
- Checklist after references; delete its instruction block before
  submission; answers use the provided macros only.
- 2-page tiny-paper track is the undiscussed fallback.
