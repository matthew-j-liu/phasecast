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

## Title (working; owner-picked 2026-08-04)

> Simulated petrological labels enable transfer learning for
> probabilistic mineralogy prediction from rock geochemistry

Runner-up kept for reference: "Simulated petrological labels and few-shot
transfer to measured rocks enable probabilistic mineralogy prediction from
bulk chemistry". Open micro-option: "whole-rock geochemistry" as the
field-standard term for the input, at the cost of one word.

## Abstract (draft v5, 2026-08-04; punctuation budget + ML-audience glosses; ~230 words)

> Mineralogy, the identity and abundance of the mineral phases in a
> rock, guides both petrological interpretation and industrial
> extraction. Measuring it remains slow, requiring an expert to count
> mineral grains under a microscope or diffraction measurements on
> carefully prepared samples. Whole-rock chemical assays are the
> opposite case,
> with standardized instruments reporting a rock's elemental
> concentrations cheaply and at scale. Predicting mineralogy from an
> assay would therefore convert an abundant measurement into the
> quantity decisions act on. Two obstacles stand in the way. The mapping
> is one-to-many, since a single composition crystallizes into different
> mineral assemblages depending on formation conditions the rock does
> not record, and no compilation pairs rock chemistry with measured
> mineralogy at training scale. Here, we supervise probabilistic
> mineralogy predictors with simulated petrological labels: a
> transformer encoder pretrained self-supervised on
> \sysPretrainBulksAllrock measured whole-rock compositions, heads
> trained on equilibrium assemblages from the MAGEMin thermodynamic
> simulator along \sysCorpusTrainDraws sampled (pressure, water) paths
> per composition, and a \sysRecalibParams-parameter recalibration fit
> on measured volcanic mineralogy. Scored zero-shot against \rrBenchN
> erupted rocks, the model ranks minerals and rocks correctly but
> overstates confidence. A recalibration fit on \sysBenchTrainRocks
> training rocks reduces validation joint negative log-likelihood from
> \rrValZsProd to \rrValAdProd nats, and \sysFewshotBudgetKnee labeled
> rocks alone reach a median \fsTwentyFiveMatrixProd. Notably, sharper
> simulator sampling degraded transfer while
> improving every simulator-side metric, and selecting models on
> validation likelihood favored overconfidence through a hidden
> \sysEffLambdaFactor{}x loss-weight mismatch we identify and correct.
> Simulated labels thereby enable transfer learning where measured
> labels cannot exist at scale.

## 1. Introduction (draft prose, ~0.75 page)

Paragraph plan with draft text; macros stand where numbers go.

Draft v5, 2026-08-04, written against style.md incl. the punctuation
budget (one colon, one semicolon per section, [cite]/[slot] brackets
exempt) and ML-audience glosses (techniques described plainly, domain
shorthand spelled out). Five paragraphs. The intro's one colon is spent
on P2's simulator triplet, its one semicolon on P5's width-not-location
sentence. The old P5/P6 collapse into the tour, and the (i)(ii)(iii)
contributions list is absorbed by it (restorable if reviewers expect
explicit bullets).

> **P1 -- the funnel.** Mineralogy is the identity and abundance of the
> mineral phases in a rock. It is the starting material of petrological
> investigation, because crystals record the temporal, compositional,
> and thermal evolution of the magmatic systems that grew them [cite
> mineralML and refs therein]. The same phases govern industrial
> processing. How a critical element is extracted, from the choice of
> acid to the temperature to the separation route, depends on which
> mineral hosts it and what surrounds that host [cite Minerals 2025].
> Identical elemental content can therefore demand entirely different
> processing. Demand for these elements is projected to grow steeply
> under the energy transition, and recovering them from ores and waste
> streams is a stated policy priority [cite DOE 2023; IEA 2024].
> Determining mineralogy, however, stays expensive. The standard routes
> are an expert counting mineral grains under a microscope on a prepared
> slice of rock, or X-ray diffraction on powdered samples, and both
> demand preparation and expert time [cite mineralML]. Whole-rock
> chemical assays sit at the opposite extreme. Instruments such as X-ray
> fluorescence (XRF) and inductively coupled plasma (ICP) spectrometry
> report a rock's elemental concentrations, are standardized and cheap,
> and already run at scale in exploration and production [cite NRMP
> 2023]. Thus, reading mineralogy directly from a chemical assay would
> convert the measurement everyone has into the quantity decisions act
> on. The assay alone could then inform how separations are designed.
>
> **P2 -- what exists (a one-condition equilibrium simulator).**
> Predicting mineralogy from chemistry means modeling rock formation,
> and rock formation resists direct attack. Melts stall, cool, and
> crystallize over kilometers and millennia. Thermodynamics, kinetics,
> and transport couple across those length and time scales, and no
> instrument observes the process end to end. Laboratory petrology
> constrains the endpoint one point at a time, with each experiment
> equilibrating one composition at one pressure, temperature, and water
> content. Decades of such experiments have been distilled into
> thermodynamic databases [cite database]. Gibbs-energy minimizers such
> as MAGEMin turn those databases into a fast forward simulator [cite
> Riel 2022]: one composition, one fully specified condition, the
> equilibrium assemblage. To date, first principles reach no further.
> The full formation process is too multi-scale and multi-physics to
> simulate outright.
>
> **P3 -- the inverse gap, named.** The question practitioners ask runs
> opposite to the simulator. An assay fixes the composition, while the
> mineralogy and the conditions that produced it are unknown. That
> inverse is one-to-many. A single bulk composition can crystallize
> into, in principle, infinitely many mineral sets and abundances,
> depending on the pressure and water path it followed. An honest
> predictor must therefore return a distribution over mineralogies
> rather than an answer. Supervised learning cannot close the gap
> directly, because no public compilation pairs rock chemistry with
> measured modal mineralogy at training scale. A model fit to the small
> labeled collections that exist would be confined to their
> compositions, with no notion of its own uncertainty. Existing tools
> bracket the problem without solving it. Normative calculations
> idealize an anhydrous equilibrium mineralogy from bulk chemistry [cite
> CIPW/pyrolite]. Grain-level classifiers such as mineralML identify
> minerals from chemistry measured on individual crystals [cite
> mineralML]. Sibling fields in materials informatics routinely train
> composition models on simulated labels, precisely because measured
> ones are scarce [cite]. Thus, the obstacle is a label gap, and
> simulated petrological labels are the available way through.
>
> **P4 -- the system (three reasoned bets).** We therefore built the
> predictor in three layers, each a bet on where its information should
> come from. We pretrained our own transformer encoder self-supervised
> on \sysPretrainBulksAllrock measured whole-rock compositions [cite
> gard2019]. The hypothesis was that a representation learned from
> nature, never from the simulator, keeps user inputs in-distribution
> regardless of the label source. The architecture adapts the core logic
> of CrabNet and none of its surrounding machinery [cite CrabNet]. Each
> element enters as a text-mined embedding (mat2vec [cite]) summed with
> a learned encoding of its abundance, which plays the role position
> embeddings play in language models. We also reasoned from
> transfer-learning theory. The feature space of the pretrained model
> determines its capability to transfer downstream ("features are fate"
> [cite Rotskoff group, ICML 2025]), so pretraining should happen on
> exactly the features practitioners measure. The \sysInputTokens
> elements of a routine assay are cheap and high-throughput physically,
> and by that theory they form the one representation positioned to
> serve both simulator-labeled training and measured-rock transfer. We
> then supervised probabilistic heads on the frozen encoder with
> simulated petrological labels. Each of \sysCorpusTrainBulks
> compositions is crystallized along \sysCorpusTrainDraws sampled
> pressure and water paths, so the within-rock spread of assemblages
> becomes trained uncertainty. A hurdle head factorizes which minerals
> are present from their abundances given presence. A conditions head
> returns the inverse posterior over temperature, pressure, and water
> content, conditioned on the assemblage. Last, we fit a
> \sysRecalibParams-parameter recalibration on measured volcanic
> mineralogy, confining the sim-to-real correction to a module the size
> of a linear probe.
>
> **P5 -- in this work.** In this work, we evaluated the stack on
> \rrBenchTotal erupted rocks compiled with both whole-rock chemistry
> and measured mineral proportions, of which \rrBenchN least-altered
> rocks form the headline cohort, split by geological unit. Zero-shot,
> with heads that had never seen a real rock, ranking and per-mineral
> discrimination transfer while stated confidence does not; the failure
> concentrates in predictive width, not location. The recalibration
> reduces validation joint NLL from \rrValZsProd to \rrValAdProd nats.
> A few-shot curve over labeled-rock budgets reaches a median
> \fsTwentyFiveMatrixProd nats at \sysFewshotBudgetKnee rocks. What no
> constant correction absorbs is rock-dependent width, with
> residuals running heavier-tailed than the head's distributional family
> at the fitted mean. Importantly, two controlled comparisons expose how
> the simulator should be used. Regenerating the corpus with sharper,
> per-rock condition sampling improved every simulator-side validation
> metric and degraded real-rock transfer, surviving identical
> recalibration [slot: cascade-2 winner]. Broad condition sampling had
> been hedging over exactly the uncertainty real rocks present. And
> selecting loss weights on validation likelihood favored an
> overconfident model even in-distribution, through a hidden averaging
> mismatch that put the tuned weight at \sysEffLambdaTuned{}x the
> properly scored joint density, where misspecification theory
> prescribes tempering down [cite Grunwald; Bissiri et al.]. Overall, this work establishes
> simulated petrological labels as a workable supervision source for
> probabilistic mineralogy prediction, and demonstrates that how the
> simulator is sampled, and how models are selected against it, decide
> whether the result transfers. The encoder and heads are
> re-supervisable slots, laying a foundation for label sources beyond
> the simulator as measured compilations grow.

Related work is now partially folded (option B in effect): P3 carries
the existing-tools survey (CIPW/pyrolite, mineralML, materials
informatics precedent); the methodological citation families (claims.md
A-G) ride sections 4-5 where each is used; the fuller family-by-family
discussion (mineralML and other models, in the ANN-sample register)
moves to the appendix. Section 6 in the skeleton below stays only if the
owner wants a visible stub.

The five-paragraph intro runs ~1 page against the 0.75 budgeted; section
2 tightens to compensate, and the motivation is owner-directed emphasis,
so the budget bends around it, not the reverse.

Scope guard for P1's industrial framing: the system predicts phase
presence and abundances (the "surrounding mineralogical environment"),
NOT per-element host-phase assignment -- trace partitioning is a deferred
head. Motivation prose stays on "informs separations"; never "identifies
the host of element X".

Motivation pillars (owner, 2026-08-04), worked into draft v2 above --
kept here as the record of what the prose must always carry:

1. For a CS audience, establish how vast, unexplored, and complicated
   crustal rock formation is: real measurements are hard; lab experiments
   are hard in their own right; those experiments are what calibrated the
   thermodynamic databases MAGEMin minimizes over. First principles alone
   cannot be thrown at a system that traverses length and time scales
   with coupled kinetics, thermodynamics, and transport -- so a simulator
   is unavoidable. What distinguishes this work is judicious use of that
   simulator to supervise an encoder that is itself trained on a large
   corpus of measured rock compositions, which is what enables transfer.
2. The problem is inherently one-to-many: a single bulk measurement can
   be produced by, in principle, infinitely many mineral sets and
   abundances. The mathematics is intractable as a point-prediction
   problem, not just the physics -- which is why the modeling is
   probabilistic and why the corpus explores many (P, H2O) paths per
   rock.

The confluence of the two is the stated reason the work takes a
sim-to-real approach to probabilistic prediction.

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
- Mixture postscript, discussion register (journal 2026-08-04; exhibit
  explorations/archive/mode_mixture/): a K=2 assemblage mixture
  faithfully captures the corpus's two-hump amphibole structure yet
  transfers tie-leaning-worse -- at matched chemistry the benchmark
  population holds the corpus's full width in ONE hump, the second hump
  being a product of the deliberately broad hidden-(P, H2O) proposal;
  the 7-group instrument passes 61% of the corpus advantage, so
  coarsening is not the bottleneck. A simulation-side prior limitation,
  not a model or procedure failure -- connects corpus-design choices to
  transfer behavior and motivates #62 (benchmark populations with
  genuine path multimodality).
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
- Benchmark population is one-humped at matched chemistry, so the
  mixture mechanism is validated on corpus structure only and the dial
  stays K=1; populations that genuinely crystallize both wet and dry at
  matched bulk chemistry are the #62 search.
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

## Candidate figures (DRAFT -- owner iterates this list before any figure is built; PNG mode; build after structure approval)

| # | figure | sources | placement |
|---|---|---|---|
| F1 | system overview schematic: three layers, what each trained on, output views | drawn (conventions from slides/overview.py) | sec 2 |
| F2 | transfer headline: per-version zero-shot joint NLL split into presence/magnitude + recalibrated val with paired CIs | real_rock_ledger.yaml; both contrast JSONs | sec 3 |
| F3 | few-shot curve, 3 checkpoints, matrix tier, val | curve_val.json x3 | sec 3/4 |
| F4 | selection divergence: teacher-forced val NLL vs sampled-support / real-rock direction across versions | ledger.yaml; runs/mode_head/*/metrics.json; real_rock_ledger.yaml | sec 5 |
| A1 | residual decomposition (width vs location) | explorations/real_rock_fewshot residual decomposition outputs | appendix D |
| A2 | cohort card (production) -- NOT the climatology-null panels | results/joint_heads_supportmag_seed0/cohort_card.png regenerated into paper/figures/ | appendix F |
| A3 | mixture mechanism: bimodal corpus amphibole-share truth histogram with the K=2 gate-weighted components on both humps | regenerate at paper quality from explorations/archive/mode_mixture/run_mixture_figures.py -- do NOT inherit the mockup results/beta_overlay_k2_amp.png | appendix, beside the sec 4/7 mixture discussion |
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
