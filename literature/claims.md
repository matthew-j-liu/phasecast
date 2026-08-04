# Claim-to-citation map

What the manuscript asserts, what our own evidence for it is, and which paper
plays which role.  Organized by argument rather than by paper, because a claim
usually draws on several references and a reference usually serves several
claims.  Entries in [literature/references.bib](references.bib).

**Roles.**  `supports` = the paper backs the claim.  `contextualizes` = places
our result in an established frame.  `complicates` = cuts partly against us and
must be engaged, not omitted.  `we-extend` = the gap our work fills.

**Verification.**  Author, title, and venue for every entry were cross-checked
against model knowledge and are sound.  Volume, page, and DOI fields came from
literature-review agents and are **not confirmed against primary sources**.
Nothing here ships until each cited entry is opened and checked.  Items marked
UNVERIFIED could not be corroborated at all and must be confirmed or dropped.

---

## A. The sim-to-real failure is calibration, not accuracy

**Claim.**  Heads trained on simulator labels rank real rocks correctly but
state their confidence wrongly; the transfer gap is dominated by predictive
width, not by predictive location.

**Our evidence.**  Per-group Spearman and AUROC unchanged across model versions
while joint NLL moves 10.4 -> 24.4 nats; empirical coverage of stated 80%
credible intervals is 30-39% for every version; per-group Murphy decomposition
gives reliability ~2.3x resolution.

- `gneiting2007calibration` -- **contextualizes.**  The "maximize sharpness
  subject to calibration" paradigm.  Sharpness gained while calibration is lost
  is not an improvement.  This is the frame for the coverage-sharpness figure.
- `guo2017calibration` -- **contextualizes.**  Canonical result that models
  trained with a proper log loss are still overconfident; establishes that our
  symptom is expected rather than anomalous.  Also: single-scalar temperature
  scaling beat more expressive vector/matrix maps, which overfit the
  calibration set -- independent support for what our own recalibrator tiers
  found.
- `minderer2021revisiting` -- **complicates.**  Argues the overconfidence trend
  is architecture-dependent and much weaker in newer architectures.  Cite so
  the claim is not overstated as universal.
- `nixon2019measuring` -- **contextualizes.**  ECE is binning-sensitive and
  rank-orders recalibration methods inconsistently; justifies why we report
  coverage curves and Brier decomposition rather than leaning on ECE alone.

## B. Log score and CRPS disagree here for a known, formal reason

**Claim.**  The metric disagreement (NLL badly degraded, CRPS statistically
tied) is a property of the scoring rules, not evidence that one is wrong.

**Our evidence.**  Pooled CRPS 0.093 / 0.097 / 0.099 with overlapping
unit-clustered intervals against joint NLL 10.4 / 18.4 / 24.4; worst 10% of
rocks carry 64% of the NLL gap while the median rock moves 4 nats.

- `bernardo1979expected` -- **supports.**  Log score is the unique proper and
  *local* rule: it sees only the density at the realized observation.
- `gneiting2007proper` -- **supports.**  Restates locality; CRPS is proper, in
  the units of the observation, and reduces to MAE for a point forecast.
- `selten1998axiomatic` -- **supports.**  Names the two mechanisms exactly:
  *hypersensitivity* (diverging penalty as assigned probability approaches
  zero) and *insensitivity* (one catastrophic miss swamps the record).  This is
  the closest formal statement of our tail behaviour.
- `bolin2023local` -- **complicates, and this one is essential.**  CRPS scales
  linearly with forecast dispersion, making it comparatively *under-sensitive*
  to concentration changes -- exactly the quantity that differs between our
  models.  So the CRPS tie is partly a property of CRPS.  Both metrics are
  biased with respect to what we measure, in opposite directions.  Their SCRPS
  is a candidate addition to our metric suite.

## C. Our joint loss: what the hurdle factorization does and does not license

**Claim.**  At weight 1 the objective is exactly the joint hurdle log-density.
The relative weight has no likelihood-based justification; it acts only through
the shared encoder.

**Our evidence.**  `mode_head_loss` sums a phase-averaged presence BCE and a
batch-averaged masked Dirichlet NLL, so the effective weight against the
properly-scaled joint NLL is 21x the nominal `magnitude_weight`: lambda = 5.3
at the baseline dials and **163** at the tuned dials, against lambda = 1 at
`magnitude_weight` 0.048.

- `cragg1971some`, `mullahy1986specification` -- **supports.**  The two-part /
  hurdle likelihood is additively separable when the two components' parameters
  are variation-independent, so joint and separate MLE are numerically
  identical and a cross-term weight is not a likelihood parameter.
- `cox1987parameter` -- **contextualizes.**  The general parameter-orthogonality
  framework the hurdle case specializes.
- `manning1987monte` -- **contextualizes.**  Two-part models have uncorrelated
  decision and outcome processes, unlike Heckman selection, which does not
  factorize.
- `bissiri2016general` -- **supports.**  At weight != 1 the objective remains a
  valid composite / general-Bayes loss but is no longer the log-density of any
  single probability model.  This is the precise wording to use; do **not**
  write "improper scoring rule," which the literature does not support -- a
  positively-weighted sum of proper rules stays proper for eliciting each
  sub-target.
- `kong2020deep` -- **supports, closest architectural precedent.**  Shared
  encoder feeding a multivariate probit occurrence head and a log-normal
  abundance-given-presence head.  Couples the heads with a structural
  covariance penalty, **not** a scalar loss weight.
- `ovaskainen2017how`, `tikhonov2020joint` -- **contextualizes.**  When hurdle
  components share structure, the established route is shared latent factors in
  a proper joint posterior, not a tuned scalar.

## D. Tempering direction: we moved the weight the wrong way

**Claim.**  Under misspecification the established prescription is to
down-weight the suspect likelihood factor.  Our HPO up-weighted it 163-fold.

**Our evidence.**  Amphibole bimodality measured in 45% of bulks, which a single
Dirichlet structurally cannot represent; trial 22 raised the magnitude weight
31x nominal while real-rock coverage fell and NLL rose 6 nats.

- `grunwald2017inconsistency` -- **supports.**  Under misspecification,
  likelihood-based inference can be inconsistent via hypercompression; the
  repair is a learning-rate exponent that *lowers* the likelihood's influence,
  and SafeBayes lowers it precisely when this occurs.
- `carmona2020semi`, `jacob2017better` -- **supports, structurally closest.**
  Semi-modular inference tempers one module's likelihood factor by
  gamma in [0, 1] specifically to stop a misspecified module contaminating a
  **shared** parameter.  Our weight is functionally an influence parameter.
  Nothing in this literature motivates gamma > 1.
- `holmes2017assigning`, `miller2019robust` -- **contextualizes.**  Independent
  routes to the same temper-down prescription.
- `wenzel2020good` -- **complicates.**  The cold-posterior effect superficially
  favours tempering up; follow-up work attributes it largely to data curation
  and augmentation artifacts in image classification.  Mention for completeness;
  do not present as support for our 163x.

## E. How in-distribution model selection chose an overconfident model

**Claim.**  Selecting the loss weight on validation NLL is the mechanism that
produced the miscalibration, and it would have done so even without
distribution shift.

**Our evidence.**  Trial 22 won on teacher-forced corpus NLL by ~4x the tie
band while degrading real-rock coverage and the sampled-support regime.

- `mukhoti2020focal` -- **supports, most direct.**  Documents NLL overfitting:
  validation NLL keeps improving past the point where calibration error starts
  degrading, because the loss keeps pushing already-confident correct
  predictions further.  Validation NLL is not a calibration proxy even
  in-distribution.
- `seitzer2022pitfalls` -- **supports, second mechanism to rule in or out.**
  Gaussian NLL implicitly down-weights poorly-fit high-variance points via the
  1/sigma^2 term, producing premature overconfidence with no multi-task
  weighting involved.  Check whether our Dirichlet head has an analogous
  1/alpha_0 implicit weighting; if so, two mechanisms stack.

## F. Post-hoc recalibration is the standard remedy, with a shift caveat

**Claim.**  A mean-preserving, single-scalar rescale of the predicted spread is
the established fix, and its guarantee weakens under domain shift.

**Our evidence.**  Coverage deficit is roughly uniform in direction across
groups; the recalibrator's linear tier already outperformed richer tiers.

- `laves2020well` -- **supports, closest recipe.**  "Sigma scaling": one scalar
  multiplying predicted spread, fit post-hoc, mean preserved exactly, framed as
  temperature scaling's regression analogue.  This is the template for an
  alpha-scale correction on a Dirichlet.
- `levi2022evaluating` -- **supports.**  Simple variance scaling matches more
  elaborate methods; also shows a degenerate failure mode in the standard
  regression-calibration definition.
- `kuleshov2018accurate`, `song2019distribution`, `zelikman2020crude`,
  `chung2021beyond` -- **contextualizes.**  The regression-recalibration family
  we are choosing from.
- `ovadia2019trust` -- **complicates, and must be cited.**  Under shift,
  temperature scaling fit on in-distribution validation is significantly
  outperformed and can score *worse* than the uncalibrated model.  Directly
  bears on fitting our recalibrator on one real-rock slice and applying it to
  another.
- `tomani2021posthoc` -- **contextualizes.**  Same mechanism, with a practical
  partial fix (perturbing the calibration set before fitting).

## G. Conformal prediction is not a free lunch here

**Claim.**  Distribution-free coverage guarantees do not survive our
simulator-to-real shift.

- `angelopoulos2021gentle` -- **contextualizes.**  The guarantee holds under
  exchangeability of calibration and test data.
- `tibshirani2019conformal` -- **complicates.**  Extends to covariate shift only
  with a known or estimable likelihood ratio, which we do not have cheaply.
- `barber2023conformal` -- **complicates.**  For non-exchangeable data the
  guarantee becomes a *bound* on the coverage gap in terms of a
  total-variation-type distance, not restored coverage.

## H. What is genuinely novel here

- No paper applies a hurdle-plus-Dirichlet structure to mineral assemblages.
  The compositional-zeros literature (`tsagris2018dirichlet`, citing geology as
  a motivating domain; `aitchison1986statistical`) sets it up but does not make
  the connection.  **we-extend.**
- No proper scoring rule purpose-built for Dirichlet or simplex-valued
  forecasts exists; no named method for post-hoc mean-preserving Dirichlet
  concentration rescaling exists.  State as gaps, not as things we found.
- No paper states the discrete-surprisal versus differential-entropy scale
  asymmetry in a hurdle context, and none runs our narrowed-proposal
  experiment.  **we-extend.**
- The rounded-versus-essential zero distinction (`tang2019zero`,
  `connor1969concepts`, `koslovsky2023bayesian`) is the existing formalism for
  our censored trace entries -- adopt rather than invent.

## K. Motivation: mineralogy is the gateway from chemical assay to action

Owner-directed 2026-08-04; carried by abstract sentence 1 and intro P1.
None of these entries are in references.bib yet -- fields get pulled from
primary sources at the verification pass. Scope guard: the system predicts
phase presence and abundances, not per-element host assignment; prose says
"informs separations", never "identifies the host of element X".

**Claim.**  Mineralogy, not bulk chemistry, is what geological
interpretation and industrial extraction act on; it is measured slowly
(point counting, XRD) while whole-rock chemical assays (XRF, ICP) are
rapid and routine, so predicting mineralogy from assay chemistry bridges
the abundant measurement to downstream decisions -- including selective
extraction of critical elements, whose procedures depend on the host
phase and the surrounding mineralogical environment.

- mineralML paper (PDF: `~/Downloads/mineralml_tex_eartharxiv_withsi.pdf`)
  -- **supports** the fundamental-geology half (phase assemblages as the
  basis of petrological investigation; crystals record magmatic
  evolution) and orients the reader on measurement methods; cite its refs
  therein (Putirka 2008, Blundy & Cashman 2008, ...) selectively rather
  than wholesale. PENDING bib fields.
- Minerals 2025, 15(5) 533 -- **supports** the critical-materials
  extraction half. UNVERIFIED: owner-supplied locator only; confirm
  authors/title/scope before citing.
- US DOE Critical Materials Assessment, 2023 -- **contextualizes** supply
  pull. Government report; fields from the primary document.
- IEA, Recycling of Critical Minerals, 2024 -- **contextualizes**. NOTE:
  the body is the International Energy AGENCY (owner note said
  Association); fix at bib entry.
- Nature Reviews Methods Primers (2023) 3:52 -- **supports** the assay
  orientation (ICP methods). UNVERIFIED: confirm the primer's identity
  and scope from the primary source.

## L. Encoder design rationale: decisions borrowed from AI for science

Owner-directed 2026-08-04; carried by intro P4 (the "rationale
hypothesis" framing is the owner's and should survive editing). Neither
entry is in references.bib yet.

**Claim.**  The encoder is our own, with two borrowed decisions: CrabNet's
core logic (element embeddings summed with a fractional-abundance
encoding, standing where position embeddings stand in language models)
without its surrounding machinery; and the choice to pretrain on exactly
the feature space users measure (routine XRF/ICP assay elements), which
has a physical motivation (cheap, high-throughput, commonly measured) and
a theoretical one (the pretrained feature space determines transfer
capability, so one representation can serve both the simulator-labeled
corpus and real-rock transfer).

- CrabNet (Wang et al., npj Computational Materials 2021; PDF on disk:
  `s41524-021-00545-1-4.pdf`) -- **supports** the adapted core logic.
  Companion cite needed wherever mat2vec is named: Tshitoyan et al. 2019
  (Nature), the embedding's source. PENDING bib entries.
- "Features are fate: a theory of transfer learning in high-dimensional
  regression" (Rotskoff group, ICML 2025 per owner) -- **supports** the
  rationale hypothesis. The owner-supplied thesis quote ("the feature
  space of the pretrained model determines the capability of the model to
  transfer to downstream tasks") must be verified verbatim against the
  paper; authors/venue unconfirmed on disk. PENDING bib fields.

## I. Do not cite / do not confuse

- **Bickel (2007)** is *not* about log-score outlier robustness (it concerns
  rank-order properties and nonlinear-utility elicitation).  Cite
  `selten1998axiomatic` for the outlier-sensitivity claim.
- **`kull2019beyond` "Dirichlet calibration" is a different thing.**  There the
  Dirichlet is the calibration *map* applied to a classifier's probability
  vector; it does not rescale a predicted Dirichlet's concentration.  Not our
  method.  Cite only to disambiguate.
- **`bengs2022pitfalls` / `bengs2023second` do not apply to us.**  Their
  impossibility results target *second-order* Dirichlets (distributions over
  class probabilities, as in evidential learning, `sensoy2018evidential`).  Ours
  is first-order -- the composition is the observable.  Close enough in
  vocabulary that a reviewer may raise it, so preempt in the text.
- **Kendall & Gal weighting is contested for our case.**  The ecology review
  offered `kendall2018multi` as the principled fallback for a shared-trunk
  hurdle weight; the scoring-rule review advised against it, since both our
  terms are already NLLs of the same observation whereas the derivation assumes
  each task carries its own free observation-noise scale.  Unresolved; do not
  adopt silently either way.
- **Whether specialized multi-task optimizers help at all is contested.**
  `kurin2022defense` and `xin2022current` both find fairly-tuned scalarization
  competitive, and disagree with each other on the mechanism.  Do not present
  any of them as a settled fix.
- **`cannon2022investigating`**: no mitigation strategy for simulation-based
  inference misspecification worked in all cases.  Whatever we adopt must not
  be framed as solving a solved problem.

## J. UNVERIFIED -- confirm or drop before submission

- A 2025 *IJCV* critique of uncertainty weighting proposing a softmax-normalized
  alternative (UW-SO).  Reported by an agent; sits past reliable knowledge.
- Mimori et al. (2020/2021) alpha-calibration -- reported as the closest
  Dirichlet-specific mean-preserving concentration rescale.  Would be the single
  most useful citation for section F if real; must be confirmed.
- Amaral et al. (2025/2026), conformal prediction for compositional data.
  Reported from abstract only, with an unreliable PDF parse.
- `cannon2022investigating`, `holmes2017assigning`, `miller2019robust`,
  `jacob2017better`: venue and year plausible but unconfirmed.
- Gulrajani & Lopez-Paz, "In Search of Lost Domain Generalization" (ICLR 2021)
  was suggested for OOD model-selection pitfalls, but the attribution to our
  exact formulation was not confirmed.  Omitted from the .bib pending a check.
