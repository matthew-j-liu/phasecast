# The supervised heads: inverse mineralogy + conditions posterior

The supervised heads read the frozen encoder latent `h` and emit a
**calibrated, multimodal posterior** over (mineral assemblage, modal
abundances, per-phase composition, formation conditions), where every posterior
sample is internally **mass-balanced** to the input bulk. For the value
proposition, the inverse-vs-forward framing, the four-axis differentiation, and
the encoder/tokenization that produces `h`, see `encoder.md`; this doc owns the
heads.

Input is the 22-token cation+trace vector (no T/P/H2O -- those are CondHead
targets; no SSL masking at inference). Output is a joint distribution the user
reads two ways: a flattened marginal table and a ranked set of complete
scenarios (both defined below).

## The transfer-learning lens

The lens for every corpus and head decision. We can simulate abundant,
high-quality, physics-rooted labels at ONE point in the magmatic chain: the
equilibrium crystal cargo a stored melt grows at sampled (P, H2O). The task we
actually want sits further DOWN that chain -- the mineralogy a geologist
measures on the erupted rock -- and its labels are scarce to nonexistent (no
public bulk + modal-mineralogy compilation exists). So we train earlier in the
chain, where labels are simulable, and hypothesize the heads still predict on
real rocks. That hypothesis is the central bet, and it is measured, not
assumed: validate against a small real set (the eruption benchmark:
point-counted phenocryst modes + bulk chemistry from well-studied eruptions),
kept validation-only -- fine-tuning on it would spend the only measurement of
the gap we have.

Two transfer layers with different risk profiles:

- **Encoder: classic pretrain transfer.** SSL on real rock chemistry (gard), so
  a user's input is in-distribution; nothing synthetic touches the
  representation.
- **Heads: sim-to-real / weak supervision.** Inputs are real chemistry in both
  train and test; only the labels come from the proxy process. The whole
  sim-to-real gap is confined to label semantics -- how well the reference
  assemblage tracks the real rock's phenocryst cargo.

Consequences to carry:

- **Mineralogy is the transfer target**; conditions are the latent mechanism
  that generates the label variance (CondHead a bonus rider, never a design
  driver). Score design choices by mineralogy value.
- The label proxy is cheap to revise (frozen encoder; heads retrain in days),
  so v1 ships a defensible proxy, not a perfect one.
- Upgrade path if a large real modal set ever materializes: fine-tune the
  synthetic-trained heads on real modes; the architecture is already shaped
  for it.

## Architecture

A frozen SSL encoder produces a pooled latent `h`; parallel heads read `h`.
All heads are probabilistic (NLL, never MSE), so the output is a distribution
with honest uncertainty rather than a point estimate.

- **ModeHead (hurdle / zero-inflated)** -- modal abundances over the 21-phase
  solid vocabulary (below).
- **CompHead** -- per-phase oxide-wt% composition for the solid phases.
- **CondHead** -- formation conditions (T, P, H2O), conditioned on the
  assemblage support (below).
- **G_sys head** -- auxiliary scalar (system Gibbs energy); staged in after
  CompHead.
- **MeltHead** -- deferred; would predict silicate-melt composition and make
  mass-balance closure exact at inference.

The encoder is the durable contribution; the heads are application slots
re-supervisable against any label source.

## The label and its vocabulary

The primary label is the **crystals-renormalized solid assemblage**: liq and
fl are dropped from each corpus row and the solids renormalized to sum to 1,
matching the granularity of real point-count data (itemized phenocrysts, one
lumped groundmass bucket). Renormalization and the 0.1 wt% presence cutoff
are training-time rebuilds from raw stored modes, never materialized.

The vocabulary is **21 solid phases** (`cnmineral.data.phase_vocab`), derived
from the MAGEMin corpus phase names by three rules (locked 2026-07-15 on a
full-corpus coverage analysis):

- **Feldspar split.** MAGEMin's single `fsp` solution model emits plagioclase
  and alkali feldspar as anonymous instances (both coexist below the solvus
  in 13-19% of feldspar-bearing draws). Instances are routed to `plag` /
  `afs` by the vendored mineralML stoichiometric feldspar rule
  (`cnmineral.data.feldspar_rule`: molar An/Or against the DHZ two-feldspar
  boundary curves; inside-the-gap ternary compositions route by dominant
  component). The split recovers the A/P ratio that defines the QAPF
  classification and keeps CompHead targets off the miscibility gap.
- **SiO2 polymorphs merged.** `trd`/`crst` fold into `q`: identical
  chemistry (unlearnable from a chemistry-only input), ~2 in 10,000 draws,
  but carrying ~half the crystal cargo when present -- merging keeps that
  mass in the silica bucket where dropping would corrupt those rows'
  renormalization. The Al2SiO5 polymorphs (`and`/`ky`/`sill`) stay separate:
  each is comfortably learnable and the polymorph identity encodes pressure,
  which the support-conditioned CondHead exploits.
- **Learnability floor.** Phases under 0.1% presence at both training rungs
  (`chl`, `fper`) are dropped; a presence head trained on tens of positives
  in ~200k rows learns "never" and the magnitude slot never trains. `cor`
  sits just above the floor and is kept.

All other multi-instance phases (cpx/mu/spl, each <1% of their draws) sum
instance modes; when CompHead lands, same-token instances get the
mass-weighted mean composition (exactly the closure-consistent choice).

## ModeHead: the hurdle parameterization

A single Dirichlet over the full 21-phase simplex cannot represent sparse
assemblages. Two reasons, the second deeper than the first:

1. Its mean `alpha_i / sum(alpha)` is strictly positive for every phase, so
   the point estimate can never be a hard zero -- it smears ~1 wt% onto each
   truly-absent phase and buries real sub-1% phases.
2. (The load-bearing reason.) The Dirichlet shares one concentration
   `alpha_0`. Representing "absent phases confidently near zero" needs small
   `alpha_i` and large `alpha_0`; honest spread on the present phases needs
   small `alpha_0`. These fight through the normalizer, so the head spends
   capacity suppressing absent mass instead of sizing real mass, and the
   present-phase alphas get distorted.

The fix factorizes the likelihood `P(modal) = P(present) * P(modal | present)`
into two heads trained on separate clean targets, never multiplied:

- **Presence head** -- per-phase Bernoulli `p_i = P(phase i present | bulk)`,
  trained by BCE against `modal > 0`. The natural target is the *within-bulk
  presence frequency* over the FC paths, which is a genuine marginal of the
  multimodal posterior (a phase present on 40% of plausible paths is real
  signal, not a coin flip to round off).
- **Magnitude head** -- a Dirichlet evaluated **only on the phases the label
  marks present**, against the present-only renormalized truth. Absent phases
  are masked out of the magnitude loss entirely, so the head only ever learns
  conditional abundances over the real support and its alphas are undistorted.
  Masking is principled, not a hack: a Dirichlet sub-composition is itself
  Dirichlet on the retained parameters (Aitchison), so `Dir(alpha_S)` against
  the renormalized present-only truth is a proper density. (Equivalently: the
  magnitude head is simply *defined* on the present support; absent phases are
  owned by the Bernoullis.)

Training teacher-forces presence from the ground-truth mask -- the magnitude
loss and the mass-balance term both use the true present set, staying
differentiable in M and C with no fragile `p_i * m_i` product. The prediction
is the magnitude head's mean over a support, not a product of two guesses.

### The output is the joint distribution, not a hard set

`p_i` is part of the distributional output, and for the path-sensitive
phases (amp, opx, ne) the presence frequency lands intermediate (~0.3-0.7) --
that is exactly the multimodal signal, and the multimodal posterior is the
contribution. So the model keeps the **joint object** (presence Bernoullis x
conditional Dirichlet); a calibrated modal posterior is obtained by
**sampling supports** (draw presence from the Bernoullis, abundances from the
conditional Dirichlet) -- an ensemble over supports. Hard-thresholding to a
single vector is one *presentation view* rendered on demand, structurally the
same kind of collapse as renormalizing away liq/fl for a hand-sample view. It
is a view, not the model.

## CompHead: per-mineral composition

A per-phase Dirichlet over **oxide wt%** (`Comp_wt`, the 11 MAGEMin oxides;
no MSE). The endmember-vector target was rejected on measurement -- MAGEMin
endmember fractions on real volcanic bulks are signed affine combinations
that leave the positive simplex (amphibole per-mineral endmember-sum median
2.3, worst component -110%) -- so oxide wt% is the target for all phases,
with endmember bonus heads on clean phases (fsp/ilm/ol/opx) loss-masked per
instance. The Dirichlet (over a heteroscedastic per-oxide Gaussian, locked
2026-07-15) buys valid compositions by construction: the mass-balance term
and scenario sampling consume composition vectors directly, and an
independent-Gaussian sample violating the sum is a parameterization artifact
carrying no diagnostic value. Costs accepted: hard-zero oxides are
eps-clipped and renormalized, and one shared concentration per phase blurs
per-oxide error bars toward each other -- small in absolute terms because
composition spreads are tight (below). Targets with two same-token instances
use the mass-weighted mean oxide composition; the feldspar split at the
vocabulary level keeps the plag/afs solvus pair out of that averaging.
Conventional scalars (An#, Mg#) are projections of the oxide vector via
stoichiometry, derived not separately predicted.

Composition is **bulk-controlled**: it reads `h` only and is largely
scenario-independent (empirically, per-phase composition CV is 10-30x tighter
than modal CV -- composition is set by the bulk, modes by the path). This is
why CompHead is not conditioned on the sampled conditions. Endmember-basis
dimensionality varies by phase (feldspar ~3, clinopyroxene ~6-8, amphibole
~10-12); the high-dimensional bases have noisier label extraction, validated
per phase in the corpus pre-regen checks.

## CondHead, G_sys, MeltHead

- **CondHead** -- a GMM per axis for (T, P, H2O); the inverse-inference
  posterior, often multimodal (the deep-wet vs shallow-dry clusters). Linear
  space, no log transform of T/P/H2O. **Conditioned on the assemblage
  support** (locked 2026-07-15): the scenario view needs assemblage and
  conditions coupled in joint samples, and independent heads assume a
  conditional independence the corpus violates by design (the within-bulk
  label variance is driven by the shared hidden (P, H2O) path -- marginals
  don't remember which condition mode went with which support). The chain
  rule fixes it without assumptions: p(support | h) from the presence
  Bernoullis, then p(T, P, H2O | h, support) with the presence vector fed to
  CondHead alongside `h` -- teacher-forced true mask at train,
  ModeHead-sampled at inference, so every ancestral sample is a coherent
  scenario. Known cost: mild exposure bias (inference-time supports are
  predictions, not labels); mitigations if it measures as real are mask
  noising at train or restricting scenario supports to corpus-observed
  combinations. T/P/H2O remain supervision targets, never encoder inputs.
- **G_sys head** -- single-Gaussian NLL, auxiliary; an encoder regularizer
  giving a second gradient, not a polymorph disambiguator. Added after
  CompHead so its marginal gradient value is measured, not assumed.
- **MeltHead (deferred)** -- a Dirichlet over the ~11 melt oxides. Adds
  inference-time mass-balance closure and a residual-melt-composition output.
  Built only if those capabilities are wanted; see its own issue.

## Mass-balance closure

`B = (1 - F_liq - F_fl) * sum_solid m * C(predicted)
     + F_liq * C_liq(corpus) + F_fl * C_fl(corpus)`

Predicted: the crystals-renormalized solid modes `m` (ModeHead) and solid
oxide compositions `C` (CompHead). Teacher-forced from the corpus row: the
melt/fluid fractions `F_liq`, `F_fl` and their oxide compositions, stored per
rung row for exactly this (ModeHead no longer predicts liq/fl -- at a fixed
melt-fraction endpoint the melt fraction is a constant with nothing to
learn). Pinning the melt side keeps the term a genuine constraint on the
predicted solids instead of a free error-sink, and avoids the
ill-conditioned divide-by-small-M_liq of solving for the melt. This is a
training-time constraint; closure holds per posterior sample, not across the
flattened marginal table. **Open rethink:** the inference-time
closure-residual-as-OOD-signal degrades now that the label melt fraction is
teacher-forced -- at inference there is no corpus row to pin F_liq/C_liq, so
the residual needs either a MeltHead or a different confidence signal.

## Outputs

### View 1 -- flattened marginal

Per phase: presence probability + conditional abundance (`mean +/- wt%`, read
as "if present"). This view does **not** sum to 100, by design: each row is
marginalized over a different support. It maps one-to-one onto the two
ModeHead heads (presence column / magnitude column).

### View 2 -- top-K scenarios (the coherent answer)

Closure holds only within a fixed support, so the faithful answer is the
ranked set of scenarios: draw a support from the presence Bernoullis,
conditions from the support-conditioned CondHead, abundances from the
conditional Dirichlet; cluster samples by support / condition-mode, rank
clusters by posterior weight, report the dominant few plus a tail line so
nothing is silently dropped. Modal numbers are crystals-renormalized (solids
sum to 100; groundmass/melt is not part of the label) and T is a derived
endpoint, not a sampled axis. Worked example, an andesite:

```
Scenario A -- deep & wet (amphibole-bearing) . posterior weight 45%
conditions   P 1.0-1.4 GPa   H2O 5-7 wt%   T_end 920-970 C
assemblage (crystals-renormalized wt%, sums to 100):
  plagioclase   40 +/- 4     An42 +/- 4
  clinopyroxene 23 +/- 3     Mg# 75
  amphibole     19 +/- 4     Mg-hornblende
  magnetite      9 +/- 2
  biotite        6 +/- 2     (hydrous phase, consistent with high H2O)
  ilmenite       3 +/- 1
mass balance:  (1-F_melt) * sum(mode*comp) + F_melt * melt = input bulk   ok

Scenario B -- shallow & dry (orthopyroxene-bearing, no amphibole) . weight 35%
conditions   P 0.3-0.6 GPa   H2O 1.5-3 wt%   T_end 960-1000 C
assemblage:
  plagioclase   48 +/- 4     An48 +/- 4   (more calcic, drier melt)
  clinopyroxene 22 +/- 3     Mg# 72
  orthopyroxene 16 +/- 3     Mg# 70       (replaces amphibole at low H2O)
  magnetite     10 +/- 2
  ilmenite       4 +/- 1
mass balance:  ok

Scenario C -- intermediate (amph + opx coexisting) . weight 12%
conditions   P 0.6-0.9 GPa   H2O 3-4.5 wt%   T_end 940-980 C
assemblage:
  plagioclase 45   cpx 23   amphibole 11   orthopyroxene 9
  magnetite 8   ilmenite 4        (sums to 100)
mass balance:  ok

Scenario weights:  A 45% . B 35% . C 12% . tail 8%   (sum 100%)
```

**How to read this (one correction baked in).** What varies across scenarios
is **presence and modes** (path-controlled) -- amphibole in A, orthopyroxene
in B -- and the **conditions** that drive them. **Composition is
bulk-controlled** and barely moves across scenarios: the An42 -> An48 drift
shown is illustrative and sits near the upper end of what is realistic; in
practice plagioclase is closer to a single An44 +/- 4 band in both scenarios,
inside the +/- spread. So read the scenarios as "which phases and how much,
under what conditions," and read composition as a per-phase endmember vector
that is essentially the same across scenarios.

The marginal table (View 1) is the projection of these scenarios; you can
flatten scenarios to marginals but cannot reconstruct coherent scenarios from
marginals -- the scenarios are the primary object.

Each scenario is a falsifiable hypothesis with a test: a geologist who finds
amphibole in thin section has selected Scenario A and, with it, its formation
conditions. That is the inverse-inference payoff.

## Evaluation philosophy

Per ADR 006, the objective is a proper scoring rule, not a threshold-dependent
metric:

- **Objective:** the joint hurdle NLL, assembled explicitly at eval (the
  training loss is a sum of two surrogates -- BCE + masked-Dirichlet-NLL --
  so compute the joint likelihood of the modal vector directly, do not assume
  training loss equals it), plus calibration of `p_i` against empirical
  presence frequency `f_i` and CondHead coverage. Report **stratified by phase
  rarity**, so rare-phase failure cannot hide inside an aggregate dominated by
  common phases.
- **Guardrails (not the headline):** a **rare-phase presence recall floor**
  (burying ne/mu/amp is the original sin), and an **FP-mass ceiling**
  (measure false-positive *mass*, not count -- a phantom at 0.3 wt% is noise,
  at 15 wt% it corrupts closure and forces the magnitude head to size a phase
  it never learned in that support).
- **Credibility anchor:** the held-out **real-rock eval** (mineralML grains,
  EDS maps), never seen in training. Showing the right scenario carries
  meaningful weight on rocks whose true assemblage is known is the test that
  this is a calibrated inverse model, not amortized MAGEMin.
- **Checkpoint selection and HPO ranking (locked 2026-07-19):** each head is
  snapshotted at its own best raw val epoch (mode on the unweighted
  assemblage NLL `bce + Dirichlet`, comp/cond on their own NLLs) -- a single
  weighted-total epoch let the largest term decide and starved cond.  The
  joint HPO ranks trials on mode's raw assemblage NLL at mode's best epoch,
  ties within the val-roster wobble band broken by comp then cond; no
  weighted aggregate anywhere in selection, since the weights are the
  search variables and any aggregate re-imports an arbitrary exchange rate
  between heads.  Cross-head consistency of mixed-epoch heads is measured
  by the predicted-modes closure residual, which doubles as the OOD
  diagnostic.

## How this design was arrived at (durable summary of the review)

The hurdle direction and several framings were sharpened by review. Recorded
so the reasoning is not re-litigated:

- The kill-shot against a single Dirichlet is the *training-coupling*
  argument (shared `alpha_0`), not just the positive-mean argument.
- Masked present-only Dirichlet-NLL is licensed by the Dirichlet
  sub-composition property; it is a proper density, not an approximation.
- Presence is *not* hard-collapsed in the model. `p_i` is distributional
  output; the joint object is sampled; the hard vector is a presentation view.
- Proper-scoring (joint NLL + calibration, stratified by rarity) is the
  objective; rare-phase recall and FP-mass are guardrails. The
  distributional-eval principle lives in ADR 006, not in a separate ADR
  (ADR 007 exists but covers the SSL pretrain objective, a different topic).

## Open items

- **Plain-Dirichlet baseline not yet measured.** The hurdle ModeHead was
  built directly (locked at the Track B start rather than gated on a smear
  measurement); a plain full-simplex Dirichlet remains the natural ablation
  baseline when the eval framework lands. Sparsemax stays rejected
  (conflicts with the probabilistic-head mandate).
- **MeltHead** deferred (inference-time closure + melt output).
- **Public-facing renormalization** (drop liq/fl for the hand-sample view) is
  a presentation-layer concern, volcanic-vs-plutonic aware.
- Composition is bulk-controlled, so CompHead is not condition-conditioned;
  revisit only if a real need for scenario-specific composition appears.
