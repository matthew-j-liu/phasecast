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

## Architecture

A frozen SSL encoder produces a pooled latent `h`; parallel heads read `h`.
All heads are probabilistic (NLL, never MSE), so the output is a distribution
with honest uncertainty rather than a point estimate.

- **ModeHead (hurdle / zero-inflated)** -- modal abundances over ~18 phases
  (including liq, fl).
- **CompHead** -- per-mineral composition for the ~16 minerals.
- **CondHead** -- formation conditions (T, P, H2O).
- **G_sys head** -- auxiliary scalar (system Gibbs energy); staged in after
  CompHead.
- **MeltHead** -- deferred; would predict silicate-melt composition and make
  mass-balance closure exact at inference.

The encoder is the durable contribution; the heads are application slots
re-supervisable against any label source.

## ModeHead: the hurdle parameterization

A single Dirichlet over the full 18-phase simplex cannot represent sparse
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

`p_i` is part of the distributional output, and for the path-sensitive rare
phases (amp, fl, ne) the presence frequency lands intermediate (~0.3-0.7) --
that is exactly the multimodal signal, and the multimodal posterior is the
contribution. So the model keeps the **joint object** (presence Bernoullis x
conditional Dirichlet); a calibrated modal posterior is obtained by
**sampling supports** (draw presence from the Bernoullis, abundances from the
conditional Dirichlet) -- an ensemble over supports. Hard-thresholding to a
single vector is one *presentation view* rendered on demand, structurally the
same kind of collapse as renormalizing away liq/fl for a hand-sample view. It
is a view, not the model.

## CompHead: per-mineral composition

A per-phase Dirichlet over **endmember mole fractions** (no MSE; melt/fluid
have no endmembers and are excluded). Endmember fractions x the fixed
Green2025 stoichiometry matrix -> oxide wt% per phase, the form the
mass-balance closure consumes. Conventional scalars (An#, Mg#, jadeite
component) are projections of the endmember vector, derived not separately
predicted.

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
  space, no log transform of T/P/H2O.
- **G_sys head** -- single-Gaussian NLL, auxiliary; an encoder regularizer
  giving a second gradient, not a polymorph disambiguator. Added after
  CompHead so its marginal gradient value is measured, not assumed.
- **MeltHead (deferred)** -- a Dirichlet over the ~11 melt oxides. Adds
  inference-time mass-balance closure and a residual-melt-composition output.
  Built only if those capabilities are wanted; see its own issue.

## Mass-balance closure

`B = sum_solid M * C(predicted) + M_liq * C_liq(corpus) + M_fl * C_fl(corpus)`

Predicted: all modal abundances M (ModeHead, including M_liq, M_fl); solid
compositions C (CompHead via the Green matrix). Given: the bulk B (input); the
liq/fl oxide compositions (corpus labels, not predicted -- pinning them keeps
the term a genuine constraint on the predicted solids instead of a free
error-sink, and avoids the ill-conditioned divide-by-small-M_liq of solving
for the melt). This is a training-time constraint; at inference the
**residual is itself a confidence / OOD signal** (a large mismatch means the
heads disagree). Closure holds per posterior sample, not across the flattened
marginal table.

## Outputs

### View 1 -- flattened marginal

Per phase: presence probability + conditional abundance (`mean +/- wt%`, read
as "if present"). This view does **not** sum to 100, by design: each row is
marginalized over a different support. It maps one-to-one onto the two
ModeHead heads (presence column / magnitude column).

### View 2 -- top-K scenarios (the coherent answer)

Closure holds only within a fixed support, so the faithful answer is the
ranked set of scenarios. Draw K posterior samples, cluster by support /
condition-mode, rank clusters by posterior weight, report the dominant few
plus a tail line so nothing is silently dropped. Worked example, an andesite:

```
Scenario A -- deep & wet (amphibole-bearing) . posterior weight 45%
conditions   P 1.0-1.4 GPa   H2O 5-7 wt%   T 920-970 C
assemblage (modal wt%, sums to 100):
  plagioclase   38 +/- 4     An42 +/- 4
  clinopyroxene 22 +/- 3     Mg# 75
  amphibole     18 +/- 4     Mg-hornblende
  magnetite      8 +/- 2
  biotite        6 +/- 2     (hydrous phase, consistent with high H2O)
  ilmenite       3 +/- 1
  residual melt  5 +/- 2
mass balance:  sum(mode*comp) = input bulk   ok (within 1 wt%/oxide)

Scenario B -- shallow & dry (orthopyroxene-bearing, no amphibole) . weight 35%
conditions   P 0.3-0.6 GPa   H2O 1.5-3 wt%   T 960-1000 C
assemblage:
  plagioclase   45 +/- 4     An48 +/- 4   (more calcic, drier melt)
  clinopyroxene 20 +/- 3     Mg# 72
  orthopyroxene 15 +/- 3     Mg# 70       (replaces amphibole at low H2O)
  magnetite      9 +/- 2
  ilmenite       4 +/- 1
  residual melt  7 +/- 2
mass balance:  ok

Scenario C -- intermediate (amph + opx coexisting) . weight 12%
conditions   P 0.6-0.9 GPa   H2O 3-4.5 wt%   T 940-980 C
assemblage:
  plagioclase 41   cpx 21   amphibole 10   orthopyroxene 8
  magnetite 8   ilmenite 4   residual melt 8        (sums to 100)
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

- **ModeHead sparsity build is gated on a smear measurement** -- fit a plain
  Dirichlet ModeHead on the existing FC corpus, quantify misallocated mass and
  buried sub-1% phases, and size the hurdle effort to the measured problem.
  Post-hoc thresholding is a baseline to beat; sparsemax is rejected (conflicts
  with the probabilistic-head mandate).
- **MeltHead** deferred (inference-time closure + melt output).
- **Public-facing renormalization** (drop liq/fl for the hand-sample view) is
  a presentation-layer concern, volcanic-vs-plutonic aware.
- Composition is bulk-controlled, so CompHead is not condition-conditioned;
  revisit only if a real need for scenario-specific composition appears.
