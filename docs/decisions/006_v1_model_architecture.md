# Decision 006: v1 model architecture — shared encoder + Dirichlet ModeHead + GMM CondHead + joint Optuna HPO

**Date:** 2026-05-19
**Status:** Active

## Decision

The v1 model is a **single shared encoder feeding two probabilistic heads**:

```
Input: 11 oxide wt% (bulk only — no T, P, or H2O at inference)
   ↓
Encoder (SubCrab wrapper) → pooled d_model latent
   ├─→ ModeHead: Dirichlet(α) over n_minerals modal vector
   └─→ CondHead: 3 × K_gmm GMM over (T, P, H2O) conditions
```

Trained with joint NLL loss: `L = 0.5·NLL_dirichlet + 0.5·NLL_gmm` (after
z-scoring the GMM contribution for scale parity). K=5 deep ensemble for
σ_epistemic. Hyperparameters chosen via joint single-stage Optuna TPE search.

This is the **inverse-inference architecture**. From bulk chemistry alone the
model predicts the joint posterior `P(T, P, H2O, modal | bulk)` — the
deployed model never sees (T, P, H2O) at inference, but the training corpus
supplies them as labels.

## Why this decision

### Project differentiation argument (from 2026-05-19 journal entry)

The two existing tools live on different sides of the problem:
- **MAGEMin**: forward map (bulk + conditions → assemblage). Needs conditions
  the geologist doesn't have.
- **mineralML**: per-grain classification from spot analyses. Wrong granularity
  for whole-rock analysis.

Geologist's workflow today: run MAGEMin over a (T, P, H2O) sweep, stare at an
unweighted range of possible mineralogies. Our model **collapses that range
into the conditioned posterior** by learning, from a corpus of realistic
(bulk, conditions, assemblage) triples, which conditions are likely for a
given bulk.

A pure modal-prediction model (deterministic OR probabilistic) is Position B —
"MAGEMin emulator marginalized over T, P" — which was explicitly rejected at
project inception. Our architecture must do something MAGEMin can't, and
**condition inference is exactly that.**

### Per-component rationale

**Shared encoder.** Multiple heads benefit from a common bulk representation.
This also sets up future CompHead (mineralML compositional fusion) and
GsysHead (thermodynamic regularizer) without further architecture work.

**Dirichlet ModeHead.** Mineral wt% labels sum to exactly 1 (MAGEMin output
is normalized). Dirichlet is the natural distribution over the simplex,
parameterized by `α` (concentration parameters). Provides:
- Sum-to-1 enforced by construction
- Aleatoric uncertainty per phase (Dirichlet variance)
- Multimodal-friendly via ensemble aggregation of K=5 `α` vectors
- Label smoothing ε=1e-3 handles exact zeros in sparse targets

**GMM CondHead.** Conditions (T, P, H2O) are continuous, and the corpus has
many paths per bulk → the marginal posterior over conditions for a given
bulk is genuinely multimodal (a basalt could plausibly be either shallow
volcanic OR deeper plutonic). A unimodal Gaussian would average over those
modes and miss the multimodality. GMM with K_gmm components captures it.
Linear-space (no log transform) — predictions outside the physical range
are diagnostic information, not problems to hide.

**Joint single-stage HPO.** A two-stage HPO (e.g., tune model size first,
loss weights second) assumes separability that doesn't hold — `d_model` and
`λ_cond` interact (a bigger encoder can carry condition signal at lower λ;
a smaller encoder needs higher λ). Joint search over all dims with TPE
sampler.

**NLL objective (not R²).** NLL is a proper scoring rule that automatically
yields calibrated uncertainty estimates. R² and calibration coverage are
reported *separately* in the evaluation, but they don't drive the
optimization. (Optimizing R² alone would reward overconfidence; optimizing
NLL doesn't.)

## Alternatives considered

### Alt 1: Deterministic centroid head (rejected — Position B)

A single deterministic head predicting the mean modal vector. Marginalizes
over the 8 paths per bulk into one "average" assemblage. Was rejected
because:
- It IS Position B in different clothing. Bayesian-amortization of MAGEMin
  over the rock-class-conditional (T, P, H2O) prior.
- Loses the "calibrated multimodal" differentiation row of the 4-axis table
  in CLAUDE.md.

### Alt 2: Probabilistic ModeHead only (no CondHead) — rejected

Just the Dirichlet head over modal. Captures uncertainty but doesn't predict
conditions. Reviewer's natural rejoinder: "what does the model do that
MAGEMin doesn't?" Without CondHead the answer is just "marginalizes over T,
P, H2O with calibrated uncertainty" — which is amortization, not new
science.

### Alt 3: CondHead only, then pipe to MAGEMin — partially adopted

A model that predicts only (T, P, H2O), then the user calls MAGEMin with
those conditions to get modal. This is a valid simpler architecture and we
should evaluate against it in Stage 3 (two-step pipeline as upper bound for
modal prediction accuracy, since MAGEMin generated the labels).

But the joint architecture has two advantages:
- Single-forward-pass inference (microsecond claim)
- ModeHead can later be trained against non-MAGEMin labels (mineralML
  fusion) to deviate from pure MAGEMin output

Stage 3 will validate whether the ModeHead is materially better than the
two-step baseline. If not, the project pivots toward "CondHead + MAGEMin" as
the deployed product and ModeHead becomes optional.

### Alt 4: ALIGNN structural embeddings as v1 head — rejected, deferred to v2

Per ALIGNN-related decisions from project inception. Marginal value for
modal prediction; only meaningful for polymorph disambiguation in v2.

### Alt 5: DFT-derived ΔG catalog as physics-constraint loss — rejected, deferred

User direction: the "we have better ΔG than HP" argument is debatable on
its own merit (HP is itself partly DFT-informed). Not pursuing in v1.

### Alt 6: mineralML CompHead in v1 — deferred to v1.1

The most natural escape from "MAGEMin surrogate" framing is empirical
per-mineral composition fusion. But it requires real data engineering work
on mineralML's grain database, and the v1 architecture works regardless of
which corpus deviations get added later. So mineralML fusion is v1.1 and v1
ships with MAGEMin-only labels.

### Alt 7: Two-stage HPO — rejected

Fixing some hyperparameters in stage 1 and tuning others in stage 2 misses
interactions. Joint single-stage TPE search is the principled choice even
though it explores a higher-dimensional space.

### Alt 8: MSE on Dirichlet's mean instead of NLL — rejected

Loses the calibration signal. The architecture is probabilistic by design;
the loss should match.

### Alt 9: Log-space transformation for T and P in CondHead — rejected

Adds complexity (transform target on input, exp on output) to enforce
non-negativity that the data already enforces (T, P >> 0). A model predicting
negative T is a diagnostic signal, not a problem to hide via transform.

### Alt 10: Multivariate normal CondHead (correlated T, P) — viable, deferred

(T, P) are anti-correlated for plutonic-vs-volcanic. Independent GMMs per
axis don't model this. A joint MVN-GMM would. But independent is cheaper
and we don't have evidence yet that the correlation matters. Revisit if
Stage 2 results show smearing across the volcanic/plutonic divide.

## Open implementation questions

These don't affect the architectural decision but need to be settled during
implementation:

- **Stratification key for k-fold CV**: `qap_name`, matching v0. 64 distinct
  classes, some very rare — may need `StratifiedShuffleSplit` with
  smallest-class fallback.
- **`λ_cond` z-scoring reference**: against the first trial's NLL_cond value,
  cached in the Optuna study.
- **`K_gmm` discrete search range**: 2, 3, 4. Probably converges to 3.
- **Batch size search range**: must be compatible with MPS memory.
- **Train epoch budget per HPO trial**: 30 (matches upstream CV_EPOCHS).
  Final non-HPO run uses 200+ epochs.

## Revisit when

- Stage 3 evaluation completes. If ModeHead is no better than two-step
  baseline, architecture is overweight — consider stripping back to
  CondHead-only.
- mineralML fusion is built (v1.1). CompHead becomes a third head, joint
  loss adds a fourth term, HPO search space gains another dimension.
- DFT-derived ΔG or G_sys is captured (v2). GsysHead becomes a fourth head.

## Cross-reference

- Project differentiation framing: JOURNAL.md 2026-05-11 "Position C
  crystallized as the design"
- Inverse-inference reframing: JOURNAL.md 2026-05-19 entry
- Corpus generation: JOURNAL.md 2026-05-17 to 05-19 entry, ADR 005
- Upstream Optuna reference:
  - `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn/src/objective.py`
  - `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn/scripts/run_scaling.py`
- Stage-1 implementation will live in:
  - `src/cnmineral/models/{encoder,heads,model,losses,dataset,training,objective}.py`
  - `scripts/{03_train_base.py, 04_hpo.py}`
  - `configs/{base_v1.yaml, hpo_v1.yaml}`
