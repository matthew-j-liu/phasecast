# v1 / v2 model reference (historical)

Living reference for the CrabNet-Mineral v1 model. Explains the architecture
in plain terms; updated in place as the model evolves. For *why* each choice
was made (and what was rejected) see ADR 006; for the chronological build log
see `JOURNAL.md`.

As-built as of the `v1-trained` tag (2026-05-22).

> **Direction forward (2026-06-02):** v1 and v2 are stable reference
> points; the v3 envisioned section near the end of this doc sketches
> the upcoming architecture in the same pipeline / inputs / heads /
> training format used for v1/v2 above. The implementation handoff
> and execution plan live in `docs/FRESH_START.md` and
> `docs/V3_WORKFLOW.md` respectively. Update v3 architecture details
> in this doc as they crystallize, but the v1/v2 sections above are
> historical and should not be edited to describe v3.

## What the model does

One forward pass takes a whole-rock bulk chemistry and returns, jointly:

1. the **mineral assemblage** -- which of 18 phases are present and in what
   modal wt%, as a probability distribution
2. the **formation conditions** -- temperature, pressure, water content, as a
   probability distribution per axis

This is *inverse inference*: bulk chemistry in, formation conditions +
mineralogy out. MAGEMin runs the forward map (conditions -> assemblage) and
needs T/P/H2O as input; this model runs the inverse and needs none of them.

## Pipeline at a glance

```
10 oxide wt%  ->  9 cation wt%  ->  SubCrab encoder  ->  pooled latent
                                                              |
                                              +---------------+---------------+
                                              |                               |
                                         ModeHead                        CondHead
                                       (Dirichlet)                     (GMM x 3 axes)
                                              |                               |
                                   distribution over              distribution over
                                   18-mineral modal vector        (T, P, H2O)
```

The whole thing is trained 5 times (different seeds) into a K=5 ensemble.

## Input

The model takes **10 anhydrous major-oxide wt%**: `sio2, al2o3, cao, mgo,
feo, fe2o3, k2o, na2o, tio2, cr2o3`. Any total is fine -- the input is
renormalized.

Conversion (`dataset.py`, `oxides_to_cations`):

- each oxide is converted to its cation wt% via a fixed molar-mass factor
- `feo` and `fe2o3` are summed into a single `Fe` column
- result: **9 cation wt%** -- Si, Al, Ca, Mg, Fe, K, Na, Ti, Cr
  (`V1_CATIONS_ORDER`)
- divided by 100 so values land in [0, 1], which is what kingcrab's
  `FractionalEncoder` expects (it does *not* want wt% in [0, 100])

Each sample becomes two tensors: `src` (atomic numbers of the 9 cations) and
`frac` (their mass fractions).

**Why no water in the input.** A measured rock analysis includes H2O, and an
earlier version fed it in as an 11th oxide (`h2o_tot`). But in the FC corpus
that column was set equal to the CondHead's H2O *target* (`h2o_path_wt`), so
feeding it in target-leaked the H2O axis -- the model was copying the answer,
not inferring it. v1 drops `h2o_tot`; the model infers H2O from anhydrous
chemistry instead. (Recovering the legitimate part of that signal is v1.1
Option C -- see Known limitations.)

T and P are never inputs either -- they are supervision targets for the
CondHead only.

## Encoder

`encoder.py`. A thin wrapper around `kingcrab.Encoder` (the vendored CrabNet
transformer). It embeds each cation token, runs self-attention over the 9
tokens, then **masked-mean pools** -- averages the per-token outputs over the
real (non-padded) cations -- to produce a single pooled latent vector of size
`d_model`.

That one latent vector feeds both heads. There is no T/P/H2O anywhere in the
encoder; its only input is the 9 cations.

## The two heads

Both heads (`heads.py`) are small residual MLPs that read the shared latent.
Neither emits bare point estimates -- each emits the parameters of a
probability distribution.

### ModeHead -- Dirichlet over the modal vector

Predicts the 18-mineral modal vector (wt% fractions: non-negative, sum to 1).

A **Dirichlet distribution** is the natural distribution over exactly that
kind of vector -- points on the simplex. The head outputs a concentration
vector `alpha` of length 18 (softplus + a small floor keeps it strictly
positive). From `alpha`:

- predictive **mean** = `alpha / alpha.sum()` -- the modal prediction
- predictive **variance** = the aleatoric (irreducible) uncertainty per phase

So instead of "fsp = 45%" the head says "fsp is distributed with mean ~45%
and this much spread."

The 18 phases: `cpx, ol, q, spl, fsp, bi, ilm, sph, fl, ru, opx, ne, mu, ep,
liq, amp, g, cd`.

### CondHead -- Gaussian mixture per condition axis

Predicts the 3 formation conditions (T, P, H2O).

For each axis independently it outputs a **Gaussian Mixture Model**: `K_gmm`
Gaussian components, each with a mean, a standard deviation, and a mixture
weight. A mixture (rather than a single Gaussian) lets the head express a
*multi-modal* answer -- e.g. "this bulk is consistent with formation at
~700 C or ~1000 C" as two separate peaks, instead of being forced to a single
blurred mean between them. This matters because the corpus is many-to-one by
design: one bulk has 8 FC paths at different (P, H2O), so the honest answer
to "what conditions formed this rock" is genuinely a distribution.

Conditions are z-scored during training (`cond_mean`, `cond_std`, saved in
the checkpoint); inference de-normalizes back to physical units (C, kbar,
wt%).

## Uncertainty: the ensemble and the two kinds

The model reports **two distinct uncertainties**:

- **Aleatoric** -- inherent ambiguity in the data. The same bulk really does
  map to a range of assemblages and conditions (the 8 FC paths). Captured
  *within* a single model by the Dirichlet variance and the GMM spread.
  Irreducible -- more data will not shrink it.
- **Epistemic** -- the model's own uncertainty about its weights. Captured by
  the **K=5 deep ensemble**: train the same architecture 5 times with
  different random seeds (different weight init, different shuffle order);
  each lands in a slightly different solution. Where the 5 members *disagree*,
  the model is unsure for reasons of training randomness. Reducible with more
  data or better training.

Ensemble aggregation at inference (`06_eval_ensemble.py`): average the 5
Dirichlet `alpha` vectors; member-to-member spread of the modal means is the
epistemic sigma. GMM components from all 5 members are pooled into one
mixture per axis.

## Training

**Loss -- joint NLL** (`losses.py`). NLL (negative log-likelihood) asks "how
probable is the true label under the distribution the model predicted" and
minimizes the negative log of that. With two heads the loss is a weighted sum:

```
joint_loss = 0.5 * NLL_dirichlet(modal)  +  0.5 * lambda_cond * NLL_gmm(T,P,H2O)
```

Both heads train together against this one objective. `lambda_cond` balances
the two terms and is an HPO knob. The Dirichlet target is label-smoothed by a
small `eps` so exact-zero phases do not send `log(0)` to infinity.

**Data splits** (`training.py`, `make_loaders_v1_bulk_split`). Bulk-level,
not row-level: each bulk's 8 FC paths share almost identical inputs, so all 8
must go to the *same* split or the model memorizes across train/val. The
split is **64 / 16 / 20** train / val / test, stratified by `qap_name` (rock
class), at the bulk level. Classes with fewer than 25 bulks are dropped from
the corpus so per-class test metrics stay interpretable. The exact split is
frozen in `data/splits_v1.json`.

**HPO** (`objective.py`, `04_hpo.py`). Optuna TPE search over architecture +
loss + GMM dims. Each trial runs 3-fold **bulk-level** CV (`StratifiedGroupKFold`
so no bulk spans two folds) and is scored by mean joint NLL. Storage is a
filesystem JournalStorage (resumable).

The v1 winning configuration (HPO trial #32):

| knob | value |
|---|---|
| d_model | 512 |
| N (transformer layers) | 4 |
| heads (attention) | 16 |
| K_gmm (mixture components/axis) | 3 |
| lr | 1.25e-4 |
| weight_decay | 1.0e-5 |
| batch_size | 512 |
| lambda_cond | 3.44 |
| dropout | 0.13 |
| eps_smooth | 1.0e-4 |

## Inference contract

**Input:** 10 anhydrous oxide wt% (any total -- renormalized). No T, P, H2O,
no h2o_tot.

**Output:**
- a Dirichlet over the 18-mineral modal vector -- per-phase mean wt% +
  aleatoric spread + epistemic spread (from the ensemble)
- a GMM over each of T, P, H2O -- mean, component modes, component weights

A worked example lives in the "simulated user inference" section of
`notebooks/v1_eval.ipynb`.

## As-built performance (v1-trained tag)

K=5 ensemble, held-out **test** set (4,239 bulks / 30,212 path-rows):

| metric | value |
|---|---|
| T MAE | 15.4 C |
| P MAE | 1.76 kbar |
| H2O MAE | 0.535 wt% |
| joint NLL | -43.4 |
| 1-sigma coverage (T / P / H2O) | 0.88 / 0.64 / 0.76 |
| phases with R^2 > 0.5 | 7 / 18 |

Anhydrous rock-forming majors are strong: fsp R^2 0.93, q 0.97, cpx 0.91,
ol 0.97, opx 0.75.

## v2 update (2026-05-29): SSL-pretrained encoder, corrected single-seed ablation

(The earlier 2026-05-27 v2 entry was retracted -- the SSL pretrain hit a
kingcrab padding-mask collision bug and the encoder weights never updated.
The corrected entry below is the real result. Full bug analysis in JOURNAL
2026-05-29.)

A four-cell ablation (`scripts/10_ablation_v2_eval.py`) on the same frozen
bulk-level test split tested whether SSL pretraining on 474k unlabeled
igneous bulks from `complete.csv` and/or adding trace elements as model
inputs improves per-phase `R^2_bulk` (the fair metric from B2). Result:
**SSL pretrain delivers material gains, particularly on hydrous and
foid-bearing phases; trace inputs at supervised time still do not add
value.** Adopted v2 path is cell (c) -- SSL-pretrained encoder + majors-
only inputs (same 9-cation forward graph as v1, only the encoder init
weights differ).

| phase | ceiling | v1 (a) | v2 cell (c) | delta |
|---|---|---|---|---|
| **bi**  | 0.98 | +0.53 | **+0.71** | **+0.18** |
| **ne**  | 0.97 | +0.48 | **+0.70** | **+0.22** |
| **mu**  | 0.82 | -0.62 | -0.39 | +0.23 (less bad) |
| amp | 0.67 | +0.32 | +0.40 | +0.08 |
| liq | 0.69 | +0.43 | +0.47 | +0.04 |
| fl  | 0.61 | +0.16 | +0.21 | +0.05 |
| cpx | 0.98 | +0.93 | +0.93 | tied |
| ol  | 0.99 | +0.98 | +0.98 | tied |
| fsp | 0.98 | +0.95 | +0.97 | +0.02 |
| q   | 1.00 | +0.97 | +0.98 | +0.01 |

Cell (c) single-seed test condition MAE: T **14.8** C, P **1.73** kbar, H2O
**0.51** wt% -- **beats v1's K=5 ensemble on every axis** (v1: T 15.4 C, P
1.76 kbar, H2O 0.54 wt%). The encoder warm-started from SSL on unlabeled
chemistry generalizes meaningfully better than the random-init v1 baseline,
even before ensembling.

Adding traces at supervised time (cell d) is slightly worse than cell (c)
on most phases despite richer inputs. The mechanism is consistent with B2:
the FC labels are deterministic functions of majors only, so trace inputs
cannot improve prediction of labels that don't depend on traces. The SSL
pretrain still benefits from seeing major+trace co-occurrences during
pretraining (that's where the ne / bi / mu gains come from -- the encoder
learned the foid-bearing and hydrous-class structure from unlabeled
geochemistry), but the supervised forward graph stays on majors. The
unlabeled corpus does most of the trace-leveraging work; supervised time
keeps the inference contract simple.

Latent-space k-NN sanity check on the pretrained encoder: cosine of test
modal vs k=10 nearest-training-neighbor mean modal in the 512-D latent =
**0.975 vs random-neighbor 0.813, delta +0.162** (vs the earlier broken
+0.145 which only measured mat2vec's input-embedding contribution). Now
the delta reflects the actual SSL pretrain.

**The SSL-pretrained encoder is also a strong rock-type classifier.** The
linear (logistic regression) probe under-reports the SSL gain; a nonlinear
1-hidden-MLP probe reveals it:

| encoder | supergroup LR | supergroup MLP | qap_name LR | qap_name MLP |
|---|---|---|---|---|
| random-init | 0.601 | 0.653 | 0.516 | 0.599 |
| **SSL-pretrained** | **0.615** | **0.763** | **0.537** | **0.720** |
| cell-c fine-tuned | 0.701 | 0.787 | 0.659 | 0.750 |

SSL pretrain alone (no mineralogy supervision) reaches 76% supergroup
accuracy and 72% qap_name accuracy with an MLP probe -- within 2-3pp of
the fully supervised cell-c. The encoder learned class-discriminative
structure from unlabeled chemistry alone. This validates the encoder as a
**foundation model** for rock chemistry, not just a mineralogy feature
extractor. See `docs/PRETRAIN_APPLICATIONS.md` for the family of capabilities
this enables beyond modal prediction.

K=5 ensemble retrain pending; with single-seed cell-c already beating the
v1 ensemble, a v2 ensemble should widen the lead further.

## v3 envisioned (2026-06-02): heads expand, encoder durable, parameterization choices pending pilots

After v2 cell (c) validated the SSL-pretrained encoder as the durable
substrate, v3 commits to the SSL-pretrained encoder architecture validated by v2 cell-c the encoder design and rewrites the supervised heads to add per-phase composition (CompHead) and auxiliary
representation-shaping heads. The crystallized rationale and
implementation handoff live in `docs/FRESH_START.md` and
`docs/V3_WORKFLOW.md`; this section sketches what the trained v3 model
will look like, in the same layout as v1/v2 above. Numbers will land
on the corresponding lines once each piece trains.

### Pipeline at a glance (v3)

```
22-token input  ->  own thin transformer  ->  pooled latent (frozen after SSL pretrain)
   src [22]              (mat2vec + N=4, heads=16,                  |
   frac [22]              d_model=512; explicit padding             |
                          from src==0 only)                    latent h [512]
                                                                    |
        +------------------+------------------+------------------+------------------+
        |                  |                  |                  |                  |
    ModeHead          CompHead           CondHead          ClassHead         G_sys head
   (Dirichlet,       (Dirichlet per     (GMM x 3 axes,    (CE supergroup,    (Gaussian scalar,
    16-18 phases)     SS phase over      T, P, H2O)        aux, free          mu+log_sigma,
                      endmember moles)                     latent shaping)    aux, optional)
        |                  |                  |                  |                  |
   modal vector      per-phase            conditions         supergroup         G_sys est
   distribution      composition          distribution        distribution
                     distribution
```

K=5 ensemble at deployment (same as v1); single-seed acceptance gate
first. Optional Track C end-to-end fine-tune of the encoder runs only
if Track B's frozen-encoder result leaves measurable headroom.

### Input (v3)

Up to **22 cation+trace tokens** per sample (vs v1's 9 cations):

- **9 majors** -- Si, Al, Ca, Mg, Fe, K, Na, Ti, Cr -- from anhydrous
  oxide wt% (FeO + Fe2O3 summed to Fe), divided by 100.
- **13 traces** -- Sr, Zr, Ba, Y, Rb, Ni, Nb, V, La, Ce, Nd, Yb, Th --
  from ppm divided by 1e6.

`src` carries atomic numbers; `frac` carries normalized values in
[0, 1].  Special `src` values:

  - `src = 0` -- padding (absent element).  `frac = 0` at these
    positions; the encoder excludes them from attention via the
    padding mask.
  - `src = MASK_TOKEN_ID` (= 119, one past the periodic table) --
    SSL-masked token (BERT-style).  `frac = 0` at these positions;
    the encoder DOES attend to them and the SSL loss demands the
    original `frac` value back.  The MASK row of the element embedding
    table is a learnable parameter, separate from any mat2vec row.

This is **convention C** from the design discussion (2026-06-02): mask
and padding are distinguishable by `src`, so the encoder knows which
slots to predict.  Earlier convention A (mask = padding, both `src=0`)
was rejected because it deprived the encoder of the "predict here"
signal.  The v3 padding-mask derivation is `padding_mask = (src == 0)`
explicitly -- this is the v2 bug fix and remains the load-bearing
invariant.  See `docs/FRESH_START.md`.

T, P, H2O remain supervision-only -- never forward-graph inputs. h2o_tot
also stays out of the input (the v1 leak fix carries through).

### Encoder (v3)

**Own thin transformer (~70 lines)**, not a wrapper around vendored
kingcrab.  Same `d_model=512`, `N=4`, `heads=16` as v1/v2.  Per-token
construction mirrors CrabNet's Element Derived Matrix pattern:

  - **Element embedding**: mat2vec rows 1-118 + a learnable row
    at MASK_TOKEN_ID, projected from 200-D to `d_model`.
  - **Frac encoding**: two `FractionalEncoder` instances (sinusoidal
    lookup over discretized `frac`), one linear and one log10.  Each
    emits `d_model // 2` channels; concatenated they fill `d_model`.
    The log channel gives the encoder real resolution at low frac --
    important for traces (Sr ~10 ppm vs 100 ppm).
  - **Combination**: three learnable scalars (`emb_scaler`,
    `pos_scaler`, `pos_scaler_log`) let the model re-weight element
    vs linear-frac vs log-frac contributions during training.
  - `token = elem_e * emb_scaler + concat(linear_frac * pos_scaler, log_frac * pos_scaler_log)`

The `FractionalEncoder` is lifted verbatim from CrabNet's
`kingcrab.py` into `src/cnmineral/models/fractional_encoder.py` with
attribution; the rest of kingcrab is NOT vendored.  Padding mask is
derived **explicitly** from `src == 0`, never from `frac > 0`.
MASK_TOKEN_ID positions are NOT padding; the encoder attends to them.
Masked-mean pool over non-padding tokens yields a `[d_model]` latent.

After SSL pretrain (next section), the encoder is **frozen** during
supervised head training.  End-to-end fine-tune is opt-in via Track C.

### Pretraining (v3)

**Masked-element modeling with pooled-latent decoder** (Option H from
the v2 bug-fix design, refined with convention C masking 2026-06-02):

- Per training batch, pick 25% of populated tokens per sample,
  uniformly at random.
- At each masked position set `src = MASK_TOKEN_ID` and `frac = 0`
  (BERT-style sentinel; distinguishable from padding by `src`).
- Run the corrupted `(src, frac)` through the encoder.  Padding
  tokens (`src == 0`) are excluded from attention; MASK_TOKEN_ID
  positions are included.
- A small decoder MLP (`d_model -> d_model//2 -> n_tokens=22`) reads
  the **pooled latent** and predicts `frac` for **every** position.
- Loss = MSE on originally-masked positions only.
- Class-balanced sampling by supergroup so basalt-heavy `complete.csv`
  doesn't dominate.

Pretrain corpus: ~474k unlabeled igneous bulks from `data/complete.csv`
(post igneous filter). One full pass ~16 min on MPS at d_model=512
(v2 timing). **Acceptance gate**: supergroup MLP-probe accuracy
>=75%, k-NN modal cosine >=0.95 on a held-out 5k subset --
matches v2 cell (c)'s numbers.

### The heads (v3)

Each head is a small residual MLP reading the shared pooled latent.
Every head emits the parameters of a probability distribution -- no
MSE / point-estimate heads anywhere in v3 (CLAUDE.md project rule).
All heads run in parallel; turning auxiliary heads off does not
change inference for the others.

**ModeHead** -- Dirichlet over the 16-18-phase modal vector. Same
parameterization as v1/v2. The phase vocabulary may shrink in v3 if
the corpus regen drops `liq` and `fl` from the modal targets per the
deferred v1.1 plan; the head architecture is unchanged.

**CompHead** -- per-phase composition as **per-phase Dirichlet over
endmember mole fractions**. Per solid-solution phase: a simplex of
3-12 endmember fractions (cpx ~6-8 endmembers, amp ~10-12, fsp 3,
opx 4-5, etc.); each phase's head emits a Dirichlet concentration
vector `alpha_phase` (softplus + clamp). ~80 total simplex outputs
across the SS phases.  Per-phase oxide composition is recovered at
output time via `Σ em_frac × em_oxide_stoichiometry` (fixed constant
matrix from the Green2025 endmember basis), so predicted compositions
are **physically valid by construction** -- always inside the convex
hull of endmember oxide vectors.  Mass balance is exact *within* each
phase by construction; bilinear and still-soft at the bulk level
(handled by the mass-balance term in the joint loss).

Pilot 2b in the v3 plan validates that endmember mole fractions are
extractable from PetThermoTools cleanly (or invertible from per-phase
oxide composition with low enough noise that the basis advantage
survives) before the full corpus regen commits.  Decision was for
endmember-fraction over oxide-wt% on three grounds: physical validity
by construction, lower output dimensionality (~80 vs ~143), and
symmetry with ModeHead's Dirichlet parameterization.

**CondHead** -- GMM per axis (T, P, H2O), `K_gmm=3` components per
axis. Identical parameterization to v1/v2.

**ClassHead** (auxiliary) -- cross-entropy over the 8 petrogenetic
supergroups. Reads the same pooled latent; the supervised gradient
shapes the encoder toward petrogenetically meaningful axes for free.
Not a deployment output; can be turned off without affecting inference
for the other heads.

**G_sys head** (auxiliary, optional) -- **heteroscedastic Gaussian**
over MAGEMin's system Gibbs energy at the FC endpoint.  Emits
`(mu, log_sigma)`; loss is the 1-D Gaussian NLL.  **Kept as a
separate head from CondHead, not folded in as a 4th axis.**  The
reason is semantic role in the MAGEMin forward map: (T, P, H2O) are
*conditions you give MAGEMin* (the CondHead re-infers the thermodynamic
context that produced the rock); G_sys is *what MAGEMin produces at
the resulting state* (the G_sys head re-infers a derived thermodynamic
quantity of the equilibrium).  Different semantic role → different
head, with an independent `lambda_gsys` knob and an independent
calibration story.

**Not a polymorph disambiguator** -- a scalar cannot separate Ky/Sil/And
(architectural decision in CLAUDE.md).  Useful for (1) encoder
regularization toward thermodynamic-stability axes, (2) internal-
consistency check at inference (predicted M + C + conditions should
imply the predicted G_sys; large divergence is an uncertainty signal
independent of ensemble disagreement).  **Conditional on the v3 corpus
regen persisting G_sys**, which the v1 corpus generator discarded;
include only if persistence is cheap to add to the regen.

Bump to GMM only if pilot evidence shows multimodality in per-bulk
G_sys across FC paths.  Default starts at K=1 (single Gaussian).

### Training (v3)

**Joint loss**, fully probabilistic, with mass-balance regularization:

```
joint_loss = NLL_dirichlet(ModeHead, FC_modal)                                       # phase simplex
           + lambda_comp     * sum_phase NLL_dirichlet(CompHead_phase, FC_endmember_fracs_phase)
           + lambda_balance  * mass_balance_loss(bulk, M_pred, C_pred)               # bilinear bulk reconstruction
           + lambda_cond     * sum_axis NLL_gmm(CondHead_axis, FC_axis)              # axes: T, P, H2O
           + lambda_class    * CE(ClassHead, supergroup)                             # categorical
           + lambda_gsys     * NLL_gauss(G_sys_head, FC_G_sys)                       # mean + log_sigma
```

Every term is the NLL (or CE for the discrete categorical) of a
probability distribution matched against its label.  No MSE anywhere
-- the project rule in CLAUDE.md ("Don't use MSE as the head loss")
holds end-to-end for v3.

`mass_balance_loss` reconstructs each bulk-oxide from the predicted
modal × per-phase composition (recovered from CompHead endmember
fractions × known stoichiometry matrix) and penalises deviation from
the input bulk: `Σ_oxide (bulk_oxide - Σ_phase M_phase × C_phase_oxide)²`.
`lambda_balance` swept in pilot 3 (0 / 0.1 / 1.0 / 10.0).

The last two terms are auxiliary; setting `lambda_class` or `lambda_gsys`
to zero removes those gradient signals without affecting the supervised
heads.  `lambda_cond` inherits v1's HPO-tuned region as a starting
point.

**Encoder frozen** during Track B head training; Track C unfreezes
with `encoder_lr = 0.1 * head_lr` per the standard transfer-learning
recipe.

**Data splits** -- bulk-level, same convention as v1.
`data/splits_v3.json` rebuilt once the v3 corpus regen produces
`train_fc_v3.parquet`. Splits regenerate because the v3 corpus may
have a different bulk set (filter changes, attrition differences) than
v1; bulk-level 64/16/20 stratified by `qap_name`, rare classes
(<25 bulks) dropped, same recipe.

**HPO** -- joint single-stage Optuna TPE over the new lambdas
(`lambda_comp`, `lambda_balance`, `lambda_class`, `lambda_gsys`,
`lambda_cond`) and `K_gmm`.  `JournalStorage` backend (resumable).
Search bounds inherit v1's winning region for shared knobs.

**Acceptance gate for Track B (single seed)**: per-phase `R^2_bulk`
>= v2 cell-c on hydrous + foid-bearing phases (amp, bi, mu, fl, ne);
CondHead 1-sigma coverage closer to nominal 0.68 than v1's loose
0.88/0.64/0.76; CompHead predictions land inside stoichiometric bounds
for >=90% of test bulks.

### Inference contract (v3)

**Inputs:** majors + traces in oxide wt% / ppm units, any subset
(absent elements pad to `src=0`).

**Outputs (per sample):**
- Dirichlet over the 16-18-phase modal vector (mean = predicted modal
  fractions; variance = aleatoric uncertainty)
- One Dirichlet per SS phase over its endmember mole fractions; the
  per-phase oxide composition is derivable at inference via the fixed
  endmember stoichiometry matrix (always physically valid)
- GMM per condition axis (T, P, H2O)
- Supergroup categorical distribution (auxiliary; optional surface)
- G_sys Gaussian (mean, sigma; auxiliary; optional surface, if trained)
- Latent vector h (for downstream applications -- k-NN retrieval, OOD
  scoring, imputation; see `docs/PRETRAIN_APPLICATIONS.md`)

K=5 ensemble at deployment for epistemic uncertainty (same recipe as
v1/v2).

## Known limitations of v1 / v2

- **Calibration: T was loose, P and H2O were already close** (corrected
  reading -- the Gaussian-`mean +/- sqrt(var)` coverage 0.88/0.64/0.76 used
  in the headline overstates miscalibration for a mixture predictive). The
  proper diagnostic is the PIT (predictive CDF at the target; uniform if
  calibrated). PIT central-68 coverage on test is **T 0.81 / P 0.62 / H2O
  0.71**; PIT KS vs uniform is **0.17 / 0.05 / 0.11**. Only T is materially
  miscalibrated (clean dome PIT -- predictive too wide). Post-hoc temperature
  scaling per Guo et al. 2017 (sigma-scaling regression variant -- multiply
  each component sigma by `exp(t_axis)`) is implemented in
  `scripts/07_calibrate.py`; adopted temperatures
  `T=0.655, P=0.943, H2O=0.932` shrink T's PIT KS to 0.12 with NLL gain;
  P and H2O scaling is near a no-op (their PIT lumpiness is largely an
  artifact of evaluating a continuous CDF on the discrete sampled targets --
  P takes 4 values per bulk, H2O takes 2). Calibration artifact at
  `checkpoints/base_v1_ensemble/calibration_v1.json`; PIT histograms at
  `data/diagnostics/calibration/pit_hist_v1.png`. Inference/eval wiring (one
  flag that loads the JSON and adds `t_axis` to `cond_log_sigmas`) is
  deferred.
- **Hydrous-phase modal accuracy is weak.** amp, bi, mu, fl regressed when
  `h2o_tot` was dropped to fix the H2O leak -- that column was also a genuine
  predictor for water-dependent phases. v1.1 Option C restores the legitimate
  signal by using complete.csv's *measured* `h2o_tot` (which differs from the
  FC `h2o_path_wt` target) as an input.
- **Rare phases.** sph, ep, mu, ru have very low corpus presence and negative
  R^2 -- too few examples to fit.
- **Modal abundance only.** 13 of the 18 phases are solid solutions; the
  ModeHead predicts how *much* of each phase, not its *composition*. The
  v1.1 CompHead (`[n_minerals, 11]` per-mineral oxide wt%) addresses this.

## Code map (v1 / v2)

The v3 codebase rewrites these files under a new package structure; see
`docs/FRESH_START.md` and `docs/V3_WORKFLOW.md` for the v3 layout.

| file | role |
|---|---|
| `src/cnmineral/models/encoder.py` | SubCrab encoder + masked-mean pooling |
| `src/cnmineral/models/heads.py` | ModeHead (Dirichlet), CondHead (GMM) |
| `src/cnmineral/models/model.py` | `BaseModelV1` = encoder + the two heads (also v0 `BaseModel`) |
| `src/cnmineral/models/losses.py` | `dirichlet_nll`, `gmm_nll_1d`, `joint_loss` |
| `src/cnmineral/models/dataset.py` | `TrainFCv1Dataset`, oxide->cation conversion, split column lists |
| `src/cnmineral/models/training.py` | train/eval loops, bulk-level split + CV-fold helpers |
| `src/cnmineral/models/objective.py` | Optuna HPO objective |
| `scripts/03_train_base.py` | single-model train (v0/v1 dispatch) |
| `scripts/04_hpo.py` | HPO study driver |
| `scripts/05_train_ensemble.py` | K-member ensemble training |
| `scripts/06_eval_ensemble.py` | ensemble eval, per-phase / per-class metrics, predictions dump |
| `notebooks/v1_eval.ipynb` | eval dashboard + simulated-user inference demo |
| `configs/base_v1.yaml`, `configs/hpo_v1.yaml` | training + HPO config |
| `data/splits_v1.json` | frozen bulk-level train/val/test split |

## Keeping this doc current

Two living areas in this doc:

- **v1 / v2 sections above are historical** -- frozen at the
  `v1-trained` and `v2-traces-ssl-final` tags. Edit only to correct
  errors of fact about those trained checkpoints.
- **v3 envisioned section** -- update as design crystallizes during
  the Track A and Track B work. As specific lambdas, head shapes, and
  acceptance numbers land, replace the "pending" language with the
  measured result and the corresponding JOURNAL entry. When v3 ships,
  this section graduates from "envisioned" to a permanent v3 reference
  alongside v1/v2.

Note significant changes here and in `docs/JOURNAL.md`.
