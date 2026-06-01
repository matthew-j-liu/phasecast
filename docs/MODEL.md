# v1 model: how it works

Living reference for the CrabNet-Mineral v1 model. Explains the architecture
in plain terms; updated in place as the model evolves. For *why* each choice
was made (and what was rejected) see ADR 006; for the chronological build log
see `JOURNAL.md`.

As-built as of the `v1-trained` tag (2026-05-22).

> **Direction forward (2026-06-01):** v1 (this doc) and v2 (SSL pretrain +
> traces, see JOURNAL 2026-05-29) are stable reference points. The next
> direction is a fresh-codebase v3 rewrite that adds CompHead, ClassHead,
> optional G_sys head, and uses the SSL-pretrained encoder as the durable
> substrate. The crystallized design and implementation handoff is in
> `docs/FRESH_START.md`. This MODEL.md document remains the reference for
> the v1 / v2 trained checkpoints; do not edit it to describe v3.

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

## Known limitations

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

## Code map

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

Update in place when the model changes. The likely next edits: v1.1 Option C
(input set goes back to including a measured `h2o_tot`), the CompHead (a third
head), and any calibration fix. Note the change here and in `JOURNAL.md`.
