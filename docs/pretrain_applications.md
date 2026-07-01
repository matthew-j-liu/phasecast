# SSL-pretrained encoder: capabilities beyond mineralogy

The SSL-pretrained encoder is a **rock-chemistry foundation model**, not
just a feature extractor for the mineralogy task. Trained via masked-element
modeling on ~167k unlabeled igneous bulks from `data/bulks/gard2019.csv`, it
organizes rocks in a latent space that captures real geochemical
structure. Mineralogy is one downstream application. This doc enumerates the
others -- capabilities the existing encoder already supports with no further
training, or with tiny supervised heads on the pretrained latent.

**v3 encoder is available.**  Canonical checkpoint at
`checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt` (per-element
standardized log-space masked MSE, ADR 007); hyperparameters in
`configs/pretrain/pretrain_production.yaml`.  See
`docs/roadmap.md` for current probe numbers and gate-close
status.  All capabilities in this doc target the v3 encoder;
v2's encoder remains usable as a reference.

Living document; update as capabilities are built or evaluated.

## What the pretrain head + pooled latent already give you

These run on the **frozen** encoder with no encoder retraining. k-NN (#3)
and OOD (#4) need no training at all; element imputation (#1) and the anomaly
flagging built on it (#2) add a small read-out head trained in minutes on the
frozen latent (same cost class as the heads in the next section).

### 1. Element imputation from partial analyses

A user with majors only (or majors + a few traces) can get predicted values
for the missing elements. Use cases:

- Legacy / incomplete datasets where ICP-MS traces were never measured
- Labs where major-element results arrive faster than trace
- Cross-validation: predict each measured element from the others, compare
  to reported value, flag large discrepancies as possible analytical errors

**Imputation runs through a read-out head, not the SSL `PretrainHead`.** A
read-out head is the standard foundation-model pattern: freeze the encoder
and train a small head on its pooled latent to read out one quantity (here,
the masked element's value). It is *not a new architecture* -- it is the same
application-slot pattern the v3 heads use (ModeHead / CompHead / CondHead /
ClassHead all read the same frozen `h`). It shares `PretrainHead`'s small-MLP
shape but differs in role:

| | `PretrainHead` | imputation read-out head |
|---|---|---|
| trained | jointly with the encoder during SSL | alone, encoder frozen |
| objective | per-element standardized log-space MSE (ADR 007) | the same objective; capacity (hidden width, dropout) tuned to imputation |
| role | SSL machinery: drives the masked-element loss that shapes the encoder | downstream application head, re-supervised post-hoc |
| lives in | the pretrain checkpoint | its own `ImputationHeadCheckpoint` |

`PretrainHead` is not deleted -- it must exist to pretrain the encoder, and
its imputation outputs were always a side effect. Under the *original*
linear-frac SSL objective those outputs were decent on majors but collapsed on
traces: a linear-frac loss is dominated by the majors, so trace predictions
optimized to a near-constant and went negative in log-space R2. The
trace-imputation gate (`explorations/trace_signal_in_latent/`, JOURNAL
2026-06-15) showed this is a *head-objective* problem, not encoder capacity --
XGBoost on the frozen masked latent reaches ~94% of the input-element XGBoost
ceiling across the 13 traces. So the fix is the right objective on the
*existing* frozen latent, not a re-pretrain: the SSL objective itself moved to
standardized log-space (ADR 007), and this separately-tuned read-out head reads
the frozen latent at the imputation task's own capacity.

**Shipped.** The production head is Optuna-tuned (single-stage sweep, objective
median trace log-R2 on a held-out val split) and reaches test median trace
log-R2 0.836 = 95% of the 0.877 frozen-latent yardstick, with no major
regression; per-trace 95-98% of the yardstick (JOURNAL 2026-06-24).
`configs/imputation/imputation_head.yaml` regenerates it on the canonical
encoder; the checkpoint + readout artifacts are gitignored (regenerable).

Implementation (built):
- `cnmineral.models.imputation_head.LogImputationHead` -- linear or
  one-hidden-MLP read-out emitting per-element standardized `log10(frac)`,
  with de-standardization buffers carried in `state_dict`.
- `cnmineral.losses.pretrain_loss.masked_element_log_mse` -- the standardized
  log-space objective, shared by the SSL pretrain and this read-out head.
- `scripts/training/imputation_head.py` -- the single entry, mirroring
  `pretrain.py`: a plain `--config` run trains the head on a frozen encoder
  (`train_imputation_head`) then scores the test split once, writing the
  log-space scorecard to `results/imputation/<encoder_tag>/readout/`; `--hpo`
  runs the head sweep instead. Configs: `configs/imputation/imputation_head.yaml`
  and `imputation_head_hpo.yaml`. Dataset/splits reconstruct from the encoder
  checkpoint's config.
- `cnmineral.training.imputation_head_hpo.run_hpo` -- single-stage Optuna
  sweep of the head HPs (the `--hpo` path).
- `cnmineral.eval.imputation.run_imputation` -- log-space leave-one-out scoring
  against the gate's two reference lines; the scoring core the single run calls
  after training.

The frozen-latent XGBoost number stays a *yardstick*, never a shipped model.
It is a tuned gradient-boosted forest, so it measures a strong nonlinear
readout's reach on the latent, not a hard ceiling: a linear or one-layer head
can trail it on readout capacity even when the latent holds the signal (the
MLP head closes most of that gap; a linear head leaves more on the table).
Closing the remaining gap from that target to the input ceiling (worst on
HREE) is a different problem -- it would need a stronger encoder, the separate
re-pretrain epic, not this head.

### 2. Per-element anomaly / measurement-error flagging

Same machinery, run as an outlier detector: predict each element from all
the others, compare to reported value, flag standardized residuals beyond
a threshold. Could be deployed as a "sanity check" pass on incoming user
analyses.

Effort: ~half day, mostly UX (deciding thresholds, presentation).

### 3. k-NN similarity lookup against 1M+ bulks

Compute the pooled latent of a user's bulk; find the K closest bulks in
`complete.csv` (which has reference IDs, geographic origin, age info, and
rock-class metadata). Return: "your rock looks most like these N samples
from [location], [age], previously classified as [name]."

This is *the* canonical user-facing feature of any embedding-trained
foundation model. The infrastructure is already partly built inside
`scripts/09_visualize_latent.py:knn_mineralogy_check` -- needs lifting out
into a reusable inference function and a UI.

Effort: ~1 day. Mostly: precompute the full 1M-row latent cache once,
then nearest-neighbor lookup at inference (FAISS or sklearn for the
search index).

### 4. Out-of-distribution flagging

Already implemented as a diagnostic in
`scripts/09_visualize_latent.py:ood_distance_distribution`. At inference,
compute distance from the user's bulk to its k=10-th nearest training-set
neighbor in the pretrained latent. Flag bulks above the 99th-percentile
training threshold as out-of-distribution.

Lifted from mineralML's "Stage 2 returns OOD" practice. This is a real
safety net: a user submitting a contaminated sample, a measurement error,
or a rare rock type should get an explicit "out of training distribution"
warning rather than a confident-but-wrong prediction.

Effort: already built; ~half day to wire into an inference pipeline.

## Capabilities that need a tiny supervised head

These add a small classification or regression head on top of the
pretrained latent. Trained in minutes (logistic regression or 1-hidden
MLP); no encoder retraining required.

### 5. Tectonic-setting classification

`complete.csv` doesn't have direct tectonic-setting labels, but 60+ years of
geochemical-discriminant-diagram literature defines MORB / IAT / OIB / CFB /
arc / continental categorizations as deterministic functions of trace ratios.
Examples:
- Pearce (1982) Nb-Y
- Cabanis & Lecolle (1989) La-Y-Nb
- Wood (1980) Th-Hf-Ta
- Shervais (1982) Ti-V
- Pearce & Cann (1973) Zr-Ti-Y-Sr

Compute these labels from `complete.csv` for free (they're just functions of
trace columns we already have). Train a small classifier `latent ->
tectonic class`. Resulting model takes a bulk's chemistry and returns
"this looks like a MORB" with class probabilities.

**This is where trace info actually pays off cleanly** -- the labels
depend on trace ratios that the encoder learned from SSL. Should be
strong predictions; the model has the right inputs (encoded chemistry)
and trustworthy targets (deterministic discriminants).

Effort: ~1-2 days. Most of the work is selecting which discriminants to
use and handling the trace coverage holes in complete.csv.

### 6. Magma-series classification

Tholeiitic / calc-alkaline / alkaline. Targets computable from majors +
alkali content via the standard diagrams (Irvine & Baragar 1971 AFM,
Miyashiro 1974, Le Bas 1986 TAS-derived). Small classifier on the latent.

Effort: ~half day after #5's infrastructure is in place.

### 7. Petrogenetic-ratio regression with uncertainty

Predict named, interpretable ratios from the latent: Sr/Y, La/Yb, Nb/Y,
Th/Yb, Mg#, K/Rb, Ba/La, Eu/Eu* (Eu anomaly), etc. These are computable
deterministically from input chemistry, but a regression head that returns
**predicted value + uncertainty** has real value as a sanity check or for
incomplete-input scenarios.

Effort: ~1 day. Multi-output regression, Bayesian last layer for
uncertainty if desired.

### 8. Differentiation index / position on liquid line of descent

Continuous "how evolved is this magma" scalar. Loosely correlated with
Mg# but more nuanced (accounts for residual Th, K, Rb buildup, REE
patterns). Useful for batch comparison of cogenetic samples.

Effort: ~1 day. Mostly: defining the index labels (e.g., from existing
LLD calculators in pyrolite).

### 9. Isotopic-ratio prediction

`complete.csv` has columns for `epsilon_Nd`, `epsilon_Sr`, `epsilon_Hf` on
a non-trivial fraction of rocks. Regression head on the latent; for bulks
where isotopes weren't measured (the majority), get predicted values plus
uncertainty.

**Genuine new value** -- isotope measurements are expensive (~$200+/sample).
An honest predicted value with uncertainty is useful for screening,
exploration, regional studies.

Effort: ~2 days. Need to filter to bulks with isotope data, train, validate
against held-out measured isotopes.

### 10. Approximate age regression

Many bulks in `complete.csv` have rough age info (numeric or geological era
binned). Chemistry-from-age isn't strongly determined for most rocks, but
distinctive populations (Archean greenstones, Proterozoic granites, Cenozoic
arc rocks) do have characteristic signatures. A regression head with **large
honest uncertainty bars** is the right framing.

Practically more useful as a coarse classifier (Archean / Proterozoic /
Phanerozoic) than a continuous regression.

Effort: ~1-2 days including the binning and honest-uncertainty modeling.

### 11. Geographic / craton-of-origin inference

Crustal regions have characteristic rock populations. A classifier could
surface "rocks like this most commonly come from [region]." Less rigorous
but interesting for prospecting / exploration / provenance contexts.

Effort: ~1-2 days. Geography labels in `complete.csv` would need cleaning;
target should probably be at the craton or geological-province level, not
country.

## Capabilities the encoder does NOT support (without further work)

To stay honest about what's NOT in scope of the pretrained encoder alone:

- **Modal mineralogy prediction.** The original mineralogy track. Needs
  supervised training against FC labels (which is what v2 cells did). The
  pretrained encoder is a strong warm start (cell (c) wins over v1) but
  the modal head is still trained.
- **Per-mineral solid-solution composition.** The deferred CompHead (B1)
  work. Needs corpus regen + mineralML grain-data fusion.
- **System Gibbs energy.** The deferred GsysHead (B8). Needs corpus regen
  with `G_sys` columns.
- **Trace-element discriminative tasks with FC-derived labels.** Traces
  don't influence FC labels, so trace inputs don't help predict them. The
  tectonic-setting / petrogenetic-ratio tasks in this doc are different
  because their labels DO depend on traces.

## Prioritization for a "geochemistry copilot" app

If building an app that exposes these capabilities to a user, the
prioritization roughly:

| priority | capability | rationale |
|---|---|---|
| 1 | k-NN similarity (#3) | Most directly useful "find rocks like mine" UX; uses encoder + a precomputed cache; no training |
| 1 | Element imputation (#1) | Solves a real lab pain point (fill in partial analyses); shipped, via a tuned read-out head on the frozen latent |
| 1 | OOD flagging (#4) | Safety net; already built |
| 2 | Tectonic-setting classification (#5) | Big interpretability win; trace-discriminator labels |
| 2 | Petrogenetic-ratio regression with uncertainty (#7) | Interpretable, well-defined |
| 3 | Magma series (#6) | Cheap once #5 is built |
| 3 | Anomaly flagging (#2) | Pairs naturally with #1 and #4 |
| 4 | Isotope prediction (#9) | Real new value but data is sparse |
| 5 | Age regression / coarse classifier (#10) | Honest uncertainty important |
| 5 | Geographic / craton (#11) | Niche but interesting |

## Relationship to the mineralogy track

Both tracks share the same encoder. The mineralogy track (`v1-trained`,
v2 cell-c) adds the ModeHead + CondHead and trains supervised against FC
labels. The applications track in this doc adds zero or one small head
per capability without retraining the encoder.

Both tracks benefit if the next encoder iteration is stronger (e.g., from
per-token-normalized loss, larger pretrain corpus, longer training).
Improvements to the encoder propagate to both tracks; improvements specific
to the mineralogy heads or specific to the applications heads do not
interfere with each other.
