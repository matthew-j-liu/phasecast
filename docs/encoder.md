# The encoder: foundation model, data, and value proposition

The canonical reference for the shared substrate: the value proposition, the
SSL-pretrained encoder, the tokenization and data flow, the three corpora, the
pretraining mechanic, and the single load-bearing code invariant. The encoder
is the durable contribution; the supervised heads that read its latent are a
separate concern.

Companion to `heads.md` (the supervised heads + posterior). The last two
sections are the representation scorecard and the encoder's capabilities
beyond mineralogy.

## Value proposition (the one-sentence pitch)

> A probabilistic emulator of MAGEMin integrated over geologically plausible
> formation conditions, with a rock-chemistry foundation-model encoder
> learned from real bulks, parameterized so real labeled data can replace
> synthetic labels later without retraining the encoder.

Three things this captures that "MAGEMin surrogate" doesn't:

1. **Marginal not conditional.** MAGEMin computes `Y = f(X, C)` for a single
   condition `C = (T, P, H2O)`. The model learns `p(Y | X) = ∫ p(Y|X,C) p(C) dC`
   from samples. The user gives a bulk; the model gives the distribution
   over mineralogy and conditions consistent with that bulk. That integral
   is what a petrologist computes mentally. MAGEMin doesn't.

2. **Distribution not point estimate.** With a probabilistic ModeHead + GMM
   CondHead, the within-bulk variance across FC paths becomes the trained
   uncertainty, not noise to average out. The 100-path degeneracy
   visualization (see `data/diagnostics/degeneracy_viz/`) makes this
   concrete: same bulk, different (P, H2O), modal abundance varies up to
   100% relative for hydrous phases. The right output is a distribution.

3. **Foundation-model encoder.** SSL pretraining on real bulks (~167k
   igneous in the 22-token generation, ~788k all-rock in the accepted
   23-token generation)
   organizes the latent by petrogenetic structure WITHOUT MAGEMin labels.
   Empirically: ~80% supergroup MLP-probe accuracy with zero supervision
   (vs ~76% for v2 cell-c); strong k-NN neighborhood structure on
   qap_name -- a large multiple over a random-neighbor baseline. The
   encoder is the durable contribution that survives any specific
   generator's biases. Heads are application slots; the encoder is the
   substrate.

### Four-axis differentiation

The contribution is covering all four axes simultaneously -- not any single
one:

| Tool | bulk-only input | no spot analyses | multimodal output | hydrous + sedimentary phases |
|---|---|---|---|---|
| MAGEMin | needs T,P | yes | one assemblage | yes |
| mineralML | yes | needs spots/pixels | no | partial |
| CIPW (pyrolite) | yes | yes | single answer | no |
| **phasecast** | **yes** | **yes** | **yes** | **yes** |

## Encoder composition

The encoder composes three pieces, implemented in `src/cnmineral/models/`:

- **mat2vec embeddings** -- a 200-dim chemistry-aware vector per element
  (learned from a materials-science text corpus). CSV at
  `src/cnmineral/data/element_properties/mat2vec.csv`. Row 0 is padding,
  1-118 are elements, 119 is the learnable `[MASK]` row.
- **dual FractionalEncoder** -- sinusoidal lookup over discretized `frac`, one
  linear and one log10 variant, giving the trace-element resolution a single
  linear projection would lose (lifted from CrabNet's kingcrab with
  attribution).
- **TransformerEncoder** with an explicit `src == 0` padding mask (NOT derived
  from `frac` -- see the load-bearing invariant below) and masked-mean pooling
  over non-padding tokens.

## Input / output / architecture sketch

```
INPUTS (per sample)
  src  : [22]  atomic numbers, 0 = absent (padding)
  frac : [22]  values in [0, 1] (majors: wt%/100, traces: ppm/1e6)

  V3_CATIONS_ORDER (22 tokens; ordering fixed):
    majors (9): Si Al Ca Mg Fe K Na Ti Cr
    traces (13): Sr Zr Ba Y Rb Ni Nb V La Ce Nd Yb Th

  The all-rock generation (pretrain_allrock_redox_seed0) appends a 23rd
  token: the Fe-redox ratio, populated where a source reported the
  FeO + Fe2O3 pair and padded elsewhere.

ENCODER (shared across all heads)
  element embedding (mat2vec)  -> per-token vector
  + frac projection             -> per-token vector
  -> Transformer encoder (d_model / N / heads set per pretrain config, stamped in ckpt.config)
  -> masked-mean pool over non-padding tokens
  -> latent h : [B, d_model]
```

The parallel supervised heads all read the same pooled latent `h`; their
parameterization (the hurdle ModeHead, CompHead, CondHead, the auxiliary
ClassHead / G_sys heads) lives in `heads.md`.

## Where G_sys fits (and where it doesn't)

G_sys (system Gibbs energy at the FC endpoint) is a single scalar per path.
It is **not a polymorph disambiguator** -- explicitly rejected at project
inception (the same bulk can have different polymorph identities with very
similar G_sys; a scalar can't disambiguate). Don't pitch it as one.

What it IS useful for:

1. **Auxiliary head as encoder regularizer.** Predicting G_sys forces the
   encoder latent to organize along thermodynamic-stability axes -- rocks
   close to the silica-undersaturated solidus vs evolved felsic vs ultramafic
   should all have characteristic G_sys ranges. Free gradient signal,
   cheap to add (`d_model -> 1` MLP), runs in parallel with the other heads.

2. **Internal-consistency check at inference.** Predicted mineralogy +
   composition + conditions should imply a G_sys close to the predicted
   G_sys. Large divergence flags inconsistent prediction -- useful as an
   uncertainty signal independent of ensemble disagreement.

3. **Phase-equilibrium loss term.** If you train CompHead to also predict
   endmember mole fractions (pilot 2b option B), the implied G_sys is
   derivable from the endmember activities. Adding a loss term
   `||G_sys_predicted - G_sys_from_(M, C)||^2` ties the heads via
   thermodynamics, not just mass balance.

The early v1 corpus discarded G_sys during FC; the regen persists it per
rung (see ADR 004 "Storage shape"), so a G_sys head is
now a free decision -- no corpus cost either way.

**Don't add G_sys in v3 if it would delay shipping.** It's a Tier 2 head,
worth a section in the regen if it's cheap, but not blocking.

## Training data

Three corpora, all generated upstream of the model:

1. **SSL pretrain corpus**: `data/bulks/gard2019.csv`. Two generations.
   The igneous 22-token generation filtered to ~167k bulks (igneous, all 9
   majors reported including both FeO and Fe2O3, non-null `qap_name` in a
   class of >=25; see `bulk_chemistry.py` for the walk) and pretrained
   `pretrain_v3_logmse_std_seed0` -- the previous canonical encoder,
   superseded by the cascade-2 encoder verdict (2026-08-05). The all-rock
   23-token generation (LANDED 2026-07-28, `pretrain_allrock_redox_seed0`)
   is the canonical encoder and the one under the production heads: the
   tuned-vs-tuned contrast against the igneous encoder was a statistical
   tie leaning all-rock, so adoption rides its independent merits. It relaxes
   to the Fe fallback chain, drops the igneous filter (~788k rows, all rock
   types), and adds the optional Fe-redox ratio input token, populated
   where a source reported the FeO + Fe2O3 pair (24.1% of the corpus) and
   padded elsewhere -- strict Fe pairs matter for MAGEMin
   label generation, not encoder pretraining. Acceptance record in
   the scorecard below; guardrails in #24; JOURNAL 2026-07-23/28.

2. **Supervised corpus**: gard2019 bulks with MAGEMin FC labels, chunked
   parquet under `data/corpus/`. Two measured generations. The old
   broad-draw corpus -- the one the PRODUCTION heads
   (`joint_heads_allrock_tuned_seed0`) were trained on, and the one cascade 2
   trained on: 10,000 train bulks × 48 draws + 500 valtest bulks × 128
   draws, rungs 50/30, committed roster 221 val / 278 test. The traced
   regen (`data/corpus/train_traced`, 2026-07-27): 35k train bulks × 24
   trace-conditional draws + 2000 valtest × 128 on the frozen 887/1112
   roster (instrument `valtest128-traced@2000`). Heads trained on the
   traced corpus lost to the old-corpus model on the real-rock benchmark
   after identical recalibration (journal 2026-08-03), so the traced
   corpus is a measured non-production arm; its wins are quartz presence
   and assemblage ranking.

3. **Held-out evaluation corpora**: real-rock targets (mineralML grains at
   `data/grains/mineralml.csv` for CompHead evaluation; EDS maps if available)
   -- never used in training, only for cross-modal validation.

Critical schema requirements for the v3 corpus regen, beyond what v1 has:

- per-phase composition columns (`<oxide>_<phase>`, cumulate mass-weighted
  across FC steps; see `scripts/experiments/visualize_degeneracy.py` on tag `v2.0-traces-ssl` for
  the extraction logic that figures out the PTT short-code mapping and
  the FC-cumulate vs FC-snapshot distinction)
- optional `G_sys` scalar per path
- optional per-phase endmember mole fractions (pilot 2b decides whether
  to use these as CompHead targets instead of oxide-wt%)
- bulk_oxide_renormalized column for the mass-balance loss term

### Strategy modules (corpus-gen priors)

The corpus-gen pipeline's petrologic priors live in `src/cnmineral/data/`:

- `fe_redox.py` -- Fe3+/ΣFe ratio (`ClassRatioStrategy`, Middlemost 1989
  per-class defaults; a perturbed table is the sensitivity-analysis hook)
- `condition_proposal.py` -- (P, H2O) sampling: the bounded uniform proposal
  under the Newman & Lowenstern 2002 rhyolite solubility, and the
  trace-conditional mixture of literature boxes (per-class Wallace/Plank and
  Putirka 2008 ranges where traces are missing)

Tier-1 (measured) -> use directly. Tier-2 (assumed) -> strategy applies. A
provenance label is carried through the pipeline so evaluation can stratify by
signal quality.

## Pretraining (the SSL mechanic)

Masked-element modeling, pooled-latent decoder (Option H from the
v2 bug-fix). The convention that worked:

```
For each batch:
  pick a tuned fraction of populated tokens per sample, uniformly at random
    (mask_prob, Optuna-tuned, ~0.23 in the production config)
  set src = MASK_TOKEN_ID (= 119) and frac = 0 at those positions
    (distinguishable from padding by src; the encoder attends to them)
  pass the corrupted (src, frac) through the encoder
  decoder MLP on the pooled latent predicts frac for EVERY position
  loss only at originally-masked positions
```

This is convention C (see the load-bearing invariant below): masked tokens
are NOT padding, so the encoder must attend to them to predict their frac
from context.

Decoder shape: `d_model -> d_model//2 -> n_tokens=22`. Output is a vector
of predicted frac values for all 22 positions; loss reads off the masked
ones only. The non-masked predictions are unsupervised side outputs --
they don't get a loss because they'd just be teaching the model to
reproduce its own input.  The canonical objective is per-element
standardized log-space MSE (ADR 007); linear-space MSE is the retired
original.

The decoder architecture is fixed, not tuned: the `d_model//2` bottleneck
and the identity output are baked in, neither exposed as a hyperparameter.
Identity output is the correct choice (not a deferred one) because the
log-space target is unbounded in R, so a squashing activation would be
wrong.  The head is a training-time scaffold -- it exists only to shape the
encoder via the reconstruction gradient and is discarded at inference (only
the encoder ships).

Class-balanced sampling by supergroup during SSL prevents basalt
over-fitting (complete.csv is ~50% mafic-volcanic).

## What's deferred (do not pre-build)

These are scoped in the plan but should NOT be implemented before pilots
prove their value:

- ALIGNN structural stream (v2 deferral; still deferred)
- Polymorph head (Ky/Sil/And, Qz/Crs/Trd) -- needs explicit polymorph
  labels that MAGEMin doesn't provide
- fO2 sweep (currently class-conditional via fe_redox; sweeping is v3+)
- TracePartitionHead (the 4th head from the trace-usage design taxonomy;
  separate corpus-gen requirement around Kd database lookups)

## The single most important code invariant

```
padding mask is derived from `src == 0` ONLY.
masked-for-SSL sets `src = MASK_TOKEN_ID` (= 119) and `frac = 0`.
masked-for-SSL is therefore DISTINGUISHABLE from padding by `src`,
and the encoder attends to MASK positions (it must, to predict).
```

This is **convention C** from the 2026-06-02 design discussion.  An
earlier v3 sketch used convention A (mask = padding, both `src=0`),
which makes the padding-mask one-liner above the only safeguard.  We
switched to C so the encoder can BERT-style attend specifically to
masked positions; the explicit padding-mask line remains the
load-bearing v2-bug-prevention invariant.

If you remember nothing else from the v2 codebase, remember the
padding-mask line.  It's the difference between "SSL works" and "SSL
silently has zero encoder gradients for a week."

## Representation scorecard

A living, deliberately-provisional rubric for the *strength* of the encoder's
latent representation. It is not a gate and nothing here is locked -- it is the
synthesis layer over the individual eval investigations, kept in one place so
"is the representation strong?" is a constant, anchored question rather than an
occasional vibe.

### How to read it

- **Every axis carries a floor and a ceiling anchor.** A raw number is
  meaningless on its own; the signal is where the current value sits *between its
  floor (no information) and its ceiling (the most any method extracts)*. An axis
  without anchors does not belong here -- that is the lesson of
  [input_knn_ceiling](../explorations/archive/input_knn_ceiling/README.md). The "fluff"
  trap is reporting a value with nothing to compare it to.
- **Status is a lifecycle, not a grade.** `candidate` = an axis we intend to
  measure (usually has an open issue); `provisional` = measured and anchored but
  the target/verdict is not fixed; `locked` = promoted to an ADR as a standing
  criterion. Axes earn their place as evidence lands; some may be dropped.
- **Do not optimize UMAP-2D appearance.** Rock chemistry is largely continuous,
  so clean 2D separation is partly the wrong target.

### Scope and refresh

Numbers below are for the canonical encoder `pretrain_allrock_redox_seed0`
(`epoch_best`), scored on the inherited shared-row roster
(`pretrain_inherited_test_roster.csv`; 32,950 of its 33,371 ids survive the
all-rock corpus filters) with the redox token padded -- the deployment
regime, and the condition the previous canonical encoder can be compared
under. Values in parentheses marked "prev" are the previous canonical
encoder `pretrain_v3_logmse_std_seed0` re-scored on the same rows (its
recorded own-split values are close but not identical). The canonical
designation was confirmed downstream 2026-08-05: cascade 2's
tuned-vs-tuned heads contrast on recalibrated real-rock val tied
leaning all-rock, and the production heads
(`joint_heads_allrock_tuned_seed0`) now sit on this encoder. The standing
trigger: every new canonical encoder (the #15 pretext cascade -- re-probe,
re-tune head, re-confirm ADR 007) **re-scores on these axes in the same
pass**. When an axis stabilizes into a criterion worth fixing, it
graduates: candidate (issue) -> provisional (here) -> locked (ADR).

### Re-score 2026-07-28: the all-rock + redox encoder rides as-is

787,721-row all-rock corpus, redox token, ADR 007 HPs ridden as-is; cross-seed
spread measured at 0.002-0.006 (seed1: qap MLP 0.765, supergroup MLP 0.812).
Igneous-probe deficit vs the previous canonical encoder is consistent across
seeds but measured under the previous corpus's optimization HPs -- the
Strategy-A re-HPO is deferred to #55. Conclusion (owner-approved): even
without re-tuning the encoder's HPs for the new corpus, we barely took a loss
to performance, if any at all (probe deltas at or inside the noise band,
imputation a tie), while gaining a ~4.7x broader all-rock training set and
therefore, in principle, a richer latent representation -- with the
rock_group axis and redox slot as its first realized evidence.

### Information / decodability -- how much is recoverable

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| trace imputation | median log-R2, 13 traces | 0.832 (prev 0.836) | ~0 (mean-fill) -> 0.877 (latent-XGB) -> 0.892 (input-XGB); gates stamped to the previous corpus, not re-run | provisional | #8, [trace_signal_in_latent](../explorations/archive/trace_signal_in_latent/README.md) |
| major imputation | median log-R2, majors | 0.894 (prev 0.891) | ~0 (mean-fill) -> input-XGB per-major ceiling | provisional | #8, [imputation_head_design](../explorations/archive/imputation_head_design/README.md) |
| qap decodability | probe acc (MLP / logreg) | 0.759 / 0.704 (prev 0.769 / 0.722) | 0.100 (random) -> 0.220 (majority) -> **0.787 / 0.681** (raw-assay probe) | provisional | probe eval, [input_probe_ceiling](../explorations/archive/input_probe_ceiling/README.md) |
| supergroup decodability | probe acc (MLP / logreg) | 0.814 / 0.731 (prev 0.821 / 0.749) | 0.151 (random) -> **0.813 / 0.659** (raw-assay probe) | provisional | probe eval, [input_probe_ceiling](../explorations/archive/input_probe_ceiling/README.md) |
| rock_group decodability | probe acc (MLP / logreg), full all-rock test split | 0.919 / 0.894 | 0.810 (majority: always-igneous) -> *TBD* | provisional | probe eval |
| redox imputation | log-R2, redox slot (never in trace medians) | 0.306 | ~0 (mean-fill) -> *TBD* | provisional | imputation readout |
| redox token lift | populated-minus-padded qap MLP on measured-pair rows | +0.015 (0.774 populated) | 0 (token ignored) -> *TBD* | provisional | probe eval, #46 |

Each encoder scores with its own sweep-tuned read-out head (the head re-tunes
per encoder so the axis measures the latent, not a stale head). The
latent-XGB / input-XGB gates were measured on the previous corpus and are not
yet re-run on the all-rock corpus, so the 95% / 94% recovery framing carries
over only approximately. The probe ceilings are now measured, and the latent
sits **below** both: on qap the same MLP reads 0.787 from the raw assay against
the latent's 0.759, and 0.834 against 0.792 when both arms are tuned under one
search space. `qap_name` is computed from LOI-free major-element chemistry
(Gard et al. 2019, Sect. 4.2/4.4), so this axis measures how deterministic the
label is at least as much as how organized the latent is -- do not cite it as
evidence of latent organization. What the latent does add is *linear*
accessibility (0.704 vs 0.681 on qap, 0.731 vs 0.659 on supergroup); since a
linear readout is invariant to linear reparameterization, no linear
representation of the assay -- PCA included -- can exceed the raw logreg
number, so the linear leg of #19's linear-null bracket is settled without
running PCA. rock_group is
the one axis scored on the full all-rock test split rather than the shared
roster (the roster is all-igneous, where the probe's 0.982 is merely igneous
recall); per-class recall runs igneous 0.97 / sedimentary 0.82 / metamorphic
0.51 -- metamorphic chemistry overlaps its protoliths, so group membership is
genuinely underdetermined from composition there.

### Nonlinearity premium -- does the learned nonlinearity earn its keep

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| encoder vs linear, linear readout | qap logreg, latent minus any linear representation | **+0.023** (0.704 vs 0.681) | raw-assay logreg bounds every linear representation -> encoder | provisional | [input_probe_ceiling](../explorations/archive/input_probe_ceiling/README.md), #19 |
| encoder vs linear, nonlinear readout | imputation + MLP-probe delta over PCA | *TBD* | linear-PCA baseline -> encoder | candidate | #19 |

The decisive "is the transformer worth it over a linear learned representation"
test. Without it, "beats raw-input ceiling" does not separate *learning* from
*nonlinearity*. The linear-readout leg needs no PCA run to settle: a linear
probe is invariant to linear reparameterization of its input, so a linear probe
on *any* linear map of the assay (PCA included) is bounded by the raw-assay
logreg, and the latent clears it by 0.023 on qap and 0.071 on supergroup. Only
the nonlinear-readout leg -- PCA features under an MLP probe, and PPCA
imputation -- still needs the run #19 describes.

### Geometry / distance usability -- is the latent usable by distance

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| qap kNN purity | cosine k=5 | 0.577 (prev 0.582) | 0.100 (random) -> 0.607 (raw-input, previous-corpus measurement) | provisional | [input_knn_ceiling](../explorations/archive/input_knn_ceiling/README.md) |
| supergroup kNN purity | cosine k=5 | 0.670 (previous encoder; not re-measured this cascade) | 0.151 (random) -> 0.676 (raw-input) | provisional | [input_knn_ceiling](../explorations/archive/input_knn_ceiling/README.md) |
| distance-metric recoverability | post-hoc kNN lift toward probe | *TBD* | raw-distance kNN -> probe acc | candidate | #16 |

qap is ~96% of a data ceiling that is itself well below 1.0 (qap is largely not
kNN-separable from chemistry). The open question is how much of the probe-vs-kNN
gap a learned/whitened metric recovers without re-pretraining.

### Geometry health -- anisotropy

| axis | metric | current | ideal <- latent -> input | status | source |
|---|---|---|---|---|---|
| first-order (mean cone) | mean random-pair cosine | 0.266 (centered 0.003; prev 0.259) | 0.000 (isotropic) <- 0.266 -> 0.531 (raw_frac, all-rock) | provisional | [latent_anisotropy](../explorations/archive/latent_anisotropy/README.md) |
| second-order (shape) | effective rank / d | 0.191 (48.9/256; top-10 var 0.546; prev 0.125) | 0.996 (isotropic) <- 0.191 -> 0.101 (raw_frac, all-rock) | provisional | [latent_anisotropy](../explorations/archive/latent_anisotropy/README.md) |

The encoder removes the compositional input cone (first-order, harmless to
kNN); the all-rock latent spreads variance over ~49 effective dims, up from
the previous canonical encoder's ~32 (no catastrophic collapse, still far
from isotropic). Anisotropy scores each encoder on its own corpus's test
split, so the prev comparison mixes encoder and population. The
before-number for any future anti-collapse objective.

### Interpretability -- is the structure chemically meaningful

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| axes <-> geochemical trends | R2 of top PCs vs SiO2 / Mg# / alkalis | *TBD* | 0 (no correlation) -> 1 (axis is the index) | candidate | #17 |

### Robustness / transfer -- does it hold off-distribution

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| OOD detection | AUROC (Mahalanobis / recon-error) | *TBD* | 0.5 (chance) -> 1.0 | candidate | #18 |
| transfer to real-rock corpora | kNN purity of real-rock labels | *TBD* | random -> in-corpus equivalent | candidate | #18 |

### Downstream utility -- the actual product

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| head task performance | ModeHead / CondHead / CompHead NLL + calibration | measured per version | marginal/prior baseline -> oracle | provisional | `results/ledger.yaml` (corpus), `results/real_rock_ledger.yaml` + journal 2026-08-03 (real rocks) |

Track B landed; head performance is tracked per version in the ledgers rather
than as one scorecard number, and the real-rock benchmark is the cross-era
judge (then-production `joint_heads_supportmag_seed0`: sealed-test joint NLL 0.478
after recalibration, vs 0.769/1.371 for the traced-corpus arms). The
multimodal-posterior risk resolved in the corpus's favor: the latent supports
it, and the surviving gap is output-family width (single Dirichlet vs the
measured bimodality), not latent capacity.

## Capabilities beyond mineralogy

The SSL-pretrained encoder is a **rock-chemistry foundation model**, not
just a feature extractor for the mineralogy task. Trained via masked-element
modeling on unlabeled bulks from `data/bulks/gard2019.csv` (~167k igneous in
the 22-token generation, ~788k all-rock in the 23-token generation), it
organizes rocks in a latent space that captures real geochemical
structure. Mineralogy is one downstream application. This section enumerates
the others -- capabilities the existing encoder already supports with no further
training, or with tiny supervised heads on the pretrained latent.

**Canonical encoder:** `checkpoints/pretrain_allrock_redox_seed0`
(all-rock, 23-token with the Fe-redox ratio; accepted 2026-07-28, adopted
as canonical 2026-08-05 on the cascade-2 tuned-vs-tuned verdict), riding
the ADR-007 objective and hyperparameters.  The production heads
(`joint_heads_allrock_tuned_seed0`) sit on it; the earlier igneous
checkpoint `pretrain_v3_logmse_std_seed0` is the previous canonical
encoder.  Current probe numbers and acceptance record live in
the scorecard section above.  Capability numbers below were measured on the encoder
generation named in each entry.

Update this section as capabilities are built or evaluated.

### What the pretrain head + pooled latent already give you

These run on the **frozen** encoder with no encoder retraining. k-NN (#3)
and OOD (#4) need no training at all; element imputation (#1) and the anomaly
flagging built on it (#2) add a small read-out head trained in minutes on the
frozen latent (same cost class as the heads in the next section).

#### 1. Element imputation from partial analyses

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
trace-imputation gate (`explorations/archive/trace_signal_in_latent/`, JOURNAL
2026-06-15) showed this is a *head-objective* problem, not encoder capacity --
XGBoost on the frozen masked latent reaches ~94% of the input-element XGBoost
ceiling across the 13 traces. So the fix is the right objective on the
*existing* frozen latent, not a re-pretrain: the SSL objective itself moved to
standardized log-space (ADR 007), and this separately-tuned read-out head reads
the frozen latent at the imputation task's own capacity.

**Shipped.** The production head is Optuna-tuned (single-stage sweep, objective
median trace log-R2 on a held-out val split); on the igneous encoder it
reached test median trace log-R2 0.836 = 95% of the 0.877 frozen-latent
yardstick (JOURNAL 2026-06-24), and the re-tune on the all-rock encoder ties
at 0.832 (acceptance 2026-07-28; the 0.877/0.892 gates are stamped to the
previous corpus, not re-run).
`configs/imputation/imputation_head_allrock_redox.yaml` regenerates it on the
canonical encoder (`imputation_head.yaml` for the igneous one); the
checkpoint + readout artifacts are gitignored (regenerable).

Implementation (built):
- `cnmineral.models.imputation_head.LogImputationHead` -- linear or
  one-hidden-MLP read-out emitting per-element standardized `log10(frac)`,
  with de-standardization buffers carried in `state_dict`.
- `cnmineral.losses.pretrain_loss.masked_element_log_mse` -- the standardized
  log-space objective, shared by the SSL pretrain and this read-out head.
- `scripts/training/imputation_head.py` -- the single entry, mirroring
  `pretrain.py`: a plain `--config` run trains the head on a frozen encoder
  (`train_imputation_head`) then scores the test split once, writing the
  log-space scorecard to `runs/imputation/<encoder_tag>/readout/`; `--hpo`
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

#### 2. Per-element anomaly / measurement-error flagging

Same machinery, run as an outlier detector: predict each element from all
the others, compare to reported value, flag standardized residuals beyond
a threshold. Could be deployed as a "sanity check" pass on incoming user
analyses.

Effort: ~half day, mostly UX (deciding thresholds, presentation).

#### 3. k-NN similarity lookup against 1M+ bulks

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

#### 4. Out-of-distribution flagging

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

### Capabilities that need a tiny supervised head

These add a small classification or regression head on top of the
pretrained latent. Trained in minutes (logistic regression or 1-hidden
MLP); no encoder retraining required.

#### 5. Tectonic-setting classification

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

#### 6. Magma-series classification

Tholeiitic / calc-alkaline / alkaline. Targets computable from majors +
alkali content via the standard diagrams (Irvine & Baragar 1971 AFM,
Miyashiro 1974, Le Bas 1986 TAS-derived). Small classifier on the latent.

Effort: ~half day after #5's infrastructure is in place.

#### 7. Petrogenetic-ratio regression with uncertainty

Predict named, interpretable ratios from the latent: Sr/Y, La/Yb, Nb/Y,
Th/Yb, Mg#, K/Rb, Ba/La, Eu/Eu* (Eu anomaly), etc. These are computable
deterministically from input chemistry, but a regression head that returns
**predicted value + uncertainty** has real value as a sanity check or for
incomplete-input scenarios.

Effort: ~1 day. Multi-output regression, Bayesian last layer for
uncertainty if desired.

#### 8. Differentiation index / position on liquid line of descent

Continuous "how evolved is this magma" scalar. Loosely correlated with
Mg# but more nuanced (accounts for residual Th, K, Rb buildup, REE
patterns). Useful for batch comparison of cogenetic samples.

Effort: ~1 day. Mostly: defining the index labels (e.g., from existing
LLD calculators in pyrolite).

#### 9. Isotopic-ratio prediction

`complete.csv` has columns for `epsilon_Nd`, `epsilon_Sr`, `epsilon_Hf` on
a non-trivial fraction of rocks. Regression head on the latent; for bulks
where isotopes weren't measured (the majority), get predicted values plus
uncertainty.

**Genuine new value** -- isotope measurements are expensive (~$200+/sample).
An honest predicted value with uncertainty is useful for screening,
exploration, regional studies.

Effort: ~2 days. Need to filter to bulks with isotope data, train, validate
against held-out measured isotopes.

#### 10. Approximate age regression

Many bulks in `complete.csv` have rough age info (numeric or geological era
binned). Chemistry-from-age isn't strongly determined for most rocks, but
distinctive populations (Archean greenstones, Proterozoic granites, Cenozoic
arc rocks) do have characteristic signatures. A regression head with **large
honest uncertainty bars** is the right framing.

Practically more useful as a coarse classifier (Archean / Proterozoic /
Phanerozoic) than a continuous regression.

Effort: ~1-2 days including the binning and honest-uncertainty modeling.

#### 11. Geographic / craton-of-origin inference

Crustal regions have characteristic rock populations. A classifier could
surface "rocks like this most commonly come from [region]." Less rigorous
but interesting for prospecting / exploration / provenance contexts.

Effort: ~1-2 days. Geography labels in `complete.csv` would need cleaning;
target should probably be at the craton or geological-province level, not
country.

### Capabilities the encoder does NOT support (without further work)

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

### Prioritization for a "geochemistry copilot" app

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

### Relationship to the mineralogy track

Both tracks share the same encoder. The mineralogy track (`v1.0-mode-cond`,
v2 cell-c) adds the ModeHead + CondHead and trains supervised against FC
labels. The applications track in this doc adds zero or one small head
per capability without retraining the encoder.

Both tracks benefit if the next encoder iteration is stronger (e.g., from
per-token-normalized loss, larger pretrain corpus, longer training).
Improvements to the encoder propagate to both tracks; improvements specific
to the mineralogy heads or specific to the applications heads do not
interfere with each other.
