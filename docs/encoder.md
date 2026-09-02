# The encoder: foundation model, data, and value proposition

The canonical reference for the shared substrate: the value proposition, the
SSL-pretrained encoder, the tokenization and data flow, the three corpora, the
pretraining mechanic, and the single load-bearing code invariant. The encoder
is the durable contribution; the supervised heads that read its latent are a
separate concern.

Companion to `heads.md` (the supervised heads + posterior) and
`pretrain_applications.md` (the SSL-encoder capability list).

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
rung (see `docs/corpus_regen_spec.md` storage shape), so a G_sys head is
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
   `docs/scorecard.md`; guardrails in #24; JOURNAL 2026-07-23/28.

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
  across FC steps; see `scripts/experiments/visualize_degeneracy.py` for
  the extraction logic that figures out the PTT short-code mapping and
  the FC-cumulate vs FC-snapshot distinction)
- optional `G_sys` scalar per path
- optional per-phase endmember mole fractions (pilot 2b decides whether
  to use these as CompHead targets instead of oxide-wt%)
- bulk_oxide_renormalized column for the mass-balance loss term

### Strategy modules (corpus-gen priors)

The corpus-gen pipeline encodes petrologic priors as toggle-able strategies,
not hardcoded constants (`src/cnmineral/data/`). Mirror this pattern for any
new prior:

- `fe_redox.py` -- Fe3+/ΣFe ratio (Middlemost 1989 per-class defaults;
  ClassRatioStrategy + ConstantRatioStrategy)
- `tp_priors.py` -- (T, P) sampling (Putirka 2008 per-class ranges). The P axis
  carries forward into FC's (P, H2O) sampling; T sampling is replaced by
  per-bulk T_FC_end at which residual liq < 5%.
- `h2o.py` -- H2O wt% (Wallace 2005 + Plank 2013 + Burnham 1979 per-class
  ranges; ClassRangeH2OStrategy + ConstantH2OStrategy)

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
