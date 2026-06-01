# Fresh-codebase implementation handoff

A from-scratch reimplementation guide. Crystallizes the value proposition,
architecture, training data, and what to carry forward from the v2 codebase.

Living doc; written 2026-06-01 after a long debugging arc converged on a
stable design. Companion to `MODEL.md` (current-state architecture reference)
and `PRETRAIN_APPLICATIONS.md` (the SSL-encoder capability list).

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

2. **Distribution not point estimate.** With Dirichlet ModeHead + GMM
   CondHead, the within-bulk variance across FC paths becomes the trained
   uncertainty, not noise to average out. The 100-path degeneracy
   visualization (see `data/diagnostics/degeneracy_viz/`) makes this
   concrete: same bulk, different (P, H2O), modal abundance varies up to
   100% relative for hydrous phases. The right output is a distribution.

3. **Foundation-model encoder.** SSL pretraining on 474k real bulks
   organizes the latent by petrogenetic structure WITHOUT MAGEMin labels.
   Measured: 76% supergroup classification accuracy via MLP probe with
   zero supervision; 0.975 cosine on modal vectors via k-NN in latent vs
   0.813 for random neighbors. The encoder is the durable contribution
   that survives any specific generator's biases. Heads are application
   slots; the encoder is the substrate.

## Should you keep CrabNet (kingcrab.py) or roll your own?

**Verdict: keep mat2vec, write your own encoder.**

What CrabNet gives:
- mat2vec embeddings (a 200-dim chemistry-aware vector per element, learned
  from a materials-science text corpus). Genuinely valuable; not trivial
  to replicate. Keep the CSV.
- FractionalEncoder (discretizes continuous `frac` into indexed lookups).
  Cute but not essential; an MLP on continuous `frac` works.
- TransformerEncoder + padding convention. **Drop this.** The padding
  convention (`frac > 0` means populated) collided with our SSL mask token
  and cost a full day of debugging plus an entire round of invalidated
  ablation results. Owning the padding semantics matters.

What to write yourself (~1 day):

```python
class Encoder(nn.Module):
    def __init__(self, d_model=512, N=4, heads=16, n_tokens=22):
        self.elem_emb = nn.Embedding.from_pretrained(mat2vec_matrix, freeze=False)
        self.frac_proj = nn.Linear(1, d_model)
        # Or keep CrabNet's discretization if you like it
        self.element_proj = nn.Linear(200, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, N)

    def forward(self, src, frac):
        # src: [B, 22] atomic numbers, 0 = padding
        # frac: [B, 22] values in [0, 1]
        pad_mask = (src == 0)  # EXPLICIT, not derived from frac
        token = self.element_proj(self.elem_emb(src)) + self.frac_proj(frac.unsqueeze(-1))
        h = self.transformer(token, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).float().unsqueeze(-1)
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # masked-mean pool
```

That's the whole encoder. Padding mask derived from `src` only; SSL masking
zeros both `src` and `frac` to make masked = padding indistinguishable, as
in the v2 bug-fix design (Option H). No accidental gradient holes.

## Input / output / architecture sketch

```
INPUTS (per sample)
  src  : [22]  atomic numbers, 0 = absent (padding)
  frac : [22]  values in [0, 1] (majors: wt%/100, traces: ppm/1e6)

  V3_CATIONS_ORDER (22 tokens; ordering fixed):
    majors (9): Si Al Ca Mg Fe K Na Ti Cr
    traces (13): Sr Zr Ba Y Rb Ni Nb V La Ce Nd Yb Th

ENCODER (shared across all heads)
  element embedding (mat2vec)  -> per-token vector
  + frac projection             -> per-token vector
  -> Transformer encoder (d_model=512, N=4, heads=16)
  -> masked-mean pool over non-padding tokens
  -> latent h : [B, 512]

HEADS (parallel, all read the same h)
  ModeHead    : h -> Dirichlet alpha over 18 phases       [B, 18]
  CompHead    : h -> per-phase composition                [B, 18, K]
                K = 11 (oxide-wt%) OR per-phase endmembers (pilot 2b decides)
  CondHead    : h -> GMM per axis (T, P, H2O)             [B, 3, K_gmm, 3]
                last dim = (mean, sigma, weight) per component
  ClassHead   : h -> supergroup logits (auxiliary)        [B, 9]
  G_sys head  : h -> scalar G_sys regression (auxiliary)  [B, 1]
```

ClassHead and G_sys head are auxiliary: they regularize the encoder by
adding gradient signals along petrogenetically meaningful axes. They are
not deployment outputs. Both can be turned off without changing inference.

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

**Critical**: the current production corpus DISCARDED G_sys during FC. The
column doesn't exist in `data/train_fc_v1.parquet`. A G_sys head requires
either (a) the v3 corpus regen to persist `res["sys"]["G"]` (or equivalent
PTT key), or (b) post-hoc re-extraction by re-running FC on each bulk just
to grab the scalar -- expensive but doable.

**Don't add G_sys in v3 if it would delay shipping.** It's a Tier 2 head,
worth a section in the regen if it's cheap, but not blocking.

## Training data

Three corpora, all generated upstream of the model:

1. **SSL pretrain corpus**: `data/complete.csv`, igneous-filtered to ~474k
   bulks. Inputs only (no labels). Used for masked-element pretraining.

2. **Supervised corpus (v3)**: per-bulk × N FC paths with per-path
   (T_FC_end, P, H2O, modal_18, per-phase composition, supergroup label,
   optional G_sys). Generated via the v3 corpus regen that's behind the
   pilot tests in `~/.claude/plans/sunny-weaving-nebula.md`. Approximate
   shape: 22k bulks × 8 paths = 178k rows for the supervised loss.

3. **Held-out evaluation corpora**: real-rock targets (mineralML grains
   for CompHead evaluation; EDS maps if available) -- never used in
   training, only for cross-modal validation.

Critical schema requirements for the v3 corpus regen, beyond what v1 has:

- per-phase composition columns (`<oxide>_<phase>`, cumulate mass-weighted
  across FC steps; see `scripts/experiments/visualize_degeneracy.py` for
  the extraction logic that figures out the PTT short-code mapping and
  the FC-cumulate vs FC-snapshot distinction)
- optional `G_sys` scalar per path
- optional per-phase endmember mole fractions (pilot 2b decides whether
  to use these as CompHead targets instead of oxide-wt%)
- bulk_oxide_renormalized column for the mass-balance loss term

## Pretraining (the SSL mechanic)

Masked-element modeling, pooled-latent decoder (Option H from the
v2 bug-fix). The convention that worked:

```
For each batch:
  pick 25% of populated tokens per sample, uniformly at random
  zero BOTH src and frac at those positions (masked = padding, indistinguishable)
  pass the corrupted (src, frac) through the encoder
  decoder MLP on the pooled latent predicts frac for EVERY position
  MSE loss only at originally-masked positions
```

Decoder shape: `d_model -> d_model//2 -> n_tokens=22`. Output is a vector
of predicted frac values for all 22 positions; loss reads off the masked
ones only. The non-masked predictions are unsupervised side outputs --
they don't get a loss because they'd just be teaching the model to
reproduce its own input.

Class-balanced sampling by supergroup during SSL prevents basalt
over-fitting (complete.csv is ~50% mafic-volcanic).

## What to bring forward from the v2 codebase

- `src/cnmineral/data/element_properties/mat2vec.csv` (the embedding matrix)
- `src/cnmineral/data/fe_redox.py`, `tp_priors.py`, `h2o.py` (the strategy
  modules for Fe3+/H2O/P sampling priors -- ADRs explain why these are
  injectable, not hardcoded)
- `scripts/production/_phase_map.py` PLUS the `_ALL_SHORT_TO_SHORT` mapping
  added in `scripts/experiments/visualize_degeneracy.py` (you'll need both
  when extracting per-phase composition from PTT)
- `data/splits_v1.json` if reusing v1 split; rebuild if changing input
  vocab or filter rules
- `data/min_df_v1.csv` (mineralML grain corpus) for held-out CompHead eval
- ADRs 004, 005, 006 from `docs/decisions/` (label-design, corpus, model arch)

## What to drop

- `src/cnmineral/kingcrab.py` (write your own encoder; padding bug isn't
  worth carrying)
- The dual `mass%_<phase>` interpretation (fraction-vs-percent ambiguity
  that bit the degeneracy viz). Standardize: all masses internal are
  fractions in [0, 1] that sum to 1; convert to wt% only at display time
- The per-mineral wt% representation as a hardcoded constant. Make the
  phase vocabulary a YAML config, not a Python list

## What's deferred (do not pre-build)

These are scoped in the plan but should NOT be implemented before pilots
prove their value:

- ALIGNN structural stream (v2 deferral; still deferred)
- Polymorph head (Ky/Sil/And, Qz/Crs/Trd) -- needs explicit polymorph
  labels that MAGEMin doesn't provide
- fO2 sweep (currently class-conditional via fe_redox; sweeping is v3+)
- TracePartitionHead (the 4th head from the trace-usage design taxonomy;
  separate corpus-gen requirement around Kd database lookups)

## The minimal-shippable v3 spine

If you want a tight scope for the first fresh-codebase pass:

1. Encoder (your own thin transformer + mat2vec)
2. ModeHead (Dirichlet over 18 phases)
3. CompHead (oxide-wt%, simplest version; upgrade after pilot 2b)
4. CondHead (GMM per axis)
5. ClassHead (supergroup CE; auxiliary, cheap, latent-shaping)
6. SSL pretraining via MEM with the Option H pooled-latent decoder
7. Mass-balance soft loss (`λ * ||bulk - Σ M × C||²`)
8. K=5 ensemble after the single-seed ablation passes

Add later: G_sys head, TracePartitionHead, ALIGNN, polymorph head.

## The single most important code invariant

```
padding mask is derived from `src == 0` ONLY.
masked-for-SSL means BOTH src AND frac are zeroed.
masked-for-SSL is therefore indistinguishable from padding.
```

If you remember nothing else from the v2 codebase, remember that one. It's
the difference between "SSL works" and "SSL silently has zero encoder
gradients for a week."
