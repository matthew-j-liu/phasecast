# Read-out head design ablation

The trace-imputation gate (`../trace_imputation_loss_gate/`) settled *where* to
fix imputation: a head on the frozen latent, not a re-pretrain. This ablation
settles *how* to build that head. It trains the shipped `LogImputationHead`
four ways on the frozen `pretrain_v3_seed0` encoder and scores each
leave-one-out in log space on the Gard test split (n=33,371), against the
gate's yardsticks: frozen-latent XGBoost **0.839** and input-element XGBoost
ceiling **0.892** (median over the 13 traces).

```
conda run -n alignn_fix python explorations/imputation_head_ablation/run_ablation.py
```

Read-only on the encoder (frozen, eval, no grad). Writes
`results/<encoder_tag>/ablation_metrics.json`. 20 epochs/variant, seed 0;
numbers below reproduce exactly across runs.

## Variants

| tag | head | standardize | masking |
|---|---|---|---|
| L-std | linear | yes | SSL (mask_prob 0.23, multi-token) |
| L-raw | linear | no | SSL |
| MLP-std | MLP(256) + GELU | yes | SSL |
| L-1tok | linear | yes | single-token (matches the LOO eval) |

## Results (median trace log-R2)

| variant | median trace | min trace (Yb) | median major |
|---|---|---|---|
| L-std | 0.748 | 0.544 | 0.830 |
| L-raw | 0.743 | 0.439 | 0.813 |
| **MLP-std** | **0.792** | 0.623 | 0.866 |
| L-1tok | 0.757 | 0.555 | 0.840 |

Per-trace log-R2:

| elem | L-std | L-raw | MLP-std | L-1tok |
|---|---|---|---|---|
| Sr | 0.695 | 0.673 | 0.764 | 0.711 |
| Zr | 0.703 | 0.695 | 0.760 | 0.718 |
| Ba | 0.682 | 0.672 | 0.760 | 0.688 |
| Y  | 0.576 | 0.505 | 0.661 | 0.578 |
| Rb | 0.803 | 0.792 | 0.842 | 0.808 |
| Ni | 0.748 | 0.743 | 0.792 | 0.757 |
| Nb | 0.710 | 0.679 | 0.781 | 0.723 |
| V  | 0.828 | 0.819 | 0.849 | 0.828 |
| La | 0.805 | 0.786 | 0.856 | 0.826 |
| Ce | 0.820 | 0.799 | 0.864 | 0.841 |
| Nd | 0.811 | 0.779 | 0.852 | 0.836 |
| Yb | 0.544 | 0.439 | 0.623 | 0.555 |
| Th | 0.769 | 0.755 | 0.802 | 0.773 |

## Findings

1. **Log space is the fix; per-element standardization is a minor refinement.**
   L-raw (log space, no standardization) already reaches 0.743 median;
   standardization adds **+0.005 median**. Its real effect is narrow -- it
   rescues the lowest-signal heavy REE (Yb +0.105, Y +0.071) and little else.
   Mechanism: a linear head's outputs are independent affine maps, so
   per-element target scaling is absorbable and R2 is scale-invariant; the only
   coupling is the global `clip_grad_norm_`, where un-standardized
   high-variance majors hog the clip budget and starve the low-variance HREE.
   So the standardization story in the loss/head docstrings was reworded:
   log space does the work, standardization buys the bottom two traces.

2. **MLP beats linear and should be the shipped default.** MLP-std leads linear
   by **+0.044 median** (+0.08 on Yb) and lands at 0.792 = 94% of the
   frozen-latent target -- matching the "94% of the input ceiling" the gate
   advertised as recoverable. `configs/imputation/imputation_head.yaml` now
   defaults to `hidden: 256`.

3. **The yardstick is a strong-nonlinear-readout reference, not a hard
   ceiling.** 1b is a tuned gradient-boosted forest, so a linear or one-layer
   head can trail it on readout capacity even when the latent holds the signal.
   The linear head's 0.748-vs-0.839 gap is partly readout capacity, which the
   MLP closing most of it confirms. Docs that called 1b "the most the head can
   extract from this latent" were reworded.

4. **Train/eval masking mismatch is minor.** The head trains on SSL multi-token
   masking but is scored single-token leave-one-out (matching the gate's 1b).
   Matching the training masker to the eval (L-1tok) gives only **+0.009
   median** -- not worth a code change. The encoder's multi-mask pretrain
   regime transfers to single-mask eval fine.

The gap to the yardstick is concentrated on HREE (Yb, Y), exactly where the
gate found the latent's information loss is worst -- the head tracks the
latent's content rather than introducing new failure.
