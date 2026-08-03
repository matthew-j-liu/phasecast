# input_knn_ceiling

## Question

The canonical encoder decodes rock type well but arranges it weakly: qap MLP
probe 0.769, yet raw-latent kNN qap purity only 0.582 (k=5) vs 0.100 random.
Issue #15 asks whether a new SSL pretext (group-masking, pull-together) could
make the latent *segregate by qap*. Before spending training hours, what is the
prize -- how separable is qap by raw chemistry at all? Some qap splits are not in
our input (basalt vs gabbro are the same chemistry, differing only in
cooling-rate texture the encoder never sees), so there is an unknown ceiling
below 1.0 that no encoder can pass.

## Method

Run the identical kNN purity (`cnmineral.eval.probe.primitives.knn_label_purity`)
on the **raw 22-element input** instead of the latent, reproducing the
encoder's exact test split via `reserve_test(ckpt.config)`. Two raw
representations: `raw_frac` (literal abundances) and `std_log` (per-element
standardized `log10(frac)`, the model's own input space). Both metrics
(cosine, euclidean), labels qap_name + supergroup, alongside the latent's own
purity. The achievable ceiling is the best raw-input number; the geometry
analogue of the imputation work's raw-input XGBoost ceiling.

    conda run -n alignn_fix python explorations/input_knn_ceiling/run_ceiling.py

## Result

kNN purity at k=5 on the canonical encoder's 33,371-row test split
(latent reproduces 0.582 exactly, confirming the split was rebuilt correctly):

| label | latent (cosine) | best raw-input | headroom |
|---|---|---|---|
| qap_name | 0.582 | 0.607 (raw_frac) | +0.026 |
| supergroup | 0.670 | 0.676 (raw_frac) | +0.006 |

Two notes:
- `std_log` is *worse* (qap 0.527), not better: weighting all 22 elements
  equally dilutes qap, which is a major-element story (silica tier, feldspar
  ratios). `raw_frac`, dominated by the majors, is the better ceiling. The latent
  sits between the two -- the encoder weights traces more than a pure-major
  distance would (it was trained to reconstruct them), costing the last ~0.025 of
  qap-kNN purity.
- The same decode-vs-cluster gap (a classifier reads qap far above 0.61, kNN
  tops out at 0.61) exists in the *raw input*, so the gap is intrinsic to the
  task, not an anisotropy artifact the SSL objective introduced.

## Decision

Raw-input qap kNN tops out at ~0.61 and the latent is already at 0.582 (~96% of
the ceiling). qap is largely not kNN-separable from bulk chemistry, and the
encoder already mirrors that faithfully. **Reconstruction-family pretexts
(group-masking, #3) are bounded near this ceiling -- don't chase qap-kNN with
them.** The headroom is ~0.025. A contrastive / pull-together objective is the
only thing that could exceed a fixed-metric input ceiling, but only if its
positive pairs carry a target the raw chemistry lacks (tectonic setting, FC
trajectory) -- which reframes #15 away from qap-kNN.

This tool generalizes: point `run_ceiling.py` at any future target to screen its
raw-input ceiling before committing to training. Canonical verdict + the full
reasoning chain are in `docs/journal.md` (2026-06-25).
