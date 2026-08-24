# latent_anisotropy

## Question

`input_knn_ceiling` concluded the weak cosine-kNN qap purity is "not an
anisotropy artifact the SSL objective introduced" -- but only indirectly (the
same kNN gap exists in the raw input). Nobody has measured the canonical
latent's anisotropy head-on. How anisotropic is it, in both senses, and does the
encoder introduce a pathological geometry or remove one?

Isotropic = looks the same in every direction. Two independent departures:

- **first-order (mean / cone):** the cloud sits off-origin in a narrow cone -- a
  shared dominant direction. Seen by cosine, removed by centering. Scored by the
  mean random-pair cosine and the mean-vector dominance `||mean(X)||/mean(||x||)`.
- **second-order (shape):** the covariance ellipsoid is elongated -- variance
  spread unevenly across directions. Computed on centered data, blind to the
  mean. Scored by the covariance spectrum's effective rank (Roy-Vetterli
  entropy) / participation ratio; dimensional collapse is its extreme tail.

## Method

Reproduce the encoder's exact test split (`reserve_test(ckpt.config)`, 33,371
rows) and compute both facets on four matrices: the `latent` [N, 256], the two
raw-input reps reused from `input_knn_ceiling` (`raw_frac`, `std_log`, [N, 22]),
and an isotropic Gaussian reference [N, 256] as the perfectly-isotropic yardstick
at matched N, d. Spectrum is of the covariance (not correlation) -- the geometry
the cosine / euclidean distance metrics actually operate on. Effective rank is
also reported as a fraction of `d` so the 256-d latent and 22-d inputs compare.

    conda run -n mineral python -m explorations.archive.latent_anisotropy.run_anisotropy

## Result

On the canonical encoder (`pretrain_v3_logmse_std_seed0`):

first-order (mean / cone):

| space | pair cosine | pair cosine (centered) | mean-vec dominance |
|---|---|---|---|
| latent | 0.259 | 0.003 | 0.488 |
| raw_frac | 0.938 | 0.013 | 0.969 |
| std_log | 0.022 | 0.021 | 0.003 |
| isotropic_ref | 0.000 | -0.000 | 0.006 |

second-order (covariance shape):

| space | eff. rank | eff. rank / d | top-1 var | top-10 var |
|---|---|---|---|---|
| latent | 32.0 / 256 | 0.125 | 0.199 | 0.653 |
| raw_frac | 2.8 / 22 | 0.126 | 0.713 | 1.000 |
| std_log | 9.1 / 22 | 0.415 | 0.346 | 0.886 |
| isotropic_ref | 255.0 / 256 | 0.996 | 0.005 | 0.046 |

Two readings:

- **First-order: the encoder removes the compositional cone, doesn't add one.**
  Raw compositional input is almost degenerate (`raw_frac` pair cosine 0.94 --
  everything in the positive orthant). The latent's residual cone is far milder
  (0.26), and centering drives it to ~0 (purely first-order). Independently,
  `input_ceiling.json` has latent cosine-kNN >= euclidean-kNN everywhere, and
  euclidean is mean-invariant -- so this residual cone is demonstrably not the
  kNN bottleneck. The "not an artifact" claim now has a direct number: the latent
  is ~4x less cone-shaped than its own input.

- **Second-order: this is where the latent is genuinely anisotropic, but not
  collapsed.** ~32 of 256 directions carry the variance (12.5% of the budget;
  top-10 components hold 65%). Effective rank 32 >> 1, so no catastrophic
  dimensional collapse, but far from the isotropic 255/256. In *fractional* terms
  the latent (0.125) concentrates about as tightly as `raw_frac` (0.126) and more
  than `std_log` (0.415); in *absolute* terms it spreads chemistry over more
  directions (32 vs ~3-9) than either input.

## Decision

Recorded baseline, no action. The canonical latent is moderately cone-shaped
(first-order, shown harmless to kNN) and uses ~12.5% of its 256-dim budget
(second-order), with no catastrophic collapse. The SSL objective de-anisotropizes
the compositional input rather than introducing a pathological geometry. If issue
#15 ever pursues an anti-collapse (VICReg-style variance/covariance) objective,
effective rank 32/256 is the before-number it would aim to raise. Canonical
verdict in `docs/journal.md` (2026-06-25).
