# Encoder representation scorecard

A living, deliberately-provisional rubric for the *strength* of the encoder's
latent representation. It is not a gate and nothing here is locked -- it is the
synthesis layer over the individual eval investigations, kept in one place so
"is the representation strong?" is a constant, anchored question rather than an
occasional vibe.

## How to read it

- **Every axis carries a floor and a ceiling anchor.** A raw number is
  meaningless on its own; the signal is where the current value sits *between its
  floor (no information) and its ceiling (the most any method extracts)*. An axis
  without anchors does not belong here -- that is the lesson of
  [input_knn_ceiling](../explorations/input_knn_ceiling/README.md). The "fluff"
  trap is reporting a value with nothing to compare it to.
- **Status is a lifecycle, not a grade.** `candidate` = an axis we intend to
  measure (usually has an open issue); `provisional` = measured and anchored but
  the target/verdict is not fixed; `locked` = promoted to an ADR as a standing
  criterion. Axes earn their place as evidence lands; some may be dropped.
- **Do not optimize UMAP-2D appearance.** Rock chemistry is largely continuous,
  so clean 2D separation is partly the wrong target.

## Scope and refresh

Numbers below are for the canonical encoder `pretrain_v3_logmse_std_seed0`
(`epoch_best`) on its 33,371-row test split. The standing trigger: every new
canonical encoder (the #15 pretext cascade -- re-probe, re-tune head, re-confirm
ADR 007) **re-scores on these axes in the same pass**. When an axis stabilizes
into a criterion worth fixing, it graduates: candidate (issue) -> provisional
(here) -> locked (ADR).

## Information / decodability -- how much is recoverable

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| trace imputation | median log-R2, 13 traces | 0.836 | ~0 (mean-fill) -> 0.877 (latent-XGB) -> 0.892 (input-XGB) | provisional | #8, [trace_signal_in_latent](../explorations/trace_signal_in_latent/README.md) |
| major imputation | median log-R2, majors | 0.891 | ~0 (mean-fill) -> input-XGB per-major ceiling | provisional | #8, [imputation_head_design](../explorations/imputation_head_design/README.md) |
| qap decodability | probe acc (MLP / logreg) | 0.769 / 0.722 | 0.100 (random) -> *TBD* | provisional | probe eval |
| supergroup decodability | probe acc (MLP / logreg) | 0.821 / 0.749 | 0.151 (random) -> *TBD* | provisional | probe eval |

The head reaches 95% of what a strong nonlinear reader (XGBoost) pulls off the
frozen latent (0.836 / 0.877) and 94% of the absolute raw-input ceiling (0.892).
Probe ceilings are still floor-anchored only -- the missing ceiling is a raw-input
classifier, which #19 supplies.

## Nonlinearity premium -- does the learned nonlinearity earn its keep

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| encoder vs linear (PCA/PPCA) | imputation + probe delta over PCA | *TBD* | linear-PCA baseline -> encoder | candidate | #19 |

The decisive "is the transformer worth it over a linear learned representation"
test. Without it, "beats raw-input ceiling" does not separate *learning* from
*nonlinearity*.

## Geometry / distance usability -- is the latent usable by distance

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| qap kNN purity | cosine k=5 | 0.582 | 0.100 (random) -> 0.607 (raw-input) | provisional | [input_knn_ceiling](../explorations/input_knn_ceiling/README.md) |
| supergroup kNN purity | cosine k=5 | 0.670 | 0.151 (random) -> 0.676 (raw-input) | provisional | [input_knn_ceiling](../explorations/input_knn_ceiling/README.md) |
| distance-metric recoverability | post-hoc kNN lift toward probe | *TBD* | raw-distance kNN -> probe acc | candidate | #16 |

qap is ~96% of a data ceiling that is itself well below 1.0 (qap is largely not
kNN-separable from chemistry). The open question is how much of the probe-vs-kNN
gap a learned/whitened metric recovers without re-pretraining.

## Geometry health -- anisotropy

| axis | metric | current | ideal <- latent -> input | status | source |
|---|---|---|---|---|---|
| first-order (mean cone) | mean random-pair cosine | 0.259 (centered 0.003) | 0.000 (isotropic) <- 0.259 -> 0.938 (raw_frac) | provisional | [latent_anisotropy](../explorations/latent_anisotropy/README.md) |
| second-order (shape) | effective rank / d | 0.125 (32/256; top-10 var 0.653) | 0.996 (isotropic) <- 0.125 -> 0.126 (raw_frac) | provisional | [latent_anisotropy](../explorations/latent_anisotropy/README.md) |

The encoder removes the compositional input cone (first-order, harmless to kNN);
its variance concentrates in ~32 effective dims (no catastrophic collapse, but far
from isotropic). The before-number for any future anti-collapse objective.

## Interpretability -- is the structure chemically meaningful

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| axes <-> geochemical trends | R2 of top PCs vs SiO2 / Mg# / alkalis | *TBD* | 0 (no correlation) -> 1 (axis is the index) | candidate | #17 |

## Robustness / transfer -- does it hold off-distribution

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| OOD detection | AUROC (Mahalanobis / recon-error) | *TBD* | 0.5 (chance) -> 1.0 | candidate | #18 |
| transfer to real-rock corpora | kNN purity of real-rock labels | *TBD* | random -> in-corpus equivalent | candidate | #18 |

## Downstream utility -- the actual product

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| head task performance | ModeHead / CondHead / CompHead NLL + calibration | *TBD* | marginal/prior baseline -> oracle | candidate | Track B (#9) |

The heads are the application and the real remaining risk (whether ~32 effective
dims support a multimodal posterior). Assessed as Track B lands, not pre-emptively.
