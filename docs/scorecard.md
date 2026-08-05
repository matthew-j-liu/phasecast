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
  [input_knn_ceiling](../explorations/archive/input_knn_ceiling/README.md). The "fluff"
  trap is reporting a value with nothing to compare it to.
- **Status is a lifecycle, not a grade.** `candidate` = an axis we intend to
  measure (usually has an open issue); `provisional` = measured and anchored but
  the target/verdict is not fixed; `locked` = promoted to an ADR as a standing
  criterion. Axes earn their place as evidence lands; some may be dropped.
- **Do not optimize UMAP-2D appearance.** Rock chemistry is largely continuous,
  so clean 2D separation is partly the wrong target.

## Scope and refresh

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

## Re-score 2026-07-28: the all-rock + redox encoder rides as-is

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

## Information / decodability -- how much is recoverable

| axis | metric | current | floor -> ceiling | status | source |
|---|---|---|---|---|---|
| trace imputation | median log-R2, 13 traces | 0.832 (prev 0.836) | ~0 (mean-fill) -> 0.877 (latent-XGB) -> 0.892 (input-XGB); gates stamped to the previous corpus, not re-run | provisional | #8, [trace_signal_in_latent](../explorations/archive/trace_signal_in_latent/README.md) |
| major imputation | median log-R2, majors | 0.894 (prev 0.891) | ~0 (mean-fill) -> input-XGB per-major ceiling | provisional | #8, [imputation_head_design](../explorations/archive/imputation_head_design/README.md) |
| qap decodability | probe acc (MLP / logreg) | 0.759 / 0.704 (prev 0.769 / 0.722) | 0.100 (random) -> *TBD* | provisional | probe eval |
| supergroup decodability | probe acc (MLP / logreg) | 0.814 / 0.731 (prev 0.821 / 0.749) | 0.151 (random) -> *TBD* | provisional | probe eval |
| rock_group decodability | probe acc (MLP / logreg), full all-rock test split | 0.919 / 0.894 | 0.810 (majority: always-igneous) -> *TBD* | provisional | probe eval |
| redox imputation | log-R2, redox slot (never in trace medians) | 0.306 | ~0 (mean-fill) -> *TBD* | provisional | imputation readout |
| redox token lift | populated-minus-padded qap MLP on measured-pair rows | +0.015 (0.774 populated) | 0 (token ignored) -> *TBD* | provisional | probe eval, #46 |

Each encoder scores with its own sweep-tuned read-out head (the head re-tunes
per encoder so the axis measures the latent, not a stale head). The
latent-XGB / input-XGB gates were measured on the previous corpus and are not
yet re-run on the all-rock corpus, so the 95% / 94% recovery framing carries
over only approximately. Probe ceilings are still floor-anchored only -- the
missing ceiling is a raw-input classifier, which #19 supplies. rock_group is
the one axis scored on the full all-rock test split rather than the shared
roster (the roster is all-igneous, where the probe's 0.982 is merely igneous
recall); per-class recall runs igneous 0.97 / sedimentary 0.82 / metamorphic
0.51 -- metamorphic chemistry overlaps its protoliths, so group membership is
genuinely underdetermined from composition there.

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
| qap kNN purity | cosine k=5 | 0.577 (prev 0.582) | 0.100 (random) -> 0.607 (raw-input, previous-corpus measurement) | provisional | [input_knn_ceiling](../explorations/archive/input_knn_ceiling/README.md) |
| supergroup kNN purity | cosine k=5 | 0.670 (previous encoder; not re-measured this cascade) | 0.151 (random) -> 0.676 (raw-input) | provisional | [input_knn_ceiling](../explorations/archive/input_knn_ceiling/README.md) |
| distance-metric recoverability | post-hoc kNN lift toward probe | *TBD* | raw-distance kNN -> probe acc | candidate | #16 |

qap is ~96% of a data ceiling that is itself well below 1.0 (qap is largely not
kNN-separable from chemistry). The open question is how much of the probe-vs-kNN
gap a learned/whitened metric recovers without re-pretraining.

## Geometry health -- anisotropy

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
| head task performance | ModeHead / CondHead / CompHead NLL + calibration | measured per version | marginal/prior baseline -> oracle | provisional | `results/ledger.yaml` (corpus), `results/real_rock_ledger.yaml` + journal 2026-08-03 (real rocks) |

Track B landed; head performance is tracked per version in the ledgers rather
than as one scorecard number, and the real-rock benchmark is the cross-era
judge (production `joint_heads_supportmag_seed0`: sealed-test joint NLL 0.478
after recalibration, vs 0.769/1.371 for the traced-corpus arms). The
multimodal-posterior risk resolved in the corpus's favor: the latent supports
it, and the surviving gap is output-family width (single Dirichlet vs the
measured bimodality), not latent capacity.
