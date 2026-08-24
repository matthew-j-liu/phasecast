# input_probe_ceiling

## Question

The canonical encoder's pooled latent decodes rock type at qap MLP probe 0.759
(`docs/scorecard.md`), and that number carries the claim that the latent is
geologically organized. But gard2019's `qap_name` is a **computed**
classification derived from LOI-free major-element chemistry (Gard et al. 2019,
Sect. 4.2/4.4 and Table 3), not an observed petrographic name -- and the encoder
reads those same majors. So 0.759 on its own cannot separate two very different
readings:

- the latent is well organized, and the probe is reading that organization; or
- the label is close to a deterministic function of the input, and *any*
  faithful compression recovers it.

The control that separates them is the same probe trained on the **raw assay**.
`docs/scorecard.md` lists this as the missing ceiling for both the qap and
supergroup decodability rows (`0.100 (random) -> *TBD*`), which #19 supplies.
This is the classifier analogue of the archived
[input_knn_ceiling](../archive/input_knn_ceiling/README.md), which answered the
same question for a distance metric (qap k=5: latent 0.582, raw-input 0.607).

Scope: only the raw-input rung. #19 also carries a PCA/PPCA linear-null bracket,
which stays open there.

## Method

Both arms are fit inside one run off **one pair of index arrays**, so identical
rows is a property of the code rather than a claim -- `runs/probe/` no longer
exists, so no artifact backed the 0.759 and re-running the latent arm was
necessary anyway.

Protocol, asserted before any fitting (a drift in the split, the roster, or the
corpus filters would otherwise yield a plausible wrong ceiling instead of a
failure): `pretrain_allrock_redox_seed0/epoch_best`, 787,721 rows, split by
`reserve_test(ckpt.config)`, scored rows restricted to
`pretrain_inherited_test_roster.csv`, then both sides restricted to qap-labeled
rows -- 458,490 train / 32,950 test, 70 classes on both sides, chance 0.0997,
majority (basalt) 0.2203. Redox token padded, the deployment regime and the
condition 0.759 was measured under. Seed 0 from the config.

Five representations through the identical probe primitives
(`train_logreg_probe`, `train_mlp_probe`), two targets (`qap_name`,
`supergroup`):

| representation | dim | what it is |
|---|---|---|
| `latent` | 256 | the paired latent arm |
| `std_log_mask` | 44 | per-element standardized `log10(frac)` + a binary populated flag |
| `std_frac_mask` | 44 | the same, standardized in linear space |
| `raw_frac_mask` | 44 | literal abundances + the same flag |
| `std_log` | 22 | the log arm without the flag |

Four raw featurizations, and the ceiling is the **best** of them -- a ceiling
should be a maximum over reasonable methods, and reporting one pick invites the
objection that it was the one that lost. `input_knn_ceiling` reported "best
raw-input" for the same reason.

Three featurization choices carry the fairness of the whole contrast, and are
worth stating:

- **log and linear both get a fairly scaled arm.** Log space is where the
  encoder's own objective lives (ADR 007) and where production already feeds a
  small head (`cnmineral.eval.real_rock :: raw_chemistry`), and it stops the
  majors from swamping the traces. Linear `frac` won `input_knn_ceiling`
  because k-NN in linear space is scale-dominated by the majors and qap is a
  major-element story. Rather than pick, both are run standardized.
- **`raw_frac_mask` is carried, but it is not a fair MLP arm.**
  `train_mlp_probe` does not scale its inputs, and unstandardized abundances
  span roughly `1e-6` to `0.5` across columns; on a subsample its MLP collapsed
  to 0.430 against its own logreg's 0.590 (the logreg standardizes internally).
  It is kept because it is the `input_knn_ceiling` representation verbatim, so
  the two explorations compare directly -- but `std_frac_mask` is what answers
  "how good is linear space." Since the ceiling is a max, the collapse costs
  nothing.
- **the missingness flag is not optional.** The encoder sees missingness
  explicitly (`src == 0` vs `src == Z`) and roughly half the trace columns are
  unreported per rock. A raw arm denied that signal would be handicapped on
  input representation rather than on representation learning -- a strawman in
  the direction that flatters the encoder. `std_log` measures what the flag is
  worth.

Center/scale come from the **train** indices only (`log_frac_stats` for log
space, its local linear counterpart for `std_frac`); absent elements fill with
standardized 0 (the neutral train mean), not `log10(0)`. The redox slot is
dropped from every raw matrix, because `PaddedRedoxView` zeroes it for the
latent arm.

**Tuning budget.** Headline is fixed-protocol on both arms, because the paper
reports the fixed-protocol number (ADR 007's table shows fixed 0.769 vs HPO'd
0.790 -- materially different, which is exactly why the two must not be mixed
across arms), and because the MLP defaults are standard SSL-eval values not
chosen by looking at this latent. `--tune` adds a matched Optuna leg on the
latent and the winning raw arm under `configs/probe/probe_hpo.yaml`'s own search
space, trial count, and sampler seed. That leg's objective is test accuracy,
matching `run_hpo` -- fair between arms but optimistically biased in absolute
terms, which is why it is the robustness check and not the headline.

**The asymmetry that cannot be removed:** the encoder pretrained (unsupervised)
on all 630,177 trainval rows while the raw probe trains on the 458,490
qap-labeled ones. That is not a fairness violation -- it is precisely what SSL
buys -- but the honest framing is "latent vs raw assay, the encoder having had
unlabeled access to a superset," not "the same information."

    conda run -n mineral python explorations/input_probe_ceiling/run_ceiling.py \
        --checkpoint checkpoints/pretrain_allrock_redox_seed0/epoch_best.pt

Expect roughly 45-90 min for the fixed-protocol pass, extrapolated linearly
from an 8,000-row subsample where the 20 fits took ~40 s in total; the tuning
leg adds ~1-3 h.

## Result

The latent arm reproduced 0.759 exactly, so this is the arm the scorecard and
the paper draft report. It loses.

qap_name, 70 classes, chance 0.0997, majority (basalt) 0.2203:

| probe | latent (256) | best raw | margin |
|---|---|---|---|
| linear | **0.704** | 0.681 (`std_frac_mask`) | **+0.023** |
| MLP, fixed protocol | 0.759 | **0.787** (`std_frac_mask`) | -0.028 |
| MLP, 30 trials each | 0.792 | **0.834** (`std_frac_mask`) | **-0.042** |

supergroup: linear 0.731 vs 0.659 (**+0.071**); MLP 0.814 vs 0.813 (tie).

Two findings, in opposite directions.

**The latent carries no more rock-name information than the assay.** It cannot
-- it is a compression of the assay -- and it in fact loses. Matched tuning did
not rescue it: both arms gained, the raw arm gained more, and the deficit grew
from 0.028 to 0.042. This is the same sign and nearly the same magnitude as the
archived [input_knn_ceiling](../archive/input_knn_ceiling/README.md), where raw
beat latent by 0.025 at qap k=5, so two unrelated methods agree.

**The latent linearizes what the assay carries nonlinearly.** The linear probe
reverses the ordering, and dimensionality cannot explain it: a linear readout is
invariant to linear reparameterization, so any linear map of the 44 features is
bounded by the raw arm's 0.681. Only the encoder's nonlinearity can produce
0.704. It roughly halves the accuracy a nonlinear probe adds over a linear one
(qap 0.106 -> 0.055, supergroup 0.154 -> 0.083).

Three secondary results worth keeping:

- **Standardized linear beats standardized log** (qap MLP 0.787 vs 0.774). The
  README predicted the opposite -- that `input_knn_ceiling`'s linear win was a
  distance-metric artifact that a per-feature-scaling classifier would erase. It
  is not; linear space wins for qap under both metrics.
- **The missingness flag set the verdict.** In log space it is worth +0.025
  (0.774 masked vs 0.749 unmasked). Denied that signal, the raw arm would have
  scored below the latent and this control would have returned the opposite
  answer. That was a judgment call made before any measurement.
- **`raw_frac_mask` is not a fair MLP arm at small n.** On an 8,000-row
  subsample its MLP collapsed to 0.430 against its own logreg's 0.590; at the
  full 458,490 it recovers to 0.726. Unscaled inputs cost the MLP, and the cost
  shrinks with data.

This also settles half of #19's linear-null bracket without running PCA: a PCA
of the assay is a linear map, so a linear probe on it is bounded by 0.681 -- 
below the latent. Only the PCA-plus-nonlinear-probe leg remains open.

## Decision

**The qap probe axis is not evidence that the latent is organized, and the
scorecard rows now say so.** The label is computed from the same majors the
encoder reads, and a 44-column read of those majors decodes it better under
every budget tested. The axis is retained with its ceiling filled rather than
deleted: a measured negative anchors the two decodability rows that were
floor-only.

The claim that survives is linearization, and it is the one worth carrying
forward: the encoder makes class structure linearly accessible without adding
information the analysis does not already contain.

**Not in the paper.** The result was drafted into the Results section and cut
2026-08-15 for space -- the argument needed roughly 110 words of hedging to land
a conceded result, against a manuscript a page over the limit. The surviving
imputation sentence claims that the latent *preserves* chemistry, not that it
beats raw chemistry, which is the claim this control leaves standing. Do not let
that sentence drift toward superiority in a later edit: imputation carries the
same shape of control and points the same way (scorecard: latent-XGB 0.877
against input-XGB 0.892). The macros are emitted and unreferenced, so the number
is one `\cite`-free sentence away if space appears.

Canonical verdict in `docs/journal.md` (2026-08-15).
