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

    conda run -n alignn_fix python explorations/input_probe_ceiling/run_ceiling.py \
        --checkpoint checkpoints/pretrain_allrock_redox_seed0/epoch_best.pt

Expect roughly 45-90 min for the fixed-protocol pass, extrapolated linearly
from an 8,000-row subsample where the 20 fits took ~40 s in total; the tuning
leg adds ~1-3 h.

## Result

Not yet run. The slice reproduces the protocol exactly (458,490 / 32,950 / 70
classes / chance 0.0997 / majority 0.2203, verified 2026-08-14) and the fit
path is smoke-tested end to end on a subsample, but a subsample is not the
measurement: 8,000 train rows is a far harsher regime for a 256-dim latent than
for a 44-dim raw matrix, so nothing about the ordering there carries over.
`results/probe_ceiling.json` and this section land together.

## Decision

Pending the result. The three readings and what each does to the paper sentence:

- **best raw >= 0.759** -- the probe measures the label's determinism, not the
  latent's organization. Drop the qap-probe evidence for "geologically
  organized" or state plainly that it sits at the raw-input ceiling, and
  re-anchor on an axis with margin. The encoder-vs-raw-chemistry result on real
  rocks (+0.5-0.67 nats) is already measured and is *not* at risk from this
  control, since it contrasts the same two inputs on a task with no such
  determinism.
- **latent clearly above best raw** -- the claim survives and strengthens into a
  control rather than a bare number.
- **margin inside the 0.002-0.006 cross-seed band** -- indistinguishable at this
  resolution; report both, do not claim organization from this axis alone, and
  run the tuning leg before writing.

Either way the scorecard's two `*TBD*` ceilings get a number. Canonical verdict
goes to `docs/journal.md`.
