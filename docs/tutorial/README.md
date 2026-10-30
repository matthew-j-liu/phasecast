# Running the model from an application

This directory is the handoff for building a web app or a local tool on the
trained model. `inference.ipynb` is the reference path: a table of rock
compositions in, a posterior over mineralogy and formation conditions out, with
the figure an app would draw. This page is the contract and the setup facts
around it.

## Reference docs

- `inference.ipynb`, here. Executed outputs are committed, so it reads on
  GitHub without running. `inference_helpers.py` beside it holds the
  validation, sampling and drawing code it calls, the interim stand-in for
  the stable API.
- The paper, [*Simulated petrological labels enable probabilistic mineralogy
  prediction from whole-rock geochemistry*](https://openreview.net/forum?id=XH2fnz0Enx)
  (NeurIPS 2026 Sim2Science workshop). Figure 2 is the inference card the app
  generalizes.
- The overview deck, `slides/overview.pptx`: one anatomy slide per head with
  exact parameter counts. The paper has no system schematic.
- `docs/encoder.md` and `docs/heads.md` for the model in prose.

## Setup

```bash
conda env create -f environment.yml
conda activate mineral
pip install -e .
jupyter lab docs/tutorial/inference.ipynb
```

Inference needs none of the simulation stack: no Julia, no MAGEMin. The
runtime imports are torch, numpy, pandas, scipy, scikit-learn and pyyaml, plus
matplotlib for the figure. Loading the encoder also imports optuna through the
training package; that is an import-graph wart on the cleanup list, not a real
dependency.

| what | measured on a laptop CPU, four threads |
|---|---|
| import torch and the package | 1.1 s, once per process |
| load both checkpoints | under 0.1 s |
| encoder, per rock | 0.2 ms |
| full posterior per rock, 512 samples | 15 ms |

Checkpoints, both tracked in git: the encoder is 44 MB (18 MB of weights,
26 MB of optimizer state from pretraining) and the heads 7 MB. The whole
model is about 5 million parameters.

## Input contract

One row per rock, plain CSV with one header row, lowercase column names.

| columns | unit | required |
|---|---|---|
| `sio2 al2o3 cao mgo k2o na2o tio2` | oxide wt% | yes |
| one of `feo` + `fe2o3`, `feo_tot`, `fe2o3_tot` | oxide wt% | yes |
| `sr_ppm zr_ppm ba_ppm y_ppm rb_ppm ni_ppm nb_ppm v_ppm la_ppm ce_ppm nd_ppm yb_ppm th_ppm` | ppm | no |

- A blank and a 0 both mean not measured and pad their slot. A majors-only
  rock is valid.
- Nothing is renormalized. Each row goes through on the total its source
  reports; do not rescale to 100.
- Below-detection entries (`<0.5`, `< DL`) are left blank.
- A single total-iron value can go in `feo_tot` / `fe2o3_tot` or in `feo` /
  `fe2o3` alone; the tokenizer applies the same factor either way, so a
  single-rock form needs only FeO and Fe2O3 boxes. Supplying `feo` and
  `fe2o3` separately does not change the prediction today; the redox slot is
  reserved for a future retrain.
- `cr2o3`, MnO, P2O5, LOI, totals, rock names and any temperature, pressure
  or water estimate are ignored.
- Validate before predicting with `check_input` in the helper module. Its
  errors refuse a rock: fewer than four measured elements (the pretraining
  floor), an oxide outside 0 to 100 wt% or a trace outside 0 to 1,000,000
  ppm (either overflows the encoder's lookup table). Its warnings let a rock
  through with a note: a missing major oxide, or a major-oxide total outside
  85 to 103 wt%.

There is no standard lab export. ALS certificates report majors as oxide wt%
and traces in ppm under a multi-row header (method code, analyte, unit,
detection limit); Actlabs uses four header rows and `< DL` for below
detection. Both parse to the table above by column aliasing. Commercial labs
usually report total iron as Fe2O3, which is the `fe2o3_tot` form.

## Output contract

Per rock, `predict` in the helper module returns the input chemistry, the
seven group names (feldspar, pyroxene, amphibole, biotite, olivine, Fe-Ti
oxides, quartz) and, under `posterior`:

- `presence`: probability of each group.
- `share_mean`, `share_p25`, `share_p75`: share of the crystalline mass per
  group over all 512 samples, percent.
- `scenarios`: the five most likely mineral lists, each with its probability,
  mean `share` and `share_p25`, `share_p75`.
- `other_scenarios`: how many mineral lists fall past the top five, their
  combined probability and the largest single one.
- `conditions`: 512 samples each of temperature (C), pressure (kbar) and
  water content (wt%). The front end draws the temperature-pressure density
  from these points; it is not a model output. A few samples fall just below
  0 kbar or 0 wt% water (the condition head's Gaussian tails), so start those
  axes at 0.

Every seven-number array follows the order of the group names. About 10 KB
of JSON per rock.

The app shows the model as trained on simulated labels. The recalibrator, a
119-parameter correction fit on measured volcanic rocks, stays a research
tool and is not part of the app (issue #70).

## Things that bite

- The heads checkpoint stores the encoder path relative to the repository
  root. Run from the root or rewrite `ckpt.encoder_checkpoint` before loading.
- On macOS, `import cnmineral` before `import torch`, or set
  `KMP_DUPLICATE_LIB_OK=TRUE`. Two OpenMP runtimes collide otherwise. Linux
  servers are unaffected.
- Sampling is seeded. Same rock, same seed, same card. A port that keeps the
  seed and the order of the random calls in `_posterior` reproduces the
  notebook's numbers to the digit.
- In the deck's output card the "Possible Assemblages" panel is hand-drawn
  illustration, not model output. The notebook's scenario panel is the real
  one.

## Where the app code lives

In its own repository. This repository is the library; the stable inference
API (issue #70) promotes the helper module's `predict` into the package, and
the app imports that rather than copying the helper.
