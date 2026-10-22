# Running the model from an application

This directory is the handoff for building a web app or a local tool on the
trained model. `inference.ipynb` is the reference path: a table of rock
compositions in, a posterior over mineralogy and formation conditions out, both
zero-shot and recalibrated, with the figure an app would draw. This page is the
contract and the setup facts around it.

## Reference docs

- `inference.ipynb`, here. Executed outputs are committed, so it reads on
  GitHub without running. `inference_helpers.py` beside it holds the
  sampling, recalibration and drawing code it calls, the interim stand-in
  for the stable API.
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
| fit the recalibrator on 372 rocks | under 1 s |

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

- A missing trace element is padding, not zero. A majors-only rock is valid.
- Nothing is renormalized. Each row goes through on the total its source
  reports; do not rescale to 100.
- Below-detection entries (`<0.5`, `< DL`) are left blank.
- Supplying `feo` and `fe2o3` separately does not change the prediction
  today; the redox slot is reserved for a future retrain.
- `cr2o3` is accepted and ignored. MnO, P2O5, LOI, totals, rock names and any
  temperature, pressure or water estimate are ignored.
- Validate before tokenizing: every oxide within 0 to 100 wt% and every trace
  within 0 to 1,000,000 ppm, or the encoder's lookup table overflows.
  `check_input` in the helper module is the reference.

There is no standard lab export. ALS certificates report majors as oxide wt%
and traces in ppm under a multi-row header (method code, analyte, unit,
detection limit); Actlabs uses four header rows and `< DL` for below
detection. Both parse to the table above by column aliasing. Commercial labs
usually report total iron as Fe2O3, which is the `fe2o3_tot` form.

## Output contract

Per rock, `predict` in the helper module returns the input chemistry, the
seven group names, and for each of the two arms:

- `presence`: probability of each of the seven mineral groups (feldspar,
  pyroxene, amphibole, biotite, olivine, Fe-Ti oxides, quartz).
- `share_mean`, `share_p25`, `share_p75`: share of the crystalline mass per
  group, percent.
- `scenarios`: the top mineral lists with their probability and shares.
- `conditions`: 512 samples each of temperature (C), pressure (kbar) and
  water content (wt%). The front end draws the temperature-pressure density
  from these points; it is not a model output.

Plus `composition_wt_pct`, the oxide composition of the most likely minerals.
About 20 KB of JSON per rock.

## The two arms

Zero-shot is the model as trained on simulated labels. Recalibrated passes
the seven group presences and magnitudes through a 119-parameter layer fit on
372 measured volcanic rocks; it does not touch the condition or composition
heads. The app shows both, labeled (owner ruling 2026-10-15, overriding the
zero-shot-only default on issue #70). The fit is seeded and takes under a
second, so the app can fit at startup from the benchmark CSVs or load the
saved state dict the notebook writes.

## Things that bite

- The heads checkpoint stores the encoder path relative to the repository
  root. Run from the root or rewrite `ckpt.encoder_checkpoint` before loading.
- On macOS, `import cnmineral` before `import torch`, or set
  `KMP_DUPLICATE_LIB_OK=TRUE`. Two OpenMP runtimes collide otherwise. Linux
  servers are unaffected.
- Sampling is seeded. Same rock, same seed, same card.
- In the deck's output card the "Possible Assemblages" panel is hand-drawn
  illustration, not model output. The notebook's scenario panel is the real
  one.

## Where the app code lives

In its own repository. This repository is the library; the stable inference
API (issue #70) promotes the helper module's `predict` into the package, and
the app imports that rather than copying the helper.
