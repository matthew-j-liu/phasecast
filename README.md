# phasecast: probabilistic mineralogy from whole-rock geochemistry

From one whole-rock chemical analysis, the model predicts a distribution over
modal mineralogy and the conditions of formation.

This repository accompanies the submission *Simulated petrological labels
enable probabilistic mineralogy prediction from whole-rock geochemistry*.

## Overview

Modal mineralogy, the mineral identities and abundances in rocks, guides
geological studies and industrial resource extraction. While measuring modal
mineralogy is time-intensive, whole-rock chemical measurements furnish
elemental concentrations at higher throughput. Predicting mineralogy from
elemental data, however, poses a one-to-many inverse problem, and labeled
mineralogical datasets are too sparse to train on at scale. We address both
challenges by simulating petrological labels that express multiple formation
conditions per composition and at a scale unattainable by measurements.

The labels supervise a three-layer model: a transformer encoder
self-supervised on roughly 788,000 whole-rock compositions, prediction heads
that turn the encoding into distributions over mineral assemblages, and a
recalibration layer that aligns the predictions with measured mineralogy.
Zero-shot on 640 volcanic rocks, the model identifies the minerals present
and predicts which rocks are richer in each mineral; recalibration fit on as
few as 25 labeled rocks brings the predicted amounts into agreement with
measured values, recovering 76 percent of the improvement from fitting on the
full 372-rock fitting split.

## Installation

```bash
conda env create -f environment.yml
conda activate mineral
pip install -e .
pytest tests/        # optional smoke check
```

Only stage 3 below (label generation) additionally needs Julia with MAGEMin.
Install Julia 1.12 (for example through juliaup), then instantiate the pinned
environment in `magemin/`:

```bash
julia --project=magemin -e 'using Pkg; Pkg.instantiate()'
```

## Pipeline

Stages marked *(runnable here)* execute against files in this snapshot.
Stages marked *(pre-computed)* involve long computation and ship their
outputs instead.

**1. Tokenize whole-rock chemistry** *(in-package)*. The model reads one
whole-rock analysis as a set of element tokens (element identity plus amount)
with an optional iron-redox row; temperature, pressure, and water content are
never inputs. Tokenizer: `src/cnmineral/data/tokens.py`. The pretraining
compilation (Gard et al. 2019) is a public download.

**2. Pretrain the encoder** *(pre-computed -- ships `epoch_best.pt`)*.
Masked-element self-supervision over roughly 788,000 compositions spanning
all rock types.

```bash
python scripts/training/pretrain.py --config configs/pretrain/pretrain_allrock_redox.yaml
```

**3. Generate simulated labels** *(pre-computed -- ships the excerpt)*. Each
label pairs one measured composition with the mineral assemblage `MAGEMin`
computes for it. Sampling several pressure and water conditions per
composition lets the labels express multiple formation conditions for one
rock.

```bash
python scripts/build/generate_mineralogy_corpus.py --bulks <bulk csvs> --out data/corpus --draws 24
```

**4. Train the heads** *(pre-computed -- ships `heads_best.pt`)*. Three heads
predict distributions over mineral abundances, per-phase compositions, and
formation conditions from the frozen encoder's representation.

```bash
python scripts/training/joint_heads.py --config configs/heads/joint_heads.yaml
```

**5. Evaluate on measured rocks** *(runnable here)*. Scores the shipped model
against the 640-rock benchmark, zero-shot and recalibrated. The third command
refits the recalibration layer at each fitting-set size, reproducing the
paper's few-shot experiment.

```bash
python scripts/eval/real_rock.py --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt
python scripts/eval/real_rock.py --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt --fit-recalibrator matrix
python figures/src/fit_fewshot_curve.py --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt
```

**6. Regenerate the paper's figures** *(runnable here)*. Each `fig_*.py`
redraws one paper figure from the frozen inputs in `figures/data/`. The
paired `extract_*.py` rebuild those inputs from the artifacts in this
snapshot (two exceptions read the full simulated corpus and are noted in
their docstrings).

```bash
python figures/src/fig_cohort_card.py
python figures/src/fig_inference_card.py
```

## Checkpoints, data, and figure inputs

Everything the runnable stages above read is either in the source tree or
listed here.

| path | contents | size |
|---|---|---|
| `checkpoints/pretrain_allrock_redox_seed0/epoch_best.pt` | the pretrained encoder: a plain `torch.save` dict (encoder and optimizer state, training config, loss history) -- load with `cnmineral.training.Checkpoint.load` | 44 MB |
| `checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt` | the three supervised heads (mineral abundances, phase compositions, formation conditions): state dicts plus their head, loss, and label configs -- load with `JointHeadCheckpoint.load` | 7 MB |
| `data/real_mineralogy/samples.csv` | benchmark chemistry and metadata: 730 volcanic samples with major oxides, 13 trace elements, and petrographic notes, keyed by source and sample id | 240 KB |
| `data/real_mineralogy/modes.csv` | the same samples' measured mineral abundances, one row per sample-phase pair, carrying each source's own phase name and its mapping into the vocabulary the model is scored against | 240 KB |
| `data/corpus_excerpt.csv` | a readable slice of the simulated label corpus: one composition's sampled crystallization runs and the resulting assemblages | few KB |
| `figures/src/` | one `fig_*.py` per paper figure, the `extract_*.py` scripts that rebuild their inputs from the artifacts above, and the `fit_*.py` scripts that recompute the few-shot experiment | -- |
| `figures/data/` | the frozen inputs the figure scripts read: one small CSV per figure plus the two few-shot result JSONs | ~700 KB |

The benchmark evaluation joins the two CSVs; its headline scores the 640
least-altered rocks that quantify at least one mineral abundance (724 rocks
carry both chemistry and modes) -- the cohort every benchmark number in the
paper refers to. The train/val/test split rosters ship inside the
package (`src/cnmineral/data/`).

Not included: the full simulated corpus (3.5 GB of parquet, regenerable end
to end from the public compilation via stage 3) and transient run outputs.
The full corpus will be released with the de-anonymized repository on
publication.

## License

MIT -- see `LICENSE`.
