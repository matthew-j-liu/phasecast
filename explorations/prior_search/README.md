# prior_search

## Question

Which (P, H2O) prior should the generator draw from so that a head trained
on the corpus gets better on feldspar, pyroxene, amphibole and biotite as
the corpus grows (#79, epic #74 step 3)? #78 showed the equilibrium recipe
reaches the measured silicate share of most benchmark rocks at some draw
from the broad box while the marginal over the box buries it. This
exploration screens candidate priors before any corpus is generated (stage
0, no MAGEMin), and will hold the generator and the ladder figures for the
schemes that go on to be generated.

## Method

Rulings (owner, 2026-09-16): the target is the four silicates, about 90%
of the crystal cargo on the typical rock; the scaling ladder is 128 / 256 /
512 / 1,024 / 2,048 compositions (five points, four doublings, one night of
MAGEMin per scheme, every size by nesting); seeds on the ladder's endpoints
are training seeds, so no scheme is generated more than once; no
intermediate screen at 512; local first, the cluster delayed.

- **Family.** A scheme is a rectangle in (P, H2O), uniform in P and
  uniform in H2O given P, clipped to the broad box (P 0.5 to 12 kbar, H2O
  0 to 8 wt%) and pointwise to the solubility bound `w_max(P)`: the box
  construction of `cnmineral.data.condition_proposal`, copied into
  `schemes.py` so a corpus drawn from a scheme carries exactly the density
  the screen weighted by. The grid is centers P 6 / 8 / 10 kbar x H2O 3 /
  4.5 / 6 wt% x two half-widths, narrow (2 kbar, 1.5 wt%) and wide (3 kbar,
  2.5 wt%): 18 schemes covering 14% and 35% of the broad box's area. The
  #78 best-draw quartiles for the four silicates (P 5 to 11 kbar, H2O 2.5
  to 6 wt%) say the hits are broad, so both widths are defensible.
- **Controls.** `broad`, the production broad box (every draw weight 1, so
  its marginal is #78's `d_marginal`), and `traced`, the production
  trace-conditional prior `bulk_proposal` evaluated per rock on its TAS
  class (the qap stand-in; `generator_reach/results/compositions.csv`) and
  its reported Th, Nb, Yb, V, Y and TiO2 (`data/real_mineralogy/samples.csv`;
  rocks missing them fall to the class tier, as production does).
- **Weights.** Each of a rock's 128 draws is weighted by the scheme density
  over the broad-box density stored on the draw. Both rung rows of a draw
  (50 and 30, as #78 pools them) carry its weight; the weighted mean share
  per category is the marginal a corpus drawn under the scheme would
  present for that rock. Effective draws per rock are `(sum w)^2 / sum
  w^2` over the draws that reached a rung; a rock no draw reaches is
  counted and excluded.
- **Score.** Per subset, the per-category absolute error from the measured
  share averaged over the rocks quantifying it, the head's share-error
  definition (`magnitude_parity_stats`); `silicate` is the mean over fsp,
  px, amp and bi, `other` over ol, opq and q. The train split (372 rocks)
  ranks, the val split (106) reads, test is untouched. The weighted
  within-rock spread of the silicate share is reported as the stage-0
  stand-in for the collapse guard.
- **Caveat.** A reweighted mean from fewer effective draws is noisier, and
  noise inflates an absolute error, so a narrow scheme reads slightly worse
  than the same center wide. The ranking by center is the robust read;
  widths are compared knowing the bias direction.
- **Rosters.** `rosters/n{N}.csv` are the first N ids of the md5 hash order
  `corpus_scaling` builds its ladder from (`hash_order`, imported), written
  sorted by id as those rosters are; nested by construction, and 2,048 sits
  inside the on-disk 5,000. The screen writes them when missing and refuses
  a file that no longer matches.

```
python explorations/prior_search/run_prior_screen.py                 # seconds
python explorations/prior_search/run_prior_screen.py --figures-only
python explorations/prior_search/run_predictability.py               # seconds
```

Reference cells for the production prior at the five sizes, last-epoch
state at both depths (about a minute each):

```
for N in 128 256 512 1024 2048; do
  python explorations/corpus_scaling/run_variant.py --tag final56_allrock_n$N \
      --roster explorations/prior_search/rosters/n$N.csv --batches-per-epoch 8 \
      --set epochs=7 --mode-selection final
  python explorations/corpus_scaling/run_variant.py --tag final224_allrock_n$N \
      --roster explorations/prior_search/rosters/n$N.csv --batches-per-epoch 8 \
      --set epochs=28 --mode-selection final
done
```

Committed under `results/`: `schemes.csv` (one row per control and scheme:
rectangle, area fraction, per-category and pooled errors per subset,
spread, effective-draw facts), `screen.json` (settings, ranking, top three,
the reproduction check), `prior_ranking.png` (schemes sorted by train
silicate error, val filled and train hollow, the controls as lines, the
cost on the others beneath), `prior_grid.png` (val error by rectangle
center, one heatmap per width), and from the predictability read
`predictability.csv`, `predictability.json` and `predictability.png`
(ceilings and predicted windows on val, and the predicted against the
true hit centers).

## Result (stage 0, 2026-09-16)

The broad control reproduces #78's `d_marginal` means to floating point.
Absolute share error of the reweighted marginal, train / val:

| scheme | silicates | others (val) | effective draws |
|---|---|---|---|
| broad box (production) | 0.116 / 0.136 | 0.190 | 128 |
| trace-conditional (production) | 0.126 / 0.144 | 0.181 | 28 |
| p10_w3_wide (best) | 0.108 / 0.130 | 0.211 | 35 |
| p10_w4.5_wide | 0.110 / 0.137 | 0.224 | 35 |
| p10_w3_narrow | 0.110 / 0.133 | 0.214 | 17 |
| p6_w6_narrow (worst) | 0.130 / 0.148 | 0.209 | 17 |

- **The family's ceiling is small.** The best rectangle takes 0.007 off the
  silicate error on train and 0.006 on val against the broad box, and the
  whole grid spans 0.02 with the broad box in its middle. The best
  reweighted marginal is still 0.13 on val, against 0.002 for the best
  single draw in #78: the hits exist for nearly every rock, but they sit in
  different places for different rocks, so no one rectangle recovers them.
- **Pressure is the mild lever, water is not.** Error falls with the
  pressure center at every water center and rises with the water center at
  every pressure; 6 wt% is the worst column everywhere. Read past the grid
  edge (scratch, not in `run_prior_screen.py`): an 11 kbar center equals
  10, drier centers (1.5 and 0 wt%) are worse, and a pressure band of 8 to
  12 kbar with water left at the broad box's full range scores the same as
  the best rectangle (0.108 / 0.131).
- **The cost is real.** Every silicate-aimed scheme raises the error on
  oxides, olivine and quartz, 0.190 to 0.21 at the best scheme, the
  direction #78 predicted (their hits are low and dry). The
  trace-conditional prior is the mirror image: worse on the silicates,
  better on the others.
- Narrow and wide agree on the ordering of centers; narrow reads slightly
  worse at the same center, the direction of the effective-draw bias.

## Does chemistry predict where a rock's hits sit? (2026-09-18)

The rectangle result leaves one family: a prior that changes with the
rock. Before writing one, `run_predictability.py` reads on the same draws
whether any could work. A rock's hits are the (draw, rung) rows whose mean
absolute share error over its quantified silicates is within 0.05 (623
rocks quantify a silicate; 234 have no such row and keep their 8 best).
Two ceilings, then the same rectangle centered where a predictor fit on
the train rocks puts it, read on val:

| prior | silicates train / val | others val |
|---|---|---|
| broad box (stage 0) | 0.116 / 0.136 | 0.190 |
| best rectangle (stage 0) | 0.108 / 0.130 | 0.211 |
| hit rows only (ceiling) | 0.079 / 0.088 | 0.143 |
| wide window on the rock's own hit center (ceiling) | 0.095 / 0.107 | 0.149 |
| wide window on the train-mean center | 0.119 / 0.139 | 0.204 |
| wide window on the TAS-class mean center | 0.112 / 0.137 | 0.183 |
| wide window on a ridge prediction from the majors | 0.114 / 0.137 | 0.179 |
| wide window on a random-forest prediction from the majors | 0.101 / 0.131 | 0.187 |

- **The joint ceiling is far above the per-category one.** #78 found a
  draw within 0.05 for 80 to 95% of rocks per category; on the four
  silicates together, 38% of rocks have none, and the median best joint
  error is 0.037. Feldspar's best draw is not amphibole's. Even the hit
  rows alone leave 0.088 on val.
- **The hits are localized but not predictable from majors.** Within a
  rock, hits spread 1.4 kbar and 1.0 wt% (medians), so a window around
  them is meaningful; across rocks the centers spread 5 to 10 kbar and 2.7
  to 5.0 wt% (interquartile). The forest recovers R2 0.17 of the pressure
  center and 0.04 of the water center on val; ridge and the class mean
  recover none. The best predicted window scores 0.131, the best global
  rectangle's number.
- **The heads already beat every marginal but the oracle.** Zero-shot
  silicate share error on the val rocks (scratch, `magnitude_parity_stats`
  on the split): production head 0.119, the 5,000-composition heads at 224
  and 56 steps 0.123 and 0.112. A head trained on the broad-box corpus
  sits between the best global rectangle (0.130) and the oracle per-rock
  window (0.107), so the most a perfectly conditioned prior could open is
  about 0.01, and majors predict none of it.

## Decision

Owner, 2026-09-18 (journal that date). For the current (generator, ruler)
pair, single-condition equilibrium against du Bray's phenocryst counts,
there is no way to change how the generator runs to attain scaling. The
pair has to change: the recipe, fractional crystallization of the rock's
own bulk with the accumulated crystals as the label against the same
ruler, or the ruler, plutonic whole-rock modes read on the current
generator at a solidus rung. The ruler first, because it costs a harvest
and a rung and its outcome sorts the recipe question. #79 closed on its
gate; no scheme was generated; the rosters under `rosters/` stay as the
ladder for the next pair; the condition prior is a settled decision in
CLAUDE.md.
