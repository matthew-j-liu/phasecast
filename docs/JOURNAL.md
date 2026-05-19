# Research journal

Append-only chronological log of design discussions, observations, hypotheses
tested, and decisions made during the v1 paper development. The lab-notebook
analog for the research process.

**Where to put what:**
- *Binding architectural decisions* (hard-to-reverse choices, with alternatives
  considered and revisit triggers) → `docs/decisions/NNN_topic.md` (ADRs).
- *Observations, in-flight reasoning, hypotheses, daily updates* → this file.
- *Living architecture documents* → `docs/POSITION_C.md` and similar; updated
  in place when the design itself evolves.
- *Project orientation* → `CLAUDE.md`.

Each entry below uses a rough Q / Context / Considered / Decision / Next
template, adapted where some fields are N/A. Entries are appended forward
chronologically. Older context up top, newest at the bottom.

---

## 2026-05-11 — Differentiation framing established

**Q:** How do we position our project relative to mineralML and other existing
mineral-prediction tools?

**Context:** Started the session by deeply reviewing mineralML's codebase, paper,
and stoichiometric framework. Wanted to understand what they do and don't do
before claiming a contribution.

**Conclusion:** mineralML and our model are **inverse problems on the same input
space**. mineralML takes per-grain oxide chemistry from a single spot (EPMA /
EDS pixel) and classifies that one grain. We take per-rock bulk chemistry (ICP /
XRF) and regress the assemblage of multiple coexisting phases. Same oxide
vocabulary, semantically opposite tasks. This framing crystallized into the
4-axis differentiation table (no T/P input, no spot data, multimodal output,
hydrous + sedimentary phase coverage), which becomes the paper's headline
contribution claim — no existing tool covers all four axes.

**Next:** Use the table as the litmus test for every architectural decision —
does this change widen the table or just strengthen one row?

---

## 2026-05-11 — Position C crystallized as the design

**Q:** Where does our contribution actually live, given that mineralML, MAGEMin,
and CIPW each cover parts of the space?

**Context:** Worked through three candidate framings:
- *Position A — mineralML wrapper.* Bootstrap our training corpus by running
  mineralML on EDS map data and treating its outputs as labels.
- *Position B — MAGEMin emulator marginalized over T, P.* Train a model that
  amortizes MAGEMin's bulk + (T, P) → assemblage map, with T, P marginalized
  using rock-class-conditional priors.
- *Position C — MAGEMin × mineralML training-data fusion.* Use MAGEMin to
  generate equilibrium assemblage labels at sampled (T, P), then replace
  MAGEMin's idealized phase compositions with samples from mineralML's
  empirical per-mineral grain corpus.

**Considered:** A makes us downstream of mineralML — collapses the contribution
to "mineralML wrapper for non-imaging data," weakens the paper. B is a Bayesian
amortization of an existing tool — defensible but methodologically thin without
adding information neither tool has. C combines two information sources that
nobody combines today: physics-derived assemblage labels + empirically realistic
per-phase compositions.

**Decision:** C is the v1 contribution. B is the methodological scaffolding
inside C (the marginalization happens during training-data generation). A is
not in the pipeline at all.

**Next:** Build the pipeline that produces Position C training tuples. Reserve
ALIGNN structural stream + polymorph branch for v2.

---

## 2026-05-11 — Deterministic calculators aren't ML targets

**Q:** Should we use CIPW norm as a training target or auxiliary output?

**Context:** Initially proposed CIPW as a possible auxiliary head. User pushed
back: CIPW is a deterministic algorithmic calculator (pyrolite implements it
analytically). Training an ML version is strictly worse than calling the
function.

**Conclusion:** Don't build ML surrogates for deterministic calculators where
the analytical version is available. The same logic later applied to MAGEMin:
we use it as a *label-generation oracle*, not a prediction target. Our model's
user-facing input is bulk only; the T/P marginalization and the
assemblage prediction happen inside the model, not by emulating MAGEMin.

**Next:** Apply this principle to architectural decisions going forward — never
propose learning what physics or geometry already gives us exactly.

---

## 2026-05-12 — Class-conditional priors as the unifying design pattern

**Q:** Across three independent decisions (training data stratification, Fe
redox handling, T/P sampling), what's the right pattern?

**Context:** Hit the same shape three times in a row:
- Stratify training bulks by IUGS class label (`qap_name`)
- Sample Fe³⁺/ΣFe from per-class Middlemost 1989 ratios
- Sample (T, P) from per-class Putirka 2008 ranges

Each is a function of rock class with a literature default, where the user
might want to swap in alternatives for sensitivity analysis.

**Conclusion:** Encode petrologic priors as **injectable strategy objects with
literature defaults**, not as hardcoded constants. Pattern:
```
class XStrategy:
    def __init__(self, table=LITERATURE_DEFAULT): ...
    def __call__(self, rock_class, ...) -> value
```
Sensitivity analysis becomes a one-line config change. Now codified in
`fe_redox.py` and `tp_priors.py`; will extend to `h2o.py` next.

**Next:** Apply this same pattern to H2O when we build it.

---

## 2026-05-12 — Tiered Fe redox handling with provenance tracking

**Q:** complete.csv has measured Fe partitioning (`feo` + `fe2o3` directly) for
30% of rows; the rest report total iron in one of two forms. How do we use
measured data where it exists without forcing assumptions on every row?

**Context:** Upstream Stage 1 filter required *both* `feo` and `fe2o3`
explicitly present — which is why their Stage 1 collapsed from 740k igneous
rows to 161, badly biased toward older wet-chemistry samples (mostly mafic).

**Decision:** Three-tier strategy:
- **T1 (measured):** both `feo` and `fe2o3` present → use directly, no assumption
- **T2 (assumed_class):** only a `_tot` column → split using Middlemost 1989
  per-class ratio
- **T2c (derived_partial):** one of feo/fe2o3 + a `_tot` → derive the other
- **T3 (no_iron):** nothing → drop

Each training tuple carries a `fe_provenance` label so we can stratify
evaluation by Fe-tier and report tier-specific accuracy.

On the smoke set: 33% T1 / 39% T2 / 28% T2c / 0% T3. The methodological
transparency makes the paper's iron-handling section auditable.

**Next:** Symmetric tiered structure to apply to H2O.

---

## 2026-05-12 — Stage 1 H5 from CrabNet-Mineral is not v0 training data

**Q:** Can we save effort by reusing the existing MAGEMin outputs from the
other agent's pipeline?

**Context:** Asked the other agent to summarize how their `filter_complete_csv.py`
narrowed 1M rows to the 161 compositions in Stage 1. Their filter required
all 11 MAGEMin oxides non-null, which (because H2O is missing in ~98% of rows
and Cr2O3 in ~97%) implicitly demands wet-chemistry analyses with full Fe
partitioning. Result: 141/161 basalts, 11/161 picrites, single Frost class,
SiO2 capped at 60.2%.

**Conclusion:** Using Stage 1 as v0 training data would teach the model "the
world is basalts and picrites." Not usable. We build our own filter with:
- Stage-2-style permissive Cr2O3 / H2O (fill missing with 0)
- Tier 2 Fe fallback (use total-Fe columns when split unavailable)
- Quality bounds (sum 80-120 wt%, drop outliers)
- Stratified sampling by qap_name with class-balanced targets

Output: 522 bulks at smoke scale (~22k at full scale planned), spanning
basalt + andesite + dacite + rhyolite + granite + dunite + harzburgite +
syenite + monzogranite + foidite + many more.

**Next:** Write the filter (`02_filter_complete.py`); run smoke; verify Fe and
class distributions look right before going further.

---

## 2026-05-12 — Two upstream bugs surfaced because our problem setting exercises edge cases

**Q:** While validating the MAGEMin smoke run, found two issues that look like
mistakes in the upstream pipeline. Are they real bugs?

**Context:** First validation pass reported phase wt fractions summing to mean
0.86 (should be ~1.0) and amphibole / biotite at 0.3% / 0.2% of training
labels.

**Bug 1 (hard correctness):** MAGEMin's `out.ph` returns duplicate phase names
when two compositions of the same solution phase coexist (e.g., plagioclase
and alkali feldspar both as `fsp`, or two `cpx` compositions). The upstream
`parse_magemin_output.py` overwrites by phase name — silently dropping the
duplicate's mass. Same bug exists in our parser (vendored from upstream). Hit
in 43.5% of our smoke rows; visible in the mean-0.86 sum statistic.

**Why we hit it and they didn't (much):** Their brute 28-point T/P grid lands
in fewer two-feldspar / two-pyroxene cells than our class-conditional
sampling does. Same bug, different exposure rate.

**Fix:** `run_magemin.jl` now emits SS info as a list of per-phase records
preserving duplicates. `parse_magemin.py` sums Y_wt for duplicate phase names
and mass-weighted-averages SS endmembers. Phase wt sums now exactly 1.0.

**Bug 2 (design mismatch, not strictly a bug):** The brute T/P grid is Stage-1-
shaped, not Position-C-shaped. Position B requires class-conditional T/P
sampling. Refactored `run_magemin.jl` to be stateless — reads T, P per row.
T/P sampling moved upstream into `02b_sample_tp.py` using rock-class priors.

**Next:** Worth a heads-up to the other agent — if they ever run Stage 2 at
scale, the bug 1 fix should be backported.

---

## 2026-05-12 — Hydrous-phase gap forces a v0-blocking design decision

**Q:** Smoke MAGEMin labels show amphibole 0.3% and biotite 0.2% of training
samples — effectively absent. Is this a bug in our pipeline?

**Context:** Not a bug, a real consequence: 98% of `complete.csv` has no H2O
measurement → our schema layer defaults to H2O = 0 → MAGEMin's equilibria for
"dry" rocks don't crystallize amphibole or biotite, even at conditions where
real rocks routinely contain them (mid-crustal andesites, granodiorites, etc.).

**Considered:**
- (A) Accept the limitation, document the hydrous-phase gap, start model work
- (B) Build a tiered H2O strategy symmetric to Fe redox, then resume

**Decision:** B. Two reasons:
1. The paper's "hydrous + sedimentary phases" differentiation axis would carry
   a serious asterisk if amp/bi are absent from training labels.
2. Training a ModeHead on labels where amp/bi are functionally zero would
   collapse their gates permanently. Retrofitting H2O later means re-training
   from scratch, not incrementally improving. Cheaper to land the fix in v0.

**Design (parallel to fe_redox.py):**
- T1 (measured): `h2o_tot` reported in complete.csv → use directly (~1% of rows)
- T2 (class-conditional): ranges per qap_name from Wallace 2005 + Plank et al.
  2013 + Burnham 1979. Ranges not point estimates — within-class H2O variance
  is genuinely large (arc rocks 1-8 wt%). Uniform sample within range, like
  TPRangeStrategy.
- T3 doesn't exist — the class default is always defined.

**Open:** Per-class ranges still being compiled; literature gives ranges with
sometimes-conflicting central tendencies for the same class. Will reference-
cite each major class in the module docstring.

**Next:** Write `src/cnmineral/data/h2o.py` + integrate into `schema.py`.
Re-run smoke to verify amp/bi presence rises to a learnable level (target
~15-30% of arc/dacite/granite samples). Then proceed to model side.

ADR for the binding parts of this decision: `docs/decisions/003_h2o_strategy.md`.

---

## 2026-05-12 — H2O strategy implementation results

**Q:** Did the tiered H2O strategy actually move amp/bi presence into a
learnable range, and what side effects did it produce?

**Context:** Implemented `src/cnmineral/data/h2o.py` (ClassRangeH2OStrategy with
Wallace 2005 + Plank 2013 + Burnham 1979 per-class ranges) and threaded it
through `schema.py` + `02_filter_complete.py` + `parse_magemin.py`. Re-ran the
smoke pipeline end-to-end (522 bulks × 5 (T, P) = 2,610 attempts; ~41 s wall).

**Result — hydrous-phase presence:**

| Phase | Dry baseline | With H2O strategy | Change |
|---|---|---|---|
| amphibole (amp) | 0.3% | 11.2% | ~37× |
| biotite (bi)    | 0.2% | 11.7% | ~58× |
| muscovite (mu)  | absent | 0.8% | new |
| epidote (ep)    | absent | 0.1% | new |
| cordierite (cd) | absent | 11.0% | new |

Decision validated: amp and bi each in >10% of training labels — comfortably
learnable. Within-class targeting also looks right (mu in felsic only, ep at
specific T-P windows).

**Side effects on the equilibrium assemblages (all petrologically expected):**
- `liq` rose from 45% to 98% (H2O lowers the solidus; most sampled T conditions
  now above the wet solidus)
- `q` (quartz) dropped from 44% to 5% (SiO2 dissolves preferentially into
  hydrous melt)
- `ru` (rutile) dropped from 20% to 0.5% (rutile destabilizes with H2O)
- `fsp` dropped from 77% to 55% (hydrous phases compete for K, Na, Al)
- `spl`, `opx`, `cpx`, `ilm` all reduced (smaller fractions of solid phases
  because more mass is in the melt)

Mass balance still exactly 1.0 on every row.

**Open issue noted but deferred:** With 98% of training rocks having `liq` in
the assemblage, the trained model will learn to predict melt presence for
typical bulk inputs. Real users querying with fully-crystallized rocks (which
is what ICP/XRF analyses are) would get a non-zero `liq` prediction — physically
the rock is at room temperature with zero melt. This is a Position-B-shaped
artifact: we're predicting *equilibrium-at-sampled-T,P*, not
*final-crystallized assemblage*. Two possible v1.5 responses:
- (a) At evaluation time, marginalize the `liq` channel out of the prediction
  (report only solid phase fractions, renormalized)
- (b) Constrain T sampling per class to favor subsolidus conditions (lower
  end of each Putirka range), shrinking the liq-dominated fraction of training
  data
- (c) Reframe the model's claim: "predict the assemblage that *would form*
  during crystallization," with melt presence as a separate output
Not addressing now — flag for after v0 architecture validation. Pure model-
output post-processing (option a) is essentially free if it turns out to
matter.

**Decision confirmed:** H2O fix is right and ready. Conclude data-side
prerequisites for v1; next session begins model implementation
(`src/cnmineral/models/`).

**Next:** Commit H2O changes. Begin model side: encoder wrapper around
kingcrab.py, ModeHead with sigmoid-gate sparsity, training loop on smoke H5
to validate architecture end-to-end.

---

## 2026-05-12 — Model side v0 trains end-to-end on smoke H5

**Q:** Does the architecture train, and do we hit the v0 success criteria from the plan (`/Users/mliu/.claude/plans/start-to-plan-model-tender-sunset.md`)?

**Context:** Implemented v0 model side per the plan — but simpler than the
original draft. Reading kingcrab.py confirmed that `SubCrab` already does
encoding + masked-mean pooling + `value × sigmoid(gate)` output in one class.
For ModeHead-only v0, that's the whole model. No separate "encoder wrapper"
needed in v0; will refactor when v1 adds multi-head sharing.

**Files added:**
- `src/cnmineral/models/dataset.py` — `TrainMageminDataset`. Reads H5 eagerly,
  converts 11-oxide schema → 10-cation wt% (sums FeO + Fe2O3 → Fe), divides
  by 100 to give mass fractions in [0, 1] (the encoder expects fractions, not
  wt% — a non-obvious gotcha)
- `src/cnmineral/models/model.py` — `BaseModel` wrapping one `SubCrab(out_dims=2*n_minerals)`. Essentially upstream's BaseCrabNet with naming adapted
- `src/cnmineral/models/losses.py` — `mode_loss` = MSE + soft sum-to-1
  regularizer
- `src/cnmineral/models/training.py` — `train_one_epoch`, `evaluate`
  (per-phase R² + sparsity + sum-to-1 violation), `EarlyStopping`,
  `make_loaders` with `StratifiedShuffleSplit` on qap_name
- `configs/base_v0.yaml` — all hyperparameters in one place
- `scripts/03_train_base.py` — sole CLI `--config PATH`; everything else
  in YAML
- `tests/test_shapes.py` — shape-check tests run in <1s

**Two integration fixes during testing:**
1. `element_properties/` CSVs needed to live alongside `kingcrab.py`
   (it uses `dirname(__file__)/data/element_properties`). `git mv`d from
   `data/element_properties/` → `src/cnmineral/data/element_properties/`.
2. CrabNet's `FractionalEncoder` expects fractions in [0, 1], not wt% in
   [0, 100]. The encoder discretizes the input into a lookup index using
   `frac × resolution`, so wt%=50 produces frac_idx=2.5e5 — way out of
   bounds. Fix: divide cation wt% by 100 in the dataset.

**Result — first training run on smoke H5 (2,545 points, 22 phases, CPU, 97.7s):**

Plan's 6 success criteria, scored:
- Loop runs without errors → ✓ 95 epochs (early-stop at 94, best 79)
- Val loss decreasing → ✓
- Top-5 phase R² > 0.5 → 4 of 5 (spl was 0.36; others 0.69 - 0.99)
- Sum-to-1 violation < 0.05 → ✓ 0.0105
- Mean sparsity ~16-19 gates closed → ✓ 0.775 → ~17 of 22 closed per sample
- Checkpoint written → ✓ `checkpoints/base_v0/best.pt`

Per-phase R² at best epoch (sorted by val-set presence):

| Phase | Presence | R² |
|---|---|---|
| liq | 94% | +0.82 |
| fsp | 51% | +0.69 |
| cpx | 39% | +0.72 |
| spl | 37% | +0.36 |
| ol | 24% | +0.99 |
| opx | 21% | +0.88 |
| fl | 11% | +0.30 |
| amp | 10% | +0.18 |
| bi | 9% | +0.81 |
| cd | 8% | +0.19 |
| (rare phases, presence < 5%) | — | noisy, mostly negative |

**Notable findings:**
- Olivine R²=0.99 — Fo# is highly predictable given bulk MgO + FeO
- Biotite R²=0.81 despite only 8.6% presence. The H2O fix (journal entry above)
  is doing real work — the model can learn hydrous phases from limited samples
- Amphibole R²=0.18 despite similar presence to biotite — within-class
  composition variance is likely the issue; need to investigate
- Rare phases (< 5% presence) have noisy R² — expected at 2,545 samples;
  should improve at scale (--scale full ≈ 440k points)

**Decision validated:** architecture trains. ModeHead-only as v0 was the right
scoping; multi-head additions can wait. Next direction is scaling MAGEMin to
`--scale full --k 20` for a ~440k-point training corpus, then comparing the
same architecture's per-phase R² at scale. After that, multi-head extensions
(CompHead, GsysHead, ClassHead) can land additively.

**Next:** Commit model-side v0. Decide whether to scale up MAGEMin to full
corpus next session, or first add the comp / gsys / class heads on the smoke
H5 to validate multi-task plumbing.

---

## 2026-05-12 — Optuna HPO is on the deferred list; upstream code exists

**Q:** Where does Optuna hyperparameter optimization fit in the v1 roadmap?

**Context:** v0 training used hand-picked hyperparameters (d_model=256, N=2,
heads=8, lr=3e-4 — defaults inherited from upstream BaseCrabNet). These were
fine for architecture validation but almost certainly not optimal for the
Position C training corpus we're producing. Per-phase R² values in the v0
run vary widely (olivine 0.99 vs amphibole 0.18), and some of that spread
likely reflects hyperparameter mismatch rather than fundamental task
difficulty.

**Decision:** Optuna integration is a deferred v0.5 / v1 task — **NOT v0**.
v0 ships with hand-picked hyperparameters and characterizes ModeHead
behavior at scale. Optuna comes after we've:
1. Trained at full MAGEMin scale (~440k points) with v0 hyperparameters
2. Decided whether the next bottleneck is hyperparameters, data quality, or
   architecture
3. Optionally added multi-task heads (which add their own hyperparameters)

**Upstream code to port from:** The CrabNet-Mineral repo
(`~/git_clones/CrabNet_PyPi/CrabNet-Mineral/`, branch `structure-edm`) has
working Optuna integration we should adapt rather than rewrite:
- `alignn/src/objective.py` — Optuna objective functions for BaseCrabNet
  and StructureCrabNet, including k-fold CV inside the objective
- `alignn/scripts/run_scaling.py` — multi-scale Optuna study orchestration
  (runs HPO at each data scale separately, since optimal hyperparams
  shift with data volume)
- Storage: they use Optuna's `JournalStorage` (filesystem-based) rather
  than SQLite — important on parallel filesystems (HPC Lustre/GPFS) where
  SQLite breaks. We should follow the same convention.
- Their pattern: separate Optuna study DB per (model, scale) combination,
  with `tag-prefix` arg controlling naming. Worth mirroring once we have
  multiple model variants (base / multi-task / Dirichlet output / ensemble).

**Scope for porting (when we get to it):**
- `src/cnmineral/models/objective.py` — Optuna objective for `BaseModel`,
  later for the multi-head variant. Reuse k-fold CV pattern from upstream
  but adapt to our stratified-by-qap_name split.
- `scripts/04_hpo.py` — thin wrapper invoking the objective with a
  configurable n_trials and study storage path
- `configs/hpo_base.yaml` — search-space specification (d_model in
  {128, 256, 512}, N in {1, 2, 3}, heads in {4, 8}, lr log-uniform in
  [1e-4, 1e-2], dropout in [0.0, 0.3], lambda_sum in [0.001, 0.1])
- JournalStorage path: `data/optuna/<study_name>.journal`. Gitignore the
  `data/optuna/` directory.

**Why not earlier:** Optuna at v0 would have meant tuning hyperparameters
on the smoke H5 (2,545 points) and likely getting different optima than
at full scale (440k). Their `run_scaling.py` showed this directly —
optimal `d_model` shifts as data scale changes. Better to characterize
the architecture's behavior at v1 scale first, then tune.

**Next:** No action this session. Note left so future sessions know HPO
is a planned step and where to find the reference code.

---

## 2026-05-13 — Full-corpus training: architecture validated, label-design issue surfaced

**Q:** Does the v0 architecture generalize to ~170× more training data, and what
does it tell us about the next direction?

**Context:** Trained the same `BaseModel` (kingcrab.SubCrab wrapped) on the full
MAGEMin-derived H5 (447,368 samples, 25 phases). Three pipeline issues handled
during data generation:
- Deterministic SIGSEGV in `libMAGEMin.dylib reset_SS` on 5 of 92 chunks. Solved
  by subdivision retry (`scripts/retry_failed_chunks.py`): split each 5,000-row
  failed chunk into 10 sub-chunks of 500 rows; ~90% per chunk recovered. Net
  loss ~2,500 of 457,000 rows = 0.55%.
- Switched device selection to prefer MPS over CPU (`pick_device()` helper).
  ~9× speedup vs CPU.
- Bumped batch size to 1024 for MPS efficiency.
- Added tqdm batch-level progress bars to `train_one_epoch` and `evaluate`.

**Training results:**
- Wall time: 996.2s (16.6 min) on MPS. Best epoch 26, early-stop at 41 with
  patience 15. Smoke trained in 97s for 95 epochs on CPU.
- Best val loss: **0.00115** (vs smoke 0.00147 — 22% improvement)
- Sparsity 0.85, sum-to-1 violation 0.005
- Train-val gap at best epoch: +0.000042 — **no overfitting**

**Per-phase R² scaling (smoke vs full):**

| Phase | Pres (smoke / full) | R² (smoke / full) | Δ R² |
|---|---|---|---|
| liq | 0.94 / 0.97 | 0.82 / 0.82 | ~flat |
| fsp | 0.51 / 0.52 | 0.69 / 0.64 | -0.05 |
| cpx | 0.39 / 0.35 | 0.72 / 0.70 | flat |
| **spl** | 0.37 / 0.32 | **0.36** / **0.82** | **+0.46** |
| opx | 0.21 / 0.24 | 0.88 / 0.88 | flat |
| ol | 0.24 / 0.20 | 0.99 / 0.99 | flat |
| amp | 0.10 / 0.11 | 0.18 / 0.47 | +0.29 |
| fl | 0.11 / 0.10 | 0.30 / 0.53 | +0.23 |
| **cd** | 0.07 / 0.08 | **0.19** / **0.88** | **+0.69** |
| bi | 0.09 / 0.07 | 0.81 / 0.82 | flat |
| ilm | 0.06 / 0.04 | 0.17 / 0.39 | +0.22 |
| q | 0.04 / 0.04 | 0.46 / 0.69 | +0.23 |
| **sill** | 0.03 / 0.03 | **-0.09** / **0.72** | **+0.81** |
| ne | 0.04 / 0.02 | 0.64 / 0.97 | +0.34 |
| **fper** | 0.03 / 0.02 | **-0.02** / **0.89** | **+0.91** |
| g | 0.02 / 0.02 | 0.05 / 0.50 | +0.45 |
| **sph** | 0.01 / 0.01 | **-1.89** / **0.63** | **+2.52** |

**Pattern:** Phases that had <5% presence at smoke scale (sph, sill, fper, g,
cd, cor) had random or negative R² — too few samples to learn from. At 170×
more data, they all become learnable, with R² 0.50-0.89.

**14 of 17 phases with > 1% presence now have R² > 0.5.** Three exceptions:
- `fsp` 0.64 (lower than smoke 0.69) — possibly affected by class redistribution
- `amp` 0.47 (still the worst >10%-presence phase) — within-class composition
  variance is plausibly the cause; worth investigating in v1
- `ilm` 0.39 — fewer training samples than expected (4.2% presence vs smoke 5.7%)

**Conclusion: architecture scales. No bottleneck on the model side at this size.**

---

**Label-design problem surfaced during inspection:**

Investigated `liq` more carefully — it's not just present in many samples,
it's actually the dominant phase by mass.

```
liq presence (>0.01):     97.4%
liq presence (>0.50):     76.1%
liq presence (>0.95):     27.9%
mean liq mass fraction:   0.697
fraction of total training-set mass that is liq: 70%
```

**70% of the predicted "rock" is silicate melt, not crystals.** This isn't a
MAGEMin bug or H2O-strategy bug — it's an upstream sampling issue. We sample
T from Putirka 2008 ranges, which are *crystallization temperatures* —
conditions where the magma was actively crystallizing, by definition with
substantial melt. Adding H2O (fixed amp/bi gap correctly) made the solidus
lower → even more melt at the same T.

For users of this model — people mining rocks and running ICP/XRF on
surface samples — the target is **room-temperature mineralogy**, not in-situ
crystallization snapshots. Our labels are answering the wrong question.

For different cooling histories:
- **Volcanic** (basalt, andesite, rhyolite — quenched at eruption): melt
  becomes glass on rapid cooling. `liq` IS conceptually a real component of
  the room-T sample. Model output here is approximately correct.
- **Plutonic** (granite, gabbro, diorite — slow-cooled at depth): melt fully
  crystallizes during cooling. `liq` in the prediction corresponds to nothing
  in the room-T sample.
- **Sedimentary**: not in our training corpus (rock_group == 'igneous' filter).

The simple post-processing fix (drop liq, renormalize) reshuffles 70% of the
predicted mass into a ~30% remainder. Not a defensible cleanup.

**Decision (locked, see decision 004 to be written):** Regenerate labels at
**subsolidus T** rather than crystallization T. Implementation:
- Add `SubsolidusTPStrategy` to `tp_priors.py` with per-class T ranges below
  the H2O-saturated solidus (granite ~500-650°C, basalt ~800-950°C, ultramafic
  ~700-1000°C, etc.)
- Re-run filter (unchanged bulk selection) → sample_tp (new strategy) →
  orchestrate → parse → retrain
- Validation target: liq mean mass fraction collapses to near-zero; per-phase
  modes correspond to what users observe in surface-collected rocks

Three rejected alternatives:
- Bracket the solidus per bulk (more accurate but ~5× compute cost; defer)
- Post-process the current labels (defensibility issue per above)
- Add a Glass head separately (only helps volcanic cooling; doesn't fix
  plutonic-rock predictions)

**Action items:**
1. Patch `scripts/03_train_base.py` to write more per-epoch info to disk
   (unfilter r2_per_phase, add wall-time, auto-write eval.json, metadata header)
2. Add `SubsolidusTPStrategy` to `src/cnmineral/data/tp_priors.py`
3. Regenerate labels: full pipeline with `--strategy subsolidus`
4. Retrain on subsolidus H5; compare per-phase R² and `liq` mass fraction
   against the current crystallization-T results
5. Then ship v1 paper-side claim: "model predicts surface-relevant modal
   mineralogy from ICP/XRF bulk chemistry"

**Insight worth carrying forward:** the v0 result *does* validate the
architecture-to-data interface (BaseModel learns this kind of regression at
scale, 14 of 17 R² > 0.5, no overfit). The subsolidus regeneration changes
**what the model predicts**, not whether the architecture works. So the v0
training was not wasted — it's a methodology baseline ("this is what
crystallization-T predictions look like") for the eventual paper's
methods section.

**Next:** Patch training script (action 1), then build SubsolidusTPStrategy
(action 2).

---

## 2026-05-13 — Small infra notes from the full-corpus run (parking for future debugging)

Brief observations not worth their own decision docs but useful if anything
behaves unexpectedly later:

**OpenMP / `KMP_DUPLICATE_LIB_OK` on macOS.** Both PyTorch and MAGEMin link
`libomp.dylib`. When MAGEMin runs concurrently with a PyTorch script, the
second process aborts at startup with "OpenMP already initialized." Fix is
documented in the handoff brief but I missed it initially. Now
`os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` is set at the top of
`scripts/03_train_base.py` and `scripts/eval_checkpoint.py` to make those
scripts immune to whether MAGEMin happens to be running. Symptom if this
ever breaks again: SIGABRT immediately after the script imports torch.

**Deterministic segfaults in `libMAGEMin.dylib reset_SS`.** ~5% of chunks
in the full-corpus run crashed deterministically inside MAGEMin's
solid-solution reset code. Runtimes within 0.5% of each other across re-runs;
the bug is data-deterministic (specific composition × T × P rows trigger an
edge case in the SS solver). Subdivision retry (`scripts/retry_failed_chunks.py`)
splits each failed 5,000-row chunk into 10 sub-chunks of 500 — only the
sub-chunk containing the problematic row crashes, recovering ~90% of rows
per failed chunk. Total loss at full scale: ~2,500 of 457,000 rows (0.55%).
The same bug almost certainly exists in upstream Stage 2 runs; if Sarah/the
other agent ever scales there, recommend porting our subdivision script.

**MPS device selection + batch size 1024.** Initial training ran on CPU
because the script's device check was CUDA-only. Adding `pick_device()` to
fall through CUDA → MPS → CPU + bumping `batch_size: 256 → 1024` gave a
roughly 20× speedup on the full corpus (5.5 hr CPU estimate → 16.6 min on
MPS in practice). Larger speedup than my predicted 4-6×; partly the MPS
backend and partly the batch-size reduction in per-batch Python overhead.

**tqdm batch-level progress bars** in `train_one_epoch` and `evaluate`.
`progress=True` flag (default) wraps the loader with `tqdm(..., leave=False)`
so the bar disappears at end of epoch and the per-epoch summary stays in
scrollback. Live `loss=` postfix updates per batch. Was originally absent;
the user requested it after seeing how long training ran without visible
within-epoch progress.

These are mostly small but each cost real debugging time when they bit us.
Documenting so future-me doesn't re-derive them.

---

## 2026-05-13 — Revised near-solidus T ranges (correction to earlier table)

**Q:** What T ranges to use for the regenerated training labels?

**Context:** Earlier journal entry "Full-corpus training: architecture
validated, label-design issue surfaced" proposed wider, deeper-subsolidus
T ranges as a fix (granite 400-600, basalt 700-900, etc.). Subsequent
external assessment via ChatGPT pushed back: at very low T, MAGEMin's `ig`
database leaves its calibrated regime and produces metamorphic-style
assemblages (greenschist: chl, ep, ms) that don't reflect what users see
in granitic surface samples. The 400°C floor on granite is in this drift
regime.

**Revised ranges (use these, not the earlier table):** Pick T ranges 0-150°C
*below* the wet solidus per class — the "kinetic arrest" regime where real
igneous rocks stopped reacting during cooling and where MAGEMin's `ig`
calibration is still trustworthy:

| Class | Near-solidus T (°C) | Wet solidus (~) |
|---|---|---|
| Granite (I/S-type) | 550-650 | 650 |
| Granodiorite, tonalite | 600-700 | 700 |
| Rhyolite | 550-700 | 700 |
| Dacite | 650-800 | 800 |
| Andesite | 750-900 | 900 |
| Basalt (all variants) | 850-1000 | 950-1000 |
| Gabbro (all variants) | 850-1000 | 1000 |
| Komatiite, picrite | 950-1100 | 1100 |
| Dunite, harzburgite, lherzolite | 900-1100 | 1100 (wet) |
| Foidite, nephelinite | 850-1050 | 1100 |
| Trachyte, phonolite | 650-800 | 750 |

**Rationale:** Stays within MAGEMin's calibrated range. Avoids metamorphic
drift. Captures the "frozen-at-kinetic-arrest" state that approximates what
users observe in surface samples. Each range is wide enough (~150°C) to
preserve some marginalization, but tight enough to be class-specific.

**Status:** Proposed but **not yet implemented in code**. The PetThermoTools
experiment described in the next journal entry should run first — it might
indicate that subsolidus equilibrium is too biased a proxy for the
fractional-crystallization endpoint, in which case we'd go to fractional
crystallization directly rather than implementing the near-solidus strategy.

---

## 2026-05-13 — PetThermoTools experiment design: equilibrium vs fractional crystallization

**Q:** Before committing to near-solidus equilibrium as the label-generation
strategy, is its modal-mineralogy output actually a good proxy for the
endpoint of proper fractional crystallization?

**Context:** Two label-generation strategies for the room-temperature
crystalline assemblage of igneous bulk samples:

- **(A) Equilibrium at near-solidus T** (the "Phase A" fix): one MAGEMin call
  per (bulk, T_ns, P). Cheap. Treats the system as closed at T_ns. Every
  phase has uniform composition (no zoning, no fractionation memory).
- **(B) Fractional crystallization** ("Phase B"): step-cool the bulk from
  liquidus to subsolidus, removing some fraction of solids at each step.
  The integrated record includes zoned crystals (early Fo-rich olivine +
  late Fo-poor olivine) and disproportionately Q-Kfs-heavy terminal solids
  for granitic systems. More physically correct but ~50× more compute and
  requires either PetThermoTools or a custom Julia loop.

I had asserted earlier that "central tendency converges between (A) and (B)."
That assertion was sloppy. It's plausible for basalts/gabbros where the
assemblage is roughly invariant across the cooling T range, but probably
wrong for granitic systems where the residual melt evolves substantially.
We don't know which class it's wrong for and by how much, so the assertion
shouldn't be load-bearing for the v1 plan.

**Experiment:** Empirically measure modal-mineralogy divergence between (A)
and (B) across rock classes.

**Design:**
- Sample ~50 bulks per rock class from existing `magemin_inputs.csv`.
  Classes to include: basalt, andesite, dacite, rhyolite, granite, gabbro,
  diorite, dunite, foidite, syenite (the petrologically important spectrum).
- Per bulk, run two MAGEMin scenarios:
  1. **Equilibrium-at-T_ns**: single MAGEMin call at the near-solidus T per
     the revised table above. Record modal vector + per-phase composition.
  2. **Fractional crystallization**: start at T_liquidus + 50°C, cool in ~25°C
     steps to T_ns, removing 90% of solids per step. Final modal output =
     accumulated removed solids + remaining solids at termination, normalized.
- Compare with L1 distance on the modal vectors: `Σ |mode_eq[i] − mode_frac[i]|`.
  Per-phase delta (signed) tells us *which* phases the equilibrium proxy
  over- vs under-predicts.
- Aggregate per class: median + IQR of L1 distance, mean per-phase delta.

**Tooling decision: use PetThermoTools.** Two candidates were considered:

- *Custom Julia loop* in `scripts/julia/fractional_crystallization.jl`: ~50
  lines wrapping our existing `point_wise_minimization` call. Cheap, transparent.
- *PetThermoTools* (https://petthermotools.readthedocs.io): community-vetted
  Python wrapper around MAGEMin_C.jl with `Frac_solid=True` flag and proper
  defaults (cooling step size, solid removal fraction, liquidus search).

We chose PetThermoTools. Reason: if the experiment shows fractional
crystallization is needed for production label generation (Phase B), we'd
want PetThermoTools anyway for its solidus-bracketing and step-size logic.
Investing the integration effort once amortizes over both the experiment
and potential v1.5/v2 production work. The custom Julia loop saves
implementation time short-term but doesn't reuse.

See ADR `docs/decisions/004_petthermotools_for_fractional_crystallization.md`.

**Decision rule based on results:**
- **Median L1 distance < 0.10 across all classes**: subsolidus equilibrium
  (Phase A) is a defensible proxy. Implement `NearSolidusTPStrategy` per the
  revised T table, regenerate corpus, retrain.
- **Median L1 distance > 0.25 for one or more classes** (likely granite per
  the petrologic intuition): proxy is too biased. Skip Phase A, go straight
  to PetThermoTools-generated labels (Phase B) for production.
- **In between**: hybrid — Phase A for mafics, Phase B for felsics (since the
  computational cost is class-localized; doing fractional crystallization on
  only the felsic ~30% of the corpus is much cheaper than doing it on all).

**Cost estimate:**
- PetThermoTools install + integration: ~2-3 hours
- MAGEMin compute: 500 bulks × ~21 calls each (1 equilibrium + 20 fractional
  steps) = ~10,500 calls × ~11 ms = ~2 min actual compute
- Comparison + analysis script: 1-2 hours
- Journal entry + plots: 1 hour
- **Total: ~half a day.** Much cheaper than re-running production training
  on bad labels twice.

**Implementation location:** `scripts/experiments/eq_vs_frac/` (new subdir, so
the experiment script stays separate from the main pipeline). Outputs to
`data/experiments/eq_vs_frac/{results.csv, plots/}`.

**Next session priorities (cold-start for the fresh agent):**
1. Clone PetThermoTools to a path of user's choosing (user is doing this)
2. Plan integration: how to call from our Julia env, output format adaptation,
   etc. Enter plan mode for this.
3. Implement `scripts/experiments/eq_vs_frac/run_comparison.py` + supporting
   Julia bits.
4. Run experiment on a small sample first to validate, then full ~500-bulk run.
5. Analysis + journal entry with results table.
6. **Then decide Phase A vs Phase B vs hybrid for production label generation.**

The experiment is the *gate* before committing to either Phase A or Phase B.
Don't implement `NearSolidusTPStrategy` (the Phase A code) until the
experiment validates that approach.

---

## 2026-05-13 — PetThermoTools integration + eq-vs-frac experiment results: adopt fractional crystallization

**Q:** Is equilibrium-at-near-solidus a good enough proxy for the
fractional-crystallization endpoint to use as production labels?

**Context:** Integrated PetThermoTools v0.4.5 (Gleeson, UC Berkeley) into the
`alignn_fix` env via pip-editable install of `~/git_clones/PetThermoTools`.
Ran `ptt.install_MAGEMinCalc()` which provisioned an isolated Julia env at
`~/.petthermotools_julia_env` (juliapkg auto-fetched a Julia 1.11.9 sidecar
alongside our existing 1.12.6) with `MAGEMin_C@2.1.5 + MAGEMinCalc@v0.6.1`.
The isolated Julia env does NOT conflict with our `scripts/julia/Project.toml`
pin. `ptt.test_MAGEMinCalc()` passed. The crash printed at the end of
`install_MAGEMinCalc` (`curl_multi_assign` + SIGSEGV during `Pkg.precompile()`
cleanup) is a known cosmetic bug in juliacall on macOS — MAGEMin_C is fully
functional.

Two implementation surprises along the way:

1. **DataFrame-bulks batching is broken for ptt's MAGEMin backend.** The
   wrapper's docstring claims `bulk: dict | DataFrame`, but in practice ptt's
   PythonCall path raises `Dict(kv::PyPandasDataFrame)` when handing a
   DataFrame to MAGEMinCalc. Fell back to a per-bulk serial loop.

2. **`find_liquidus=True` is the dominant cost (~15 s/bulk, ~85% of wall).**
   Replacing it with a class-conditional fixed `T_start_C` (set ~50-100 C above
   each class's Putirka upper-bound liquidus) cut per-bulk wall to 2-3 s.
   Verified on a basalt + a granite that the FC endpoint modal vector matches
   `find_liquidus=True` to L1 < 0.005 — empirically equivalent. Production
   cost: 550 bulks in 26 min.

**Experiment:** `scripts/experiments/eq_vs_frac/`. 550 bulks across 11 classes
(50 per class, gabbro capped at 98 available). Equilibrium leg used our
existing `run_magemin.jl` (database `ig`, same Project.toml pin as the main
pipeline) at near-solidus T midpoints. Fractional leg via ptt with `Model =
"Green2025"`, `Frac_solid=True`, `dt_C=25`, `T_end_C = T_lo` of the
near-solidus range per class. Both legs use the same 22 bulks per class, same
H2O strategy already applied to `magemin_inputs.csv`. Convergence: 526/550 eq,
547/550 FC.

**L1 distance, equilibrium vs fractional modal vectors (per class, median):**

| Class | N | median L1 | mean L1 | IQR |
|---|---|---|---|---|
| rhyolite     | 49 | 0.131 | 0.153 | [0.11, 0.17] |
| monzogranite | 49 | 0.157 | 0.195 | [0.12, 0.26] |
| granodiorite | 50 | 0.214 | 0.250 | [0.19, 0.29] |
| basalt       | 50 | 0.292 | 0.293 | [0.18, 0.41] |
| andesite     | 50 | 0.376 | 0.418 | [0.27, 0.54] |
| syenite      | 50 | 0.411 | 0.495 | [0.23, 0.69] |
| foidite      | 32 | 0.413 | 0.527 | [0.31, 0.58] |
| dunite       | 47 | 0.486 | 0.633 | [0.04, 1.04] |
| diorite      | 49 | 0.547 | 0.529 | [0.27, 0.70] |
| gabbro       | 50 | 0.674 | 0.688 | [0.50, 0.86] |
| **dacite**   | 50 | **1.144** | **1.138** | [0.90, 1.36] |

Overall median 0.350, mean 0.482. Plots at
`data/experiments/eq_vs_frac/plots/{l1_by_class.png, phase_delta.png}`.

**Decision per ADR 004 rule:** dacite median 1.14 ≫ 0.25 threshold; 9/11
classes exceed 0.25; only rhyolite is anywhere near the 0.10 "adopt eq"
threshold. **Adopt fractional crystallization (Phase B) for production label
generation.** ADR 004 status updated to Resolved → Phase B.

**Three findings the experiment surfaced that the petrologic intuition got
wrong:**

1. **Felsic classes were the *best* match, not the worst.** Earlier
   reasoning (journal "PetThermoTools experiment design") predicted granitic
   systems would diverge most because their residual melt evolves
   substantially across cooling. Empirically rhyolite (0.13) and monzogranite
   (0.16) are the lowest-L1 classes. Plausible explanation: felsic eutectics
   are narrow — both legs converge on the same near-eutectic qz+fsp+bi
   assemblage. The melt's compositional drift mostly happens in the *liq*
   channel of the eq snapshot, which discharges into qz+fsp on either path.

2. **Dacite is catastrophic at L1=1.14.** This wasn't predicted by anyone.
   The signed-delta heatmap shows the eq leg retains ~50% mass in `liq` while
   the FC leg has redistributed that mass into fsp (+0.3) + minor solids. The
   dacite near-solidus T (650-800 C, midpoint 725 C) appears to sit *above*
   the wet solidus for many dacite bulks given their H2O loadings, so the eq
   snapshot is essentially "magma in progress" again — exactly the failure
   mode we set out to fix. Lowering T further would push dacite eq into
   MAGEMin's metamorphic-drift regime. There's no T window where eq is both
   thermodynamically calibrated and below the wet solidus for hydrous dacite.
   This is the cleanest argument for FC.

3. **The dominant signed delta across most classes is `liq` -0.2 to -0.5 with
   compensating `fsp` and ferromagnesian gains.** The eq leg systematically
   over-predicts melt mass at the chosen near-solidus T. FC redistributes
   that mass into the realistic crystalline endpoint. This is the same
   failure mode the original v0 labels had (mean liq 0.70 in the crystallization-
   T corpus) — just less severe. Near-solidus equilibrium reduces but does
   not eliminate the melt-inflation problem.

**Implications for v1:**
- Production label generation switches to FC via PetThermoTools. Drop the
  `NearSolidusTPStrategy` idea from the May 12 / May 13 entries — don't
  implement it.
- Per-bulk cost ~3 s on this machine; ~22k bulks × ~20 (P, H2O) samples per
  bulk would be ~13 wall-hours single-process. Need to parallelize across
  bulks via separate python workers (ptt's intra-process multi_processing
  doesn't help with single-bulk calls) or accept the wall time.
- The FC modal vector schema is wider than eq (FC produces `ep`,
  occasionally other phases eq didn't show in this sample). The full
  production phase vocabulary should be discovered fresh at scale and may
  differ from the v0 25-phase set; check before retraining.
- Single-T per bulk loses the multimodal-posterior framing of the v0
  pipeline. Phase B at production should sample multiple (P, H2O, fO2)
  combinations per bulk to preserve the many-to-one assemblage distribution
  the model is meant to learn.

**Next:**
1. Update CLAUDE.md "Active concern" + "Status snapshot" to reflect the
   resolved label-strategy question.
2. ADR 005: production label generation via PetThermoTools FC — settings,
   sampling strategy, performance budget.
3. Design the production FC orchestration (chunking, retry, error
   handling — mirroring the existing MAGEMin chunked pipeline).
4. Regenerate corpus → retrain → compare to v0 baseline.

**Files added:**
- `configs/eq_vs_frac.yaml`
- `scripts/experiments/eq_vs_frac/{01_sample_bulks, 02_run_equilibrium, 03_run_fractional, 04_compare}.py`
- `scripts/experiments/eq_vs_frac/_phase_map.py`
- `environment.yml` adds `juliacall`, `juliapkg`, editable install of
  PetThermoTools (sibling clone)
- `.gitignore` excludes `data/experiments/`
- `data/experiments/eq_vs_frac/{bulks.parquet, equilibrium.parquet, fractional.parquet, results.csv, plots/}`
   (gitignored)

---

## 2026-05-15 — eq-vs-frac v2: anchored terminal T + liq-threshold-driven T_FC_end

**Q:** v1's L1s mixed three different effects (path-dependence, terminal-T
gap between eq and FC, eq running above the wet solidus for hydrous bulks).
What does the comparison look like when those confounds are removed?

**Context:** Post-v1 audit (`docs/eq_vs_frac_diagnostics.md`) showed that:
- eq ran at T_midpoint, FC ended at T_lo — a 50-100 C gap that contaminated
  the L1 with terminal-T differences.
- For hydrous classes (dacite especially), the eq T sat *above* the wet
  solidus — eq was "magma in progress" while FC was at the crystallized
  endpoint. Most of dacite's L1=1.14 was this artifact, not path-dependence.
- liq_eq vs liq_fc at the chosen T's were grossly mismatched in some classes
  (dacite eq=0.45, FC=0.00; syenite eq=0.17, FC=0.01).

**v2 design changes:**

1. **Terminal T anchoring.** Both legs run at the same `T_FC_end_class` per
   class. The eq leg no longer uses T_midpoint.
2. **Threshold-driven T selection.** For each class, pick the highest T (snap
   to nearest 25 C) where the class-aggregate mean `liquid1` from a per-bulk
   FC sweep falls below 5 wt%. This empirically defines "subsolidus enough"
   per the bulk H2O distribution rather than relying on literature wet-solidus
   tables.
3. **Per-bulk audit.** Bulks whose own `liquid1` at the picked T still exceeds
   5 wt% are excluded from the L1 comparison (but kept in the underlying
   modal data).
4. **In-situ L1.** All phases including `liq` and `fl` stay in the modal
   vector. The L1 measures "do these two procedures agree on the system
   state at this T?" not "do they agree on eventual surface mineralogy?"

`liq` vs `fl` distinction matters here: `liq` is silicate melt, `fl` is a
free aqueous fluid. Below the wet solidus, melt → 0 but excess H2O that
doesn't fit in amp/bi/mu lives as fl. Threshold gates on `liq` only —
forcing `fl` to zero would push hydrous bulks below MAGEMin's calibrated
range.

**Procedure:**

1. `01_sweep_fc.py`: per-bulk FC from T_start down to T_floor_v2 (initially
   525-800 C per class, adjusted to 600-525-800 after the basalt/gabbro/foidite
   classes hit the floor with mean liq still > 5%).
2. `02_pick_T_and_extract_fc.py`: derives `T_FC_end_class` from the sweep
   trajectory, audits, extracts FC modal at that T (cumsum solids + residual
   liq + fluid).
3. `03_run_eq_v2.py`: shells out to `run_magemin.jl` at `T_FC_end_class` per
   bulk's class. 526/550 converged in 3.6 s.
4. `04_compare_v2.py`: L1 over full phase union including liq + fl.

547/550 bulks converged on FC; the floor adjustment for basalt/gabbro/foidite
(initial 725-750 C → 600 C) brought their mean liq under the threshold.
Final per-class T's:

| class        | T_FC_end_class (C) | mean liq @ T (wt%) | p95 liq | n flagged |
|--------------|---------------------|---------------------|---------|-----------|
| basalt       | 675                 | 3.6                 | 9.6     | 15        |
| andesite     | 650                 | 0.0                 | 0.0     | 0         |
| dacite       | 650                 | 0.0                 | 0.0     | 0         |
| rhyolite     | 650                 | 0.0                 | 0.0     | 0         |
| monzogranite | 650                 | 0.6                 | 0.0     | 2         |
| granodiorite | 650                 | 0.0                 | 0.0     | 0         |
| diorite      | 650                 | 0.0                 | 0.0     | 0         |
| gabbro       | 700                 | 3.7                 | 14.0    | 15        |
| dunite       | 975                 | 4.7                 | 15.1    | 19        |
| foidite      | 625                 | 4.9                 | 33.0    | 7         |
| syenite      | 650                 | 0.8                 | 3.9     | 1         |

59 of 547 bulks flagged (11%); excluded from L1. Most felsic and intermediate
classes converged on 650 C — that T is below the wet solidus for those H2O
loadings. Basalt/gabbro hold residual fractionation-trapped evolved melt
down to 675-700 C. Dunite picked 975 C because its 5%-quantile cools through
mean-<5% earlier than its 95%-quantile.

**L1 results (in-situ, liq + fl included, post-audit):**

| class        | n   | median L1 | mean L1 | v1 median | Δ      |
|--------------|-----|-----------|---------|-----------|--------|
| monzogranite | 47  | **0.038** | 0.071   | 0.157     | -0.119 |
| dunite       | 28  | **0.070** | 0.412   | 0.486     | -0.416 |
| rhyolite     | 49  | **0.084** | 0.098   | 0.131     | -0.047 |
| syenite      | 49  | 0.127     | 0.249   | 0.411     | -0.284 |
| granodiorite | 50  | 0.166     | 0.193   | 0.214     | -0.048 |
| dacite       | 50  | 0.288     | 0.336   | 1.144     | **-0.856** |
| andesite     | 50  | 0.467     | 0.550   | 0.376     | +0.091 |
| basalt       | 35  | 0.609     | 0.623   | 0.292     | +0.317 |
| gabbro       | 35  | 0.715     | 0.803   | 0.674     | +0.041 |
| diorite      | 49  | 0.719     | 0.728   | 0.547     | +0.172 |
| foidite      | 26  | 0.787     | 0.803   | 0.413     | +0.374 |

Overall: v1 median 0.350, v2 median 0.324. Mean v1 0.482, v2 0.411.

**Three groups emerge cleanly in v2:**

- **Path-irrelevant (median L1 < 0.10): monzogranite, dunite, rhyolite.**
  Closed-system eq @ 650-975 C and open-system FC at the same T converge on
  the same near-eutectic / cumulate assemblage. Felsic shallow systems sit on
  a narrow eutectic; cooling path doesn't reorder the final phases. Dunite is
  thermodynamically simple (ol + pyx + sp) and produces the same modal whether
  closed or open.
- **Mild path effect (0.10-0.25): syenite, granodiorite.** Both legs agree
  qualitatively but differ in modal proportions by 10-15%. Acceptable for v1
  production labels.
- **Strong path effect (>0.25): dacite, andesite, basalt, gabbro, diorite,
  foidite.** Fractionation-accumulated Mg-rich olivine + early cpx + zoned
  feldspars yield a substantially different assemblage from the closed-system
  snapshot. Plutonic mafic-intermediate compositions are the most affected
  (gabbro 0.72, diorite 0.72, foidite 0.79). These are exactly the classes
  for which eq labels would mislead the model.

**Decision (confirms ADR 004):** adopt fractional crystallization for
production label generation. Worst class median (foidite 0.79) is 8x the
0.10 "adopt eq" threshold; 6 of 11 classes are above 0.25.

**Refinements to the production strategy implied by v2:**

- Felsic shallow (rhyolite, granite, granodiorite): a hybrid path *could*
  work — eq labels at near-eutectic T would lose <0.15 L1. But for pipeline
  simplicity, use FC everywhere; the marginal compute cost is small.
- The T_FC_end_class table here (650 C for most felsic/intermediate, 675-700
  for basalt/gabbro, 975 for dunite, 625 for foidite) is a starting point for
  production. Production may want per-bulk T based on each bulk's individual
  liq trajectory rather than class-aggregate T's; this avoids the 11% audit
  attrition we hit here.
- The 5% threshold is reasonable. A 1% threshold would push 3-4 more classes
  toward MAGEMin's metamorphic-drift floor; not worth the marginal "cleaner
  subsolidus" gain.

**Files added under `scripts/experiments/eq_vs_frac_v2/`:**
- `01_sweep_fc.py` (FC trajectory sweep, supports `--classes` filter and
  `--append` for targeted re-runs)
- `02_pick_T_and_extract_fc.py` (T_FC_end_class derivation + audit + FC modal
  extraction)
- `03_run_eq_v2.py` (eq leg at the picked T's via run_magemin.jl)
- `04_compare_v2.py` (L1 comparison + plots)
- Outputs in `data/experiments/eq_vs_frac_v2/`: `fc_sweep.parquet` (13670
  rows × 92 cols), `T_FC_end_table.csv`, `audit_flags.parquet`,
  `equilibrium_v2.parquet`, `fractional_v2.parquet`, `results_v2.csv`,
  `plots/{l1_by_class_v2,phase_delta_v2}.png` (all gitignored).

Config additions to `configs/eq_vs_frac.yaml`: per-class `T_floor_v2` and a
`v2:` block with `liq_threshold: 0.05`, `T_quantize_C: 25`, `dt_C: 25`.

**Next:**
1. Update `CLAUDE.md` "Active concern" / "Status snapshot" to reflect resolved
   label-strategy question.
2. ADR 005: production label generation via PetThermoTools FC — per-bulk T
   selection algorithm, (P, H2O, fO2) sampling design, chunking/retry
   infrastructure, performance budget.
3. Design production FC orchestration. ~22k bulks × ~20 (P, H2O) samples ×
   ~3 s = ~13 wall-hours single-process; needs python-level worker pool.
4. Regenerate corpus → retrain BaseModel → compare to v0 baseline.

---

## 2026-05-17 to 05-19 — Production corpus generation: memory leak crisis and resolution

**Q:** Generate the v1 production training corpus (~178k path-rows) per ADR 005.

**Context:** Built `scripts/production/01_find_T_FC_end_per_bulk.py` (pre-pass)
and `02_generate_corpus.py` (orchestrator). Pre-pass ran cleanly: 22,359 bulks,
21,446 clean (95.9%) at ~3 s/bulk over ~5 hr wall.

Orchestrator crashed laptop three times. Diagnosed via macOS Jetsam logs at
`/Library/Logs/DiagnosticReports/JetsamEvent-*.ips`: workers grew from ~7 GB
to ~180 GB RSS over ~4 hr. Kernel-level memory pressure → Jetsam cascade →
WindowServer crashes → full reboot. Root cause is a **per-call memory leak
in juliacall + MAGEMin_C / MAGEMinCalc** that's invisible at petrology-session
scale (10³ calls) but catastrophic at ML-corpus scale (10⁶ calls).

**Diagnosis layers:**

1. First hypothesis: VSCode + Pylance file watcher amplification. Settings.json
   exclusions for `data/` quieted that but didn't fix the crash.
2. Second hypothesis: `maxtasksperchild=1` on `mp.Pool` would recycle workers
   between chunks. Tested at smoke scale and confirmed memory recycling works.
   But chunk_size=100 → 800 FC calls per chunk → ~36 GB growth per chunk per
   worker → 4 workers × 36 GB = 144 GB peak, still exceeded 128 GB RAM.
3. Third fix layer: `chunk_size: 100 → 25`. 200 calls per chunk → ~9 GB
   growth per chunk per worker → 4 workers × 10 GB = 40 GB peak. Comfortable.
4. Defense-in-depth: explicit `juliacall.Main.GC.gc(True)` between paths in
   `process_chunk`. Slowed leak rate by ~36% but doesn't eliminate it (the
   leak is in C-level mmap regions juliacall's GC can't reclaim).

**Final settings (locked):**
- `--workers 4 --chunk-size 25` with `maxtasksperchild=1` on the `mp.Pool`
- Explicit Julia GC call between FC paths in `process_chunk`
- `caffeinate -i` to prevent macOS idle sleep during the long run
- Detached via `nohup` so VSCode close / IDE crashes don't kill the orchestrator
- `.vscode/settings.json` excludes `data/` from Pylance + file watcher

**Result:** 730 of 731 chunks completed cleanly (one chunk hung at the very
end on what appears to be a juliacall init deadlock, lost ~200 path-rows of
the ~146k total — negligible). Total wall time ~31 hours including the three
crashes. The user manually killed the hung orchestrator at user request to
recover; `merge_chunks.py` produced `data/train_fc_v1.parquet`.

**Final corpus stats:**
- **177,976 path-rows** across 22,245 unique bulks × 8 paths
- **64 rock classes** (basalt-dominated per Gard et al. 2019 natural prevalence)
- **18 mineral phases** discovered in the modal vocabulary
- **Mean liq mass fraction: 0.049** (v0 baseline was 0.70 — **14× reduction**, the
  headline metric that motivated this whole regenerate)
- **99.96% convergence**, 69 fc_errors (mostly `JuliaError` on edge-case bulks)
- 22.8% of rows have `liq > 5%`; 10.2% have `liq > 15%` — concentrated in
  foidite-family and gabbro classes where fractionation-trapped melt persists
  below MAGEMin's calibrated T range

**Open upstream work to file:** This juliacall + MAGEMin_C memory leak is a
real upstream bug. Gleeson (UC Berkeley) is local; worth opening a ptt issue
with the minimal reproducer (8 paths × 100 bulks shows the growth pattern in
~30 min). Their package is validated at petrology-session scale and the
millions-of-calls workload exposes a regime they haven't tested.

**Files committed in this work:**
- `scripts/production/01_find_T_FC_end_per_bulk.py` — pre-pass with mp pool
- `scripts/production/02_generate_corpus.py` — orchestrator with mp pool +
  per-path Julia GC + maxtasksperchild=1
- `scripts/production/merge_chunks.py` — generic chunk-parquet merger
- `.vscode/settings.json` — Pylance + file watcher exclusions for `data/`
- `.gitignore` updates for new corpus + cache directories

---

## 2026-05-19 — v1 model architecture: inverse-inference reframing + Stage 1 locks

**Q:** Given the v1 corpus exists, what's the model architecture that justifies
its existence (and isn't just another MAGEMin amortization)?

**Context:** Spent time stress-testing the project's value proposition.
Started from the centroid-prediction framing (deterministic head, bulk only)
which I'd recommended for v1 simplicity. User pushed back: that's exactly
Position B (MAGEMin emulator marginalized over T, P) which we explicitly
rejected at project inception (see 2026-05-11 "Position C crystallized").

After working through several possible escapes, **arrived at the inverse-
inference framing**: the model's headline contribution is that it predicts
**both formation conditions (T, P, H2O) AND mineralogy from bulk chemistry
alone**. Geologist's workflow today is "run MAGEMin over a (T, P, H2O) sweep
and stare at an unweighted range of possible assemblages." Our model collapses
that range into the conditioned posterior `P(T, P, H2O, modal | bulk)`. No
existing tool does this:
- MAGEMin: forward map only, needs (T, P, H2O) as input
- mineralML: per-grain classification, not per-rock
- CIPW: single deterministic answer, no condition awareness

**The decomposition that justifies each architecture piece:**
- If corpus = MAGEMin and head is single (T, P, H2O) prediction: model is
  ML wrapper for "predict conditions then call MAGEMin." Real value (saves
  the unprincipled sweep) but smaller.
- If corpus = MAGEMin and head is single modal prediction: Position B, rejected.
- If corpus ≠ MAGEMin (mineralML fusion, DFT-corrected thermodynamics, etc.)
  AND head is joint conditions + modal: full Position C with inverse-inference
  layer. **This is the v1 model + v2 corpus extension.**

For v1: ship the architecture with corpus = MAGEMin (no fusion yet); the
inverse-inference contribution stands on its own. mineralML fusion (CompHead +
empirical compositions) is a v1.1 / v2 escalation that makes the ModeHead's
predictions irreducible to MAGEMin.

**Architecture locks (Stage 1):**

| Component | Decision |
|---|---|
| Encoder | Shared SubCrab → pooled latent (no T/P/H2O input — bulk only at inference) |
| ModeHead | Dirichlet over mineral wt%, label-smoothing ε=1e-3 |
| CondHead | Mixture of Gaussians per (T, P, H2O) axis, linear-space, heteroscedastic, K_gmm component count in HPO search space |
| Ensemble | K=5 deep ensemble for σ_epistemic |
| Loss | Joint Dirichlet NLL + GMM NLL with weight λ_cond |
| HPO | Joint single-stage Optuna TPE search over all dims (d_model, N_layers, heads, dropout, lr, weight_decay, batch_size, λ_cond, K_gmm, ε) |
| HPO objective | `0.5 × NLL_dir + 0.5 × NLL_gmm_zscored` (proper scoring rule, automatic calibration) |
| HPO storage | `optuna.storages.JournalStorage` at `data/optuna/v1_hpo.journal` |
| Evaluation report | Per-phase R², calibration coverage, mean liq, per-class breakdowns (separate from HPO objective) |
| Device | MPS via existing `pick_device()` helper |
| Corpus load filter | `fc_status == "ok"` AND `liq ≤ 0.10` (preserves data on disk; applied in dataset.py) |

**Inference contract:**
- Input: 11 oxide wt% from the user's ICP/XRF measurement (any total; wrapper
  renormalizes to 100%)
- Output:
  - Distribution over mineral modal vector (Dirichlet mean + ensemble
    epistemic σ + intrinsic aleatoric σ)
  - GMM over (T, P, H2O): mean + modes + per-component weights
- Geologist sees both inferred conditions AND assemblage, with multimodal
  uncertainty over both, in microseconds

**Aleatoric vs epistemic uncertainty (definitions for future-me):**
- Aleatoric = inherent randomness in the data (same bulk → different paths →
  different assemblages). Captured by the within-model distribution width
  (Dirichlet variance, GMM component spread). Irreducible.
- Epistemic = model-state uncertainty (different ensemble members predict
  different things). Captured by ensemble disagreement. Reducible with more
  data or better training.

**Open implementation questions to address in Stage 1:**
- File layout: `encoder.py`, `heads.py`, `model.py` refactor, `losses.py`,
  `dataset.py` update for liq filter + condition labels, `objective.py`, `04_hpo.py`
- Stratified k-fold on `qap_name` (matches v0 split convention)
- Port the upstream Optuna scaffolding from
  `~/git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn/src/objective.py`
- Train-from-scratch script first (sanity check), then HPO sweep

**Inverse-inference experiment to design for Stage 3 (validation):**
The two-step baseline (CondHead → MAGEMin → modal) is the upper bound for
modal accuracy (since MAGEMin generated the corpus labels). Quantify how
close the joint model's ModeHead gets to two-step's MAGEMin call. Small gap =
good ModeHead approximation. Large gap = ModeHead is the bottleneck. Either
result is informative.

**Documents created/updated in this work:**
- `notebooks/train_fc_v1_eda.ipynb` — corpus EDA with liq-threshold filtering
- ADR 006 (this commit) — v1 model architecture decisions
- ADR 005 status updated to Resolved → Executed
- CLAUDE.md status snapshot updated
