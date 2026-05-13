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
