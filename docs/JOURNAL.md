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

## 2026-05-22 — Stage 1-3 executed: v1 model built, trained, two leaks fixed

**Q:** Build, train, and evaluate the ADR 006 architecture end to end.

Covers ~3 days of work. Two data-handling bugs were caught and fixed along
the way — both were my misreads of the corpus structure, not findings. Notes
kept terse but complete enough to retrace.

**Stage 1 — architecture built.** Files: `src/cnmineral/models/{encoder,
heads,model,losses,dataset,training,objective}.py`. `BaseModelV1` = shared
SubCrab encoder → Dirichlet ModeHead (18-phase modal wt%) + GMM CondHead
(T/P/H2O, K_gmm Gaussians/axis). Joint NLL loss. Driver scripts
`03_train_base.py` (v0/v1 dispatch), `04_hpo.py`, `05_train_ensemble.py`,
`06_eval_ensemble.py`. Configs `base_v1.yaml`, `hpo_v1.yaml`.

**Bug 1 — row-level split leak.** First HPO + K=5 ensemble used
`StratifiedShuffleSplit`/`StratifiedKFold` on *row* indices. Each bulk has 8
FC paths sharing ~identical encoder input (10 of 11 oxides identical across
paths), so row-level splits scattered a bulk's paths across train and val →
memorization. Reported metrics (val_loss -55, T MAE 7.6) were inflated.
Fix: bulk-level split on `sample_id`. `make_loaders_v1_bulk_split` does a
two-stage stratified `train_test_split` (64/16/20 train/val/test);
`make_cv_folds_v1` uses `StratifiedGroupKFold` for HPO CV. Classes with
< 25 bulks dropped from the corpus (31 classes, 199 bulks, 0.9%) so per-class
test metrics stay interpretable. Split persisted to `data/splits_v1.json`.
Row-leaked artifacts archived `*_rowleak` / `*_rowsplit`.

**Bug 2 — H2O target leak.** Input oxide `h2o_tot` == CondHead H2O target
`h2o_path_wt` (corr 1.0, the corpus set bulk water to each FC path's sampled
H2O). The CondHead H2O axis was copying an input, not inferring it (fake MAE
0.034). Fix: dropped `h2o_tot` from `V1_OXIDE_COLS` → model takes 10 oxides /
9 cations. New `V1_CATIONS_ORDER` keeps v0's `TrainMageminDataset` untouched.
Cost: `h2o_tot` was also a genuine input for hydrous-phase modal prediction;
removing it regressed amp/bi/mu/fl/ne modal R² (~0.13-0.25 each). Anhydrous
majors unaffected. v1.1 fix is Option C — use complete.csv's *measured*
`h2o_tot` (≠ `h2o_path_wt`) as input.

**HPO.** Bulk-level 3-fold CV, joint-NLL objective, JournalStorage. Search
space widened (d_model/N/heads kept pegging the ceiling). 10-oxide run
stopped at 16 complete + 25 pruned (50-trial target abandoned — pace ~34
min/complete on the big d=512 config, top-3 tied; diminishing returns).
Best trial #32: `d_model=512, N=4, heads=16, K_gmm=3, lr=1.25e-4,
weight_decay=1.0e-5, batch=512, lambda_cond=3.44, dropout=0.13, eps=1.0e-4`.

**v1 model = K=5 deep ensemble at trial #32, honest 10-oxide.** Bulk-level
test set (4,239 bulks, 30,212 rows):
- T MAE 15.4 C, P MAE 1.76 kbar, H2O MAE 0.535 wt% (genuine inference now).
- joint NLL -43.4; 1σ coverage T/P/H2O = 0.88/0.64/0.76 (nominal 0.68).
- per-phase R² > 0.5 for 7/18; anhydrous majors solid: fsp 0.93, q 0.97,
  cpx 0.91, ol 0.97, opx 0.75. Hydrous + rare phases weak.

**Artifacts:** `checkpoints/base_v1_ensemble/` (honest v1, seed_1000-1004 +
eval_v1.json + predictions parquets). Archived: `base_v1_ensemble_h2oleak`
(11-oxide), `base_v1_ensemble_rowsplit`, `base_v1_rowleak`. HPO journals
`data/optuna/v1_hpo_{10ox,wide,smallsearch,rowleak}.journal`.
`notebooks/v1_eval.ipynb` — eval dashboard + simulated-user inference demo.

**Next:**
- Tag `v1-trained` at this commit.
- v1.1: Option C (measured-`h2o_tot` input) to recover hydrous-phase modal;
  CompHead for per-mineral solid-solution chemistry (the phases predicted are
  mostly solid solutions — modal abundance alone is half the answer).
- Calibration is loose (T/H2O underconfident); post-hoc temperature scaling
  is cheap if needed.

---

## 2026-05-22 — A2 calibration + B2 amp diagnostic; B1/B8 corpus gating

**Q:** Pick up after `v1-trained`. HANDOFF Tier-2 (A2 calibration) and Tier-1
analysis (B2 amp). And: do the cached FC chunks already carry per-mineral
`Comp_wt` and `G_sys`, so B1+B8 can be a cheap re-parse?

**B1/B8 gating finding.** No. All 731 chunks in `data/fc_corpus_cache/` (and
the 40 in `_legacy/`) carry 37–38 cols: meta + 11 oxides + per-phase modal
wt% only. `scripts/production/02_generate_corpus.py:final_modal_from_res`
extracts `res["mass_g"]` and discards the rest of the PetThermoTools FC return
dict; per-phase compositions and system Gibbs energy were computed in-memory
and never persisted. B1 + B8 therefore require a **full FC re-run** (the
31-hour multi-crash path), not a re-parse. Do them in one bundle; needs
explicit time/infra buy-in before starting.

---

**A2 — post-hoc temperature scaling on the GMM CondHead.** New
`scripts/07_calibrate.py`. Method = Guo et al. 2017 ("On Calibration of Modern
Neural Networks") adapted to a probabilistic regression head: one scalar per
axis fit on val by minimizing NLL, applied as `log_sigma → log_sigma + t_axis`
(Guo et al.'s sigma-scaling regression analog, cf. Laves 2020, Levi 2022).
Coverage = diagnostic, not the fit target.

NLL-fit temperatures (adopted): **T=0.655, P=0.943, H2O=0.932** — all shrink
sigma. Test before → after:

|axis|NLL|cov1 (gauss)|cov1 (PIT)|KS vs uniform|
|---|---|---|---|---|
|T  |-2.570 → -2.652|0.878 → 0.861|0.808 → 0.781|0.169 → 0.120|
|P  |-0.009 → -0.011|0.635 → 0.618|0.620 → 0.601|0.051 → 0.060|
|H2O|-0.534 → -0.536|0.758 → 0.732|0.706 → 0.686|0.114 → 0.117|

**Reframing finding.** The headline Gaussian-σ coverage (0.88/0.64/0.76) is
the wrong metric for a multimodal mixture predictive. The proper diagnostic
is the **PIT** (probability integral transform — predictive CDF evaluated at
the target; uniform if calibrated). PIT central-68 coverage is 0.81/0.62/0.71
on test — H2O is essentially calibrated already, P is a hair under, only T is
materially over-covered (PIT histogram is a clean dome). KS vs uniform
agrees: P 0.05 (near-uniform), H2O 0.11, T 0.17.

Temperature scaling **helps T meaningfully** (KS 0.17→0.12, NLL clearly down,
central-68 coverage 0.81→0.78); for P and H2O it is near a no-op because
their PIT non-uniformity is not a scale problem. Concretely the v1 corpus
samples h2o_path_wt at 2 discrete values per bulk and P_kbar at 4 — so PIT
lumpiness on those axes is largely an artifact of evaluating a continuous CDF
on discrete targets, not miscalibration. T_FC_end_used is per-bulk constant
and continuous globally, so T is the one axis where calibration is a
well-posed problem and Guo-style temperature scaling does its job. (Deeper
consequence, flagged not acted on: the CondHead's P and H2O outputs largely
recover the per-class sampling grid, not a bulk-conditional inverse
inference — the per-axis "inverse inference" claim is substantively only
about T.)

Artifacts: `checkpoints/base_v1_ensemble/calibration_v1.json`,
`data/diagnostics/calibration/pit_hist_v1.png` (PIT before/after per axis).
A single residual t-fit shape — a small dome remains for T after scaling —
shows a one-scalar fit can't fully flatten the T PIT; the NLL-fit
deliberately stops short of full coverage-matching (which would want t=0.27,
hurting NLL). Inference/eval wiring deferred (one flag reading the JSON).

---

**B2 — amp diagnostic.** New `scripts/b2_amp_diagnostic.py`. Pure analysis on
test split: variance decomposition + h2oleak comparison + per-class
breakdown. Decomposes amp per-path R² as `ceiling × R²_bulk` where
ceiling = SS_between/SS_tot is the highest per-path R² any bulk-only model
can reach (because the 8 FC paths share an identical input — verified honest
`pred_amp` per-bulk std = 0.00 exactly).

**amp per-path R² = 0.22 = ceiling 0.67 × R²_bulk 0.32.** Both terms matter.

|phase|presence|within%|ceiling|R²_bulk|R²_path|
|---|---|---|---|---|---|
|fl |0.16|0.40|0.60|0.17|0.10|
|**amp**|**0.33**|**0.33**|**0.67**|**0.32**|**0.22**|
|liq|0.34|0.31|0.69|0.43|0.30|
|opx|0.49|0.07|0.93|0.81|0.75|
|cpx|0.63|0.02|0.98|0.93|0.91|
|ol |0.30|0.01|0.99|0.98|0.97|
|q  |0.76|0.00|1.00|0.97|0.97|
|fsp|0.89|0.02|0.98|0.95|0.93|

The three high-within-bulk phases are exactly the (P, H2O)-sensitive ones:
amphibole, silicate melt, free fluid. All anhydrous majors have <5%
within-bulk spread. **amp's low ceiling is amp-specific, not a global corpus
property.**

**h2o_tot leak contribution.** Honest amp per-path R² 0.216, leaked
(11-oxide `base_v1_ensemble_h2oleak`) 0.342 (+0.13). The leaked `h2o_tot`
varied per path so the leaked model could chase within-bulk spread; that
0.13 is irreducible by honest means (ADR 007).

**Per-class amp R² (honest).** Negative in the amp-rich classes: gabbro
-1.70, quartz diorite -1.03, foid-bearing diorite -0.88,
foid-bearing monzodiorite -0.92, diorite -0.64, andesite -0.30. Positive
only for amp-poor classes (rhyolite +0.36, monzogranite +0.17,
phonolite +0.17). The headline R²_bulk = 0.32 is buoyed by easy
near-zero predictions in felsic classes; **in the classes where amp matters,
the model is genuinely bad — not just metric-artifact bad.**

**Implications for B1 / CompHead.** CompHead predicts per-mineral composition,
not abundance. It cannot move amp's modal R² at either the per-path or the
bulk level. amp abundance is hard for two independent reasons: (1) ceiling 0.67
(33% within-bulk path spread, structurally invisible to a bulk-only model),
(2) genuine bulk-level difficulty in plutonic mafic-intermediate classes.
CompHead's value for amp is in predicting amphibole *chemistry* (Mg#, Al, Na),
which is a separate and worthwhile target — but **B1 should not be pitched as
"fixes amp"**.

Plots: `data/diagnostics/b2_amp/{phase_variance_ceilings, amp_parity,
amp_per_class_r2}.png`.

---

**Next.**
- Update MODEL.md "Calibration is loose" bullet with the PIT reframing.
- B1+B8 require full FC re-run; cost ≈ 31 h + the leak workaround; needs
  user buy-in before starting.
- B3 (modal-haze / Dirichlet sparsity) is the remaining Tier-1 item not
  blocked on a corpus regen; it's a ModeHead change + retrain.
- Inference wiring for the calibration JSON: one flag in 06_eval / the
  notebook reads it and adds `t_axis` to `cond_log_sigmas`.

---

## 2026-05-27 — v2 ablation: SSL pretrain wins, trace inputs don't add (Strategy B partial)

**Q:** Does trace-augmented SSL pretraining (Strategy B from the v2 plan)
improve hydrous-phase modal `R²_bulk`?

**Build.** All on the `v2-trace` branch. Plumbing in three new files:
`src/cnmineral/data/unlabeled.py` (`V2_CATIONS_ORDER`: 9 majors + 13 traces
= 22-token vocab; `UnlabeledV2Dataset` over 474k igneous bulks from
complete.csv with the v1 supervised bulks excluded; `oxides_traces_to_tokens`
helper; supergroup mapping for class-balanced sampling),
`src/cnmineral/models/pretrain_head.py` (per-token reconstruction MLP), and
`src/cnmineral/models/pretrain_losses.py` (masked-element MSE with the
`(src=valid, frac=0)=MASK` convention -- no kingcrab modification needed).
New scripts: `08_pretrain.py` (SSL driver with supergroup-balanced sampling
per the mineralML §2.1 oversampling lesson), `09_visualize_latent.py`
(UMAP/PCA + k-NN sanity + OOD), `10_ablation_v2_eval.py` (four-cell
R²_bulk decomposition). Added `pretrain_init: str | None` kwarg to
`BaseModelV1.__init__` (5-line addition), `majors_only: bool` flag to
`TrainFCv2Dataset`, parameter-group AdamW (`encoder_lr = 0.1 * head_lr`) in
`03_train_base.py`'s new `run_v2` path. `TrainFCv2Dataset` extends the v1
parquet with a complete.csv join on `sample_id` for traces; trace coverage
on the supervised subset is 82% with at least one trace populated.

**SSL pretrain.** `scripts/08_pretrain.py` on 474k unlabeled bulks (5% val
split), 8 epochs, mask probability 0.25, AdamW + cosine LR, supergroup-
balanced sampling. Wall: 16.6 min on MPS. Val masked-element MSE 8.5e-3 →
4.47e-3 over 8 epochs (best at epoch 1; subsequent epochs marginal).
Encoder saved to `checkpoints/pretrain_v2/encoder_best.pt`.

**Four-cell ablation** (single seed, full corpus, frozen v1 bulk-level split):

| | (a) v1 baseline | (b) traces, random init | (c) majors-only, pretrained | (d) full v2 (traces + pretrained) |
|---|---|---|---|---|
| best epoch | -- | 35 | 51 | -- |
| T MAE | 15.4 | 16.3 | 17.0 | 17.5 |
| P MAE | 1.76 | 1.78 | 1.76 | 1.76 |
| H2O MAE | 0.54 | 0.57 | 0.54 | 0.56 |

**R²_bulk per phase (the fair B2-decomposition metric):**

| phase | ceiling | (a) v1 | (b) traces | (c) pretrain | (d) full v2 |
|---|---|---|---|---|---|
| amp | 0.67 | +0.32 | +0.28 | **+0.32** | +0.21 |
| fl  | 0.60 | +0.16 | +0.09 | +0.15 | +0.15 |
| liq | 0.69 | +0.43 | +0.41 | **+0.48** | +0.48 |
| bi  | 0.98 | +0.53 | +0.46 | **+0.54** | +0.52 |
| mu  | 0.82 | -0.62 | -0.64 | -0.60 | -0.60 |
| ne  | 0.97 | +0.48 | +0.48 | **+0.65** | +0.64 |
| cpx | 0.98 | +0.93 | +0.92 | +0.90 | +0.90 |
| ol  | 0.99 | +0.98 | +0.97 | +0.98 | +0.97 |
| fsp | 0.98 | +0.95 | +0.95 | +0.94 | +0.94 |
| q   | 1.00 | +0.97 | +0.97 | +0.97 | +0.97 |

**Decision-rule reading.** Per the plan: "(d) > (b) AND (d) > (c)" was the
condition for Strategy B success. **Not met.** The pattern is **(c) ≈ (d) >
(a) ≈ (b)** for hydrous-and-foid phases (liq, bi, ne) and **(a) > (b) ≥ (c) >
(d)** for anhydrous majors. Plain English:
- **SSL pretrain helps**, modestly to materially (ne +0.17 vs v1, liq +0.05,
  bi +0.01).
- **Trace inputs do NOT add value, and slightly hurt** (amp R²_bulk drops
  +0.32 → +0.21 when traces are added to the pretrained encoder; anhydrous
  majors give back 1-3 percentage points across cpx/ol/fsp/q).
- **mu is hopeless in every cell** -- joins sph/ep/ru on the "too-rare-
  to-fit" list.

**Why traces don't add value here -- the honest mechanism.** Labels in the
v1 FC corpus are deterministic functions of *majors only* (PetThermoTools FC
takes the 11-oxide majors and emits modal+conditions; trace elements are
irrelevant to the simulation). Adding traces as model inputs cannot improve
the model's ability to predict labels that don't depend on traces. The
hypothesis "traces discriminate amphibole-bearing diorites from look-alikes"
might be true in *nature* but is false in *our labels*. To exploit traces
honestly, the corpus would have to be regenerated against labels that DO
depend on traces -- e.g., the deferred-future mineralML grain-composition
fusion (Position C) which trains against real-rock per-mineral compositions.
This is consistent with B2: the bulk-level R² ceiling for amp isn't
information-poverty in the encoder, it's that the MAGEMin labels carry no
amp-specific signal beyond what 10 majors already encode.

**Why SSL pretrain helps despite no new input.** The encoder uses mat2vec
element embeddings (already informative) + a self-attention transformer over
22 tokens. SSL on 474k unlabeled bulks teaches the *transformer* the natural
geochemical manifold; the supervised loss then doesn't have to learn that
structure from 21k labeled bulks. Confirmed quantitatively by the latent-
space k-NN sanity check: 10-nearest-neighbor mean modal in the 512-D
pretrained latent gives **cosine similarity 0.957 vs true modal** (baseline:
random-neighbor 0.813, delta +0.145), well above the plan's ≥0.1 acceptance
bar. The encoder learned task-relevant structure WITHOUT EVER SEEING A
MINERAL LABEL.

**Latent-space diagnostics** (`scripts/09_visualize_latent.py`,
`data/diagnostics/latent_v2/`):
- UMAP/PCA of pretrained vs random-init encoder on 5k held-out bulks colored
  by petrogenetic supergroup looks visually similar -- the mat2vec input
  embeddings dominate the top-2 axes. The pretrain gain is real but
  distributed across all 512 dims, not concentrated in 2D.
- Three-panel UMAP (random / pretrained / pretrained-then-finetuned cell d)
  shows the **fine-tune does most of the visible structuring**, with clear
  foid-bearing / mafic / felsic separation only after supervised training.
- Ratio-colored UMAP (Sr/Y, La/Yb, Nb/Y, Th/Yb) on pretrained shows clear
  gradients along the manifold -- petrogenetic discriminants align with
  encoder structure without supervision.
- k-NN sanity: pretrained latent's neighbors give modal predictions cosine
  0.957 to truth vs random 0.813 (delta +0.145, above ≥0.1 bar).
- OOD distance: 1.5% of test bulks beyond the 99% train threshold. Test +
  held-out-train distributions overlap closely; calibration of the OOD
  signal is healthy.

**Pivot recommendation.** Adopt **cell (c) -- pretrained encoder + majors-
only inputs -- as the v2 production path.** Drop the trace inputs from the
fine-tune model architecture (they hurt amp and don't help elsewhere);
preserve the trace columns in pretrain so the encoder still learns
major↔trace correlations (which is the source of the ne +0.17 win). Net
v2.0 model = same architecture as v1, same input as v1 (majors only), new
encoder init from majors+traces SSL pretrain.

**Cost-benefit.** SSL pretrain = 16 min on MPS. Supervised retrain = 30-50
min per seed × K=5 ensemble = 3-4 hours. Worth doing for the +0.17 R²_bulk
on ne and +0.05 on liq alone; the ce ll-(c) win is a clear improvement on the
v1 baseline that costs less than two hours of compute.

**Artifacts.**
- `checkpoints/pretrain_v2/encoder_best.pt` -- SSL-pretrained encoder
- `checkpoints/base_v2_{b,c,d}_*/best.pt` -- per-cell single-seed
  checkpoints + predictions_test.parquet
- `data/diagnostics/latent_v2/{umap_*.png, pca_*.png, ood_distance.png,
  knn_mineralogy.json, ood_flags_summary.json}` -- latent diagnostics
- `data/diagnostics/ablation_v2/comparison.json` -- four-cell numeric
  comparison
- Plan: `~/.claude/plans/sunny-weaving-nebula.md`

**Next.**
- Update MODEL.md "as-built performance" with the cell-(c) numbers as v2.0.
- Update HANDOFF.md / CLAUDE.md status to reflect v2 pivot.
- Optional: K=5 ensemble for cell (c) to produce a v2-trained tag analogous
  to v1-trained.
- Deferred (corpus regen required): mineralML composition fusion (would
  give traces a label-side rationale); CompHead (B1); GsysHead (B8); fl/liq
  drop. These remain the Tier-1 future work; v2 doesn't unblock them.

---

## 2026-05-29 — Phase A: SSL pretrain bug fix + corrected v2 results + SSL-as-foundation-model reframe

**RETRACTION of 2026-05-27 entry.** The "Strategy B partial" result was based
on a broken SSL pretrain: the v2 mask-token convention `(src=valid, frac=0)`
collided with kingcrab's frac-based padding logic. Masked positions hit
every zeroing path in `kingcrab.py:Encoder.forward` (`src_key_padding_mask`,
the fractional multiplication by `0`, the explicit `masked_fill`). Per-token
outputs at masked positions were zero, no gradient back to the encoder.
Verified bytewise: `encoder_best.pt` was identical to a fresh seed-42
random init. The 8-epoch "training loss decrease" was the pretrain head
learning a per-position prior from a constant zero input. The four-cell
ablation thus compared random-init encoders with different encoder
learning-rate scales (`encoder_lr = 0.1*head_lr` for cells with non-None
`pretrain_init`, full `head_lr` for cell b), not pretrained-vs-not.

**Bug fix (Option H from the plan).** Stop the per-token mask convention.
Zero BOTH `src` and `frac` at masked positions so kingcrab pads them out
cleanly. Change `PretrainHead` from a per-token MLP (`d_model -> 1` applied
to `[B, T, d_model]`) to a vocab-decoder MLP (`d_model -> d_model//2 ->
n_tokens=22` applied to `[B, d_model]` pooled latent). Head reconstructs
the full chemistry from the pooled latent of the surviving tokens; loss
applies only at the originally-masked positions. ~30 lines of changes
across `pretrain_head.py`, `pretrain_losses.py`, and `08_pretrain.py`. No
kingcrab modification. Gradient sanity check confirms: 53/53 encoder
params now have nonzero `grad.norm()` after backward (was 0/53).

**Fixed pretrain.** 8 epochs, 474k unlabeled igneous bulks, supergroup-
balanced sampling. Train loss 1.60e-3 -> 4.88e-4 (~3x reduction; barely
moved before). Val loss 2.64e-4 -> 2.39e-4 (best at epoch 6). Encoder
weights now differ meaningfully from random init (diff norms 3.6-5.3
across sampled keys, vs 0.0 before).

**Corrected four-cell ablation** (per-phase R^2_bulk on test split):

| phase | ceiling | (a) v1 K=5 | (b) traces, random | (c) majors-only + SSL | (d) full v2 |
|---|---|---|---|---|---|
| **bi** | 0.98 | +0.53 | +0.55 | **+0.71** | +0.68 |
| **ne** | 0.97 | +0.48 | +0.59 | **+0.70** | +0.66 |
| **mu** | 0.82 | -0.62 | -0.64 | -0.39 | -0.48 |
| amp | 0.67 | +0.32 | +0.31 | +0.40 | +0.36 |
| liq | 0.69 | +0.43 | +0.40 | +0.47 | +0.50 |
| fl | 0.61 | +0.16 | +0.16 | +0.21 | +0.19 |
| cpx | 0.98 | +0.93 | +0.92 | +0.93 | +0.92 |
| ol | 0.99 | +0.98 | +0.97 | +0.98 | +0.98 |
| fsp | 0.98 | +0.95 | +0.95 | +0.97 | +0.97 |
| q | 1.00 | +0.97 | +0.98 | +0.98 | +0.98 |

Condition MAEs: cell (c) **beats v1 K=5 ensemble on every axis** as a single
seed: T 14.8 vs 15.4, P 1.73 vs 1.76, H2O 0.51 vs 0.54. Adopted v2 path =
cell (c) = SSL-pretrained encoder + majors-only inputs.

The earlier story ("SSL helps modestly; traces don't add") survives at
higher magnitude with the fix: SSL pretrain delivers **large gains on
hydrous and foid-bearing phases** (bi +0.18, ne +0.22, mu +0.23 vs v1)
where petrogenetic structure matters; tied-or-better on the anhydrous
majors that were already near-saturated. Cell (d) (with traces) is
slightly worse than (c) on most phases because FC labels are deterministic
functions of majors only -- trace inputs add no discriminative signal for
the labels and inject noise.

**Latent-space diagnostics with the real pretrained encoder.** k-NN
cosine of test modal vs k=10 nearest training neighbors' mean modal in
the 512-D pretrained latent: **0.975 vs 0.813 random-neighbor baseline,
delta +0.162** (vs the broken-pretrain +0.145 which measured mat2vec
alone). OOD: 1.4% of test bulks beyond 99% train threshold; healthy.
Three-stage UMAP shows pretrained adds visible cluster sharpening over
random init (less dramatic than supervised fine-tune does, but real).

**The headline non-obvious finding: SSL-pretrained encoder is a strong
rock-type classifier.** Linear probe under-reports the SSL gain; nonlinear
MLP probe (1 hidden layer, 128 ReLU) reveals it. On the supervised
train/test bulks:

| encoder | supergroup LR | supergroup MLP | qap_name LR | qap_name MLP |
|---|---|---|---|---|
| random-init | 0.601 | 0.653 (+5pp) | 0.516 | 0.599 (+8pp) |
| **SSL-pretrained** | **0.615** | **0.763** (+15pp) | **0.537** | **0.720** (+18pp) |
| cell-c fine-tuned | 0.701 | 0.787 (+9pp) | 0.659 | 0.750 (+9pp) |

The MLP-LR delta is *3x larger for the SSL encoder than for random init*:
SSL specifically introduced nonlinear class structure that linear
classifiers couldn't access. The SSL-pretrained encoder + MLP probe
reaches **76.3% supergroup accuracy and 72.0% qap_name accuracy** -- only
2-3pp behind the fully supervised cell-c. **SSL captures ~95% of the
classification capacity that joint supervised mineralogy training
contributes.** Hard-staged designs are more viable than the earlier
linear-probe number implied (24% supergroup miss rate, not 40%).

**Bigger reframe -- the SSL-pretrained encoder is a rock-chemistry
foundation model.** It supports a family of useful predictions beyond
mineralogy, with no further training or with tiny supervised heads:
element imputation (for partial analyses), per-element anomaly flagging,
k-NN similarity lookup against the 1M-row complete.csv, OOD detection,
tectonic-setting classification (Pearce-diagram derived labels), magma
series classification, petrogenetic-ratio regression (Sr/Y, La/Yb, etc.),
isotopic-ratio prediction where complete.csv has labels, approximate age
regression, geographic origin inference. Full list and effort estimates
in new `docs/PRETRAIN_APPLICATIONS.md`. This positions the project as
having two downstream tracks rather than one: a *mineralogy track* (the
original v1/v2 work) and a *non-mineralogy applications track* (the
SSL-derived capabilities).

**Architecture / API changes shipped this session:**
- `src/cnmineral/models/pretrain_head.py` -- vocab-decoder MLP shape
- `src/cnmineral/models/pretrain_losses.py` -- mask convention docs +
  optional per-token std normalization (config-gated, off by default)
- `scripts/08_pretrain.py` -- mask both src and frac; default-pool encoder;
  optional per-token-norm pipeline
- `src/cnmineral/models/encoder.py` -- `return_tokens` kwarg kept for
  backward compat but no longer used by pretrain
- `scripts/11_linear_probe.py` -- now reports both LR and MLP probe per
  encoder; gracefully skips missing checkpoints

**Follow-ups coded but not re-run yet:**
- Per-token loss normalization (`pretrain.per_token_norm: true` in
  `pretrain_v2.yaml`) -- standardizes per-position MSE so trace
  reconstruction isn't drowned out by major-element magnitudes. Would
  re-pretrain to test; deferred to next iteration.

**Artifacts:**
- `checkpoints/pretrain_v2/encoder_best.pt` -- real SSL-pretrained encoder
  (replaces the broken one, which is archived as
  `checkpoints/pretrain_v2.broken-bug-frac-padding/`)
- `checkpoints/base_v2_{b,c,d}_*/best.pt` + `predictions_test.parquet`
  per cell (broken cells archived alongside)
- `data/diagnostics/latent_v2/{umap_*,pca_*,ood_*.png, knn_mineralogy.json,
   linear_probe.json}` -- diagnostics
- `data/diagnostics/ablation_v2/comparison.json` -- four-cell numbers

**Next.**
- New `docs/PRETRAIN_APPLICATIONS.md` capturing the foundation-model
  capability list (this entry references it; write next).
- Reframe `docs/MODEL.md` v2 section against the corrected results.
- Update `CLAUDE.md` "Project in one paragraph" + "Status snapshot" to
  position the project as two tracks (mineralogy + non-mineralogy SSL
  applications).
- Commit on `v2-trace` branch in logical chunks (pretrain code + cells +
  probe + docs as separate commits).
- After commit: kick off two parallel agents -- mineralogy track
  (joint multi-task aux supergroup head + soft-staged comparison) and
  non-mineralogy applications track (build out the items in
  `PRETRAIN_APPLICATIONS.md`, prioritizing k-NN similarity / element
  imputation / OOD as zero-additional-training deliverables).
- Per-token loss normalization follow-up (`per_token_norm: true`)
  remains queued; worth a one-off re-pretrain comparison either inside
  the mineralogy agent's session or as a standalone experiment.

## 2026-06-01 -- Design crystallization for fresh-codebase v3 rewrite

**Q.** Before committing to another corpus regen + retrain, can we
crystallize the value proposition, architecture, training data, and
implementation plan into a single doc, so a from-scratch rewrite can
proceed cleanly without re-deriving design choices?

**Context.** Several long sessions of debugging (kingcrab padding-mask
bug, mass%_Liq scale collision, per-phase composition extraction)
revealed both the right invariants and the wrong ones. The v3 design
is now stable enough to write down. The user wants primary ownership
of the rewrite with agentic help only in specific spots; that requires
a tight handoff doc.

**Conversational arc this session:**

1. Built `scripts/experiments/visualize_degeneracy.py` and
   `plot_degeneracy.py` to visualize per-bulk (P, H2O) degeneracy on
   one real Arc basalt (sample_id=356546). Ran 100 paths, 10x10 grid
   over (P 1-5 kbar, H2O 2-6 wt%), per-path T_FC_end where liq<5%.
   Three bug rounds before extraction worked: (a) mass%_Liq is a
   fraction not a percent, (b) FC-snapshot vs FC-cumulate semantics
   differ (production uses cumulate via mass_g cumsum), (c) res["All"]
   uses PTT short codes (Cpx/Plag/Sp) that aren't in _phase_map.py.
   Documented why production never hit these: it tackled a narrower
   problem (modal only, pre-computed T) with a cleaner data path.

2. Results: modal CV across paths is 0.03-1.6 (tight for anhydrous
   majors, blowing up for amp/ol/fl/opx); per-phase composition CV is
   10-30x tighter than modal CV. Empirical evidence that CompHead's
   per-path R² ceiling is meaningfully higher than ModeHead's --
   composition is mostly bulk-controlled, modal is mostly
   path-controlled.

3. P / H2O sampling argument: T's "rock exists at surface = rock has
   cooled" is uniquely clean. P has a partial analog
   (volcanic-vs-plutonic). H2O has no clean per-rock analog (degassing
   confound). Recommendation: condition the P sampling grid on
   qap_name volcanic-vs-plutonic; keep H2O on tectonic-class
   population prior.

4. Value-proposition deep-dive (with the user's own ChatGPT framing
   as a useful structuring tool). Crystallized one-sentence pitch:
   "A probabilistic emulator of MAGEMin integrated over geologically
   plausible formation conditions, with a rock-chemistry foundation-
   model encoder learned from real bulks, parameterized so real
   labeled data can replace synthetic labels later without retraining
   the encoder." Three pieces: marginal not conditional, distribution
   not point estimate, encoder as foundation model independent of
   MAGEMin's correctness.

5. UMAP recoloring: added `scripts/09b_latent_chem_axes_and_3d.py`
   producing `umap_by_chem_axes.png` (SiO2 / Mg# / Na2O+K2O gradients
   across the SSL latent) and `umap_3d_supergroup.png` (3D UMAP).
   The chem-axis figure is the headline -- smooth continuous gradients
   on the SSL latent, showing the encoder organized along
   differentiation and alkalinity axes without supervision. mineralML's
   cleaner-looking UMAP is because their loss is supervised
   classification of discrete mineral species; ours is unsupervised
   SSL on continuous bulk chemistry. Different signals, both faithful.

6. CompHead parameterization discussion. User proposed flattening
   ModeHead+CompHead into a single endmember-wt% head. Mass-balance
   argument is correct but the flattened head has 4x output dim,
   extreme sparsity, database-coupled basis, and loses interpretability.
   Refined alternative: keep two heads, parameterize CompHead as
   per-phase endmember mole fractions (simplex per phase). Physically
   valid composition by construction. Decision deferred to new pilot
   test 2b added to the plan (and to memory
   [[comphead_parameterization_options]]).

**Deliverables this session:**

- `scripts/experiments/visualize_degeneracy.py` + `plot_degeneracy.py`
  -- 100-path degeneracy viz on one Arc basalt; outputs in
  `data/diagnostics/degeneracy_viz/`
- `scripts/09b_latent_chem_axes_and_3d.py` -- continuous-chem-axis
  recoloring + 3D UMAP; outputs in `data/diagnostics/latent_v2/`
- `docs/FRESH_START.md` -- the consolidated implementation handoff
  for the v3 rewrite. Crystallizes value prop, architecture, training
  data, kingcrab decision, G_sys placement, what to keep vs drop
- Plan update: pilot 2b added to the v3 corpus regen plan
  (`~/.claude/plans/sunny-weaving-nebula.md`), gating the regen on an
  explicit oxide-wt% vs endmember-fraction CompHead decision
- Memory: `comphead_parameterization_options` capturing the tradeoff

**Crystallized design choices for v3:**

- 22-token input vocab (9 majors + 13 traces); src=0 for absent
- Encoder: own thin transformer with mat2vec embeddings; drop kingcrab
- 4-5 heads in parallel: ModeHead (Dirichlet, 18 phases), CompHead
  (parameterization pending pilot 2b), CondHead (GMM per axis),
  ClassHead (CE supergroup, auxiliary), optional G_sys (scalar regression,
  auxiliary; only if corpus regen persists it)
- SSL pretraining: masked-element modeling, Option H pooled-latent
  decoder (zero both src and frac at masked positions)
- Loss: NLL_modal + λ_comp·comp_loss + λ_balance·mass_balance +
  λ_cond·NLL_cond + (optional) λ_class·CE_class + (optional) λ_gsys·MSE_gsys
- Padding mask derived from `src == 0` ONLY; masked-for-SSL means
  BOTH src and frac are zero; the single most important code invariant

**Next.**

- User drives the fresh-codebase implementation; agentic help in
  specific spots (mat2vec loading, ADR migration, mass-balance loss
  validation, kingcrab-free encoder verification)
- Before any corpus regen: execute pilot tests 1, 2, 2b, 3, 4 per the
  updated plan; decide oxide-wt% vs endmember-fraction CompHead
- Skipped from v3 first pass: TracePartitionHead, ALIGNN, polymorph
  head, fO2 sweep. All deferred to v4+.

## 2026-06-04 -- BulkChemistryDataset lands; HPO smoke runs end-to-end on real data

**Q.** Can the v3 pretrain pipeline run on the real Gard 2019 corpus
end-to-end, and what filter shape + wall time should the full pretrain
plan around?

**Context.** Day 4 of the v3 first-week sequence in
`docs/V3_WORKFLOW.md`. `scripts/pretrain.py` was already wired against
a stub `BulkChemistryDataset` (imported lazily, falling back to
`SyntheticBulkChemistryDataset` if absent). The real loader was the unlock
for items 1-3 of the Track A gate (UMAP + probe diagnostics + full SSL
pretrain). Built it, ran smoke + HPO smoke on the real corpus.

**`BulkChemistryDataset` in `src/cnmineral/data/bulk_chemistry.py`.**

Constructor knobs: `csv_path`, `rock_group` (default `"igneous"`),
`drop_zero_as_nan` (default `True`), `require_all_majors` (default
`True`), `include_cr2o3` (default `False`).

The strict default filter walks:

| step | rows |
|---|---|
| start (igneous) | 739,727 |
| + all 7 non-Fe majors non-null | 544,050 |
| + feo non-null | 260,718 |
| + fe2o3 non-null | 172,060 |
| + k2o, na2o, tio2 non-null | 169,024 |
| + non-negative on the 9 majors | 168,524 |
| + drop_zero_as_nan applied | 166,893 |

Sanity check: `len(BulkChemistryDataset())` is 166,893. The "474k" figure
floating in `CLAUDE.md` predates the explicit Fe-fallback convention
and corresponds to a looser filter (7 non-Fe majors + any of
feo/fe2o3/feo_tot/fe2o3_tot present), which yields ~501k rows. Under
`require_all_majors=False` the loader returns ~737k rows.

The big attrition is iron: requiring `feo` halves the pool
(544k -> 261k) and requiring `fe2o3` halves it again (261k -> 172k).
Many igneous samples report `feo_tot` or `fe2o3_tot` alone, not the
split pair. The Fe-fallback chain (`feo_tot` first, then
`fe2o3_tot`) is implemented per the handoff but only fires under
`require_all_majors=False`. Worth a follow-up decision on whether the
full SSL pretrain should relax the filter to recover the ~330k extra
rows (P2 below).

**Edge cases:**

- exact-zero in any oxide / trace -> treated as NaN (geochemists use
  0 for below-detection; the v1 convention is to drop those slots)
- negative oxide -> row dropped entirely (corrupt sample signal)
- negative trace -> only the trace slot NaN'd out (one bad trace
  shouldn't discard an otherwise-good bulk)
- `qap_name` carried as a parallel `np.ndarray` of strings (None for
  missing) so downstream split logic can stratify without rebuilding
- `sample_id` is the CSV's canonical identifier; deterministic
  ordering is preserved

**Tests landed in `tests/test_bulk_chemistry.py` (17 added):**

structural -- shape/dtype, padding convention, MASK_TOKEN_ID absent
from raw output, sample_id stable across construction, oxide-cation
conversion numerical pin (sio2=50 -> Si frac ≈ 0.2337), iron
aggregation pin, iron fallback to feo_tot pin, zero-as-NaN per-slot
pin + row-drop variant, negative-major drops row, negative-trace nans
only the slot, qap_name preserved, rock_group filter excludes
sedimentary, `include_cr2o3` toggle, trace ppm->frac pin, missing-CSV
raises. Plus one skip-if-absent regression pin against the real CSV
that asserts `160k <= len(ds) <= 175k` so a drift surfaces here
rather than after a long pretrain.

Suite: 54 passed (37 existing + 17 new) in 5.4 s.

**Real-data smoke: single-run.**

`epochs=1`, `d_model=512`, `n_layers=4`, `n_heads=16`, `batch_size=64`
on MPS:

```
split: trainval=133514 test=33379
run_pretraining: n_train=106811 n_val=26703
epoch 1/1 train=0.0008 val=0.0002 elapsed=40.8s
final: train=0.0008 val=0.0002 test=0.0002
```

40.8 s / epoch on MPS at the full production model. A 10-epoch run is
~7 min. Memory was not the bottleneck.

**Real-data smoke: HPO.**

3 trials × 1 epoch + 1 production run on best-params + 1 test eval.
Trial wall times 31-63 s (batch_size=256 -> 32 s; batch_size=32 -> 62
s). Total HPO sweep: ~4 min 11 s.

Summary JSON has all four loss fields populated:
- `best_trial_val_loss`: 0.000143 (trial 2)
- `final_train_loss`: 0.00115
- `final_val_loss`: 0.000158
- `final_test_loss`: 0.000154

(All values are absolute MSE on `frac` in [0, 1]; the model converges
quickly because the masked targets cluster tightly around per-position
medians.)

The `assert_padding_convention` check on the first batch passed for
both runs.

**Codebase change recommendations (not implemented this session per
the handoff's no-touch list).**

P1 -- **`reserve_test` should stratify by `qap_name`** when the dataset
exposes it. CLAUDE.md mandates 64/16/20 stratified by `qap_name` for
the production split, and `BulkChemistryDataset.qap_names` is now
available. Today `reserve_test` calls `train_test_split` without
`stratify=`, so rare QAP classes can end up entirely in train or
entirely in test. Fix: in `scripts/pretrain.py::reserve_test`, when
the dataset has a `qap_names` attribute, pass `stratify=` and drop
rows whose class count is below sklearn's threshold (CLAUDE.md
mentions a 25-bulk floor). The smoke runs are unaffected by this
gap because they only validate wiring, not split quality.

P1 -- **dataset construction is duplicated 5+ times per HPO run.**
`reserve_test`, each trial's `run_pretraining`, the production run,
and `evaluate_on_test` each call `create_dataset(config)`, which
reads + parses the 828 MB CSV anew (~3 s) and holds the parsed
pandas frame transiently (~1.2 GB peak per construction). Three
sequential constructions drove RSS from 265 MB to 2 GB. A 20-trial
sweep would construct the dataset ~22 times. Fix: memoize in
`scripts/pretrain.py::create_dataset` by a `(csv_path, frozenset
data_kwargs)` key so the dataset is built once per study. The peak
resident state per dataset is only ~45 MB of tensors; the bloat is
pandas not releasing back to the OS. P1 because it caps how many
trials can fit on a 16 GB machine before swapping.

P2 -- **build a parquet cache of `complete.csv` on first read.** The
CSV parse takes ~3 s but is purely deterministic given the column
subset; a parquet sidecar would drop this to <300 ms. Not a blocker
for the 167k corpus and 40 s/epoch wall times, but compounds with
the P1 memoization gap (each cache miss multiplies). Recommend
`data/complete.parquet` built once via a small `scripts/` helper,
with `BulkChemistryDataset` preferring the parquet when it exists and
falling back to CSV otherwise.

P2 -- **revisit the strict `require_all_majors=True` default.** The
strict filter discards ~330k rows that have computable Fe via
`feo_tot` / `fe2o3_tot`. For SSL pretrain (where more data is almost
always better and the supervised loss isn't dependent on label
quality), it may be worth toggling to `require_all_majors=False` to
recover those rows. The CompHead / mass-balance numerics in Track B
have stronger reasons to want strict majors, so the toggle should
land in the config not the default. Worth a 1-epoch smoke comparison
on both settings before the full run.

P3 -- **`num_workers` on MPS.** Single-process loading is fine at this
scale; MPS + multiprocessing data loading is fragile on macOS
(libomp conflicts + spawn semantics). Defer until the wall time
becomes a real constraint.

**Deliverables this session:**

- `src/cnmineral/data/bulk_chemistry.py` -- the real CSV-backed dataset
- `tests/test_bulk_chemistry.py` -- 17 tests pinning the contract
- `data/optuna/v3_pretrain_hpo_realsmoke.journal` +
  `data/optuna/v3_pretrain_hpo_realsmoke_summary.json` -- HPO smoke
  artifacts
- `checkpoints/pretrain_v3_smoke/epoch_001.pt` -- single-run smoke
  checkpoint with test fields populated

**Next.**

- Day 6: port UMAP + probe diagnostics from v2's
  `scripts/09_visualize_latent.py` +
  `scripts/09b_latent_chem_axes_and_3d.py` against
  `src/cnmineral/eval/`.
- Day 7: full-corpus SSL pretrain. Plan around ~7 min/epoch on MPS at
  the production model; a 20-epoch run is ~2.3 h, a 50-epoch run is
  ~6 h. Memoize `create_dataset` first (P1) so an HPO sweep doesn't
  thrash memory.
- Decide whether to relax `require_all_majors` for SSL pretrain
  (P2) before the production run.

## 2026-06-11 -- Track A closed: 20-epoch encoder beats v2 cell-c by ~5 points

**Q.** Did the v3 SSL pretrain produce an encoder that clears the
Track A acceptance gate (>= 76% supergroup MLP, >= 72% qap_name
MLP -- the v2 cell-c numbers)?  And is 20 epochs enough or do we
need more?

**Context.** Trained the v3 encoder at the Optuna-tuned
hyperparameters from `data/optuna/v3_pretrain_hpo_long.journal`
trial #44 (d_model=256, n_layers=4, n_heads=8, lr=1.47e-4,
mask_prob=0.230, weight_decay=7.67e-3, dropout=0.013).  Worried
mid-session that 40 epochs might still be undertraining since the
train and val SSL curves were both still descending in lockstep
(zero train/val gap) -- the classic "no overfitting but possibly
not converged" pattern.  Decided to compare 20-epoch and 40-epoch
probe results to settle it empirically.

**Considered.**

A. Take the 40-epoch probe number as the headline (more compute,
   more training).
B. Take the 20-epoch number as the headline; relegate 40 to a
   methodology footnote.
C. Run multi-seed at both to get error bars before deciding.

**Decision.** B.  Probe accuracy on the 20-epoch encoder at
library-default probe HPs (`scripts/eval/probe_encoder.py
--checkpoint epoch_best.pt`):

  qap_name LR     0.708
  qap_name MLP    0.762  (gate 0.72; +4.2)
  supergroup LR   0.737
  supergroup MLP  0.806  (gate 0.76; +4.6)
  k-NN qap purity 0.582 / 0.555 / 0.494 at k=5/10/50  (random 0.10)

40-epoch deltas vs 20 (same seed):

  qap_name LR     +0.003
  qap_name MLP    -0.003
  supergroup LR    0.000
  supergroup MLP  +0.003

All within probe stochasticity (>= +/- 0.005 from MLP init / data
shuffle / single-seed noise).  The SSL loss kept descending past
epoch 20 but the downstream metric did not -- the SSL-loss-as-
proxy curve flattens at the downstream level well before SSL loss
itself does.  20-epoch is the canonical budget; 40-epoch is kept
as `epoch_best_40.pt` for the paper ablation ("we trained to 40 to
verify convergence; no measurable gain past 20").  Skipped C
(multi-seed) because gate clears by ~5 points; the qualitative
answer doesn't change with error bars and the work is better spent
moving forward.

Probe HPO on the 20-epoch encoder afterwards tuned qap_name MLP
from 0.762 to 0.776 (30-trial TPE sweep over the MLP probe HPs;
summary in `results/probe_hpo/`).  The library-default 0.762 stays
the headline for the v2 comparison since cell-c was also probed at
library defaults.

Also closed this session: collapsed three confusingly-named eval
scripts into two (`probe_encoder.py --hpo` + `calibrate_probe.py`),
eliminated 7 CLI flags that overrode or duplicated YAML keys,
dropped `--pretrain-config` entirely by reading from `ckpt.config`
(the YAML and stamped config could silently disagree, e.g. a
post-training `frac_resolution` edit would rebuild the dataset at
the wrong tokenization), promoted the four probe modules into a
`cnmineral/eval/probe/` subpackage that mirrors `cnmineral/training/`
structurally, deleted one dead skipped test.  Net: cleaner CLI
surface, fewer redundant flags, structurally consistent package
layout.

**Next.**

- Track A part 2: PRETRAIN_APPLICATIONS, starting with element
  imputation (the natural first capability -- the pretrain head
  literally predicts missing elements from context).  See
  `docs/PRETRAIN_APPLICATIONS.md` for the 11-capability list and
  prioritization.  Target files: `scripts/eval/impute_elements.py`
  + `src/cnmineral/eval/imputation.py`.  Plan mode before writing
  code.
- Track B remains gated on the FC corpus regen pilots in
  `~/.claude/plans/sunny-weaving-nebula.md`.  Not the next move.
- Defer multi-seed and probe-HPO methodology rigor work to the
  paper-writing phase if reviewers ask for error bars.
- Two open questions for the next probe HPO run: the qap_name
  sweep's best hit the ceiling on both `hidden_dim` (best=1024 of
  [256, 512, 1024]) and `n_hidden_layers` (best=2 of [1, 2]).
  Methodology caveat before expanding: a bigger search space will
  push the probe number up, but past a certain probe complexity
  the metric stops measuring "is the latent organized well?" and
  starts measuring "can a big classifier extract signal?" --
  SimCLR/MoCo/CLIP standardize on *linear* probes for exactly this
  reason.  Expand ONE dimension at a time (start with `hidden_dim`
  to [256, 512, 1024, 2048]) and resist doing both at once.

## 2026-06-12 -- qap probe ceiling: widening hidden_dim to 2048 is within-noise

**Q.** Does widening the qap_name MLP probe past `hidden_dim=1024`
raise the latent-quality ceiling, or just measure classifier
capacity?

**Context.** The 06-11 sweep left the best railed on both
`hidden_dim` (1024 of [256, 512, 1024]) and `n_hidden_layers` (2 of
[1, 2]).  Ran a fresh 40-trial TPE study over `hidden_dim`
[256, 512, 1024, 2048] with only that dimension widened --
`n_hidden_layers` held at [1, 2] per the recorded methodology
decision (an arbitrarily deep/wide probe stops measuring whether
the latent is organized and starts measuring whether a big
classifier can extract signal).  Same 20-epoch encoder
(`epoch_best.pt`), same reproduced test split.  Separate study
`probe_hpo_qap_wide` so the 3-choice and 4-choice spaces don't
comingle in one TPE model; lands at `results/probe_hpo_wide/`.

**Result.** Best 0.7824 at `hidden_dim=2048`, `n_hidden_layers=2`,
vs the old 3-choice tuned 0.776 (+0.6pp).  LR baseline 0.708,
unchanged.  TPE leaned on 2048 (20 of 40 trials) and the best does
rail on the largest option again -- but the read is dominated by
noise, not capacity: the best *1024* trial in this same sweep hit
0.7791, only 0.0033 below the 2048 best and itself above the old
0.776.  Top-5 mixed 2048 and 1024 (0.7824 / 0.7815 / 0.7791 /
0.7790 / 0.7770), all inside the +/- 0.005 probe stochasticity band
the 06-11 entry already cites.  So most of the "gain" over 0.776 is
re-sampling the 1024 space with more trials, not 2048 capacity; the
2048-over-1024 increment is within noise.

**Considered.**

A. Chase further -- add 4096, since the best still rails on the max.
B. Stop -- accept ~0.78 as the qap probe ceiling at sane complexity.
C. Now expand `n_hidden_layers` too.

**Decision.** B.  A +0.6pp move for 2x probe width, with the 2048
edge inside probe noise of 1024, is the diminishing-returns regime
the methodology caveat warned about -- chasing 4096 would measure
the classifier, not the latent.  `n_hidden_layers` stays unexpanded
by the same logic: no evidence the latent needs more probe depth to
read out.  The library-default 0.762 remains the headline for the
v2 comparison (cell-c was probed at defaults too); the tuned number
nudges 0.776 -> 0.782 but changes no conclusion.  Closes the
`hidden_dim` open question; the `n_hidden_layers` one is retired on
methodology grounds rather than tested.

Housekeeping: reconciled the orphaned probe-HPO artifact tag.  The
0.776 study's journal + results dir carried tag
`pretrain_v3_seed0_epoch_best_20`, derived from the pre-rename
checkpoint `epoch_best_20.pt`; that file is now `epoch_best.pt`, so
the artifacts were renamed to the canonical
`pretrain_v3_seed0_epoch_best` tag and the summary's resolved-path
fields repointed (the `config` block kept as the sole record of the
3-choice space).  Old 3-choice study now under
`results/probe_hpo/`, new 4-choice under `results/probe_hpo_wide/`,
both tagged by the same encoder.

**Next.** Unchanged: Track A part 2 (element imputation, plan mode
before code); Track B still gated on the FC corpus pilots.

## 2026-06-15 -- element imputation lands; trace failure is loss-balance not architecture; foundation-model framing + Phase 1 diagnostic

**Q.** Why does the frozen encoder+head impute major elements well but
fail completely on traces, and what does that failure tell us about how
to choose a pretraining objective for a model whose whole point is to
serve many downstream capabilities?

**Context.** Element imputation (PRETRAIN_APPLICATIONS capability #1)
landed: `src/cnmineral/eval/imputation.py` + `scripts/eval/impute_elements.py`.
It runs the SSL `PretrainHead` at inference -- deterministic
leave-one-element-out over the Gard test split reconstructed from
`ckpt.config`, mask exactly element X (`src=119, frac=0`), predict it,
compare to the withheld truth, back-transform to wt% (majors) / ppm
(traces).  No new training; a standalone use of encoder+head.  Needed a
head-returning checkpoint loader (`load_encoder_and_head_from_checkpoint`
in `eval/latent.py`); the probe path discards the head, imputation needs
it.

Result on `pretrain_v3_seed0/epoch_best.pt`:
- Majors impute well: R2 0.68-0.94 (Si 0.94 / 1.9 wt% MAE, Na/Ti
  weakest ~0.68), uniform n=33371 (require_all_majors guarantees all
  eight non-Cr majors present).  Cr drops out (slot padded under
  include_cr2o3=False).
- Traces fail: negative R2 across the board (Yb -2505, Th -1965),
  predictions pinned to a ~1e-3 frac floor regardless of truth.  n
  varies 11k-24k (trace coverage is patchier than majors).

Diagnosed root cause, confirmed not a bug.  The SSL loss
(`losses.pretrain_loss.masked_element_mse`) is raw-frac squared error
pooled uniformly across masked positions.  Per-element contributions:
Si frac ~0.25, 10% error squared ~6e-4; Yb frac ~3e-6, *100%* error
squared ~9e-12 -- a ~1e7-1e8 ratio.  The optimum is to nail majors and
emit a generic small constant at trace slots, which is exactly what
happens (spot-check: Yb true frac 3e-6, predicted 9e-4).  The dual
linear+log `FractionalEncoder` is INPUT-side resolution only (it lets
the encoder *read* small fracs); it does not touch the loss, so traces
remain effectively unsupervised.  The negative-R2 trace numbers are a
broken-by-objective floor, not a fundamental ceiling.

**Discussion (the framing this surfaced, recorded because it governs
the next several experiments).**

- *SSL is self-supervised, not supervised, despite the MSE.*  The
  mechanics are identical to supervised regression -- target, loss,
  backprop.  The distinction is the *source of the target*: the masked
  frac is a piece of the input we hid from ourselves, not an external
  annotation.  No labeler, no separate measurement; the same value is
  input on one masking draw and target on another.  This is what lets
  us pretrain on 474k raw Gard bulks with zero labels.  Contrast: the
  supervised heads (ModeHead/CondHead) take genuinely external targets
  (MAGEMin/PetThermoTools outputs).  SSL = "supervised learning where
  the labels are free."

- *The contribution is the latent representation, not any one task.*
  One pretraining run yields a reusable encoder: impute any element
  zero-shot, k-NN retrieval, OOD flagging, supergroup classification,
  etc.  An XGBoost competitor must fix (inputs -> outputs) a priori and
  is locked to that single task -- change the input set or the target
  and you retrain a fresh model; there is no latent to reuse for the
  other ten capabilities.  That is the generalist-vs-specialist trade:
  the specialist wins on its one narrow task (especially right now,
  where our traces are broken-by-objective and *anything* reasonable
  beats them -- not a fair contest), the foundation model wins on
  breadth and amortization.  The pitch is never "we beat XGBoost at
  Yb"; it is "one latent serves many jobs, and is competitive where the
  task benefits from learned structure."

- *No task-neutral objective exists (No Free Lunch).*  Every
  pretraining loss induces a latent geometry, which is a prior over
  which downstream tasks are easy.  A foundation model is a bet on a
  prior, not an escape from having one.  You cannot derive *the*
  objective optimal across all capabilities; you pick the one whose
  prior best matches the expected distribution of tasks.

- *Head vs encoder objective are different questions.*  The head
  objective should simply match its task -- for imputation, log/relative
  error -- and it costs nothing to other capabilities because the head
  is a thin, discardable adapter.  The actual mistake was reusing the
  *pretrain* head as a task head; imputation today has no objective of
  its own, it inherits the SSL loss.  The encoder objective is the
  consequential, shared one.

- *The current encoder objective is biased by accident, not design.*
  Raw-frac MSE is major-dominated, which happens to align the latent
  with major-element petrology -- rock type, QAPF/supergroup, the
  differentiation manifold are all major-driven, which is why the
  encoder hits 80.6% supergroup.  But that alignment came from the
  *units* (frac magnitudes), not from a decision that majors matter
  most for all future tasks.

- *Rising tide vs tradeoff for raw-frac -> log reconstruction.*  In
  general, rebalancing the encoder loss is a tradeoff (finite capacity,
  NFL).  But severity depends on the marginal utility of loss weight,
  and here it is wildly unequal.  Majors are easy and near-saturated:
  high coverage, high SNR, low intrinsic dimension (igneous majors live
  on a ~few-axis liquid-line-of-descent manifold) -- taking loss weight
  away barely moves their error.  Traces are hard and starved: sparse,
  noisy, higher intrinsic dimension (REE patterns, incompatible-element
  ratios carry real petrogenesis/source information) -- adding weight
  helps a lot.  Reallocating from a saturated-easy band to a
  starved-hard band is close to Pareto-improving, and log space does
  not destroy major information (majors are still perfectly predictable
  in log units).  Predicted outcome of raw->log: traces up
  substantially, majors roughly flat.  Knowable a priori: the
  *direction*; that the major cost is *small* (low-dim major manifold =>
  capacity headroom).  Not knowable without the run: the exact
  magnitudes.

The four objectives in play (current is one loss in two roles; the
proposed fix differs by scenario):
- Current (both scenarios): `L = mean_{i in M} (f_hat_i - f_i)^2`, raw
  frac, uniform pooling.  Shaped the latent AND trained the head we
  read.
- Proposed, Scenario A (latent already has trace info): new read-out
  head on frozen `h`, per-element, log target:
  `L = mean_n (g_e(h_n) - log10(ppm_{e,n}))^2`.  Encoder untouched.
- Proposed, Scenario B (latent is trace-blind): rebalanced *pretraining*
  loss, head emits log-frac directly:
  `L = mean_{i in M} (y_hat_i - log10(f_i))^2`, back-transform 10^y_hat
  (also removes negative predictions).  Variants: per-token standardized
  MSE, per-token weights.  Requires re-pretrain.

**Considered.**

A. Assume the latent has the trace info -- just fix the head loss
   (cheap) and ship.
B. Assume it does not -- re-pretrain with a rebalanced loss.
C. Run a diagnostic gate first, then fix at the layer the data
   indicates.

**Decision.** C.  Phase 1 (cheap, read-only on the canonical checkpoint,
no retraining): establish two reference points per element in log-ppm
space on the same held-out test, all scored through the existing
`imputation.py` harness (it scores any predictor, not just the head) --
(1a) input-space XGBoost/ridge baseline predicting each element from the
*other measured elements* (this is also the deferred specialist baseline
and the practical ceiling), (1b) the same regressor on the frozen
latents `h`, (1c) the current pretrain-head number for reference.  Gate
on 1b vs 1a: 1b ~ 1a => Scenario A (latent has it; cheap head fix,
encoder untouched, and 1b *is* the Scenario-A read-out); 1b << 1a =>
Scenario B (latent blind; rebalanced re-pretrain).  Do not touch the
canonical encoder until the gate says B; if B, the required eval is both
the imputation re-score AND re-running the supergroup/qap probe suite to
measure the rising-tide-vs-tradeoff prediction (traces up, majors flat).
Even if the gate is A, defer any rebalanced encoder to whenever the
*other* trace-dependent capabilities (source discrimination,
OOD-on-traces) prove they need it -- imputation alone should not trigger
a re-pretrain.

Review fixes applied to the landed eval: dropped duplicate
`_EXTRACT_BATCH_SIZE` / trivial `_N_MAJORS` constants, guarded the
log-scale identity line + per-panel log toggle against unconstrained
(possibly non-positive) head output.  Fe back-transforms to cation wt%
(it is a FeO+Fe2O3 blend, no single-oxide inverse); flagged for the
tutorial.

**Next.** Phase 1 diagnostic in a fresh session; user supplies prior
XGBoost reference codebases.  Don't pay for an encoder re-pretrain
unless the gate lands on B.

## 2026-06-15 -- trace-imputation gate result: latent carries the trace signal; fix the head, encoder untouched

**Q.** Does the frozen encoder latent still carry trace-element
information (so a cheap read-out head can recover imputation), or is the
latent trace-blind (forcing a rebalanced re-pretrain)?

**Setup.** Ran `explorations/trace_imputation_loss_gate/run_gate.py`
(40 Optuna trials, no subsample) on `pretrain_v3_seed0/epoch_best.pt`
(d_model 256, epoch 20, n_test 33,371).  Two controlled arms per element,
both XGBoost scored in log10-frac space on the Gard test split, differing
only in the feature set:
- 1a input-XGB: predict each element from the *other measured elements*
  (the practical ceiling / specialist baseline).
- 1b latent-XGB: predict from the frozen latent with the target token
  masked before encoding (leakage-guarded).
The current pretrain head was dropped as an arm -- trained on linear
raw-frac MSE, it is not comparable in log space; its fair linear-space
numbers stay in `results/imputation/`.

**Result.** Over the 13 traces, median log-space R2: 1a 0.892, 1b 0.839.
1b reaches a median 94% of the 1a ceiling's R2 (per-trace 84-96%), and
every trace recovers at 1b R2 0.72-0.90.  1a beats 1b on *every* trace --
the gap is real, not noise: median 0.05, largest on the HREE (Yb -0.14,
Y -0.11), smallest on Th/Rb/Ni/V (~0.04).  Majors: 1b within 0.01-0.04
of 1a.

**Reading the numbers (the gap is genuine -- recorded because it tempers
the verdict).** The decision the gate makes is blind vs not-blind, not
"1b == 1a."  A trace-blind latent would give 1b R2 ~ 0; instead 1b is
strongly positive and tracks the ceiling closely, so the trace
information clearly survives the encoder.  The persistent 1a-1b gap is
real signal the 256-d latent dropped relative to the raw inputs -- a
lossy-but-high-fidelity compression, worst on heavy REE.  Hence: the
imputation failure is the *head's objective* (linear raw-frac MSE), not
encoder capacity; and a log-space read-out head on the frozen latent
should recover most of the traces, bounded above by 1b (so still below
the 1a input ceiling, especially on HREE).

**Decision.** Build the cheap fix first: a log-space read-out head on the
frozen latent (issue #5).  Do NOT re-pretrain the encoder for imputation
-- the latent is not blind; encoder loss unchanged, so no ADR.  The
residual gap (esp. HREE) is the legitimate motivation for an *optional,
later* rebalanced re-pretrain (the rising-tide experiment), paid for only
if a downstream capability needs ceiling-level traces -- not forced by
imputation alone.

**Caveats.** Both Optuna studies landed on the same seeded startup trial
(`best_1a == best_1b`), so the 40-trial search added little beyond random
startup; 1b may be mildly under-tuned (true gap to 1a possibly smaller).
The coded verdict ("A" if median 1b >= 0.8x median 1a) is an advisory
heuristic that collapses two distinct questions -- "is the latent blind?"
and "how close to the ceiling?" -- into one ratio; report them separately
going forward (issue #6).

**Next.** Read-out head (#5, unblocked).  Validate the traces-only tuning
objective and split the gate's blindness-vs-ceiling reporting (#6).
Open: set the downstream bar for trace imputation -- "useful" (read-out
head suffices) vs "match the input ceiling on HREE" (would justify the
rebalanced re-pretrain).
