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
0.13 is irreducible by honest means (ADR 008).

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

**Setup.** Ran `explorations/trace_signal_in_latent/run_gate.py`
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
should recover most of the traces, approaching 1b (a strong nonlinear
readout's reach on the latent, not a hard capacity ceiling -- a linear head
can trail it on readout capacity; see the 2026-06-16 ablation) and staying
below the 1a input ceiling, especially on HREE.

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

## 2026-06-15 -- log-space imputation read-out head lands (the head-gap fix)

**Q.** Build the cheap fix the gate pointed to: a head on the frozen
encoder that imputes masked elements in log space, replacing the SSL
head's linear-frac outputs that collapse on traces.

**What landed.** A production read-out head, encoder untouched:
- `LogImputationHead` (`models/imputation_head.py`) -- linear or
  one-hidden-MLP read-out emitting per-element *standardized* `log10(frac)`;
  de-standardization buffers (`target_mean`/`target_std`) live in
  `state_dict` so they travel with the checkpoint and stay
  encoder-version-agnostic.
- `masked_element_log_mse` (`losses/pretrain_loss.py`) alongside the
  linear-frac `masked_element_mse` (kept as the contrast).  The log space is
  the rebalance that lifts traces off the floor; per-element standardization
  is a smaller refinement that mainly rescues the lowest-signal heavy REE via
  the global gradient clip (it barely moves a linear head -- see the
  2026-06-16 ablation).
- `train_imputation_head` (`training/imputation_head.py`) -- frozen
  encoder (eval, `requires_grad_(False)`, encode under `no_grad`), head-only
  optimizer; dataset/splits reconstructed from the *encoder* checkpoint's
  config so train/val/test line up with the gate and probe suite.  Produces
  `ImputationHeadCheckpoint` (head + buffers + a pointer back to the encoder
  checkpoint; no encoder copy).
- `run_imputation_logspace` (`eval/imputation_logspace.py`) -- log-space
  leave-one-out scoring with the gate's two reference lines (input-element
  XGBoost ceiling, frozen-latent XGBoost target) read from the gate JSON;
  writes `results/imputation/<encoder_tag>/readout/`.

**Why a separate head, not a re-pretrain.** Settled by the gate above: the
frozen latent already carries the trace signal, so fixing imputation needs
only the right objective on the existing latent.  Re-supervising a head
post-hoc on a frozen encoder is the documented application-slot pattern
(same family as ModeHead/CompHead/CondHead) -- not a new architecture.
`PretrainHead` stays as SSL machinery; this head is the imputation
application.  Concept written up in `docs/PRETRAIN_APPLICATIONS.md` #1.

**Naming.** Avoided the gate's `1a`/`1b` code-names in code/output: metric
keys `log_r2`/`log_mae`/`log_rmse`, reference keys `input_xgb_ceiling_r2`
/ `frozen_latent_xgb_r2`; the eval prints the criterion + measured values
(head recovers X% of the frozen-latent target, Y% of the input ceiling)
with no pass/fail verdict -- this is the "before" baseline the optional
re-pretrain (the rising-tide experiment) is measured against.

**Next.** Run the head on the full corpus, ablate standardize and
linear-vs-MLP, and confirm median trace `log_r2` rises off the floor toward
the frozen-latent target.  Results in the 2026-06-16 entry below.

## 2026-06-16 -- imputation head full-corpus run: MLP default, and what standardization actually buys

**Q.** Does the head close the head-gap on the full corpus, is the linear
read-out enough, and is per-element standardization really "the rebalance"?

**Setup.** Trained on `pretrain_v3_seed0/epoch_best.pt` (frozen), scored
log-space leave-one-out on its Gard test split.  Yardsticks from the gate:
frozen-latent XGBoost median trace R2 0.839, input-element ceiling 0.892.
Harness + per-element JSON: `explorations/imputation_head_design/`.

**Results** (median trace log-R2 / Yb, the worst trace / median major):

| variant | median trace | Yb | median major |
|---|---|---|---|
| linear + standardize | 0.748 | 0.544 | 0.830 |
| linear, no standardize | 0.743 | 0.439 | 0.813 |
| MLP (hidden=256) + standardize | 0.792 | 0.623 | 0.866 |
| linear + standardize, single-token mask eval | 0.757 | 0.555 | 0.840 |

**Three conclusions, two of which corrected overstated framing:**

1. **Log space does ~all the work; standardization is a narrow grad-clip
   refinement, not "the rebalance."** Linear-no-standardize already hits
   0.743 median; standardization adds +0.005 median.  Its real effect is
   local -- it rescues the lowest-signal heavy REE (Yb +0.105, Y +0.071) and
   little else.  Mechanism: for a linear head each output is an independent
   affine map, so per-element target scaling is absorbed into the weights and
   R2 is scale-invariant; the only coupling is the global
   `clip_grad_norm_`, where un-standardized high-variance majors consume the
   clip budget and starve the HREE.  Kept `standardize: true` (the HREE
   rescue is free), but the loss/model docstrings now say "log space fixes
   the collapse; standardization buys the bottom traces."  (MLP-without-
   standardize not run; the HREE rescue is the linear-ablation finding.)

2. **MLP is the shipped default, not linear.** MLP beats linear by +0.044
   median (+0.08 on Yb) and lands at 0.792 = 94% of the frozen-latent
   target -- the same 94% the gate advertised as recoverable.  The artifact
   owner reaches for the default, and the default should be the head that
   reaches the documented target.  `configs/imputation/imputation_head.yaml`
   now ships `hidden: 256`.

3. **The frozen-latent XGBoost number is a strong-readout reference, not the
   head's hard ceiling.** It is a tuned gradient-boosted forest, so a linear
   or one-layer head can trail it on readout capacity alone even when the
   latent holds the signal -- which is exactly what the MLP closing most of
   the linear gap shows.  Reworded the eval/model/PRETRAIN prose to stop
   implying the head's ceiling is that forest; the per-element "% of target"
   print was already fair and stays.

**Minor.** The train(multi-mask)/eval(single-token LOO) mismatch barely
matters: matching it lifts median only +0.009 -- not worth a code change.
Fixed `_plot_logspace_r2`'s hard-coded `ylim(0, 1)` so a genuinely negative
per-element R2 renders as a negative bar instead of clipping to the floor.

**Net.** The head closes the head-gap (negative -> 0.79 median trace,
majors 0.87); the MLP recovers ~94% of the frozen-latent target.  The
residual to the input ceiling is the latent gap, the encoder's job -- the
optional rebalanced re-pretrain (rising-tide experiment), not this head.

## 2026-06-16 -- modal target: keep liq + fl in the corpus (do not renormalize to solids)

**Q.** The cumulate FC modal target includes residual liquid (`liq`) and
free fluid (`fl`).  Should the ModeHead target instead be solids-only --
drop `liq`/`fl` and renormalize the minerals to 1?

**Context (the two mass balances).** MAGEMin equilibrates a *closed* bulk:
for every component, `bulk_j = sum_phases mass_p * X_{p,j}` (Gibbs
minimization under element conservation).  The cumulate mode (all
fractionated solids + residual liq + fl at `T_FC_end`) therefore
mass-balances exactly to the bulk we fed it.  The model inherits that
closure as a consistency check: ModeHead (Dirichlet on the simplex) x
CompHead (per-phase composition) should reconstruct the *dry* input oxides,
`sum_p mode_p * X_{p,j} ~= bulk_j` -- the same equation, now a training
signal / the motivation for a future single endmember-wt% head.  H2O is the
one component outside that dry sum (model never sees it; CondHead infers
it).

**Considered.** Solids-only renormalized would be *more comparable to
observable mineralogy* (thin-section modes, the mineralML grain corpus we
hold out for cross-modal validation) and might *raise the ModeHead per-path
R2 ceiling* by removing `liq`, one of the most path-variable quantities.
But it **breaks the closure to the input bulk**: the discarded residual melt
+ fluid are small in mass (`T_FC_end` is defined at residual liq < 5%, so
~5-6% total) yet hold the *incompatible/volatile* budget disproportionately
(excess SiO2, K2O, Na2O in the melt; H2O in the fluid), so the per-element
reconstruction error is composition-skewed, not uniform.  It also discards
the melt-fraction = degree-of-crystallinity signal and the free-water
bookkeeping.

**Decision.** **Keep `liq` + `fl` in the corpus modal target.**  They are
small (~5-6% at `T_FC_end`), and keeping them preserves the exact
closed-system mass-balance closure, which keeps model-building simpler --
the `sum mode*comp = bulk` consistency check stays valid and the
endmember-wt% head stays reachable.  Renormalizing to solids would trade
that closure for marginal comparability gains and a skewed residual; not
worth it at this stage.

**Deferred (decoupled).** How the *user-facing* output is presented (e.g.
reporting a solids-only mineral mode, since a petrologist reads minerals,
not trapped melt/fluid) is a **presentation-layer** choice to revisit at
productization.  It does not require changing the training corpus -- drop
`liq`/`fl` and renormalize at display time if/when a user-facing surface
wants it.  Corpus stays closed-system; the public view can be post-processed
later.


## 2026-06-18 -- SSL re-pretrain objective: standardized log-space MSE beats linear-space MSE

**Q.** Does re-pretraining the SSL encoder with a trace-aware loss raise the
frozen latent's trace-imputation accuracy without lowering the major-driven
classification probes, relative to the linear-space masked MSE the canonical
encoder (`pretrain_v3_seed0`) trains with?

**Context.** The canonical encoder trains with linear-space masked-element
MSE, where per-element target magnitudes span ~1e7-1e8x (majors at wt%/100,
traces at ppm/1e6), so trace residuals contribute almost nothing to the loss.
The 2026-06-16 imputation-head ablation showed that log space, not the head
architecture, is what makes the frozen latent's trace readout work.  This
asks whether moving the encoder's own pretraining objective into a trace-aware
space carries the same benefit into the latent itself.

**Setup.** Three objectives, each Optuna-tuned (30 trials) over
`{lr, weight_decay, dropout, mask_prob, batch_size}` at the pinned 256/4/8
architecture, seed 0, so the objective is the only variable vs the canonical
encoder: per-element standardized log-MSE (`log_mse`, `standardize: true`),
plain log-MSE (`log_mse`, `standardize: false`), and inverse-variance-weighted
linear-MSE (`weighted_mse`).  Configs:
`configs/pretrain/pretrain_hpo_{logmse_std,logmse_plain,weighted}.yaml`.  Each
candidate scored on two read-only instruments on the same Gard test split:
the trace-imputation gate (frozen-latent XGBoost, target masked before
encoding; `explorations/trace_signal_in_latent/`) and the classification
probes (frozen-latent logreg + 1-hidden MLP on supergroup and qap_name;
`scripts/eval/probe_encoder.py`, fixed protocol + 30-trial MLP HPO).

**Results** (frozen-latent metrics; baseline = canonical linear-MSE encoder):

| metric | baseline | std log-MSE | plain log-MSE | weighted lin-MSE |
|---|---|---|---|---|
| gate median trace log-R2 | 0.839 | 0.877 | 0.871 | 0.706 |
| gate Yb log-R2 (worst trace) | 0.724 | 0.831 | 0.807 | 0.526 |
| supergroup MLP probe | 0.806 | 0.821 | 0.812 | 0.549 |
| qap MLP probe (fixed) | 0.762 | 0.769 | 0.758 | 0.514 |
| qap MLP probe (HPO) | 0.776 | 0.790 | 0.781 | 0.609 |

Standardized log-MSE improves every element in the gate (8/8 majors
+0.002..+0.032, all 13 traces, worst trace Yb +0.107) and all three probes;
nothing regresses.  Plain log-MSE moves the same direction but is uniformly
weaker.  Inverse-variance-weighted *linear* MSE degrades both instruments
(trace median 0.706, supergroup MLP 0.549): equalizing per-element loss
contribution by weight does not address a dynamic-range problem that lives in
the residual magnitudes -- the optimization still chases the large-magnitude
majors and learns a latent carrying less trace signal and less class
structure than even the linear-MSE baseline.

**Seeds.** The standardized-log winner was re-pretrained at seed 1 from the
same HPs; both instruments replicate (gate median trace 0.873, Yb 0.841;
supergroup MLP 0.820, qap MLP 0.771, qap HPO 0.788).  The 1a input-element
ceiling is unchanged across encoders (median 0.892 vs 0.891), confirming the
gate's reference arm is encoder-blind.

**Decision.** SSL pretraining with a per-element standardized log-space
masked-element MSE loss yields higher frozen-latent trace-imputation accuracy
(median trace log-R2 0.877 vs 0.839, worst-trace Yb 0.831 vs 0.724) and higher
classification-probe accuracy (supergroup MLP 0.821 vs 0.806, qap MLP 0.769 vs
0.762) than the linear-space MSE loss, replicated across two seeds;
inverse-variance-weighted linear-space MSE is worse on both.  We adopt the
standardized log-space MSE objective as the canonical SSL encoder
(`checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt`), retiring the
linear-MSE `pretrain_v3_seed0`.  Binding choice in ADR 007.

**Next.** Downstream (corpus regen #9, supervised heads #10/#13, imputation
read-out tune #8) loads the new canonical encoder.  The imputation head (#8)
re-scores against the new frozen latent: its earlier frozen-latent yardstick
(0.839 median trace log-R2) becomes 0.877.

## 2026-06-22 -- trace-gate tuning objective audit: traces-only HPs are objective-invariant

**Q.** The gate (`run_gate._tune_shared`) tunes XGBoost by minimizing mean
log-space val-MSE over the 13 traces only, then reuses those trace-tuned params
for the major final fits too.  Two unchecked assumptions: (a) the trace R2s and
the 1a/1b verdict don't shift under a different tuning objective; (b) the major
R2s aren't handicapped by trace-tuned params.  Validate both before #8 leans on
the gate again (issue #6).

**Context.** Cheap, read-only methodology audit on the canonical encoder
(`pretrain_v3_logmse_std_seed0/epoch_best.pt`).  Not on the critical path; run
now while the gate code is fresh.

**Setup.** `explorations/trace_signal_in_latent/audit_tuning_objective.py`
(throwaway) imports the gate helpers unchanged.  Each masked-token latent is
encoded once and cached, so only XGBoost re-tunes/refits.  Six Optuna studies =
arm {1a input-XGB, 1b frozen-latent} x objective {traces-only, majors-only,
all-22}, 40 trials each on all 106787 fit / 26697 val rows (production parity).
Final per-element fits on the full 33371-row test split: traces under
best_traces vs best_all (question a); majors under best_traces vs best_majors
(question b); materiality threshold 0.005.  Harness cross-check: traces under
best_traces reproduces the cached gate exactly (median 1a 0.892, 1b 0.877).

**Results.** All six studies converge to a *single* XGBoost param set
(max_depth 10, lr 0.018, subsample 0.59, colsample 0.88, min_child_weight 10,
gamma ~0).  Every final-fit R2 delta is therefore exactly 0.000 (max |delta|
1a/1b = 0.000 for both questions); the trace verdict stays
`latent_carries_signal` under both best_traces and best_all.  Per-element trace
R2s span 0.820 (Y, 1b) to 0.951 (Ce, 1b); majors 0.809 (Na, 1b) to 0.942 (Fe,
1b).

The objectives are genuinely distinct problems -- their best values differ
(0.0277 traces / 0.0154 majors / 0.0230 all, for 1a), as expected from averaging
different element subsets.  The identical optimum is the expected signature of a
controlled paired comparison, not an artifact: all studies share the run seed by
design, so they share the 10 random TPE-startup trials and only the objective
varies.  After startup, TPE's choices are driven by the *ranking* of trials
under the objective; an objective that preferred different HPs would diverge the
trajectory after startup and land on a different best trial/params.  None
diverged (all six lock to best trial 22) -- so the three objectives induce the
same HP preference ordering, which is the finding.  A per-objective distinct seed
would reintroduce a tuning-luck confound (a better final R2 could mean a better
objective or just a luckier draw), so the shared seed is the correct control.
This validates along the single trajectory the gate runs in production (one
seed), which is the condition that matters.

**Decision.** The traces-only tuning objective is validated: on this corpus the
XGBoost HP optimum is invariant to which element subset defines the objective,
so the gate's trace verdict and its major R2s are unchanged whether tuned on
traces, majors, or all 22.  No reason to change the gate's objective.  Results
in `results/pretrain_v3_logmse_std_seed0_epoch_best/audit_tuning_objective.json`;
the audit script is kept as a documented one-off (not wired into the gate).

**Next.** Close #6.  #8 (imputation read-out tune) proceeds on the canonical
latent unblocked.

## 2026-06-24 -- imputation read-out head tuned + shipped (application #1)

**Q.** Tune the log-space imputation read-out head on the frozen canonical
encoder and ship it as production application #1, the first standalone use of
the SSL latent.

**Context.** The 2026-06-16 design ablation settled the head *family* (one-hidden
GELU MLP, per-element log-standardized) at a single fixed HP point on the
now-retired linear-MSE encoder; nobody had tuned the HPs, and the canonical
encoder (`pretrain_v3_logmse_std_seed0`, ADR 007) is a different latent.  Its
trace yardsticks (gate diagnostic): frozen-latent XGBoost median 0.877,
input-element ceiling 0.892.

**Setup.** Single-stage Optuna (TPE, JournalStorage), objective = median trace
leave-one-out log-R2 on a held-out val split carved from trainval (test never
touched in tuning).  Encoder frozen, head-only; per-trial val-loss early stopping
with best-weight restore; no pruner (the objective is an end-of-trial LOO).
Search: hidden, lr, weight_decay, dropout, mask_prob.  Untuned reference (the
ablation's fixed config) scored on the same val split: 0.8247 traces / 0.8854
majors.  Two sweeps: a 40-trial production sweep, then a 120-trial ceiling sweep
with hidden extended to 4096 and the lr floor lowered.

**Results.** Production sweep best (trial 34): val 0.8327 traces / 0.8948 majors
at hidden=1024, lr=1.40e-4, wd=2.84e-7, dropout=0.066, mask_prob=0.217.  Ceiling
sweep best: val 0.8426 at hidden=4096 (+0.010).  But hidden pegged at the search
max in *both* sweeps and the optimization history plateaued (+0.001 over the last
~80 trials), so ~0.842-0.845 is the one-hidden-layer MLP's ceiling on this latent
(~96% of the GBT readout); the residual to 0.877 is a readout-class gap (single
MLP layer vs gradient-boosted forest), not capacity the head should chase.
Shipped the lean hidden=1024 head -- a 4x-smaller artifact (285k vs 1.1M params)
for -1pp, parsimony over the marginal gain.

Final head on the 33371-row test split: median trace log-R2 0.8363 (95% of the
0.877 GBT yardstick, 94% of the 0.892 input ceiling), median major 0.8913 (no
regression vs the 0.8854 untuned baseline).  Per trace:

| trace | head | %GBT | trace | head | %GBT |
|---|---|---|---|---|---|
| Sr | 0.821 | 96% | Nb | 0.836 | 95% |
| Zr | 0.809 | 95% | V  | 0.876 | 97% |
| Ba | 0.812 | 96% | La | 0.924 | 98% |
| Y  | 0.780 | 95% | Ce | 0.936 | 98% |
| Rb | 0.883 | 97% | Nd | 0.930 | 98% |
| Ni | 0.826 | 95% | Yb | 0.792 | 95% |
| Th | 0.859 | 96% |    |       |     |

The two lowest (Y 0.780, Yb 0.792) are the HREE -- the latent's information loss
(#7's closed job), not the head's: the head tracks 95-98% of the GBT on every
trace.

**Decision.** Ship the hidden=1024 head as the production element-imputation
head; application #1 is shipped.  `configs/imputation/imputation_head.yaml`
regenerates it; the checkpoint + readout artifacts stay gitignored (regenerable,
per repo convention).  No verdict -- this lands an artifact, not a finding.

**Next.** Capability refresh on the locked encoder (imputation summary + latent
UMAP), then tag the foundation-encoder-plus-applications stable point, then Track
B corpus regeneration (#9).

## 2026-06-24 -- close-out housekeeping: SSL decoder knobs settle on evidence, weighted_mse retired

**Q.** Before tagging the foundation-encoder-plus-applications stable point, can
the deferred SSL-decoder-knob investigation (#1) close, and on what grounds?

**Context.** #1 deferred an ablation of two `PretrainHead` settings -- the hidden
bottleneck (`d_model // 2`) and the output activation (`None` vs `sigmoid`) --
with three triggers to revisit: SSL reconstruction MSE plateaus high vs v2,
supergroup probe drops >2pts below v2 cell-c's 76.3%, or predicted frac lands
outside [0,1] for >5% of masked positions.  The settings have since been baked
into the constructor (no config knobs; `output_activation` does not exist as a
parameter), and the encoder is locked (canonical = `pretrain_v3_logmse_std_seed0`).

**Considered.** Re-running the ablation requires re-pretraining with alternate
head settings, which would either be moot (we ship the existing encoder) or break
the lock.  But the close does not rest on that:

- **Output activation is moot under log-space, by correctness not cost.** The
  canonical objective is standardized log-space MSE (ADR 007); the head output is
  consumed as standardized `log10(frac)`, unbounded in R.  Sigmoid squashes to
  (0,1) and is provably wrong for that target -- identity is the only correct
  choice.  Trigger 3 ("frac outside [0,1]") is moot for the same reason: the head
  no longer predicts raw frac.
- **The probe trigger fired in the close direction.** The capability refresh on
  the canonical encoder gives supergroup MLP probe test acc = 0.821, +5.8pp over
  v2 cell-c's 0.763; trigger 2 asked to revisit on a regression, and there is
  none.
- **The hidden-width knob's predicted irrelevance is corroborated.** #1 expected
  small deltas (decoder capacity incidental).  The #8 imputation-head HPO (same
  MLP-on-latent shape) measured ~1pp per 4x width with a plateau -- capacity is
  incidental on this latent, as predicted.
- **Supporting:** the SSL decoder is a training-time scaffold, discarded at
  inference; only the encoder ships, and it is the validated artifact.

Doc-rot fixed in the same change: `PretrainHead`'s docstring still claimed "MSE
pulls predictions toward frac in [0,1]" (pre-log-space); corrected in the
docstring and `docs/encoder.md`, both now stating the fixed-scaffold rationale.

Separately retired `weighted_mse` (inverse-variance-weighted linear MSE) as dead
code: ADR 007 rejected it, its runtime artifacts were deleted, and the loss fn +
objective branch + config + tests are gone.  The ADR and the 2026-06-18 entry
keep the record of why.

**Decision.** Closed #1 with no code change beyond the docstring/doc fixes; the
refresh reconfirmed the supergroup probe at 0.821 >= 0.763 (the live re-check of
trigger 2 cleared).  The fixed SSL decoder -- `d_model // 2` bottleneck, identity
output, neither a hyperparameter -- is confirmed correct, not merely deferred.

**Next.** The #4 splits-hardening fix, full pytest, tag, merge v3 -> main.

## 2026-06-24 -- encoder geometry audit: the 40-epoch result was the linear encoder; kNN ~0.58 is inherent geometry, not under-training

**Q.** Auditing the canonical encoder's probe / UMAP / kNN before tagging: are the
weak-looking UMAP separation and kNN purity (~0.58) a sign of under-training (too
few epochs), or something else?

**Context.**
- Probe refresh on the canonical log encoder: supergroup MLP 0.821, qap MLP 0.769,
  kNN qap purity 0.582 / 0.554 / 0.487 (k=5/10/50) vs 0.100 random.  Encoder val
  loss best = epoch 20 of 20 (the last), still descending, constant LR (no
  scheduler) -- which superficially looks under-trained.
- Recalled a prior 20-vs-40-epoch experiment that found no gain.  Verified the
  artifact: it was `checkpoints/pretrain_v3_seed0/epoch_best_40.pt` -- the
  **linear-space** encoder (val_loss 5e-5, old HPs from trial #44), NOT the
  canonical log encoder.  So "40 = 20" is only resolved-by-analogy for the
  canonical log encoder; it has never been tested at 40 epochs directly.

**Considered / findings.**
1. The probe MLP-loss curves (24/28 epochs) are PROBE early-stopping (patience 5),
   not encoder training -- a red herring for the encoder-epochs question.
2. "Best = last epoch, SSL loss still descending" is the known pattern: the
   2026-06-11 entry already established that SSL loss descends past 20 while the
   downstream metric saturates well before.  SSL-loss descent does not imply
   downstream gain.
3. Decisive: kNN k=5 purity is ~0.582 across THREE encoders -- linear-20 (0.582),
   linear-40 (0.585), log-20 (0.582).  Stability across both objective and epoch
   count means ~0.58 is a data/geometry property, not a training-duration artifact.
4. The real signal: a supervised MLP probe reads qap at 0.769 while raw-latent kNN
   purity is 0.58 -- a ~19pp gap.  The class information IS in the latent (a
   classifier extracts it) but the latent does not arrange same-class rocks as
   neighbors.  This is the masked-LM "anisotropic embedding" symptom
   (decode-but-don't-cluster), not under-training and not too-messy data.

**Decision.** The weak unsupervised geometry is a representation-objective
limitation, not an epoch-budget problem -- more epochs of the same masked-element
objective will not move it (linear 20-vs-40 null + cross-encoder kNN stability).
Improving it needs a different SSL pretext that shapes the embedding geometry;
opened as epic #15 (P1-high).  Measurement discipline going forward: raw-latent
probe + distance-weighted / supergroup kNN, NOT UMAP-2D aesthetics (UMAP is lossy,
and rock chemistry is largely continuous, so discrete 2D separation is partly the
wrong target).

**Open caveat.** One 40-epoch run of the CANONICAL log encoder + reprobe would
convert "resolved-by-analogy" into "directly verified" for ~20 min compute;
deferred as low expected value given the cross-encoder stability, noted in #15.

**Next.** Pursue #15 (stronger representations via new SSL pretext objectives --
pull-together / forgery-detection / group-masking) on the `encoder-representation`
branch.

## 2026-06-25 -- input-kNN ceiling: qap is ~not kNN-separable from bulk chemistry; the latent is already at the data limit

**Q.** Before building any new SSL pretext for #15, what is the prize? The
canonical encoder reads qap at 0.769 (MLP probe) but raw-latent kNN qap purity is
only 0.582 (k=5). Is that 0.58 a fixable representation gap, or a property of the
data?

**Context.** The 06-24 geometry audit established 0.58 is stable across encoders
and epoch counts (not under-training) and read it as the masked-LM "anisotropic
embedding" symptom -- a training-objective limitation a new pretext could fix.
That read was untested against the one obvious baseline: how separable is qap by
the raw chemistry *at all*. Some qap splits are textural (basalt vs gabbro are
the same chemistry, differing only in cooling rate the encoder never sees), so
there is an unknown ceiling below 1.0 that no encoder can pass.

**Method.** Ran the identical kNN purity on the raw 22-element input instead of
the latent (geometry analogue of the imputation raw-input XGBoost ceiling),
reproducing the encoder's exact test split. Two raw representations (raw frac;
per-element standardized log-frac), both metrics, qap + supergroup. Exploration:
`explorations/input_knn_ceiling/` (committed result + repro). The latent number
reproduced 0.582 exactly, confirming the split was rebuilt correctly.

**Result (k=5, 33371-row test).**

| label | latent (cos) | best raw-input | headroom |
|---|---|---|---|
| qap_name | 0.582 | 0.607 (raw frac) | +0.026 |
| supergroup | 0.670 | 0.676 (raw frac) | +0.006 |

Raw-input qap kNN tops out at ~0.61; the latent at 0.582 is ~96% of that. Two
secondary findings: (1) the standardized-log space is *worse* (qap 0.527) -- equal
element weighting dilutes qap, which is major-driven; raw-frac (major-dominated)
is the better ceiling, and the latent's slight shortfall vs raw is the encoder
weighting traces more (it was trained to reconstruct them). (2) The decode-vs-kNN
gap (classifier 0.77 vs kNN 0.61) exists in the *raw input* too, so it is intrinsic
to the task, not an anisotropy artifact the SSL objective introduced -- this
partly overturns the 06-24 anisotropy attribution.

**Considered.** A learned representation is NOT hard-bounded by a fixed-metric
raw-input kNN -- metric learning routinely beats raw-feature distance (face
embeddings vs pixels). But our encoder is reconstruction-trained, which *preserves*
input geometry rather than reshaping it, so it sits at the input ceiling as
expected. The only family that could exceed a fixed-metric input ceiling is
contrastive / pull-together, and only if its positive pairs carry a target the raw
chemistry lacks. For qap specifically there is no natural "same-qap" augmentation
short of labels, so even pull-together is unlikely to break this particular ceiling.

**Decision.** qap-kNN purity is a near-maxed data ceiling, not a representation
gap. Reconstruction-family pretexts (group-masking, #3) are bounded near it --
do not chase qap-kNN with them. The honest reframing of #15: the question is not
"can a better pretext segregate qap" (chemistry caps that at ~0.61) but "is there
a target worth injecting -- tectonic setting, FC trajectory -- where a
metric-learning objective lets the latent exceed what raw chemistry alone
supports." Tectonic setting is in `data/grains/mineralml.csv` only (per-grain,
majors-only, dirty labels, eval-only policy), so it is not a quick win; the
FC-trajectory target (Track B #9) is the clean non-derivable path. No production
code changed; the diagnostic lives in `explorations/input_knn_ceiling/` and
generalizes (point it at any future target to screen its raw-input ceiling first).

**Next.** Take the input-kNN ceiling finding back to #15 to re-scope the epic away
from qap-kNN; the FC-corpus half of Track B (#9) is independent of the encoder
geometry and can proceed in parallel.

## 2026-06-25 -- latent anisotropy measured head-on: encoder removes the input cone, latent uses ~12.5% of its dim budget

**Q.** `input_knn_ceiling` argued the weak cosine-kNN purity is "not an anisotropy
artifact the SSL objective introduced," but only indirectly (the same gap exists
in the raw input). What does the canonical latent's anisotropy actually measure,
in both senses?

**Context.** Anisotropy = departure from isotropy, and there are two independent
ones, conflated in the prior journal prose. First-order (mean / cone): a shared
dominant direction, seen by cosine, removed by centering. Second-order (shape):
an elongated covariance ellipsoid, computed on centered data, summarized by
effective rank. The existing `input_ceiling.json` already showed latent cosine-kNN
>= euclidean-kNN, and euclidean is mean-invariant -- so first-order anisotropy was
already implicated as *not* the kNN bottleneck, but nothing had been measured
directly.

**Result (33371-row test split, `pretrain_v3_logmse_std_seed0`).**

| facet | latent | raw_frac | std_log | isotropic ref |
|---|---|---|---|---|
| first-order pair cosine | 0.259 | 0.938 | 0.022 | 0.000 |
| ... centered | 0.003 | 0.013 | 0.021 | -0.000 |
| second-order eff. rank / d | 32/256 (0.125) | 2.8/22 (0.126) | 9.1/22 (0.415) | 255/256 (0.996) |
| top-10 var fraction | 0.653 | 1.000 | 0.886 | 0.046 |

First-order: the encoder *removes* the compositional cone rather than adding one.
Raw composition is almost degenerate (pair cosine 0.94 -- positive orthant); the
latent's residual cone is ~4x milder (0.26) and vanishes under centering (0.003).
Combined with cosine-kNN >= euclidean-kNN, this residual cone is demonstrably not
the kNN bottleneck. The "not an artifact" claim now rests on a direct number.

Second-order: this is where the latent is genuinely anisotropic but not collapsed.
~32 of 256 directions carry the variance (top-10 hold 65%). Eff. rank 32 >> 1, so
no catastrophic dimensional collapse, but far from isotropic (255/256). In
fractional terms it concentrates about as tightly as `raw_frac` (0.125 vs 0.126);
in absolute terms it spreads chemistry over more directions than either input.

**Decision.** Recorded baseline, no action. The SSL objective de-anisotropizes the
compositional input rather than introducing a pathological geometry; the residual
first-order cone is harmless to kNN. If #15 ever pursues an anti-collapse
(VICReg-style variance/covariance) objective, eff. rank 32/256 is the
before-number it would aim to raise. Backing exhibit + repro in
`explorations/latent_anisotropy/`.

**Next.** None required; this closes the anisotropy question the 06-24 audit and
06-25 input-kNN entry left as inference.

**Cross-cutting verdict (anisotropy + imputation + kNN-ceiling + probe).** Across
four orthogonal probes the encoder reads as strong: it sits at ~95-96% of the
raw-input ceiling on imputation (trace log-R2 0.836 / 0.877) and qap-kNN (0.582 /
0.607), decodes qap at 0.769, and is geometrically healthy (mild first-order cone,
no catastrophic collapse). The earlier worries -- smeared UMAP, low kNN -- are
properties of the task (qap not kNN-separable; chemistry is a continuum), not
encoder defects, and are NOT a blocker for proceeding to Track B (the heads, where
the real remaining risk lives -- whether ~32 effective dims support a multimodal
posterior). The one honest caveat: probe (0.77) > kNN (0.58) plus the variance
concentration means the latent is strong as a *decodable* representation but its
*geometry-native* uses (retrieval, clustering, distance-OOD) under-weight the
low-variance directions where some signal lives. Three follow-ups spun out as
issues #16 (post-hoc latent distance metric -- recover the probe-vs-kNN gap without
re-pretraining; the probe is a learned metric in disguise), #17 (map top latent
axes to named geochemical trends -- cheap positive validation of the
petrogenetic-structure claim), #18 (OOD detection + transfer to real-rock corpora
-- the foundation-model claim is untested out-of-distribution). #16 and #18 share
the Mahalanobis/whitening machinery.

## 2026-06-29 -- commit to the log-space SSL objective: drop linear + the config-loss harness

**Q.** ADR 007 made standardized log-space masked MSE the canonical SSL
objective but kept the linear-space loss and the `loss:`-selectable machinery
in place as a comparison harness. Should the codebase keep them?

**Context.** The linear bits were retained only to re-run the linear-vs-log
comparison someday. But the tagged history already preserves that: tag
`v3.1-log-imputation` (commit `6ba9695`, 2026-06-24) points at a pre-cleanup state
holding both objectives and all linear configs, so `git checkout
v3.1-log-imputation` restores the full harness. Keeping dead code in the live tree
to hedge a one-off reproduction is the smell. A one-item menu (`loss:` with a
single value) also can't name its item losslessly -- `log_mse` hides the
standardization, `std_log_mse` hides that it is per-element and masked -- so the
honest move is to name the objective in full in code + ADR, not compress it into
a config token.

**Considered.** (1) Keep the harness as ADR 007 literally said. (2) Keep the
seam but rename the token honestly. (3) Drop the token entirely, keep only the
`make_loss_fn` builder seam (the present-tense decoupling the train loop and
test eval rely on).

**Decision.** Option 3. Deleted `masked_element_mse`, the rejected plain-log
variant, `_loss_config`, and the `loss:` token + type-dispatch;
`compute_loss_stats` / `make_loss_fn` lost their `config` param and build the one
objective unconditionally. Standardization is unconditional everywhere -- the
`standardize` toggle is gone from the SSL path and from `LogImputationHead`
(`build_head` ignores a stale key on old checkpoints). Collapsed the misnamed
`eval/imputation.py` (linear scaffold) + `eval/imputation_logspace.py` (log
read-out) pair into one honest `eval/imputation.py` (`run_imputation`), and
repointed `scripts/eval/impute_elements.py` at a trained head checkpoint. ADR
007 amended to record the removal and cite the tag. `pretrain_production.yaml` is
now the seed-0 canonical config (HPs verified against the canonical checkpoint's
stamped config), no `loss:` block. The linear-architecture trap was avoided:
`LogImputationHead(hidden=None)` and the `FractionalEncoder` linear path are
untouched -- only linear-*space* magnitude weighting was deleted.

**Next.** Track B heads (ModeHead / CompHead / CondHead).

## 2026-06-30 -- Track B corpus scoping: equilibrium-on-volcanic, oxide labels, deferred-prior sampling

**Q.** Reverse-engineering from the desired output (a mass-balanced posterior over
mineralogy + endmembers + conditions per bulk): what do we generate with MAGEMin,
what labels do we store, how do we sample conditions, and what makes any of it more
than a fast forward MAGEMin emulator (Candioti 2026) or a from-scratch regressor?

**Context.** The v3 rewrite deleted the FC pipeline (recoverable from `2fd5559^`).
Recovered `run_magemin.jl` + its Project/Manifest to verify claims directly on real
gard volcanic bulks (Julia 1.12.6, MAGEMin_C 2.1.9). Two sub-agents ran in parallel
(condition sampling; MagmaDB analogs). The honest split that fell out: the *corpus*
(many conditions per bulk) is what makes the multimodal inverse posterior exist and
beats Candioti; the *latent* earns its keep only via things a from-scratch regressor
can't do -- OOD/abstention, imputation of partial input, few-shot transfer to real
labels, and grounding in real analogs. The scenario *format* is not itself
constitutively latent, and saying so is more defensible than pretending otherwise.

**Considered.** (1) FC vs equilibrium for the generator. (2) CompHead target:
per-phase endmember mole fractions (heads.md Option B) vs per-phase oxide wt%. (3)
Condition sampling: product grid vs Monte-Carlo-with-deferred-prior. (4) CondHead /
scenario weights: are they real or circular? (5) MagmaDB as a real-analog panel. (6)
Whether an FC leg leaves useful data on the table.

**Decision.**
- **Equilibrium, not FC, scoped to volcanic + has-iron bulks; cool to near-solidus
  and read the equilibrium assemblage.** A volcanic rock crystallizes closed-system
  in place, which *is* equilibrium; FC models open-system cumulate-shedding. This
  amends ADR 004 (which chose FC on a mixed set for modal realism) for the volcanic
  case -- the formal ADR amendment is still to be written. Single-point equilibrium
  also returns modes + `SS_vec.emFrac_wt` + `G_system` natively, sidestepping the
  MAGEMinCalc fork FC needs for per-step composition.
- **CompHead target = per-phase oxide wt% (`Comp_wt`) for all phases**, not the
  endmember vector. Recon on real volcanic bulks: `emFrac_wt` is a signed affine
  combination (sums to 1 but components go negative); amphibole per-mineral
  `sum|emFrac|` median 2.3, worst component -110%, so a Dirichlet/simplex head is
  infeasible for the phase that matters most. `Comp_wt` is non-negative, mass-
  balance-grade, free from the same call; mineral names come from stoichiometry
  classifiers on oxide/apfu. `emFrac_wt` still stored as diagnostic; endmember heads
  are an optional bonus on clean phases (fsp/ilm/ol/opx) via loss-masking. See
  [[comphead-parameterization-options]].
- **Conditions: Monte-Carlo, prior deferred to inference.** Sample (P,H2O) by Latin-
  Hypercube from a broad proposal `q=Uniform(P[0.5,12]kbar x H2O[0,8])`, store
  `(P,H2O,q)` per row, bake NO prior into the corpus; defer every prior to inference
  via importance weights `w=p/q` (change prior -> recompute targets -> retrain heads,
  no MAGEMin rerun). The grid-density artifact is real (P-dense vs H2O-dense grids
  disagree; MC matches balanced). T is derived (near-solidus), not sampled. Fe split
  class-conditional Middlemost, fO2 fixed.
- **Scenario weights are "prior-weighted plausibility," not nature's probability.**
  The circularity (weight = physics integrated against our condition prior) lives
  only in the prior factor; the assemblage<->condition coupling is real physics. De-
  circularize at inference by reweighting toward evidence-based priors -- Sr/Y +
  La/Yb (volcanic-calibrated but soft, single-sample-noisy, composition-gated),
  latent-neighbor analogs (shown as evidence, not reweighted), user-supplied
  constraints (#21). The `p/q` reweight literally divides out the corpus prior.
- **MagmaDB assessed and shelved.** Embeddable and coherent on the arc slice, but too
  sparse/narrow (78 distinct bulks, ~29% igneous coverage, arc basalt-andesite only)
  to carry the analog panel. Revisit only with a better reference set. Output-layer
  only; never blocked corpus regen. (`explorations/magmadb_analogs/`.)

**Standing risk (unresolved).** The equilibrium generator may NOT reproduce the
heads.md phase-identity multimodality. MC test (N=1, wet andesite 6875): amphibole on
every (P,H2O) path, opx absent -- closed-system keeps its water so the assemblage
converges; the amp-in/opx-in switch was an FC (melt-drying) phenomenon. If this holds
across compositions, the "top-K scenarios" headline shifts toward a "conditions +
proportions + minor-phase-presence" posterior. A composition-stratified small test
decides it (regen spec, now ADR 004, item 7). See [[equilibrium-multimodality-risk]].

**Next.** New agent owns items 2-6 (near-solidus rule, volcanic subset+QC, productionize
the extractor, phase vocab, output schema) + the small composition-stratified test,
per the corpus regen spec (retired into ADR 004). Owner-confirm items: near-solidus liq threshold, the
qap list, and the multimodality headline reframe. Deferred: plutonic/cumulate branch
via FC-of-parents (#22, high priority, starts after volcanic v1) -- cumulates are the
one real rock class equilibrium can't generate faithfully. Formal ADR 004 amendment
still to be written.

## 2026-07-14 -- endpoint + label granularity locked; corpus regen fully specced

**Q.** Finalize the synthetic-label design for the Track B heads: where on the
cooling ladder to read the assemblage, at what granularity, and everything else
gating corpus regen -- so a generation agent can run without design questions.

**Context.** The re-pick experiment had shown the amp/opx multimodality
survives at 50% melt and that near-solidus carries artifacts (phase-list +40%,
spurious muscovite, wet-attrition tail). The real-mineralogy EDA had shown
quartz / alkali feldspar / biotite are felsic-restricted phenocrysts (0%
occurrence in mafic rocks). The sharpest residual: do felsic bulks need a
DEEPER crystallinity stop to saturate those late-looking diagnostic phases --
i.e. a composition-conditional endpoint?

**Considered.** (1) Single vs composition-conditional endpoint. (2) Lumped vs
itemized groundmass. (3) Melt fraction as a secondary predicted diagnostic.
(4) du Bray visual modes for magnitude scoring. (5) Water bound: VolatileCalc
vs Moore 1998 vs an in-house envelope. (6) Scale: 10k vs 30k bulks vs
two-tier. (7) Presence cutoff 0.1 vs 1 wt%.

**Decision.** (Spec detail in the corpus regen spec, retired into ADR 004; evidence in
`explorations/volcanic_proxy_definition/felsic_saturation.py` + updated
`endpoint_repick.py`, both re-reading the stored 143k-point ladders.)

- **Composition-conditional endpoint REJECTED -- the hypothesis inverted.**
  Quartz saturates EARLY in felsic bulks (median first appearance at 75% melt
  remaining); at the 50%-melt rung, subalkaline felsic draws show quartz 0.95
  and K-feldspar 0.50 vs real du Bray 0.53 / 0.49. Real felsic lavas are the
  crystal-poorest (~10-30 vol% phenocrysts), so a deeper felsic stop moves
  AWAY from the real reference state. Near-solidus invents felsic phenocrysts
  in mafic bulks (quartz 61%, biotite 91%, K-fsp 5% vs 0/0/0 real) -- a
  categorical sim-real failure no prior reweighting fixes, and the strongest
  quantitative argument for the phenocryst regime. Biotite is the one
  counter-pull (under-shoots real at every phenocryst-regime rung, matches
  only near-solidus); it is water-hungry and confounded with the flat H2O box
  -- the SENTINEL PHASE for the post-water-redesign re-measurement.
- **Endpoint: stop at melt < 30%; label rungs 60/50/40/30 re-picked from one
  ladder; single composition-independent rung picked by label-level pre-screen
  (presence-per-band first, rank proportions second); heads trained on top
  1-2 rungs only.** The amp/opx switch is endpoint-insensitive (51-60% of
  bulks at every rung 70->1% melt), so multimodality constrains nothing here.
- **Labels crystals-renormalized; melt-fraction head DROPPED as unbuildable:**
  at a fixed melt-fraction endpoint the label is a constant (attrition 0.7%
  at the 50 rung) -- nothing to learn. Consequence for heads work: training
  teacher-forces the label melt fraction in the mass-balance closure; the
  inference-time closure residual as an OOD signal degrades (open, heads-side).
- **Water: N&L 2002 rhyolite-curve solubility bound capped at 8 wt%.** Owner
  call: well-established literature over in-house envelopes. The extrapolation
  objection dissolves -- the 8 cap (melt-inclusion supply ceiling, Wallace
  2005 / Plank 2013) binds above ~4-5 kbar, exactly where VolatileCalc's
  calibration ends, so the curve is never evaluated out of range. Accepted
  gap: low-P alkaline melts dissolve somewhat more; documented, not patched.
- **Scale two-tier: 10k bulks x 48 draws (production) + ~500 x 128 (val/test
  precision tier).** Training tolerates noisy presence targets (SE 0.07);
  calibration eval and the rung ablation need SE 0.04. ~13M MAGEMin points,
  ~2 days accumulated laptop runtime, chunked + resumable (atomic per-chunk
  parquet + manifest; resume = rerun, at most one chunk redone).
- Subset: total-Fe sufficient (Middlemost split is imposed regardless) with
  measured-Fe provenance tier; QC flags weight bulks out of the stratified
  draw rather than hard-drop. Presence cutoff starts at 0.1 wt% (raw modes
  stored; cutoff is a retrain, never a regen); pre-screen runs both cutoffs.

**Next.** Regen agent: bounded sampler -> 36-bulk water A/B (isolate the water
fix; watch biotite) -> ~300-bulk pilot -> pre-screen picks the rung ->
production. heads.md still describes the liq/fl-bearing ModeHead vocabulary
and worked example -- update when heads work starts, not a regen blocker.

## 2026-07-14 -- regen machinery landed; water A/B: switch holds, biotite gap is prior-side

**Q.** Implement the locked regen spec (bounded sampler, chunked orchestrator,
pilot/pre-screen tooling) and run the water A/B: does the amp/opx
multimodality survive the solubility-bounded proposal, and does biotite move
toward real arc frequencies?

**Context.** All design locked earlier today (previous entry). Biotite was the
sentinel: under the flat box it under-shot real presence at every
phenocryst-regime rung.

**Considered.** (1) Curve-fit the N&L figure vs port the model: ported the
VolatileCalc pure-H2O rhyolite branch exactly (the figure is an output of the
code, and `q` should be the model's value, not a digitization's); validated by
round-trip through the third-party VolatileCalc implementation at 14 pressures
(<0.05%), six values pinned as a regression test. (2) Adaptive per-T-step
cooling vs one fixed-ladder batch per chunk: measured on the stored ladders,
the ~29% of points below the 30%-melt stop cost about what ~31 extra Julia
startups per chunk would -- a wash, so one batch per chunk and truncate at
parse time. (3) `q` construction: uniform in P, uniform in H2O given P, so LHS
stratification and the closed-form density hold exactly (q varies with P;
inference divides by it).

**Decision / result.** A/B on the same 36 bulks, N=128, paired seeds (P draws
identical; only the 22.3% of draws below 3.05 kbar changed, mean H2O 4.1 ->
2.8 wt%): the amp/opx switch holds at 0.57-0.60 of bulks at every rung in both
arms (intermediates 0.636, unchanged) -- the heads.md headline is intact.
Biotite did NOT move (+/-0.02 everywhere): the bound removes wet-shallow
draws, but the gap comes from dry draws the prior-agnostic proposal keeps by
design. Verdict: the biotite gap is an inference-prior question (Wallace/Plank
via p/q, now well-defined -- q went from 1 unique value to 1030), not corpus
coverage. Training-side wrinkle for the rung pick: intermediate-band biotite
has ZERO positives at rungs 60/50, matches real exactly (0.40) at rung 30,
while quartz/K-feldspar match best at 50-60 -- expect the pre-screen to argue
for two training rungs. Owner enlarged the pilot from ~300 to ~500 bulks
(first 5 chunks of the shuffled train table). MAGEMin output gotcha recorded
in the corpus module: total Fe as FeO + excess O, no Fe2O3 column.
Evidence: `equilibrium_multimodality/results/water_ab_{switch,presence}.csv`.

**Next.** Pilot (first 5 chunks, `--max-chunks 5`) -> health report -> rung
pre-screen -> owner picks rung(s) -> production (10k x 48 train + 500 x 128
val/test).

## 2026-07-14 -- explorations/ cleanup before corpus EDA

**Q.** Pre-EDA housekeeping: explorations/ had drifted from its own rules
(index listed 3 of 10 dirs; a production script imported exploration code;
one exploration had no README/test).

**Decision.** (1) `build_subset` + `VOLCANIC_QAP`/`ALKALINE_QAP` +
`total_fe_as_feo` promoted from `equilibrium_multimodality/selection.py` to
`cnmineral.data.volcanic_subset` -- production corpus tables now import from
src, and the exploration imports back the right way round (`selection.py`
keeps only the stratified test-bulk picking). Fe2O3->FeO factor deduplicated
against `magemin_oxides.FE2O3_TO_FEO_MASS`. Verified: 335,042 bulks, 25
classes, unchanged. (2) `lepr_validation` brought to compliance: README
(2,368 experiment pairs with measured (P, T); presence-only; no use decided
yet -- complements, not duplicates, `real_mineralogy`), `__init__.py`,
tripwire tests. (3) Index rewritten to cover all 10 explorations.
`equilibrium_multimodality`'s 533 MB is the gitignored `work/` ladders that
`volcanic_proxy_definition` re-reads -- kept. `imputation_head_design` is
results+README by documented intent (code retired) -- kept as is.

**Next.** Corpus EDA (health-report `--figures` + pre-screen plots), rung
pick, production launch.

## 2026-07-14 -- real-benchmark fine-tuning stance: zero-shot first, benchmark stays eval-only

**Q.** The pre-screen shows real label-level disagreement (presence MAE ~0.19,
concentrated in biotite/quartz/amp). Should the heads be fine-tuned on the
real volcanic benchmark to close it?

**Context.** The harvest-time README said "never train on this," but that was
never owner-ratified. The architecture explicitly supports re-supervising
heads on real labels, so the question is legitimate.

**Considered.** (1) Fine-tune on a benchmark split: costs power on an
already-small instrument (730 samples, 13 with quantitative modes).
(2) Fine-tune on LEPR instead: REJECTED -- structural mismatch. LEPR has no
modal proportions (nothing for ModeHead's primary target), no melt fraction
(experiments can't be placed on the rung axis, so labels would smear the
endpoint definition), and its (P, T) means "imposed lab conditions," not
"phenocryst-regime scenario conditions." Being equilibrium-at-known-conditions,
LEPR is structurally closer to MAGEMin than to natural rocks -- its honest
role is presence-level generator validation, not head supervision.
(3) Zero-shot first: ship sim-trained heads, measure the sim-to-real gap on
the intact benchmark, decide fine-tuning afterward with a number in hand.

**Decision.** Option 3 (owner). The benchmark stays eval-only *until the
zero-shot gap is measured*; the fine-tuning decision then reopens with "on
what data" deliberately unresolved (benchmark split, or a future itemized-mode
harvest, e.g. Rietveld XRD). The pre-screen's label-level agreement is the
zero-shot ceiling and the attribution tool: head errors decompose into
corpus-level gaps (pre-screen shows them), prior-level gaps (fixable by p/q
reweighting at inference), and model-level gaps (only the part below the
pre-screen ceiling is the head's fault).

**Next.** Rung pick (owner leaning 50+30), production regen launch. The rung
choice gates head training only -- all four rungs are stored per draw, so
generation is rung-agnostic and can launch first.

## 2026-07-14 -- rung pick: train on rungs 50 + 30

**Q.** Which of the four stored rungs (60/50/40/30% residual melt) supply the
training labels?

**Context.** Pre-screen on the passed 500-bulk pilot: rung 50 wins both
aggregate scores at both cutoffs (presence MAE 0.193/0.218, rank Spearman
0.79); rungs 40/30 clearly worse in aggregate. But intermediate-band biotite
(real 0.40 presence) is 0.07/0.15 at rungs 60/50 and reaches real only at 30,
while quartz/K-feldspar overshoot grows monotonically with depth. The
benchmark's mafic band rests on very few real samples, so the pick weighted
the intermediate and felsic bands.

**Decision (owner).** Two rungs: **50 + 30**. 50 as the primary label (best
overall presence and proportion match); 30 specifically so biotite exists in
the label distribution for intermediates -- training on 50 alone would make
biotite in andesites unlearnable (zero positives). Both rungs come from the
same stored ladders; this is a label-rebuild choice, not a generation choice.

**Next.** Production launch (rung-agnostic, all four rungs stored): train tier
10k x 48, then valtest 500 x 128. Heads-side consequence to carry: the model
sees the same bulk at two label depths (e.g. amphibole median share 0.22 at
rung 50 vs 0.32 at 30) -- accepted consciously; rung is implicit in the label
distribution.

## 2026-07-15 -- Track B start: four head decisions locked, hurdle ModeHead lands end-to-end

**Q.** Heads work starts on the ~45% partial corpus (owner directive:
ModeHead only, end-to-end, before any other head). Four design questions had
to be locked first: how the scenario view gets assemblage-conditions
coupling, the phase vocabulary, instance handling, and the CompHead
parameterization.

**Context.** Full-corpus coverage analysis (44 chunks, ~210k draws per
training rung) confirmed the pilot picture: 24 solid phases; chl/trd/crst/
fper at ~1-2 in 10,000 draws (unlearnable); trd/crst are SiO2 polymorphs
carrying a median ~half the crystal cargo when present; cor just above a
0.1% floor at both rungs. MAGEMin's lumped `fsp` emits two coexisting
feldspars (plag + alkali) in 13% (rung 50) / 19% (rung 30) of
feldspar-bearing draws; cpx/mu/spl multi-instances all <1%. Also verified:
the per-draw renormalized `in_*` oxides encode the sampled H2O through their
common scale factor, so head inputs must come from the original gard bulk
row (ADR 008 carry-over); relaxing `require_all_majors` recovers 4497/4500
corpus bulks (3 stragglers dropped with a logged count, incl. poison bulk
976755).

**Considered.** Coupling: independent heads (assumes a conditional
independence the corpus violates by design -- marginals forget which
condition mode pairs with which support), chain-rule support-conditioning,
shared discrete mixture (heavier redesign). Vocabulary: strict floor
without merges (renormalizes trd/crst's large mass into wrong phases),
keep-all-24 (dead output slots). Instances: lump feldspar for v1 (bakes the
inside-the-solvus comp smear into 1-in-6 fsp draws and forfeits the
QAPF-defining A/P ratio), full classifier vendoring (blocks on sub-1%
phases that don't earn it). CompHead: heteroscedastic per-oxide Gaussian
(exact error bars, but samples violate the simplex -- an uninformative
artifact needing post-hoc repair wherever composition vectors are consumed).

**Decision (owner).** (1) CondHead conditions on the presence support:
p(support | h) x p(T,P,H2O | h, support), teacher-forced true mask at train,
ModeHead-sampled at inference; mild exposure bias accepted. (2) Vocabulary
21 solids: trd/crst -> q; chl + fper dropped; and/ky/sill + cor kept
(polymorph identity encodes pressure for the support-conditioned CondHead);
fsp split into plag/afs. (3) Feldspar routed by the vendored mineralML
stoichiometric rule (DHZ boundaries; miscibility-gap compositions by
dominant component); all other phases sum instance modes. (4) CompHead:
per-phase Dirichlet over oxide wt%, eps-clipped targets.

What landed (ModeHead end-to-end): `phase_vocab` + `feldspar_rule` (100%
agreement with mineralML on all 17,776 non-gap chunk-0 feldspars) +
`mineralogy_labels` + `mineralogy_dataset` (inputs via
`BulkChemistryDataset(require_all_majors=False)`; hash-stable provisional
bulk split, to be discarded when the valtest tier lands) + `ModeHead` +
`mode_loss` (masked Dirichlet NLL verified against `torch.distributions`) +
`training/mode_head.py` with explicit joint-hurdle-NLL eval and per-phase
calibration table. All partial-corpus numbers are provisional scaffolding.

**Next.** CompHead (Dirichlet over `Comp_wt`, plag/afs targets clean by
construction), CondHead + support conditioning, mass-balance term (needs
CompHead), joint loss composition. Open rethink carried: inference-time
closure-residual-as-OOD degrades under teacher-forced melt fractions.

## 2026-07-16 -- ModeHead interrogation harness + first read; head reporting system; split mapping locked

**Q.** What does the smoke ModeHead actually do (presence calibration, the
multimodal-signal claim, magnitude honesty, guardrail mass), and how should
every future head be reported so orientation doesn't restart each session?

**Context.** Handoff step 1: interrogate before iterating. 47/100 chunks,
4,670 resolvable bulks, provisional hash split (~930 val bulks, 86.6k val
rows). Owner spent a long review session interrogating the semantics
(prediction-vs-label counting, BCE mechanics, bin definitions) -- those
answers were folded into the deck as they were settled.

**Considered.** Notebook vs eval-module for the interrogation (hybrid
module+notebook proposed; owner chose module + thin script per repo
conventions). Per-head vs per-metric-family eval slides (owner: per head,
maximum detail).

**Decision.** Landed `cnmineral/eval/mode_head.py` + `scripts/eval/mode_head.py`
(+10 tests): reliability/ECE, p-vs-within-bulk-f, stratified joint hurdle NLL,
magnitude parity, Beta-marginal PIT, alpha concentration, FP/FN *mass*, banded
recall, rung-50-vs-30; artifacts to `results/mode_head/<tag>/`. First read
(all provisional): presence side healthy -- ECE <= 0.033 every phase, p tracks
each bulk's presence frequency (amp r 0.93; the multimodal claim measured,
not assumed). Magnitude side is the weak half -- amp conditional MAE 0.14
with U-shaped PIT (overconfident; truth plausibly bimodal across paths),
oxide accessories humped (underconfident), plag systematically under-sized.
Garnet presence near-blind (recall 0 even at within-bulk f > 0.9). FP-mass
mean 0.15 (half amphibole) is substantially the intrinsic cost of collapsing
a multimodal posterior to one hard support, not pure head error. Joint NLL is
not comparable across support sizes (Dirichlet density scaling) -- rung strata
differences are mostly bookkeeping. Deck: standard two-slide-per-head report
locked (8-item checklist in `slides/overview.py` docstring, rule added to
CLAUDE.md), plus a one-rock-96-labels fan-out slide and a reliability
worked-example slide; loss weight renamed lambda_mag on slides (config key
stays `magnitude_weight`). Split mapping locked (owner): all 10k train-tier
bulks train; the 500 valtest bulks divide 16:20 -> ~222 val / ~278 test,
stratified by qap_name (CLAUDE.md settled decisions updated).

**Next.** Step-2 iteration scouting (retrains ~1 min): hidden-width sweep +
no-trunk linear baseline + magnitude_weight; plain-Dirichlet ablation
baseline (heads.md open item). Candidate design change if the amp PIT U-shape
survives tuning: support-conditioned magnitude branch (mirrors the locked
CondHead pattern; owner sign-off required). Candidate concerns named, none
triaged (owner's ledger): (a) FP-mass-at-0.5 conflates head error with
hard-view cost; (b) amp conditional bimodality vs a single per-bulk Dirichlet;
(c) NLL support-size comparability caveat on strata; (d) eval imports
`_bulk_latents`/`_row_split_indices` privately from the training module;
(e) split seed/val_fraction not stamped in `ModeHeadCheckpoint`.

## 2026-07-17 -- Dirichlet-branch orientation: parity read, slide corrections, bimodality work queued

**Q.** Figure-review phase 2: what does the magnitude (Dirichlet) branch
actually claim, is the worked-example slide telling that story truthfully,
and where does the amphibole failure come from mechanically?

**Context.** Owner orientation session on `magnitude_parity.png` + the
Dirichlet worked-example slide, before PIT. Numbers drifted mid-session as
train-tier chunks landed (val grew ~930 -> 1,614 bulks); all provisional.

**Considered.** For the parity regime panels: ranked all 21 phases on MAE,
r, bias, and spread compression (pred std / true std) rather than eyeballing.
For capturing within-bulk bimodality: support-conditioned magnitude branch
vs mixture-of-Dirichlets vs condition-on-conditions chain vs flows/diffusion
(last rejected: forfeits closed-form NLL/PIT and the microsecond-inference
pitch for a 21-dim problem).

**Decision.** `plot_magnitude_parity_subset` added to the harness -- THREE
regimes (owner relaxed the 2x2 protocol to "up to four, only as many as the
figure defends"): alkali feldspar = healthy (r 0.89); amphibole =
regression-to-the-middle of a two-humped truth (pred std 0.13 vs true 0.23);
spinel = MAE-flattered accessory (bias ~80% of median magnitude).
Plagioclase dropped: on parity it is the same regime as afs -- its small
systematic under-sizing is invisible in a hexbin and belongs to the PIT
figure. Both parity plots now shade +/-1 within-bulk sd around the diagonal, with
the width binned by magnitude level (`within_bulk_sigma_profile`; owner
observed a flat band overstates the floor at low magnitudes -- confirmed,
afs sd 0.058 at level ~0.1 vs 0.165 at ~0.66; the pooled scalar
`within_bulk_conditional_sigma` stays as the `within_bulk_sd` per-phase
metric and subset-title stat): the vertical scatter a bulk-level predictor
cannot remove. Amp's pooled band is 0.19 (81% of its total sd is
within-bulk); notably amp's measured r 0.68
EXCEEDS the bulk-mean-predictor ceiling (~0.59) because the teacher-forced
true support leaks path information into the sliced mean -- direct evidence
the support carries magnitude signal, strengthening the support-conditioned
branch candidate (#25). Plag under-sizing challenged by owner and re-verified
the valid way -- binned by *prediction* (truth exceeds prediction in 9/10
deciles, +0.041 top decile; 75% of bulks negative; median bias -0.028) --
after retracting the truth-binned visual argument (regression-to-the-mean
artifact, not evidence). Dirichlet worked-example slide redrawn at owner
request: two tracks converging at a scoring node (truth contributes support
only as a grading mask; modes meet prediction at scoring), plus an expanded
masked-NLL box (normalizer-vs-shape-term reading, definitions, worked
numbers in notes). Mechanism settled in discussion: presence stays honest
under path-ignorance because Bernoullis are closed under mixtures; a single
Dirichlet is a one-hump family, so the bulk-level optimum is
middle-mean/compressed-spread -- and one shared alpha0 per row couples all
phases' spreads. Filed #25 (P1: amp multimodality + the tuning/architecture
round, support-conditioned branch as sign-off-gated candidate) and #26 (P2:
full-simplex-mean vs Bernoulli consistency probe; absent-phase alphas get
zero gradient, so it measures trunk leakage, not correctness).

**Next.** Finish the PIT walkthrough (subset landed: quartz flat / amp
U-with-tilt / rutile hump; note the full-tier head flattened plag's PIT --
the "piled at 1" example migrated to afs, deck caption needs a pass once
valtest lands), then alpha_concentration and mass_guardrails; then plan-mode
design of the ModeHead-only Optuna scouting sweep (#25). Decision gate for the bimodality fix: stratify amp PIT by
support -- U-shape dissolving within fixed supports means support-conditioning
suffices, persisting means mixture components.

## 2026-07-17 -- Amphibole bimodality measured in the corpus: support-conditioning first, mixture second

**Q.** Is the within-bulk amphibole bimodality real in the conditional
(given-presence) proportion labels, and where do the humps live -- across
supports, across rungs, or within a fixed (support, rung) cell? The answer
picks the fix: support-conditioning can only dissolve humps that separate by
support; within-cell humps need explicit mixture components.

**Context.** Owner review of the Optuna scouting-sweep framing killed its
three questions: the no-trunk linear baseline is decision-free (no outcome
changes what we build next), hurdle-vs-plain-Dirichlet is deferred to
whatever tuning round exists later, and tuning-vs-structural for amphibole
was already answered by construction (many-to-one corpus is a settled
decision; the 51% amp<->opx switch confirmed it empirically 2026-06-30).
Order of operations locked this session: PIT-by-support diagnostic + this
data exam now -> valtest tier lands (4/5 chunks done) -> split swap +
figure/caption refresh -> CompHead/CondHead build (plan mode) -> joint
tuning per ADR 006 -> amphibole fix ladder measured against that baseline.
Rationale for heads-first: CondHead's locked chain-rule design builds the
support-conditioning machinery anyway, its GMM builds the mixture plumbing,
and the mass-balance term gives the fix ladder the right baseline to beat.
Owner ordered a direct corpus examination of the bimodality claim.

**Considered.** Three bimodality statistics. Split-based Ashman D from exact
1-D 2-means: rejected after exemplar review -- the optimal split gives D
~2.6 for a pure Gaussian, ~3.5 for a uniform, and 6.0 for a near-degenerate
point mass, so "98% bimodal at D>2" was measuring spread shape, not humps.
Silverman-bandwidth KDE mode counting: rejected in the opposite direction --
at n~68, sd~0.18 the bandwidth is ~0.08, which merges humps closer than
~0.16 and reported only 2% bimodal. Adopted: per-bulk 1-vs-2 component
Gaussian mixture picked by BIC, called bimodal only if both weights >= 0.15,
hump separation >= 0.05 proportion units, and separation >= 2x within-hump
spread. 2,000-bulk sample of the 7,480 train-tier bulks with >= 30
amp-present rows.

**Decision.** The bimodality is real and large: 45% of sampled bulks are
bimodal in conditional amp proportion (68% among amp-rich bulks, mean >
0.5); median hump separation 0.35. Where the humps live: support identity
explains a median 81% of within-bulk conditional amp variance (null
inflation for that group count is ~0.2), support x rung 86%, rung alone 11%;
conditioning on the (support, rung) cell drops typical residual spread from
~0.19 to ~0.07. But 34% of testable bimodal bulks remain bimodal within
their dominant support (27% within the dominant support x rung cell) --
bimodality a support-conditioned single Dirichlet cannot represent, sitting
where amp is abundant. Caveats: within-support tests only possible where the
dominant support has >= 20 rows (329/906 bimodal bulks; biased toward
support-concentrated bulks), and small cells underpower the mixture test, so
the irreducible rates are likelier under- than over-estimates. All numbers
provisional-tier. Consistent with the parity finding that teacher-forced
support leaks path information (amp r 0.68 > bulk-only ceiling ~0.59).
**Recorded expectation: a combination of fixes will be necessary, staged
one at a time -- support-conditioned magnitude branch first (it addresses
the ~80% of variance the support carries, and reuses CondHead machinery),
mixture-of-Dirichlets added only after support-conditioning's improvement
is measured in isolation, so each change's contribution is interrogated
separately.** Analysis scripts were scratch (not repo artifacts); method and
numbers recorded here and in #25.

**Next.** PIT-by-support diagnostic + per-bulk overlay on the current
checkpoint (eval-harness code, plan mode first); valtest lands -> delete
`provisional_bulk_split`, refresh figures and deck captions; then
CompHead/CondHead plan mode.

## 2026-07-18 -- Split swap: sealed valtest roster replaces the provisional split; first val-roster baseline

**Q.** Retire the provisional hash split and install the final two-tier
split: what carries the sealed val/test assignment, how do the degenerate
valtest bulks enter it, and what replaces "provisional" in reporting?

**Context.** Both tiers complete and audited (948k train rows; valtest
125,105 usable rows, 97.7%); bulk 411132 dropped by owner decision
2026-07-17, leaving 499 roster bulks. New facts surfaced at implementation:
7 roster bulks have zero usable rows at any rung (3 tephritic phonolite,
2 rhyolite, 1 phonolitic tephrite, 1 alkali feldspar trachyte), and 5 of
the 6 short bulks (<200 rows) are tephritic phonolite / phonolitic
tephrite -- solver attrition concentrates in the alkaline classes.
Stratification is otherwise clean: 22 qap classes, no singletons.

**Considered.** Roster membership for the odd bulks: assign all 499
(adopted -- a zero-row bulk never enters a dataset, so it contributes no
signal anywhere, and the roster stays stable if those bulks are ever
regenerated) vs dropping the 7 vs pinning them to val. Persistence:
committed CSV generated once (adopted -- the sealed test assignment must
not drift with sklearn versions, and the roster is reviewable in git) vs
seed-recompute at load. Reporting: factual caption "val (N bulks, 128
draws/bulk)" with N computed at runtime; metrics.json records the csv
path, subset, and counts -- no adjectives.

**Decision.** `scripts/build/build_valtest_split.py` (one-shot,
--force-guarded) wrote `src/cnmineral/data/valtest_split.csv`: 221 val /
278 test, stratified 16:20 by qap_name (seed 0), every class present in
both subsets. `provisional_bulk_split` deleted; the trainer now trains on
every train-tier bulk (9,933 after the chemistry join) and early-stops on
the val roster (218 effective bulks, 55,576 rows); the eval harness takes
only `--checkpoint` and rejects pre-roster checkpoints. Retrained head:
best val at epoch 3 (early stop at 8), joint hurdle NLL -1.0066 nats/row
on the val roster -- a new baseline series, not comparable to the
provisional-split -0.818. First val-roster read against the
provisional-era story: garnet's high-frequency miss is gone (recall 1.0
at f > 0.9, 0.0 below f = 0.5 -- misses are now purely
intermediate-frequency, alongside opx 0.23 / ru 0.13 / ol 0.04); plag's
PIT flattened (tails 0.12); amphibole keeps the U-with-high-tilt (tails
0.27, mean 0.58: spread too narrow and conditionally under-sized, the #25
target); oxide accessories stay humped (spl/ilm/ru tails ~0.01). Deck
updated with fixed-roster populations, the 256-rows/rock worked example,
and the regen-status slide closed out. Issues #29/#30 filed as
low-priority assessments (shared eval-figure helpers once Comp/Cond evals
exist; inspect_checkpoint doesn't know ModeHeadCheckpoint).

**Next.** CompHead/CondHead build (plan mode; two heads + joint loss +
mass-balance) -> joint tuning per ADR 006 -> staged amphibole fixes
(support-conditioning first, mixture second), each measured against this
baseline.

## 2026-07-18 -- CompHead + CondHead + joint trainer built (code complete; baseline run pending)

**Q.** Build the two remaining supervised heads and the joint loss with
the mass-balance closure term -- what shape does the joint stack take, and
what did the corpus reveal on the way?

**Context.** ModeHead-only baseline landed on the sealed val roster
(-1.0066 nats/row). Owner sign-off on two structural choices: the joint
trainer retrains all three heads together against the frozen encoder
(closure is differentiable in both modes and comps; ADR 006's single-stage
HPO assumes one objective), and the build ships end-to-end. The standalone
ModeHead trainer stays as the harness for isolated amp-fix measurements
(#25).

**Considered.** Comp parameterization: per-oxide alphas vs the locked
mean x shared-concentration factorization (adopted: softmax mean x one
softplus scalar per phase; the mean is invariant to the concentration).
Closure norm: L1 vs squared L2 (adopted L2 -- constraint penalty, not a
head likelihood, so the no-MSE rule doesn't apply). Cond conditioning
mechanism: concat(h, presence) into the trunk (adopted) vs FiLM (deferred,
no evidence it's needed). Checkpoint schema: extend ModeHeadCheckpoint vs
a new JointHeadCheckpoint (adopted new -- three state dicts + a loss
bundle are a different shape, not optional fields).

**Decision.** Landed: `models/comp_head.py` (65,404 params at d_model 256)
and `models/cond_head.py` (39,067; input concat(h, presence[21]), per-axis
K=3 GMM in raw units); losses `comp_loss` / `cond_loss` / `closure_loss` /
`joint_loss`; `load_joint_labels` + `JointHeadDataset` (comp targets with
mass-weighted instance merge, feldspar split before averaging; cond
targets from the rungs T_end + draws join; closure fixings with the
fe2o3 -> FeO-equivalent + excess-O basis conversion, verified against the
corpus at 9.5e-5 max residual on well-formed rows); the joint trainer with
JointHeadCheckpoint; comp + cond eval harnesses (incl. the teacher-forced
vs ModeHead-sampled exposure-bias measurement); 47 new tests, full suite
286 green; anatomy slides for both heads in the deck (exact param counts).
Two training fixes forced by raw-unit cond targets: `init_output_bias`
seeds the mixture at the targets' mean/std (an unseeded head starts ~1,300
sigmas from typical T and Adam needs ~mu/lr steps to close), and gradient
clipping is per-head so CondHead's early gradients can't starve the
others. Surprises: (1) ~0.3% of corpus rows do not self-close in MAGEMin's
own output (multi-instance melt immiscibility; dropped-phase mass), tail
bounded at 5.24 wt%, 99.7% of rows within 0.1 wt% -- owner reviewed the
distribution and declined to act; the real-corpus closure test asserts
quantile thresholds. (2) lambda_cond is nearly inert for optimization
(L_cond's gradient reaches only CondHead's own weights; frozen encoder +
separate trunks + teacher forcing sever every cross-head path, and Adam
absorbs constant rescaling); its one real lever is the val-total early
stop, where 0.01 down-weights condition progress 100x -- owner classed
this an observation/feature (mineralogy-first), not a bug. Exposure bias
filed as #31 with the mitigation ladder (observed-combinations filter
first, mask noising second, joint support model never-yet).

**Next.** Baseline joint training run (owner-executed) -> both eval
harnesses -> read exposure delta (#31), closure residuals, comp
concentration blur, GMM component structure -> eval-semantics deck slides
from the real figures + heads.md/roadmap status -> joint single-stage HPO
(ADR 006) -> staged amp fixes against the tuned baseline.

## 2026-07-18 -- Joint baseline readout: stopping arbitration, underconfident comp, over-eager accessories

**Q.** What did the first joint baseline run + both eval harnesses reveal?

**Context.** `joint_heads_seed0` (11 epochs, early stop kept epoch 6),
comp/cond harnesses run, orientation figures 1-7 in
`results/summary/joint_heads_seed0/`.

**Findings.** (1) **Heads disagree on stopping** -- quantified from the raw
monitor: the joint epoch-6 checkpoint cost mode 0.004 and comp 0.017 nats
(wobble scale) but cond >=0.057 *and its convergence* (still improving
~0.01 nats/epoch at cutoff). Comp's numerically largest term decides the
checkpoint. (2) **Closure** <1 wt%/oxide (accepted) but with a systematic
sign bias (SiO2 +0.9 / Al2O3 -0.6) numerically invisible to the penalty at
weight 1. (3) **Comp** accurate on majors (plag An# r=0.92; ilmenite
FeO/TiO2 the outlier) yet underconfident everywhere: alpha0 compressed to
~100-650, fixed-recipe phases get no tail mass -- re-measure after tuning
before questioning the locked shared-concentration design. (4) **Mode**
nails majors including amp's bimodal spread; accessories over-predicted in
presence and magnitude (spl 12%->44%; ilm ~8x presence on the basalt
case). (5) **Cond** teacher-forced well calibrated; sampled-support
coverage collapses 68%->30-36% (#31 now measured; demoted per
mineralogy-first).

**Next.** Plan the heads study order-of-operations before any code.

## 2026-07-19 -- Heads study locked: per-head selection, mode-primary HPO objective, 3-prime sweep design

**Q.** In what order do we tune/ablate the heads, and what objective does
the joint HPO (ADR 006) rank trials by?

**Context.** Baseline readout above. Candidate objectives: weighted total
(rejected: self-gaming -- the lambdas are search variables), mode-primary
with feasibility constraints, Pareto front, per-head checkpointing.

**Considered.** The curves show **no real mode<->comp tradeoff**: disjoint
parameter blocks on a frozen encoder + AdamW scale-invariance leave only
two couplings -- the closure term (~0.0015 at baseline weight) and
checkpoint selection -- and the selection artifact's victim was cond, not a
mode/comp fight. So fix the mechanism: **per-head best checkpoints** (each
head snapshotted at its own raw-val best; stop when every head is
`patience` past its own best), and the objective simplifies to the
**unweighted assemblage NLL** (bce + Dirichlet, ungameable by
`magnitude_weight`) at mode's best epoch, ties <=0.03 nats broken by comp
then cond. Architecture splits by **objective visibility**: mode's
width/depth joins the main sweep (7 dims: lr, wd, magnitude_weight,
lambda_balance across seven decades, dropout, mode_hidden, mode_layers);
comp/cond architecture is invisible to a mode objective and gets
conditional **mini-sweeps** against their own NLLs (adoption requires mode
inside the tie band via `mode_reference`). Shared-lr remains the one live
coupling -- audited by the per-trial mode-vs-comp scatter; per-head lr
param groups are the escape hatch. **Gate before adopting**: predicted-
modes closure, per-head-best pairing (rerun) vs the old epoch-6 checkpoint.

**Decision.** Locked and built: `layers` on all three heads, per-head
config keys, per-head selection trainer (`selection_series`),
`predicted_modes_closure` + `run_predicted_closure_eval`,
`joint_heads_hpo` harness (enqueued baseline trial 0, user-attr audit
data, `tiebreak_best` read-out, fANOVA + scatter figures) + config
(n_trials 150 ceiling, no pruner, seed 0 every trial).

**Next.** Epic + child issues; owner reruns the baseline config (doubles
as the gate input), then the main sweep, mini-sweeps, production readout.

## 2026-07-19 -- Cross-epoch closure gate: PASS; rerun corrects the comp-overfitting read

**Q.** Does pairing mode@its-best with comp@its-best (37 epochs apart)
degrade the predicted-modes closure diagnostic relative to the co-trained
epoch-6 pairing?

**Context.** Baseline rerun under per-head selection: mode -5.7523@ep2
(exact replication of the old trajectory -- the refactor changed no
training semantics), comp -50.7946@ep39, cond 8.4569@ep57 (ran the full
60-epoch ceiling, ~0.002 nats/epoch at the end). Correction to
2026-07-18: comp's "overfitting after epoch 7" was a local bump -- with
room to run it improved another 0.59 nats by ep39, so the joint epoch-6
checkpoint was costing comp ~0.6 nats, not 0.017.

**Decision.** PASS: cross-epoch l2 mean 0.0324 / p95 0.0699 vs co-trained
0.0329 / 0.0672 -- within a few percent, mean slightly better. Per-head
selection adopted. Note the signed per-oxide bias is state-dependent
(ep6 pairing: Al2O3 -0.010 dominant; cross-epoch: SiO2 -0.011), and both
differ from the teacher-forced instrument's stable +SiO2/-Al2O3 -- the
closure-penalty-form ablation should use the teacher-forced instrument.

**Next.** Main sweep overnight (expect trials to run the full 60-epoch
ceiling; cond rarely plateaus 10-in-a-row before 60); epic + child issues.

## 2026-07-19 -- Main sweep sealed at 42 trials; trial-21 tiebreak winner adopted; ceiling 120

**Q.** When to stop the main sweep, does mode_layers railing at 3 warrant
extending the range, and what gets adopted?

**Context.** 42 complete trials. Last material incumbent jump at trial 20
(-5.8562); the next 16 trials added 0.0015 (trial 36, -5.8577) --
fourth-decimal polish. 15 trials inside the 0.03 tie band, all
mode_hidden 256, 14/15 at mode_layers 3, dropout 0.32-0.36, lr ~2-4e-4.
Untuned -> tuned mode gain 0.105 nats; the epoch-2 stopping worry from
the baseline era is settled -- the old settings were simply wrong for mode.

**Considered.** Layers extension: best by layer count 1/2/3 = -5.8028 /
-5.8478 / -5.8577. The 2->3 step (0.010 nats) is a quarter of 1->2
(0.045) and already under the tie band; layers-3's 22-vs-8 trial-count
advantage inflates it if anything; and fANOVA puts both architecture
knobs at ~0 (mode_hidden 0.02, mode_layers <0.01). Rejected on those
three consistent lines. fANOVA's headline lambda_balance 0.59 is a
whole-space number: closure weights toward the 1e4 end wreck mode
training (the scatter's -4.8..-5.4 stragglers), while inside the basin
the tie band tolerates 2.9e-2..62 -- inert near the optimum, consistent
with the baseline "penalty invisible" read. Scatter: **no mode-comp
tradeoff** -- the best-comp corner sits inside the best-mode cluster, and
the entire shared-lr tax on comp is 0.044 nats (best-anywhere trial 11 vs
the winner), so per-head lr (#39) has no comp case left; cond is starved
in every tuned trial (8.55-8.66 vs untuned 8.457). The winner's comp/cond
best epochs are 58/60 -- pinned to the 60-epoch ceiling by the cool tuned
lr. Under per-head selection a taller ceiling is free for mode (its best
froze at ep30), so the choice was 120 vs 200: the lr-scaling hint
(baseline-lr cond converged ~ep57; tuned lr is ~4.7x cooler) says even
200 may rail at 1.7x the cost, and if cond rails at 120 the finding is
"per-head lr, not epochs" either way.

**Decision.** Sweep sealed at n_trials 42 (journal +
results/heads_hpo/joint_heads_hpo/ are the record). Tiebreak winner
**trial 21** (mode -5.8381@ep30, comp -53.796@ep58, cond 8.634@ep60,
predicted-modes closure 0.027): lr 2.15e-4, weight_decay 1.04e-5,
magnitude_weight 0.25, lambda_balance 54.5, dropout 0.348, mode 256x3.
Adopted **verbatim** into joint_heads.yaml -- seed 0 kept, so the
production retrain replicates the trial through ep60 and then lets
comp/cond continue. The Optuna incumbent (trial 36, -5.8577) ceded 0.020
in-band mode nats for 0.32 comp nats: the tiebreak working as designed.
**Epoch ceiling 60 -> 120** in the production config and both
mini-sweeps (owner call, over 200). Mini-sweep configs authored
(joint_heads_minisweep_{comp,cond}.yaml): search = that head's hidden x
layers ([64..512] x [1..3]), base = winner verbatim, mode_reference
-5.8381, trial 0 = the current 128x1; **n_trials 16**, not the planned
~30 -- the space is 12 discrete cells with a fixed seed, so extra trials
mostly re-measure cells. Sweeps run sequentially, not in parallel:
~57 GB resident each (mostly MPS-allocator hoard) on 128 GB, and the
shared GPU voids any wall-clock gain. #35 closed (manually -- Closes
keywords on a feature branch only fire on merge to main), #36
in-progress; comp sweep launched (~19 s/epoch, <=38 min/trial).

**Next.** Cond sweep after comp; each writes its summary + figures on
natural completion (after a Ctrl-C instead: lower n_trials to the
completed count and rerun --hpo -- the workaround #42 exists to remove).
Mini-sweep NLLs are **not comparable** to main-sweep numbers (120- vs
60-epoch ceiling): judge architecture against the mini-sweep's own
trial 0. Then #37: production retrain + full readout (alpha0
compression, accessory reliability, teacher-forced signed closure bias,
truth_vs_pred baseline-vs-tuned), deck refresh from the new checkpoint,
heads.md/roadmap/journal. #39's remaining case is cond-only and demoted;
#38 ablates the closure-penalty form on the teacher-forced instrument.

## 2026-07-21 -- Architecture mini-sweeps sealed: comp keeps 128x1, cond adopts 512x2

**Q.** Do the comp/cond trunks (invisible to the main sweep's mode
objective) want different architecture, judged against their own NLLs
under the tie-band admissibility guard?

**Context.** Both sweeps ran the 12-cell grid (hidden {64,128,256,512} x
layers {1..3}) at the tuned environment, 120-epoch ceiling, sealed at 20
complete trials each (budget bumped 16 -> 20 mid-run; journal resume
counts only COMPLETE trials, so user-killed trials cost nothing across
relaunches). TPE re-measured the incumbent cells heavily -- and every
duplicate reproduced to 14 decimals: training is fully deterministic on
this stack at fixed seed, so within-config wobble is exactly zero.

**Considered.** Comp (9/12 cells): capacity monotonically HURTS above
128x1 -- 512x3 worst at -51.56 and early-stopping by ep13, and bigger
comp trunks drag mode ~0.010 nats through the closure coupling. The three
unmeasured cells sit on downhill trend lines. Trial 0's 0.053-nat margin
over 64x1 is within comp's seed-noise scale (~0.15, measured below), so
the honest read is "<=128 ties, >128 hurts beyond noise" -- either way the
incumbent stands. Cond (5/12 cells, winner re-measured 8x): opposite
story -- monotone in width (64s ~8.72 < 128x1 8.5999 < 256x3 8.3623 <
512x2 8.3110), confirming starvation rather than overfit; winner sits at
the measured grid's edge (512x3 was proposed twice, both user-killed;
accepted per mineralogy-first, cond is the bonus head). Bonus
measurement: cond-architecture swaps perturb mode/comp only through the
shared RNG stream (init draw order, dropout masks), giving a free
seed-noise estimate at fixed data -- mode ~0.021 nats, comp ~0.15. The
0.03 mode tie band is appropriately sized (noise eats two-thirds of it).

**Decision.** Comp keeps `comp_hidden null / comp_layers 1` (no diff).
Cond adopts `cond_hidden 512 / cond_layers 2` (0.289 nats, ~8x its noise
scale, beats even the untuned 8.457 every shared-lr trial had starved;
mode untouched at -5.83842 vs reference -5.83809). #36 closed; #39
(per-head lr) closed -- its only case was cond starvation, fixed by
capacity; cond's best epoch 101 < 120 also retires the
"rails at the ceiling" worry.

**Next.** #37 production retrain (expected to replicate the cond sweep's
trial-12 cell exactly, given determinism) + full readout.

## 2026-07-21 -- Production retrain + tuned-vs-baseline readout: two diseases cured, two confirmed structural

**Q.** Does the adopted config reproduce the sweep's numbers in
production, and which of the baseline's diagnosed pathologies did tuning
actually fix?

**Context.** Retrain of `joint_heads.yaml` (no --hpo) stopped at ep111
with per-head bests mode -5.8384@ep31 / comp -53.7574@ep93 / cond
8.3110@ep101 -- an exact replica of cond-sweep trial 12, the free
pipeline check. Baseline checkpoint preserved at
`checkpoints/joint_heads_seed0_baseline/` (with `heads_joint_ep6.pt`
gate provenance); baseline eval artifacts renamed to
`results/*/joint_heads_seed0_baseline/`. All evals rerun on the tuned
checkpoint; every number below is same-instrument, same-roster.

**Considered.** (1) Comp alpha0 compression: CURED -- baseline hedged
every phase into alpha0 100-650; tuned spans 25-32,000 and orders by
real mineralogy (q/ru/sph/sill sharp, spl/ilm/cpx soft); eval comp NLL
-50.25 -> -53.85, majors' PIT tails near nominal. (2) Closure signed
bias: CURED -- the stable +SiO2/-Al2O3 signature collapsed to
zero-centered per-oxide residuals, L2 mean 0.0181 -> 0.0134 (#38 closed
on this evidence). (3) Cond: teacher-forced honest (NLL 8.60 -> 8.32,
coverage_68 0.65-0.69 -> 0.67-0.72) but the exposure gap WIDENED (T
delta 2.91 -> 3.94, H2O 1.53 -> 2.40, P_kbar inf on some sampled-support
rows) -- sharper under truth = more wrong under mismatch; recorded on
#31, stays P3. (4) Mode (first full joint-era interrogation, see next
entry): hurdle NLL -0.996 -> -1.162; amp presence improved (ECE 0.029 ->
0.014, recall 0.868 -> 0.893) but amp magnitude UNMOVED (MAE 0.1357 ->
0.1353; 65% of its (rock, support) PIT groups still mean-shifted) --
the within-bulk bimodality is structural, exactly as staged (#25).
Accessories still over-predicted (zero recall on ep/mu/ky/cor/and with
leaked sample mass); garnet's high-frequency recall miss is back in the
joint model (0.0 at f > 0.9, n=156) -- watch during #25/#41.

**Decision.** Tuned model is production. Readout verdicts recorded;
issue audit closed #12/#29/#30/#36/#38/#39 (rationales on the issues),
commented #31. Deck refreshed: three anatomy slides at checkpoint-pulled
counts (mode 208,170 / comp 65,404 / cond 418,843 = 692,417 head params,
~21% of the 3.23M-param trainable encoder), ModeHead dials repointed to
joint_heads.yaml, comp+cond eval-semantics slides authored (their
two-slide reports are now complete).

**Next.** #41 (accessory hurdle recalibration -- post-hoc calibration on
the new reliability instrument before any structure), then #25 (amp
mixture magnitude branch). #37 + epic #32 close with this entry.

## 2026-07-21 -- Eval consolidation: one joint-checkpoint entry point; standalone ModeHead pipeline deleted

**Q.** The mode eval could only load the pre-joint `ModeHeadCheckpoint`
format, so no joint model ever had the full ModeHead interrogation --
bandaid an adapter, or consolidate?

**Context.** Owner call: no bandaid sprawl. Joint training is the only
production path; the standalone pipeline existed purely as history.

**Considered.** Adapter-only (two formats forever) and three-scripts +
shared loader (keeps triple setup cost: each eval script rebuilt the
dataset and re-ran the encoder) both rejected for the clean shape: one
entry point, engines as pure measurement functions.

**Decision.** Deleted: `training/mode_head.py` (ModeHeadCheckpoint,
standalone trainer), its script + `configs/heads/mode_head.yaml`,
`ModeHeadDataset`, `load_mode_labels`, the three per-head eval scripts,
`tests/training/test_mode_head_train.py`. Landed:
`cnmineral/eval/joint_heads.py` (`run_joint_eval` +
`val_dataset_and_latents`, one dataset rebuild + one encoder pass shared
across heads) behind `scripts/eval/joint_heads.py --checkpoint ...
[--heads mode,comp,cond]`; engines now take `(ckpt, dataset, latents,
*, device, tag)`; `bulk_latents` public in `eval/latent.py`;
`evaluate_mode_head` lives with its only caller in
`training/joint_heads.py`; metrics stamp `checkpoint_tag` instead of a
path. Label-semantics tests ported onto `load_joint_labels` (the
deleted loader was their oracle; testing an inlined copy would test
nothing). Full suite 296 green; refactored path reproduces pre-refactor
comp metrics bit-identically; mode interrogations backfilled for BOTH
joint checkpoints (`results/mode_head/joint_heads_seed0{_baseline,}/`).
`results/mode_head/mode_head_seed0/` stays as the standalone-era record,
no longer regenerable. #30 closed obsolete, #29 closed satisfied.

**Next.** None -- the instrument now exists; #41 consumes it first.

## 2026-07-21 -- Results restructure: tracked `results/` (curated) + untracked `runs/` (machine output)

**Q.** Where do the new cross-version presentation artifacts (model-
improvement ledger, frozen per-version headline reports) live, given
`results/` was gitignored scratch?

**Context.** Session locked four canonical headline figures per model
version (skill trajectory, mode truth-vs-pred, inference card, scenario
view); the trajectory needs numbers from every past version, but old
checkpoints and eval pipelines die (the standalone-ModeHead record is
already unregenerable), so its inputs cannot live in gitignored dirs.
Meanwhile `explorations/*/results/` were already tracked -- "results"
meant opposite things at different depths.

**Considered.** (a) keep untracked results/, add tracked `reports/`
(cheap, but two near-synonym siblings forever); (b) rename + tracked
results/ (one invariant repo-wide); (c) b plus a top-level `archive/`
tier for closed-era dirs (third category with a boundary to
re-adjudicate every time something closes -- deferred until a second
unregenerable artifact exists).

**Decision.** Option b. Untracked `results/` renamed `runs/` (gitignore
now anchors `/runs/`); everything instruments emit is unchanged in
layout, just re-rooted (the four runs-rooted `RESULTS_ROOT` constants,
renamed `RUNS_ROOT` to match -- explorations-rooted ones keep their
name -- plus script defaults, 6 HPO config `output_dir`s, deck notes,
live doc refs). New tracked `results/` holds
curated history: `ledger.yaml` (one entry per shipped model version;
schema lands with the trajectory work) + `reports/<version>/` frozen
headline figures. Repo invariant: results = tracked conclusions (top
level and in explorations), runs = untracked machine output.
`runs/mode_head/mode_head_seed0/` remains the unregenerable
standalone-era record (checkpoint on disk at
`checkpoints/mode_head_seed0/`, format readers deleted); flagged in
`results/README.md`, its numbers become durable via the ledger.

**Next.** Skill-score reference + ledger schema + inference-card and
scenario-view builders (this session), writing into the new layout.

## 2026-07-21 -- Headline reporting layer: ledger + trajectory + two per-model figures; results/ goes untracked with a role split

**Q.** What does "show the model is good and getting better" look like as
a small, repeatable artifact set -- and where does it live?

**Context.** 20+ diagnostic figures per version is an instrument panel,
not a report. Presentations need a fixed format an audience can re-read
across versions; iteration needs a pithy per-version summary the owner
can judge in a minute.

**Considered.** Skill scores over an input-blind climatology baseline
(built, validated, then REMOVED -- a constructed reference invites
"baseline shopping" suspicion and cancels out of every version-to-version
delta anyway; raw NLL says the same thing without a defendable choice).
Per-dimension NLL normalization for a shared cross-head axis (rejected:
mode's divisor is arbitrary within ~3x). Teacher-forced condition panels
in the card (rejected: the card is the product demo; every panel now
conditions on the model's own predictions). Tracked results/ (adopted in
the morning, REVERSED by evening: headline artifacts stay private;
tracking them bought durability at the cost of publishing dev history).

**Decision.** Reporting layer: `results/ledger.yaml` (append-only, one
entry per shipped version, metrics always re-measured on the current
instrument; three entries backfilled: standalone-modehead 07-15,
joint-baseline 07-18, joint-tuned 07-21) + `results/trajectory.png`
(per-head raw-NLL small multiples, y inverted) + per-model
`results/<checkpoint-tag>/` holding the inference card
(`scripts/view/inference_card.py`: hurdle split into is-it-there /
how-much-when-there panels, false positives rate-panel-only, no teacher
forcing anywhere; bulk = median per-bulk mode NLL, `--pick best` for the
internal showcase) and the scenario view (`scripts/view/scenario_view.py`:
truth-bimodality auto-pick, top-2 sampled assemblages + per-scenario
conditions). truth_vs_pred stays a runs/summary diagnostic. Repo rule:
results/ = headline summaries, runs/ = transient snapshots, both
untracked; explorations/*/results/ stay tracked. One-command ship
consolidation deferred to #44 until the manual flow has been exercised.

**Next.** #41 opens with a new observation from the median-NLL card:
ilmenite presence UNDER-predicted (truth 0.65 of draws vs 0.15 predicted)
-- accessory miscalibration may be two-sided, check the reliability
tables both ways.

## 2026-07-21 -- Instrument correction: pre-study baseline eval dirs scored the joint-ep6 checkpoint, not per-head bests

**Q.** Why did regenerating `runs/{comp,cond}_head/joint_heads_seed0_baseline/`
move the numbers (comp -50.2508 -> -50.9005, cond 8.6023 -> 8.4653)?

**Context.** The ledger backfill required current-instrument stamps; the
relic dirs carried rotted absolute-path stamps predating the eval
consolidation.

**Decision (root cause).** Not eval code (consolidation reproduced comp
bit-identically) and not the roster (both post-swap). The per-head best
checkpointing change landed after the baseline's original evals: the old
dirs scored comp/cond at the joint-best epoch 6, today's heads_best.pt
holds each head at its own best epoch. Both heads improve under their own
best -- exactly the direction expected. Consequence: the 07-21 readout's
tuned-vs-baseline deltas were inflated by a selection-rule mismatch
(cond gain is ~0.15 nats, not ~0.28; comp -50.90 -> -53.85; baseline
closure L2 0.0161, coverage_68 0.650-0.661). All conclusions survive
directionally. Ledger and all six runs/ metric dirs now carry
current-instrument, per-head-best numbers; the mode dirs were already
correct (regenerated during consolidation backfill).

**Next.** None -- recorded so nobody re-derives the discrepancy.

## 2026-07-22 -- Presence-fraction ceiling probe: the corpus is the ceiling; regen becomes the goalpost

**Q.** The head hedges mid-range on spl/ilm/ru/sph because it can't tell
which bulks are the high-presence ones (#41 reframe). Is the missing
discrimination available anywhere -- raw chemistry, the frozen latent --
or does it not exist in the corpus?

**Context.** Per-bulk presence fraction is deterministic in chemistry up
to binomial draw noise, so three predictors bound the failure:
the production ModeHead, XGBoost on the 22-token log chemistry (input
ceiling), and a 1-hidden MLP on the frozen pooled latent. All scored on
the val roster against `within_bulk_frequency` truth
(explorations/presence_fraction_ceiling, instrument verified by
reproducing the #41 headline MAEs exactly).

**Considered.** Post-hoc recalibration (dead by measurement: reliability
already on-diagonal, isotonic worsens held-out). Per-phase BCE weighting
/ head capacity (dead: latent probe ~ head). bulk_frac skip connection
(dead: latent probe ~ raw ceiling, encoder drops nothing).

**Decision.** Headline mean per-bulk MAE: head 0.106, raw-XGB 0.098,
latent-MLP 0.102, all inside the 0.02 tie band; binomial floors
0.016-0.024. Verdict: corpus is the ceiling, enrichment (more bulks AND
more draws/bulk) is the lever, stop architecture work on presence
discrimination. Two structural findings: (1) Cr is populated in 0% of
train inputs (`include_cr2o3=False`; gard2019 coverage ~2%) and the
MAGEMin inputs were effectively Cr-free too (2.1% nonzero, corr with
f_spl +0.02) -- consistently Cr-blind, so Cr is NOT the spl gap;
deferred until real-rock validation forces it. (2) The Fe3+/Fe2+ split
(measured or Middlemost-class-assumed) is visible to the generator but
absent from the input (total-Fe token only) -- the prime suspect for spl
staying worst (0.159) even at the raw ceiling. Feature importances:
ilm leans Ti (0.31, hypothesis confirmed); ru/sph lean Ca/alkalis (which
Ti phase wins is Ca-controlled); spl leans Mg/Fe (Cr-free spinel
physics). Direction locked with the owner: the next corpus regen is the
goalpost; #45 is the pre-regen decision ledger (redox #46, sizing via
#27, P/H2O proposal revisions, valtest instrument upgrade, rare-class
rescue, Cr deferral); #25 (amphibole) and #18 (real-rock transfer,
which also freezes the comparison instrument for current-vs-regen
corpus) run first.

**Next.** #25 amphibole magnitude. #46 redox measurements (provenance
split + `bulk_11` true-input ceiling) before the fO2 design decision.

## 2026-07-22 -- Support-conditioned magnitude branch (#25 stage 1): adopted

**Q.** Amphibole's conditional magnitude collapses to the mean (truth
bimodal by (P, H2O) path within a bulk; one bulk-level Dirichlet says 0.4
when the answer is 0.15-or-0.6). The 2026-07-17 measurement said support
identity carries ~81% of the within-bulk conditional variance. Does
feeding the support to the magnitude branch recover it?

**Context.** ModeHead's magnitude branch got its own trunk on
[h, support(21)], mirroring CondHead's locked chain rule (teacher-forced
true mask at train, ModeHead-sampled at inference); the presence trunk
stays support-blind by construction. Trial-21 config, seed, epochs
unchanged for isolation; new checkpoint dir
`joint_heads_supportmag_seed0` (no legacy loading code -- the old
single-trunk checkpoint reads only under old git; baseline numbers live
in the ledger and runs/). Eval instrument moved to row-level
teacher-forced alphas; the fp/fn-mass guardrail queries the branch under
the thresholded predicted support; inference sampling became two-step
ancestral (support drawn from the Bernoullis, then that support's own
Dirichlet -- `sample_assemblages` takes an `alpha_fn`, unique supports
batched).

**Considered.** An explorations/ pilot before touching src (rejected: the
build decision was already recorded; a standalone pilot drops the closure
coupling and the tuned loss weights, so it measures the pilot, not the
change). Mixture-of-Dirichlets first (rejected earlier: staged after
support-conditioning so each fix is measured in isolation).

**Decision.** Adopted. Val roster: joint hurdle NLL -1.1621 -> -2.4565
(all Dirichlet: -6.1046 -> -7.4538; bernoulli 4.9425 -> 4.9973, a real
but small presence cost from losing the shared trunk's multi-task
signal). Amp magnitude parity r 0.715 -> 0.876, MAE 0.135 -> 0.085;
PIT frac_shifted 0.649 -> 0.406. Gains broad, as the shifted-groups
finding predicted: plag r 0.834 -> 0.943, cpx 0.853 -> 0.926,
q 0.903 -> 0.959. Amp frac_two_sided rose 0.026 -> 0.114 -- the
within-support bimodality surfacing once the between-support shift is
explained; that residue matches the ~30% irreducible measurement and is
the mixture candidate's target, not a regression. Exposure bias measured
(new `dirichlet_nll_sampled_support` mirror of the cond keys): -7.45
teacher-forced vs -1.59 under one Bernoulli-sampled support per row
(delta +5.87 nats; amp parity r 0.876 -> 0.539). Read it as a
sensitivity diagnostic, not the deployed score: the instrument scores
sampled-support alphas against the true-support sub-composition, a
mismatch coherent ancestral samples never produce (each sample's
Dirichlet lives on its own support); the support-blind baseline scores
its teacher-forced -6.10 on this instrument by construction (alphas
ignore the support), so the delta prices how much the new head trusts
its lineup input -- the flip side of the conditioning gain.

**Next.** Mixture parameterization for the within-support humps only if
its expected gain survives the exposure-bias framing (amp two-sided
groups are 11% of groups vs 41% still shifted). #25 possibly contributes
a regen line item to #45: narrower trace-conditional (P, H2O) sampling
shrinks the within-support spread the mixture would chase. #46 redox and
#18 real-rock transfer remain the queued measurements.

## 2026-07-22 -- Fe redox attribution (#46): invisibility confirmed, one column recovers most of it

**Q.** The ceiling probe left spinel worst at the raw-chemistry ceiling
itself (per-bulk presence-fraction MAE 0.159 vs 0.024 binomial floor)
with fO2 as the residual hypothesis: the generator saw an Fe3+/Fe2+
split per bulk, the input carries only total Fe. Is redox invisibility
the driver, and how much is recoverable?

**Context.** Two measurements in `explorations/fe_redox_attribution`
(reusing the ceiling probe's targets, XGB protocol, and scoring; new
production checkpoint `joint_heads_supportmag_seed0`). (1) Per-bulk
errors sliced by `fe_provenance` (measured vs Middlemost class-assumed,
stamped in the corpus draws tables), on the val roster and a 2k-bulk
train holdout. (2) The XGB ceiling arm re-run under the generator's own
anhydrous 10-oxide basis with (`bulk10`) and without (`bulk9`) its
excess-oxygen column, plus the current input with excess-O appended
(`token22_xo`). Anchor reproduced (token22 spl 0.159).

**Considered.** Class-composition confound (measured provenance skews
old wet-chemistry mafic): ruled out -- the measured-group penalty
persists within every supergroup with n >= 20 per side. Label-noise
confound (measured labels merely noisier): ruled out -- per-group
binomial floors are equal (0.038-0.041); the penalty is discrimination
loss (r 0.54 vs 0.90). Base-rate confound: group mean f differs 0.59 vs
0.53, far too small for a 2x MAE ratio.

**Decision.** Redox invisibility confirmed as the spinel driver, and it
is mostly recoverable from one scalar. Val spl MAE: token22 0.159,
bulk9 0.158, bulk10 0.059 (r +0.94), token22_xo 0.072 -- removing
excess-O from the generator basis reproduces the 22-token ceiling to
0.001, and appending it to the current input recovers ~87% of the gap.
Gains are corpus-wide across Fe-bearing phases (ilm 0.118 -> 0.072, ol
0.040 -> 0.023, g 0.041 -> 0.027, plus ru/ep/cpx/opx), with bulk9 ~
token22 throughout; excess-O is the rank-1 spl feature in both redox
bases. Provenance split (holdout): measured +0.109 MAE over
assumed_class under token22 (CI [+0.094, +0.122]), collapsing to +0.023
under bulk10 on the identical bulks. Side finding: assumed_class bulks
also improve (0.109 -> 0.053) -- the class ratio keys on `qap_name`,
which chemistry does not fully determine, so even rule-assigned redox
is partly input-invisible. Residual: measured group still ~2x floor
under bulk10 (0.076 vs 0.039). Status-quo option is dead on these
numbers; the input-vs-sampled-condition call is the owner's, recorded
on #45. Also revises the ceiling probe's verdict: "the corpus is the
ceiling" was true of the 22-token input, but the binding constraint was
the input basis, not corpus size or draw noise.

**Next.** Owner call, same day: promote Fe redox to an optional input
token at the regen, uniform optional-token semantics -- fed where the
generator saw a value, padded/masked otherwise, unknown-redox inference
marginalizing implicitly through the probabilistic heads. No explicit
inference-time redox path: imputation readout and pin-and-sweep fall
out of the generic token machinery, and inference-time redox sampling
would duplicate what calibrated heads carry in one pass (a special
path would also imply redox outranks every other missing token, which
nothing measured). The distinction that earns redox the slot at all:
it is the only generator input that was not a model input; missing
traces create no label-input inconsistency because MAGEMin never
consumed them. Spec knobs deferred to #45: token parameterization,
fed-vs-padded for class-assumed bulks, train-time mask rate, and the
eval layer's padded-redox calibration check + redox-availability-
stratified reporting (real rocks mostly report total Fe, so the padded
regime is the deployment regime). The input change rides the regen's
encoder re-pretrain. Then #47 step 0 per the pre-regen queue.

## 2026-07-22 -- Lineup coherence (#47 step 0): off-corpus mass is material, the observed-pattern restriction is nearly free

**Q.** `sample_assemblages` draws each phase's presence Bernoulli
independently, so it can compose lineups the corpus never pairs; the
magnitude branch was trained teacher-forced on true supports only. How
much sampled assemblage mass lands on patterns appearing zero times
among the bulk's truth draws, and zero times corpus-wide -- and what
would restricting to corpus-observed patterns cost?

**Context.** `explorations/lineup_coherence/run_measurement.py`, val
roster (222 bulks), production checkpoint
`joint_heads_supportmag_seed0`. Off-menu mass computed exactly
(closed-form Bernoulli products over a pattern library, renormalized
for the sampler's empty-support rejection); 2000-draw MC cross-check
agrees within binomial noise (max dev 0.030, sd 0.011). Libraries:
bulk's own truth draws (pooled rungs, matching the rung-blind p),
train-tier observed patterns (5763 unique across 10k bulks), train +
val-truth union (5871).

**Considered.** Monte-Carlo measurement (rejected: the exact sum over a
finite library is cheaper and noise-free, and is itself the mitigation
math). Coarsened pattern identity (deferred: #47 design decision 1 is
downstream; this measurement is exact-21-phase by construction).

**Decision.** Material, by the issue's pre-registered criterion. Mean
10.7% of sampler mass (median 8.1%, p90 22.5%, max 48.4%) lands on
patterns never observed in the train tier -- pure extrapolation for the
support-conditioned magnitude branch and dead weight in a ranked
scenario list; union library barely differs (10.5%). It concentrates
where the product matters: off-train mass rises monotonically with the
count of intermediate-probability phases (0.001 -> 0.028 -> 0.058 ->
0.123), worst in rhyolites (top offenders 0.34-0.48). The mitigation
cost is negligible: only 0.6% of truth-draw mass (mean; max 17.6%, one
foid-bearing latite) sits outside the train library. Off-own-menu mass
is much larger (mean 65%) but ~55 points of it is corpus-plausible
patterns absent from the bulk's finite 128-draw menu -- accessory
flicker territory, the coarsening decision's problem, not the
coherence filter's. Bonus confirmed feasible: with a 5763-pattern
library, scenario weights become exact (one [5763 x 21] matmul per
bulk), removing MC noise from the #47 ranking. Adopting the heads.md
mitigation for the #47 build: restrict sampled lineups to
train-observed patterns and renormalize; report the per-bulk excluded
truth share rather than hiding it. This realizes mitigation 1 on #31's
ladder (incoherent sets removed exactly); the distribution-shift
component of exposure bias remains separately tracked by
`exposure_bias_nll_delta`.

**Next.** Per the pre-regen queue: #27, then the #47 build (pattern
library + exact weights + coarsening and threshold-semantics decisions
+ prediction-set eval), #18, (#40), #48 dry run, close #45, regen.

## 2026-07-23 -- N-scaling (#27): bulks are where the nats are; draws flat past ~24; regen re-shaped by NeurIPS deadline

**Q.** How does joint-heads quality scale with training-bulk count, and
does the bulks-vs-draws split justify the regen shape -- now under a
changed constraint set: NERSC down for weeks, NeurIPS submission in ~5
weeks, so the regen is a ~60 h laptop weekend (~875k draws at the
measured ~4 draws/s), not the 10x HPC run.

**Context.** `explorations/corpus_scaling/`: nested md5-hash bulk
subsets (625/1250/2500/5000/10000, prefixes nest by construction),
standard config retrained per point (joint heads + support-conditioned
magnitude, 120 epochs, seed 0), scored on the committed 222-bulk val
roster; plus draws-axis points at full bulks (24, 12 of the 48).
Reusable knobs added to the trainer (`train_bulk_ids`,
`train_draw_limit`, stamped into `label_config` when set). Full-N point
reproduces production `joint_heads_supportmag_seed0` to 4 decimals --
subset plumbing validated.

**Considered.** Pure log-linear fit (rejected: misses by 0.06 nats;
per-doubling gains shrink 0.42 -> 0.24) vs saturating power law
`L_inf + a*N^-b` (adopted: fits within 0.02, b = 0.225, floor ~ -3.9
but poorly constrained by 5 points / 1 seed -- planning-grade).
Draws-heavy regen shapes 15k x 48 / 10k x 72 (rejected: draws axis
measured flat -- 12/24/48 spans 0.08 nats non-monotonically while a
bulk doubling is worth 0.25-0.3; at matched rows all-bulks-fewer-draws
wins by 0.3-0.5 nats). 60k x 12 (rejected: 12 draws shows the one
visible cost, CondHead ~0.06 nats/doubling -- draws are the (P, H2O)
variation itself -- and thins the within-bulk multimodality margin).
Reusing old broad-proposal draws next to new trace-conditional draws
(rejected: the within-bulk distribution IS the proposal; mixing two
proposals contaminates the posterior the regen is fixing). Dense-draw
sub-tier insurance (~1k bulks x 96; deferred: two-tier bookkeeping in
deadline crunch, insurance against a risk the narrower proposal should
reduce).

**Decision.** Per-phase verdicts: garnet data-limited (pf_r 0.66 ->
0.83, still climbing); spinel weakly data-limited but structurally
capped (~0.70 pf_r; #46 redox token is the lever); amphibole not
data-limited (flat across the ladder; #25 mixture question); ilmenite
saturated. Sizing input to #45: allocate to bulks at 24 draws/bulk --
recommended weekend shape ~30k bulks x 24, keeping the existing 10k
train bulks and the sealed 500-bulk valtest roster (regenerated under
the new proposal, valtest at 256 draws), rare-class quotas on the new
bulks. Caveat: the 24-draw knee was measured under the broad proposal;
the #48 dry run checks it under the trace-conditional one. Owner locks
the final shape on #45.

**Next.** Compressed pre-Saturday queue: #48 dry run (proposal check,
sampler-level, fast), valtest-256 instrument bookkeeping, rare-class
quota list, lock #45 shape, launch generation. #47 build and #18 slip
past the weekend (don't gate generation; val-roster NLL +
interrogation harness are the frozen instrument). Redox token is
model-side, rides the regen's encoder re-pretrain.

## 2026-07-23 -- SSL pretrain corpus: relax the Fe pair rule, open to all rock types

**Q.** The #46 redox token already forces an encoder re-pretrain at the
regen. Should that re-pretrain keep the strict FeO+Fe2O3 pair filter
and the igneous-only filter on gard2019, or widen the corpus?

**Context.** The strict pair requirement is the dominant attrition step
in the SSL filter walk (544k rows with clean non-Fe majors -> 261k
requiring feo -> 172k requiring fe2o3; 2026-06-04): most sources report
total iron only, so the strict corpus keeps ~23% of the usable rows to
preserve a redox nuance the 22-token input cannot represent anyway. The
relaxed Fe fallback chain yields ~493k igneous / ~737k all-rock bulks
vs the production 166,893, and the supervised corpus already ships on
the relaxed loader (4497/4500 bulks recovered, 2026-07-15). #46
resolved redox invisibility with an optional Fe-redox input token --
measured-pair rows feed it, everything else pads it (implicit
marginalization) -- so relaxing the filter no longer discards redox
information; it moves into the optional token. Deployment argument:
real rocks at inference report whatever their lab measured; pretraining
only on strict-pair rows trains on a biased slice (sources that ran
ferrous titration), the wrong match for few-shot transfer beyond the
igneous corpus.

**Considered.** Keep strict (167k; rejects ~75% of usable rows for
nuance the input cannot see). Relax Fe only (~493k igneous). Relax Fe +
all rock types (~737k; the non-null qap_name criterion must drop for
the new rows -- QAPF is an igneous taxonomy, 0% non-null on sedimentary
-- so stratification/probing moves to rock_group as the coarse label
with qap probes restricted to the igneous slice, per #24).

**Decision (owner).** Ingest as much data as the encoder can get: the
next re-pretrain relaxes `require_all_majors` and drops the igneous
filter. Strict Fe matters where labels are made, not where the encoder
learns: MAGEMin generation wants measured speciation for label
fidelity, with the class-conditional literature fallback
(`fe_redox.py`, provenance stamped) covering unmeasured bulks.
Guardrails from #24 stand: a 1-epoch strict-vs-relaxed smoke
comparison, then the full scorecard cascade with no regression on the
igneous slice (the mineralogy heads consume that region of the latent).

**Next.** Batch with the regen-forced re-pretrain; spec knobs recorded
on #45 next to the redox-token knobs; #24 part 1 becomes this decision.
#24 part 2 (Andrews XRD sediment head) stays exploratory -- a cheap
transfer probe, no expectations attached.

## 2026-07-23 -- Trace-conditional (P, H2O) proposal dry run (#48): material shift, knee holds, adopted at default knobs

**Q.** Before the regen commits ~60 h of MAGEMin to the trace-conditional
proposal, does it change the per-bulk condition draws enough to justify
adoption, which bulks change most, and does #27's 24-draw knee survive?

**Context.** `explorations/trace_conditional_proposal/` (sampler-level, no
MAGEMin; both rosters, 512 draws/bulk/proposal, ~25 s). Candidate sampler
landed in `cnmineral.data.condition_proposal` beside the untouched broad
one: immobile-ratio cascade (Pearce 2008 Th/Yb-Nb/Yb -> Shervais 1982 Ti-V
-> Winchester & Floyd Nb/Y), soft logistic setting posterior, nine
(maficity, setting) literature boxes, fallback to the recovered ADR 003
Wallace/Plank H2O x Putirka P class tables, 8% broad floor; mixture density
closed-form so stored-q reweighting survives. Pearce constants pinned to
Sun & McDonough 1989 via pyrolite's data files; Shervais 20/50 and Nb/Y
0.67 on textbook values.

**Considered.** Routing felsic bulks to fallback (rejected: evolved melts
inherit the ratios; kept in the cascade with doubled bandwidth -- the
felsic-out variant drops trace-tier occupancy 0.585 -> 0.443). Knob
variants: sigma x0.5/x2 near-inert on every headline; eps 0.04-0.16 spans
median shrink 0.21-0.44 (the one real lever); all measured in
`knob_sensitivity.csv`.

**Decision.** Adopted for the regen at default knobs (sigma 0.15, eps
0.08, felsic in) -- owner locked 2026-07-23; knob effects on holdout and
real-rock transfer are speculative pre-regen, so tuning is deferred to a
future regen with every knob kept off extremes. Measured: 63% of bulks
reach a discriminant rung (30/26/7 by rung), median joint-envelope shrink
0.26 (H2O and P widths ~0.52 each), zero bulks widen (floor is the widest
component), 24-draw envelope contracts to 0.25 so the #27 knee holds with
margin; 56% of trace-tier bulks keep >0.5 nats of setting ambiguity (the
deep-wet/shallow-dry mass, preserved by design). 354 flags (3.4%; 324
`basalt` reassigned wet-arc) reviewed and accepted: the per-rock trace
evidence (published discriminant) overrules the hand-assembled class
table's single dry-basalt range -- the arc/MORB conflation inside the
`basalt` class is the gap the revision targets, so the flips are the fix
working, not a defect.

**Next.** Remaining pre-Saturday queue: valtest-256 instrument
bookkeeping, rare-class quota list, #45 shape lock, then wire
`draw_conditions_traced` into the generator at the regen build.

## 2026-07-23 -- Instrument bookkeeping: segmented trajectories, no cross-instrument re-measurement

**Q.** The regen doubles valtest draws (128 -> 256, noise floor 0.024 ->
~0.017) and replaces the broad proposal with the trace-conditional one, so
every ledger number is measured on a ruler that is about to change. How
does the version ledger survive the swap?

**Context.** `results/ledger.yaml` held four versions measured on the
current instrument, and its header promised the opposite convention
(re-measure old versions whenever the instrument changes). Re-measuring
would conflate distribution shift with model quality: old checkpoints were
trained against old-proposal draws, and the within-bulk distribution IS
the proposal.

**Considered.** Re-measure the three readable checkpoints on the new
valtest for one continuous trajectory (rejected: forces a comparison that
is never apples-to-apples, and the standalone-modehead checkpoint is
unreadable so the trajectory would lose its first point anyway). A one-off
bridge eval of the production checkpoint on the new instrument to price
the world-shift (withdrawn: the real-rock benchmark is the common test set
that ultimately compares era-best models -- tuned on real data, judged on
real data -- so no synthetic-corpus bridge is needed).

**Decision.** Record which ruler measured what; never compare across
rulers. `ledger.yaml` gains an `instruments:` block (roster, draws/bulk,
proposal, noise floor, tie band -- the durable home for those constants)
and a per-entry `instrument` stamp; entries are never re-measured on a
later instrument. `eval/trajectory.py` validates the stamp and draws one
disconnected segment per instrument (dotted boundary, per-era labels); the
first post-regen model is point 1 of the new instrument's segment, not
point 5 of the old one. Cross-instrument model comparison is the real-rock
benchmark's job. Also un-hardcoded the "128 draws/bulk" eval figure tags:
`JointHeadDataset` now exposes `draws_per_bulk` and the three head evals
derive the tag from it (the only 128s found outside display strings were
none).

**Next.** Rare-class quota list, #45 shape sign-off, generator wiring of
`draw_conditions_traced`. When the first post-regen eval lands, define the
new instrument entry in `ledger.yaml` (measure its noise floor and tie
band on the 256-draw roster rather than assuming the 1/sqrt(2) scaling).

## 2026-07-23 -- Rare-class quotas locked: alpha=0.5 temperature smoothing

**Q.** How should the regen's ~20k new train bulks divide across the 22
qap classes, so the allocation is deliberate rather than an incidental
replay of pool abundance?

**Context.** Eligible pool 324,327 bulks; the big four hold 89%, the
bottom ten classes have 116-508 bulks total. The current 10k left tail
classes at 32-44 bulks each, and #27 measured rare-phase skill (garnet)
still data-limited at 10k while #41 named corpus enrichment the
remaining presence lever. Class abundance is power-law shaped
(rank-frequency slope -2.59, R2 0.948), which makes temperature
smoothing (quota ~ pool^alpha) the natural single-knob family: it maps
a power-law pool to a power-law corpus with slope scaled by alpha.
Analysis + candidate tables in `explorations/rare_class_quota/`.

**Considered.** Proportional (alpha=1: tail gets literally zero new
bulks). The marginal-utility optimum alpha = 1/(1+b) = 0.82 from the
#27 scaling exponent under frequency-weighted loss (rejected: "optimal"
for the headline but isolates the rare classes -- bottom ten get ~250
new bulks, smallest class frozen at 32). Take-all-tail + alpha=0.82
hybrid (rejected: the hard cutoff makes final counts non-monotone in
abundance -- foid-bearing trachyte ends at 502 while the more abundant
alkali feldspar rhyolite ends at 162). Uniform / alpha<0.5 (no benefit:
tail is availability-bound below alpha ~ 0.5, the knob only reshuffles
head vs middle).

**Decision.** alpha = 0.5 (square-root sampling: published precedent in
Mahajan et al. 2018; word2vec 0.75 / XLM-R 0.3 bracket it),
availability-capped, monotone by construction, ~95% of the maximum
possible tail rescue with saturation emergent rather than imposed.
Owner condition: the paper appendix shows the candidate table and this
reasoning. Landed: `expansion_quotas` in
`cnmineral.data.volcanic_subset` (tested), `--expand` mode on
`build_corpus_bulk_tables.py`, official table
`data/corpus/train_bulks_expansion.csv` (20k at seed 20260714: big four
+10.3k, middle eight +7.6k, bottom ten +2.1k with 8/10 exhausted,
smallest class 32 -> 111). The locked item is the rule (alpha=0.5 +
caps); the budget flexes with actual weekend hours. Side checks:
measured-Fe-pair share in the tail (22-58%) matches the head, and tail
exhaustion costs no NERSC headroom -- gard2019 has no more of these
classes at any scale.

**Next.** #45 shape sign-off, then generator wiring
(`draw_conditions_traced` + redox stamping) and launch.

## 2026-07-23 -- Regen generator wiring: traced proposal in, nested-block expansion emit, redox needs no new column

**Q.** Wire the adopted trace-conditional proposal (#48) into the corpus
generator for tonight's launch (#45) without touching the orchestrator's
fault-tolerance machinery, consume both bulk tables, and make an early
stop preserve the alpha=0.5 quota shape.

**Context.** The swap point is a single call inside
`magemin_corpus.build_inputs` -- `bulk_proposal(...).draw()` replaces
`draw_conditions()` -- one layer upstream of `run_group`'s bisection,
`run_chunk`'s atomic flush, and the manifest resume, all untouched.
Per-bulk seeding (`seed + sample_id`) carries over, so draws stay
chunk-boundary-independent and a resumed chunk regenerates identical
draws (bisection boundaries included). The sampler additionally needs
`tio2` for the Ti/V rung, passed alongside the ppm traces.

**Considered.** Stamping nothing from the proposal (rejected: tier/rung
routing records what the generator actually did against future knob
drift; two per-bulk categoricals, owner sign-off). A new redox column
(rejected: `in_feo`/`in_fe2o3` are ratio-invariant under the H2O
renormalization and, with `fe_provenance`, derive both #46 token
parameterizations exactly -- stamping enough to defer the choice is the
point). Draw order as returned (rejected: `BulkProposal.draw` groups
draws by mixture box, so a draw-index prefix -- what `train_draw_limit`
takes -- would be a single-box biased subset; draws are permuted
per bulk after sampling).

**Decision.** Landed: traced sampler + `proposal_tier`/`proposal_rung`
stamps in `build_inputs`; orchestrator `--bulks` takes multiple tables
(concat in order, duplicate-sample_id guard); `--expand-block M` on
`build_corpus_bulk_tables.py` emits the expansion as two shuffled blocks
(M-budget quota selection first, then the increment; per-class nested
picks, min() against largest-remainder wobble) so stopping mid-window
realizes ~alpha=0.5 at the achieved N. Nested emit provably reorders
without changing the selection (tested). Also fixed the orchestrator
test double, red since the solver-hang timeout landed (missing `timeout`
kwarg) -- all four fault-tolerance tests were failing on HEAD.

**Next.** Owner stamps the budget on #45, re-emits the expansion at the
launch allocation (`--expand N --expand-block M`), runs valtest first
(~9 h, early attrition read on the narrower boxes), then the train tier;
during-regen track switches to #18 transfer/few-shot.

## 2026-07-24 -- Sim-to-real zero shot: the gap is calibration, except for amphibole, where it is discrimination

**Q.** Measure the transfer bet intact -- does a ModeHead trained on MAGEMin
reference-state labels predict the minerals a geologist counts? -- and read
the residual shape closely enough to specify the thin adapter (#18 build 2)
rather than guess it.

**Context.** Path A, zero-shot, nothing fit on real rocks: the benchmark
stays eval-only until the gap is measured, since fine-tuning first spends the
one honest transfer number we get. The model speaks 21 minerals; real modal
data resolves ~7 groups. Both halves of the coarsening turn out exact --
presence composes as a noisy-OR (the head emits independent Bernoulli
marginals), magnitude by Dirichlet aggregation, and dropping the phases no
source reports is the sub-composition property `masked_dirichlet_nll` already
rests on -- so the measured gap is the model's, not the mapping's.

**Considered.** Metric: the headline is the loss the adapter minimizes,
evaluated at the identity correction (per-group Platt scaling *is*
Bernoulli-NLL fitting), so build 2's gain reads as nats recovered on one
scale. Trace rows: imputation rejected on measurement -- phenocryst totals
are small (median 13.5 vol%), so renormalization puts any imputed constant
inside the range of genuinely-reported values; left-censoring adopted
instead. The censored term is reported beside the headline, never summed in:
density-on-observed times a marginal tail probability is not a joint
likelihood, because the Dirichlet couples the components. Ledger: a separate
file, since corpus entries are per-head NLLs stamped to a valtest instrument
and never re-measured across instruments -- which leaves the real rocks as
the only cross-era comparison. No instrument stamp, but a `benchmark`
version, since the roster itself will move.

**Decision.** Landed `cnmineral.eval.real_rock` + `scripts/eval/real_rock.py`
+ `results/real_rock_ledger.yaml` (`eruption-v1`, entries pending), and
`tokenize_bulk_frame` extracted so real rocks tokenize through the corpus's
own path. Headline on `joint_heads_supportmag_seed0`: **joint NLL 10.404**
over 635 least-altered rocks (bernoulli 5.708 + dirichlet 4.696).

The read: **ranking transfers, absolute numbers do not.** Against an
always-present baseline, four groups carry real signal (ol +0.32 F1, q +0.25,
px +0.09, bi +0.06), two are saturated at ~95% base rate (fsp, opq -- nothing
to discriminate), and **amphibole is no better than trivial** (+0.02, AUROC
0.58) despite being the phase the transfer story turns on. Magnitude R2 is
negative for four of seven while Pearson r runs 0.5-0.76: the model orders
rocks correctly and reports on the wrong scale. Per-group offsets are large
and structured (amp presence +0.39, q -0.41; opq under-sized ~2.7x), and the
residual-vs-SiO2 read says the adapter needs a silica term on q and amp
presence only. Off-vocabulary mass is ~0, so nothing is hallucinated.

Three traps found and encoded: coarsening leaves exact-zero concentrations
and `mask * lgamma(0)` is `0 * inf` = NaN (fixed at the call site, not in the
shared loss); the Dirichlet term's scale moves with composition size, so a
subset of simpler rocks scores "better" (every subset block now reports
`mean_composition_size`); and a group receiving both a number and a trace
counts as quantified, or the truth vector silently stops summing to 1.

Retracted in review: per-group `r^2` was quoted as a recalibration ceiling.
It is the R2 of an optimal affine map on shares, while the adapter is a
multiplicative shift on concentrations plus renormalization -- a different
model class that couples all groups -- it was fit with nothing held out, and
it forecasts R2 for a thing that minimizes NLL. Not a bound. The only
rigorous version is fitting the adapter under cross-validation.

**Next.** Sequencing agreed with the owner: (1) mapping sensitivity audit --
perturb densities, the 0.5 vol% trace bound, the alteration rule, the support
forcing, and see how far the headline moves, since everything else rests on
it; (2) the thin per-group adapter under k-fold, to establish an honest
held-out baseline; (3) regen results as the separate track they already are.
Group-coupled correction (the only form that can move ranking), hierarchical
shrinkage across groups, and folding the 413 censored instances into the
objective are held until there is a held-out number to beat. Still open: how
`R_theta` is fit versus preserving a sealed estimate, and the benchmark's
noise floor -- most of the set is visual estimates at +/-5-10 vol%, and 6
point-counted rocks cannot pin it.

## 2026-07-24 -- Truth-mapping audit: robust where it measures, fragile where it invents

**Q.** The zero-shot headline (10.404 nats) and the residual shape build 2 will
be designed against both pass through mapping choices nobody measured --
mid-series densities, a 0.5 vol% trace bound, a leading-digit freshness parse, a
rule for splitting group-level truth into cpx/opx. Does any defensible reading
of those change a decision? Step 1 of the agreed sequencing, run before the
adapter.

**Context.** Framed as three decisions rather than a nats tolerance: the
adapter's parameterization (per-group constants + SiO2 slopes on q and amp), the
amphibole verdict (discrimination failure, AUROC 0.58), and the headline's
quotability. `explorations/real_rock_mapping/run_sensitivity.py` re-scores one
frozen checkpoint under ~30 variants across six knobs, always through the
shipped loader and loss -- an audit of a reimplementation audits nothing.
Production gained exactly two seams (`load_benchmark(densities=,
trace_bound_vol_pct=)`, `truth_forced_support(member_threshold=)`); every other
variant is derived in the exploration, including the imputed-trace policy, which
is an exact algebraic transform of the loaded arrays rather than a second
mapping. Nothing is fit, so the adapter's held-out estimate is untouched.

**Considered.** A materiality-free sign test flagged amphibole's magnitude
offset as flipping under seven variants; its shipped value is -0.002, so every
perturbation reverses a correction that does not exist. The flip test now
requires both sides to clear 0.05 of presence probability / 0.10 log units on at
least 100 rocks. Also caught in review: `fill_between(x, v, v, step="mid")` does
not render as zero height -- it fills the span between neighbouring groups, so
the first figure drew a density uncertainty band onto the presence panel, where
density provably has none. Per-group segments now.

**Decision.** The mapping is robust where it encodes a measurement and fragile
where it encodes an invention, so **build 2 proceeds on the shipped mapping and
the residual shape as measured**.

Two invariances are exact, not small: the Bernoulli term is bit-identical
(5.708414604) across every density variant, and the headline is bit-identical
across trace bounds from 0.1 to 1.0 vol%. So every presence-side conclusion --
the amphibole verdict included -- cannot move with either, and both are asserted
as tests. D1 holds: no presence-offset sign flips, one mechanical magnitude flip
(px under `all_members`, which splits the group's concentration by
construction), and the Monte Carlo pins every adapter constant to roughly ±0.06
log units (opq +0.942..+1.027, bi +0.445..+0.568, q -0.541..-0.519). D2 holds
outright -- amphibole AUROC spans 0.497-0.592 across every variant, never better
than trivial, below chance under one. D3 holds for the measurements (densities
±0.3 nats, alteration cutoff ±0.3, uncertain IDs and xenocrysts ±0.02) and
**fails for the invented rule: support forcing spans 9.86-13.38**, a 3.5-nat
lever, an order of magnitude more than any physical assumption.

Two findings worth carrying. The vol%-to-mass conversion does real work in the
conservative direction: skipping it improves the headline by 2.4 nats, more than
the entire plausible density band, because it is what exposes the opq
under-sizing (oxide offset +0.991 -> +0.583 in volume units). MAGEMin modes are
mass fractions, so converting is correct and not converting would have flattered
the model on a basis it does not predict -- and *which* density values are used
matters ~8x less than the choice to convert. And the groundmass drop is what
makes visual estimates commensurable with point counts: the 13 groundmass rows
are exactly the 13 point-counted samples, dropping those rocks moves the
headline +0.04 nats, and the literal "keep groundmass as an unscored denominator
component" is not scoreable at all (no `liq` in the 21-phase vocabulary to pair
it with, and a Dirichlet needs its components to renormalize).

**Next.** Build 2, the thin per-group adapter under k-fold, unblocked. Two
carry-forwards from this audit: treat the support-forcing rule as a modelling
choice with a 3.5-nat lever rather than a mapping detail, and pin amphibole's
magnitude constant at zero instead of fitting it, since its Monte Carlo band
straddles zero and no density makes the sign recoverable. Left-censoring stands
(imputing costs 2.2 nats, but at composition size 4.23 vs 3.61, so that is not a
like-for-like read and is reported with the size stamped). Still open, unchanged:
how `R_theta` is fit versus preserving a sealed estimate, and the benchmark's
noise floor.

## 2026-07-24 -- Support density and magnitude calibration: backwards, and one metric is coverage-selected

**Q.** Two readings off the real-rock eval suggested ModeHead's magnitude branch
is better calibrated when the support it conditions on is *sparser* than truth:
the forced-support ladder improved monotonically as it thinned (6.45 phases on
support 13.38 nats, 4.92 10.40, 4.59 9.86), and the exposure probe went the
"wrong" way, the model's own support beating the measured one by 0.31 nats. If
the cause were a train/inference mismatch -- teacher forcing only ever shows the
branch a true assemblage, while `sample_supports` draws from the presence
marginals and over-calls -- then presence miscalibration would also be a
mode-magnitude problem, and presence work would be justified by mode value.
Testable entirely on the corpus.

**Context.** Stratifying the observed penalty by support size cannot separate
that from its rivals, since a row's support error correlates with how hard the
row is. So the probe decomposed instead: perturb the conditioned support in two
controlled directions from truth (pad with the most probable absent phases, drop
the least probable present ones) while the evaluated mask stays the true support.
`masked_dirichlet_nll` masks by truth, so the Dirichlet's dimension is fixed
across rungs and density is the only moving part. Frozen
`joint_heads_supportmag_seed0`, valtest val roster, 218 bulks / 55,576 rows,
nothing fit.

**Considered.** Two rival explanations, both corpus-testable. The exposure probe
is defined only where the sampled support covers every measured group, and those
are the rows whose presence the model got right, so its sign could be selection.
And the forced-support ladder runs through a coarsening step that sums member
concentrations into groups, inflating alpha0 and sharpening the density whether
or not the branch is miscalibrated -- a mechanism with no corpus analogue.

**Decision.** **The hypothesis is refuted and both rivals hold.** A denser
conditioned support is monotonically *worse*, ~1.25 nats per extra phase
(+1.26 / +2.60 / +3.87 / +5.40 at +1/+2/+3/+5), with share MAE degrading 0.042
-> 0.098; padding *dilutes* alpha0 on the true support (45.9 -> 35.5) rather
than sharpening it. The premise fails outright: expected sampled support size is
4.79 against a true 4.81, so the presence head does not over-call in aggregate
(largest per-phase bias bi +0.012; amphibole is *under*-called by 0.012).

Restricting the corpus to covering supports moves the exposure penalty from
**+5.865 to +1.635** -- 72% of it -- with no model output changed. Corpus
coverage is 19% against the real-rock probe's 45% over 7 coarse groups instead
of 21 phases, so the surviving subset there is more selected still, enough to
carry a +1.6-style number past zero. The all-rows figure reproduces the shipped
`exposure_bias_nll_delta` to the digit. The forced-support ladder is the
coarsening artifact: on real rocks extra members are summed *into* the scored
groups and sharpen; on the corpus extra phases sit outside the scored mask and
dilute. Opposite signs. That ladder measures the invented support rule, not the
head, consistent with the mapping audit's own verdict.

What survives: support errors cost in both directions, misses ~2.1x spurious per
phase (+2.71 vs +1.26), and a sampled support of exactly the right *size* but
wrong membership costs +4.77 -- membership dominates, size barely matters.

**Exploration deleted at the owner's call as fluff**; the probe, its tests, and
its figures are gone and this entry is the record. The return did not justify a
directory: one negative result and one metric correction.

**Next.** The single actionable consequence, on #52: the exposure-bias
instruction as filed ("adaptation that improves teacher-forced magnitude while
degrading self-support magnitude is not a win") is not comparable across
adapters, because the adapter changes presence and therefore changes which rocks
are coverable. Freeze the coverage set at the zero-shot baseline's and score
every adapter on those same rocks. Nothing else here informs the adaptation
module: the miss/spurious asymmetry is already handled by minimizing the joint
NLL, and it was measured in 21-phase space where the coarsening flips the
density effect's sign, so it should not be assumed to transfer.

## 2026-07-25 -- Few-shot sim-to-real: eight eruptions buy most of the gap, and coupling pays

**Q.** How many labelled real rocks bridge the sim-to-real gap while still
generalizing to eruptions the adapter never saw (#52)? The deliverable is a
learning curve, not a recovered-nats point estimate: per-rock joint NLL has sd
13.8 against mean 10.4, so a level on a ~160-rock slice carries +/- 2 nats, and
only the shape survives that.

**Context.** Two decisions taken before any code. Pinatubo's chemistry was
transcribed first so the curve and its baseline share one benchmark version.
And overfitting is controlled by **one weight-decay constant per tier, chosen
once on the fixed val slice** and shared across every budget and draw, rather
than per-draw early stopping -- so every drawn rock on the x-axis is a fitted
rock and "N rocks buys this much" is reproducible by harvesting N rocks. The
val slice's entire contribution to the curve is three scalars.

**Considered.** Transcription was made checkable rather than trusted: Table 7's
majors are normalized volatile-free so each row's ten oxides sum to 100
(asserted in the parser), and each block prints an Average the transcribed rows
must reproduce. That caught a misread FeTO3 (6.98 could not reproduce the
printed 9.01 block mean; it is 8.98). 5 of 6 rocks landed -- `CN6791-h` has no
Table 7 row, the nearest being `CN6791d`, a different split of the same
locality, and equating them would invent a measurement. Split feasibility got
one constraint rather than a scored term: the test slice's silica range must
sit inside train's, since the adapter carries SiO2 terms and scoring it beyond
its fitted range measures extrapolation. The tails are too thin to balance on
(6 rocks above 80 wt%, the largest at 90.2 against a 99th percentile of 80.0),
so scoring the endpoints would hand split selection to an outlier.

**Decision.** Landed `cnmineral.models.real_rock_adapter` (three nested tiers,
diagonal in linear in mlp, the 23-param affine being the degenerate case),
`cnmineral.training.real_rock_adapter` (fit, cluster bootstrap, weight-decay
selection), the committed unit roster, and
`explorations/real_rock_fewshot/`. Benchmark bumped to **eruption-v2** (724
scoreable / 640 headline); the zero-shot headline moves 10.404 -> **10.406**,
which is the transcription's first check.

**Adaptation is cheap and steeply diminishing.** About **25 labelled rocks --
eight eruptions -- recovers 3.4-3.8 of the 5.0 nats** that all 368 recover.
Fourteen times more data buys the remaining third. At the full train slice the
linear tier reaches +5.03 nats on the Dirichlet term (+8.06 joint, CI
[+5.73, +10.49]) and share MAE 0.1068 -> 0.0876.

**Coupling pays, nonlinearity does not.** `linear` wins at every budget from 50
rocks up, and on 10 of 10 paired draws from 200 up; the MLP's extra capacity
never repays itself. So the correction is genuinely group-coupled rather than
seven independent constants. The MLP does take the better median at 25 rocks,
but on the paired draws (all tiers see identical ones) it wins only 6 of 10 by
a mean 0.22 nats against a 2.13-nat draw-to-draw spread, carried by one bad
draw where its heavier shrinkage limits the damage -- a regularization
robustness effect, not capacity paying off.

**Much of the gap is over-confidence, not wrong means.** The fitted diagonal
drops alpha0 on the scored support 46.2 -> 25.1, widening the Dirichlet by half.
At 25 rocks it *worsens* share MAE (0.107 -> 0.114) while still improving the
Dirichlet term: at that budget it buys calibration, not accuracy.

Two corrections to my own plan. Its verification step said the fitted diagonal
should recover the measured per-group offsets (opq +0.99, bi +0.51, q -0.53) --
wrong, and the same confusion this journal already retracted once: those are
share moments, while the adapter is a multiplicative shift on concentrations
whose 7 dof split into 6 mean-directions plus one sharpness, and the NLL spends
much of that budget on sharpness. And no adapter entry goes in
`real_rock_ledger.yaml`: an adapter is scored on 161 held-out rocks against a
zero-shot entry measured over 640, which is exactly the mismatched-row-set
comparison that file's reading rules exist to prevent.

**One bug worth carrying.** The coupled tiers read `log(alpha)` as an input, and
coarsening leaves exact zeros where a group has no member on the support.
Clamping to `1e-12` before the log hands the mixing matrix a **-27 spike** on
precisely the groups carrying no information; `linear` and `mlp` diverged on it
(val NLL 1e20, erratic non-monotone weight-decay curves) and the first read was
"the coupled tiers do not help." They do -- an off-support feature is the
neutral 0. The wrong finding was one sweep away from shipping.

**Next.** HPO on train/val, now that the curve establishes adaptation works and
what it costs. Re-run the same curve on post-regen heads: under the
segmented-ledger convention this benchmark is the cross-instrument judge (#45),
so that pre/post comparison is the regen's transfer evidence. Untouched by
design: amphibole detection (a monotone per-group map cannot move AUROC --
hands off to regen and #49) and the benchmark's noise floor, which ~11
point-counted rocks still cannot pin. The curve's spread band is not a noise
floor.

## 2026-07-26 -- Valtest grows by extension to 2000: aggregate precision is buyable, rare-class resolution is not

**Q.** The traced train tier finished (35k bulks x 24 draws, 71.5 h, 0.306
s/draw, `data/corpus/train_traced`) while the measurement instrument stayed at
500 bulks. What valtest does the new instrument deserve?

**Context.** A new instrument starts regardless -- the traced proposal changes
the label distribution, so valtest numbers open a new ledger segment either way
(the reading rules in `results/README.md`). The open question is only the
roster: size, composition, draw budget.

**Considered.** Three shapes. *Keep 500*: free, but nine mid classes sit at
5-14 bulks -- anecdote, not measurement. *Redraw 2000 fresh* from the
contamination-safe remainder: dead on arrival, because the +25k train expansion
left 9 of the 22 classes with zero bulks outside train -- a fresh draw would
carry only 13 classes, and the committed 500 is the sole holder of the other
nine. *Extend to 2000*: keeps everything the 500 holds and grows where bulks
still exist. Rare-class resolution is foreclosed at any budget -- that was
spent, knowingly, when the expansion took the bulks. On draws, 128 rather than
256: quadrupling rocks already averages per-rock noise 4x, while more draws buy
only per-rock precision the aggregate does not need.

**Decision.** Extension, never redraw (`build_valtest_split.py --extend`): the
499 committed assignments are frozen -- growing a sealed roster must not move a
bulk from test to val -- and only the added bulks get assigned, 1999 now 887
val / 1112 test. Drawn at alpha=0.5, the same rule the 25k train expansion
used, so the roster mix tracks the realized 35k train mix to ~1pp; an
instrument weighted differently from the training distribution measures
something the model was never asked to do (`expansion_quotas` targets the
expanded tier's shape, so all 2000 carry it, not just the added 1500).
Instrument stamp `valtest128-traced@2000`, noise floor left null until
measured. Plumbing that fell out: `draw_expansion` takes have/used from the
caller so one contamination-guard definition serves both tiers, and
`report_corpus_health --bulks` accepts several tables (the train tier is two
CSVs, which was silently skipping its representativeness plot).

**Next.** Generation is ~22 h (lands 2026-07-27); then the cascade -- corpus
build, head retrain on the traced tiers, mini-HPO, re-score on the new
instrument -- with the real-rock benchmark judging the era transition
(zero-shot entry + the #52 curve re-run, per the segmented-ledger convention).

## 2026-07-28 -- Cascade 1 kickoff: heads wired to the traced tiers, the instrument becomes code-read

**Q.** With both traced tiers generated and the encoder re-pretrain running
in a parallel session, how does the heads retrain sequence, how big is the
mini-HPO, and where does the tie band live now that the instrument it was
sized for is retired?

**Context.** Every ledger entry so far is stamped `valtest128-broad`; the
new instrument's stanza sat with null floor and band while `tie_band: 0.03`
lived on in three sweep configs nothing connected. The #32 capacity answers
(mode 256x3, comp 128x1, cond 512x2) were bought against the same frozen
latent cascade 1 keeps.

**Considered.** Sweep-first (fastest headline, but corpus and tuning
effects fold together and the tiebreak runs on a stale band). Full #32
repeat (~82 trials; re-buys certainty on capacity answers whose mechanism
did not change). Tie band in a new tracked instruments file, or manual
copies plus a consistency test (both keep a second copy alive).

**Decision.** Owner-locked, four pieces. (1) *Baseline first*: retrain at
the incumbent dials on the traced tiers (`joint_heads_regen_seed0`),
re-score, then sweep -- two ledger entries mirroring joint-baseline ->
joint-tuned, so the corpus change is measured alone. (2) *Weights-only
mini-HPO*: 24 trials over lr / weight_decay / magnitude_weight /
lambda_balance / dropout with the incumbent enqueued as trial 0 and head
sizes locked; escalate to the arch mini-sweeps only if the winner lands at
a search edge or comp/cond degrade beyond the band; the full space re-opens
when the encoder changes (#54). (3) *One source of truth for the band*: the
ledger's instruments stanza is canonical -- `run_hpo` resolves it via
`ledger_tie_band` and refuses an unmeasured (null) instrument; the three
config copies are deleted and `results/` is tracked from here on. (4) *The
floor is measured where it is used*: `bulk_frequency_targets` +
`binomial_floor` promoted from the presence-ceiling exploration into
`eval.mode_head`, so every ModeHead eval stamps per-phase floors into its
`metrics.json`. The stanza field is renamed `noise_floor` -- the value is
presence-fraction MAE, not nats -- while the band stays `tie_band_nats`,
adjudicated by the owner from the floor plus the seed-noise scale rather
than unit-converted. Riders: the #49 amp gate check (report-only) and
recalibrator HPO join this cascade's real-rock batch; #53 waits for
cascade 2, where it decomposes encoder-vs-heads in one shot. Wiring smoke:
34,474 train bulks x 24 draws (1.65M rows at rungs [50, 30]); 871 of the
887 val-roster bulks carry corpus rows (3 unresolvable chemistry, 13
rowless -- same ~1.5% as the old instrument's 218/221); nine classes sit
permanently at 5-6 valtest bulks, recorded as `thin_classes` in the stanza
so per-class numbers for them are structurally caveated as anecdote.

**Next.** Owner launches the baseline retrain once the parallel session's
imputation HPO frees MPS (both trainers auto-pick it; no CPU flag exists).
The first eval prints the measured floor, the owner sets the band, the
stanza fills, and the sweep unlocks.

## 2026-07-28 -- Acceptance: the all-rock + redox encoder rides as-is

**Q.** Does pretraining on the ~4.7x all-rock corpus with the optional
redox token cost the igneous representation the heads consume -- and does
the token itself pay?

**Context.** Spec locked 2026-07-27 on #24 / #45. The first training run
was superseded by the valtest-pin amendment (the merged roster pins the
previous encoder's surviving test rows AND the supervised eval bulks into
SSL test; the dedup-prefer knob makes pinned rows win their duplicate
groups). The production run early-stopped at epoch 16 (val 0.2000). All
head-to-head numbers are on the inherited shared-row roster (32,950 rows
scored in the new corpus, 33,371 in the old), new encoder redox-padded;
the old-encoder numbers are fresh roster-restricted re-runs, not the
recorded own-split scorecard values.

**Considered.** A mechanical tie-band gate (rejected at spec time -- the
owner adjudicates the table, with the ADR 007 cross-seed spread as reading
aid). Delivering on one seed (rejected: the deltas sat at the band edge,
so the pre-authorized seed1 tiebreaker ran). Running the Strategy-A
re-HPO before adoption (owner declined in the interest of time).

**Decision.** Adopted ride-as-is; owner adjudicated same day. Numbers:
qap MLP 0.759 vs 0.769, supergroup MLP 0.814 vs 0.821, kNN qap purity
0.577 vs 0.582, LR probes -0.018; seed1 (0.765 / 0.812) puts cross-seed
spread at 0.002-0.006, so the igneous deficit is real, small, and an
upper bound -- the HPs are the previous corpus's optimum ridden unchanged
for clean attribution. Trace imputation is a tie: re-tuned head 0.832 vs
0.836 on the same rows (sweep best-val 0.8557, winner hidden 1024).
Redox evidence: populated-vs-padded qap MLP +0.015 (0.774, above the
previous encoder), and since the inherited roster is entirely
measured-pair rows this IS the measured-slice delta; redox-imputation
readout log-R2 0.306. Anisotropy: effective rank 48.9 vs 32.0 of 256
(own-population characterization). Scorecard re-scored under the
shared-row protocol; the Strategy-A re-HPO is deferred to #55 (P2-medium)
with the adoption bar unchanged (beat ride-as-is on this battery; adopting
later re-triggers the heads cascade). One correction caught during the
scorecard write: the roster-scored rock_group probe (0.982) is igneous
recall on an all-igneous scoring population, not group separability --
the full-test-split number is being measured and the scorecard line
stays pending until it lands.

**Next.** Cascade 2 consumes `pretrain_allrock_redox_seed0` (heads
retrain; the redox pad-dropout rate and the padded-redox calibration eval
are decided there). #55 revisits the encoder optimization HPs against the
same battery. The seed1 checkpoint stays as the spread reference until
cascade 2 lands (untracked; deletion is final).

## 2026-07-29 -- Cascade 1 measured: the traced corpus moves the dials, sharpness buys teacher-forced fit at sampled-support cost

**Q.** At fixed dials, what did the corpus change do; what did the
weights-only sweep move; and did the incumbent dials survive?

**Context.** First eval on `valtest128-traced@2000` measured the floor:
binomial max 0.0206 (presence-fraction MAE) vs the old instrument's 0.024.
Owner carried the 0.03-nat band (the ~0.021 seed-noise cross-check is an
old-corpus figure; a fresh one only materializes if the arch mini-sweeps
ever run). Baseline retrain at incumbent dials (`joint_heads_regen_seed0`,
120 epochs, ~36 s/epoch at 1.65M rows): mode -5.4187, comp -56.2998, cond
7.4035, closure 0.0107.

**Findings.** (1) *The incumbent lost beyond the band.* Sweep
(`joint_heads_traced_hpo`, 24 trials, 60-epoch horizon): tiebreak winner =
optuna best = trial 22, mode monitor -8.9421 vs trial 0's -8.8246 -- 0.118
nats, ~4x the band -- with comp and cond also better than trial 0. Neither
escalation trigger fired: the winner is interior on every dial (the
magnitude_weight cap region was probed at 9.4-9.8 and lost), and nothing
degraded. (2) *The dial story is magnitude_weight 0.25 -> 7.76 (31x) and
lambda_balance 54.5 -> 0.012.* The closure penalty is redundant on
mass-balanced labels: flat within noise from 1e-3 to ~50, harmful >= ~500,
and the tuned run's converged closure (0.0109) is identical to baseline's
-- the 60-epoch closure loosening was comp non-convergence, not penalty
loss. (3) *Production retrain reproduced the sweep exactly* (mode best
epoch 43 = trial 22 to four decimals; early stop at 102): eval mode
-5.4774 (0.059 better than regen-baseline, ~2x band), comp -56.2860
(0.014 in-band deficit), cond 7.3577. (4) *Sharpness trades sampled-support
robustness for teacher-forced fit*: dirichlet on sampled supports -4.08 ->
-3.47 (exposure delta +4.93 -> +5.64), cond-T sampled 6.98 -> 7.39. The
support-conditioned magnitudes got more specialized to the true support,
so self-sampled support errors cost more -- the objective never priced
that regime. (5) *Three rows printed inf on cond-T sampled-support*: their
sampled supports gained 3 spurious phases, CondHead read the bloated
lineup as colder (T posterior 720-790 C, sigma 10-16, truth 1020-1040 --
15-20 sigma out), and the density underflowed float32 because
`clip(1e-300)` is 0.0 in float32 -- a dead guard. `gmm_nll_per_axis` now
computes via `logsumexp` (regression test added); true row NLLs are
118-141 nats, and the finite T sampled mean is 7.3895.

**Decision.** Trial 22 dials adopted verbatim (precedent #32);
`joint_heads_regen_tuned_seed0` is the production checkpoint. Ledger
entries (regen-baseline + regen-tuned) land at ship per #44 order.

**Next.** Real-rock batch on the production checkpoint: zero-shot (the
cross-era judge; 10.406 to beat), #52 few-shot curve re-run, recalibrator
HPO (sweep designed with the owner, carve-out selection only), #49 amp
gate check (report-only). Then ship flow and the v3.4 tag.

## 2026-07-29 -- Era tags renamed by contribution; weights stay out of git until submission

**Q.** What should the era tags be named, what does an annotation carry,
and where do production weights live between now and publication?

**Context.** The old names were uneven: `v1.0-trained` said nothing,
`-final` rotted the moment v3 existed, `v3.1-log-imputation` was a
lightweight tag whose displayed "message" was an off-topic commit message,
and proposed replacements had traps of their own ("traced" reads as
tracing-a-source; "emulator" is v1's explicitly rejected Position-B
framing).

**Decision.** Rename map (all re-created at their original commits with
original tagger dates recovered from the unreachable old tag objects;
stale references in pretrain_applications.md, ADR 007, and supergroup.py
updated): v1.0-trained -> v1.0-mode-cond; v2.0-traces-ssl-final ->
v2.0-traces-ssl; v3.0-refactor-imputation -> v3.0-own-transformer;
v3.1-log-imputation -> v3.1-logmse-objective (now annotated);
v3.2-encoder-representation -> v3.2-label-corpus-limited; v3.3-sim-to-real
kept, re-annotated. Naming principles: the slug names the era's
contribution in the repo's locked vocabulary; no status or temporal words;
the number carries ordering. The cascade-1 ship will tag
v3.4-trace-conditioned-draws.

Weights policy, owner-locked: the repo stays lean -- no tracked
checkpoints, no releases while the models are research iterations
(publishing a checkpoint implies it should be used). Tag annotations carry
the era's production-checkpoint sha256s (v3.3's rewritten format is the
standard: pithy prose + hash block); the hashes recorded at tag time are
what make checkpoints attachable and verifiable as GitHub release assets
at NeurIPS submission. Tracked surface: code, configs, docs, tests,
slides source, `results/` (headline layer), `explorations/*/results/`,
split rosters. Out: `data/`, `runs/`, `checkpoints/`, optuna journals.
Accepted risk, named: until submission the era weights exist on one
machine plus ordinary backups.

**Next.** v3.4 annotation at ship: prose + frontier statement ("last
version before the all-rock 23-token encoder cascade") + sha block.

## 2026-07-30 -- Real-rock metric selection: four instruments promoted, the null wins on all of them

**Q.** The zero-shot headline is one number in nats that nobody can size.
Which additional instruments earn a permanent place beside it, and what do
they say about the cascade-1 checkpoint?

**Context.** Six candidate metrics were prototyped on the 640 least-altered
eruption-v2 rocks, scored across three arms (old corpus
`joint_heads_supportmag_seed0`, traced baseline `joint_heads_regen_seed0`,
traced tuned `joint_heads_regen_tuned_seed0`), each against a climatology
null -- per-group base rates plus the mean measured composition, fit on the
train/val carve-out with one tuned concentration. The null was added
because "4.7 nats doesn't mean anything to me" was correct: a cost has no
scale without a do-nothing baseline.

**Considered.** (1) *Coverage and sharpness*: stated-80% intervals are
really 39/32/30% intervals; climatology sits near the diagonal. Rejected
from the promoted set -- calibration is the recalibrator's job downstream,
and no single concentration rescale reaches parity anyway (a global
multiplier tops out at 69% coverage at the 80% level, and opq/bi never
reach it at all, so part of the deficit is location, not width). (2)
*CRPS*: pooled 0.0929/0.0972/0.0990 against the null's 0.0820. Overlapping
marginal intervals initially read as a tie; the paired contrast excludes
zero for every pairing (tuned - old +0.0061 [+0.0035, +0.0089]). Not
promoted -- Bolin & Wallin 2023 show CRPS scales linearly with dispersion,
so it is structurally under-sensitive to the concentration changes that
differ between arms, and once the null is on the NLL figure it adds sign
without magnitude. Raw CRPS is also incomparable across groups: opq has
one of the lowest bars and the worst skill (-0.50 vs its own climatology),
while ol is the only group beating the null (+0.32). (3) *Aitchison
distance*: medians 1.40/1.47/1.54 vs the null's 1.19, all paired contrasts
excluding zero. Not promoted, but it contradicts the standing
"accuracy did not move" reading -- log-ratio geometry weights every mineral
equally, so it sees point-estimate degradation on the accessories that
Spearman/AUROC/MAE, all dominated by the large shares, cannot. (4)
*Assemblage ranking*: median rank 11 -> 9 -> 8 of 128, paired tuned-minus-old
-2.0 [-3.0, -2.0] with Jaccard +0.0104 [+0.0033, +0.0187]. The cleanest
across-arm win in the set. Not promoted because climatology ties the best
arm at rank 8 and dominates the whole top-K curve, so as a per-run headline
it reports a deficit without pointing anywhere.

**Decision.** Promoted, all four emitted by `run_real_rock_eval` into
`runs/real_rock/<tag>/` and by `scripts/view/real_rock_card.py` into
`results/<tag>/`: `cohort_card` (the single-rock card generalized to the
cohort -- chemistry violins, presence rates, and modal mineralogy on the
measured and on the model's own assemblage), `magnitude_ratio` (paired
per-rock error, the view the cohort card's marginals structurally cannot
give), `nll_distribution` (the headline split by term against the null),
and `presence_brier` with its Murphy decomposition. `climatology` and
`murphy` are production functions; the null's headline is recorded in
`metrics.json`, not only drawn.

**What they say.** All three arms lose to the null on every promoted axis.
The tuned arm: joint 24.4 against 1.5, presence 6.5 against 3.5,
composition 17.9 against -1.9. Amphibole Brier 0.505 is worse than a coin
flip (AUROC 0.539, model calls it in 19% of rocks against 65% present).
Olivine looks like the same failure and is not -- AUROC 0.841 with every
probability too small to cross any threshold, which is a scale failure a
recalibrator repairs, where amphibole has nothing underneath to rescale.
The decomposition is the actionable line: reliability 0.081 against
resolution 0.036, with the null at zero resolution. The model knows
something real about specific rocks and hands back more than it earns
through miscalibration.

Two findings the six-figure sweep produced that nothing in the existing
eval surfaced. First, a signed multiplicative bias table (tuned arm, median
per-rock ratio): fsp x1.16 and q x1.86 over-called, ol x1.00, and amp x0.75,
px x0.63, bi x0.63, opq x0.36 under. The felsic framework is over-called
and everything mafic and accessory under; opaques at x0.36 was invisible to
every absolute-error metric because opaque shares are small. Second, the
retrain moved exactly one group: quartz presence, +49 points over-called on
the old corpus to +28 on the traced arms, corroborated independently by
Brier (-0.0802 [-0.0992, -0.0595]) and CRPS (-0.0143 [-0.0242, -0.0003]).
Every other group sits within a few points across all three arms.

**Also.** `_unit_bootstrap` hands `stat_fn` positions into the `units`
array it was given; a caller scoring a subset with absolute row indices
resamples the wrong rocks and returns intervals that do not contain the
estimate. Production had one call site and it was correct; the trap is now
documented. And `tokenize_bulk_frame` never renormalizes to a common oxide
total, so gard2019 (median 97.8, raw analyses with untokenized volatiles)
and the du Bray benchmark (median 99.7, recalculated volatile-free) present
to the encoder on different scales -- filed as #56.

**Next.** Coverage is absent from the promoted set by choice; if the
recalibrator ships, its calibration claim needs that instrument back. The
promoted panels also mix aggregations (entry-weighted pooling in one,
equal-weight-per-group in another), unstated.

## 2026-07-31 -- Residual decomposition after the recalibrator fit: location is absorbed, the deficit is width no sane knob buys

**Q.** Before spending an HPO sweep on the recalibrator, what can its
functional form reach at all? The zero-shot probe behind the 2026-07-30
entry (global concentration multiplier tops out at 69% coverage; opq/bi
never reach parity under free per-group multipliers) suggested those two
groups were location-limited, in which case concentration-space knobs chase
the wrong quantity for them.

**Context.** New instrument `explorations/real_rock_fewshot/
run_residual_decomposition.py`: fit vector and matrix on the committed
train roster, then per group decompose the remaining miss on train and val
into width vs location. The oracle reparameterizes each group's Beta
marginal into mean and precision -- `k` scales precision at fixed mean,
`delta` shifts the mean logit at fixed precision -- exactly the two
directions the recalibrator's own alpha offsets entangle (a per-group
`magnitude_offset` moves mean and concentration together; the only
mean-preserving direction in the parameterization is an offset shared by
all seven groups). Grid search each knob's reachable coverage, record the
sharpness at the optimum, bootstrap over units. Run on both regen arms;
test slice untouched.

**What it measured.** (1) The zero-shot location hypothesis does not
survive the fit. Constant per-group offsets absorb most of the location
error: zero-shot medians run to 2.8x off (bi, opq under; fsp, q over);
after the matrix fit every group's median ratio is within 1.4x and mean
PIT sits near 0.5. opq -- the group the probe flagged hardest -- lands
nearly centered (median x1.2-1.3 over, PIT 0.45-0.49). (2) What remains is
under-dispersion everywhere: coverage of the nominal 80% interval is
0.45-0.75 after the matrix fit, and reaching nominal by pure widening
needs precision cuts of 6x-1000x, at which point most groups' intervals
span half the simplex or more -- coverage bought vacuously. The two groups
whose optimum stays sharp (px at k 0.12-0.17, bi at k 0.04) are exactly
the two whose width ceiling falls short of nominal (0.75-0.84). So the
residual is heavier-tailed than any Beta at the fitted mean: 25-50% of
rocks sit outside intervals already a fifth of the simplex wide. (3) The
surviving location structure is SiO2-dependent, not constant: Spearman of
signed residual vs SiO2 is bi +0.6, q and px +0.5, stable across arms and
tiers. The recalibrator's SiO2 slopes act on presence only. (4)
Recalibration nearly closes the tuned-vs-baseline zero-shot gap: val joint
NLL 31.7 vs 25.4 collapses to 2.04 vs 1.71 after the matrix fit -- the
tuned arm's 6.3-nat zero-shot penalty shrinks to 0.3. Direct input to the
open production adjudication: the regression the cross-era judge measured
is almost entirely recalibrator-absorbable.

**Decision implications, owner to adjudicate.** HPO on the fit
hyperparameters optimizes NLL and will not close a coverage gap that any
Beta-shaped correction can only close vacuously; a mean-shifting term is
not the priority either, since the fit already absorbs constant location
error. The two extensions the data argues for: heavier-tailed magnitude
structure (mixture or per-rock precision), and an SiO2 slope on the
magnitude side. Neither is scheduled here.

**Also.** The oracle's max-coverage ceiling degenerates for mid-share
groups once `k` is allowed small enough -- a diluted Beta's 80% interval
tends to (0, 1) when the group's mean share exceeds 0.1, so its ceiling
saturates at 1.0 vacuously; below 0.1 the interval collapses toward zero
instead, which is why small-share groups (bi, px) are the only ones whose
ceiling binds. The instrument therefore records median interval width at
every optimum, and the figure prints the `k` that buys each ceiling.

**Next.** Owner call on the recalibrator HPO docket item in light of the
above, and on the production adjudication now that the tuned arm's
real-rock penalty is measured as recalibrator-absorbable. The 423 censored
trace-only observations remain outside the fit objective.

## 2026-08-03 -- lepr_validation: closing disposition

**Q.** What did the LEPR harvest settle, before its directory is archived?

**Closure.** The harvest stands at 2,368 deduplicated (bulk -> assemblage)
pairs with measured (P, T, H2O) -- presence-only, no modes, no melt
fraction. Fine-tuning on it was rejected 2026-07-14 (imposed lab conditions,
no modal targets); the surviving honest role is presence-level generator
validation, unscheduled. Recorded as the study's own closure line because
its disposition was previously scattered across two 2026-07-14 entries.

## 2026-08-03 -- explorations archived to the minimal active set; closed-era runs/ flushed

**Q.** The working tree carried 18 explorations (~580MB) and 13 `runs/`
snapshot dirs (189MB) spanning closed eras -- what is the minimal informative
set to carry forward?

**Motion.** Fourteen closed studies moved to `explorations/archive/` via
`git mv` (history follows); each has its verdict journaled and a one-line
summary in `explorations/archive/README.md`. Their ten mirrored tests were
removed (git keeps them; `tests/TESTS.md` updated). Four stay active, each
for a reason beyond preference: `real_rock_fewshot` (current workbench),
`corpus_scaling` (#60 reuses its machinery), `real_mineralogy` (its
`harvest.py` is the sole builder of the gitignored benchmark inputs
`data/real_mineralogy/{samples,modes}.csv`), `real_rock_mapping` (the
benchmark's error-bar instrument).

Deleted outright: `equilibrium_multimodality/work/` (532MB of untracked
MAGEMin ladders). This engages the 2026-07-14 note that kept it for
`volcanic_proxy_definition` reruns -- that reason expired with the corpus
regen landing 2026-07-27, and the ladders live in the verified desktop
backup (`crabnet-mineral-v2_2026_08_03`) and regenerate in ~25 min. Also
flushed from `runs/` (all gitignored, all in the same backup): `imputation`,
`imputation_hpo`, `imputation_hpo_wide`, `probe`, `probe_hpo`,
`pretrain_hpo`, `summary` -- closed-era snapshots whose headline numbers
live in the scorecard/ledger/journal. Held pending the production
adjudication paperwork: `real_rock`, `heads_hpo`, `pretrain`, `comp_head`,
`cond_head`, `mode_head`. Optuna sweep state was verified to live in
`data/optuna/journals/`, untouched by any of this.

Companion edits so nothing dangles: `GATE_RESULTS_ROOT` in
`cnmineral/eval/imputation.py` repointed to the archive (it was a silent
soft-fail read), four cross-exploration imports rewritten to
`explorations.archive.*`, doc links repointed (scorecard, corpus_regen_spec,
pretrain_applications, ADR 004, two build-script docstrings), and the
`.gitignore` scratch pattern deepened to `explorations/**/work/`.

## 2026-08-03 -- Three-way checkpoint contrast: the old corpus wins after identical recalibration; test slice spent

**Q.** The residual decomposition recalibrated only the two regen
checkpoints; the old-corpus production model had never been corrected under
the same protocol. Does the traced corpus's real-rock penalty survive
correction, or was it all calibration?

**Instruments.** `explorations/real_rock_fewshot/run_adapted_figures.py`
(the four root-results canvases redrawn post-fit, val slice),
`run_checkpoint_contrast.py` (paired unit-clustered contrasts, val + the
one sealed test scoring), and `run_curve.py --eval-split val` (few-shot
curves for all three checkpoints with the test slice left alone). Protocol
everywhere: matrix tier fit on the full 368-rock train roster, identical
across checkpoints; the regen numbers reproduce the residual decomposition
bit-for-bit.

**Result.** Post-correction val joint NLL 1.376 / 1.707 / 2.035 for
supportmag / regen / regen-tuned -- but val alone cannot separate old from
regen (paired delta -0.33 [-0.76, +0.05]). The sealed test slice can:
**0.478 / 0.769 / 1.371**, every pairwise contrast excluding zero
(old-vs-regen -0.29 [-0.54, -0.07]; old-vs-tuned -0.89 [-1.51, -0.24];
regen-vs-tuned -0.60 [-1.12, -0.10]). Few-shot curves: the old-corpus
model wins at every labelled budget (25 rocks: median joint 3.86 / 5.97 /
6.84), no crossover, so there is no few-shot regime where the traced
corpus pays. Reading: the traced corpus's zero-shot penalty is mostly
calibration (11.8 of 12.1 nats absorbed) but a real residual survives, and
it points the wrong way for "sharper supervision is better." Hypothesized
mechanism (not isolated -- the regen changed sampler, draws, and roster
together): narrowed draws deleted rock-dependent hedging that broad-draw
marginalization provided; constant corrections cannot restore per-rock
width, which is the same missing structure the decomposition's heavy-tail
finding and the measured amp bimodality point at. What the traced corpus
did buy stands: quartz presence and assemblage ranking.

**Consequence.** The real-rock test slice is SPENT. Future recalibrator
work selects and validates on train/val; any later test number is a
disclosed second look.

## 2026-08-03 -- Owner decisions: production, cascade 2, mixture, recalibrator order

Recorded with rationale in the 2026-08-03 recap doc (retired 2026-09-03; the effective-lambda argument is in heads.md's loss-weighting section, the measurements in the entry above); terse ledger here.

1. **Production remains `joint_heads_supportmag_seed0`.** The provisional
   2026-07-29 adoption of trial 22 is reversed -- it loses to its own
   baseline on every paired real-rock metric, and both regen checkpoints
   lose to the old-corpus model after identical correction. No v3.4 tag
   rushed; historical tags stay as-is.
2. **Cascade 2 = all-rock 23-token encoder x the OLD broad-draw corpus.**
   Baseline-first at current production dials (isolates the encoder
   effect), then the sweep under the corrected criterion: effective
   lambda reported and searched at or below 1 (magnitude_weight <= 0.048),
   sampled-support NLL in the tiebreak. Judged on the real-rock benchmark.
   Supersedes the earlier plan to run cascade 2 on the traced corpus.
3. **ModeHead Dirichlet mixture** attempted after cascade 2, on the old
   corpus with whichever encoder wins; kept only if better.
4. **Recalibrator upgrades wait on the mixture verdict**; the SiO2
   magnitude-side term and constant offsets are needed regardless.
5. **#60 filed**: corpus scaling re-measured on the real-rock axis, after
   cascade 2; decides whether a larger broad-draw regen is generated.

## 2026-08-03 -- Cascade-2 baseline: the encoder swap alone is a real-rock tie; both encoders go to corrected sweeps

**Q.** Does the all-rock 23-token encoder transfer better than the
igneous-only 22-token one, with everything else pinned to production?

**Context.** Baseline per decision 2: heads retrained on the old broad-draw
corpus at the support-conditioned-magnitude production dials with only
`encoder_checkpoint` swapped to `pretrain_allrock_redox_seed0`
(`joint_heads_allrock_seed0`). The redox slot stays padded throughout, so
heads-train, eval, and deployment share one regime and the swap measures
the representation change alone; the fed-where-measured arm is filed as
#61. The corrected sweep criterion also landed in code this session:
`magnitude_weight` search bounded at or below 0.048 (effective lambda <= 1)
in the sweep configs, `dirichlet_nll_sampled_support` as a band-veto in
`tiebreak_best` (attr-less pre-criterion trials exempt), and
`effective_lambda` stamped on every trial.

**Result.** Corpus val (valtest128-broad, 218 bulks): teacher-forced joint
NLL -2.6157 vs production's -2.4565 (better ~0.16 nats) while
sampled-support dirichlet regressed -1.5884 -> -1.3074 (~0.28 nats).
Real-rock zero-shot (eruption-v2, 640 rocks): 11.412 vs 10.406, +1.0 nats,
nearly all dirichlet width. Under the identical matrix recalibration (fit
on the 368-rock train roster) the paired unit-clustered val contrast is
+0.078 [-0.396, +0.407] -- a statistical tie; the test slice was not
touched. Reading: the encoder swap alone is roughly neutral where it
counts, it carries the all-rock encoder's independent merits at no measured
real-rock cost, and the corpus/real-rock divergence appeared again -- with
the sampled-support axis agreeing with the rocks, the first live
confirmation that the veto metric tracks the transfer direction while
teacher-forced NLL points the other way.

**Decision (owner).** Sweep BOTH encoders under the corrected criterion at
matched budget: identical search space (ModeHead architecture re-opened per
#54), instrument, sampler seed, and 48-trial ceiling --
`configs/heads/joint_heads_hpo.yaml` (all-rock) and
`configs/heads/joint_heads_igneous_hpo.yaml` (igneous-only), separate
journals. Each sweep's tiebreak winner retrains at 120 epochs and the
paired real-rock protocol re-judges the encoders at their tuned best.

**Also.** `run_checkpoint_contrast.py` writes a fixed output name, so
today's two-checkpoint run overwrote the tracked three-way JSON that held
the sealed-test record; restored from git history as
`checkpoint_contrast_matrix_three_way.json` alongside the current one.
Ledger entries appended (`allrock-baseline` in both ledgers, measured on
production's own instrument); trajectory regenerated.

## 2026-08-04 -- Cascade-2 all-rock arm complete: corrected-criterion winner ties production on recalibrated val at no measured cost

**Q.** At the all-rock encoder's tuned best under the corrected criterion,
does the model transfer better to real rocks?

**Context.** The joint_heads_allrock_hpo sweep completed its full 48-trial
ceiling on natural termination (~18 min/trial). The corrected criterion's
first live exercise was decisive in both halves: TPE explored the interior
of the lambda <= 1 range rather than railing at the cap (tie set spans
lambda 0.13-0.83), and the sampled-support band-veto dropped 10 of an
11-trial mode tie set -- including the raw Optuna best (trial 23) -- to
pick trial 32 (effective lambda 0.59, ModeHead 512x3, dropout 0.26; the
re-opened architecture went to the largest searched trunk). Retrained at
the production 120-epoch horizon (`joint_heads_allrock_tuned_seed0`),
early-stopped at 66 with per-head bests replicating the sweep trial
exactly, confirming the 60-epoch sweep horizon was adequate.

**Result.** Corpus (valtest128-broad, val 218): best numbers on record on
both axes -- teacher-forced joint -2.6713 and sampled-support dirichlet
-1.9620 (vs -1.3074 allrock-baseline / -1.5884 production), so the
corrected criterion recovered the baseline's sampled-support regression
without giving up teacher-forced ground. Real-rock zero-shot (eruption-v2,
640): 11.638, worst of the three arms -- bernoulli improves (5.679 ->
5.506) while dirichlet width widens (5.732 -> 6.132); tempering lambda
below 1 does not by itself close the sim-to-real confidence gap, because
that gap is a property of the corpus-to-rocks shift, not the loss
weighting. Under identical matrix recalibration (fit on the 368-rock train
roster) the paired val contrast is a three-way statistical tie with the
tuned model's point estimate best -- 1.376 production / 1.298 baseline /
1.277 tuned; production minus tuned +0.099 [-0.395, +0.501], baseline
minus tuned +0.021 [-0.271, +0.237]. Reading: the corrected criterion's
pick transfers at no measured cost and did NOT lose after correction (the
zero-shot penalty was absorbable width, as predicted), but a real-rock win
claim is unproven -- val's third consecutive non-separation, and a test
read remains an owner-authorized disclosed second look.

**Decision (owner).** Production unchanged pending the encoder verdict.
The igneous twin sweep (joint_heads_igneous_hpo, matched budget and seed)
launched 2026-08-04 to run in the background -- Ctrl-C-resumable, so it
does not block the queue; the owner's prior expectation, recorded here, is
a small tuned-vs-tuned difference, with the sweep's value being measured
attribution for the adoption argument (all-rock encoder's independent
merits at no measured real-rock cost). The broader order of operations
proceeds meanwhile to the ModeHead Dirichlet mixture -- the component
aimed at the rock-dependent width that constant recalibration provably
cannot fix.

## 2026-08-04 -- ModeHead dirichlet layer (mixture) built: K=2 assemblage-level components behind a K=1-identity dial

**Q.** Build the mixture-of-Dirichlets magnitude branch -- the component
aimed at rock-dependent width -- such that dropping it costs one commit
and production is never at risk.

**Context.** Three measured evidence lines motivate it: 34% of testable
bimodal bulks stay bimodal within their dominant support (the residue
support-conditioning cannot represent in principle; amp two-sided PIT
groups 0.026 -> 0.114 when #25 landed), the recalibrator residual is
heavier-tailed than any Beta at the fitted mean, and the broad-draw
corpus's hedging value. Design settled in conversation before any code.

**Considered.** Per-phase mixtures: rejected -- on a simplex a
single-phase mixture is not a coherent object (moved amp mass must come
from somewhere), and the humps co-vary through the shared hidden (P, H2O)
path (the 51% amp<->opx switch), the same argument that locked
support-conditioned CondHead. An explorations/ pilot: rejected on the #25
precedent (a standalone pilot drops closure coupling and tuned weights;
droppability comes from the K=1 identity plus checkpoint separation, not
directory placement). A gate entropy regularizer: rejected for build 1 --
measure before medicating; gate entropy and component separation ship as
read-after engagement diagnostics instead, so collapse reads as "never
engaged", not a verdict. Per-component recalibrator corrections: rejected
-- one shared per-group correction applies to every component so
recalibration capacity stays fixed across arms.

**Decision.** Built and gated. One gate over K=2 whole-assemblage alpha
vectors on the magnitude trunk (`mode_n_components`, soft default 1);
loss is logsumexp over components (CondHead's GMM pattern moved onto the
Dirichlet); ancestral samples draw a component, so every scenario commits
to one hump. K=1 is the pre-mixture head at the parameter level -- the
build gate PASSED: both `joint_heads_supportmag_seed0` and
`joint_heads_allrock_tuned_seed0` load strictly under the new code and
reproduce their ledgered numbers digit-for-digit (-2.4565 = 4.9973 -
7.4538, sampled -1.5884; -2.6713, sampled -1.9620). Full suite 427
passed (12 new tests). Config ready:
`configs/heads/joint_heads_mixture.yaml` (allrock-tuned dials + K=2 ->
`joint_heads_allrock_mixture_seed0`). Verdict pre-registered before any
result exists: recalibrated val paired NLL against the same-dials
single-Dirichlet control is primary; an NLL tie with two-sided PIT
collapsing toward base rate, coverage toward nominal, and no per-phase
regression counts as better; a tie on both axes drops it (the K>1 code
path is deleted -- point-estimate metrics cannot judge this component,
the mixture mean stays in the valley by construction). Deferred to a
visualization pass: mixture quantiles (`share_interval` has no closed
form under a mixture), card whisker semantics, beta-overlay K-curves;
those paths skip with a printed note or refuse loudly at K>1.

**Next.** Owner trains K=2 (GPU frees when the igneous twin sweep is
Ctrl-C'd after its current trial; resumable by study name), then the
interrogation and the pre-registered verdict. A mixture kept likely moots
the recalibrator's heavy-tail extension; dropped sends recalibrator work
forward against the single-Dirichlet reality. If the igneous twin WINS
rather than ties, the mixture config's encoder + dials re-point first.

## 2026-08-04 -- Mixture trained and interrogated: corpus wins on both axes, real rocks lean against, verdict is a third path -- kept as a dial

**Q.** Does K=2 earn the pre-registered keep?

**Context.** Trained on the owner's GPU (`joint_heads_allrock_mixture_seed0`,
early stop at 58, mode best at epoch 8; the igneous sweep was Ctrl-C'd at 14
of 48 complete to free the GPU -- resumable by study name, production-dials
trial 0 leading its mode tie set outright at the pause). Engagement first,
as pre-registered: the gate is alive (entropy mean 0.6021 of a 0.6931 max,
component separation mean 0.2635), so the result is a verdict, not a
never-engaged diagnosis.

**Result.** Corpus (valtest128-broad, val 218): best numbers on record on
both axes -- teacher-forced joint -3.0819 (vs control -2.6713), sampled-
support dirichlet -2.1161 (vs -1.9620), the gain entirely in the magnitude
term (bernoulli 4.877 vs 4.860). The targeted residue collapsed: amp
two-sided PIT groups 0.120 -> 0.023, afs 0.079 -> 0.011, mu 0.087 -> 0, no
material two-sided regression anywhere (q 0.014 -> 0.024 is 2 groups of
210). No per-phase regression: magnitude MAE/r improve or hold on
essentially every phase; presence flat. Real rocks: zero-shot 13.799 vs the
control's 11.638 (eruption-v2, 640), and unlike the tuned arm's regression
last round this one is NOT absorbable width -- under identical matrix
recalibration the paired val contrast leans against the mixture, control
1.277 vs 1.642, difference -0.365 [-0.864, +0.118], located entirely in the
dirichlet term (adapted bernoulli identical across arms). The coverage
clause of the pre-registered rule proved unmeasurable at K>1 (the share-
interval instrument skips; the PIT diagnostic's `coverage` field is
data-side). The gate hedges rather than switches: weights near 50/50 on
showcase bulks, broad gate-weight distribution -- always-on hedging, which
is exactly the shape that wins corpus NLL and pays width on rocks.

**Decision (owner).** Kept as a dial -- the pre-registered binary
(promote/delete) resolved to neither. `mode_n_components` stays, K=1
remains the soft default for investigations; the corpus beta-overlay
artifact (genuinely bimodal truth with the two components on the humps)
demonstrated the mechanism captures real corpus structure worth keeping
reachable. Main-line figure semantics stay K=1; K>1
visualization remains a design conversation if ever promoted. Standing
figure preference recorded: for main-paper real-rock cards the owner
prefers the full predicted density drawn against the measured dot over
the point-and-whisker summary. No results/ or ledger entries -- the
mixture is not adopted. Interpretation of the transfer failure ran in the
same session (next entry); #62 (find more real benchmarks) filed P1 out
of the discussion.

## 2026-08-04 -- Why the mixture does not transfer: the instrument passes it, nature is one-humped at matched chemistry, and gate re-pointing only ties

**Q.** Three candidate mechanisms for the real-rock lean: the 7-group
instrument marginalizes the residue away; the real conditional lacks the
corpus's second hump; the gate is population-mis-weighted with no
recalibration lever.

**Considered / Result.** Three experiments, tmp/ scripts, one session.
(1) Instrument transplant -- population fixed, instrument changed: corpus
val rows scored through the real-rock coarsening (on-support concentration
sums, off-vocab mass dropped, sub-composition renormalized; off-vocab truth
mass mean 4%). The mixture-minus-control teacher-forced dirichlet gap goes
-0.4279 (21-phase) -> -0.2618 (7-group): 61% survives, and amp's marginal
Beta gap survives intact (-0.105 -> -0.104; amp maps to its own group).
Refuted as the primary mechanism -- had real rocks been distributed like
corpus draws, the mixture would have won the real contrast. (2) Real-data
bimodality -- 312 clean rocks with numeric nonzero amp (75 units), k-NN=30
neighborhoods in the frozen encoder latent with per-unit cap 3, 1-vs-2
component GMM BIC on log share: 38.8% prefer two components (9.6% strongly,
median dBIC -1.9), and the model-free SiO2-window check agrees (27%/18%,
median -4.3). The matched corpus reference (nearest val bulk in the same
latent space, within-bulk draws at rung 50) prefers two components in 100%
of testable bulks (73.7% strongly, median dBIC +26.3). The twist: total
log-share spread is nearly identical (median std 0.305 real vs 0.312
corpus) -- nature at matched chemistry is as uncertain as the corpus says,
but organizes that width into ONE hump. Chemistry variation inside real
neighborhoods biases toward finding humps, so the negative is a fortiori.
(3) Gate-offset probe -- one global scalar on component B's gate logit,
matrix recalibrator refit at every grid point on the train roster,
selection on train NLL: delta -1.50 (mean gate weight on B 0.58 -> 0.26)
takes val 1.642 -> 1.415 and the paired contrast to -0.138 [-0.615,
+0.281] -- a clean tie, ~60% of the penalty recovered by one parameter.
Full collapse onto either single component is worse than the partial shift
and never reaches the control: each component is individually narrower
than nature's single wide hump (they split the corpus width between them),
consistent with the recalibrator-residual under-dispersion finding.

**Reading.** The mixture learned true corpus structure; the corpus's
second hump is a property of the deliberately broad hidden-(P, H2O) prior,
not of this benchmark's population. The real penalty decomposes ~60%
population-level gate weighting (recoverable by one recalibration
parameter, to a tie -- never a win) and ~40% hump shape (components too
narrow once separated). A population that genuinely crystallizes both wet
and dry at matched bulk chemistry is where K=2 would pay -- that is #62's
search criterion.

**Next.** Locked order resumes: igneous sweep tail -> encoder verdict,
then recalibrator upgrades against the K=1 default reality (the gate-
offset term is a known cheap addition only if K>1 is ever fielded), then
#60. Scripts, figures, and the K=2 figure mockups are parked as the
backing exhibit at explorations/archive/mode_mixture/ (born-archived: the
question is settled here); the bimodality analysis doubles as #62's
candidate-benchmark screening test, reusable for any phase.

## 2026-08-05 -- Cascade-2 igneous arm complete: encoder verdict ties leaning all-rock; all-rock encoder canonical, allrock-tuned promoted to production

**Q.** At tuned best on both encoders, does the old igneous encoder hold
a real-rock edge that should block all-rock adoption?

**Context.** The igneous twin sweep (`joint_heads_igneous_hpo`, matched
budget/seed/criterion) resumed after the mixture pause and completed its
48-trial ceiling on natural termination (~20 min/trial). The corrected
criterion repeated its all-rock-arm behavior: TPE explored the
lambda <= 1 interior, the 0.03 band held 12 trials, and the
sampled-support band-veto dropped 11 -- including the raw Optuna best
(mode -7.2480, sampled-support -0.862) and the runner-up by 0.007 --
leaving trial 43 (effective lambda 0.609, ModeHead 512x3, dropout 0.36,
sampled-support -1.820, the band's best on the veto axis; lambda_balance
soft-railed at ~0.002, the balance-off finding again). Production-dials
trial 0, the sole tie-set member at the 14-trial pause, finished rank 23
of 48. The two arms' winners converged on near-identical dials (0.59 vs
0.609, both 512x3), strengthening the matched-treatment claim. Retrained
120 epochs as `joint_heads_igneous_tuned_seed0`: mode best replicated
the sweep exactly (-7.2343 at epoch 18); comp/cond kept improving to
epochs 117/114, so the run reached the cap rather than early-stopping.

**Result.** Corpus (valtest128-broad, val 218): trails allrock-tuned on
both axes (teacher-forced joint -2.5547 vs -2.6713, sampled-support
-1.8200 vs -1.9620). Zero-shot (eruption-v2, 640): 11.691 vs 11.638 --
near-identical. Under identical matrix recalibration (368-rock train
roster; val 111 rocks / 39 units): adapted joint 1.486 vs 1.277
allrock-tuned / 1.376 production; allrock-tuned minus igneous-tuned
-0.209 [-0.460, +0.090], a statistical tie leaning all-rock; supportmag
minus igneous-tuned -0.110 [-0.425, +0.129], so on the old encoder the
corrected-criterion dials tie production with the point lean negative --
the sweep's corpus gain bought no real-rock transfer (the corpus/real
divergence again, now with tuning treatment matched across encoders).
Every axis ties or leans all-rock; none leans igneous.

**Decision (owner).** The pre-registered rule fires -- a tie licenses
adoption on independent merits (redox recovers the qap probe, effective
rank 49 vs 32, ~788k-row SSL corpus): `pretrain_allrock_redox_seed0` is
the canonical encoder and `joint_heads_allrock_tuned_seed0` is promoted
to production (ledger prose amended per the regen-tuned precedent;
per-model results/ figures generated at promotion). #54 closes -- the
comp/cond mini-sweeps are waived as superseded: the planned re-tunes
re-open the full search space anyway. NeurIPS-window strategy (deadline
2026-08-29): freeze this stack and write; run #60 corpus scaling as the
background priority (D over C -- the corpus axis owns every large
measured real-rock effect, and Sim2Science is a simulator-side venue);
#55 encoder re-HPO stays open, parked past the deadline; no fixed
swap-cutoff date, the owner owns any late-swap decision. Work branches
into two sessions: one on #60, one on paper analysis (recalibrator
upgrades first, then the few-shot curve re-run on production, then the
padded-calibration eval, which gates the implicit-marginalization
claim).

**Next.** Ship-flow figures for the promoted model; deck refresh rides
the paper-analysis session; the recalibrator design conversation opens
that session.

## 2026-08-05 -- Recalibrator tailed tier: the contamination term earns the whole gain, the silica slopes measure nothing

**Q.** The 2026-07-31 decomposition prescribed two recalibrator extensions --
an SiO2 location term on the magnitude side and heavier-tailed magnitude
structure. Built as the `tailed` tier on the frozen production stack, do they
deliver on the four criteria fixed before the code was written?

**Context.** Design ratified with the owner before code: `tailed` = matrix
plus (1) pure-location silica slopes on px/bi/q -- tilt then rescale the free
block so total concentration is exactly preserved, keeping the amp pin's
letter and spirit -- and (2) a shared two-parameter precision contamination:
with fitted probability eps the forecast Dirichlet is diluted by fitted
factor k, mean-preserving by construction, identity at init via k=1.
`vector`/`matrix` stay frozen semantically so matrix-stamped verdicts remain
reproducible; eps/k/slopes are ordinary train-fit parameters (no selection on
the reporting slice beyond the standing weight-decay grid, which picked an
interior optimum at 1.0: val 3.35/2.55/3.56 at wd 0.1/1/10). Instrument:
`explorations/real_rock_fewshot/run_tailed_acceptance.py`, fit on the
368-rock train roster, reported on val 111 (test untouched).

**Result, against the pre-registered criteria.** (a) PASS, decisive: paired
unit-bootstrap val joint NLL, matrix minus tailed +0.311 [+0.058, +0.660]
(levels 1.277 -> 0.965), entirely in the Dirichlet term. (d) PASS: presence
term flat to three decimals (the hurdle factorizes; the upgrade is
magnitude-side only). (b) PARTIAL, informative: median interval widths are
flat (no vacuous widening) but stated-80% coverage barely moves (px
0.75->0.77, bi 0.68->0.70, opq 0.74->0.76, q 0.61->0.55). The NLL-optimal
contamination (eps 0.052, k 0.51) fattens the tails beyond the 10-90
quantiles rather than widening the central interval -- proper-score repair,
not stated-interval repair, consistent with the decomposition's finding that
nominal coverage is only reachable vacuously. (c) FAIL: Spearman(residual,
SiO2) is unchanged (px 0.38->0.41, bi 0.53->0.53, q 0.51->0.53; note the
matrix fit had itself already worsened q's from the 0.37 zero-shot value).
Attribution ablation (frozen-term refits, same protocol): the contamination
alone is the whole gain, +0.306 [+0.051, +0.660]; the slopes add +0.007
[-0.004, +0.016] -- statistically nothing, fitted values tiny and q's
negative against the marginal residual trend. Mechanism check: the median
rock is a wash (-0.007); the gain concentrates exactly where the design
aimed -- +0.471 mean on the 67% of val rocks with at least one group outside
the matrix interval, -0.007 on fully covered rocks, top-5 rocks carrying
roughly the entire net gain. Insurance against heavy-tail misses, purchased
at ~log(1-eps) on well-covered rocks.

**Also.** The coverage-and-sharpness panel is restored as the fifth promoted
canvas (#59) via `plot_coverage_sharpness` + `mixture_share_interval`
(bisected mixture quantiles); headline aggregation is equal weight per group,
now stated in the captions of both group-aggregating panels (#58). Internal
reference, never for the paper: climatology val joint 2.175 vs tailed 0.965
-- the recalibrated production stack now clears the bar the zero-shot arms
lost to.

**Decision.** Owner adjudication pending at the debrief: keep or excise the
silica slopes (measured at nothing; parsimony argues excise, the
decomposition's evidence argued for them). The tailed tier becomes the
headline transfer instrument going forward; rendered verdicts stay stamped
to the matrix tier that produced them.

**Next.** Few-shot curve re-run on production with vector/matrix/tailed (mlp
off the curve per #52); then the padded-redox calibration eval; then
provenance swaps + deck.

## 2026-08-06 -- Silica slopes excised: tailed = matrix + precision contamination

**Decision (owner).** The silica location term measured +0.007 [-0.004,
+0.016] nats against the full tailed fit -- statistically nothing -- so it is
removed rather than shipped as three dead parameters. The canonical
recalibration layer going forward is the `tailed` tier = matrix scaling plus
the two-parameter precision contamination. Re-measured after excision on the
production stack (same instrument, same protocol): val joint 1.277 -> 0.972,
matrix minus tailed +0.304 [+0.049, +0.659], all in the Dirichlet term,
presence flat; fitted eps 0.052, k 0.507; coverage/width table unchanged
within noise. The per-group Spearman(residual, SiO2) diagnostic stays in the
acceptance JSON; the silica hypothesis (from the 2026-07-31 decomposition) is
recorded as tested and refuted at the joint-likelihood level -- the marginal
correlation is real, but a location term cannot monetize it against the
shared-simplex cost.

**Next.** Few-shot curve re-run on production (vector/matrix/tailed),
owner-run; then padded-redox calibration; then provenance swaps + deck.

## 2026-08-06 -- Few-shot curve re-run on production: #52's shape holds, the contamination needs claims history

**Q.** Does the labelled-rock learning curve keep its shape on the frozen
production stack, and does the tailed tier's tail insurance pay at small
labelled budgets?

**Context.** `run_curve.py --select-wd` on
`joint_heads_allrock_tuned_seed0`, val-scored per the post-spend convention,
#52 conventions otherwise (whole-unit draws, budgets 25-400, 10 draws each,
paired across tiers); curve tiers vector/matrix/tailed, mlp off per the #52
finding. Weight decay re-selected on this checkpoint's val: vector 0.1,
matrix 1.0, tailed 1.0 -- unchanged from the recorded constants. Owner-run.

**Result.** Zero-shot magnitude 7.63 nats; full-train tailed reaches -1.92
(cross-instrument consistent with the acceptance read's -1.924). The #52
headline survives, slightly stronger: ~26 rocks recover ~77% of the 9.6-nat
magnitude gain, steeply diminishing after. Tier ordering: matrix and tailed
are indistinguishable at ~26 rocks (paired per-draw delta -0.001) -- the
contamination's two parameters are fit to rare interval misses and 25 rocks
carry too few to identify them -- then tailed wins 10/10 paired draws at
every budget from 50 up with a monotonically growing margin (-0.002 at 50,
-0.11 at 100, -0.29 at 200, -0.30 at full train). Vector lags from 50 rocks
on and never goes negative (+0.2 to +0.6 across budgets). At 25 rocks the
10-90 draw band spans ~2.5 nats, so which eruptions were drawn dominates
which tier was fit.

**Next.** Padded-redox calibration eval (gates the implicit-marginalization
claim); then provenance swaps + deck.

## 2026-08-06 -- #60 corpus scaling on the real-rock axis: flat, not data-limited; broad-draw regen dead; background thread pivots to real data

**Q.** Does #27's corpus-side scaling (0.25-0.3 nats/doubling, no
saturation) survive re-measurement with the real-rock benchmark as the
y-axis -- and does a larger broad-draw regen pay?

**Context.** The #27 ladder retrained on the frozen production stack
(all-rock encoder, trial-32 dials, old broad-draw corpus, 120 epochs):
nested md5 subsets 625/1250/2500/5000 under new `corpus_scaling_allrock_`
tags, n10000 = the production checkpoint itself, plus a seed-1 repeat at
2500 to price training noise. Each point scored corpus-side
(valtest128-broad val), zero-shot (eruption-v2, 640) and recalibrated on
the val slice (fit on the 368-rock train roster; test untouched).
Statistics are paired and unit-clustered: all pairwise contrasts plus
per-rock slopes of adapted NLL on log2(bulks) -- levels on a 111-rock /
39-unit slice carry more unit noise than per-doubling effects. Scored
under the matrix tier, then re-scored under the tailed tier the paper
session shipped the same morning (`run_realrock.py --tier`; tier-suffixed
results files, zero-shot shared).

**Result.** Corpus-side replicates #27 at the tuned dials: -1.119 ->
-2.671, monotone, no saturation. The real-rock axis diverges. Zero-shot
*worsens* with corpus size (7.999 at n625 vs 11.638 at n10000), +3.1 nats
almost entirely Dirichlet width while presence stays flat -- more data of
the same kind sharpens confidence real rocks do not support. Adapted val
under matrix leans harmful (slope +0.114 [-0.043, +0.333] nats/doubling;
every contrast a tie; seed noise +0.053 [-0.244, +0.375]). Under the
tailed tier the penalty mostly vanishes -- slope +0.031 [-0.081, +0.176],
top-of-ladder -0.013 [-0.236, +0.203], point spread 0.54 -> 0.21 nats --
so the harmful lean was recoverable overconfidence: the width mechanism
confirmed by intervention, and the traced-regen signature reproduced by a
different knob (corpus growth instead of draw narrowing). Accuracy is
insensitive to corpus size across the full 16x range (share MAE
0.093-0.099, adapted presence NLL 2.87-2.91). Production point
cross-checks both sessions digit-for-digit (11.638 / 1.277 / 0.972).

**Decision (owner).** No larger broad-draw regen: the real-rock axis is
flat within ~0.1-0.2 nats/doubling under the canonical instrument, and
the sim-to-real gap is not a data-quantity problem. #60 closes measured;
the paper's scaling claim extends from "corpus-side measured, real-rock
open" to measured-on-both-axes with a dose-response divergence on the
same five models. The freed background thread pivots: the next session
finds new real evaluation data and re-audits the existing benchmark chunk
by chunk (source, extraction path, assumptions, notes per chunk) --
every transfer verdict currently rests on one dataset, fit and scored on
slices of itself, and the tailed tier's heavy-tail structure may partly
encode its visual-estimate measurement style. New-benchmark numbers land
as new instruments, provenance-scoped, never re-litigating frozen
verdicts.

**Next.** Real-data session opens with the audit + harvest design; paper
session still owes the padded-redox calibration eval, provenance swaps,
deck refresh.

## 2026-08-06 -- Fe provenance tiers audited: `derived_partial` is mostly class-assumed, and the missingness tracks publication era, not composition

**Q.** The padded-redox eval needs to know which corpus bulks carry a real
per-rock Fe3+/total-Fe value. Where does the redox information actually live,
and does each stamped tier mean what its name says?

**Context.** Tabulated all 10,500 production-corpus bulks (10,000 train, 221
val, 278 test, 1 unassigned) against `split_fe`'s branch structure and
gard2019's source metadata: publication year, `data_source`, `qap_name`,
SiO2. Read-only.

**Result.** Tier counts: measured 3070, derived_partial 3095, assumed_class
4335 (train 2915/2959/4126, val 75/56/90, test 80/79/119). The middle tier
does not mean what its name says. Only 81 of its 3095 bulks come from the
documented "one of feo/fe2o3 plus a `_tot`, derive the remainder" branch and
carry a real per-rock ratio; the other ~3014 reported a lone `feo` (1742) or
lone `fe2o3` (1259) with no total, which `split_fe` treats as total iron and
splits with the same Middlemost class lookup `assumed_class` gets. Real
accounting: 3151 of 10,500 bulks (30%) carry per-rock redox, 7349 (70%) carry
a number that is a deterministic function of `qap_name`. Visible in the values
themselves -- 3029 distinct ratios across 3070 measured bulks, 36 across 4335
assumed_class. The SSL token's populated condition (raw `feo` and `fe2o3` both
> 0) coincides with the `measured` tier exactly, verified by join: no bulk is
populated for the encoder but class-assumed for the generator, or the reverse.

Where the measured ones come from is publication era, overwhelmingly.
Pre-1970 analyses are 97% measured, 1970s 88%, 1980s 53%, 1990s 32%, 2000s
18%, 2010+ 10% (year missing for 38% of bulks). Ferrous titration was routine
before XRF and ICP; the compilations inherit whatever the source paper
published. By database, georoc supplies most of the measured mass (1782) at
29%, the small regional compilations run high (ozchem 79%, geoatlas 60%,
earthchem-usgs 49%), and earthchem navdat/petdb contributes zero measured
bulks. By rock class the rate is flat: 26-48% across the top twelve
`qap_name` values, median SiO2 52.9 measured vs 53.8 assumed_class.

That last part revises a standing phrasing. The #46 exploration named "the
historical wet-chemistry mafic bias" as the confound it controlled for within
supergroups; the era half is emphatically right, the mafic half is not -- the
tiers are compositionally near-identical. Whether a bulk has measured
speciation is mostly a fact about when and where its analysis was published,
which is not something a chemical analysis exposes. That makes the padded
state less likely to act as a provenance indicator than assumed, and turns a
worry into a directly testable question.

**Correction of record.** `magemin_oxides.py`'s tier docstring described only
the 81-bulk branch; corrected in this change to name both paths and say which
one dominates. The 2026-05-12 entry's T2c line has the same gap and stands as
written -- the log is append-only and this entry is the correction. #46's
headline contrast is unaffected: it compared `measured` against
`assumed_class` by name, so the ambiguous tier never entered the number.

**Next.** Padded-redox eval design, populated arm scoped to the `measured`
tier per the owner's confirm (75 val bulks).

## 2026-08-06 -- Padded redox does not marginalize: it is a low default, and the blank itself is a provenance signal

**Q.** The docs describe a padded Fe-redox token as marginalizing implicitly
over the unknown ratio. Padding is the deployment regime and not by choice, so
does the padded state actually behave like an average over plausible redox, or
like something else?

**Context.** Three facts found while verifying the setup reshaped the
question before any code. The production heads train, validate and deploy at
22 tokens -- the pad-dropout knob left open on the cascade-2 handoff was never
turned, it was resolved by omission, so the heads have never seen the slot
populated. The eruption-v2 benchmark cannot populate it at all: `samples.csv`
carries only `feo_tot` / `fe2o3_tot`, which are total-iron reportings, so 0 of
640 rocks qualify. Consequently a forecast-level padded-vs-populated contrast
on the frozen heads would confound "the encoder fails to marginalize" with
"the heads met an input distribution they never saw," and the well-posed
question lives at the encoder, which did see both regimes (24.1% populated in
pretraining, no dropout). Measured in `explorations/redox_padding`, label-free
throughout, on the 3070 `measured`-tier bulks (the tier that coincides exactly
with the tokenizer's populated condition) and the 787,721-row SSL corpus. The
owner scoped this to the descriptive measurements; pricing what padding costs
on the labels, and any populated-arm retrain, are parked.

**Result.** Four readings, and the claim fails on three of them.

Padding is exactly a no-op: a padded 23-token input reproduces the 22-token
latent to 0.0 absolute difference on the production weights. Nothing in the
token path is position-aware, `src == 0` masks the slot out of attention, and
the pool divides by the non-padding count. The production config's one-regime
assumption is correct and is now pinned by a unit test rather than an
argument.

The token is not inert -- sweeping the ratio over 0.05-0.95 moves a bulk's
latent by 26% of the median between-rock distance (p10-p90 0.20-0.33) -- but
the padded state sits off the populated family rather than inside it: 0.52 of
the sweep's own span from the nearest curve point, 0.31 after stripping the
displacement component shared by every bulk (padding drops a token from the
pool, so part of the gap is arithmetic; the shared direction is real, median
pairwise cosine 0.62). Where it lands imitates a low fixed ratio -- nearest
grid value median 0.15 raw, 0.25 offset-corrected, against a measured-tier
true median of 0.33 -- and it barely tracks the individual rock: Pearson 0.23,
Spearman 0.18, unchanged by the correction. A marginalizing state would track
the rock; a global-default state would not.

There is also little conditional to marginalize over. The imputation head
recovers the redox slot at log-space R2 0.307 against 0.766-0.944 for all 22
elements -- the petrologically expected answer, since the same composition
crystallizes under different fO2, and it means any honest conditional over
redox is wide.

And the blank is not neutral. Whether a row reported the pair is predictable
from the rest of the composition at AUROC 0.907 (raw inputs, 22 values plus
presence flags, redox excluded), and the frozen padded latent carries
essentially all of it at 0.898. The top features are trace presence flags (Th,
Yb, Nd), not compositions: measured-pair analyses are overwhelmingly older
(97% pre-1970, 10% post-2010) and older analyses ran different trace panels.
This refutes the read offered in the 2026-08-06 provenance-tier entry, that
the compositional flatness of the tiers made a provenance-indicator effect
unlikely -- flat major-element distributions did not stop the encoder, because
the signal is in which traces were run, not in what the rock is. PR-AUC (0.849
/ 0.832) reads against the test slice's 0.374 base rate, not the corpus 0.241;
the pinned roster over-represents measured rows by construction.

**Decision.** The padded slot is a distinct learned "no speciation reported"
state, not an implicit average over redox: it behaves like a low default, does
not adapt to the rock, and is legible to the encoder as a fact about the
analysis. Nothing here argues against the production configuration -- training
and deployment share one regime, that regime is forced by what real rocks
report, and the equivalence making it coherent is now tested. What goes is the
mechanism story attached to it; the doc wording is the owner's to place.

**Next.** Provenance swaps and the deck, which is where the encoder-doc
wording (still "fed where measured", against the locked populated/padded
terminology) gets corrected. Pricing probe and populated-arm retrain stay
parked; the retrain re-scores if the real-data session finds a benchmark
carrying FeO and Fe2O3.

## 2026-08-06 -- sediments do not transfer, and the encoder is not what is missing: Andrews XRD through the canonical figures

**Q.** Does the frozen stack transfer to sedimentary rocks measured a
different way, and does the all-rock encoder -- which saw sedimentary
chemistry at pretraining -- predict sediments any better than the igneous-only
one?

**Context.** The Andrews set (`dp_crab_Andrews.pkl` in the old CrabNet-Mineral
repo, reached from #24) is 605 sedimentary rocks with bulk chemistry and modal
mineralogy quantified by XRD. An independent measurement modality and an
independent rock class, so it tests #63's worry that the recalibrator's
heavy-tail structure partly models du Bray's visual-estimate style. Built in
`tmp/` as a mock-up: the source is anonymised `Rock_N` with no locality,
formation, freshness grading or identified publication, so nothing here is a
reportable benchmark number. Shaped into `RealRockBenchmark` so every shipped
plot function runs unmodified; the five canonical canvases were produced on
`joint_heads_allrock_tuned_seed0`, and the shipped du Bray figures re-ran
byte-identical afterwards.

Compatibility, measured first. 7 of 22 input tokens populate (no Ti, no Cr, no
traces; Cl and S are reported but have no token, and they are what predict the
halite and pyrite the truth contains). Truth resolves 5 of the 7 groups --
olivine and biotite appear in zero rocks -- and 43% of measured modal mass
(carbonates, evaporite, sulfide, clays) has no slot in the 21-phase vocabulary
(off-vocabulary share p5 0.151, median 0.426, p95 0.641). Chlorite, dropped
from the vocabulary at the <0.1% presence floor on igneous statistics, is in
100% of Andrews rocks. The tokenised inputs also sit on a different scale:
implied oxide total median 59.3 wt% against ~98 for gard and ~99.7 for du
Bray, because carbonate, water and chloride mass is untokenised. That is a
40% offset, not the 2% #56 was written around, and it is a confound this
mock-up does not separate from rock class.

**Result.** No transfer, and the raw score inverts without the null. Joint NLL
is *lower* on Andrews than on du Bray (7.893 vs 11.638) because sediments are
an easier forecast -- every rock is feldspar plus quartz, so climatology scores
-5.301. Against their own nulls the model is +13.194 nats on Andrews and
+10.111 on du Bray. Presence Brier 0.34 against a 0.027 null; 80% intervals
cover 50%. The model calls quartz in 23% of rocks that all contain it (-77 pp)
and olivine in 46% that contain none.

The resolution term is not evidence of anything: four of seven groups have
zero presence variance on Andrews (fsp 1.000, q 1.000, ol 0.000, bi 0.000) and
climatology's resolution is also exactly 0, so there is nothing for either to
resolve. The failure is reliability, which is the kind a correction layer can
in principle address.

The encoder contrast is the load-bearing result. Igneous-tuned against
all-rock-tuned on the same 605 rocks, paired, all-rock minus igneous:
presence **-1.156** [-1.470, -0.842], composition **+3.583** [+3.134, +4.021],
joint **+2.427** [+1.813, +3.047]. Pretraining on sedimentary chemistry made
the model measurably better at knowing which minerals are present and worse at
sizing them, net worse, with both arms 10.8 and 13.2 nats behind ignoring the
rock. The composition regression carries #60's signature -- broader training
sharpening the size distribution in a direction real rocks do not support --
and the per-group ratios agree, all-rock being more extreme in both directions
(amp x4.54 -> x8.77, fsp x0.64 -> x0.39). This is an arm contrast, not an
encoder-only ablation: each arm carries its own tuned head dials.

Per-group magnitude ratios flip sign wholesale between the two benchmarks
(du Bray then Andrews, same checkpoint): fsp x1.14 / x0.39, px x0.76 / x2.42,
amp x0.92 / x8.77, opq x0.34 / x3.39, q **x2.01 / x1.01**. A correction layer
fit on du Bray is not portable across rock class. The quartz entry is
supporting evidence for #57 from a test that could have refuted it: the
over-call vanishes exactly where the proposed mechanism cannot operate, since
a sandstone has no glass and its quartz is detrital.

Side measurement, since Andrews has no units to cluster on: on du Bray the
per-rock bootstrap is **2.08x narrower** than the unit-clustered one (135
units, 640 rocks). Andrews intervals are the per-rock kind and are captioned
with that factor.

**Framing (owner, in progress).** The sediment failure becomes structure
rather than an aside: preempt the reviewer who is unsurprised that an igneous
pipeline scored on igneous rocks, and point future work at simulating
sedimentary samples, which the workshop is about. The measured claim is
*not* "broadening the encoder changed nothing" -- it is the division of labour
above, representation breadth improving recognition and leaving quantification
untouched, which localises the gap to head supervision. Existing results that
carry the same load: the ledgered igneous-vs-all-rock du Bray contrast is a
tie (-0.209 [-0.460, +0.090]); #55 measured the all-rock encoder ~0.7-1pp
*under* the igneous one on igneous probes; #60 measured the real-rock axis
flat across 16x corpus scaling; #52 measured ~26 labelled real rocks
recovering ~77% of the magnitude gain. Open: whether the headline swaps to
igneous-tuned canon (narratively cleaner, costs re-running the recalibrator,
the #52 curve and the figures against the freeze) or keeps all-rock-tuned and
inverts the rhetoric (free, and puts the sediment-aware encoder inside the
claim rather than beside it). Also open, and #53 is the experiment for it:
the claim that the encoder is not the limitation is currently inferred, not
controlled.

**Next.** Andrews stays a `tmp/` mock-up until provenance is identified (#63
workstream 2). The vocabulary, not just the label source, is a cap on any
sedimentary extension: clays, carbonates and evaporites are low-temperature
diagenetic products outside MAGEMin's regime, so "simulate sediments" means a
different generator and a vocabulary extension, not the current pipeline
pointed at sediment bulks.

## 2026-08-06 -- canon stays all-rock; the sediment result goes in the main claim, not an appendix

**Decision (owner).** The framing left open in the Andrews entry above is
settled: **no canon swap.** The headline stack stays
`joint_heads_allrock_tuned_seed0`, the freeze holds, and nothing is re-run
against the igneous-tuned twin. Reverting to an igneous-only headline would
have bought narrative tidiness ("igneous end to end, so du Bray is
unsurprising") at the cost of re-running the recalibrator fit, the #52 curve
and the five figures three weeks from the deadline.

The rhetoric inverts instead, and lands stronger. Rather than "we were igneous
throughout, and separately here is an appendix where broadening did not help",
the claim is: *the canonical encoder already saw sedimentary chemistry, so the
representation is not the limitation, and sedimentary prediction still fails*
-- which localises the gap to head supervision by construction rather than by a
side experiment the reader has to trust. The sediment-aware encoder sits inside
the main claim. The igneous-vs-all-rock du Bray tie (-0.209 [-0.460, +0.090])
becomes a supporting line, and the Andrews arm contrast supplies the mechanism:
breadth improves recognition, cannot touch quantification.

Unchanged by this: the claim that the encoder is not the limitation is still
inferred rather than controlled, and #53 remains the experiment that would
underwrite it.

**Next.** #24 closed with the measured verdict. Andrews proposition (b) --
fitting the recalibration layer on an Andrews slice to price how much of the
failure is recoverable by rescaling -- is unrun and newly more interesting,
since the failure is reliability rather than resolution.

## 2026-08-06 -- The benchmark's own ruler, measured: point counts recovered for 70 rocks, and the two rulers disagree in opposite directions

**Q.** Every real-rock verdict is scored against modal mineralogy that was
eyeballed rather than counted for 713 of 724 rocks. How much of the measured
sim-to-real gap belongs to the model and how much to the measurement -- and is
the canonical recalibrator's heavy-tail component modelling nature or the
geologist's habit?

**Context.** `dubray_ds1099/Appendix10.xlsx`, "Results of microscope-based point
counts", has been on disk since 2026-07-01 and was never opened by the harvest:
95 rocks at 389-867 counted points, of which 70 are already in the benchmark
carrying the visual estimate from Appendix 02. The harvest now parses it and
emits those rows *beside* the visual ones, with `load_benchmark(prefer_basis=)`
choosing; the default is the visual estimate and a regression test asserts the
default load is bit-identical to the benchmark without the new rows, so nothing
already scored moved. Both arms are restricted to the five groups Appendix 10
counts, since it never looks for pyroxene or olivine, and the owner had already
scoped the headline to du Bray (MSH and Pinatubo, 11 hand-transcribed rocks worth
+0.04 nats, stop being averaged in).

**Result.** Neither ruler is uniformly better, which was not the expected shape.
On magnitude the point count wins by 3.72 nats (Dirichlet 20.749 -> 17.025;
paired median +4.55, 69% of rocks); on presence the visual estimate wins by 2.17
nats (Bernoulli 3.650 -> 5.822, 27% of rocks), because a point count has a
detection limit near one part in 600 and misses rare phases a geologist sees
while scanning -- opaques present in 64 rocks visually against 49 counted,
quartz 61 against 42. The joint nets to +1.55, a wash. Per-group medians move
where it matters: opq share halves (0.101 -> 0.048) and quartz falls
(0.163 -> 0.110), which are exactly the two ends of the standing signed-bias
table (opq under-called x0.36, q over-called x1.86) -- so a real part of that
result is the ruler. The binomial floor at 602 points is +/-0.008 to +/-0.015,
4-6x smaller than the opq disagreement, so this is method, not counting noise.
On the question that opened the thread, the tailed recalibrator's per-rock
outlier weight correlates with ruler disagreement at Spearman +0.22, weak and
carried by one rock: the contamination component is not mostly encoding
visual-estimate sloppiness. Scope is hard, and stated in the artifact: the 64
rocks are two eruptions of ash-flow tuff at silica 64-81, sitting 61 train / 2
val / 1 test, so this speaks to the recalibrator's fit roster and barely at all
to the val slice that renders verdicts.

**Considered.** Promoting the point counts to a new benchmark version and
re-scoring was on the table and was not taken: with 61 of 64 paired rocks in the
fit roster the val slice would barely move, and a version bump would put every
frozen verdict on a different instrument for a change whose net joint effect is
a wash. Moving those two units into val to put good truth on the scoring side is
the live follow-up, and it now has to price the presence-side loss too, not just
the magnitude gain.

**Decision (owner).** Four uses of the paired data were put up: always visual,
always point count, split by term (presence from the eyeball, quantity from the
count, with the geologist's uncounted sightings censored at the ~1-in-600
detection limit), or leave the scored benchmark alone and carry the pair as a
measurement error bar. The owner took the last: **the scoring rule stays one
uniform convention, the visual estimate, across all 629 rocks**, and the
disagreement becomes an appendix discussion of what would change if point counts
were used. Rationale: least disruptive, one rule for the whole dataset, and with
61 of 64 paired rocks in the fit roster the other options would pay a
re-derivation cost for a change nobody could see in the scored numbers. The
split-by-term option is the principled version and is named in the paper as
such; it only becomes live if those two eruptions are ever moved into the
scoring slice.

Point counts therefore land as a second ruler behind a default-preserving
parameter, not as eruption-v3. The benchmark composition tables are generated
(`run_composition_tables.py` -> `results/benchmark_composition.{md,json}`) rather
than hand-counted -- the invariant test caught a silica table that had been
computed on the wrong stage of the funnel and posted to #63. Three audit
findings are logged and deliberately not acted on: unit labels are raw source
strings that fragment 148 spellings into what should be 141 clusters, with one
rock leaking train into val; the harvest reads 8 of 42 petrography columns
(`Cpx_Opx` would split the lumped `px` group on 357 rocks) and 24 of 68 chemistry
columns; and nothing pins which revision of the USGS workbooks is on disk.

**Next.** Agenda points 2 and 3 of the du Bray audit -- the unread columns, then
the unit-key fix and what it does to the committed split. The split placement of
the point-counted units waits on those.

## 2026-08-06 -- amendment: the sediment failure is recoverable, and that is the stronger claim

**Q.** How much of the zero-shot sediment failure survives a correction layer
fit on real sedimentary labels? Proposition (b) from the Andrews entry above,
run "for good measure" and expected to show nothing.

**Context.** The du Bray protocol on a benchmark whose test slice has never
been spent: weight decay chosen once on val, layer fit on the 363-rock train
slice, scored on the 152-rock test slice, all three tiers.
`explorations/sediment_transfer/run_recalibrate.py`. The eruption-v2 side was
re-measured on the same checkpoint rather than quoted, because the recorded
tailed number is val-scored and the margin over a null only means something on
a matched slice.

**Result.** The layer recovers **13.595 nats** [-14.706, -12.483] and lands the
model **1.12 nats ahead of climatology**, not behind it (tailed -5.832 against
a null of -4.709; matrix -4.749, vector -0.590). On eruption-v2 val the same
layer sits 1.20 nats ahead of a climatology of 2.175. **The margins are
near-identical at comparable fit budgets** (363 vs 368 rocks). What differs is
how far the layer had to drag things: ~13.6 nats on sediments against ~10.7 on
volcanic.

This amends the flat "sediments do not transfer" reading two entries up. Zero
shot, no. Given a few hundred real labels, sedimentary prediction reaches the
same footing as volcanic. Since the recalibration layer is the cheapest
possible form of head re-supervision, this is the most direct evidence in the
project that the missing ingredient is target-domain head supervision rather
than representation -- and it is a better paper claim than the failure was.
It is also consistent with the corrected reading of the presence result: the
zero-shot failure is reliability, not resolution, and reliability is what
rescaling addresses.

Second finding, and it converges with the du Bray-side result journaled the
same day: **the tailed tier is not a du Bray artifact.** It wins on Andrews by
1.08 nats over matrix. If its contamination component were mostly modelling
visual-estimate sloppiness it should not help on XRD Rietveld data. The point
count entry reaches the same verdict from the other side (outlier weight
correlates with ruler disagreement at only Spearman +0.22). #63's suspicion
about the tailed tier is answered by two independent lines; the rest of #63
stands.

**Caveats.** Everything is within-vocabulary: 43% of Andrews truth mass has no
slot, the null is computed on the same restricted target, and no layer fixes
that. One split, one seed, per-rock intervals running ~2x narrow. The 40%
input-scale offset remains confounded with rock class. And the cross-benchmark
ratio column was computed on eruption-v2's visual basis, which the point-count
entry shows moves exactly the two groups it leans on hardest (opq and q).

**Decision.** Code preserved as `explorations/sediment_transfer/` with the
figures and JSON committed; #24 amended, #63 credited with the tailed-tier
answer, #51 carrying the appendix commitment. Andrews stays unreportable until
provenance is identified.

**Next.** Nothing owed here. If the sediment direction is picked up, the
vocabulary and the generator are the binding constraints, not the labels.

## 2026-08-06 -- Provenance swaps + deck refresh: the freeze queue closes 4/4, and the deck was wrong about the heads by 2x

**Q.** The stack froze on 2026-08-05/06 across three separate events (encoder
promotion, tailed recalibrator, #60's close) but `slides/overview.py` was last
regenerated 2026-08-04. Which permanent docs and which slides still describe a
model that no longer exists?

**Context.** Final item of the paper-freeze queue. Five files hit on the
superseded encoder tag; the deck's own note flagged its corpus-regen status
strip as a retire-or-fold candidate. Owner scoped the pass: re-stamp plus new
transfer slides, and retire the status strip into the corpora slide.

**Result.** Three of the five doc hits were legitimate and were left alone --
`scorecard.md` labels the old tag as its "prev" comparison column and already
carried the all-rock numbers, `roadmap.md` frames both generations correctly,
and `encoder.md`'s mention is stamped as superseded. Only
`pretrain_applications.md` was stale, claiming the production heads still sat
on the igneous encoder "until cascade 2" -- which had closed the day before.
`encoder.md` took the populated/padded terminology fix at both sites.

The deck was the real load: 13 of 23 slides carried stale content, and two
were substantively wrong rather than merely dated. The corpora slide named the
traced regen as production's supervised corpus when production trained on the
old broad-draw one, and the status strip still asserted `supportmag` was
production and that cascade 2 had not run. Every number was re-pulled from its
source rather than edited in place, which is how the parameter error surfaced:
the heads are **1,830,337** params, not the 895,169 the deck claimed -- ModeHead
alone is 1,346,090 (74% of the head budget) because the cascade-2 sweep widened
its trunk from 256x3 to 512x3. Heads are ~57% of the encoder, not ~28%. The
loss dials had drifted the same way: production runs magnitude_weight 0.0281
(effective 0.59) and lambda_balance 0.00308 against the deck's 0.25 and 54.5.

Two counting traps worth recording. The encoder checkpoint reports 4,514,699
parameters but the deck's 3.23M was right -- the two 640k `frac_encoder_*.pe`
tables are `register_buffer`, not parameters; trainable is 3,234,699, and the
all-rock encoder adds exactly +200 over the igneous one (one embedding row).
And the recalibrator's val levels are scored on 111 rocks, so its zero-shot
reads 13.44 where the 640-rock headline cohort reads 11.638; the two
populations are easy to mix and the new slide says so explicitly.

The deck went 23 -> 24 slides: the status strip retired into the corpora slide,
and the recalibrator gained the standard two-slide report (anatomy +
eval-semantics). It is written bespoke rather than through `head_anatomy_slide`
because that template hard-codes the latent-h input glyph and the recalibrator
reads the frozen ModeHead's outputs, not h.

**Decision.** Landed as commit `112cfca`. The `tokens.py` `REDOX_TOKEN_ID`
comment, which was the last place in the repo still asserting implicit
marginalization, is corrected to name the measured behaviour instead.

**Next.** Nothing owed on the freeze queue -- all four items are discharged.
The sealed test slice still gets exactly one disclosed read, on the final
stack, owner-authorized.

## 2026-08-06 -- du Bray's unread columns priced, and the unit-key leak measured instead of arbitrated

**Q.** The harvest reads 8 of 42 petrography and 24 of 68 chemistry columns.
Four unread ones looked promising, and the benchmark's unit key is raw source
text that fragments one eruption across several spellings. What is actually in
those columns, and does the fragmentation cost us anything?

**Context.** Workstream 1 of #63, continuing the point-count pass. The premise
throughout was that these are extras on a benchmark already harvested and
scored; nothing here could move a frozen verdict, only extend the record.

**Result.** The DS 764 / DS 1099 pamphlets define every column in a "Data
Fields" section, and neither pamphlet was on disk -- they are the authoritative
source for all four questions and answered three of them against expectation.

`Cpx_Opx` is presence plus a dominance ordering (`C`/`O`, dominant uppercase),
**not a ratio**: 357 benchmark rocks carry it, 299 naming a single pyroxene.
Splitting the lumped `px` group would change the scoring vocabulary from seven
groups to eight and make every ledger number incomparable, for resolution on
half the scored rocks. Owner kept it lumped; the cpx-vs-opx question, if it is
ever asked, goes to the 13 MSH and Pinatubo rocks whose sources name their
pyroxenes outright. `harvest.py`'s "sparsely populated" comment was wrong and
is corrected.

`ClrIndx` is *defined* as the sum of the hbl/bt/px/ol/opq abundances and is
exactly that sum in the files (745/747 DS 764 rows, 416/416 DS 1099). It is
arithmetic on columns already read, not the independent cross-check on du
Bray's visual convention it was hoped to be -- the point counts remain the only
such handle. Dead end, recorded so it is not re-opened.

`chem_src` exists in both sources, not just DS 1099: DS 764's is `Chem_Src` in
Appendix 1, a file the harvest never opens. Coverage is 713 of 717 rocks, but
all 509 DS 764 benchmark rocks trace to source 4, so the variance lives only in
the 204 DS 1099 rocks. One sentence for the paper's data section; no analysis.

`Total_I_pct` was the one that grew. It is the pre-recalculation analytical
total and includes volatiles (verified: it sits flat near 100 regardless of
volatile content, and adding volatiles would imply totals of 111). With
`Volatile_pct` it makes du Bray's volatile-free recalculation invertible per
rock, which turns #56 from an estimate into an exact test. Measured: the undo
lands the benchmark on the training corpus's convention almost precisely
(tokenized-major medians 99.63 published -> 97.74 undone, against gard2019's
97.82), the encoder is not blind to it (latent relative L2 median 0.153), and
zero-shot on 429 least-altered non-test rocks improves 12.529 -> 12.208, almost
entirely in the magnitude term. The unit-clustered interval [-0.001, +0.625]
does not clear zero and 56% of rocks improve, so it is real but second-order.
Direction is what matters: **the correction helps, so every real-rock number on
record is mildly conservative rather than flattered.**

The unit key resolved without the arbitration it appeared to need. The concern
was that raw spellings fragment eruptions (148 strings, 141 after case and
whitespace) and that Appendix 1 carries a *second* eruption labeling that
disagrees with the petrography column on 91 rocks -- adjudicating which is
authoritative was the expensive part. Instead the leak was measured directly. A
deliberately over-permissive merge rule (qualifiers and sub-facies suffixes
folded in, comma-boundary prefix match so the nineteen `Plio-Pleistocene, ...`
units stay apart) flags 15 of 111 scored val rocks as sharing an eruption with
train. Scored under the production tailed layer, the leak-free subset comes out
**better**, not worse: 0.9725 on all 111 against 0.8850 on the 96 clean ones.
That is backwards from a leak. The same pattern holds zero-shot (13.44 -> 13.88),
where leakage is impossible by construction because the heads never saw a real
rock, so the explanation is that the large well-sampled eruptions are simply
easier rocks. Granting the most permissive merge any curated table would make,
the conclusion does not move.

**Decision.** px stays lumped. `ClrIndx` and `chem_src` are closed as
documentation, not work. #56 is parked to near submission as one of two
whole-hog options -- leave it alone, or re-run all evals on the unadjusted
convention and report those -- never piecemeal, since a half-converted
benchmark is the only bad outcome; the measurement is on the issue so it is not
redone. The unit key is left exactly as committed: no rocks dropped, none
shifted, no curation table, because the leak is measurably not operating.

**Next.** Nothing owed on workstream 1. Workstream 2 of #63 (new evaluation
data, ideally plutonic) has not started. Still unpinned: which revision of the
USGS workbooks is on disk -- no checksum or download manifest, with Appendix
10's derived-percent assertion the only such guard.

## 2026-08-13 -- #53 greenlit and scoped hard: the fresh-head arm becomes a modularity demonstration, not just a control

**Q.** The paper asserts that the heads are re-supervisable application slots
over a durable encoder. What evidence carries that, and what anchors the
paper's nats now that the rock-independent reference is being dropped?

**Context.** Paper framing session (`paper/writing_journal.md`, 2026-08-13
part 2) deleted the rock-independent reference (`eval.real_rock.climatology`)
from the manuscript after measuring that its information access was never
matched to the model's: it is fit on 479 measured rocks while the zero-shot
model has seen none, and for recalibrated numbers it is fit inside the very
slice it is scored on. The asymmetry runs conservative for the recovery claim
and inflationary for the zero-shot deficit. It stays in the codebase as an
internal instrument; it leaves the paper.

Replacing it, the owner's frame is analytical chemistry's external/internal
standards. External = the measured proportions themselves, reached through
self-anchoring metrics (fold error against 1x, AUROC against 0.5, coverage
against nominal). Internal = arms of our own system sharing encoder, rocks,
splits and scoring, so everything but one variable cancels. Three internal
standards already exist on disk (zero-shot arm; the vector/matrix/tailed tier
ladder; the Broad/Sharp/SharpWeight sampling arms). The missing one isolates
whether simulator-supervised heads are worth anything at all.

**Considered.** (a) Keep the reference and add a paragraph disclosing the
asymmetry. (b) Demote it to an appendix. (c) Delete it and lean on internal
standards plus self-anchoring metrics. (d) Leave #53 unbuilt and let claim 8
stand on architectural assertion.

**Decision.** (c) plus build #53, scoped hard. Scope cuts from the issue as
filed: ONE readout rather than linear-and-MLP, and the 1-parameter temperature
arm dropped entirely, since the tier ladder already answers "is it just widen
everything" (vector 3.76 against tailed 0.97 on recalibrator val). Everything
else from the issue holds -- same budgets, same unit draws, paired per draw,
same fixed test slice, same hurdle likelihood, one shrinkage constant per arm
chosen once on val.

The framing changes, and that is what makes it worth the deadline risk. As a
control, #53 has an outcome that damages the paper (fresh head wins ->
simulator supervision was a detour). Read as a modularity demonstration, both
outcomes report in the same words: the frozen latent supports heads trained on
either label source, and the budget where the arms cross is an exchange rate --
how many measured rocks the simulated labels are worth. The asymmetry that
survives either result is already in the issue: a fresh head speaks only the 7
benchmark groups, while the simulator-trained head speaks 21 phases plus
per-phase compositions plus formation conditions, and requires no measured
labels to exist at all. State it, do not engineer it away.

**Next.** Build the scoped arm in `explorations/real_rock_fewshot/` against the
existing `run_curve.py` harness. It carries two paper claims on one axis: "25
labeled rocks buy most of the recovery" and the exchange rate. Then the
encoder-durability appendix figure (imputation R2 per element plus one
latent-organization panel), whose source currently sits under gitignored
`runs/imputation/` and needs an extract into `paper/figures/data/`.

---

## 2026-08-13 -- #53 built, and the control the design was missing: the encoder is worth measuring

**Q.** Does the simulation-trained head's supervision transfer, or is the frozen
latent doing the work? And -- the question the design had not asked -- was the
encoder needed at all?

**Context.** #53 shipped as scoped: a 7-group hurdle head (`MeasuredModeHead`,
34,839 params, 1-hidden MLP, 2-component Dirichlet mixture) trained on measured
proportions over the frozen latent, on the same budgets and the same unit draws
as the correction tiers, same hurdle objective, val slice. Then the owner asked
whether anything took raw geochemistry into a head and predicted mineralogy.
Nothing did. Every arm, old and new, sat downstream of the encoder, so the
latent-ablated arm controlled for "the labels are easy" but nothing controlled
for "the encoder was not needed." A third input variant was added: the same head
reading the 22-element assay in log space.

**Considered.** Adding the raw arm as a separate sweep and merging was rejected:
arms are paired within a draw, so a merged file cannot support a paired
contrast. `run_curve`'s loop was restructured from arm-major to budget/draw-major
instead, which makes the pairing hold by construction. The rng sequence is
unchanged, and the three pre-existing correction arms reproduce their committed
numbers bit-for-bit (150/150 rows) -- the regression check on the restructure.

**Decision / findings.** Six arms, val slice, 10 draws per budget. Read the
paired per-draw difference, never the difference of medians: at 26 rocks the two
differ by nearly a nat (+1.38 vs +2.28) because the arms' worst draws are
different draws. For the same reason the 10-90 bands overlap almost completely
while paired signs run 9-10 of 10 -- the bands understate every contrast here.

1. **The encoder earns its place, and it does so on presence.** A head given the
   latent beats one given only the assay by +0.50 to +0.67 nats at every budget
   (9-10/10 draws, CI-clear from 106 rocks up), and one given neither by +0.77
   at full roster. Presence term alone, 368 rocks: 2.77 latent / 3.36 assay /
   3.55 neither. Below ~50 rocks the assay-given head is beaten outright by the
   one given neither -- 22 raw numbers overfit where the latent does not. First
   measurement in this project of the encoder beating raw chemistry on real
   rocks.
2. **Nothing a head reads changes the amounts.** Given the measured mineral
   list, the amount term is -2.50 / -2.50 / -2.51 across the three inputs; the
   head that cannot tell rocks apart matches the one reading full chemistry.
   Unresolved whether that is petrology or the benchmark's 7-group resolution.
3. **The simulation-trained head is the only per-rock source of amount
   information anywhere in the experiment.** It wins absolute share error
   (0.0954 vs ~0.1035 for all three re-supervised heads, 10/10 draws from 210)
   while losing the likelihood badly (0.97 vs 0.27). Better centre, wrong
   confidence.
4. **The exchange rate does not exist.** The arms never cross: no budget swept
   makes the simulated labels worth more on the joint score. The reframe #53
   shipped under is retired.
5. **Claim 5 does not survive the metric switch.** "123 parameters and 25
   labeled rocks recover the gap" is a likelihood statement; on share error the
   recalibration layer sits at 0.1139 at 26 rocks against 0.1101 for no fit at
   all, clearing the zero-shot line only at ~106 rocks and winning unanimously
   from 210. Under the intuitive-metrics-lead rule the claim becomes **25 rocks
   fix the confidence, roughly 200 fix the accuracy.** Owner accepted.

**Caveat carried.** The comparison is lopsided against the simulation-trained
head: a 123-parameter rescale of its *outputs* against a 34,839-parameter head,
a ~280x mismatch, with its weights never adapted. Whether parameter-efficient
fine-tuning inside the head closes the gap is open and is being spec'd
separately.

**Next.** Paper writing resumes on the encoder-first Results ordering (owner
decision; `paper/writing_journal.md` session 17). The three re-supervised heads
are named in prose by what they are given, never by code name. Figures 1 and 2
stay the only main-text figures; this material is appendix, carried in main text
as prose plus a pointer. The figures owe a paired-difference panel, since the
spread bands understate what they are drawn to show.

---

## 2026-08-14 -- adapting the head from the inside: the 280x parameter mismatch was not the reason

**Q.** The 2026-08-13 entry closed on a caveat: #53 compared a 123-parameter
rescale of the simulation-trained head's *outputs* against a 34,839-parameter
head trained on measured labels, a ~280x mismatch, with the head's own weights
never touched. Does parameter-efficient fine-tuning inside the head close the
gap?

**Context.** A seventh arm (`adapted`) on the few-shot curve: hand-rolled LoRA,
rank 5 on all eight of `ModeHead`'s linear layers, B zero-initialized, trained
on the same measured labels through the same `recalibrator_nll`, on the same
unit draws as the other six. 33,595 trainable parameters against
`MeasuredModeHead`'s 34,839 -- matched within 4%, which is a fairness gesture
and not a capacity claim, since a low-rank perturbation of a trained
1.35M-parameter head is not the hypothesis class of a 35k-parameter head fit
from scratch. Everything else frozen: encoder, base weights, and the 21-phase
support the amount branch is conditioned on.

**Considered.**

- *The support was pinned at the frozen head's values.* `truth_forced_support`
  is a threshold plus an argmax, so it carries no gradient however the adapters
  move; letting it track them buys a discontinuous objective, not capacity. It
  is also the invariant guaranteeing every truth-present group holds a member on
  the support, so a scored group's coarsened concentration is never the exact
  zero `lgamma` cannot take -- `recalibrator_nll`'s guard covers off-mask
  entries, not that.
- *No `src/` change was needed.* The pinned support turned out to be
  recomputable from `presence` and `latent`, both already on `GroupPredictions`,
  plus the injected frozen head; it reproduces `predict`'s own `support21` on 0
  of 15,204 cells differing, across MPS and CPU. The planned schema field was
  dropped. Everything lives in `explorations/real_rock_fewshot/`.
- *The coarsening was rewritten in torch, with the noisy-OR in log space.* The
  head now runs inside the training loop, and the shipped numpy path recovers a
  nearly-absent group's probability by subtracting a near-one product from one
  -- worth up to 0.01 in group logit here, survivable in a reported number and
  not in a gradient. Cost of the switch: the arm's zero-shot origin is not
  bit-identical to the shared baseline. Measured at +2.4e-5 nats on the val mean
  (max per-rock 7.9e-4), which is negligible against the ~1-2 nat effects and
  the slice's own +/- 2.
- *Bias adapters were rejected* on the grounds that `presence_out.bias` is
  literally the vector tier's correction in 21-space, and folding it in makes
  the arm "adapted and also offset." That decision turned out to decide the
  result; see below.
- Gradient correctness was proved rather than assumed: `gradcheck` on the
  coarsening, an end-to-end finite difference on the objective itself, and a
  structural check that a member phase moves only its own group -- a transposed
  21x7 map passes `gradcheck` perfectly happily.

**Two method errors, both mine, both corrected.** The plateau probe was
specified at zero shrinkage so the plateau would be "the optimizer's, not the
regularizer's." That rationale is wrong: 33k adapters against at most 368 rocks
drive the Dirichlet density down without bound, the train loss reached -8 nats
on 29 rocks and was still falling 0.4 nats per 500 epochs at epoch 2000, and
there is no plateau to find. Shrinkage is what bounds the objective, so the
probe runs at the value the measured arms and both surviving correction tiers
already selected -- inherited, not chosen. Then the first full probe rejected
the entire learning-rate grid, because the stability and convergence tolerances
were absolute constants an order of magnitude below the noise a converged Adam
run makes forever; both were rewritten to scale off the run's own descent and
running minimum. The probe now persists its traces before anything adjudicates
them, so a criterion bug costs a reread rather than twenty minutes.

Settled at lr 1e-3, 1636 epochs (train loss only, never val), weight decay 10 --
an order of magnitude above what any other arm wanted. Over-shrinking past it
returns the frozen head exactly (val joint 5.66 -> 9.35 -> 13.4437 against a
zero-shot 13.4436), which is the identity-at-initialization contract showing up
in a fit rather than in a test.

**Decision / findings.** Median val joint NLL, zero-shot 13.44, 10 draws:

| budget (rocks) | 26 | 54 | 106 | 210 | 368 |
|---|---|---|---|---|---|
| adapted | 7.19 | 5.75 | 5.86 | 4.97 | 4.86 |
| adapted, per-budget oracle shrinkage | 7.62 | 3.11 | 3.97 | 1.74 | 1.08 |
| matrix | 3.97 | 2.62 | 2.36 | 1.37 | 1.28 |
| tailed | 3.97 | 2.59 | 2.08 | 1.11 | 0.97 |
| measured | 1.69 | 1.25 | 1.12 | 0.38 | 0.27 |

1. **The parameter mismatch was not the reason, and the caveat is discharged.**
   Paired per-draw against the measured head, the gap is 4.59 to 4.82 nats at
   every budget with the whole 95% CI above zero in 48 of 50 draws (`tailed`, by
   contrast, runs 0.70 to 1.38). Given the same rocks, the same objective and a
   matched trainable budget, adapting the head from the inside is *worse* than
   correcting its outputs with 121 parameters at four of five budgets.
2. **It holds under an advantage no other arm gets.** The oracle row picks
   shrinkage per budget on val and is then scored on val; it still loses to
   `tailed` at all five budgets. (3 draws, different draws from the sweep's, so
   the 26-rock entry is noisy -- an upper bound, not a curve.)
3. **The loss is almost entirely presence, and it is a parameterization fact.**
   At the swept shrinkage the Bernoulli term is 5.813 against `matrix`'s 2.896,
   essentially unmoved from zero-shot's 5.813: 82% of the gap. A per-group
   presence correction is an intercept, and a low-rank update with no bias term
   cannot express one cheaply -- it has to route a constant through
   `B @ A @ latent`. That intercept is exactly what the correction arms get from
   seven `presence_offset` parameters and what `MeasuredModeHead` gets from
   biases its fit deliberately exempts from shrinkage. Zero-initialized bias
   deltas (~3.1k parameters, identity at init preserved) is the targeted test;
   not run, because the oracle bound already loses to `tailed` everywhere.
4. **The amount branch goes the other way.** At a shrinkage of 1 its Dirichlet
   reaches -2.05 against `matrix`'s -1.62 and its joint beats `matrix` at the
   full roster (1.185 vs 1.277); its share MAE at 26 rocks is 0.099, the lowest
   of any arm on the curve (`tailed` 0.114, `measured` 0.111). Internal
   adaptation helps the amounts and cannot touch the calls.
5. **A protocol pinch this arm is the first to expose.** Its best shrinkage
   slides 30x across the swept range -- 10, 3, 3, 1, 0.3 -- while
   `select_weight_decay` searches sizes 25 to 200 and the curve sweeps to 400.
   For a 23- to 123-parameter correction the optimum is flat and the mismatch is
   invisible; here a near-tie at small budgets (5.76 against 5.66) picked a
   constant costing 3.7 nats at the largest. The single val-selected constant is
   kept anyway: it is what makes every arm's x-axis mean the same thing, and no
   conclusion depends on the handicap.

**Next.** Nothing further is owed on #53's caveat. `share_mae_val.png` still
plots the original four arms, so the adapted arm's best-on-curve share error at
26 rocks does not appear on the figure that would show it -- a figure decision,
not a measurement gap. Paper writing resumes; this material is appendix at most,
and the sentence it earns in main text is that the comparison was tested against
its own strongest objection and survived.

## 2026-08-15 -- the qap probe measures the label, not the latent: raw chemistry decodes rock name better, and the claim that survives is linearization

**Q.** The Results draft opened on the encoder and offered a QAPF rock-name
probe at 0.759 as evidence that the pooled latent is geologically organized.
But `qap_name` in gard2019 is a *computed* classification derived from LOI-free
major-element chemistry (Gard et al. 2019, Sect. 4.2/4.4, Table 3), and the
encoder reads those same majors. Does 0.759 show an organized latent, or a
label any faithful compression of the input recovers?

**Context.** `docs/scorecard.md` carried the qap and supergroup decodability
rows as floor-anchored only (`0.100 (random) -> *TBD*`), with a standing note
that the missing ceiling was a raw-input classifier. `runs/probe/` no longer
existed, so no artifact backed the 0.759 either.

**Considered.** Extending `run_scoring` with a raw arm (would have left the
paired comparison implicit and paid the UMAP suite); running the latent arm
through the production script and cross-referencing its JSON (lands in
gitignored `runs/`). Chose a new exploration,
`explorations/input_probe_ceiling/`, that fits both arms off one pair of index
arrays -- the sibling of the archived `input_knn_ceiling`, same target, distance
metric swapped for a classifier.

**Decision.** The axis is a measured negative and the scorecard now says so.

The protocol reproduced exactly (458,490 / 32,950 / 70 classes / chance 0.0997 /
majority 0.2203) and the latent arm returned 0.759 to three decimals, so this is
the arm the scorecard reports.

| probe | latent (256) | best raw (44) | margin |
|---|---|---|---|
| linear | 0.704 | 0.681 | +0.023 |
| MLP, fixed protocol | 0.759 | 0.787 | -0.028 |
| MLP, 30 trials each arm | 0.792 | 0.834 | -0.042 |

supergroup: linear 0.731 vs 0.659 (+0.071); MLP 0.814 vs 0.813 (tie).

1. **The latent carries no more rock-name information than the assay**, and
   matched tuning does not rescue it -- both arms gained, the raw arm gained
   more, and the deficit widened from 0.028 to 0.042. Same sign and nearly the
   same magnitude as `input_knn_ceiling` (raw beat latent by 0.025 at qap k=5),
   so two unrelated methods agree the latent sits slightly behind raw chemistry
   on qap. Consistent with that exploration's reading: the encoder is trained to
   reconstruct traces, and qap is a major-element story.
2. **What it does add is linear accessibility**, and dimensionality cannot
   explain it. A linear readout is invariant to linear reparameterization, so
   any linear map of the 44 features is bounded by the raw arm's 0.681; only the
   encoder's nonlinearity reaches 0.704. It roughly halves the premium a
   nonlinear probe earns over a linear one (qap 0.106 -> 0.055, supergroup
   0.154 -> 0.083). This settles the linear-readout leg of #19's linear-null
   bracket without running PCA at all -- PCA is a linear map, so its linear
   probe cannot clear 0.681.
3. **The missingness flag set the verdict.** In log space it is worth +0.025
   (0.774 masked, 0.749 unmasked). A raw arm denied the signal the encoder gets
   free from `src == 0` would have landed below the latent and the control would
   have returned the opposite answer. The call was made before any measurement,
   on the argument that the encoder sees missingness explicitly and roughly half
   the trace columns are unreported per rock.
4. **Standardized linear beat standardized log** (qap MLP 0.787 vs 0.774),
   refuting the exploration's own prediction that `input_knn_ceiling`'s
   linear-space win was a distance-metric artifact a per-feature-scaling
   classifier would erase.
5. **`supergroup` should not appear in the paper, on any axis.** It is our own
   9-way collapse (`cnmineral.eval.supergroup`), a substring-matching rule
   ported from v2 whose docstring names its purpose as a UMAP color palette. It
   is uncited and internally inconsistent: `syenite` -> felsic-plutonic but
   `trachyte` -> intermediate-volcanic, though trachyte is syenite's volcanic
   equivalent; `anorthosite` -> mafic-plutonic for a rock that is >90%
   plagioclase. It also happens to carry the largest margin here (+0.071),
   which is the worst possible place for an invented taxonomy.

**Next.** The result was drafted into Results and **cut for space on 2026-08-15**
-- the manuscript is a page over, and landing a conceded result honestly cost
~110 words. The surviving imputation sentence claims the latent *preserves*
chemistry rather than beating raw chemistry, which is exactly the claim this
control leaves standing; that wording is now guarded, because imputation carries
the same shape of control pointing the same way (latent-XGB 0.877 against
input-XGB 0.892). `build_numbers.py` reads the eleven probe macros from
`probe_ceiling.json` instead of hand-carrying four literals, so they are emitted
and unreferenced -- one sentence away if space appears. Still open on #19: PCA
features under a nonlinear probe, and PPCA imputation.

## 2026-08-18 -- the recalibrator's presence silica slopes measure nothing, and the read that motivated them never supported quartz

**Q.** Do the q/amp presence SiO2 slopes earn their two parameters? The B7
close read described them as one of two terms extending Eq. 6, alongside the
precision contamination -- but the contamination was measured (+0.304 [+0.049,
+0.659]) and the slopes never were.

**Context.** The slopes entered the parameterization on 2026-07-24 from a
residual-vs-SiO2 read and are unconditional across tiers, so they are 2 of the
23/121/123 counts. The magnitude location term from the same read was built,
measured at +0.007 [-0.004, +0.016] and excised (2026-08-06); the presence half
was never contrasted against its own absence. `fit_recalibrator` gained
`fit_presence_sio2`, which withholds the gradient from a parameter that
initializes at zero, so the ablation is the parameterization without the slopes
and the default path is untouched. `run_presence_slope.py` fits the tailed tier
twice on the 368-rock train roster and scores both paired on val over units.

**Considered.** Measured: **+0.0016 [+0.0009, +0.0023] nats**, entirely
presence-side; val joint 0.9740 pinned against 0.9725 free, share MAE identical
to four decimals, fitted shifts q +0.048 and amp +0.003 logit per 10 wt% SiO2.
Re-selecting the pinned arm's shrinkage returns the same 1.0. The interval
excludes zero only because both arms are deterministic at zero/one init, with no
seed noise to widen it. The abundance term is bit-identical across the arms --
presence and abundance parameters draw gradients from separate loss terms --
which is now a test.

Why it is so small: the residual read was reproduced on
`joint_heads_supportmag_seed0`, the checkpoint the term was designed from, and
on production. Spearman(presence residual, SiO2) is q **+0.02 on both**, amp
+0.33/+0.34, bi +0.38/+0.32, fsp -0.17/-0.38, opq -0.39/-0.31. Binned by silica
quintile, q's gap arches (-0.28, -0.46, -0.54, -0.48, -0.19 then; the same shape
now) -- real silica structure, but non-monotone, so a linear slope has nothing to
fit. Biotite is the one monotone category and was passed over. The module
comment's claim that presence residuals are flat outside these two is false on
both checkpoints.

**Decision.** The paper keeps the parameters and the sentence describing them,
because every reported number was fitted with them and the counts only reconcile
with them present; B7's justification clause ("the two categories whose presence
residuals vary with silica") is cut as unsupported. Removing them now would mean
re-running the acceptance read, the eleven-minute curve sweep, the inference-card
figure and `numbers.tex` in order to move four integers and nothing else --
0.0016 nats sits below the display precision of every number the paper prints.
The canonical layer drops them after submission, by flipping the default rather
than deleting the parameter, so the reported results stay reproducible from HEAD.

**Next.** Post-submission: flip `fit_presence_sio2`, re-fit, counts become
21/119/121 and `\sysResupervisedOverRecalib` 283 -> 288. The silica-residual
figure that was on the near-term docket is dropped -- built honestly it shows two
arches and a control that slopes more than either.

## 2026-08-18 -- density table aligned to the mean-of-members rule it now claims

**Q.** B8's rebuild wanted a citable source and a reproducible rule for the
volume-to-mass densities. Checked against webmineral.com (Barthelmy's
Mineralogy Database), 12 of 14 values match the measured averages within 0.1.
The two that did not conform to any statable rule: Fe-Ti oxides 5.00 against a
magnetite/ilmenite mean of 4.94, and silica phases 2.40 against a
tridymite/cristobalite mean of 2.32.

**Decision (owner).** `MINERAL_DENSITY` updates to the exact member means --
Fe-Ti oxides 5.00 -> 4.94, silica phases 2.40 -> 2.32 (members = the
polymorphs only: the label is MSH groundmass silica, 7 rocks, 0-1.1 vol%,
never co-reported with a quartz row, and quartz is its own table entry) -- so
the paper's sentence "labels with no composition to interpolate along are
assigned the mean of their members" is exactly true and self-checkable.
Reported numbers, stamped artifacts, and figures are NOT regenerated: an
executive decision, the shifts are 1.2% and 3.4% on density ratios where only
ratios enter, the mapping audit already measured density perturbations far
larger than this as immaterial, and the presence factor is provably invariant
to densities. Re-runs from HEAD will differ from stamped values below display
precision.

**Also.** B8 cites the database directly (owner Zotero entry
`barthelmy_webmineral`); the DHZ/Klein comment provenance was superseded by
the check. Olivine stays 3.50 = Fo80 interpolation (0.8 x 3.27 + 0.2 x 4.39),
now stated with the equation in B8.

## 2026-09-02 -- presence silica slopes deleted; silica leaves the recalibrator entirely

**Q.** #67 item 1: remove the q/amp presence SiO2 slopes from the canonical
recalibration layer, now that the submission no longer pins them.

**Context.** Measured 2026-08-18 at +0.0016 [+0.0009, +0.0023] nats, below the
display precision of every reported number, and kept through submission only so
the reported parameter counts (23/121/123) reconciled. The `mlp` tier also read
silica as a trunk feature; that term was never separately measured, and `mlp`
is a comparison arm, never canonical.

**Considered.** Flip the `fit_presence_sio2` default and keep the parameter (the
08-18 plan, so results stay reproducible from HEAD) vs delete. The submitted
state is reproducible from tag `sim2sci-neurips`, so HEAD does not need to
carry a dead parameter behind a flag. Keep silica for the `mlp` trunk only vs
remove it everywhere: one tier reading an argument three ignore is not worth a
signature.

**Decision.** Deleted. Silica leaves the recalibrator entirely: `forward` is now
`(logits7, alpha7[, log_w])`, the `mlp` trunk reads `2n` features, and no tier
reads silica, so the benchmark split's silica-containment constraint (#67
item 2) has lost its justification. `run_presence_slope.py`, its result JSON,
the exploration README section, and the three tests that existed only to check
the ablation go with it. Counts from the module: vector 21, matrix 119, mlp
1061, tailed 121. Deck slide updated and regenerated.

**Next.** No re-stamp yet. Items 3 and 4 also change the layer and item 2
re-splits the benchmark, so every real-rock number (acceptance read, few-shot
curve, ledger, results figures) is restamped once at the end of #67.

## 2026-09-02 -- amphibole magnitude pin re-measured on production: still unidentifiable, pin stands

**Q.** #67 item 3: does amphibole's abundance-correction sign still flip under
the plausible density range on the production checkpoint and current benchmark?
If not, un-pin it.

**Context.** The pin dates from the July mapping audit on
`joint_heads_supportmag_seed0`: amphibole's implied magnitude offset spanned
-0.033..+0.045 over 200 density draws, inside the audit's 0.10 materiality
floor, so a fitted value would read the density assumption. Since then the
production checkpoint, the benchmark cohort (640 du Bray rocks), and the
density table (08-18) all changed without the band being re-read.

**Considered.** Re-running `run_sensitivity.py` on production (nothing is fit,
so no stamped fit moves): amphibole -0.026..+0.050, straddling zero and inside
the floor; the other six groups keep sign and size (opq +0.949..+1.036, bi
+0.480..+0.598, q -0.517..-0.495, fsp -0.146..-0.121, px +0.139..+0.222, ol
+0.061..+0.143); the one magnitude sign flip is px under `all_members`, the
same mechanical case as July; amp AUROC 0.509-0.592. 312 of the 650 clean
rocks carry a numeric amphibole share, so this is not a small-sample artifact.

**Decision.** The pin stands, for the original reason: the correction is not
identifiable from this benchmark at any plausible density. Recorded in the
audit README and the model docstring; the `pin_amp_magnitude` switch stays a
real parameter because a future benchmark could resolve it.

**Next.** #67 item 4 (K=2 mixture), then item 2 (re-split) and the single
re-stamp.

## 2026-09-02 -- amphibole pin retired: the fit reproduces it, and a benchmark-specific special case does not belong in the layer

**Q.** Owner, after the re-measurement above: would it be more robust to fit
amphibole's correction and let it land at zero, rather than hardcode the pin?

**Context.** The pin encoded one benchmark's identifiability finding into the
model class. The July build's unpinned fit landed at -0.002, and the audit
itself predicted "the recalibrator should learn ~0 there". Weight decay already
shrinks an unidentifiable correction toward zero; the pin added nothing to the
numbers, only a claim that stops being true when the data change -- the
plutonic point-count sources may resolve amphibole, and #70's layer is fit on a
caller's own labels, where refusing to correct one mineral is indefensible.

**Decision.** Un-pinned and the switch deleted: `pin_amp_magnitude`, the
`magnitude_free` buffer, the two tests that checked the pin, and the deck
caption clause. `per_group_offsets` reports the raw fitted constant. The
interpretability caveat stays where it belongs, in the mapping audit README and
the model docstring: amphibole's fitted value reads the density assumption.
Numbers move by ~nothing; restamp at the end of #67 with everything else.

**Next.** #67 item 4.

## 2026-09-02 -- K=2 ModeHead mixture deleted (#67 item 4)

**Q.** Field the K=2 Dirichlet mixture on the ModeHead's abundance branch
against a benchmark where it pays, or delete it.

**Context.** Built 2026-08-04 to express the corpus's two-humped amphibole
abundance at matched chemistry; won the corpus on both axes, leaned against
the K=1 control on recalibrated real-rock val (-0.365 [-0.864, +0.118]), and
the attribution found why: real amphibole populations at matched chemistry
have one hump, the second being a property of sampling both wet and dry
conditions for every composition. The submission's ModeHead is K=1; the
2-component mixture the paper describes is the tailed recalibration tier's
contamination, a different object. The K>1 head path has been unused since.

**Considered.** Fielding it against the incoming plutonic benchmark. Rejected:
the failure is structural (any one locality crystallized under one set of
conditions), and a rung-mismatched plutonic benchmark would confound the read.

**Decision.** Deleted: the head's `n_components`, gate, and `magnitude_mixture`;
the `mode_n_components` config plumbing and `mode_log_w` loss plumbing; the
gate-aware sampler branch; the head-K branches and skip-notes in the mode and
real-rock interrogations; `joint_heads_mixture.yaml`; the mixture checkpoint;
six tests that checked K>1 head behaviour. The mixture-shaped scoring
contract (`mixture_dirichlet_nll`, `mixture_conditional_mean`,
`mixture_share_interval`) stays because the tailed tier and the measured heads
emit mixtures. Archived exploration `explorations/archive/mode_mixture/` is
the record. Production is K=1 throughout, so no number moves.

**Next.** #67 item 2 (re-split without the silica constraint) and the single
re-stamp of every real-rock number.

## 2026-09-02 -- benchmark split rebuilt without the silica-containment rule; unit spellings canonicalized (#67 item 2)

**Q.** #67 item 2: rebuild the real-rock roster selecting on the matching
criterion alone, now that no recalibrator tier reads silica. Which of the other
roster-moving changes ride along?

**Context.** The rule rejected any candidate whose scored test slice held a rock
outside train's silica range -- 292 of 400 candidates -- and its only
justification was the recalibrator's SiO2 terms, which item 1 deleted. A
read-only re-run of the builder's own loop reproduced the committed roster
bit-for-bit and showed the rule did almost nothing for balance: dropping it
moved the winning candidate's score from 0.0794 to 0.0792 while moving 129 of
724 rocks, because it changes which candidate wins, not how good the winner is.
Two other pending changes move the roster. The #63 unit-key fragmentation:
147 spellings collapse to 140 on whitespace and case, six pairs inside one
subset and one straddling (`FractionTuff`, 1 val rock against 36 train); and
any change to the unit set reshuffles the whole roster because the candidate
sequence diverges, so that fix is never free later. And #56 whole-hog.

**Considered.** #56: the undo factor exists for 573 of 640 scored rocks; the 6
MSH and 5 Pinatubo rocks -- the entire precision tier -- and 67 du Bray rocks
carry no raw total, and Pinatubo's Table 7 is normalized to 100 by
construction, so a full conversion is impossible and a partial one is the
piecemeal state ruled out on 2026-08-06. The silica-quantile term in
`split_balance` shared the rule's stated rationale but has a model-free one:
it is the benchmark's counterpart of the corpus's `qap_name` stratification,
and a slice concentrated at one end of the silica trend scores a different
task; dropping it would leave no chemistry stratification at all. Weight decay
for the restamp: the protocol defines the constant as a function of the val
slice, so it is re-selected on the new one.

**Decision (owner).** All four as recommended. `unit_grouped_split` keeps only
the `split_balance` argmax. `canonical_units` in the loader maps whitespace /
case variants to their most common spelling (`?` qualifiers stay distinct; the
`Strat_Name` disagreement stays unrepaired). Roster rebuilt at seed 0. At the
July default of 400 candidates the best draw balanced worse than the old
roster (0.090 against 0.079, val's feldspar base rate 6.7 pp off the
population), because the smaller unit set changes the candidate sequence and
the best-of-400 is itself a sample; the criterion reads truth marginals only,
so searching it harder cannot favour the model, and the default is now 2000
(`SPLIT_CANDIDATES`), which lands at 0.078 (rate 0.050, silica 0.028).
Scored 372 train (51 units) / 106 val (43) / 162 test (46); the 90.2 wt%
rhyolite now sits in train. #56 closed with the benchmark as published; the
no-renormalization convention is recorded in CLAUDE.md's settled decisions
and in `tokenize_bulk_frame`'s docstring as a check every new source must
pass. Test 29 deleted; a loader test for the canonical key added and the
shipped-benchmark invariant extended.

**Next.** The single restamp of every slice-dependent real-rock number on the
new roster: weight-decay re-selection, tailed acceptance read, few-shot curve,
results figures, deck counts, README. Zero-shot ledger entries do not move,
being scored on all 640 rocks.

## 2026-09-03 -- the tailed tier's margin was the amphibole pin: matrix returns to canonical

**Q.** The restamp's first read on the rebuilt roster scored matrix and tailed
identically (val joint 0.829 / 0.828) with the contamination term sitting at
its initialization (eps 0.047, k 0.99). The roster, or yesterday's layer
changes?

**Context.** Bisected on the old roster with the production checkpoint, the
same 368 train / 111 val rocks throughout, code checked out per commit:

    before item 1                matrix 1.270   tailed 0.967   k 0.51
    after item 1 (silica out)    matrix 1.271   tailed 0.968   k 0.51
    after item 3 (amp unpinned)  matrix 0.688   tailed 0.688   k 0.99

The first two rows reproduce the 08-05 record (the residual 0.005, and
zero-shot 13.444 -> 13.394, is the 08-18 density-table change). The third is
the whole effect: freeing amphibole's magnitude correction recovers 0.58 nats
for matrix on identical rocks, after which the contamination term has nothing
left to fit. The mechanism is that the per-group magnitude offsets are
relative, not independent: the fit widens every group's Dirichlet by scaling
alphas down (vector-tier offsets -0.5 to -2.1), and a zero pinned inside that
vector held amphibole at its zero-shot sharpness while everything around it
widened -- inflating amphibole's implied share and leaving exactly the heavy
tail the contamination term was then fitted against (on val the free fit takes
the median amphibole alpha 4.5 -> 2.0 with the rest). The pin applied through
the mixing matrix too, so no tier could route around it. The mapping audit's
band was a statement about that one offset's location in isolation, and it
was right about that; it said nothing about the pin's effect on the rest of
the vector.

**Overturned.** The 2026-08-05 verdict that the contamination term earns the
whole tailed gain (it earned the pin's cost back), and the item 3 entry's
"numbers move by ~nothing" (they move by 0.58 nats on val). The paper stands
as submitted at the tag; this goes on the camera-ready reconcile queue.

**Decision (owner).** Matrix (119 parameters) returns to canonical. Tailed is
deleted as its own step after this restamp lands, by the rule that deleted the
K=2 mixture: an exact tie, parameters at initialization, the record kept here
and at the tag. The restamp records the tie with tailed still present.

**Next.** Finish the restamp (curve with re-selected weight decay, encoder
pair contrast, trajectory figure, deck, READMEs), then the tailed deletion.

## 2026-09-03 -- restamp on the rebuilt roster (#67 item 2 closes)

**Q.** The single restamp of every slice-dependent real-rock number, on the
rebuilt roster (372 / 106 / 162 scored) and the layer without silica or the
amphibole pin, tailed still present so the tie is on record.

**Decision.** Weight decay re-selected per arm on the new val slice: vector
0.1 -> 0.01, measured_ablated 1 -> 10, adapted 10 -> 1, the rest unchanged;
the constants are recorded in the fewshot scripts. Acceptance read, val joint:
zero-shot 11.29, vector 1.42, matrix 0.83, tailed 0.83 (contrast +0.000).
Encoder pair under identical matrix recalibration, val: allrock-tuned 0.829,
supportmag 0.957, igneous-tuned 0.946; allrock-tuned minus igneous-tuned
-0.118 [-0.282, +0.024], supportmag minus allrock-tuned +0.128 [-0.028,
+0.289] -- ties leaning allrock-tuned, as before. Curve (val, ten draws): at
full train matrix 0.83, measured 0.61, adapted 1.27; matrix wins share error
against every re-supervised arm (0.102 vs 0.117-0.118, 9-10/10 at every
budget) and now clears zero-shot share error at 25 rocks; the latent head's
presence margin over the assay head holds from 50 rocks (+0.37 to +0.51,
8-10/10) but not at 25; the adapted arm at shrinkage 1 moves presence (6.01 ->
3.52) and has the best share error from 200 rocks, trailing matrix by +0.44
joint at full train, all of it presence. The three-way regen contrast JSON is
left as measured on the old roster: re-running it would either drop its test
column or spend the new test slice, which nothing has scored. The oracle
shrinkage read and the plateau probe were not re-run. Results figures, runs/
metrics and the trajectory figure regenerated; deck regeneration folds into
the tailed deletion, whose slides it rewrites.

## 2026-09-03 -- tailed tier deleted; matrix is the recalibrator

**Q.** Delete the tailed tier now that it ties matrix exactly with its two
parameters at initialization, or keep it as a free option.

**Decision (owner).** Deleted, by the rule that deleted the K=2 mixture: an
exact tie, parameters at initialization, the record kept here and at the
submission tag. Gone: the tier, `CONTAM_LOGIT_BIAS` and the contamination
branch in `RealRockRecalibrator`; the two tailed tests; the arm on the curve
(`matrix` is now the rung the figures name); `run_tailed_acceptance.py`, which
becomes `run_acceptance.py` (val levels, the vector-vs-matrix contrast,
coverage and width; `acceptance.json`, `acceptance_coverage_sharpness.png`);
the point-count contrast's contamination-responsibility read (Spearman +0.22,
2026-08-06, stands as measured); the tier from the unit-leakage audit, the
sediment scripts and the corpus-scaling reader; the `precision_contamination`
technique entry. Recorded results measured under the tier (corpus scaling's
tailed ladder, the sediment margin, the +0.22) stay in their READMEs marked as
such. The `MeasuredModeHead` keeps `n_components=2` so its recorded arms
reproduce; the default now has no canonical-tier reason behind it.

**Acceptance read, matrix canonical (val, 106 rocks / 39 units, fit on 372):**
zero-shot 11.293, vector 1.417, matrix 0.829; vector minus matrix +0.588
[+0.248, +0.943] joint, +0.397 Dirichlet, +0.191 Bernoulli -- the coupling
pays on both terms. 80% coverage per group 0.19-0.55 zero-shot -> 0.64-0.89
matrix at median width 0.20-0.31. Curve re-run without the arm on the recorded
constants; matrix recovers 49% of its 7.4-nat magnitude gain at 25 rocks and
86% at 50, ties vector on paired draws through 50 rocks and wins 10/10 from
100 up. Deck's two recalibrator slides rewritten around matrix and
regenerated.

**Left alone, flagged.** `scripts/build/build_numbers.py` still reads
`tailed_acceptance.json` and the tailed curve rows: it is the submitted
paper's number builder and belongs to the camera-ready reconcile, not to this
deletion.

## 2026-09-03 -- number builder moved to HEAD's artifacts; the reconcile list exists; the leak control reversed

**Q.** Keep the paper's number builder describing the submission, or move it to
HEAD now that the tailed artifacts it read are gone?

**Decision (owner).** Move it: the submitted state is frozen four times over
(tag, mirror, backup, Overleaf) and the builder's job is to describe HEAD.
`build_numbers.py` now reads `acceptance.json` (matrix canonical), the matrix
rows of the curve, the matrix entries of the sediment and unit-leakage
results and the matrix corpus ladder; the tailed macros are gone and
`rrCoupling*` (vector minus matrix) arrive; the re-supervised fits get one
shrinkage macro each since they no longer share a constant. Against the
submitted macro file (kept beside the paper as `numbers_submitted.tex`): 341
unchanged, 102 changed, 17 removed, 12 added. The list is
`docs/camera_ready_reconcile.md`, prospective, to be deleted when camera-ready
lands; it also names the artifacts still stale at HEAD (inference-card figure
data, corpus ladder, sediment results, the three-way contrast).

**Finding.** The unit-leakage control, re-run under matrix on the rebuilt
split, reads the opposite direction from the paper's control paragraph: the
25 flagged validation rocks (sub-facies and `?` variants of large tuffs)
evaluate at 0.334 against 0.981 for the other 81, so removing them worsens
the score where on the old split it had improved it. Unit-clustered 95% on
the difference is [-2.81, +1.12]: not established either way, on 11 flagged
units. The control cannot claim a direction. The manuscript never cited it
(the `ctlLeak*` macros are emitted and unused), so nothing in the paper moves;
what moves is the reasoning on #67 and in the 2026-08-06 entry, which gave
the backwards direction as the reason not to repair the unit key. Whether the
remaining `?` and sub-facies fragmentation deserves the curated eruption table
#67 declined is reopened by this, not decided.

**Next.** Issue sweep, then the research week.

## 2026-09-03 -- Two planning-era docs retired into ADR 004 and the journal; techniques.md rewritten around the checkpoint config stamp

**Q.** Which of the docs under `docs/` still earn their place after
submission and #67?

**Context.** Outside the ADRs and the technique index, `docs/` held two
files past their own expiry and one live file with a stale status.
`corpus_regen_spec.md` declared itself due for deletion when the regen
landed 2026-07-27 and was held as the generation-parameter record;
`2026_08_03_recap.md` was a dated snapshot whose measurements and decisions
are the two 2026-08-03 entries above and whose effective-lambda argument is
in heads.md. Both were verified present on `origin/main`, on
`supervised-mineralogy`, and in the 2026-09-02 pre-rename Desktop backup
before deletion. `camera_ready_reconcile.md` stays until camera-ready; it
exists only in unpushed local commits.

**Decision.** `corpus_regen_spec.md` deleted; its parameter record is now
the "Generation record" section of ADR 004's amendment, keeping the section
names code docstrings cite (Conditions, Storage shape, Orchestration) and
stating which corpus is production (first generation, 10k x 48 broad draws,
rungs 50 + 30; the traced tier is a measured non-production arm). Every
reference repointed: CLAUDE.md, ADR 005, encoder.md, roadmap.md, three
`cnmineral.data` docstrings, two build scripts, four archived explorations,
and the dated mentions in this journal. `2026_08_03_recap.md` deleted; the
two numbers only it carried, baseline effective lambda 5.3 (magnitude_weight
0.25 x 21) and trial 22 at 163 (7.76 x 21), are cited from the 2026-07-29
cascade-1 entry by the number builder. `techniques.md` rewritten: the
per-run activation registry it proposed in June was never built, and the
question it was for ("did this run use X?") is answered by the config every
checkpoint stamps (`Checkpoint.config`, `JointHeadCheckpoint.head_config` /
`loss_config` / `label_config`); the doc now says so and drops the hook map.

**Next.** `roadmap.md`'s status section is stamped 2026-08-05 and predates
submission, #67 and the 2026-09-02 thrust ordering; its back half is
spring rewrite-era text. Separate change.

## 2026-09-03 -- #69 read: real-rock error is a hump in corpus size, not a plateau, and a 78-composition head beats production on every mineral category

**Q.** #69 asked for zero-shot absolute error per corpus-scaling checkpoint,
the ruler #60's NLL read folded calibration into. Does share error fall
with the number of simulated compositions?

**Instruments.** The corpus-scaling ladder
(`explorations/corpus_scaling/run_ladder.py`) extended below 625 with
nested 78 / 156 / 312 subsets plus a second training seed at 78, each the
standard joint-heads config on the frozen all-rock encoder, 96 rows per
composition (48 draws x rungs 50 + 30). Zero-shot `scripts/eval/real_rock.py`
on every point, all eight re-evaluated on HEAD (the cached 08-06 values
reproduce to the third decimal). Per-category magnitude absolute error on
the 640 evaluated rocks, all three splits, quantified rocks per category;
the constant forecast (train-split mean share) as the reference. Scripts,
CSVs and figures in `tmp/` (gitignored): `group_mae_vs_corpus.py`,
`olivine_mae_vs_corpus.py`, `draw_vs_bulk_variance.py`,
`climatology_mae.py`. New checkpoints under `checkpoints/corpus_scaling_
allrock_n{78,156,312}` and `_n78_seed1`, runs under `runs/real_rock/`.

**Result.** Absolute error rises from 78 to about 625 compositions and
plateaus, in every category. Mean of seven: 0.102 at 78 (0.101 on the
second seed), 0.116 at 625, 0.113 at 10,000; constant forecast 0.114. At 78
the model beats the constant forecast on five of seven categories; at
10,000 it loses on five of seven (the 08-13 finding reproduced). Both 78
seeds beat the 10,000 head in every category (fsp 0.125 / 0.119 vs 0.145;
px 0.073 / 0.078 vs 0.081; amp 0.105 / 0.111 vs 0.121; bi 0.077 / 0.071 vs
0.078; ol 0.098 / 0.097 vs 0.100; opq 0.073 / 0.068 vs 0.084; q 0.165 /
0.162 vs 0.180). Corpus-side val joint NLL improves monotonically across
the same ladder (0.96 -> -2.67) while the real-rock headline NLL worsens
monotonically (3.6 at 78, 8.0 at 625, 11.6 at 10,000). Presence bias per
category (mean predicted probability minus observed rate) is identical at
625 and 10,000: amp -0.39, q +0.38, px +0.15, ol -0.16, opq -0.16. Scale
transfer (magnitude R2) rises only for olivine, 0.52 -> 0.61; every other
category sits at or below 0.23 or negative at every size. Olivine is the
one category that recovers past 625, back to where 78 already was.

Two hypotheses tested against the corpus and rejected as the separator.
Coverage: about a third of bulks at every ladder rung have MgO above 6 wt%,
so olivine-bearing compositions are not a tail the ladder fills in. Draw
noise: the within-bulk (draw-set) share of simulated share variance at rung
50 on 1,991 bulks is amp 0.67, ol 0.58, fsp 0.39, bi 0.36, px 0.21, opq
0.20, q 0.09, which does not sort the categories by slope (pyroxene and
quartz are composition-set yet flat; olivine is draw-set yet recovers).

**Reading.** Seventy-eight compositions over the frozen encoder are enough
to learn the coarse composition-to-mineralogy map, and that map transfers.
Further doublings add the simulator's fine structure, which is exactly
where an equilibrium assemblage at half-crystallized disagrees with a
phenocryst count, so learning it adds error. Olivine is the exception
because its simulated label is the one that transfers. Two failure
signatures separate: amphibole is draw-dominated in the corpus and its
presence discrimination is near chance, so whatever water content is right
is averaged away by the draw prior; quartz, pyroxene and feldspar are
composition-set regardless of draw and biased at every draw, so the rung is
the suspect. Biotite's rank correlation is negative at every size, and
opaques are set by the fixed fO2 the heads never see. Corpus scaling helps
exactly where the simulated label and the measured mineralogy describe the
same thing.

**Decision (owner, 2026-09-03).** The 09-02 Lawrencium rule resolves on the
dropped side: the cluster has no corpus-size use. The direction is
calibrating the generator against measured mineralogy, formalized as epic
#74 with one child per experiment (#75 to #79): three tests on
labels already on disk (rung-60 heads, a regularized 10,000-composition
head, the matrix recalibrator on the 78 head), then the generator run on
the 640 benchmark compositions at all four rungs to separate "right
conditions buried under the draw prior" from "wrong rung" from "wrong
recipe", and only then a generator-prior search with its objective locked
in advance. Three seeds per point before anything is paper-bound; the
submitted numbers are untouched. Also landed: `run_real_rock_eval` had
crashed on HEAD since the mixture deletion (a dangling `log_w` argument);
fixed, one line; the zero-shot eval path has no end-to-end test.

**Next.** The three on-disk tests, about an hour combined, before the
plumbing day the benchmark-composition run needs.

## 2026-09-03 -- Zero-shot evaluation split into scoring, metrics and figures, with the smoke test the mixture deletion showed was missing

**Q.** `run_real_rock_eval` crashed on HEAD for a day after the mixture
deletion left one dangling argument, and no test noticed. What guards the
instrument every verdict rests on?

**Decision.** The function is now three: `score_real_rock` (one forward pass
and every per-rock array and group statistic, no I/O), `real_rock_metrics`
(the JSON summary of those arrays) and `plot_real_rock` (the nine figures);
`run_real_rock_eval` composes them with the same signature as before and
writes `metrics.json` before any figure is drawn, so a plotting failure can
no longer lose the numbers. Verified on the production checkpoint: the
post-split `metrics.json` is byte-identical to the pre-split run. A flag to
skip figures was rejected as a test-serving switch on production code; the
split is the separation the architecture audit (#71) would want anyway.

The smoke test (`test_zero_shot_scoring_runs_end_to_end_without_a_figure`)
runs a randomly initialised ModeHead over the shipped benchmark through
scoring and metrics in 60 ms, with no checkpoint or encoder, and patches
`plt.subplots` / `plt.figure` to raise, so a figure creeping into the
scoring path fails it. It uses the shipped benchmark rather than a synthetic
one because the null forecast fits on the committed roster, and because the
benchmark's own edge cases (trace-only rocks, single-group compositions,
censoring) are what the path has to survive.

## 2026-09-03 -- #75, #76, #77: the hump is fitting depth, not corpus size, and the recalibrator absorbs it

**Q.** Epic #74 step 1, the three tests on labels already on disk: do
rung-60 labels shrink the quartz / pyroxene / feldspar error (#75); does a
regularised 10,000-composition head match the 78-composition head (#76);
does corpus fidelity matter after matrix recalibration (#77)?

**Instruments.** `tmp/train_variant.py` trains the standard joint-heads
config with key overrides (rungs, weight decay, ModeHead width, an epoch
truncated to its first N shuffled batches) and evaluates zero-shot;
`tmp/variant_read.py` reads per-category absolute error, presence bias, R2
and AUROC beside the ladder points; `tmp/recalibrated_78.py` fits the
matrix recalibrator on the train roster and reads val, paired between
checkpoints by the unit bootstrap. Checkpoints
`checkpoints/{rung60,steps*}_allrock_n*`, runs under `runs/real_rock/`,
logs under `tmp/logs/`. Gradient steps count batches of 1,024 rows: the 78
head's ModeHead was selected at epoch 7 of 8 batches, 56 steps; production's
at epoch 6 of 938, about 5,600.

**#75, rung.** Rung-60 heads at 625 and 10,000 compositions (ModeHead
selected at epochs 9 and 11, no early-stop trouble, so [60, 40] was not
needed). At 10,000, absolute error against the rung 50 + 30 head: feldspar
0.141 vs 0.145, pyroxene 0.086 vs 0.081, quartz 0.196 vs 0.180; mean of
seven 0.116 vs 0.113; seed gap 0.004-0.006 per category. None of the three
clears the gate and two go the wrong way. What the rung does move is
presence: quartz presence bias falls from +0.41 to +0.14 at both sizes, and
biotite flips from +0.05 to -0.23 (the generator makes little biotite at 60%
melt, the reason ADR 004 kept rung 30). Headline NLL 8.2 vs 11.6 at 10,000,
all in the width term. The read-out depth sets which phases the equilibrium
assemblage contains, not how much of each; the share error of the
composition-set categories is upstream of the rung.

**#76, regularisation.** The arm that answers it is the step-matched one. A
10,000-composition head trained with 8 batches per epoch for 7 epochs, 56
gradient steps, the schedule the 78 head was selected under, lands at
mean-of-seven 0.104 and 0.102 on two seeds (78 head 0.102 / 0.101;
production 0.113), beats production in six of seven categories, and sits
inside the seed gap of the 78 head on the mean. A step ladder at 10,000
compositions reproduces the corpus-size hump as a hump in fitting depth:
24 steps 0.103; 56 steps 0.104 / 0.102; 112 steps 0.110; 224 steps 0.111;
448 steps 0.115; about 950 steps selected on corpus val 0.115; production
at about 5,600 steps 0.113. Headline NLL 2.3, 3.4 / 3.0, 4.0, 5.0, 6.8, 7.7,
11.6, monotone. The 56-step head's corpus val hurdle NLL is -0.23 nats per
row against production's -2.67: the emulator that transfers best is the one
that has barely fit the simulator. The weight-decay arms (10x, 100x, 1000x) and the narrow-ModeHead arm,
finished after the pause, do not reproduce it: 10x is production to the
third decimal (0.113, ModeHead selected at the same epoch); 100x and 1000x
land at 0.114 with headline NLL 15.1 / 14.8 (the ModeHead selects later,
epoch 18, about 16,800 steps); the 64-wide ModeHead at 0.110 with NLL 7.0
(selected at epoch 61). Only stopping early reproduces the 78 head; a
weight-norm or width brake does not, so the effect is optimization depth,
not parameter capacity.

**#77, recalibration.** Matrix recalibrator fit on the 372-rock train
roster, read on 106 val rocks / 39 units, weight decay re-selected at 1.0
for every head. Val joint: production 0.829 (reproduces the restamp), the
78 head 0.849 and 0.727 on its two seeds, the 625 head 0.971. Paired,
production minus 78: -0.020 [-0.555, +0.440] and +0.102 [-0.343, +0.500];
production minus 625: -0.142 [-0.382, +0.080]; the 78 seed gap +0.122
[+0.031, +0.237]. Share error after recalibration 0.102 / 0.103 / 0.100 for
production and the two 78 seeds (zero-shot val 0.121 / 0.114 / 0.111). A
7-8 nat zero-shot gap on the same rocks (11.3 vs 4.5 / 3.5) becomes a tie
inside the seed gap after one 119-parameter layer. The 56-step 10,000 head
recalibrates to 0.673 / 0.685 (production minus it +0.155 [-0.204, +0.512]
and +0.144 [-0.212, +0.468]) and the rung-60 10,000 head to 0.851 (-0.022
[-0.343, +0.293]); share error after recalibration 0.102 / 0.101 and 0.104.
Every head on disk is a tie after the layer.

**Reading.** The three results are one finding. Fitting the simulator past a
few dozen gradient steps adds per-category location and width bias that
measured mineralogy does not share; that bias is what the corpus-size
ladder measured (a larger corpus reaches the same steps sooner), what the
78 head avoided by starving, and what the matrix recalibrator removes. The
rung changes the phase list, not the share sizes. For the epic: the
zero-shot baseline every #74 experiment must beat is the early-stopped
10,000 head (0.102-0.104), not the 78 head; #79's objective must be
recalibrated error or the slope of a ladder read after recalibration,
since zero-shot differences of several nats do not survive the layer; the
rung leaves the search as a share dimension and stays a presence dimension
for quartz and biotite, which #78 reads directly. The head-selection rule
(early stopping on corpus val) is what places production at 5,600 steps,
the worst real-rock point of the ladder; selecting on the real-rock train
split, which #74's rules allow, is a candidate the owner has not ruled on.

**Next.** Owner reads the verdicts before #78; every #76 arm is finished. The two
ladders overlaid on gradient steps at ModeHead selection are
`tmp/group_mae_vs_steps.py` (+ `.csv`, `.png`): the mean-of-seven traces one
curve. No point has more than two seeds.

## 2026-09-04 -- Corpus size at fixed fitting depth: the hump was the stopping rule, and the two rulers want different heads

**Q.** Once gradient steps are held fixed, does a larger simulated corpus
help real rocks at all? The #69 ladder and the #76 step ladder had varied
corpus size and fitting depth together (journal 2026-09-03): each ladder
point was stopped where corpus val was best, which is 56 steps at 78
compositions and about 5,600 at 10,000.

**Instruments.** `tmp/train_variant.py` with every training epoch cut to
its first 8 shuffled batches of 1,024 rows, so every head sees one step
schedule and only the composition pool differs; a grid of 78 / 312 / 1,250
/ 5,000 / 10,000 compositions by 24 / 56 / 112 / 224 / 448 steps (25
heads, the 10,000 row being the #76 step ladder), the traced-regen corpus
at 10,000 compositions as a second row, and `tmp/depth_ladder_read.py` for
the zero-shot and matrix-recalibrated reads (fit on the 372 train rocks,
read on 106 val rocks / 39 units, unit bootstrap on per-rock slopes).
Bias here is always relative to this benchmark's convention, du Bray's
phenocryst point counts on quenched lavas; nothing below says the
simulator is wrong, only that what a head learns past a few dozen steps
is not what that convention counts (owner's framing).

**Result.** Zero-shot mean-of-seven absolute error at 24 or 56 steps is
flat in corpus size: 0.101 to 0.104 from 78 to 10,000, slope +0.0002 and
+0.0000 per doubling. Every depth line rises with depth at every size. After
recalibration corpus size helps modestly at 56 to 224 steps, 0.02 to 0.05
nats per doubling: 0.85 at 78 down to 0.64 to 0.69 at 5,000 to 10,000
against production's 0.83, share error unchanged at 0.101 to 0.104. The
recalibrator wants a deeper head than zero-shot does: the 24-step heads
recalibrate worst (0.85 to 1.01) and get worse with size, the 56 to 224
step heads best. The traced corpus is 0.003 to 0.007 better zero-shot at
24 and 56 steps (0.0965 and 0.0999, three seeds, range under 0.001) and
identical after recalibration. Per category, biotite is worse than a
constant forecast at every size and depth, olivine is flat everywhere, and
feldspar, opaques and quartz are pure depth effects. One caveat found in
the read: the trainer keeps the best corpus-val epoch, so the 78 and 312
rows never reached the 112 to 448 cells (their corpus-val optimum stops
them at 56 and 176 steps); the next entry fills them.

**Reading.** Corpus size is a weak lever on both rulers and fitting depth
a strong one (0.104 versus 0.113 zero-shot; 0.65 versus 0.83
recalibrated), so on this benchmark the stopping rule matters more than
the generator. The three experiments that round this out are the fair
learning curve with selection on the real-rock train split, true fixed
depth for the small corpora, and seeds; next entry.

## 2026-09-04 -- The fair learning curve is flat: real-rock selection picks 16 to 24 steps at every corpus size, and the recalibrated read wants a different head

**Q.** Three follow-ups to the grid, planned with the owner. (1) The
community's learning-curve method with the right validation set: stop
the ModeHead where the real-rock train split says to, once on hurdle NLL
and once on share absolute error, read val. (2) True fixed depth for the
small corpora: keep the last-epoch state. (3) Seeds on the cells the
conclusions rest on.

**Instruments.** `train_joint_heads` gained an optional per-epoch
`epoch_probe` whose series are stamped in the checkpoint monitor, and a
`mode_selection` config key (`val`, today's rule and the default; `final`;
or a monitor series name), tested in `tests/training/test_joint_heads_train.py`
and indexed as `target_split_model_selection`. `tmp/train_variant.py
--mode-selection` builds the probe (per epoch: the mode head's hurdle NLL
terms and share error on the 372 evaluated train rocks, from one cached
set of latents). Runs: `stop{nll,mae}_allrock_n{78..10000}` on the 8-batch
schedule with patience 10 on the chosen series; `final{112,224,448}_allrock_n{78,312}`;
seeds 1 and 2 at 10,000 x {56, 112, 224}, 78 x 56, 5,000 x 56, traced x
{24, 56}, and the stop rows at 78 and 10,000. Thirty-eight heads, forty
minutes in three detached chains. Read by `tmp/depth_ladder_read.py`;
figures `tmp/results/depth_ladder.png`, `depth_ladder_groups.png`,
`stop_on_real_rock.png`.

**Result, selection on the real-rock train split.** The per-epoch curves
are nearly the same at every corpus size: train hurdle NLL bottoms at 16
to 24 steps and rises monotonically to 8 to 10 nats by 1,000 steps; share
error bottoms at 16 to 56 steps and rises to 0.11 to 0.13. The NLL rule
selects 16 to 24 steps at all five sizes (three seeds at 78 and 10,000:
16 / 24 / 24 both), the share-error rule 16 to 56. Zero-shot mean-of-seven
under either rule is flat in corpus size, 0.100 to 0.103 (NLL rule) and
0.098 to 0.101 (share rule), slopes -0.0004 and -0.0001 per doubling; the
10,000 head under the share rule is the best zero-shot head measured,
0.098 (three seeds 0.097 to 0.100). After recalibration the same heads
are among the worst: 0.82 to 0.95 val joint (NLL rule, slope +0.016
[-0.001, +0.036] per doubling), no better than production's 0.83 and 0.2
nats behind the 56 to 224 step heads.

**Result, true fixed depth.** With the last-epoch state, 78 compositions
at 112 / 224 / 448 steps give 0.110 / 0.119 / 0.125 zero-shot and 0.82 /
0.98 / 1.00 recalibrated; 312 gives 0.108 / 0.113 / 0.117 and 0.85 / 0.89 /
1.00. A small corpus pushed past its corpus-val optimum overfits its own
compositions. With those cells filled, the deep lines slope down with
corpus size on both rulers: at 224 steps -0.0010 per doubling zero-shot
and -0.046 [-0.087, -0.010] nats per doubling recalibrated, the one
corpus-size slope whose interval excludes zero.

**Result, seeds.** Three seeds at 10,000 x 56 / 112 / 224 span 0.002
zero-shot and 0.03 to 0.05 nats recalibrated; 5,000 x 56 spans 0.001 and
0.04; 78 x 56 spans 0.004 and 0.12. The traced edge at 24 steps holds on
three seeds (0.0964 to 0.0968 against 0.103). Every grid conclusion
survives.

**Reading.** The learning curve done the community's way, each corpus
size stopped on the target split, is flat in corpus size on the zero-shot
ruler and picks the same depth everywhere, so a fixed budget of about 24
steps reproduces it with no real labels in the loop. The corpus-size hump
of #69 was the selection set, nothing else. But the zero-shot optimum and
the recalibrated optimum are different heads: the recalibrator wants a
head fit four to ten times deeper (112 to 224 steps), where corpus size
does help and where the best heads sit 0.15 to 0.19 nats below
production. Selection on real-rock train NLL therefore serves the
zero-shot claim and hurts the recalibrated one. Whichever ruler the paper
reports through decides the rule, and the rule can be a fixed step budget
either way. At fixed deep depth more simulated data helps in the ordinary
sense, by preventing a small corpus from overfitting its compositions;
that is a different claim from "more data improves the best achievable
head", which stays unsupported zero-shot and weakly supported after
recalibration.

**Decision.** The stopping rule is the owner's call; not taken here.
Candidates named, not adopted: selection on recalibrated train NLL (the
layer fit inside the epoch loop, untested), and a three-seed 10,000 x 112
to 224 head as the recalibrated baseline for #74. tmp/ stays as laid out
in `tmp/README.md` until the rule is decided.

**Next.** Owner reads; then #78 or the rule.

## 2026-09-04 -- Rulings after the depth grid: the target is a scaling pair, no stopping rule yet, plutonic parked, one figure

**Decision (owner).** Four rulings on the day's results. (1) The target of
epic #74 is a **scaling pair**: a simulated label convention and a
measurement convention whose error falls with corpus size at fixed depth,
after recalibration, on seeds; on the canon figure, a line that falls in
the zero-shot and recalibrated share-error panels to below the shallow
heads. The volcanic search for one is the priority, exhaustive on the
simulator side (rung, water and pressure prior, redox, recipe, the
phenocryst/groundmass split of the label) since the benchmark side is thin.
(2) No stopping rule is chosen. There is no time pressure, the canon figure
shows every depth side by side, and a convention picked now would be
picked on less evidence than a convention picked after the candidate
corpora are drawn; the only step that needs a written convention is #79's
objective, which comes last. (3) #63 and #22, the plutonic branch, are
parked until every volcanic pair tried fails to scale. (4)
`tmp/results/depth_ladder.png` as regenerated today (56 / 224 / 448-step
lines, the head stopped on the real-rock train split, a no-information
reference per ruler, production as a star) is the figure every scaling
experiment is reviewed on; seeds are spent only on lines that look like a
pair. Recorded in `tmp/README.md`; roadmap re-sequenced.

**Next.** #78. Before its plumbing day, a one-bulk smoke run of the
generator (`scripts/build/generate_mineralogy_corpus.py` over
`magemin/run_magemin.jl`) to confirm Julia and MAGEMin still run on this
machine; nothing has generated since the split rebuild.

## 2026-09-09 -- The depth-grid method graduates to explorations/corpus_scaling; two review figures in their current style; tmp/ emptied

**Q.** What from the corpus-scaling scratch directory survives, and where?

**Context.** `tmp/README.md` tied its own graduation to the stopping-rule
decision, which the 09-04 rulings deliberately left open, while the canon
figure's generator sat in a gitignored directory where deletions are
permanent. On 09-08 the figure script had gained an unrecorded `_clean`
variant (no references, no production star, larger text) for a talk.

**Decision (owner).** Two figures are the ones every scaling experiment is
reviewed on: `depth_ladder.png` (four rulers) and its per-category
companion `depth_ladder_groups.png`, in their current matplotlib style.
The clean variant was a one-off presentation export and is dropped. Paper
styling (the ratified frame, fonts, cividis) waits until a scaling pair is
found. Graduation no longer waits on the stopping rule. Everything in
`tmp/` not behind those two figures is flushed: the per-category table and
step-ladder readers (the step ladder is the 10,000-composition row of the
grid), the three run.sh chains and their logs, the #69 and #77 csv/json
side products, the stop-curve figure (its per-epoch series stay stamped in
every checkpoint's monitor), and the clean exports. The run.sh files were
not redundant with the checkpoint stamp, which records neither the seed,
the batches per epoch, the selection rule nor the roster; the tag grammar,
written once in the README, covers all 63 grid heads and the 25 whose
run.sh had already gone.

**Landed.** `explorations/corpus_scaling/run_variant.py` (the cell trainer)
and `run_depth_grid.py` (the reader and the two figures), the latter with
the constant-forecast reference folded in from the deleted
`climatology_mae.py` (reproduces its json to the last digit) and without
the clean and stop-curve drawing; `results/depth_ladder.{csv,json,png}` and
`depth_ladder_groups.png` moved beside the #27 and #60 artifacts, and the
`--figures-only` redraw from the new location is byte-identical to the
09-08 files. Three tripwire tests (cell rule to gradient steps, seed
aggregation and the final-over-grid line rule, the truncated loader).
README fourth section with the tag grammar, the two-figure ruling, the
scaling-pair definition and the seeds protocol, plus an inline correction
to the #60 decision (the ladder confounded depth with size; the verdict
stands). Roadmap and memory repointed.

**Next.** #78. The one-bulk generator smoke run passed the same day: one
dacite at two trace-conditional draws, 62 MAGEMin points, all converged,
7.6 s wall including Julia startup under Julia 1.12.6, four-rung
assemblages as expected (quartz, feldspar, biotite, spinel, amphibole in
one draw); Julia and MAGEMin are intact after the split rebuild.

## 2026-09-10 -- Production code is not an experiment's configuration surface: the rule lands, and #81 lists its first eleven cases

**Q.** Where should the rule that experiments do not patch production live,
and what does the codebase already violate?

**Context.** Preparing #78, the owner ruled that its scripts must be a
self-contained exploration: the generator's expansion step hard-codes the
trace-conditional prior, and the fix is a copy inside the exploration, not
a flag on `scripts/build/generate_mineralogy_corpus.py`. The rule behind
that ruling existed only in pieces: `explorations/README.md` said reuse
flows one direction, `docs/code_style.md` said production classes serve
callers rather than tests, CLAUDE.md said leave unrelated code alone and
delete dead code. None said that a flag, branch, hook or config key whose
only caller is an experiment is that experiment's scaffolding.

**Decision (owner).** The rule is written in three places: CLAUDE.md
("Production code is not an experiment's configuration surface", the
authoritative statement), `explorations/README.md` (the operational
version: copy the step into the exploration and change the copy) and
`docs/code_style.md` (one sentence extending the tests rule to
experiments). The `mode_selection` key and `epoch_probe` hook added to
`train_joint_heads` on 09-04 are the rule's first case and will be
reverted into the corpus-scaling exploration. Three read-only audits over
`src/cnmineral/training` + `models`, `eval` + `data`, and `scripts/`
produced the rest; every candidate's caller list, originating commit and
recorded decision were verified by hand before filing. #81 (P1, refactor)
holds eleven items with per-item instructions, ground rules (one commit
each, technique index regenerated, `numbers.tex` byte-identical before and
after, explorations carry their own copies) and a checked-but-borderline
list so the next sweep does not re-find them. Nothing is removed yet;
sequenced before or as the opening step of #71. Two audit claims fell on
verification and were dropped: the predicted-modes closure function is
called by the head HPO objective, and the `--redox` probe flag has a
standing scorecard axis behind it (recorded as borderline instead).

**Also found.** `paper/figures/src/extract_inference_card.py` fits the
deleted `tailed` tier and asserts a two-component head, so it raises on
HEAD; `paper/` is gitignored, so this goes on the camera-ready reconcile
list rather than #81.

**Next.** #78 as a self-contained exploration: the benchmark composition
table, the exploration's own expansion over the broad box at 128 draws,
the run, the three-way read.

## 2026-09-16 -- #78: equilibrium on the benchmark rocks' own chemistry reaches the silicates but the marginal buries them; the four-silicate subset is the target and #79 becomes a prior screen

**Q.** Can the equilibrium generator reproduce the measured mineralogy of
the 640 evaluated benchmark rocks at all, and where it cannot, is the miss
the draw prior, the read-out stage, or the recipe?

**Context.** Every corpus tried is flat on the real-rock axis at fixed
fitting depth (journal 2026-09-04). #78 removed the model:
`explorations/generator_reach/` runs MAGEMin on each evaluated rock's own
chemistry at 128 draws from the broad box the production corpus used,
keeps all 31 ladder points, and per rock and category measures three
absolute errors from the measured share on the paper's crystal
normalization (the mean over the pooled rung-50 and rung-30 rows, the best
single draw at those rungs chosen per category, the best step along that
draw's whole path), sorted at tolerance 0.05 into reproduced / buried /
wrong stage / wrong recipe (owner rulings 09-09/10 and 09-15, written into
the README because the earlier record sat in the emptied `tmp/`). Plumbing
facts the build found: the repo has no QAPF-from-chemistry classifier
(gard's names are precomputed), so the iron split takes a TAS class
computed from the majors; benchmark ids collide numerically with gard2019
ids, so the table carries its own index; shapely was an undeclared
leftover of PetThermoTools, so the classifier uses matplotlib's path
geometry. 2.54M points, 6.1 h, no solver crashes, every draw reached both
rungs.

**Result.** At 0.05, buried dominates feldspar (0.71), amphibole (0.69),
quartz (0.59), pyroxene (0.50) and biotite (0.39): some draw reaches
within 0.05 for 80 to 95% of the silicate rocks while the marginal reaches
16 to 45%. Wrong stage is 3 to 17% everywhere and the best-step melt
fraction barely tracks crystallinity (Spearman 0.04 to 0.27), so the rung
is not the share lever. Olivine (0.43) and Fe-Ti oxides (0.34) are wrong
recipe, the best step on the best draw still off by about 0.1. The hits
are a mass for pyroxene and biotite and a tail for feldspar and amphibole.
The categories disagree on where the hits sit: the four silicates at high
pressure and moderate water (median 8 to 10 kbar, 3.6 to 4.7 wt%), oxides,
olivine and quartz low and dry (1 to 3 kbar, 1.3 to 2.4 wt%). Read jointly
(scratch, not in the read script), one (draw, rung) reaches every
quantified category of a rock within 0.05 for 28% of rocks and within 0.10
for 59%, halving the typical rock's error against the marginal, with
conditions spread over the whole box; oxides block the joint hit most
often. Not measured: reading each rock at its own melt fraction, and joint
reach over the four silicates alone.

**Considered.** Reachable is not scalable: the run shows the recipe can
produce the silicate shares somewhere in the box, not that the right draw
is predictable from chemistry. A prior aimed at one corner helps the
silicates and hurts oxides, olivine and quartz, and shares renormalize
over a rock's quantified categories, so an oxide that stays wrong drags on
the same rock's silicates. Per-category verdicts flatter against the joint
read.

**Decision (owner).** Pursue the coherent subset, feldspar, pyroxene,
amphibole and biotite (about 90% of the crystal cargo on the typical rock,
hits in one direction), high pressure and relatively wet. #79 is unblocked
and rewritten as a screen of candidate (P, H2O) priors ranked by a
scalability score written before any corpus is generated: the per-doubling
slope of matrix-recalibrated val share error on the four categories across
312 / 1,250 / 5,000 compositions at fixed fitting depth (224 steps as the
primary, 56 as the zero-shot check), a level check against the
production-prior cells already on disk, a spread term, three seeds on any
winner's endpoints. Stage 0 reweights the #78 draws with no MAGEMin
(selection on the train split, reading on val); stage 1 generates the top
schemes at 1,250 compositions locally in the production table shape; stage
2 extends survivors to 5,000 with seeds and is the Lawrencium case.
Oxides, olivine and quartz are reported beside the score, direction not
chosen; their recipe verdict goes to #22 on the volcanic side. #75, #76,
#77 and #78 closed with their verdicts; #74 and the roadmap updated.
Nothing here touches production or the paper's numbers.

**Next.** Lawrencium readiness for stage 2, a discussion before any port:
what of the environment, Julia and MAGEMin, the corpus tables and
checkpoints must be on the cluster. Then #79 stage 0.

## 2026-09-18 -- #79: no condition prior moves the silicate labels, chemistry cannot place them, the heads already sit at the labels' ceiling; the scaling pair changes ruler first

**Q.** Can any (P, H2O) prior for single-condition equilibrium, global or
conditioned on the rock's chemistry, bring the corpus labels close enough
to the phenocryst counts for corpus size to matter?

**Context.** #78 found that some draw reaches the measured silicate share
of most benchmark rocks while the broad-box marginal buries it. Owner
rulings 2026-09-16: the ladder is 128 / 256 / 512 / 1,024 / 2,048
compositions (four doublings, one night of MAGEMin per scheme, every size
by nesting), seeds on the endpoints are training seeds so no scheme is
generated twice, no intermediate screen, the cluster delayed.
`explorations/prior_search/` screens priors without MAGEMin: every #78
draw carries the broad-box density, so any prior inside the box is scored
by the reweighted mean share over the pooled rung rows against the
measured share, per category over the rocks quantifying it, the
four-silicate mean, the train split selecting and val reading. The broad
control reproduces #78's marginal to floating point.

**Result.** The rectangle family (centers 6 / 8 / 10 kbar x 3 / 4.5 / 6
wt%, two widths): the best takes the silicate error from 0.116 / 0.136
(train / val) to 0.108 / 0.130 and the grid spans 0.02 with the broad box
in its middle; pressure helps mildly, water not at all, 6 wt% is the worst
column everywhere, and a pressure band of 8 to 12 kbar with water left
open equals the best rectangle; every silicate-aimed scheme raises the
oxide, olivine and quartz error from 0.19 to about 0.21. The production
trace-conditional prior is the mirror image (0.126 / 0.144 on the
silicates, 0.181 on the others): its literature boxes sit at storage
depths of 1 to 6 kbar while the silicate hits want 7 to 12. The
predictability read defines a rock's hits as the rows within 0.05 on the
four silicates together: 38% of rocks have none (median best joint error
0.037; feldspar's best draw is not amphibole's). Ceilings on val: 0.088
under the hit rows themselves, 0.107 under a wide window on each rock's
own hit center. Windows on a predicted center: random forest on the
majors 0.131 (R2 0.17 for pressure, 0.04 for water), ridge and the
TAS-class mean 0.137, the train mean 0.139. The existing heads, zero-shot
on the same val rocks: production 0.119, the 5,000-composition heads at
224 and 56 steps 0.123 and 0.112, between the best global window and the
oracle.

**Considered.** The reweighting is a label-side proxy, but the heads
anchor it: trained on the broad box, they already beat every global
marginal, so the prior is not what limits them, and the most a perfectly
conditioned prior could open is about 0.01. Untested single-condition
knobs, the iron oxidation state, the database and traces as conditioning
inputs, act mostly on the oxide side, which #78 already ruled
recipe-limited. ADR 004's amendment chose equilibrium for volcanics on the
argument that a volcanic rock crystallizes closed-system, and records that
no test was run; this is the first measurement bearing on it. The
argument holds for the whole rock, but the ruler counts phenocrysts,
which formed over a history before eruption, and no single condition
reproduces all four silicates for 38% of rocks. The generator and the
ruler count different things.

**Decision (owner, 2026-09-18).** For the current (generator, ruler) pair
there is no way to change how the generator runs to attain scaling. The
pair has to change: rethink the generator and introduce fractional
crystallization against the same ruler, or find a new ruler. Two paths.
(1) Same ruler, new recipe: fractional crystallization of the rock's own
bulk with the accumulated crystals as the label, the phenocryst cargo as
it forms. (2) Same generator, new ruler: du Bray's plutonic whole-rock
modes, read on the current generator at a solidus rung, which does not
require FC (FC-of-parents is for cumulates only, the crux recorded on
#22). Path 2 first, because it costs a harvest and a rung and its outcome
sorts path 1: if whole-rock equilibrium scales on whole-rock modes, the
volcanic failure is the phenocryst ruler and FC is that ruler's fix; if
it does not, the problem is deeper than the recipe. #79 closes on its
own gate; #74's step 4 becomes the active thrust; stage 1 and the
reference cells are not run; the rosters stay as the ladder for the next
pair. The condition prior joins the settled decisions in CLAUDE.md. Every
open issue was re-audited against this: #79 closed, #74, #22, #63, #57
and #21 updated by comment, the rest unchanged, and step 4 filed as #82,
#83 and #84. Nothing here touches production or the
paper's numbers.

**Next.** The plutonic harvest (#82: DS 942 and PP 1793, the #63
workstream-2 shortlist), then the model-free solidus-rung reach on each
plutonic rock's own bulk and the ladder on the two review figures (#83);
fractional crystallization of the rock's own bulk against the phenocryst
ruler (#84) opens after #83 reads.

## 2026-09-29 -- #85: most of #79's silicate error is offset, not missing information; feldspar and quartz carry signal past a constant, amphibole and biotite none

**Q.** Does the evidence support #79's verdict that no way of running the
generator makes the volcanic pair scale? #79 read only the level of the
simulator's label (its average absolute error against the measured share),
never whether the label tracks differences between rocks once that offset
is removed, and it carried no constant baseline and no intervals.

**Context.** Re-audit in discussion with the owner. The flat real-rock line
under the production prior is measured with trained heads (#60, #69, #76,
the depth grid); #79's inference past it is simulator-only and involved no
recalibration. A scratch check put a constant forecast, the train-mean
share renormalized over each rock's mineral list, at 0.106 on the four val
silicates: level with #79's per-rock oracle window (0.107), below every
head (0.112 to 0.123) and below the generator average (0.136). The owner
filed three follow-ups as #85 with no gate: the gate is written after every
metric is read.

**Instruments.** `explorations/generator_reach/run_signal.py`, seconds and
model-free on the #78 rows: #79's rows rebuilt rock by rock (reproducing
the committed numbers to 1e-16), the constant, and the 5,000-composition
head at 224 steps zero-shot, matrix-recalibrated and given no chemistry.
Per category on the val split: absolute error, Pearson r across rocks, and
the error after a per-category least-squares line from the source's share
to the measured share, fit on train, clipped and renormalized over the
mineral list. Unit-bootstrap bands over the 39 val units. A second read on
the 63 rocks outside test that carry both a visual estimate and a DS 1099
point count of the same thin section. `run_depth_grid.py` gained val-only
per-category share error and r for every cell and per-doubling slopes with
bands for every fixed-depth line, for the third follow-up.

**Result.**
- On the four silicates the line takes the simulator from 0.136 to 0.102
  [0.087, 0.120], the constant's level: -0.004 [-0.011, +0.002] against it.
  The zero-shot head after the line reads 0.100, the recalibrated head
  0.101, the head given no chemistry 0.106.
- Per category after the line, against the constant: feldspar -0.015
  [-0.030, -0.003], quartz -0.043 [-0.064, -0.023], pyroxene -0.006
  [-0.016, +0.003], opaques 0.000, amphibole +0.005 [-0.014, +0.022].
  Biotite's simulated share runs against the measured one (r -0.27, fit
  slope -0.44), and even the oracle hit rows leave biotite at the constant.
- The constant correlates with the measured silicate share at 0.48, the
  simulator at 0.27: the mineral list carries more of the rank information
  than the simulator's variation on top of it.
- Visual estimate and point count of the same thin sections differ by
  0.127 in feldspar share (visual lower by 0.120, counting floor 0.015),
  0.021 amphibole, 0.046 biotite, 0.067 opaques, 0.072 quartz. Against the
  point count the simulator's error is about half its error against the
  visual estimate for amphibole, biotite and opaques, and unchanged for
  feldspar and quartz. Two tuff eruptions, 61 of the 63 rocks in train: a
  hint at the measurement's own error, not a floor.

**Considered.** The line stands in for a recalibration layer acting on the
simulator itself, chosen so the error sits in share units beside the
constant; the real recalibrator on the head is the second view and agrees.
The constant varies only through the mineral list, so its r is what the
list alone carries; a joint fit of simulator and constant, the signal past
the mineral list, was not run. The 224-step line's -0.046 nats per doubling
(journal 2026-09-04) is an observation, not an example of a scaling pair
(owner).

**Reading.** #79's narrow claim stands: no condition prior inside the box
moves the silicate labels. Its broad claim, that no generator setting can
produce a scaling pair on this benchmark, does not: the label's signal is
per category, present for feldspar and quartz and absent for amphibole and
biotite, and part of the gap may be the measurement convention.

**Next (owner, 2026-10-02).** (1) The 224-step line with every point a
last-epoch head on three seeds (13 new heads, `final224_allrock_n{N}` with
`_seed{K}`), read per category on recalibrated share error, feldspar,
quartz and pyroxene first. (2) The same signal read as the first step of
#83, on the plutonic point-count benchmark, before any ladder.

## 2026-10-02 -- #82: the plutonic benchmark is built; zero-shot, the head sits 2.8 nats over its null there against 10.6 on eruption-v2, the gap closing mostly in the amounts

**Q.** Is a whole-rock benchmark built, and how does the production head
read on it before anything is fit to it?

**Context.** #79 sent the scaling pair to a new measurement convention
first: whole-rock point counts of plutonic rocks, which a solidus read of
the current generator might match where phenocryst counts could not
(journal 2026-09-18). #82 harvests du Bray DS 942 (1.4 Ga A-type
granitoids) and PP 1793 (Boulder batholith) into
`data/real_mineralogy/plutonic/`, apart from eruption-v2, whose tables stay
byte-identical. DS 454 is pinned as the authority on PP 1793's columns.
Owner rulings during the build: DS 942's unit is `Shorthand` (55 units,
keeping an intrusion's facies together) and PP 1793's is `intrs_name` (30
after canonicalization); every rock the sources kept counts as fresh,
since neither grades alteration per rock; the 71 rocks with a category
unmeasured (the oxides counted with the accessories, or only a mafic total)
stay in the split and out of the evaluated set; numbers are read on train
and val only. The split deals whole intrusions in random order with at
least 15 per subset, because the eruption split's largest-first dealer left
val with amphibole in 21% of rocks against 57% overall.

**Result.**
- The funnel (`results/plutonic_composition.md`): 1,555 rows in the two
  sources' tables, 579 point-counted, 578 with chemistry, 507 evaluated.
  Train 291 rocks, val 61, test 155 (sealed). 363 of the 507 carry measured
  mass outside the seven categories, a median 1.9% of the rock:
  accessories on 300 rocks, secondary minerals on 134, muscovite on 98.
- Zero-shot on the 352 train and val rocks, each benchmark read against
  its own climatology null: joint NLL -1.46 against the null's -4.26, so
  +2.80 nats; eruption-v2 +10.63. The presence term is +1.74 against
  eruption-v2's +2.11. The amounts term is +1.06 against +8.52.
- Presence AUROC holds where a category varies: amphibole 0.76 (eruption-v2
  0.58), pyroxene 0.78. Biotite (0.48) and the oxides (0.46) sit at chance,
  but 97% and 93% of rocks carry them, so the AUROC rests on a few rocks.
- Amounts rank rocks moderately: Spearman 0.55 amphibole, 0.52 biotite,
  0.59 quartz, 0.44 pyroxene, 0.40 oxides, 0.30 feldspar. Share absolute
  error is lower than eruption-v2's in every category.
- The matrix recalibrator fit on 25 train rocks from 6 intrusions takes the
  val joint NLL from 0.60 to -2.86 over 61 rocks. The paper's two cards are
  redrawn on these rocks (`run_plutonic_figures.py`).

**Considered.** Raw NLL does not compare across the benchmarks: a plutonic
rock spans 4.4 categories against 3.6 and the null is sharper (Dirichlet
concentration 31.6 against 12.5), so each is read against its own null.
Lower share error is not yet evidence of signal: plutonic shares may simply
vary less from rock to rock, and #85 showed a constant over each rock's
mineral list can match the heads. That comparison has not been run here.

**Reading.** The head transfers to whole-rock modes better than to
phenocryst counts, and most of the improvement is in the amounts, which is
where the two conventions count different things. Whether the head's
rock-to-rock variation beats a constant on this benchmark is open.

**Next (owner, 2026-10-02).** Two rulings shape #83's readout.

Every source is read under the two settings the scaling review figures
use: no correction, and the production matrix recalibrator fit on train.
With unlimited data a head converges to the generator's draw
distribution, so the generator read that way is the ceiling of each
scaling line. Spearman rank correlation, with its sign, is the
fitting-free check. #85's per-category straight line is retired: it
matched neither panel, added where production multiplies, ignored
trade-offs between categories and credited backwards relations.

Signal is judged against a ladder of baselines, not the constant alone.
Two floors use no chemistry: the constant over the measured mineral list
(amounts only, and plutonic lists barely vary) and the head given no
chemistry. Two chemistry baselines use no simulator: the CIPW norm read
like any source (no amphibole or biotite; a comparator, not a training
target) and the head given raw geochemistry fit on train.

The head side runs first and needs no MAGEMin. Then comes #83's generator
run on these rocks. #83 is rewritten to this readout with no gate until
every metric is read, #84's pre-written gate is withdrawn, and #85's
volcanic read is re-run once under the new settings.
#82's body overstates PP 1793 (259 rocks and 35 intrusions where the
harvest finds 246 and 30); the owner closes the issue.

## 2026-10-05 -- #83: the plutonic benchmark shows almost no cumulate chemistry; no cumulate rule is built

**Q.** Is a cumulate rule worth building for #83? The plan was to flag rocks
that are crystal piles rather than frozen melts and report them separately.

**Context.** The generator treats a rock's chemistry as a melt that cooled in
place. That fails for a cumulate, so a misfit there says little about the
generator. The europium anomaly, the usual marker, exists for few of these
rocks. The handoff cited PP 1793 as suggesting plagioclase accumulation in
three plutons covering 126 rocks.

**Result.** A scratch read; no repo files were written. Each of the 507
evaluated rocks was placed against 47,105 gard2019 volcanic rocks (named by
known origin) within ±1.5 wt% SiO2. Volcanic rocks are mostly frozen melts,
so they mark out the melt range.
- Major elements: 1 rock has Al2O3 above the 95th percentile and CaO above
  the 90th; at the 80th and 75th, 16 rocks (3%). No rock has a MgO and FeO
  excess.
- Sr: 21 rocks (4%) sit above the 97.5th percentile, 20 of them in PP 1793
  and 13 in Donald. All 21 are also high in Rb, where a plagioclase pile
  would be low. The whole Boulder batholith sits high in both against
  volcanic rocks worldwide (medians at the 92nd and 95th percentiles).
- The source's claim rests on one sample per pluton with a small positive
  Eu anomaly (Eu/Eu* 1.09, 1.18, 1.28), against a batholith average of 0.87.
  Rader Creek, 78 of the 107 evaluated rocks in those plutons, shows nothing
  here. DS 942 never discusses accumulation.

**Considered.** The check catches only strong cumulates: the volcanic
reference carries crystals, and a feldspar-and-quartz pile is chemically
close to a granite melt. That limit cuts both ways. Where chemistry stays
inside the melt range, mineral proportions are still largely fixed by the
chemistry, so the own-bulk read gains little error. Cumulates matter where
chemistry sits far from any melt, and almost none does here.

**Decision (owner, 2026-10-05).** No cumulate rule, no flag, no separate
reporting. Every #83 read covers all evaluated rocks together. Recorded on
#83 (body and comment) and on #22, whose plutonic case has no rocks to
serve here.

**Next.** #83's head-side read (step 0), then the remaining open decisions:
the recipe and box, redox, muscovite, and the joint fit.

## 2026-10-05 -- #83 step 0: after correction the production head ties the simple baselines on plutonic rocks, and the CIPW norm under the same correction beats them

**Q.** Before any generator run, does the production head track the plutonic
whole-rock modes better than baselines that use no simulator?

**Context.** These are the owner's 2026-10-02 rulings, read for the first
time. Every source is read with no correction and under the production
matrix recalibrator fit on train. Spearman, with its sign, is the
fitting-free check. Signal is judged against a ladder of baselines: the
constant over the measured mineral list, the head given no chemistry, the
CIPW norm, and the head given raw geochemistry. The code is
`run_signal.py --benchmark plutonic`. pyrolite 0.3.6 supplies the norm.
The norm enters the recalibrator packaged as a head-style forecast: the
head's median concentration and a 1e-3 floor.

**Result.** On the four silicates, 61 val rocks from 19 intrusions, with 95%
bands over intrusions:
- The recalibrated head reads 0.046 share error. The constant reads 0.050,
  the head given no chemistry 0.047 and the head given raw geochemistry
  0.051. Every difference band crosses zero.
- The head ranks rocks better than those baselines (Spearman 0.38 against
  0.08 and 0.09), but the ranking does not lower the error.
- The recalibrated norm reads 0.038, below the constant (-0.011 [-0.037,
  -0.005]), the head given no chemistry and the head given raw
  geochemistry.
- Against the recalibrated head the norm is -0.007 [-0.036, +0.002].
- Joint NLL: head -3.69, norm -4.22, head given no chemistry -3.89.
- Uncorrected, the norm (0.069) already sits below the zero-shot head
  (0.080).

**Considered.** The production head learned labels read at 50% and 30%
melt, so its tie does not test the generator near the solidus; Stage A
does. A head fit on 291 real rocks' chemistry does not beat the constant,
while the norm's mass balance gives the felsic minerals almost for free.
Val is small: 8 rocks carry pyroxene and 1 olivine.

**Reading (owner, 2026-10-05).** Step 0 does not answer #83. It sets the
bar: Stage A's labels must beat the CIPW norm under the same correction,
not only the constant. That puts the joint-fit question (does the
generator add anything beyond the norm?) at the centre. The scaling test
remains Stage B, after Stage A is read.

**Next.** Settle #83's open decisions (the recipe and box, redox, muscovite,
the joint fit). Then Stage A's code through plan mode, the owner's 6-hour
run, and the read. #85's volcanic re-run under the new settings is the
next code change in `run_signal.py`.

## 2026-10-05 -- #85: on the uniform 224-step line, corpus size does not reduce recalibrated share error; only quartz falls, and it stops by 1,250 compositions

**Q.** With every point a last-epoch head at 224 steps on three seeds, does
recalibrated share error fall with corpus size per category, feldspar,
quartz and pyroxene first?

**Context.** The earlier 224-step line mixed last-epoch heads (78, 312)
with heads selected on corpus val (1,250 and up) and had seeds only at
10,000, so its -0.046 nats per doubling was an observation (journal
2026-09-29). Run 2026-10-02: 13 new `final224_allrock_n{N}[_seed{K}]`
heads, about 44 s each, then `run_depth_grid.py`. Every earlier row of
`depth_ladder.csv` is unchanged, and the three heads that repeat an
earlier one (n5000 seed 0, n10000 seeds 1 and 2) reproduce it exactly.

**Result.** Recalibrated share error on val, seed means from 78 to 10,000
compositions, with the constant over each rock's mineral list beside it
(same rocks as `generator_reach/results/signal.csv`); slopes per doubling
with 95% unit-bootstrap bands:
- feldspar 0.145 to 0.143 (constant 0.158), -0.0005 [-0.0014, +0.0005];
- quartz, 27 rocks, 0.133 to 0.111 (constant 0.165), -0.0031 [-0.0055,
  -0.0010], all of it between 78 and 1,250 (then 0.112, 0.112, 0.111);
- pyroxene 0.065 to 0.062 (constant 0.068) and amphibole 0.105 to 0.098
  (constant 0.105), bands crossing zero;
- biotite 0.095 to 0.100, above the constant (0.094) at every size;
- four silicates -0.0004 [-0.0009, +0.0003];
- recalibrated joint NLL 0.894 to 0.682 nats, -0.032 [-0.069, +0.0005].

**Reading.** For equilibrium labels against du Bray's phenocryst estimates,
corpus size does not reduce the recalibrated shares: feldspar's signal
past the constant is already there at 78 compositions, and amphibole and
biotite never pass it. The joint NLL still falls, but not through the
shares; #86 asks whether presence or magnitude spread carries it.

**Next.** #86. The 63 tuffs with both a visual estimate and a point count
are deprioritized (owner).

## 2026-10-05 -- #83 step 0 re-read: with the 68 oxide-lumped rocks and grouped 5-fold cross-validation, the production head ties the CIPW norm and both beat the simple baselines

**Q.** Does step 0's reading survive the two protocol changes the owner
ruled for screening reads?

**Context.** Two rulings the same day. First, the 68 rocks whose Fe-Ti
oxides were lumped with the accessories join every read with the oxides
marked unmeasured, through an optional presence mask on the hurdle NLL
(`RealRockBenchmark.measured`, `GroupPredictions.measured`; all True on
every other benchmark, so published numbers are bit-identical, checked on
the eruption benchmark). The evaluated set grows from 507 to 575; the
numpy scoring path has no mask and refuses a masked benchmark. Second,
screening reads fit and read by grouped 5-fold cross-validation over the
train and val rocks, intrusions kept whole: 404 rocks from 61 intrusions
read out of fold, against 61 rocks from 19 intrusions before. The
reasoning: differences near 0.007 did not clear zero on 19 intrusions.
Test stays sealed; Stage B's review figures stay train and val. The
refactored read reproduces the committed train/val step 0 exactly before
either change is applied.

**Result.** Four-silicate share absolute error, out of fold [95%
unit-bootstrap band over 61 intrusions]:
- CIPW norm, recalibrated: 0.042 [0.035, 0.053] (was 0.038 [0.026, 0.062]);
- production head, recalibrated: 0.042 [0.038, 0.049] (was 0.046);
  minus the norm -0.0003 [-0.0095, +0.0072];
- the head minus the constant, the head given no chemistry and the head
  given raw geochemistry: -0.012 [-0.033, -0.005], -0.012 [-0.029,
  -0.002], -0.010 [-0.032, -0.004];
- the norm sits about 0.011 below the same three, bands clear of zero;
- joint NLL, head minus norm: -0.10 [-0.37, +0.23] nats.

**Reading.** Step 0's first reading does not survive. The head no longer
ties the simple baselines: once the read covers 61 intrusions, it beats
them by about 0.01 and ties the norm. The old val split of 19 intrusions
was unfavourable to the head. The bar for Stage A becomes 0.042, held
jointly by the corrected norm and the corrected head. One intrusion,
Rader Creek, has 89 of the 404 rocks read; grouped folds put it in a fold
of its own.

**Next.** Stage A's read runs on this protocol once the MAGEMin run
finishes. The new `run_signal.py --benchmark plutonic` is first exercised
end to end on that run.

## 2026-10-06 -- #83 Stage A: the simulator near the solidus ties the 0.042 bar after recalibration, adds a small NLL gain beyond the CIPW norm, and tracks amphibole and biotite on whole-rock modes

**Q.** Read near the solidus on each plutonic rock's own chemistry, does
the equilibrium label track the whole-rock modes better than the baselines
that use no simulator, and does it carry anything the CIPW norm lacks?

**Context.** The run (2026-10-05, owner's machine): 578 rocks x 128 draws
over the #78 box x 31 steps to 600 C, 5.5 h, no solver crashes, read at
10% melt, 5% melt and the last converged step. 130 and 222 of 73,600
draws never reach 10% and 5%; no train or val rock falls back to its last
step. The read is the step 0 protocol: grouped 5-fold over the 404 train
and val rocks from 61 intrusions, the 68 oxide-lumped rocks with oxides
unmeasured, bands over intrusions, test sealed. Two joint fits ask whether
the simulator adds information beyond the norm: B, the norm's matrix
recalibrator plus 7x7 terms reading the simulator that start and shrink at
zero, so its no-gain case is the recalibrated norm exactly; A, the small
measured head given the norm, the simulator, or both, a nonlinear check.
First, step 0 was re-run in all four corners (the 68 rocks in or out,
cross-validation or train/val; scratch) to separate the two changes behind
its reversal.

**Result.** Which change moved the head: the protocol. Under
cross-validation the recalibrated head reads 0.042 with or without the 68
rocks (the norm 0.044 and 0.042), head minus norm crossing zero in both.
Under train/val the 68 rocks push the head up and the norm down (0.049
against 0.035, +0.014 [+0.002, +0.052]). The 19 val intrusions were a bad
draw for the head.

Stage A, four silicates, 404 rocks out of fold:
- simulator at 10% melt, zero-shot 0.060 [0.049, 0.068], Spearman 0.49,
  the best zero-shot source (head 0.080, norm 0.082) and the best ranking;
  recalibrated 0.044 [0.036, 0.067], NLL -4.71, against the bar's 0.042
  (head -4.79, norm -4.69). The 5% and last-step reads sit within 0.001.
- joint fit B minus the recalibrated norm: share error +0.0002 [-0.0015,
  +0.0045]; joint NLL -0.18 [-0.30, -0.02] at 10% melt, -0.15 [-0.25,
  -0.01] at 5%, -0.19 [-0.27, -0.08] at the last step.
- joint fit A: 0.049 to 0.051 for every input set, above the bar, as the
  head given raw geochemistry at step 0.
- Spearman per category, simulator zero-shot / recalibrated, then the
  recalibrated norm and head: amphibole (226 rocks) 0.58 / 0.52, 0.29,
  0.44; biotite (391) 0.52 / 0.52, 0.42, 0.58; pyroxene (40) 0.57 / 0.30,
  0.39, 0.44; feldspar (404) 0.31 / 0.28, 0.19, 0.23; quartz (403) 0.48 /
  0.66, 0.65, 0.65.
- reach: one (draw, rung) row within 0.05 on all four silicates for 90% of
  the read rocks, against 62% on the eruption rocks.

**Reading.** The simulator tracks these rocks. Amphibole and biotite,
which carried nothing and ran backwards on the eruption benchmark, track
at 0.5 to 0.6 here, so #85's hypothesis holds: whole-rock point counts
follow the simulator where visual phenocryst estimates did not, and the
recipe and the database are not the cause on those two. Three unrelated
sources, thermodynamics, the norm's arithmetic and the head, meet at 0.042
share error after a linear correction; share error is saturated by what
major-element chemistry plus a 7x7 correction fit on about 320 rocks can
say, and the nonlinear fit cannot beat it with 404 rocks. The simulator's
edge over the norm is probabilistic, a small NLL gain carried mostly by
the presence of minerals the norm cannot produce. The head's zero-shot gap
to the simulator, 0.080 against 0.060, is confounded by the rung: the
head emulates 50 and 30% melt. No gate (owner).

**Next** (owner, 2026-10-06; the shape may change as results come in).
First the simulator at 50 and 30% melt on the same rocks from the same
run, no MAGEMin and no training, to split the rung from emulation error:
if the simulator at 50/30 reads near the head's 0.080 the head copied the
simulator faithfully and the gap is the rung, so a solidus head starts near
the oracle and the ladder would show little on share error; if it stays
near 0.060 the gap is copying error a corpus can close. Then #87's
tracking read of the prior on these draws. Then one regeneration that
keeps the whole cooling path, so the rung is a training-time choice, under
#87's policy with the broad box as the reference line. Then the ladder,
with the simulator's own read as the oracle line and share error, NLL and
Spearman tracked together.

## 2026-10-06 -- #83: at the head's own rungs the simulator still reads 0.056 to 0.059, so the head's zero-shot gap is copying error, not the rung

**Q.** The production head emulates MAGEMin at 50 and 30% melt and reads
0.080 zero-shot on the plutonic rocks; the simulator near the solidus
reads 0.060. Is the gap the rung, or does the head fail to reproduce the
simulator on these rocks?

**Context.** The Stage A run kept every draw's whole cooling path, so the
simulator at 50 and 30% melt is read from the same rows with no MAGEMin
and no training (`run_signal.py` now reads five rungs, the ends re-derived
from the ladder and checked against the run's own). Beside it, the head's
copying error: its zero-shot share against the simulator's at each rung on
the same rocks, scored as a source is scored against the measured share.

**Result.** Four silicates, 404 rocks out of fold:
- simulator zero-shot: 0.059 [0.049, 0.083] at 50% melt, 0.056 [0.047,
  0.064] at 30%, 0.060 at 10%, 0.061 at 5%; Spearman 0.48 to 0.49 at
  every rung; the head 0.080, Spearman 0.45;
- recalibrated: 0.042 at 50%, 0.041 at 30%, tying the bar, with joint NLL
  -4.89 and -4.84, the best of any single source (head -4.79, norm -4.69);
- the head's copying error against the simulator: 0.065 at 50% melt,
  0.074 at 30%, 0.077 at 10%; per category at 50%, feldspar 0.122 against
  the simulator's 0.074 from the rocks, amphibole 0.113 against 0.051,
  biotite 0.034 both, pyroxene 0.049 against 0.076 on 40 rocks;
- median read temperature 720 C at 50% melt, 660 at 30%, 640 at 10%.

**Reading.** The rung is not the gap. The simulator at the head's own
rungs sits where it sits near the solidus, because a granitoid's
crystallization window is narrow and the assemblage changes little across
it. The head's 0.080 is its own failure to reproduce the simulator on
these rocks' chemistry: the copying error is as large as the simulator's
distance from the rocks, concentrated in feldspar and amphibole. That is
the gap a corpus could close, which gives the ladder something to show
that the volcanic ladder never had. Two causes remain open: too little
granitoid chemistry in the corpus, or an input shift, since the plutonic
rocks report trace elements on 11 to 61% of rows and the head sees mostly
blank trace tokens. The rung is no longer a corpus-shape decision for this
benchmark; keeping the whole path in a regeneration stays free.

**Next** (owner, 2026-10-06). Five steps, one decision each. First, where
the copying error comes from: the head against MAGEMin on its own
simulated val bulks, with traces as reported and blanked to the plutonic
pattern. Second, whether the line falls at all: the depth-ladder heads on
disk (78 to 5,000 bulks at 56 and 224 steps, seeds at the endpoints) read
on the 404 rocks under the Stage A protocol, with each head's copying
error against the simulator; a screen, so it ranks and never closes.
Third, #87's tracking read of the prior on the plutonic draws. Fourth, one
regeneration under #87's policy with the broad box as reference and the
whole path kept, its shape taking the first step's answer. Fifth, the
ladder as the full test, then one read on test.
