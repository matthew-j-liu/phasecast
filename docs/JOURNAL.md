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
