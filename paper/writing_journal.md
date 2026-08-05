# Writing journal -- Sim2Science workshop paper

Append-only, dated. Written so a fresh session resumes from this file alone.
Companion to the research journal (`docs/journal.md`), which owns research
decisions; this file owns the paper thread.

## 2026-08-03 -- session 1: infrastructure + structure proposal

**State at close.** Toolchain and scaffolding are up; the structure proposal
is written and awaits owner approval; no paper prose exists beyond the
proposal's draft abstract/introduction. Nothing committed (owner runs git).

**Built this session.**

- tectonic 0.17.0 installed via brew; `paper/neurips_2026.tex` compiles
  as dropped and after every edit below (last check: clean).
- `paper/src/build_numbers.py` -> `paper/numbers.tex` (70 macros, wired
  into the tex via `\input{numbers}`). Reads results/ledger.yaml,
  results/real_rock_ledger.yaml, the two checkpoint-contrast JSONs, and
  the three curve_val.json files. Missing sources degrade to a visible
  [TBD] macro, never a broken build. Rerun:
  `conda run -n alignn_fix python paper/src/build_numbers.py`.
- `paper/figures/` + `figures/src/` + `figures/MANIFEST.md` (rules header;
  no figures built -- candidates listed in outline.md, gated on structure
  approval). Render mode: PNG.
- `paper/.gitignore` (compiled PDF + TeX droppings only; numbers.tex and
  future figures stay tracked).
- `paper/outline.md` -- THE DELIVERABLE: title candidates, draft abstract
  + introduction (full prose, owner rewrites/accepts), bullet skeletons
  for sections 2-7, appendix inventory, candidate figure list, checklist
  item -> section map, submission logistics.

**Verified against sources while drafting** (all on disk today):

- Few-shot journal numbers 3.86/5.97/6.84 at 25 rocks reproduce as the
  MATRIX-tier median over 10 draws (vector tier reads 4.23/6.84/8.67) --
  the generator computes both; prose must name the tier.
- Sealed-test three-way 0.478/0.769/1.371 and all paired contrasts
  reproduce from checkpoint_contrast_matrix_three_way.json.
- SIGN HAZARD: the encoder-swap contrast is stored as
  production-minus-allrock = +0.078 [-0.396, +0.407]; the allrock-baseline
  ledger prose quotes those stored numbers, but read as allrock-vs-
  production the sign flips: allrock adapted val 1.298 vs production
  1.376, i.e. the all-rock encoder is 0.08 nats BETTER on recalibrated
  val (statistical tie). numbers.tex emits allrock-minus-production
  (\rrValConAllrockProdMean = -0.08) with the direction in the macro
  comment. Never transcribe "+0.078, allrock worse".
- #57 read in full: the quartz over-call may be label-semantics drift
  with SiO2 (corpus crystals-renormalized label vs quenched-glass point
  count), unresolved; any quartz-improvement claim stays gated.
- #59 read in full: coverage panel returns only when a calibration claim
  ships; the recalibrator's claim IS a calibration claim, so if claim F
  ships in the paper the panel question reopens -- owner call.
- Claim A's amphibole AUROC 0.539 grounded (journal 2026-07-30: presence
  AUROC on real rocks, measured on the TUNED arm, Brier 0.505, calls amp
  in 19% of rocks against 65% present).

**Open questions for the owner** (also marked in outline.md):

1. Approve / rewrite the structure, title direction, and the draft
   abstract + intro. Related work: standalone short section (my rec) or
   folded into the intro?
2. Checklist items needing owner decisions: open access to code/data
   (weights policy says nothing releases before submission), compute
   facts for appendix G (hardware, run times), whether the benchmark
   harvest counts as a released new asset, LLM-usage answer (NA per
   policy is my read; confirm).
3. Reciprocal reviewing: you must nominate one qualified reviewer at
   submission. Standing flag until done.
4. Two numbers to reconcile before they enter prose: pretrain corpus
   size (encoder.md says ~788k rows; the acceptance record says ~760k
   after dedup/floor) and the recalibrator's exact parameter count
   (recap says ~121; count from the fit code).
5. A suspicious coincidence to resolve in the claims pass: amphibole
   presence AUROC 0.539 (real rocks, tuned arm) and amphibole magnitude
   parity r 0.539 on sampled supports (corpus, slides) are different
   metrics with identical values. Verify both on disk; one may be a
   transcription of the other.
6. Claim A's AUROC arm: claims.md quotes 0.539 without naming the arm;
   the measured value is the tuned arm's. Production's own amp AUROC
   needs pulling from runs/real_rock/joint_heads_supportmag_seed0/
   metrics.json per_group before the qualification ships.

**Queued next** (order per handoff):

1. Owner reviews outline.md -> revise until approved.
2. claims.md corrections, one at a time in plain language, owner
   sign-off before file edits: (a) claim B paired-and-agreeing-on-sign +
   Bolin & Wallin; (b) claim A's two qualifications; (c) claim F's
   measured scope limit; (d) claim D in effective-lambda units.
3. Citation verification, most-leaned-on pillars first (bissiri2016,
   cragg1971/mullahy1986, grunwald2017, mukhoti2020, bolin2023,
   laves2020, ovadia2019); owner supplies unfetchable PDFs into
   literature/.
4. Figures F1-F4 after structure approval (MANIFEST rows as they land).

**Checklist audit.** Nothing answered yet (no paper body). Structure maps
every item to a section in outline.md; open items are 5 (open access),
8 (compute), 13 (new assets), 16 (LLM declaration) -- all owner decisions.
Checklist instruction block still present in checklist.tex (delete before
submission, keep headings/questions).

## 2026-08-04 -- session 2: title locked, motivation pillars recorded

**Decisions with the owner.**

- Working title (owner-picked after iterative brainstorm; set in
  outline.md and the tex \title, compile clean): "Simulated petrological
  labels enable transfer learning for probabilistic mineralogy
  prediction from rock geochemistry". Runner-up and the
  "whole-rock geochemistry" micro-option recorded in outline.md.
  Adjective reasoning kept for reuse: labels are petrological;
  "geochemical" describes the input; "mineralogical" collides with the
  prediction target.
- Two owner-posited motivation pillars added to outline.md (must be
  explicit in abstract/intro/motivation; P1-P2 drafts get rewritten
  around them): (1) crustal rock formation is too multi-scale and
  multi-physics for first principles, so a simulator is unavoidable, and
  the distinguishing move is judicious simulator supervision of an
  encoder pretrained on measured rock compositions; (2) the problem is
  inherently one-to-many (one bulk measurement, in principle infinitely
  many mineral sets), which forces the probabilistic treatment and the
  (P, H2O) exploration.

**Precision flag for the intro rewrite.** Pillar 1 contains two distinct
measurement corpora that must not blur: the encoder's corpus is compiled
whole-rock analyses of natural rocks (gard2019); the measurements that
calibrated MAGEMin's thermodynamic database are laboratory
phase-equilibrium experiments.

**Rewrite done same session.** Abstract v2 and a five-paragraph intro v2
(setting / one-to-many / system / measurement / findings) are in
outline.md, built around both pillars: P1 carries the
first-principles-intractability chain with the lab-experiments vs
natural-rock-compositions distinction kept separate; P2 carries
one-to-many and the brittle-small-labeled-set argument. "25 rocks"
became \sysFewshotBudgetKnee (numbers.tex now 71 macros; compile clean).
Flagged in outline.md: intro v2 runs ~0.85 page vs 0.75 budgeted, making
related-work option B (folded, no standalone section) the likely offset.

**Motivation broadened (owner-directed, same session).** Abstract v3 +
intro draft v3: a new opening motivation joins the pillars -- fundamental
geology (mineralML-rooted: assemblages as the basis of petrological
investigation, crystals record magmatic evolution) extended with the
owner's critical-materials argument (extraction procedures depend on host
phase + mineralogical environment; mineralogy measurement is slow while
XRF/ICP assays are routine; the model bridges assay to downstream
separations, supply-chain/sustainability pull). Intro now six paragraphs
(~1 page): related-work folds into intro/findings (option B effectively
forced) and section 2 tightens. New claims.md section K records the five
motivation citations with status (mineralML PDF pending fields; Minerals
2025 15(5) 533 and NRMP 2023 3:52 UNVERIFIED; DOE 2023 assessment; IEA
2024 -- Agency, not Association, fix at bib). SCOPE GUARD recorded in
both files: the system predicts phase presence/abundances, not
per-element host assignment (trace partitioning is a deferred head);
motivation prose says "informs separations", never "identifies the host
of element X".

**Encoder design rationale added (owner-directed, same session).** Intro
P4 gains the borrowed-decisions passage: CrabNet's core logic (mat2vec
element embeddings + learned abundance encoding standing in for position
embeddings; logic, not machinery) and the features-are-fate warrant
(Rotskoff group, ICML 2025 per owner: pretrained feature space determines
transfer) for pretraining on the assay feature space users measure --
physical motivation (cheap, high-throughput XRF/ICP) + theoretical (one
representation serves the simulator corpus and real-rock transfer).
claims.md section L records both citations (CrabNet PDF on disk; the
features-are-fate quote and fields must be verified at the citation
pass; mat2vec companion cite flagged). Intro now ~1.1 pages; P4 may
split into encoder / heads paragraphs at tex time.

**Prose style calibration started.** Owner judged the draft prose flat
and supplied three solo-authored abstracts (TiCF / Co-DIM / TiHx).
Extracted paper/style.md v0: chain architecture (one move per sentence,
object becomes next subject), domain-internal openers (never cosmic),
instrument-as-agent evidence sentences, numbers inline with full
conditions, sparse deliberate connectives, appositive/participle
subordination, and a never-list my drafts violated (balanced aphorisms,
rhetorical questions, agent-less importance claims, uniform rhythm).
Owner corrects the sheet; intros from papers 1 and 2 requested for
paragraph-scale rules. Abstract/intro v3 get rewritten against the
corrected sheet -- they predate it.

**Style sheet extended to paragraph scale.** Owner supplied intros 1-2
(TiCF, Co-DIM), example figure captions, and a CS-venue related-work
sample (ANN paper). style.md now covers: the one-paragraph
societal-to-technical motivation funnel with numbers from sentence one;
enumerating topic sentences; the multi-step NAMED gap construction with
the sibling-field contrast move; the first-person reasoned-bets rationale
register ("We identified / hypothesized / reasoned"); the final-paragraph
compressed tour ("In this work... (1)(2)(3)... Importantly... Overall");
synthesis closers; related-work family structure with explicit "We adopt
X" adoption sentences; self-contained captions with reading instructions
and per-panel conditions. The ANN sample doubles as the register model
for the appendix related-work discussion of mineralML and other models.

**v4 rewrite executed (owner said proceed, sheet as-is).** Abstract and
intro rewritten against style.md: definitional opener replaces the
banned aphorism; the rhetorical question is gone; P1 is the funnel with
[cite] slots where DOE/IEA numbers land after verification; P2 chains
lab experiments -> databases -> MAGEMin with the "resists direct attack"
image; P3 ends with the gap NAMED ("the obstacle is a label gap") and
absorbs the existing-tools survey (CIPW/pyrolite, mineralML,
materials-informatics precedent -- related-work option B now in
effect, fuller discussion to the appendix); P4 is first-person reasoned
bets (pretrained-from-nature hypothesis, CrabNet adaptation,
features-are-fate warrant); P5 is the compressed tour ("In this work...
Importantly... Overall, this work establishes X and demonstrates Y")
absorbing old P5/P6 and the (i)(ii)(iii) list. Five paragraphs, ~1
page. New [cite] obligations surfaced: the thermodynamic database
behind MAGEMin, CIPW/pyrolite, gard2019 as the pretrain compilation,
and a materials-informatics simulated-labels precedent -- all need
claims.md/bib entries at the citation pass. No new numbers; no tex
change, so no compile needed.

**v5: punctuation budget + audience glosses (owner rules, added to
style.md).** At most one colon and one semicolon per section
([cite]/[slot] bracket punctuation exempt, compiles to \citep/markers);
long sentences broken up with pithy ones interspersed; ML-practitioner
audience, so techniques get plain glosses (point counting = expert
counting grains under a microscope; XRF/ICP = instruments reporting
elemental concentrations) and domain shorthand is spelled out
((T, P, H2O) -> temperature, pressure, water content; spot analyses ->
chemistry measured on individual crystals; phenocryst modes -> measured
mineral proportions). Abstract + intro rewritten to v5 and verified
mechanically: abstract 1 colon / 0 semicolons, intro 1 / 1 (colon spent
on P2's simulator triplet, semicolon on P5's width-not-location).
Compile clean.

**Research-thread event, do not act without owner.** Regenerating
numbers.tex surfaced that explorations/real_rock_fewshot/results/
checkpoint_contrast_matrix.json was overwritten by a newer run (the
known fixed-output-name trap). It now holds supportmag vs
joint_heads_allrock_tuned_seed0 vs joint_heads_allrock_mixture_seed0 --
cascade-2 tuned and mixture arms exist. Consequences: (1) the
\rrValConAllrockProdMean/Lo/Hi macros are TBD (69 macros now); the
verified old values live in the session-1 entry (allrock 1.298 vs
production 1.376 adapted val, stored as prod-minus-allrock +0.078). (2)
The [slot: cascade-2 winner] markers may be fillable soon, but ONLY
after the owner adjudicates production; the paper thread does not
interpret fresh research output on its own.

**End-of-session duties.** numbers.tex regenerated (69 macros, one
visible TBD); compile clean; figures/MANIFEST.md unchanged (no figures
built, candidates await structure approval); checklist audit unchanged
from session 1 (no paper body yet; open owner items remain 5 open
access, 8 compute, 13 new assets, 16 LLM declaration). Commits handed
to the owner (4 thematic: prose+title+numbers, style sheet, claims
sections, journal).

**Verdict log opened (owner feedback on v5).** style.md gains a
verdict-log section for iterative GOOD/BAD exemplars; the pithy rule is
corrected (short = 12-20 word formal sentence, never a fragment) and
qualitative magnitude claims are banned in favor of macro-carried
numbers. Fixes applied to v5: "Measuring it is slow." merged into a
formal sentence; "Recalibration recovers most of the gap." replaced in
abstract and P5 with quantitative statements (validation joint NLL
\rrValZsProd -> \rrValAdProd nats, currently 13.6 -> 1.38; median
\fsTwentyFiveMatrixProd = 3.86 at 25 rocks), via three new rrValZs*
macros read from the preserved three-way JSON (numbers.tex now 72).
Punctuation budget re-verified (abstract 1:/0;, intro 1:/1;), compile
clean. The verdict log is the standing home for prose feedback; distill
each entry into the rule sections when it generalizes.

**Next session, in order.** (1) Owner verdict on abstract/intro v5 and
the sheet, plus the rest of the outline (sections 2-7, appendix,
figures). (2) Re-point the generator's encoder-swap contrast source at
whatever JSON the research thread designates (mirror the three-way
preservation pattern if needed). (3) claims.md corrections conversation
(B, A, F, D -- one at a time, plain language). (4) Citation
verification, pillars first; the motivation set (K) and encoder set (L)
need bib entries from primary sources. (5) Figures F1-F4 after
structure approval. Standing: owner nominates one reciprocal reviewer
at submission; reconcile 788k-vs-760k pretrain count and the exact
recalibrator parameter count before those macros ship.

## 2026-08-04 -- session 3: Overleaf handoff, owner drafts solo; two backlog intakes

**Owner decision: the draft moves to Overleaf and the owner writes solo
for a few days.** This supersedes queue item 1 (the v5 verdict happened
in the parallel session; see the verdict-log entry above). The paper
thread stands by until handback.

**Overleaf file set settled.** Four files: neurips_2026.tex +
neurips_2026.sty + checklist.tex (owner's original drop; git confirms
sty and checklist untouched since, the tex differs by exactly two lines
-- \input{numbers} and the \title) + numbers.tex.
\workshoptitle{Sim2Science} was set at drop time. Sync rules:
numbers.tex flows one way local -> Overleaf (generated, never
hand-edited); the tex flows back at handback (owner downloads into
paper/, commits). references.bib goes up only at the citation pass,
when a \bibliography block is added.

**Solo-writing interface.** [cite ...] and [slot: ...] stay literal in
prose; [n: value, source] marks a number awaiting a macro; % Q: ...
marks questions for me. All four get harvested at handback. Macro
spacing rule delivered: numbers.tex macros are plain \newcommand (no
xspace), so a macro followed by a space or letter needs {} --
\rrBenchN{} erupted -- while punctuation/hyphen need none. Paste-ready
converted abstract/intro blocks (P-labels as % comments, LaTeX quotes,
{} spacing) were compile-verified in scratchpad with rendered-text
spacing confirmed via pdftotext; owner pasted their own version.

**Handback protocol (agreed).** Owner says go; orientation order =
commits -> tex -> diff against outline skeletons. Passes in order:
mechanical (numbers regen + compile, literal-digit grep, marker
harvest, per-section punctuation-budget count), verification (every
factual assertion against claims.md and disk -- sign hazard,
matrix-vs-vector tier naming, val-not-test provenance, section-K scope
guard), style verdicts (log-format GOOD/BAD comments only, never
touching owner prose). outline.md collapse proposed at that point, per
the prospective-plans rule.

**Two research-thread backlog intakes (owner-directed, flat priority).**
Owner's calibration note, near-verbatim: being among the first specific
paper inputs does not give these disproportionate priority or emphasis;
fold them where the outline naturally reaches them, and expect the
owner to iterate a complete figure list later. Both verified against
docs/journal.md (the 2026-08-04 mixture entries) and the archive files
on disk before folding:

1. Candidate appendix figure -> outline table row A3: the genuinely
   bimodal corpus amphibole-share truth histogram with the K=2
   gate-weighted components landing on both humps. If used, regenerate
   at paper quality from explorations/archive/mode_mixture/
   run_mixture_figures.py; never inherit the mockup
   results/beta_overlay_k2_amp.png.
2. Discussion/limitations point -> one bullet each in sections 4 and 7
   skeletons: the mixture's transfer failure decomposes to a
   simulation-side prior limitation, not a model or procedure failure.
   The benchmark population at matched chemistry carries the corpus's
   full width in one hump; the corpus's second hump is manufactured by
   the deliberately broad hidden-(P, H2O) proposal; the 7-group
   instrument passes 61% of the corpus advantage, so coarsening is not
   the bottleneck. Connects corpus-design choices to transfer behavior;
   motivates #62 (benchmark populations with genuine path
   multimodality).

**End-of-session duties.** outline.md and this journal are the only
files touched (markdown only -- no tex change, so numbers.tex and the
compile state stand as the parallel session left them: 72 macros, one
visible TBD, compile clean). figures/MANIFEST.md unchanged (A3 is an
outline candidate; MANIFEST rows land when figures are built).
Checklist audit unchanged (open owner items remain 5 open access, 8
compute, 13 new assets, 16 LLM declaration). Commits stay with the
owner; these edits fold into the already-modified outline.md /
writing_journal.md set, so the previously handed commit commands remain
correct.

**Next, in order.** (1) STANDBY while the owner drafts in Overleaf;
respond to questions as they come. (2) Handback passes per the protocol
above. (3) Re-point the generator's encoder-swap contrast source at
whatever JSON the research thread designates (the overwritten
checkpoint_contrast_matrix.json currently holds the cascade-2 arms;
val-only, all paired contrasts cross zero -- no interpretation until
the owner adjudicates production). (4) claims.md corrections
conversation (B, A, F, D -- one at a time, plain language). (5)
Citation verification, pillars first; K and L need bib entries from
primary sources. (6) Iterate the figure outline with the owner (owner-directed
2026-08-04: the candidate table -- F1-F4, A1-A3, gated rows -- is a
draft; a full iteration pass with the owner comes before any figure is
built), then build after structure approval. Standing:
owner nominates one reciprocal reviewer at submission; reconcile
788k-vs-760k pretrain count and the exact recalibrator parameter count
before those macros ship.
