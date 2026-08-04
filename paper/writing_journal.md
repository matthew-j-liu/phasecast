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
