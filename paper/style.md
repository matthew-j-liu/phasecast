# Prose style sheet (v0, from owner abstracts, 2026-08-04)

Derived from three owner-authored abstracts (TiCF / Co-DIM electrolyte /
TiHx). Owner corrects this sheet; corrections here propagate to all
drafting. Re-read before every drafting pass, together with one sample.
Status: abstracts only -- paragraph-scale rules await intro samples.

## Sentence architecture

- One move per sentence; the chain advances by picking up the prior
  sentence's object as the next subject.
- Canonical abstract order: field heuristic or phenomenon -> gap ->
  candidate with reason as appositive -> obstacle -> "Here, we" pivot ->
  evidence chain -> quantified headline -> significance.
- Openers are domain-internal (a heuristic, a phenomenon, the finding
  itself), never cosmic importance claims. Motivation is earned by
  specificity.
- Reasons embed as appositives ("Titanium (Ti), with low
  electronegativity and high NO3RR reactivity, is a compelling
  candidate"), not as separate because-sentences.
- Subordination via participles ("limiting the achievable rate",
  "thereby decoupling X from Y") and appositives; no dash-interruptions.
- Sentence length varies with function: pivots short, evidence sentences
  long and loaded; never uniform rhythm.

## Evidence and numbers

- The instrument or study is the grammatical agent: "Control studies
  show", "DFT calculations attribute", "With GIXRD and XAS, we
  demonstrated".
- Numbers inline with complete conditions and uncertainties; full
  recipes in parentheses. Quantify instead of hedging. Comparatives get
  multipliers (60x, 5.2x, up to 0.50 V).
- Parentheses carry technical payload only: definitions (DIM = ...),
  conditions, ranges (0 < x <= 2), examples (e.g., FeNC, CuNC).

## Connectives and diction

- Sparse, deliberate connectives: one "However" per unit; "To date";
  "Here, we"; "Thus"/"therefore" only for the true consequence;
  "thereby" to chain result to purpose. No "First... Second..."
  scaffolding.
- Verb palette: leverage, elude, tailor, surpass, underscore, attribute,
  decouple, facilitate, promote, achieve. Precise and kinetic; never
  ornate.
- Closers generalize with one concrete anchor plus, often, one
  plainspoken image ("a simple handle"), then a forward gain
  ("advancing the MNC design space and sustainable NH3 production").

## Never (observed absent in all samples; my drafts violated these)

- Balanced two-part aphorisms ("Minerals, not elements, are...").
- Rhetorical questions.
- Agent-less importance claims ("X is what Y acts on").
- Uniform 15-25-word sentence rhythm.
- Hedges where a number could stand.

## Punctuation budget (owner rule, 2026-08-04)

- At most ONE colon per section (abstract counts as a section, the whole
  intro counts as one section). Spend it on the most colon-worthy
  construction and rewrite the rest as sentences.
- At most ONE semicolon per section, same accounting.
- Punctuation inside [cite ...] and [slot: ...] brackets is exempt (it
  compiles to \citep and markers, not prose).
- Long sentences get broken up, with shorter sentences interspersed --
  but short means a complete formal sentence of roughly 12-20 words,
  never a fragment. The owner's own shortest abstract sentences stay
  full clauses with subjects and qualified objects.

## Verdict log (owner-adjudicated exemplars; append as feedback lands)

- BAD, too informal (2026-08-04): "Measuring it is slow." Ultra-short
  fragments read casual. Break long sentences into mid-length formal
  ones instead.
- BAD, too qualitative (2026-08-04): "Recalibration recovers most of
  the gap." Magnitude claims carry numbers with provenance, via macros
  ("reduces validation joint NLL from X to Y nats"), never "most".
- GOOD (2026-08-04): "Instruments such as X-ray fluorescence (XRF) and
  inductively coupled plasma (ICP) spectrometry report a rock's
  elemental concentrations, are standardized and cheap, and already run
  at scale in exploration and production [cite NRMP 2023]." Mid-length,
  a series of three concrete predicates, acronyms glossed once, cited.

## Audience calibration (owner rule, 2026-08-04)

- Readers are ML practitioners, not physical scientists. Never assume a
  technique is known. Name it once with a plain-language gloss of what
  it involves ("an expert counting mineral grains under a microscope",
  "instruments that report a rock's elemental concentrations").
- Domain shorthand gets spelled out on first use ((T, P, H2O) ->
  temperature, pressure, and water content; spot analyses -> chemistry
  measured on individual crystals).

## Paragraph architecture (from intros 1-2, TiCF and Co-DIM)

- P1 is a motivation funnel that goes societal-to-technical in ONE
  paragraph, numbers from the first sentence ("132 Mt/yr production in
  2021"), every factual claim cited: compound/problem -> uses ->
  incumbent process with quantified costs -> "Thus, [the need]" ->
  feedstock/opportunity with its reason -> current practice -> the
  coupling opportunity -> the technology -> one practical advantage.
- Authority anchors where available ("which the U.S. National Academy of
  Engineering has identified as one of 14 grand engineering
  challenges"); equity/societal dimensions stated concretely when true
  ("disproportionately benefit the world's wealthiest regions").
- Topic sentences either name the paragraph's object or enumerate what
  is coming ("NO3RR can help address challenges in decentralization,
  carbon emissions, and energy consumption"); the body delivers in the
  announced order, one item per sentence.
- Gap construction is multi-step and ends NAMED: survey of what exists
  (with citation ranges, "(e.g., Fe, Co, Cu, ...)") -> "However, ...
  has rarely been..." -> cross-field comparison ("In contrast, ... is
  common in other reactions such as ORR and CO2RR") -> emerging
  heuristic -> "but this design rule has not yet been applied to X" ->
  "Notably, [candidate]..." -> "To our knowledge, ... have not been
  synthesized due to..." -> "Thus, a synthetic knowledge gap has
  prevented...". The sibling-field contrast is a signature move.
- The rationale paragraph narrates reasoned bets in first person:
  "We identified an opportunity to... by leveraging...", "We
  hypothesized that...", "We reasoned that...", "leading us to
  hypothesize", "We therefore sought to characterize...", "We were
  particularly interested in X given that Y". Each bet carries its
  mechanistic reason and a number. Closes with "Taken together, ...".
- Final intro paragraph is a compressed tour of the paper: "In this
  work, we..." + enumerated design ("(1)... (2)... (3)...") -> findings
  with instruments as agents -> quantified benchmark ("Compared to 13
  other MNC electrocatalysts...") -> "Importantly, controls show..." ->
  mechanism attribution -> "Overall, this work establishes X and
  demonstrates Y" -> one future-facing final clause.
- Synthesis closers: "Taken together,", "Together, the results
  highlight", "Overall, this work establishes".
- Semicolon-separated series for surveys of strategies; "To our
  knowledge" reserved for novelty claims; "In practice," to pivot from
  theory to reality.

## Related work (from the ANN sample; CS-venue register)

- Organized by method family; per family: definition/framing with cites
  -> history ("explored in early theoretical studies... Subsequent
  work, such as...") -> status assessment with limitation ("became
  relatively underexplored despite offering...").
- Explicit adoption sentences tie survey to the paper: "We adopt this
  framing in our work", "We adapt this cost framework in Section 3.4".
- Named failure modes for dominant methods, cited, no straw-manning.
- Final paragraph converts the survey into the paper's opening move:
  "We anticipated that... particularly in regimes where... Thus, we
  revisit...".

## Figure captions

- Self-contained narratives. Schematics get reading instructions
  ("Beginning in the bottom-right (major steps underlined), ...") and
  end with the takeaway sentence.
- Data panels: what is plotted, complete conditions per panel
  (concentrations, scan rate), how derived ("Half-wave potentials were
  measured from CVs of..."), what fits are ("The dotted Nernstian fit
  comes from nonlinear regression of the data to eq 4"), what guide
  lines mean. Conditions repeated per panel, not "same as (a)".

## Open

- Owner has not yet corrected this sheet; corrections propagate before
  the next drafting pass.
- outline.md abstract/intro v3 predate the sheet and get rewritten
  against the corrected version.
