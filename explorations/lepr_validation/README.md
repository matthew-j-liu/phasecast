# lepr_validation

**Question.** Can the LEPR experimental-petrology dump (Library of Experimental
Phase Relations, 2007 snapshot) supply real (bulk chemistry -> mineral
assemblage) validation pairs where the conditions are *known by construction* —
lab experiments that equilibrated a measured starting composition at a set
(P, T), then reported the phases? That is the one label source where CondHead's
targets are measurements rather than simulator draws.

**Method.** `extract_lepr.py` parses the two sheets of
`data/experiments/ExPetDB_download_LEPR-2007-08-17.xlsx` (gitignored raw
download), joins experiments to starting bulks on (Citation, Experiment) —
experiment IDs repeat across papers — and drops keys with conflicting duplicate
T/P/phases or bulks. Filters to the crustal-volcanic regime: starting SiO2 in
[45, 78] wt%, P <= 10 kbar, assemblage containing silicate melt plus at least
one solid. LEPR phase codes map to a coarse vocabulary (feldspar,
clinopyroxene, orthopyroxene, amphibole, olivine, garnet, oxide, quartz/silica,
mica, feldspathoid, other); deep-mantle/metal codes are dropped from
assemblages entirely.

**Result.** 2,368 experiments survive, written to
`results/lepr_crustal_volcanic.csv` (committed): starting oxides, H2O (26%
non-null, summed from LEPR's H2O+ / H2O- columns), coarse + raw assemblages,
P_kbar, T_C, and data-quality columns (bulk_conflict, unmapped/dropped codes).
Presence-level labels only — LEPR reports which phases grew, not modal
proportions.

**Decision.** None recorded yet — this is a harvested resource, not a settled
question. Extracted 2026-07-01 alongside the real-rock benchmark work; the
natural benchmark (`../real_mineralogy/`, natural rocks, modal proportions,
conditions unknown) became the primary validation set. Candidate uses when
validation work resumes: presence-level head validation at known (P, T), and a
CondHead check on the subset with reported H2O. Complements rather than
overlaps `real_mineralogy` — experiments have measured conditions but no modes;
natural rocks have modes but no conditions.
