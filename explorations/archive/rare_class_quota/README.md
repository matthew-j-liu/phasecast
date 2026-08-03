# Rare-class quotas for the +20k train expansion (#45)

**Question.** The regen adds ~20k new train bulks on top of the kept 10k. How
should the new slots divide across the 22 qap classes, given a pool where the
big four (basalt/andesite/dacite/rhyolite) hold 89% of the mass and the bottom
ten classes have only 116-508 bulks total -- and given that #27 measured
rare-phase skill (garnet) still data-limited at 10k bulks while #41 named
corpus enrichment the remaining presence lever?

**Method.** `quota_analysis.py` (sampler-side, no MAGEMin). Class abundance is
fit as a rank-frequency power law. Candidate allocations all use
`cnmineral.data.volcanic_subset.expansion_quotas` (temperature-smoothed:
quota ~ pool^alpha share of the expanded 30k, less current holdings, capped by
availability, capped mass respilled by weight):

- **alpha = 0.82**: the marginal-utility optimum `1/(1+b)` for the #27 scaling
  exponent b = 0.225 under deployment-frequency class weighting -- "optimal"
  if a basalt error matters in proportion to how often basalt occurs.
- **alpha = 0.5**: square-root sampling, the published long-tail default
  (Mahajan et al. 2018 used it outright; word2vec's 0.75 and XLM-R's 0.3
  bracket it).
- **hybrid**: take-all on availability-bound classes (avail <= 500), then
  alpha = 0.82 on the rest.

**Result** (`allocation_candidates.csv`, `class_abundance_powerlaw.png`).
Abundance is power-law shaped: rank-frequency slope -2.59, R2 0.948 over three
decades (gently curved: head ~ -1.1, tail ~ -3.2 -- a broken power law to the
pedantic eye). Temperature smoothing is family-preserving on such data
(quota slope = alpha * pool slope), which is what makes alpha the natural
single knob. The candidates:

| new bulks to | alpha=0.82 | alpha=0.5 | hybrid |
|---|---|---|---|
| big four | 16,303 | 10,287 | 14,671 |
| middle eight | 3,442 | 7,625 | 3,140 |
| bottom ten (avail <= 500) | 255 (0/10 exhausted) | 2,088 (8/10) | 2,189 (10/10) |
| smallest class ends at | 32 | 111 | 111 |

- **alpha=0.82** is theory-optimal for the frequency-weighted headline but
  isolates the rare classes: the bottom ten get ~250 new bulks combined and
  the smallest class gets zero, freezing it at 32 bulks -- the starvation the
  quota line item exists to prevent.
- **hybrid** maximizes both ends but the hard cutoff creates a non-monotone
  seam: foid-bearing trachyte (458 available) is taken 100% and ends at 502
  while the more abundant alkali feldspar rhyolite (835 available) ends at
  162. Final representation inverts against abundance at the boundary;
  patching that means hand-tuned monotonicity constraints.
- **alpha=0.5** is smooth and monotone by construction, delivers ~95% of the
  maximum possible tail rescue without any cutoff (availability saturates
  emergently), feeds the middle generously, and keeps the corpus
  abundance-ordered (basalt ends largest at 6,632).

Side checks: measured-Fe-speciation share is 22-58% in the tail vs 27-33% in
the big four, so the boost does not buy redox-blind data; and tail exhaustion
costs no future headroom -- gard2019 has no more of these classes to give, at
any scale.

**Decision** (owner, 2026-07-23). alpha = 0.5, availability-capped, via
`build_corpus_bulk_tables.py --expand 20000` -> `train_bulks_expansion.csv`.
Locked with the condition that the paper appendix shows this comparison and
the reasoning: 0.82 is "optimal" only under frequency weighting and starves
the tail; the hybrid's cutoff breaks monotonicity; 0.5 keeps the rescue,
the monotone ordering, and published precedent. The budget flexes with actual
weekend hours; the rule (alpha=0.5 + caps) is what is locked, not the 20k.
