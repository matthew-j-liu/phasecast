# corpus_scaling

## Question

How does joint-heads quality scale with training-corpus size -- and does the
curve justify the ~10x NERSC regen (#27, feeding the #45 "draws/bulks
sizing" ledger line)? A phase whose metrics are still improving at N=10k
bulks is data-limited (more corpus will help); one that went flat is
architecture/loss-limited (more corpus is wasted on it). The prior two-point
line (47-chunk vs 100-chunk, 2026-07-17) moved garnet and ilmenite but not
spinel or the amphibole magnitude; this ladder replaces it with a proper
curve on the committed val roster.

## Method

- Nested train-bulk subsets 625 / 1250 / 2500 / 5000 / 10000 (full): sample
  ids sorted by md5 hex digest, prefix of length N -- platform-stable, and
  every smaller set is contained in the larger by construction. Class mix is
  approximately proportional at every N; rare classes thin out first, as
  they would in a genuinely smaller corpus.
- Each point retrains the standard config (`configs/heads/joint_heads.yaml`:
  joint heads + support-conditioned magnitude, 120 epochs, seed 0) with only
  the train tier subset changed, then runs the ModeHead interrogation on the
  fixed 222-bulk val roster.
- Optional second axis (`--draws 24,12`): all bulks at reduced
  draws-per-bulk. Matched-row comparisons (10k bulks x 24 draws vs 5k x 48;
  10k x 12 vs 2.5k x 48) separate more-bulks from more-paths-per-bulk -- the
  actual NERSC design question.
- Single seed per point; the full-N point doubles as a plumbing check
  against the production `joint_heads_supportmag_seed0` numbers.

Run from the repo root (env active):

```
python explorations/corpus_scaling/run_ladder.py                # ~1.5-2h streaming
python explorations/corpus_scaling/run_ladder.py --draws 24,12  # optional, ~+35 min
python explorations/corpus_scaling/aggregate.py
```

## Result

Run 2026-07-22/23, 7 points (5 bulk ladder + draws 24/12 at full bulks);
`results/scaling.json`, `scaling.png`, `scaling_draws.png`.

- Plumbing check: the full-N point reproduces the production
  `joint_heads_supportmag_seed0` metrics to 4 decimals on all three NLL
  components.
- Bulk axis, no saturation: val joint NLL -1.178 -> -2.456 across
  625 -> 10000, still gaining 0.24 nats on the last doubling. A saturating
  power law `L_inf + a*N^-b` fits within 0.02 nats everywhere
  (b = 0.225, L_inf ~ -3.9, floor poorly constrained); a pure log-line
  misses by 0.06 -- the per-doubling gain is slowly shrinking
  (0.42 -> 0.30 -> 0.31 -> 0.24).
- Draws axis, flat: 12 -> 24 -> 48 draws/bulk at full bulks spans 0.08
  nats non-monotonically (24 beats 48 by 0.035 -- the single-seed noise
  scale). At matched training rows, all-bulks-fewer-draws wins by huge
  margins (240k rows: -2.41 vs -1.90; 480k rows: -2.49 vs -2.21). The one
  monotone draws signal is CondHead (~0.06 nats/doubling), consistent
  with draws being the (P, H2O) variation itself; mode/comp and every
  highlighted phase are flat in draws.
- Per-phase verdicts: garnet data-limited (pf_r 0.66 -> 0.83 and still
  climbing); spinel weakly data-limited but structurally capped at ~0.70
  pf_r (the #46 redox input token is the lever, data secondary);
  amphibole not data-limited (pf_r flat ~0.93 across the ladder,
  magnitude MAE flat since 5000 -- the #25 mixture question, not data);
  ilmenite saturated since ~1250 bulks.

The mechanism behind the bulks/draws asymmetry: the model never sees
(P, H2O), so draws are unbiased Monte-Carlo samples of each bulk's outcome
distribution -- draw count only adds variance that pooling across
compositionally similar bulks averages out, while bulk count controls
coverage of input space, a bias no draw budget can fix. Most phases are
near-deterministic given the bulk (within-bulk path sensitivity
concentrates in amphibole and accessories), so extra draws mostly
re-observe the same assemblage.

## Decision

Regen sizing (the #45 "draws/bulks sizing" line): allocate to bulks, not
draws -- 24 draws/bulk is the knee (48 -> 24 costs nothing measurable on
mode/comp, ~0.06 nats on cond, and buys a bulk doubling worth ~0.25-0.3).
Recommended weekend shape (NERSC down, NeurIPS deadline, ~60 h laptop
budget at the measured ~4 draws/s): ~30k bulks x 24 draws, keeping the
existing 10k train bulks and the sealed 500-bulk valtest roster
(regenerated under the new proposal; valtest at 256 draws/bulk), rare-class
quotas on the new bulks. Caveat carried forward: the 24-draw knee was
measured under the current broad P/H2O proposal; the #48 dry run checks
whether the trace-conditional proposal changes per-draw redundancy.
Final bulk count flexes with actual weekend hours; locked by the owner on
#45, not here.
