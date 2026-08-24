# lineup_coherence

## Question

`sample_assemblages` draws each phase's presence Bernoulli independently, so
it can compose mineral lineups (support patterns) the corpus never pairs --
e.g. amp AND opx together when the corpus says they trade off. The magnitude
branch was trained teacher-forced on true corpus supports only, so off-menu
lineups are extrapolation, and in the #47 ranked scenario list every off-menu
sample steals weight from real scenarios. Step 0 of #47: how much sampled
assemblage mass lands on patterns that appear zero times among the bulk's own
truth draws, and zero times corpus-wide? If material, the mitigation already
recorded in heads.md -- restrict sampled lineups to corpus-observed patterns
and renormalize -- gets adopted in the #47 build.

## Method

`run_measurement.py`, on the val roster (222 bulks) with the production
checkpoint's presence probabilities (one rung-blind p-vector per bulk):

Under independent Bernoullis every exact pattern's probability is a
closed-form product, so off-menu mass w.r.t. a finite pattern library is
computed exactly (one library-by-bulk log-mass matmul), renormalized for the
sampler's empty-support rejection -- no Monte-Carlo noise. A 2000-draw MC
cross-check per bulk validates agreement with the deployed sampling scheme.

Three libraries per bulk: the bulk's own truth patterns (its ~128 valtest
draws x 2 rungs, pooled to match the rung-blind p), the train-tier library
(all patterns observed across the 10k training bulks -- what the mitigation
could restrict to at inference), and train + val-truth union (the literal
"zero times corpus-wide" reading). The mitigation's cost side is measured as
the fraction of each val bulk's truth draws whose pattern is absent from the
train library. Pattern identity is the exact 21-phase support; scenario
coarsening (#47 design decision 1) is deliberately downstream of this
measurement.

```
conda run -n mineral python explorations/lineup_coherence/run_measurement.py \
    --checkpoint checkpoints/joint_heads_supportmag_seed0/heads_best.pt
```

## Result

`results/lineup_coherence.json`. Libraries: 5763 unique train-tier patterns,
1710 val-truth, union 5871 (val truth adds only 108).

| mass | mean | median | p90 | max |
|---|---|---|---|---|
| off the bulk's own truth menu | 0.653 | 0.674 | 0.900 | 0.998 |
| off the train library (extrapolation) | 0.107 | 0.081 | 0.225 | 0.484 |
| off the union library | 0.105 | | | 0.481 |
| truth draws outside the train library (mitigation cost) | 0.006 | 0.000 | | 0.176 |

Off-train mass rises monotonically with ambiguity (count of phases at
intermediate p in (0.1, 0.9)): 0.001 (0-1 such phases, n=2) -> 0.028 (2-3)
-> 0.058 (4-5) -> 0.123 (6+, n=170); r(off-own, ambiguity) = 0.58. The
worst extrapolators are rhyolites (four of the top five, off-train 0.34-0.48,
9-11 intermediate phases). The two low-ambiguity bulks show near-total
off-own mass (0.98) with near-zero off-train mass: their confident modal
pattern is corpus-observed but differs from every one of their own 128
draws -- a prediction-error mode the coherence filter neither causes nor
fixes.

MC cross-check: max |MC - exact| = 0.030 over 222 bulks at 2000 draws
(binomial sd 0.011) -- consistent with sampling noise; the exact math matches
the deployed sampler.

## Conclusion

Incoherence is material by the issue's pre-registered criterion: on average
11% of sampler mass (p90 23%, worst 48%) lands on lineups never observed in
1M+ training rows -- pure extrapolation for the support-conditioned magnitude
branch, and dead weight in a ranked scenario list. The mitigation is nearly
free: restricting to the train-observed pattern library discards only 0.6% of
truth mass on average (one bulk reaches 18%; report the excluded share
per bulk rather than hiding it). The 5763-pattern library also makes the
mitigation's bonus exact: scenario weights become closed-form Bernoulli
products over the library (one [5763 x 21] matmul), removing Monte-Carlo
noise from the #47 ranking entirely. The dominant off-own component (~55
points of the 65%) is mass on corpus-plausible patterns that the bulk's own
finite draw menu doesn't contain -- much of it presumably accessory flicker,
which is the coarsening decision's territory, not the coherence filter's.

Decision itself recorded on #47 / journal 2026-07-22; this also realizes
mitigation 1 on #31's ladder (incoherent sets are removed exactly; the
exposure-bias distribution-shift component remains separately measured by
`exposure_bias_nll_delta`).
