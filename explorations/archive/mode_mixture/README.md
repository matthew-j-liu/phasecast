# mode_mixture

**Question.** The trained K=2 ModeHead dirichlet layer (mixture) wins the
corpus on both axes but leans against the same-dials K=1 control on
recalibrated real-rock val (-0.365 [-0.864, +0.118]). Why doesn't the
corpus advantage transfer -- and what should the canonical figures look
like at K>1?

**Method.** Three attribution experiments plus the K=2 figure mockups, all
against `joint_heads_allrock_mixture_seed0` and its control
`joint_heads_allrock_tuned_seed0` (shared all-rock encoder). Scripts assume
repo-root cwd and the `mineral` env.

- `run_instrument_transplant.py` -- corpus val rows scored through the
  real-rock 7-group coarsening (population fixed, instrument changed).
- `run_real_bimodality.py` -- is the real amp-share distribution two-humped
  at matched chemistry? Latent k-NN neighborhoods (unit-capped) with a
  1-vs-2-component GMM BIC test, paired against each neighborhood's nearest
  corpus val bulk; SiO2-window check without the encoder in the loop.
  Reusable as the #62 screening test for candidate benchmarks.
- `run_gate_offset_probe.py` -- one global gate-logit offset, matrix
  recalibrator refit per grid point on the train roster, selection on train
  NLL, paired val contrast at the selected offset.
- `run_mixture_figures.py` / `run_real_rock_card.py` -- the K=2 adaptations
  of the canonical figures (violin-share inference cards, component
  decomposition, amp K-curve beta overlay, engagement map, split scenario
  view, single-real-rock density card). Caches land in `work/` (gitignored,
  regenerate in ~2 min).

**Result.**

- Instrument: the mixture-minus-control teacher-forced dirichlet gap goes
  -0.4279 (21-phase) to -0.2618 (7-group) -- 61% survives; amp's marginal
  gap survives intact (-0.105 -> -0.104). The benchmark's coarseness is not
  the bottleneck.
- Nature: real neighborhoods prefer two humps 38.8% of the time (9.6%
  strongly, median dBIC -1.9; SiO2 windows agree) vs 100% of matched corpus
  bulks (73.7% strongly, median dBIC +26.3) -- with nearly identical total
  spread (median log-share std 0.305 real vs 0.312 corpus). Nature at
  matched chemistry has the corpus's width but ONE hump; the second hump
  belongs to the deliberately broad hidden-(P, H2O) proposal.
- Gate: train-selected offset -1.5 (mean gate weight on B 0.58 -> 0.26)
  takes recalibrated val 1.642 -> 1.415, contrast -0.138 [-0.615, +0.281]
  -- a tie, ~60% of the penalty recovered by one parameter; full collapse
  onto either component is worse than the partial shift and never reaches
  the control's 1.277 (components individually narrower than nature's one
  hump).

**Decision.** Mixture kept as the `mode_n_components` dial, K=1 canonical
(owner, 2026-08-04). The transfer lean is a simulation-side prior property,
not a model/procedure failure -- paper-commentary material. Figure
earmarks: `results/beta_overlay_k2_amp.png` is the appendix-grade exhibit
(bimodal corpus truth with the components on the humps);
`results/real_rock_card_k2.png` records the full-density-vs-measured-dot
card style preferred over point-and-whisker for main-paper figures.
Journal 2026-08-04 (two mixture entries); #49 closed, #62 carries the
screening test.
