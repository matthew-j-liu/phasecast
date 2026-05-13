# Decision 001: v1 paper scope

**Date:** 2026-05-12
**Status:** Active

## Decision

v1 paper = Position C training corpus + unimodal CrabNet with 4 heads (mode, comp, gsys-as-regularizer, class) + Dirichlet output + deep ensemble for uncertainty + mineralML classifiers as deterministic post-processor for canonical naming.

## Alternatives considered

- **ALIGNN structural stream in v1.** Rejected: Position C training data is already structured (real-rock chemistry + MAGEMin equilibrium + empirical compositions). Structure's marginal value is low. Reserve it for the v2 polymorph branch where ALIGNN distinguishes Ky/Sil/And by structure (chemistry is identical).
- **Multi-output G_sys per polymorph in v1.** Rejected: requires polymorph-forced MAGEMin runs (3-5× cost), and the disambiguation story is more compelling when paired with ALIGNN. Defer the whole polymorph branch to v2.
- **Single-output paper without composition head.** Rejected: mode-only CrabNet ≈ faster CIPW. The composition head is what justifies the contribution and enables SS endmember prediction.

## Why this scope

Four-axis differentiation table is the contribution: bulk-only input + no spot analyses + multimodal output + hydrous/sedimentary coverage. v1 delivers all four. Adding ALIGNN or multi-output G_sys doesn't widen the differentiation table — it only enables the polymorph axis, which is a separate (worthy) v2 story.

## Revisit when

- v1 trained and benchmarked. If composition predictions are good but ambiguous on polymorph pairs (Qz/Crs/Trd in evolved rocks; rare in our likely test set), v2 priority confirmed.
