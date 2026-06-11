# Decision 002: Paths explicitly rejected

**Date:** 2026-05-12
**Status:** Active

Things the team has already considered and rejected, with reasons. Don't re-pitch these without new evidence.

## CIPW norm as training target or auxiliary output
**Rejected.** CIPW is a deterministic calculator (pyrolite implements it). Training a model to predict CIPW = building a worse surrogate for a free function. Only residual use: optional sanity check on synthetic rocks during training-data generation (not at inference).

## Bulk-chemistry realism filter as assemblage realism
**Rejected.** complete.csv tells you the bulk is plausible. It does NOT tell you which of many possible assemblages corresponds to that bulk. By the inverse problem's definition, bulk → assemblage is 1-to-many; the bulk-realism filter only addresses bulk → bulk.

## Empirical assemblage mining from mineralML's Sample Name co-occurrence
**Rejected.** mineralML's training corpus is 98.3% single-mineral samples (35,062 of 35,677 by Sample Name). Only 615 multi-mineral samples. Useful as a smoke test, not as a primary signal.

## Validation_Data/PetDB_allminerals/ as a training source
**Rejected.** It's mineralML's test set (partial coverage, single source). Use the mineralML grain corpus from Zenodo (`data/grains/mineralml.csv`) instead — multi-source, curated, 88k rows.

## Using mineralML as a runtime dependency
**Rejected.** Vendor only the specific classifiers needed (FeldsparClassifier, PyroxeneClassifier, etc.) into `src/cnmineral/stoichiometry/`. Avoids version coupling and lets us hot-fix boundary rules without upstream PRs.

## Position B alone: T,P sweep emulator
**Rejected as primary framing.** Position B (CrabNet as MAGEMin-marginalized-over-T,P) is a Bayesian wrapper that adds amortization but not new information. Position C (MAGEMin × mineralML fusion) is the stronger contribution because it adds empirical composition realism that neither tool produces alone. Position B is the methodological scaffolding inside Position C, not a standalone story.

## Single-scalar G_sys for polymorph disambiguation
**Rejected.** A single G_sys output cannot distinguish Ky/Sil/And (they have different G but same bulk). For real polymorph disambiguation, need either (a) multi-output G_sys head, one scalar per polymorph candidate, or (b) conditional G_sys head taking candidate assemblage as input. Both are v2. In v1, G_sys is an encoder regularizer only.

## Brute-force enumeration of C(31, k) combinatorial space
**Rejected.** Real rocks live on a low-dimensional manifold defined by petrologic processes. Condition on rock class first (~15 IUGS buckets) → within-class assemblage is canonical (3-6 phases). Combinatorial size collapses by ~3 orders of magnitude.
