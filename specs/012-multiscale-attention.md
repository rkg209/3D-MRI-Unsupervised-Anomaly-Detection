# Spec 012 · Multi-scale attention variant (STRETCH)

**Status:** stretch — do not start until 000–011 are complete and defensible (C-7)
**Depends on:** 002, 005, 009
**Fresh compute required:** **YES — `/train` only, manual invoke**

---

## Problem

The prior work listed multi-scale attention as future work. Anomalies span scales — a small lesion
and a large mass need different receptive fields — so attention over multiple scales *might*
improve localization. This is explicitly **off the critical path**: it is an architectural
experiment, and the project is defensible without it. Note that the diffusion model (Spec 013) now
occupies the headline "third paradigm" novelty slot alongside synthetic-anomaly training (009);
this stretch spec remains an optional architectural add-on, not a comparison member.

## Contract

Add a multi-scale attention variant of the best-performing model behind the existing
`AnomalyDetectionModel` interface. Zero changes to `recon/`, `eval/`, or `classical/` — if this
spec requires touching them, the Spec 002 abstraction has failed and *that* is the finding.

## Acceptance tests

1. The variant registers, loads, and runs a forward pass through the same interface as every
   other model — no special-casing anywhere downstream.
2. It appears as a new row in the Spec 005 comparison table, scored on the identical test split
   with the identical threshold strategy.
3. **Or:** a documented decision to defer, recorded in `progress_report.md` with the reason. That
   is an acceptable and expected outcome.
4. If it does not beat the baseline, that is reported. No architecture is kept for its novelty.

## Out of scope

Everything until 000–011 are done (C-7). This spec must not block the definition of done.

## Notes / deviations

Deferring this is the **expected** outcome given the compute budget. A finished, honest study
without it beats an unfinished one with it. The one thing that would make this spec worth the GPU
hours is if Spec 005 shows the fidelity/detection anti-correlation is *scale-dependent* — that
would give a real hypothesis to test rather than an architecture to try.
