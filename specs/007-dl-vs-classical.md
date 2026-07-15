# Spec 007 · Paradigm comparison — Classical vs UNETR vs Diffusion

**Status:** approved
**Depends on:** 005, 006, 013
**Fresh compute required:** no (the diffusion checkpoint comes from 013's `/train` run)

---

## Problem

Three paradigms, one test split, one honest comparison. This is the project's **headline table**:
engineered features + gradient boosting (classical), transformer reconstruction (UNETR), and a
generative denoising prior (diffusion / AnoDDPM). The temptation here is to present a DL Dice next
to a classical ROC-AUC and imply a winner — but they measure different things at different
granularities, and a reader who does not catch that is being misled. This spec puts them on a
common footing where it can, and states the caveat loudly where it cannot.

## Contract

One artifact: `artifacts/tables/paradigm_comparison.md` + overlaid ROC/PR curves + a written
narrative from the `results-analyst` agent.

**Three columns:** Classical, UNETR, Diffusion (AnoDDPM).

**Common footing.** To compare fairly, each DL model's voxel-level anomaly mask is *also* reduced
to a slice-level decision (a slice is predicted anomalous if its flagged-voxel count exceeds a
validation-tuned threshold), giving both UNETR and Diffusion a slice-level ROC-AUC directly
comparable to the classical model's. Voxel-level Dice is reported for the **DL models only**, and
marked as having no classical counterpart.

## Acceptance tests

1. All three approaches are evaluated on the **identical** test split from `SplitContract` —
   asserted by comparing split hashes, not by eye.
2. A side-by-side table with a **slice-level ROC-AUC and PR-AUC for all three**, and voxel-level
   Dice/IoU for UNETR and Diffusion only, with the classical cells marked `n/a (no localization)`.
   The diffusion column reads `n/a (untrained)` if Spec 013 has not been run.
3. Overlaid ROC and PR curves on one figure, all available models, same axes.
4. The narrative explicitly states the granularity caveat (FR-32): slice-level AUC and voxel-level
   Dice are **not** interchangeable, and the classical model does not localize.
5. The narrative identifies where each approach wins and where each fails — including, if it is
   what the data shows, that the classical baseline is competitive, or that diffusion does **not**
   beat UNETR. A result that undercuts any approach is reported, not buried.
6. Every number regenerates via `make report` from saved artifacts.

## Out of scope

New models, new features, any training (the diffusion model is trained in 013, `/train`-gated).

## Notes / deviations

- **Renamed and widened from the original "DL-vs-classical" two-way comparison** when the project
  reframed its headline from "UNet vs Attention-UNet vs UNETR" to "Classical vs UNETR vs Diffusion".
  See the D3 update in `CLAUDE.md` and `progress_report.md`. `FR-30/FR-31` originally scoped a
  two-way DL-vs-classical table; this spec supersedes that with a three-paradigm table on the same
  split.
- The DL→slice-level reduction in the Contract is an addition beyond `FR-30`, which only requires
  "the same test split". Same-split-different-metric would still be an apples-to-oranges table. The
  reduction gives one genuinely comparable number (slice ROC-AUC) while keeping Dice for the thing
  only the DL models can do. The reduction threshold is tuned on **validation**, never test — and
  the identical reduction is applied to both DL models.
