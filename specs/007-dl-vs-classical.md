# Spec 007 · DL-vs-classical comparison

**Status:** approved
**Depends on:** 005, 006
**Fresh compute required:** no

---

## Problem

Two paradigms, one test split, one honest comparison. The temptation here is to present a
DL Dice next to a classical ROC-AUC and imply a winner — but they measure different things at
different granularities, and a reader who does not catch that is being misled. This spec puts them
on a common footing where it can, and states the caveat loudly where it cannot.

## Contract

One artifact: `artifacts/tables/dl_vs_classical.md` + overlaid ROC/PR curves + a written narrative
from the `results-analyst` agent.

**Common footing.** To compare fairly, the DL pipeline's voxel-level anomaly mask is *also*
reduced to a slice-level decision (a slice is predicted anomalous if its flagged-voxel count
exceeds a validation-tuned threshold), giving DL a slice-level ROC-AUC directly comparable to the
classical model's. Voxel-level Dice is reported for DL **only**, and marked as having no classical
counterpart.

## Acceptance tests

1. Both approaches are evaluated on the **identical** test split from `SplitContract` — asserted
   by comparing split hashes, not by eye.
2. A side-by-side table with a **slice-level ROC-AUC and PR-AUC for both**, and voxel-level
   Dice/IoU for DL only, with the classical cells marked `n/a (no localization)`.
3. Overlaid ROC and PR curves on one figure, both models, same axes.
4. The narrative explicitly states the granularity caveat (FR-32): slice-level AUC and voxel-level
   Dice are **not** interchangeable, and the classical model does not localize.
5. The narrative identifies where each approach wins and where each fails — including, if it is
   what the data shows, that the classical baseline is competitive. A result that undercuts the DL
   pipeline is reported, not buried.
6. Every number regenerates via `make report` from saved artifacts.

## Out of scope

New models, new features, any training.

## Notes / deviations

The DL→slice-level reduction in the Contract is an addition beyond `FR-30`, which only requires
"the same test split". Same-split-different-metric would still be an apples-to-oranges table. The
reduction gives one genuinely comparable number (slice ROC-AUC) while keeping Dice for the thing
only DL can do. The reduction threshold is tuned on **validation**, never test.
