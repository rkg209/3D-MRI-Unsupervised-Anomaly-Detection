# Spec 004 · Evaluation harness

**Status:** implemented
**Depends on:** 003
**Fresh compute required:** no

---

## Problem

Every claim this project makes rests on Dice and IoU being computed correctly. **In the prior
work they were not.** The audit found four independent bugs in `legacy/metric-uad.ipynb`, each
sufficient on its own to invalidate the reported numbers. This spec builds the canonical metric
implementation, scores the full test split from saved `ReconResult` files, and — critically —
establishes an **honest new baseline** while explaining precisely how it differs from the number
the prior report published.

## Contract

`MetricsComputer` (scaffolded, `src/mri_ad/eval/metrics.py`) is the **single source of truth** for
Dice, IoU, PSNR, SSIM (NFR-5). No other module computes them.

- `VolumeEvaluator.evaluate(result, label) -> VolumeMetrics`
- `AggregateEvaluator.evaluate_split(results_dir: Path, labels_dir: Path) -> AggregateMetrics`
  — takes **Paths, never a model**. Passing a model raises `TypeError`. This enforces the
  inference/evaluation boundary: the harness never re-runs `forward()`.
- `ReportGenerator` → `artifacts/metrics/per_volume.csv`, `aggregate.json`, plots.

**PSNR and SSIM are explanatory context only** (NFR-6/NFR-22). They are never an optimization
target, never a headline number, and every table that shows them must label them as such.

## Acceptance tests

1. **Correct metrics.** Dice and IoU are computed against a **binarized** ground truth (`seg > 0`),
   resized **nearest-neighbour**, over **all** depth-chunks, accumulated across **all** subjects.
   Each of those four properties is asserted by its own unit test.
2. Dice and IoU on hand-constructed tensors match closed-form expected values, including the
   empty-mask edge cases (both empty → 1.0; one empty → 0.0).
3. `AggregateEvaluator.evaluate_split()` raises `TypeError` if handed a model instead of a path.
4. `make eval` scores the full test split from saved `ReconResult` files **without a single
   `forward()` call** — asserted by monkeypatching `forward` to raise.
5. Per-volume and aggregate metrics (mean ± std) are persisted to `artifacts/metrics/`.
6. **Legacy-compat mode.** `make eval LEGACY_BUG_COMPAT=1` reproduces the prior pipeline's
   behaviour — raw multi-class seg, trilinear label resize, chunk 4 only, `residual > 0.1`,
   single-subject accumulation — and lands **within ±0.02 Dice of the published 0.6255**.
7. **The honest baseline.** `make eval` (normal mode) publishes the corrected UNETR MSE+SSIM Dice
   and IoU as the new baseline, with the delta from 0.6255 attributed to specific bug fixes in
   `progress_report.md`.
8. `make report` runs in under 2 minutes with no GPU and no ML imports.

## Out of scope

The arch × loss matrix (005), the classical baseline (006), any training.

## Notes / deviations

**This spec replaces `FR-21`,** which requires reproducing the prior report's UNETR headline
(Dice ≈ 0.6255, IoU ≈ 0.4551) within ±0.005. That acceptance test must not be used, for two
independent reasons.

**First, the target number is not trustworthy.** `legacy/metric-uad.ipynb` contains four bugs:

1. `dice_scores`/`iou_scores` are re-initialized **inside** the per-subject loop, so the reported
   "average Dice" is the score of the **last subject only**, not an average over the split.
2. The BraTS segmentation is used **raw** — labels `{0,1,2,4}` — instead of binarized. The Dice
   numerator `(gt * pred).sum()` is therefore weighted by label magnitude: a tumor-core voxel
   (label 4) contributes four times an edema voxel (label 1). This is not a Dice coefficient.
3. That same segmentation is resized with **trilinear** interpolation, producing fractional
   "labels" (0.37, 1.84, …) that are then multiplied into the numerator.
4. Only **depth-chunk 4 of 8** is ever scored. Seven-eighths of every volume is ignored.

**Second, reproducing it is impossible by construction.** Spec 001 unifies the train and eval
preprocessing (which `planning/02-architecture.md:100` explicitly asks for), and Spec 003 replaces
the magic `0.1` threshold with a swept operating point. Both changes are correct, and both change
the number. A ±0.005 tolerance against a figure produced by different preprocessing, a different
threshold, a different ground truth, and a different number of subjects is not a reproducibility
test — it is a demand to reproduce a bug.

**What we do instead** is stronger, and it is the defensible thing to say in an interview:
we reproduce the prior number *in an explicit bug-compatibility mode* (acceptance test 6), which
proves we understand exactly where it came from; and we publish a corrected baseline
(acceptance test 7) with the discrepancy itemized. The corrected Dice may well be **lower** than
0.6255. That is fine, and it must be reported as-is. Publishing a number we know to be inflated
would defeat the entire purpose of a rigor-first rebuild (D2).
