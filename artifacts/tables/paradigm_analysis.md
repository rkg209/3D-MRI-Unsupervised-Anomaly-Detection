# Paradigm comparison — Classical vs UNETR vs Diffusion (Spec 007) — analysis

**Status: 0 of 4 declared columns available.** Source of truth for every number below is
`artifacts/tables/paradigm_comparison.json`; nothing here is computed, estimated, or recalled from a
planning document or a prior report.

## Current state

The comparison declares `stats.n_columns = 4` and reports `stats.n_available = 0`. Every column
carries `available: false` with a machine-readable `na_reason`, and the reasons record
*different* kinds of absence. `classical` reads `"not run"` — the baseline is implemented but has not
been executed on a split. `unetr__mse_ssim` reads `"not evaluated"` — no test-split reconstructions
have been scored. `diffusion__ddpm` reads `"untrained"`, and `stats.diffusion_available` is `false`;
that paradigm has no checkpoint to evaluate because Spec 013's training run has not happened.
Spec 009 added a fourth column, `unetr_synth__mse_ssim` (the synthetic-anomaly fine-tune), which
reads `"not trained"` — config-only until the Spec 009 fine-tune actually runs. `source_paths` is
empty for all four, so no artifact was consumed to build any cell.

`split_hash` is `null` and the sample block is empty — `n_slices`, `n_positive`, and `prevalence` are
all `null`. Acceptance test 1 (identical split, asserted by comparing hashes) is therefore not yet
demonstrable: there is no split to hash. `artifacts/figures/paradigm_curves.png` exists as the
rendered frame for the overlaid ROC/PR curves, but with zero available columns it carries no curves.

This is a fresh-repo snapshot, not a failure. `checkpoints/` and `data/` are user-supplied and are
not present on this machine, so the renderer had nothing to read for any column. It behaved correctly:
it emitted the full four-column declared table with an explicit, attributed `n/a` per cell rather
than dropping the columns. An incomplete headline table that is honest about being incomplete is the
correct artifact at this stage.

## No winner claim

`stats.best_roc_auc_key` is `null` and `stats.best_dice_key` is `null`. **No best-paradigm claim can
be made from this artifact.** Naming a winner now — including naming the paradigm anyone expects to
win, or asserting that the classical baseline is or is not competitive — would be fabrication. The
spec's acceptance test 5 requires the narrative to identify where each approach wins and where each
fails; that test cannot be satisfied against zero data and is deferred, not quietly skipped.

## The common-footing mechanism

The reason this spec exists as more than a concatenation of Specs 005 and 006 is the reduction it
introduces. The classical baseline scores a 2D axial slice; it produces a per-slice anomaly score and
nothing finer, so it cannot say *where* in the slice the anomaly is — hence `localizes: false` in the
JSON and `n/a (no localization)` in the Dice and IoU rows of the rendered table. The DL models score
voxels; they produce a 3D anomaly mask and `localizes: true`. Placing a slice-level ROC-AUC beside a
voxel-level Dice and implying a ranking would be an apples-to-oranges table.

The fix is to push the DL models *down* to the classical model's granularity rather than to invent a
localization number for the classical model. Each DL model's voxel-level anomaly mask is reduced to a
single slice-level score — the fraction of flagged voxels in that slice — and a slice is predicted
anomalous when that fraction exceeds a threshold tuned on **validation**, never on test. The identical
reduction and the identical threshold procedure are applied to both DL columns, so UNETR and Diffusion
each gain a slice-level ROC-AUC and PR-AUC that sit on the same axis as the classical model's. Voxel-
level Dice and IoU stay DL-only and are marked as having no classical counterpart.

**The granularity caveat holds regardless.** The reduction makes the ROC-AUC row comparable across all
three columns; it does not make the ROC-AUC row comparable to the Dice row. Slice-level AUC and voxel-
level Dice measure different things — detection of an affected slice versus spatial agreement with a
lesion mask — and they are not interchangeable. A high slice-level AUC is compatible with a poor Dice:
a model can reliably notice that something is wrong in a slice while flagging the wrong voxels within
it. Any reading of the completed table has to keep the two rows separate, and the caveat is repeated
in the rendered table's `granularity_note` for exactly that reason.

## What this table is for

This is the project's headline artifact, and its purpose is the comparative framing rather than a
score. The central domain fact is that a reconstruction model which rebuilds the input faithfully also
rebuilds the tumor faithfully, so the residual it is scored on goes quiet exactly where the lesion is;
better PSNR/SSIM bought worse detection in the prior work. The three columns probe that failure from
three directions. The classical column does not reconstruct at all — it detects from engineered
features, so it is structurally immune to the failure mode and functions as the floor that any
reconstruction pipeline has to clear to justify its cost. The UNETR column is the deterministic
reconstruction paradigm whose patch-based features constrained the failure best in prior work. The
diffusion column asks the sharper question: whether a *generative* prior resists rebuilding the tumor
or falls into the same trap as the deterministic autoencoders.

Fidelity metrics are deliberately absent from this table; PSNR and SSIM live in the Spec 005 matrix,
where they serve as explanatory context for the anti-correlation and never as a headline or a target.
Here the only question is detection. If the completed table shows the classical baseline matching or
beating the deep pipelines on slice-level ROC-AUC, that is a result to report plainly — it is an
interesting finding about the cost-effectiveness of the deep paradigm, not an embarrassment.

## What would fill each column

**Classical.** `make classical` (Spec 006, `scripts/run_classical.py`). CPU-only — no GPU spend, no
checkpoint needed — but it requires BraTS data in `data/`, verified with `make check-data`. This is
the cheapest column to populate and the one that should come first.

**UNETR.** Real weights in `checkpoints/` and BraTS data in `data/`, both verified with
`make check-data` (which detects the 133-byte Git LFS pointer stubs under `legacy/` — those are not
weights, and `strict=False` is never the workaround). Then `make recon` to produce test-split
`ReconResult`s, `make eval` to score them to voxel-level Dice/IoU, and `make slice-scores`
(`scripts/run_slice_reduction.py`) to apply the slice-level reduction to the saved reconstructions.
`make recon` is GPU-bound and **MANUAL ONLY**; `make eval` and `make slice-scores` read saved `.pt`
outputs and need no GPU.

**Diffusion (AnoDDPM).** Everything UNETR needs, plus a checkpoint that does not exist: Spec 013's
training run is `/train`-gated fresh compute and must happen before this column can be evaluated at
all. Until then `"untrained"` is the accurate cell.

Once any column populates, `make paradigm` (`scripts/run_paradigm_comparison.py`) re-renders the table
and curves from whatever artifacts exist; it is CPU-only and idempotent.

Every GPU-spending step above is user-invoked by rule. No agent launches `make recon`, `make train`, or
any paid run on its own initiative; the correct agent behavior is to propose the run and stop. Any
comparison drawn once these columns populate takes the corrected Spec 004 baseline as its reference
point — not the prior report's Dice 0.6255, which came from buggy code.
