# Architecture x loss study (Spec 005) — analysis

**Status: 0 of 9 declared cells evaluated.** Source of truth for every number below is
`artifacts/tables/arch_loss_matrix.json`; nothing here is computed, estimated, or recalled from a
planning document.

## Current state

The matrix declares `n_cells = 9` and reports `n_available = 0`. Every cell carries
`available: false` and a machine-readable `na_reason`, and the reasons are not uniform — they record
*different* kinds of absence, which is the point of tracking them separately. Four cells
(`unetr__mse`, `unetr__ssim`, `unetr__multiscale_mse`, plus, in a stronger form,
`unetr__perceptual`) are blocked on weights: `"no checkpoint"`, and for `unetr__perceptual`
specifically `"no training code, no checkpoint"` — that cell has no training path at all and will
stay `n/a` until one exists. Three cells (`unetr__mse_ssim`, `unet__mse`,
`attention_unet__mse_ssim`) read `"not evaluated"`. The diffusion paradigm cell
(`diffusion__ddpm`) reads `"untrained"`, and `stats.diffusion_available` is `false`. Spec 009 added
a ninth cell, `unetr_synth__mse_ssim` (the synthetic-anomaly fine-tune), which reads `"not
trained"` — it is a config-only row until the Spec 009 fine-tune actually runs (`/train`, GPU,
manual invoke only).

This is a fresh-repo snapshot, not a failure. `checkpoints/` and `data/` are user-supplied and are
not present on this machine, so the harness had no `artifacts/metrics/<model>__<loss>/aggregate.json`
to read for any cell. The renderer behaved correctly: it emitted the full 9-row declared matrix with
an explicit, attributed `n/a` per cell rather than silently dropping the rows. An incomplete matrix
that is honest about being incomplete is the correct artifact at this stage.

## Statistics

`stats.spearman_psnr_dice` is `null` with `n_pairs = 0`; the rendered table states
`Spearman rho: n/a (fewer than 3 evaluated cells, n_pairs=0)`. No correlation between reconstruction
fidelity and detection quality has been measured, in either direction.

`stats.best_cell_id` is `null` and `stats.best_dice` is `null`. **No best-configuration claim can be
made from this artifact.** Naming a winner now — including naming the configuration anyone expects
to win — would be fabrication.

## What this table is for

The project's central claim is that higher reconstruction fidelity does *not* buy better anomaly
detection: a model that rebuilds the input faithfully also rebuilds the tumor faithfully, so the
residual it is scored on goes quiet exactly where the lesion is. This table is the mechanism study
for that claim. By holding the detection metrics (Dice, IoU) and the fidelity metrics (PSNR, SSIM)
side by side across a (model x loss) grid, it is designed to make the relationship *visible* as a
shape in the matrix and *quantifiable* as the `spearman_psnr_dice` rank correlation over the
evaluated cells — an argument from measurement rather than an assertion. The loss axis is what
supplies the variation: MSE, SSIM, MSE+SSIM, and multi-scale MSE trade off fidelity differently, so
they should spread the cells along the PSNR axis and let the Dice response be read against it. The
`diffusion__ddpm` row asks the sharper version of the question — whether a generative prior resists
rebuilding the tumor or falls into the same failure mode as the deterministic autoencoders. That
row is currently `untrained` and the answer is unknown.

Two disciplines apply once cells populate. PSNR and SSIM are explanatory context only, as the
rendered table already annotates (NFR-6/NFR-22); they are never a headline number and never a
target, and a change that raises them while lowering Dice is a regression. And the direction of the
correlation must be reported as measured — if the anti-correlation does not appear, or appears
weakly across too few cells to support it, that is the finding.

## What would fill these cells

Filling a cell requires, per (model, loss) pair: real weights in `checkpoints/` and BraTS data in
`data/`, verified with `make check-data` (which detects the 133-byte Git LFS pointer stubs under
`legacy/` — those are not weights); then `make recon` to produce test-split `ReconResult`s, which is
GPU-bound and marked MANUAL ONLY; then `make eval` to score them into
`artifacts/metrics/<model>__<loss>/aggregate.json`; then a re-run of `make matrix`, which is CPU-only
and re-reads whatever aggregates exist. The `unetr__perceptual` cell additionally needs training code
that does not exist yet, and `diffusion__ddpm` needs a Spec 013 training run.

Every GPU-spending step above is user-invoked by rule. No agent launches `make recon`, `make train`,
or any paid run on its own initiative; the correct agent behavior is to propose the run and stop.
Comparisons drawn once these cells populate take the corrected Spec 004 baseline as their reference
point — not the prior report's Dice 0.6255, which came from buggy code.
