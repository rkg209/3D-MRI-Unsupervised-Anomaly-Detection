# Spec 005 · Fidelity-vs-detection study (architecture × loss + diffusion)

**Status:** implemented
**Depends on:** 004 (002 for the diffusion row; 013 to fill its cell — else `n/a (untrained)`)
**Fresh compute required:** no (except any cell we choose to fill by training — that is gated)

---

## Problem

The project's central claim is that **higher reconstruction fidelity does not buy better anomaly
detection** — in fact it hurts. That claim is only credible if it is demonstrated across a matrix,
not asserted from one data point. This spec is the *mechanism* study (the **why**): it orchestrates
reconstruction models × losses into one generated table and one written analysis, with PSNR/SSIM
shown *alongside* Dice/IoU precisely so the anti-correlation is visible. The headline three-paradigm
comparison (Classical vs UNETR vs Diffusion) lives in Spec 007; here the diffusion model earns its
place by testing whether a *generative* prior escapes the fidelity-vs-detection anti-correlation
that the deterministic reconstruction models suffer.

## Contract

Matrix rows, all reconstruction-based:
- **UNETR × {MSE, SSIM, MSE+SSIM, multi-scale MSE, perceptual}** — the core fidelity-vs-detection
  evidence (the headline DL anchor swept across losses).
- **UNet, Attention-UNet** — kept as prior-work **reference rows** (whatever loss cells have saved
  checkpoints), not the headline; they show the anti-correlation is not a UNETR artefact.
- **Diffusion (AnoDDPM)** — one row (its DDPM objective, not the loss sweep), included to test
  whether the generative prior changes the PSNR↔Dice relationship.

Output: `artifacts/tables/arch_loss_matrix.{csv,md}` — Dice, IoU, PSNR, SSIM per cell, all sourced
from saved `ReconResult` files via Spec 004's harness. Plus a written analysis authored by the
`results-analyst` agent.

**Unavailable cells render as `n/a (<reason>)`.** They are never left blank, never dropped, and
never filled with a plausible-looking number. (Diffusion cells read `n/a (untrained)` until Spec 013
is run.)

## Acceptance tests

1. One generated table covering the matrix; every cell is either a real number or an explicit
   `n/a (<reason>)`.
2. Every number traces to a `ReconResult` under `artifacts/results/recon/<run_id>/` and a
   `run_meta.json`. A test asserts no number is hardcoded in the table generator.
3. The table reports Dice/IoU **and** PSNR/SSIM, with PSNR/SSIM explicitly labelled as
   explanatory context, not performance (NFR-22).
4. The analysis states the fidelity-vs-detection relationship **quantitatively** — e.g. the rank
   correlation between PSNR and Dice across cells — rather than asserting it qualitatively, and
   states whether the diffusion row sits on or off that trend (does the generative prior escape it?).
5. The analysis names the best combination and does not claim a cell we did not measure.

## Out of scope

Training new cells (that is `/train`, manual only — UNETR loss cells, and the diffusion model via
Spec 013), the classical baseline (006), the three-paradigm headline table (007).

## Notes / deviations

**`FR-23` mandates a complete matrix including a perceptual-loss column. That column cannot be
filled from what exists.** In the prior repo the perceptual-loss code is **commented out in every
notebook**, and `legacy/UNET/3d_slices_MSE_Perceptual.pth` is an LFS stub. There is no surviving
code and no surviving checkpoint for it — despite the filename. (The live `criterion` in that
notebook is actually `multi_scale_loss`, so the checkpoint is probably mislabelled.)

Options, in order of preference:
1. Render the perceptual column `n/a (no training code, no checkpoint)` and say so in the analysis.
2. If the weights turn up in the Google Drive folder, evaluate them and fill the column.
3. Train it — **fresh GPU compute, therefore `/train` only, manual invoke, user-initiated.**

An incomplete matrix that is *honest about being incomplete* is a stronger artifact than a
complete one containing a fabricated cell. Do not fabricate the cell.
