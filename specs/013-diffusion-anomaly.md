# Spec 013 · Diffusion-based anomaly detection (AnoDDPM)

**Status:** draft
**Depends on:** 002, 003, 004
**Fresh compute required:** **YES — the second fresh-compute spec (with 009). `/train` is manual-invoke only.**

---

## Problem

The prior work compared three deterministic reconstruction models (UNet, Attention-UNet, UNETR)
that differ only marginally — all learn a direct healthy→healthy mapping and all share the central
failure mode: they reconstruct tumors too well, so the residual at the tumor is small and detection
fails. A **diffusion model** learns "normal" anatomy as a *generative* prior over healthy brains
rather than a direct reconstruction map. This spec adds AnoDDPM as a third, genuinely different
paradigm alongside classical ML (006) and UNETR, and asks a sharper version of the central
question: does the generative prior resist rebuilding the tumor, or fall to the same failure?

## Contract

**Model: AnoDDPM.** A DDPM trained only on healthy OpenBHB volumes. At inference, the input volume
is partially noised to a configurable timestep `t_noise` and then denoised back through the reverse
process to a *healthy estimate*; the residual `|input − healthy_estimate|` is the anomaly signal.

**It conforms to the existing `AnomalyDetectionModel` interface (Spec 002)** so that `recon/` (003),
`eval/` (004), and `classical/` (006) are **untouched**:

```python
DiffusionADModel(AnomalyDetectionModel)
  forward(x: Tensor) -> Tensor    # partial-noise then denoise; returns healthy estimate
  load_checkpoint(path) -> None   # loads trained DiffusionModelUNet weights; strict=True
  model_card -> ModelCard
```

- Backing: MONAI `monai.networks.nets.DiffusionModelUNet` (`spatial_dims=3`) + a MONAI scheduler
  (`DDPMScheduler`; `DDIMScheduler` optional for faster sampling). **D6 holds — MONAI-backed.**
- `forward(x)` accepts and returns shape `(B, 1, 16, 128, 128)`; the healthy estimate is bounded to
  the model's output range and is what `recon/` diffs against — no thresholding here (that stays in
  003).
- **No checkpoint exists until trained.** `load_checkpoint` raises `CheckpointError` naming the file
  if the weights are absent or an LFS stub; **never** `strict=False`.
- Config `configs/model/diffusion.yaml`: noise schedule, `num_train_timesteps`, inference
  `t_noise` (partial-noise level), sampler, seed inherited. No hyperparameter in code.

**Training objective:** standard DDPM denoising loss on healthy OpenBHB volumes. Trained from
scratch (no existing checkpoint to fine-tune from, unlike 009).

## Acceptance tests

1. `forward` on a real `(B,1,16,128,128)` batch returns that same shape; the healthy estimate is
   finite and within the documented output range.
2. **Extension guarantee.** Adding this model touches only its module + `configs/model/diffusion.yaml`
   + a checkpoint path — no change to `recon/`, `eval/`, or `classical/`. Enforced by the same
   import-graph test as Spec 002.8 / 012.1. If this spec forces an edit there, the Spec 002
   abstraction has failed and *that* is the finding.
3. The model appears as a row in the Spec 005 study and the Spec 007 comparison, scored on the
   **identical** test split with the **identical** threshold strategy as every other model.
4. `load_checkpoint` raises `CheckpointError` (not a silent partial load) when weights are missing
   or a stub.
5. All generation/training parameters (schedule, timesteps, `t_noise`, sampler, seed) are logged in
   `run_meta.json` for every training run (FR-36 analogue).
6. Training is reachable **only** via `/train` or `make train`; the agent never launches it (C-3).
7. The test split is **never** seen during training — asserted against `SplitContract`.
8. **If diffusion does not beat UNETR on Dice/IoU, that is reported as the result.** A negative
   outcome is a finding, not a failure to hide — the whole point is an honest paradigm comparison.

## Out of scope

Fine-tuning diffusion on FPI-corrupted data (that is a possible future extension, not this spec —
009 owns FPI, on UNETR). Latent diffusion or a learned VAE stage. Any change to the reconstruction
engine, evaluation harness, or classical baseline.

## Notes / deviations

- **This spec did not exist in `3d-mri-anomaly-detection-sdd-plan.md` §5.** It is added when the
  project reframed its headline comparison from "UNet vs Attention-UNet vs UNETR" (three flavours of
  one paradigm, already done in the prior work) to "**Classical vs UNETR vs Diffusion**" (three
  distinct paradigms). See the D3 update in `CLAUDE.md` and the entry in `progress_report.md`.
- Numbered **013**, not the deliberately-empty 008 slot (`specs/README.md` explains why 007→009
  must not be renumbered).
- **Budget guardrail (C-3).** This is the second fresh-compute spec (with 009). Training a diffusion
  model from scratch is the most GPU-hungry item in the project. The agent must never launch it,
  a sweep, or any paid run autonomously. `/train` carries `disable-model-invocation: true`.
- **Compute estimate, sampler choice, exact schedule, and whether the pinned MONAI version ships
  `DiffusionModelUNet`/`DDPMScheduler` (they landed when MONAI Generative merged into core) are
  resolved during this spec's `/plan` step** — confirm via `lit-scout` and bump `pyproject.toml` if
  needed.
- PSNR/SSIM of the diffusion reconstruction remain **explanatory context only**, never a target
  (NFR-6) — the same rule that governs every other model here.
