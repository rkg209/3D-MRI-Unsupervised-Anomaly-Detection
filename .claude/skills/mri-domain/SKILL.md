---
name: mri-domain
description: MRI anomaly-detection domain conventions — the (1,16,128,128) shape contract, the reconstruct-the-tumor failure mode, canonical Dice/IoU/PSNR/SSIM definitions, the OpenBHB/BraTS preprocessing recipe, and the specific metric bugs the prior work shipped. Use whenever writing or reviewing anything that touches MRI volumes, residual maps, anomaly masks, segmentation labels, or evaluation metrics.
---

# MRI anomaly-detection domain knowledge

## The failure mode this whole project is about

Train a reconstruction model on healthy brains. At inference, feed it a tumor brain. The *intent*
is that it cannot reconstruct the tumor, so the residual `|original − reconstruction|` spikes there.

**What actually happens:** a sufficiently powerful model reconstructs the tumor faithfully too —
tumors are spatially coherent structures it can simply copy. The residual stays small and the
signal vanishes. UNETR is the best performer *because* its patch-based ViT features reconstruct
less perfectly.

**Therefore: optimize detection separability (Dice/IoU), never reconstruction fidelity
(PSNR/SSIM).** Higher fidelity actively hurts. If a change improves PSNR and lowers Dice, it is a
regression. PSNR/SSIM appear in reports **only** to demonstrate this anti-correlation.

## The shape contract

`(1, 16, 128, 128)` = `(C, D, H, W)`; batched `(B, 1, 16, 128, 128)`; float32; values in `[0, 1]`.
This is a system-wide invariant. A tensor deviating from it at a module boundary is a **bug**, not
a configuration option.

## Preprocessing

One pipeline, used by **both** datasets (the prior work used two different ones — that was a bug):

```
load → orient to (D,H,W) → crop → trilinear resize H,W to 128
     → min-max normalize to [0,1] (eps 1e-8) → deterministic non-overlapping depth chunks of 16
```

- **OpenBHB** (healthy T1, training): `.npy`, source `(1,1,182,218,182)`. Crop z[50:130],
  y[20:160], x[20:196] → `(80,140,176)` → 5 chunks.
- **BraTS** (tumor, eval): `.nii`, source `(240,240,155)`. **T2 modality only.** Segmentation is
  a sibling file.

**Segmentation labels follow the same geometry but with different rules:**
- Resize **NEAREST**, never trilinear.
- **Binarize: `seg > 0`.** BraTS labels are `{0, 1, 2, 4}` (edema / non-enhancing / enhancing).

Never randomize the eval path (no `RandSpatialCrop`) — it breaks reproducibility.

## Canonical metrics

Defined once in `src/mri_ad/eval/metrics.py`. Never reimplement them anywhere else.

```
Dice = 2|pred ∩ gt| / (|pred| + |gt| + eps)      eps = 1e-6
IoU  = |pred ∩ gt| / (|pred ∪ gt| + eps)
PSNR = 20·log10(1.0 / sqrt(mse))                 assumes [0,1] data range — CONTEXT ONLY
SSIM = monai.metrics.SSIMMetric                                            — CONTEXT ONLY
```

Both `pred` and `gt` must be **binary** before they touch these. Empty-mask convention: both empty
→ 1.0; one empty → 0.0.

## The five bugs the prior work shipped — never reintroduce them

These invalidated the published Dice ≈ 0.6255. They are subtle and easy to write again by accident.

1. **Score lists re-initialized inside the per-subject loop.** Every reported "average Dice" was
   actually the **last subject's** score. Accumulate across all subjects.
2. **Raw multi-class segmentation used as ground truth.** `(gt * pred).sum()` with labels
   `{0,1,2,4}` weights a tumor-core voxel 4× an edema voxel. That is not a Dice coefficient.
   Binarize first.
3. **Segmentation trilinear-interpolated**, producing fractional "labels" (0.37, 1.84…) that then
   get multiplied into the Dice numerator. Use nearest-neighbour.
4. **Only depth-chunk 4 of 8 evaluated.** Seven-eighths of every volume was ignored.
5. **Threshold hardcoded to `residual > 0.1`** in a notebook cell. Thresholding belongs in
   `recon/`, configured, and chosen by a sweep **on validation** — never tuned on test.

Also: PSNR/SSIM were imported but never computed, and the perceptual loss is commented out
everywhere (so `3d_slices_MSE_Perceptual.pth` has no surviving training code — it is probably
mislabelled multi-scale).

## Models

| Model | Output activation | Note |
|---|---|---|
| UNet | none | MONAI `UNet`, `channels=(16,32,64,128,256)` |
| Attention-UNet | none | MONAI `AttentionUnet` |
| **UNETR** | **Sigmoid** | Ported `UNETRReconstruction` — **NOT** `monai.networks.nets.UNETR`; the state-dict keys differ and it will not load. |

Only UNETR bounds its output to `[0,1]`. Do not assume the others do.
