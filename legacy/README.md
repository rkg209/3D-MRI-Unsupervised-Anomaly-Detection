# Legacy — reference only

This is the prior research repo (IE643_TensorTitan). Per **D1** it is **reference material only**:
port *logic*, never *structure*. Nothing here is imported by `src/mri_ad/`.

| Path | What it is |
|---|---|
| `metric-uad.ipynb` | The evaluation notebook — residual → threshold → Dice/IoU. Most important reference. |
| `UNETR/` | UNETR training notebooks. Best reported result (Dice ≈ 0.63). |
| `UNET/`, `attUNET/` | MONAI UNet / Attention-UNet training notebooks, loss sweep. |
| `GUI/` | Tkinter viewer. `GUI/utilities/utility.py:15` holds the **real** `UNETR_Reconstruction` class. |

## Defects found in this code — do NOT reintroduce

These were found during the pre-rebuild audit and are why several headline numbers are untrustworthy.
The `mri-domain` skill restates them so the agent never re-derives them wrong.

1. **The `.pth` files here are Git LFS pointer stubs (133 bytes) — they contain no weights.**
   Real weights are in the Google Drive folder linked from the root `README.md`.
2. **`metric-uad.ipynb`: `dice_scores`/`iou_scores` are re-initialized *inside* the per-subject loop.**
   Every reported "average Dice/IoU" is therefore a **single subject**, not an average.
3. **The BraTS segmentation is used raw** (multi-class labels 0/1/2/4) and **trilinear-interpolated**
   to fractional values, instead of binarized (`seg > 0`) with nearest-neighbour resize. Dice is
   consequently weighted by label magnitude — it is not a real Dice.
4. **Only depth-chunk index 4 of 8 is ever evaluated.** The other 7/8 of each volume is ignored.
5. **Thresholding is a hardcoded global `residual > 0.1`.** No percentile, no Otsu, no brain mask.
6. **Train and eval preprocessing differ.** OpenBHB: crop → resize → normalize.
   BraTS: normalize → transpose → *different* crop (depth not cropped) → resize. They must be unified.
7. **`3d-unetr-mse-ssim-aug.ipynb`** derives the seg path with `t2_path.replace('_t1.nii', '_seg.nii')`
   on a `_t2` path — so `seg_path == t2_path` and the model is scored against the MRI itself.
8. **PSNR/SSIM are imported but never computed.** There is no reconstruction-quality eval code to port.
9. **Perceptual loss is commented out everywhere.** No surviving code trained `3d_slices_MSE_Perceptual.pth`.
10. **The UNETR weights are NOT loadable into `monai.networks.nets.UNETR`** — see
    `specs/002-model-registry.md`. They came from the custom `UNETR_Reconstruction`, which has a
    different output head and a final `Sigmoid`.

## What is worth porting

- The preprocessing recipe (crop / resize / min-max / chunk to `(1,16,128,128)`) — **unified**.
- The exact model configs (channels, strides, `feature_size`, etc.).
- `SSIM_MSE_Loss` and `multi_scale_loss` — the two combination losses that actually ran.
- `UNETR_Reconstruction` itself, ported verbatim.
- The `matplotlib.cm.hot` residual-heatmap colorization from `GUI/tk_app.py`.
