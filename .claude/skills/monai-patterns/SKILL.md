---
name: monai-patterns
description: The repo's house style for MONAI and PyTorch — transform composition, CacheDataset usage, deterministic dataloaders, how each model is instantiated, checkpoint loading that handles both on-disk formats, and device handling. Use when writing or reviewing any dataset, transform, model, or training/inference loop.
---

# MONAI house style

D6: adopt MONAI for 3D transforms, datasets, and backbones. Don't hand-roll what MONAI provides.

## Transforms

Compose dict-based transforms; keys are `"image"` and (BraTS only) `"label"`. Image and label take
the **same geometry but different interpolation** — label is always `nearest` and is binarized.

```python
Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),   # image only — never scale a label
    Resized(keys=["image"], spatial_size=(-1, 128, 128), mode="trilinear"),
    Resized(keys=["label"], spatial_size=(-1, 128, 128), mode="nearest"),
    Lambdad(keys=["label"], func=lambda x: (x > 0).float()),  # BraTS labels are {0,1,2,4}
])
```

Never put a random transform on the eval path — it breaks NFR-1.

## Datasets

`CacheDataset(cache_rate=...)`: `1.0` on the cluster (preprocessed volumes fit in RAM), `0.0` on a
laptop. Set it from config, never in code.

## Deterministic dataloaders

```python
DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
           shuffle=False,                       # never shuffle on eval
           generator=torch.Generator().manual_seed(cfg.seed),
           worker_init_fn=monai.data.utils.worker_init_fn)
```

## Models

```python
# UNet / AttentionUnet — MONAI natives, no output activation
UNet(spatial_dims=3, in_channels=1, out_channels=1,
     channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
     num_res_units=2, norm=Norm.BATCH, dropout=0.2)

AttentionUnet(spatial_dims=3, in_channels=1, out_channels=1,
              channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), kernel_size=3)

# UNETR — the PORTED class, composing MONAI blocks. NOT monai.networks.nets.UNETR.
UNETRReconstruction(in_channels=1, img_size=(16, 128, 128),
                    feature_size=32, norm_name="instance")   # ends in Sigmoid
```

The ported UNETR composes `monai.networks.blocks.UnetrBasicBlock / UnetrPrUpBlock / UnetrUpBlock`
and `monai.networks.nets.ViT`, so D6 holds. It exists because the trained weights use an
`nn.Conv3d` output head (`out.weight`) where MONAI uses `UnetOutBlock` (`out.conv.conv.weight`) —
the keys do not match. See `specs/002-model-registry.md`.

## Checkpoint loading

Two formats exist in the wild. Handle both, load `strict=True`, and refuse LFS stubs:

```python
raw = torch.load(path, map_location="cpu")
state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
model.load_state_dict(state, strict=True)   # NEVER strict=False
```

`strict=False` on a key mismatch gives you a partially-random model that still produces clean-
looking metrics. That is the worst possible failure — it is silent.

## Device

`DeviceManager.get_device()` → cuda → mps → cpu. Move the model **and** every tensor. Call
`model.eval()` and wrap inference in `torch.no_grad()` — the legacy GUI computed a device and then
never used it, running everything on CPU.

## Training

Adam `lr=1e-4`, `weight_decay=1e-5`; 50 epochs; batch 8; early stopping patience 10;
`ReduceLROnPlateau(factor=0.5, patience=5)`. (The old README says 100 epochs — the notebooks say 50.)

**Never launch training autonomously.** It is `/train`, manual invoke only.
