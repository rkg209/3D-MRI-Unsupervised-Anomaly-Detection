# Spec 002 · Model registry & checkpoint loading

**Status:** approved
**Depends on:** 000, 001
**Fresh compute required:** no

---

## Problem

Three architectures (UNet, Attention-UNet, UNETR) must sit behind one interface so that the
reconstruction engine, the evaluation harness, and the comparison study never branch on model
type. This spec also resolves a defect that would otherwise silently poison every downstream
number: **the trained UNETR weights cannot be loaded into the model class the architecture doc
specifies.** Getting this wrong does not throw — it produces plausible-looking garbage.

## Contract

`AnomalyDetectionModel` (already scaffolded in `src/mri_ad/models/base.py`):
`forward(x) -> Tensor`, `load_checkpoint(path) -> None`, `model_card -> ModelCard`.

**Implementations:**

| Class | Backing | Notes |
|---|---|---|
| `UNetModel` | `monai.networks.nets.UNet` | `channels=(16,32,64,128,256)`, `strides=(2,2,2,2)`, `num_res_units=2`, `norm=BATCH`, `dropout=0.2`. No output activation. |
| `AttentionUNetModel` | `monai.networks.nets.AttentionUnet` | Same channels/strides, `kernel_size=3`. No output activation. |
| `UNETRReconstruction` | **ported**, MONAI-composed | `feature_size=32`, `img_size=(16,128,128)`, `hidden_size=768`, `mlp_dim=3072`, `num_heads=12`, `patch_size=(16,16,16)`, `norm_name='instance'`. **Ends in a Sigmoid** — the only model bounded to `[0,1]`. |

`ModelRegistry` — populated from `configs/model/*.yaml`; `get(name)` returns an
`AnomalyDetectionModel`; `register()` validates the interface and raises
`ModelRegistrationError` otherwise.

**`load_checkpoint` must:**
- accept **both** on-disk layouts: `{"model_state_dict": ...}` and a bare `state_dict`;
- load with `strict=True`;
- detect a **Git LFS pointer stub** and raise `CheckpointError` naming the file (all seven
  inherited `.pth` files are stubs);
- **never** fall back to `strict=False`.

## Acceptance tests

1. Every registered model loads its real checkpoint with `strict=True` and no missing or
   unexpected keys.
2. Every model runs a forward pass on a real `(B,1,16,128,128)` batch and returns that same shape.
3. `UNETRReconstruction`'s output is bounded to `[0,1]`; a test asserts `out.min() >= 0` and
   `out.max() <= 1`. The other two are *not* asserted to be bounded — downstream code must not
   assume they are.
4. Loading an LFS stub raises `CheckpointError`, and the message says the file is a pointer stub.
5. Both checkpoint layouts load: a test saves each format and loads it back.
6. Registering a class missing any interface method raises `ModelRegistrationError`.
7. Each model exposes a `model_card` with a correct `param_count` and its `training_loss`.
8. **Extension test:** adding a model touches only its module + a YAML + a checkpoint path —
   no change to `recon/`, `eval/`, or `classical/` (FR-14/NFR-15). Enforced by an import-graph test.

## Out of scope

Training (009), thresholding (003), metrics (004).

## Notes / deviations

**Deviation from `planning/02-architecture.md:139` and `03-system-design.md:42`,** which specify
`UNETRModel` as a wrapper over `monai.networks.nets.UNETR`. **We port the custom
`UNETR_Reconstruction` from `legacy/GUI/utilities/utility.py:15` instead.** Reason:

- Its output head is `nn.Conv3d(feature_size, 1, 1)` → state-dict keys `out.weight`, `out.bias`.
  MONAI's `UNETR` uses `UnetOutBlock` → `out.conv.conv.weight`. **The keys do not match.**
- It ends in `nn.Sigmoid()`; MONAI's `UNETR` has no output activation.
- It passes `pos_embed="perceptron"`, which modern MONAI renamed to `proj_type`.

Loading the trained weights into MONAI's class therefore fails outright, or — if someone reaches
for `strict=False` to make the error go away — loads partially and yields a model that
reconstructs noise while every metric still computes cleanly. That failure is silent, which is
what makes it dangerous.

**D6 still holds:** the ported class *is* MONAI-backed — it composes
`monai.networks.blocks.UnetrBasicBlock / UnetrPrUpBlock / UnetrUpBlock` and
`monai.networks.nets.ViT`. We adopt MONAI's building blocks, not a hand-rolled transformer.

If a MONAI-native `UNETR` is wanted later, it needs an explicit key-remapping shim
(`models/checkpoint_compat.py`) plus a numerical-equivalence test — not a `strict=False`.
