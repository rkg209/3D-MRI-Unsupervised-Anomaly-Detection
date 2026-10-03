# Spec 001 · Data layer

**Status:** implemented
**Depends on:** 000
**Fresh compute required:** no

---

## Problem

Every downstream number depends on the volumes being preprocessed identically and the split being
stable. The prior work failed both tests: it applied **different preprocessing to training and
evaluation**, and its split was not recorded. This spec codifies one MONAI-backed pipeline used by
both datasets, a deterministic split contract, and a validator that fails loudly on drift — so a
shape or intensity regression is caught at dataloader init rather than silently corrupting a Dice
score four specs later.

## Contract

**Datasets** (`src/mri_ad/data/`):

- `OpenBHBDataset` → `{"image": Tensor(1,16,128,128)}` — healthy T1, `.npy`, source `(1,1,182,218,182)`.
- `BraTSDataset` → `{"image": ..., "label": ...}` — T2 `.nii` + segmentation, source `(240,240,155)`.

**The one pipeline** (identical for both; only the source-specific load/orient step differs):

```
load → orient to (D,H,W) → crop → trilinear resize H,W to 128 → min-max to [0,1] (eps 1e-8)
     → deterministic non-overlapping depth chunks of 16 → (1,16,128,128)
```

Segmentation labels take the **same geometry** but are resized **nearest-neighbour** and binarized
(`seg > 0`). BraTS labels are `{0,1,2,4}`.

- `SplitContract` — deterministic given `seed`; serialized to `artifacts/split_contract.json`.
  Using a different split raises `SplitContractViolationError` unless explicitly overridden.
- `DataValidator` — runs at dataloader init. Asserts shape `(B,1,16,128,128)`, dtype float32,
  values in `[0,1]`. Raises `DescriptiveValidationError` **naming the observed shape and range**.

**Boundary:** no file path, patient ID, or raw metadata escapes this layer (NFR-13).

## Acceptance tests

1. Both dataloaders yield batches of shape `(B,1,16,128,128)`, dtype float32, values within `[0,1]`.
2. `DataValidator` raises `DescriptiveValidationError` on a deliberately malformed batch, and the
   message contains the actual shape and value range it saw.
3. `SplitContract` produces an identical partition across two runs with the same seed, and a
   *different* one for a different seed. The contract file round-trips.
4. Reusing a split that disagrees with the on-disk contract raises `SplitContractViolationError`.
5. **The train and eval pipelines are the same function.** A test feeds an OpenBHB volume and a
   BraTS volume through the shared transform and asserts identical output shape, dtype, and range.
6. A BraTS label tensor contains **only** `{0.0, 1.0}` after preprocessing — no fractional values.
   (This is the regression test for the prior work's invalid-Dice bug.)
7. Chunking is deterministic: the same volume yields the same chunks across runs. No
   `RandSpatialCrop` on the eval path.

## Out of scope

Synthetic corruption (009), feature extraction (006), any model.

## Notes / deviations

- **Deviation from `planning/02-architecture.md:99`,** which puts `RandSpatialCrop` in the shared
  pipeline. A random crop makes evaluation nondeterministic and violates NFR-1. Deterministic
  chunking on the eval path; randomization is permitted on the *training* path only.
- **We are deliberately unifying the two pipelines**, which `planning/02-architecture.md:100`
  asks for. Consequence: our numbers will not equal the prior work's. That is expected and is
  handled in Spec 004 — it is a bug fix, not a regression.
- The prior BraTS path did not crop depth at all and discarded 27 of 155 slices. The unified
  pipeline must state its depth handling explicitly and cover the whole volume.
