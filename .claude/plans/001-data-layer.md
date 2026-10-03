# Implementation Plan — Spec 001 · Data Layer

## Context

Every downstream number in this comparative study depends on volumes being preprocessed
*identically* and the split being *stable*. The prior work failed both: it applied **different
preprocessing to training (OpenBHB) and evaluation (BraTS)**, never recorded its split, and shipped
an invalid Dice by trilinear-interpolating multi-class labels and scoring only 1 of 8 depth chunks
of only the last subject. Spec 001 replaces the Spec 000 provisional single-volume loader
(`src/mri_ad/data/slice_io.py`) with a real data layer: **one MONAI-backed preprocessing pipeline
shared by both datasets**, a **deterministic, serialized split contract**, and a **loud validator**
so a shape/intensity regression fails at dataloader init rather than silently corrupting a Dice
score four specs later. Data itself is user-supplied and git-ignored; this spec is pure harness code
+ tests exercised on synthetic tensors (no GPU, no real data required to pass CI).

Spec is **approved**; dependency (Spec 000) is implemented. Config keys already exist in
`configs/data/default.yaml` awaiting a consumer.

### Confirmed design decisions (from user)
1. **Split scope:** `SplitContract` partitions **both** datasets — OpenBHB → `train`/`val` (recon
   training), BraTS → `val`/`test` (BraTS-val for threshold tuning per trap #5, BraTS-test for final
   eval). Requires adding a `brats` split fraction block to config.
2. **Item granularity:** **chunk-level** — datasets flatten depth chunks across volumes so each item
   is `(1,16,128,128)` and a DataLoader batch collates to `(B,1,16,128,128)` directly (acceptance
   test #1). Each item carries non-leaking integer `volume_index` + `chunk_index` so eval can
   accumulate per-volume across **all** chunks (kills prior bugs #2/#3).
3. **Slice migration:** migrate `run_slice.py` onto `BraTSDataset` and remove `slice_io.py` in this
   spec (Spec 000 flagged it must not remain a parallel path).

---

## The one pipeline (shared, exact recipe)

Both datasets use the **same** transform function; only the source-specific load/orient/crop differs.
Ported from `legacy/metric-uad.ipynb` cell 3 (OpenBHB) and unified with the BraTS path, following
the `monai-patterns` house style (dict transforms, keys `"image"`/`"label"`):

```
load → orient to (D,H,W) → crop → trilinear resize (D kept, H,W→128)
     → min-max to [0,1] (eps 1e-8) → deterministic non-overlapping depth chunks of 16 → (1,16,128,128)
```

- **OpenBHB:** `.npy` `(1,1,182,218,182)` → squeeze → permute to `(D,H,W)` → crop
  `z[50:130], y[20:160], x[20:196]` → `(80,140,176)` → resize H,W→128 → **5 chunks**. `{"image": ...}`.
- **BraTS:** `.nii` `(240,240,155)`, **T2 modality only** → `(D,H,W)` → resize H,W→128 → depth
  **padded** to next multiple of 16 (155→160, covers the **whole** volume — prior work dropped 27
  slices) → chunks. `{"image": ..., "label": ...}`.
- **Label geometry (trap #1 / bug #4):** `(seg > 0).float()` **binarize FIRST**, then resize
  **nearest**. Never trilinear on a `{0,1,2,4}` label. Output must be exactly `{0.0, 1.0}`.
- **Determinism:** no `RandSpatialCrop` on the shared/eval path (deviation from
  `planning/02-architecture.md:99`, which the config already forbids at `chunking: deterministic`).
  A `RandSpatialCrop`-style augmentation is permitted **only** on an opt-in training transform variant,
  never in the eval/validation path (NFR-1).

Reference constants live in `slice_io.py` (`_CHUNK_DEPTH`, `_pad_depth`, `_chunk`, the permute/
interpolate/min-max sequence) — port the logic, then delete the file.

---

## Files to create / modify

### New — `src/mri_ad/data/`
| File | Contents |
|---|---|
| `transforms.py` | `build_transforms(cfg, *, dataset, training=False) -> Compose`. The single shared MONAI dict-transform chain (image + optional label). Source-specific load/orient/crop selected by `dataset ∈ {"openbhb","brats"}`. Deterministic by default; `training=True` may add augmentation. Exposes the chunking transform. |
| `chunking.py` | `pad_depth`, `chunk_volume` (`(D,H,W)`→`(N,1,16,128,128)`), and a `flatten_chunks` helper mapping (volume → chunk items) with `volume_index`/`chunk_index`. Ported verbatim-in-logic from `slice_io._pad_depth`/`_chunk`. |
| `datasets.py` | `OpenBHBDataset` and `BraTSDataset` (subclass `monai.data.Dataset`/`CacheDataset`, `cache_rate` from `cfg.data.loader.cache_rate`). Both enumerate subjects, apply the shared pipeline, and present **chunk-level** items. `__getitem__` → `{"image": (1,16,128,128), ["label": ...], "volume_index": int, "chunk_index": int}`. No path/patient-id in any returned field (NFR-13). |
| `split.py` | `SplitContract`: `build(seed, openbhb_ids, brats_ids) -> SplitContract`; `save(path)`/`load(path)` round-trip to `artifacts/split_contract.json`; `verify(other)` raises `SplitContractViolationError` on mismatch unless `override=True`. Deterministic partition from `seed` (seeded `random.Random`/`np.random.default_rng`, sorted ids first). Serializes seed + per-split id lists + fractions. |
| `validation.py` | `DataValidator.validate_batch(batch)`: asserts shape `(B,1,16,128,128)`, dtype float32, values in `[0,1]`; on failure raises `DescriptiveValidationError` **naming the observed shape and [min,max] range**. `wrap_loader(loader)` or a collate/`__iter__` hook that validates the first batch at dataloader init. |
| `loaders.py` | `build_dataloader(cfg, dataset, *, split)`: deterministic `DataLoader` — `shuffle=False` on eval, `generator=torch.Generator().manual_seed(cfg.seed)`, `worker_init_fn=monai.data.utils.worker_init_fn`, batch/workers from config; runs `DataValidator` on init. |

Update `src/mri_ad/data/__init__.py` to export `OpenBHBDataset`, `BraTSDataset`, `SplitContract`,
`DataValidator`, `build_dataloader`, `build_transforms`.

### Modify
| File | Change |
|---|---|
| `configs/data/default.yaml` | Restructure `split:` to name both datasets: `openbhb: {train: 0.85, val: 0.15}`, `brats: {val: 0.2, test: 0.8}` (val = threshold-tuning holdout). Keep `contract_path`. |
| `scripts/run_slice.py` | Replace `load_brats_volume(...)` call with `BraTSDataset` + per-volume chunk iteration (filter to one `volume_index`); keep the seed→RunLogger→device→record spine unchanged. |
| `src/mri_ad/data/slice_io.py` | **Delete** after `run_slice.py` no longer imports it. |

### Tests — `tests/test_data.py` (new)
Mirror existing conventions (`REPO` anchor, `pytest.importorskip("monai")`, synthetic tensors,
`tmp_path`, no real data/checkpoints). One test per acceptance criterion (see below). Extend
`tests/test_scaffold.py` with a YAML assertion for the new `split.brats` keys.

---

## Key interfaces

```python
# datasets.py
class BraTSDataset(monai.data.Dataset):
    def __init__(self, cfg, volume_ids: list[str]): ...
    def __getitem__(self, i) -> dict:  # {"image","label","volume_index","chunk_index"}

# split.py
@dataclass(frozen=True)
class SplitContract:
    seed: int
    openbhb: dict[str, list[str]]   # {"train": [...], "val": [...]}
    brats: dict[str, list[str]]     # {"val": [...], "test": [...]}
    @classmethod
    def build(cls, seed, openbhb_ids, brats_ids) -> "SplitContract": ...
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path) -> "SplitContract": ...
    def verify(self, other: "SplitContract", *, override: bool = False) -> None  # raises SplitContractViolationError

# validation.py
class DataValidator:
    @staticmethod
    def validate_batch(batch: Tensor) -> None  # raises DescriptiveValidationError w/ observed shape+range
```

Reuse (do not reinvent): `mri_ad.VOLUME_SHAPE`, `seed_everything`, `RunLogger`, `DeviceManager`,
the `DescriptiveValidationError`/`SplitContractViolationError`/`DataError` classes, and the
`slice_io` chunking logic (ported then deleted).

---

## Acceptance tests → test map (all runnable without real data)

1. Both dataloaders yield `(B,1,16,128,128)` float32 in `[0,1]` — synthetic on-disk `.npy`/`.nii`
   fixtures written to `tmp_path`, or monkeypatched loaders returning synthetic `(D,H,W)` arrays.
2. `DataValidator` raises `DescriptiveValidationError` on a malformed batch; message contains the
   actual shape and value range.
3. `SplitContract` identical across two runs with same seed, different for different seed; file
   round-trips (`save`→`load` equal).
4. Reusing a split disagreeing with the on-disk contract raises `SplitContractViolationError`
   (and `override=True` suppresses it).
5. **Train and eval pipelines are the same function** — feed one OpenBHB-shaped and one BraTS-shaped
   synthetic volume through `build_transforms`; assert identical output shape/dtype/range.
6. A BraTS label tensor contains **only** `{0.0, 1.0}` after preprocessing (the invalid-Dice
   regression test) — plant a `{0,1,2,4}` label, assert `set(unique) <= {0.0,1.0}`.
7. Chunking deterministic across runs; assert no random transform on the eval path (introspect the
   `Compose` for `Rand*` transforms).
8. **Privacy (NFR-13):** assert no returned item field / log contains a file path, `.nii`/`.npy`
   suffix, or subject-id-as-path (mirror `test_run_logger_writes_provenance_without_leaks`).

---

## Risks & mitigations
- **MONAI `Resized` vs raw `F.interpolate` numeric drift.** The recipe must match `slice_io` closely
  enough that Spec 000's slice Dice is unchanged. Mitigation: if a MONAI transform can't reproduce
  "resize H,W only, keep D" + min-max exactly, wrap the ported `F.interpolate`/min-max in a
  `Lambdad`/custom `MapTransform` rather than forcing a MONAI built-in. Verify by re-running `make slice`.
- **Chunk-level items lose volume grouping** needed by eval. Mitigation: `volume_index`/`chunk_index`
  ints carried through (non-leaking); eval (Spec 004) groups by `volume_index`.
- **`num_workers>0` nondeterminism.** Mitigation: seeded generator + `worker_init_fn` (monai-patterns).
- **Config restructure breaks `test_scaffold.py`.** Mitigation: update the split YAML assertion in the
  same change.
- **Deleting `slice_io.py` before `run_slice.py` is migrated** breaks `make slice`. Mitigation:
  migrate `run_slice.py` first, confirm import removed, then delete.

---

## Verification (end-to-end)
1. `make test` — full `tests/test_data.py` + `tests/test_scaffold.py` pass on synthetic data (no GPU).
2. `make lint` — ruff clean.
3. `make check-data` then `make slice` (only if the user has real BraTS present) — confirms the
   migrated `run_slice.py` still produces the same Dice + figure for one volume, reproducibly.
4. Confirm `artifacts/split_contract.json` is written and round-trips; confirm no `data/` or
   `checkpoints/` files are staged for commit.
5. Append a `progress_report.md` entry (what/why/how, per CLAUDE.md rule #2).

## SDD next step
After this plan is approved: `/tasks` to decompose into ordered, individually-testable tasks, then
implement task-by-task with `torch-reviewer` before merge.
