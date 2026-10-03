# Implementation Plan — Spec 000: Vertical Slice & Reproducibility Spine

> **Note on file location:** In plan mode I can only edit this harness plan file. On approval,
> the **first** implementation action will be to save this same plan to
> `.claude/plans/000-vertical-slice.md` (the location you asked for), then proceed.

---

## Context

Nothing else in the project may start until one path works end to end. Spec 000 cuts the thinnest
path through the whole stack — load the UNETR checkpoint, reconstruct **one** BraTS volume, compute
the residual, threshold it, score Dice against binarized ground truth, save one figure — **plus the
reproducibility spine** (`seed_everything` ✓ already done, `RunLogger`, `DeviceManager`,
`run_meta.json`) that every later spec hangs on. A vertical slice surfaces integration failures
(shape mismatches, checkpoint-format surprises, device placement) *now*, before four specs are built
on a wrong assumption.

The repo today is **scaffolding only**: `exceptions.py`, `models/base.py` (interface), `eval/metrics.py`
(stubs raising `NotImplementedError("Spec 004")`), `recon/types.py` (`ReconResult`, `ThresholdStrategy`),
the full `configs/` tree, `Makefile`, and `scripts/check_data.py` all exist. Missing and to be built:
`scripts/run_slice.py`, `models/unetr.py`, `recon/threshold.py`, `utils/RunLogger` + `DeviceManager`,
and one real metric (`MetricsComputer.dice`).

### Two confirmed design decisions
- **Whole volume, not a single chunk.** Iterate all depth-chunks of one BraTS volume, run `forward`
  per chunk, reassemble to full depth, compute **one** Dice over the whole volume. This deliberately
  exercises prior-work bug #2 (accumulate across *all* chunks), which is the point of a spine.
- **Threshold = 95th percentile.** Use `configs/threshold/percentile.yaml` (`percentile: 95.0`),
  implemented as the real `FixedPercentileThreshold`. Stated as the hardcoded operating point per
  the spec ("000 may hardcode one and say so"); the *sweep/justification* is Spec 003's job.

### Layering discipline (why the slice isn't one throwaway script)
CLAUDE.md fixes single-source-of-truth homes: **metrics live only in `eval/metrics.py`**,
**thresholding only in `recon/`**. So rather than inline duplicates, the slice implements the minimal
*correct* version of each piece **in its proper home**, to be extended (not replaced) by 001–004:
- `models/unetr.py` — real `UNETRReconstruction` + `load_checkpoint` (Spec 002 later adds the
  registry, other models, `model_card` polish).
- `recon/threshold.py` — real `FixedPercentileThreshold` (Spec 003 later adds Absolute/Otsu + sweep).
- `eval/metrics.py` — fill in `MetricsComputer.dice` only (Spec 004 later fills iou/psnr/ssim + aggregation).

The **one** genuine "reach past a non-existent abstraction" is BraTS single-volume loading: Spec 001
owns `data/`, so the slice inlines a minimal, clearly-marked provisional loader in a helper, to be
replaced by `BraTSDataset`.

---

## Files to create / modify

### New — the spine
- **`src/mri_ad/utils/device.py`** — `DeviceManager.get_device() -> torch.device` (cuda → mps → cpu),
  honoring `cfg.device` (`auto` | explicit).
- **`src/mri_ad/utils/run_logger.py`** — `RunLogger` context manager / class. On enter: create
  `artifacts/runs/<UTC-timestamp>/`. On exit: write `run_meta.json` with `git_sha`
  (`git rev-parse HEAD` via subprocess, non-empty), `git_dirty` bool, fully-resolved config
  (`OmegaConf.to_container(cfg, resolve=True)`), `seed`, `hostname` (`socket.gethostname()`),
  `start_time`/`end_time` (ISO-8601 UTC), and the resulting `dice`. **No file paths or patient IDs**
  written (NFR-13 / acceptance test 7).
- **`src/mri_ad/utils/__init__.py`** — export `seed_everything`, `RunLogger`, `DeviceManager` so the
  spec's import paths (`mri_ad.utils.RunLogger`, `mri_ad.utils.DeviceManager`) resolve.

### New — the flow
- **`src/mri_ad/models/unetr.py`** — `class UNETRReconstruction(AnomalyDetectionModel)`:
  - Port the architecture from `legacy/GUI/utilities/utility.py:15` (`ViT` +
    `UnetrBasicBlock/UnetrPrUpBlock/UnetrUpBlock`, `out = nn.Conv3d(feature_size, 1, 1)`, ends in
    `nn.Sigmoid()` → output bounded `[0,1]`, keys `out.weight`/`out.bias`).
  - **Do NOT use `monai.networks.nets.UNETR`** (different head keys + no Sigmoid → silent partial
    load; see spec 002 & known-trap #4).
  - `__init__` signature reads params from `configs/model/unetr.yaml` (no `pos_embed="perceptron"`
    kwarg leaking out — port cleanly; the legacy `proj_type` rename is internal to ViT construction).
  - `load_checkpoint(path)`: handle **both** on-disk layouts — `{"model_state_dict": ...}` and bare
    `state_dict`; detect Git-LFS stub (compare first bytes to
    `b"version https://git-lfs.github.com/spec/v1"`) and raise `CheckpointError` naming the file;
    missing file → `CheckpointError`; `load_state_dict(state, strict=True)`. **Never `strict=False`.**
  - `model_card` property (minimal but correct `param_count`, shapes, `output_activation="sigmoid"`).
- **`src/mri_ad/recon/threshold.py`** — `class FixedPercentileThreshold` (matches
  `configs/threshold/percentile.yaml` target). `__call__(residual) -> binary mask`; flags the top
  `(100-p)%` of residual voxels via `torch.quantile`. Conforms to the `ThresholdStrategy` Protocol
  in `recon/types.py`.
- **`src/mri_ad/data/slice_io.py`** *(provisional — replaced by `BraTSDataset` in Spec 001)* —
  `load_brats_volume(cfg, volume_id) -> (image, label, volume_id)`:
  - `nib.load` the `<id>_t2.nii(.gz)` and `<id>_seg.nii(.gz)`; orient to `(D,H,W)`.
  - Image: trilinear resize H,W→128; depth **pad to next multiple of 16** (155→160, covers the whole
    volume per Spec 001's "cover the whole volume" rule — stated as provisional); min-max → `[0,1]`
    (eps `1e-8`); reshape to N chunks of `(1,16,128,128)`.
  - Label: **same geometry, nearest-neighbour** resize, then **binarize `seg > 0`** → `{0.,1.}`;
    same depth padding + chunking.
  - **Leaks no absolute path or metadata** to caller/stdout; `volume_id` = the anonymized BraTS
    subject folder name (a de-identified dataset key, not a patient identifier).
- **`scripts/run_slice.py`** — `@hydra.main(config_path="../configs", config_name="config")`:
  1. `seed_everything(cfg.seed, deterministic=cfg.deterministic)` — **first line**.
  2. `with RunLogger(cfg) as run:` open the run dir.
  3. `device = DeviceManager.get_device(cfg.device)`.
  4. Resolve `volume_id` (`cfg.slice.volume_id` or first sorted BraTS subject).
  5. `model = instantiate(cfg.model)`; `model.load_checkpoint(cfg.model.checkpoint)`; `model.to(device).eval()`.
  6. `image_chunks, label_chunks = load_brats_volume(...)`.
  7. Under `torch.no_grad()`: forward each chunk → reconstruction; `residual = (image - recon).abs()`;
     reassemble chunks → full-depth `original`, `reconstruction`, `residual`, `gt`.
  8. `mask = FixedPercentileThreshold(p=95)(residual)` over the whole volume.
  9. `dice = MetricsComputer.dice(mask, gt)` with `gt` already binarized. Assert `0.0 <= dice <= 1.0`.
  10. Build the 5-panel figure at the depth slice with **max GT area** (tumor visible) →
      `artifacts/figures/slice_<volume_id>.png` (matplotlib, inline; reusable viz is Spec 010).
  11. `run.record(dice=dice)`; print the single Dice value (nothing else identifying).

### Modify
- **`src/mri_ad/eval/metrics.py`** — implement **`MetricsComputer.dice`** only (leave `iou/psnr/ssim`
  as `NotImplementedError("Spec 004")`). `2|pred∩gt|/(|pred|+|gt|+eps)`, `EPS=1e-6`, empty-mask
  convention (both empty→1.0, one empty→0.0). Docstring already mandates binarized `gt`.
- **`configs/config.yaml`** — add a small `slice:` block: `volume_id: null` (null → first sorted
  subject). Keeps the "no magic numbers in code" rule; threshold already comes from the `threshold`
  group.

### Reused as-is (no change)
`seed_everything` (`utils/seed.py`), `exceptions.py` (`CheckpointError`, `ConfigError`, …),
`models/base.py` (`AnomalyDetectionModel`, `ModelCard`), `recon/types.py` (`ReconResult`,
`ThresholdStrategy`), `scripts/check_data.py`, `Makefile` `slice` target (already
`python scripts/run_slice.py`), `configs/model/unetr.yaml`, `configs/threshold/percentile.yaml`.

---

## Tests (`tests/`)

Add `tests/test_slice.py` (+ fixtures) — all runnable **without** real data/checkpoints, on synthetic tensors:
- **Dice binarization (acceptance #6):** `MetricsComputer.dice` on a `{0,1,2,4}` multiclass GT
  binarized vs. raw differs; slice path binarizes `seg>0`. Regression guard for the label-magnitude bug.
- **Both checkpoint formats (spec 002 contract):** save a random `UNETRReconstruction` state as
  `{"model_state_dict": ...}` and as a bare dict; both `load_checkpoint` with `strict=True`.
- **LFS stub / missing → `CheckpointError`** naming the file (uses a legacy stub `.pth`).
- **UNETR output bounded `[0,1]`** on a random `(1,1,16,128,128)` batch; forward returns same shape.
- **`FixedPercentileThreshold(95)`** flags ≈5% of voxels (±1) and emits only `{0.,1.}`.
- **`RunLogger`** writes `run_meta.json` with non-empty `git_sha`, resolved config, seed; **no path
  or PHI substring** appears in the JSON (acceptance #7).
- **Determinism (acceptance #5):** two forward passes on the same random input + seed on this
  hardware agree to `1e-6`.

Existing scaffold tests continue to pass (`test_unetr_config_points_at_the_ported_class_not_monai`,
`test_brats_labels_are_binarized_with_nearest_interpolation`, `test_lfs_stub_detection`).

---

## Verification

**Runnable now (no data/checkpoints — this machine has neither):**
- `make test` → `test_slice.py` + scaffold suite green.
- `make lint` → ruff clean.
- `python -c "from mri_ad.utils import RunLogger, DeviceManager, seed_everything"` imports resolve.

**Full end-to-end (gated on user-supplied data — `make check-data` currently FAILS: `checkpoints/`
missing):** once the user places real weights + BraTS/OpenBHB per `README.md`:
1. `make check-data` → exit 0.
2. `make slice` → exit 0, prints one Dice in `[0,1]`.
3. `artifacts/figures/slice_<id>.png` exists (5 panels: original/recon/residual/mask/GT).
4. `artifacts/runs/<ts>/run_meta.json` has non-empty `git_sha`, resolved config, seed.
5. Run twice → identical Dice within `1e-6`, same seed in both `run_meta.json`.
6. `grep` the run dir + stdout for any absolute path / subject-ID leak → none (NFR-13).

Because I can't run the GPU path here, I'll report the test/lint results honestly and mark the
`make slice` acceptance gate as **pending user data** — never fabricate a Dice number.

---

## Risks & notes
- **`make slice` is not GPU-training** — it's a single forward pass, safe to run; but it needs real
  checkpoints/data the repo doesn't ship. I will not run `make train` or any paid job.
- **Provisional pieces to reconcile later (per spec's "refactor onto the abstractions as they land"):**
  the inline `data/slice_io.py` loader → `BraTSDataset` (001); `UNETRReconstruction` → registry (002);
  `FixedPercentileThreshold` → engine + sweep (003); `dice` → full `MetricsComputer` + aggregation (004).
  I'll leave a short `# Spec 000 provisional — see NNN` marker on each.
- **Depth handling (155→pad 160)** is a real choice Spec 001 will formalize; flagged as provisional.
- After implementation: **append a `progress_report.md` entry** (what/why/how, problems hit) and
  **no `Co-Authored-By` trailer** on any commit (per CLAUDE.md).
