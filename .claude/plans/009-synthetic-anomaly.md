# Plan · Spec 009 — Synthetic-anomaly (FPI) fine-tuning

**Spec:** [`specs/009-synthetic-anomaly.md`](../../specs/009-synthetic-anomaly.md) (approved)
**Depends on:** 002 (model registry), 003 (recon engine), 004 (eval harness)
**Fresh compute:** **YES.** Steps 1–9 need no GPU and no real data. Step 10 is a cluster session the
user runs manually via `/train`. The agent never launches it.

> **Save location:** this file should live at `.claude/plans/009-synthetic-anomaly.md` to match the
> existing `000`–`007` plans. (Plan mode permits writing only to the harness plan path, so copying
> it into the repo is the first action after approval.)

---

## Context

The project's central failure mode is that a model trained only to reconstruct healthy brains learns
to reconstruct *anything* well — tumours included — so the residual at the tumour is small and
detection fails. Spec 009 attacks this directly: corrupt healthy OpenBHB volumes with synthetic
lesions (Foreign Patch Interpolation), train UNETR to **restore the healthy version**, and measure
whether erasing-anomalous-structure transfers to real BraTS tumours. This is the project's one
performance novelty and its before/after story. Per CLAUDE.md rule 6, if Dice does not improve, that
is the result.

### Verified repo state

| Fact | Evidence |
|---|---|
| `src/mri_ad/synth/` is an empty docstring stub | `src/mri_ad/synth/__init__.py` (1 line) |
| **No training loop exists anywhere**; `scripts/run_train.py` is missing though `make train` references it | `Makefile:57-58`; grep for `optimizer|backward()` over `src/`+`scripts/` hits nothing |
| **`src/mri_ad/losses.py` is a stub** — `SSIMMSELoss` does not exist, yet it is the repo-default loss and the objective behind `unetr_mse_ssim_aug.pth` | `losses.py` (1 line) vs `configs/loss/mse_ssim.yaml:target` |
| No `configs/train/`, no `configs/synth/` | `configs/` tree |
| `OpenBHBDataset(cfg, ids)` yields `{"image": (1,16,128,128), "volume_index", "chunk_index"}`, 5 chunks/volume | `src/mri_ad/data/datasets.py:62-96` |
| `SplitContract` has OpenBHB `train`/`val` and BraTS `val`/`test` only; `.content_hash()` exists | `src/mri_ad/data/split.py:53-123` |
| `build_dataloader(cfg, ds, *, split)` exists (seeded generator, MONAI `worker_init_fn`) but is unused | `src/mri_ad/data/loaders.py:19` |
| `build_transforms(..., training=False)` — the `training` flag is a declared no-op extension point | `src/mri_ad/data/transforms.py:128` |
| Registry instantiates from `configs/model/*.yaml` via `registry.get(name)` / `checkpoint_path(name)`; `load_checked_state_dict` accepts raw state dicts *and* `{"model_state_dict": ...}`, `strict=True`, detects LFS stubs | `models/registry.py:65-78`, `models/_checkpoint.py:22-46` |
| Eval writes `artifacts/metrics/<model>__<loss>/aggregate.json` with `run_id`, `split_hash`, `n_volumes`; `RunLogger` writes the **fully-resolved config** to `artifacts/runs/<run_id>/run_meta.json` | `eval/report.py:58-132`, `utils/run_logger.py:78` |
| `.claude/skills/train/SKILL.md` **already exists** with `disable-model-invocation: true` and a preflight checklist | verified |
| `artifacts/metrics/` does not exist — **the "before" cell has never been computed** | — |

### Decisions

| # | Decision | Why | Rejected |
|---|---|---|---|
| D1 | Patch size range is **split per axis**: H/W `[8,32]` (spec), **D `[4,10]`** | A chunk is only 16 deep; a depth-16 patch spans the whole slab — that is a column, not a lesion, and it makes "bit-identical outside" trivial in D. Declared deviation, documented in the YAML. | Silently clamping `[8,32]`→`[8,16]` |
| D2 | **Foreground-only placement** (`foreground_threshold: 0.05`, `min_foreground_fraction: 0.5`) | Volumes are min-max normalised with a large zero background. A patch in background blends 0 with 0: non-empty mask, nothing changed, model trains identity. This is silent-wrong-number risk R4. | Uniform placement |
| D3 | **Donor patch taken from the same coordinates** in another subject (`donor_alignment: same`) | The FPI construction: anatomy roughly matches, only intensity/texture is inconsistent. Random-location donors are anatomically absurd and a raw-intensity threshold finds them — which fails acceptance 2. | Random donor location |
| D4 | `synth_mask` is **geometric** (α inside a placed box, 0 outside), not `corrupted != healthy` | The diff-based definition makes acceptance 1 vacuous. Geometric + a `min_changed_fraction` floor makes it a real assertion. | Diff mask; plain binary box mask |
| D5 | Per-item RNG seeded by **`blake2b(f"{seed}:{epoch}:{index}")`** | Python's `hash()` on strings is salted by `PYTHONHASHSEED`, which `seed_everything` sets *after* interpreter start — so it has no effect and worker determinism silently breaks. | `hash()`; global RNG |
| D6 | Training loop lives in **`src/mri_ad/train/`**, peer of `data`/`models`/`synth`, with one seam: a `StepFn` | Spec 013 needs a loop too (DDPM from scratch) and reuses everything but the step function. Keeps `train/` free of `recon`/`eval` imports. | MONAI `SupervisedTrainer`/Ignite (makes per-item seeding and `set_epoch` awkward for ~120 lines of loop); Lightning |
| D7 | Fine-tuned model registers as **`configs/model/unetr_synth.yaml`**, then the *existing* recon+eval path runs unchanged | `cell_id` becomes `unetr_synth__mse_ssim`, side by side with `unetr__mse_ssim`. Satisfies FR-14 (a model touches only its module + YAML + checkpoint path); **zero** changes in `recon/` or `eval/`. | Overriding `model.checkpoint=` on the `unetr` name — both runs would share `cell_id` and the "after" eval would **overwrite the "before" number in place**. That is the exact silent-wrong-number failure. |
| D8 | Checkpoint selection on **BraTS-*val* Dice** (user decision) | Selecting on restoration loss is selecting on a reconstruction number, which this project's thesis says is anti-correlated with detection. `contract.brats["val"]` exists precisely for tuning; the test split is never touched (trap #5). | FPI val loss (kept as a config-switchable fallback and always logged as context) |
| D9 | `corrupt_probability: 1.0`, single run at **lr 1e-5, 20 epochs**, nothing frozen (user decisions) | Literal reading of the spec; cheapest defensible run. If it fails, that is a reportable negative finding. | 0.8 clean fraction; two-point LR sweep; frozen ViT encoder |
| D10 | `unetr_synth` appears in **all three** places: its own before/after table, a row in the 005 matrix, a column in the 007 paradigm table (user decision) | Matrix + paradigm are both **config-only** — cells/columns are declared data. Until trained they render honestly as `n/a`. | Before/after table only |
| D11 | AMP **off** | Mixed precision breaks bitwise determinism (NFR-1). Wall-clock cost accepted. | `amp: true` |

### Prerequisite the spec does not mention

`SSIMMSELoss` must be implemented before anything can train. It is the repo-default loss, the
objective behind the checkpoint we fine-tune from, and it does not exist. `MultiScaleMSELoss` stays
out of scope — its matrix cell is already a declared `n/a`.

---

## Files

### New

| Path | Responsibility |
|---|---|
| `src/mri_ad/synth/fpi.py` | `FPIAnomalyGenerator`, `SynthResult`, `PatchSpec`. Placement + blend. No global RNG. |
| `src/mri_ad/synth/dataset.py` | `AnomalyInformedDataset` wrapping `OpenBHBDataset`; per-item seeding, `set_epoch`, donor bijection. |
| `src/mri_ad/synth/separability.py` | `best_intensity_dice(...)` — shared by acceptance test 2 and the pre-flight guard. |
| `src/mri_ad/train/__init__.py` | Package docstring + re-exports. |
| `src/mri_ad/train/loop.py` | `Trainer`, `EpochRecord`, `TrainSummary`, `StepFn`. Imports nothing from `recon`/`eval`. |
| `src/mri_ad/train/objectives.py` | `reconstruction_step`. (013 adds `ddpm_step` here.) |
| `src/mri_ad/train/checkpointing.py` | `CheckpointWriter` — writes a dict `load_checked_state_dict` can read. |
| `src/mri_ad/train/splits.py` | `resolve_training_ids` — the single path that yields training ids; raises on any BraTS-test overlap. |
| `src/mri_ad/eval/before_after.py` | Report-safe (stdlib only): load two cells, assert the comparison is controlled, render the table. |
| `scripts/run_train.py` | `make train`. GPU, manual only. |
| `scripts/run_synth_comparison.py` | `make synth-table`. Report-safe, no GPU. |
| `configs/synth/fpi.yaml`, `configs/train/finetune.yaml`, `configs/model/unetr_synth.yaml` | Config surface (contents below). |
| `tests/test_synth.py`, `tests/test_train.py`, `tests/test_before_after.py`, `tests/test_losses.py` | Acceptance + regression tests. |

### Modified

- `src/mri_ad/losses.py` — implement `SSIMMSELoss`.
- `configs/config.yaml` — add `- synth: fpi` and `- train: finetune` to `defaults` (so `RunLogger`
  auto-logs every FPI parameter; acceptance 3).
- `configs/matrix/arch_loss.yaml` — one `unetr_synth` row (`role: novelty`, `na_reason: "not trained"`).
- `configs/paradigm/default.yaml` — one `unetr_synth` column, `n/a` until trained.
- `Makefile` — add the `synth-table` target. (`train` already exists and already points at `scripts/run_train.py`.)
- `progress_report.md` — append an entry.

**Untouched:** everything in `recon/`, all existing `eval/` modules, `run_eval.py`, `run_recon.py`,
`run_report.py`.

---

## Interfaces

```python
# synth/fpi.py ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PatchSpec:
    d0: int; d1: int; h0: int; h1: int; w0: int; w1: int
    alpha: float
    donor_offset: tuple[int, int, int]

@dataclass(frozen=True)
class SynthResult:
    corrupted:  Tensor   # (1,16,128,128) f32 in [0,1] — convex combo of two [0,1] volumes
    healthy:    Tensor   # (1,16,128,128) f32 — an OWNED CLONE, never a view (risk R1)
    synth_mask: Tensor   # (1,16,128,128) f32 — alpha inside a placed box, 0.0 outside;
                         #   overlapping patches take element-wise max(alpha) (order-independent)
    patches: tuple[PatchSpec, ...]

class FPIAnomalyGenerator:
    def __init__(self, *, patch_size_range, depth_size_range, n_patches_range, alpha_range,
                 foreground_threshold, min_foreground_fraction, min_patch_contrast,
                 max_placement_attempts, donor_alignment, corrupt_probability, seed) -> None: ...
    def generate(self, volume: Tensor, donor: Tensor, *,
                 generator: torch.Generator | None = None) -> SynthResult: ...
    # raises DescriptiveValidationError on shape/dtype/range violation (message names observed values)
    # raises ConfigError if depth_size_range[1] > shape.depth
    # ALL randomness draws from `generator`. No global RNG, no `random`, no `numpy.random`.

# synth/dataset.py ─────────────────────────────────────────────────────────
class AnomalyInformedDataset(Dataset):
    def __init__(self, base: OpenBHBDataset, generator: FPIAnomalyGenerator, *,
                 seed: int, epoch_invariant: bool = False, donor_chunk: str = "same") -> None: ...
    def set_epoch(self, epoch: int) -> None: ...
    def __getitem__(self, index: int) -> dict:
        # {"corrupted", "healthy", "synth_mask": each (1,16,128,128) f32,
        #  "volume_index": int, "chunk_index": int}   — no path, no subject id (NFR-13)

# Donor choice is a bijection, never a rejection loop (a loop consumes a variable number of
# RNG draws and desynchronises the stream):
#     j = randint(0, n-1); donor = j if j < target else j + 1     # ConfigError if n < 2

# synth/separability.py ────────────────────────────────────────────────────
def best_intensity_dice(corrupted: Tensor, synth_mask: Tensor, *,
                        foreground_threshold: float, n_thresholds: int = 256) -> float
    # Oracle-optimistic: sweeps n_thresholds quantiles x both polarities, restricted to foreground,
    # and returns the BEST Dice against (synth_mask > 0) using ground truth to pick the winner.
    # If the oracle cannot reach the bound, no global-intensity shortcut exists.

# train/loop.py ────────────────────────────────────────────────────────────
StepFn = Callable[[nn.Module, dict[str, Tensor], torch.device], Tensor]   # -> scalar loss

@dataclass(frozen=True)
class EpochRecord:  epoch: int; train_loss: float; val_loss: float; extra: dict[str, float]; lr: float

@dataclass(frozen=True)
class TrainSummary:
    best_epoch: int; best_score: float; selection_key: str
    history: tuple[EpochRecord, ...]; checkpoint_path: Path
    param_l2_delta: float     # L2 distance from the init weights — proves training happened (R6)

class Trainer:
    def __init__(self, model, *, optimizer, scheduler, train_loader, val_loader, step_fn, device,
                 max_epochs, grad_clip_norm, early_stopping_patience, selection_key, selection_mode,
                 on_epoch_start=None, validate_fn=None, checkpoint_writer) -> None: ...
    def fit(self) -> TrainSummary: ...
    # on_epoch_start is where dataset.set_epoch(epoch) is wired — Trainer never knows synth/ exists.
    # validate_fn (BraTS-val Dice) is CONSTRUCTED IN scripts/run_train.py from ReconstructionEngine +
    #   eval.metrics, so train/ never imports recon/ or eval/. Entry-point scripts may import any
    #   layer (precedent: scripts/run_recon.py).
    # Asserts loader.persistent_workers is False — with persistent workers, set_epoch never
    #   propagates and every epoch silently reuses epoch 0's corruptions.

# train/objectives.py ──────────────────────────────────────────────────────
def reconstruction_step(model, batch, device, *, criterion: nn.Module) -> Tensor:
    return criterion(model(batch["corrupted"].to(device)), batch["healthy"].to(device))

# train/checkpointing.py ───────────────────────────────────────────────────
class CheckpointWriter:
    def __init__(self, path: Path, *, meta: Mapping[str, Any]) -> None: ...
    def save(self, model, *, epoch: int, metrics: Mapping[str, float]) -> Path
    # {"model_state_dict", "epoch", "selection_key", "selection_value", "val_loss",
    #  "git_sha", "seed", "split_hash", "synth": {...}, "train": {...}}
    # load_checked_state_dict picks model_state_dict and ignores the rest. NO paths.* (NFR-13).

# train/splits.py ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrainingIds: openbhb_train: tuple[str,...]; openbhb_val: tuple[str,...]; brats_select: tuple[str,...]
def resolve_training_ids(cfg: DictConfig, contract: SplitContract) -> TrainingIds
    # raises SplitContractViolationError if ANY returned id intersects contract.brats["test"]

# eval/before_after.py  (report-safe: json/csv/pathlib only — no torch, no monai) ──
def load_cell(metrics_root: Path, runs_root: Path, *, model: str, loss: str) -> CellRecord
def assert_controlled(before: CellRecord, after: CellRecord) -> None    # ArtifactError on drift
def render_markdown(before, after) -> str
def render_csv(before, after, path: Path) -> Path

# losses.py ────────────────────────────────────────────────────────────────
class SSIMMSELoss(nn.Module):
    def __init__(self, data_range=1.0, ssim_weight=1.0, mse_weight=1.0, spatial_dims=3) -> None
    def forward(self, pred: Tensor, target: Tensor) -> Tensor    # (B,1,16,128,128) -> ()
    # = ssim_weight * monai.losses.SSIMLoss(...) + mse_weight * F.mse_loss(...)
    # matches legacy/attUNET verbatim: (1 - ssim(pred, target, data_range=1.0)) + mse(pred, target)
```

### `assert_controlled` — what makes the comparison controlled

Acceptance 4 demands "identical split, same threshold strategy, same preprocessing", but
`aggregate.json` records `split_hash` and not the threshold/preprocessing. Rather than patch
`write_aggregate`, the comparison follows `aggregate.json → run_id → artifacts/runs/<run_id>/run_meta.json`,
which already holds the fully-resolved config. Zero harness change, strictly more coverage. It
hard-fails with `ArtifactError` naming the offending key unless **all** of these match:

| Checked | Source |
|---|---|
| `split_hash`, `n_volumes`, `split == "test"`, `loss` | both `aggregate.json` |
| `config.threshold` (whole subtree) | both `run_meta.json` |
| `config.data.preprocess`, `config.data.eval_subset`, `config.shape` | both `run_meta.json` |
| `config.model.params` (identical architecture) | both `run_meta.json` |

The legacy `published_dice` (0.6255) is **never** a source for the "before" cell — spec's own note.

---

## Config

**`configs/synth/fpi.yaml`**
```yaml
# Spec 009: Foreign Patch Interpolation. corrupted = (1-a)*target + a*donor inside a patch box.
# Chosen over Poisson blending (boundary solve, slow) and 3D CutPaste (sharp edges are a shortcut
# the model learns instead of anatomy).
name: fpi
target: mri_ad.synth.fpi.FPIAnomalyGenerator
params:
  # The spec's [8,32] is isotropic, but a chunk is only 16 deep. Splitting the range per axis is a
  # DECLARED deviation: a depth-16 patch spans the whole slab and is a column, not a lesion.
  patch_size_range: [8, 32]     # H and W
  depth_size_range: [4, 10]     # D — must be <= shape.depth (validated -> ConfigError)
  n_patches_range: [1, 4]
  alpha_range: [0.3, 0.7]
  # Foreground-only placement. A patch in the zero background blends 0 with 0: the mask is
  # non-empty but nothing changed, which makes acceptance 1 vacuous and trains an identity map.
  foreground_threshold: 0.05
  min_foreground_fraction: 0.5
  min_patch_contrast: 0.01      # floor only; acceptance 2 supplies the ceiling
  max_placement_attempts: 25
  donor_alignment: same         # same|random — same-location keeps anatomy aligned (harder task)
  corrupt_probability: 1.0

donor_chunk: same               # same|random depth-slab index in the donor volume
min_changed_fraction: 0.9       # acceptance-1 assertion 3

# Acceptance 2, on REAL data, before the first optimizer step.
check:
  n_samples: 32
  n_thresholds: 256
  max_intensity_dice: 0.5       # oracle-optimistic bound over all thresholds, both polarities
  abort_on_failure: true
```

**`configs/train/finetune.yaml`**
```yaml
# Spec 009 fine-tune. GPU. MANUAL INVOKE ONLY (/train, make train) — constraint C-3.
name: finetune
init_from: unetr                # registry name whose checkpoint we fine-tune from
save_as: unetr_synth            # must match configs/model/unetr_synth.yaml `name`
checkpoint_path: ${paths.checkpoint_root}/unetr_synth.pth

max_epochs: 20                  # fine-tune, not the legacy 50-epoch from-scratch schedule
optimizer:
  target: torch.optim.Adam
  params:
    lr: 1.0e-5                  # 10x below the legacy from-scratch 1e-4: a 12-layer ViT
                                # catastrophically forgets at the from-scratch LR
    weight_decay: 1.0e-5
scheduler:
  target: torch.optim.lr_scheduler.ReduceLROnPlateau
  params: {mode: min, factor: 0.5, patience: 3}
grad_clip_norm: 1.0
early_stopping_patience: 6
amp: false                      # off: mixed precision breaks bitwise determinism (NFR-1)
allow_cpu: false                # run_train.py raises ConfigError on a non-CUDA device unless true

selection:
  metric: val_dice              # val_dice | val_loss
  mode: max
  dice_every_n_epochs: 2
  n_val_volumes: 8              # BraTS *val* split only. Test is never touched (trap #5).
```

**`configs/model/unetr_synth.yaml`** — byte-identical `params` to `unetr.yaml`; only
`name: unetr_synth` and `checkpoint: ${paths.checkpoint_root}/unetr_synth.pth` differ. A test
asserts the two `params` blocks are equal and the two `checkpoint` paths are not.

---

## Tests → acceptance mapping

| Spec acceptance | Test | What it proves |
|---|---|---|
| **1** differs within mask, bit-identical outside | `test_synth.py::test_fpi_outside_mask_bit_identical` · `::test_fpi_blend_identity_inside_mask` · `::test_fpi_changed_fraction_floor` | (a) `torch.equal` outside the *geometric* mask; (b) exactly `(1−α)h + αd` inside, `atol=0`; (c) ≥90% of masked voxels actually changed — this is the assertion that catches all-background placement. |
| **2** not trivially separable | `test_synth.py::test_intensity_threshold_cannot_recover_mask` — 32 fixed-seed samples on a *textured* phantom (ellipsoid × smooth low-frequency field, built by trilinear upsampling of a small random tensor), asserting both `mean` and `max` best-Dice < 0.5. **Plus** the real-data pre-flight guard in `run_train.py`. | No global-intensity shortcut exists — on the fixture *and* on real OpenBHB before any GPU is spent. A constant-intensity phantom would fail for a fixture artefact, hence the texture. |
| **3** all params in `run_meta.json` | `test_train.py::test_run_meta_logs_every_generator_param` — derives the expected key set from `inspect.signature(FPIAnomalyGenerator.__init__)` rather than a hardcoded list. | Cannot rot when a generator parameter is added. |
| **4** the before/after table | `test_before_after.py::test_table_from_two_cells` · `::test_rejects_split_hash_mismatch` · `::test_rejects_threshold_drift` · `::test_rejects_preprocess_drift` · `::test_before_never_sourced_from_published_dice` | The table exists, every cell traces to a `run_id`, and it hard-fails rather than emitting an uncontrolled comparison. |
| **5** test split never seen | `test_train.py::test_resolve_training_ids_rejects_test_overlap` · `::test_training_ids_are_openbhb_train_only` | `resolve_training_ids` is the only path producing ids, and it raises on any leak. |
| **6** reachable only via `/train` | `test_train.py::test_train_skill_disables_model_invocation` (parses `.claude/skills/train/SKILL.md` frontmatter) · `::test_makefile_train_target_exists` · `::test_run_train_refuses_non_cuda_device` | The gate exists and the script refuses an accidental laptop launch. |
| **7** a negative result is reported | `test_before_after.py::test_negative_delta_renders_as_regression` | An "after" Dice below "before" renders a signed delta and a `REGRESSION` marker — never a dropped row. |

**Additional regression tests:** `test_synth.py::test_deterministic_across_worker_counts`
(num_workers 0 vs 2, item-by-item equality) · `::test_val_dataset_is_epoch_invariant` ·
`::test_healthy_is_not_a_view_of_cache` · `::test_donor_never_equals_target` ·
`::test_no_global_rng_in_synth` (AST walk, mirroring `tests/test_classical_boundary.py`) ·
`test_train.py::test_checkpoint_round_trips_through_load_checked_state_dict` ·
`::test_parameters_actually_move` · `test_losses.py` (`loss(x,x) ≈ 0`, monotone decrease as
`pred → target`).

All tests run on synthetic tensors at toy sizes or with a 3-parameter toy `nn.Module` — no real
data, no checkpoints, no real 3D forward passes (laptop constraint).

---

## Ordered steps

1. `losses.py::SSIMMSELoss` + `tests/test_losses.py`. Unblocks everything.
2. `synth/fpi.py` + `synth/separability.py` + acceptance-1/2 tests. Pure tensor code.
3. `synth/dataset.py` + determinism / aliasing / donor tests.
4. Config surface: `configs/synth/fpi.yaml`, `configs/train/finetune.yaml`,
   `configs/model/unetr_synth.yaml`; add both groups to `configs/config.yaml` defaults; add the
   matrix row and the paradigm column.
5. `train/` package (`loop.py`, `objectives.py`, `checkpointing.py`, `splits.py`) + `tests/test_train.py`.
6. `scripts/run_train.py` — house pattern (`@hydra.main` → `seed_everything` → `with RunLogger(cfg)`),
   device guard, `resolve_contract` → `resolve_training_ids`, **pre-flight separability check**
   (writes `artifacts/synth/separability.json`, aborts before the first optimizer step if the
   oracle Dice exceeds `check.max_intensity_dice`), `Trainer.fit()`, `run.record(**summary)`.
7. Refresh `.claude/skills/train/SKILL.md` — it currently claims 009 is the *only* fresh-compute
   spec, which is stale now that 013 exists.
8. `eval/before_after.py` + `scripts/run_synth_comparison.py` + `Makefile` target +
   `tests/test_before_after.py`. Fully testable now against fabricated
   `aggregate.json` / `run_meta.json` fixtures.
9. `progress_report.md` entry (What / Why / Problems hit / Result / **Next:**).
10. **Hand off.** The agent runs nothing below this line.

---

## Verification

Local, no GPU, no data:

```bash
make lint
pytest tests/test_losses.py tests/test_synth.py tests/test_train.py tests/test_before_after.py
make test                                   # full suite, expect the existing 221+ green
python scripts/run_synth_comparison.py      # honest all-n/a output before any training
make matrix && make paradigm                # unetr_synth appears as n/a, not omitted
```

Cluster session — **the user runs these, in this order, manually**:

```bash
make check-data                                       # must exit 0
make recon  HYDRA_OVERRIDES="model=unetr +experiment=cluster"
make eval   HYDRA_OVERRIDES="model=unetr +experiment=cluster"   # <- the "before" cell
make train  HYDRA_OVERRIDES="+experiment=cluster"                # /train only
make recon  HYDRA_OVERRIDES="model=unetr_synth +experiment=cluster"
make eval   HYDRA_OVERRIDES="model=unetr_synth +experiment=cluster"
make synth-table                                       # the before/after table
```

The "before" cell has never been computed (`artifacts/metrics/` does not exist), so those first two
eval steps are a hard prerequisite of the fine-tune, not an optional refresh.

---

## Risks — what could silently produce a confidently wrong number

| # | Risk | Caught by |
|---|---|---|
| R1 | `healthy` aliases the `CacheDataset` tensor and the blend is in-place → from epoch 2 the cache *holds corrupted volumes*, the model trains identity→identity, and the loss curve looks beautiful. | `test_healthy_is_not_a_view_of_cache`; regenerating the same index twice and asserting bitwise equality. |
| R2 | Threshold or preprocessing drifted between the before and after recon runs → the delta measures config drift, not the intervention. | `assert_controlled` deep-compares `config.threshold`, `data.preprocess`, `shape`, `eval_subset`, `model.params` across both `run_meta.json`. |
| R3 | The "before" number is the legacy 0.6255 rather than the corrected Spec 004 baseline. | The comparison sources only `aggregate.json`; a test asserts `published_dice` is never a source; every row prints `run_id` + `results_run_id`. |
| R4 | Every patch lands in background → identity task; Dice unchanged; reported as "FPI doesn't help." | `min_changed_fraction` assertion + the real-data pre-flight guard. |
| R5 | FPI is trivially separable → improvement on *synthetic* data that does not transfer to real tumours (the spec's own stated fear). | Acceptance-2 oracle test + `check.max_intensity_dice` abort before the first optimizer step. |
| R6 | Nothing actually trained (lr 0, empty param group, everything frozen) → after == before, published as a null finding. | `test_parameters_actually_move`; `Trainer` records `param_l2_delta` into `run_meta.json`; the table prints it. |
| R7 | Corruption is nondeterministic across workers/epochs → the val curve is noise and "best epoch" is a coin flip. | Worker-count equality test; val epoch-invariance test; `blake2b` not `hash()`; `persistent_workers is False` assertion. |
| R8 | The checkpoint saves in a format `load_checked_state_dict` rejects, or `unetr_synth.yaml` params drift from `unetr.yaml` — discovered *after* the GPU hours are spent. | Round-trip save→load test into a fresh `UNETRReconstruction`; a test asserting the two YAMLs' `params` are equal. |
| R9 | BraTS **test** volumes leak into Dice-based checkpoint selection. | `resolve_training_ids` raises on intersection and is the only id-producing path; selection reads `contract.brats["val"]`. |
| R10 | `n_volumes` differs between the two eval runs (`eval_subset` set once) → 20 volumes compared against 100. | `assert_controlled` equality on `n_volumes` and `eval_subset`. |
| R11 | MONAI's `SSIMLoss` signature differs by version (`data_range` as ctor vs forward arg) → a silently wrong objective. | `test_losses.py` asserts `loss(x,x) ≈ 0` and strict monotone decrease as `pred → target`. |
| R12 | Each `__getitem__` touches two `CacheDataset` entries; at `cache_rate: 0.0` that doubles disk I/O per item. | Not a correctness risk — the trainer warns (does not fail) when `cache_rate < 1.0` on CUDA. `configs/experiment/cluster.yaml` already sets `1.0`. |
