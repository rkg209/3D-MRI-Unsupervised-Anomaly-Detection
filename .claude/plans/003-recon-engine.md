# Plan · Spec 003 — Reconstruction & anomaly-map engine

**Spec:** [`specs/003-recon-engine.md`](../../specs/003-recon-engine.md) (approved) · **Depends on:** 001, 002 · **Fresh compute:** no (the sweep is cluster-run, manual-invoke)

---

## Context

Spec 003 is the heart of the method: `model + volume → reconstruction → residual → binary anomaly mask`. Today that path exists only as inlined code in `scripts/run_slice.py` (Spec 000's vertical slice), and the threshold is a single hardcoded strategy — the same shape of mistake the prior work made when it buried `residual > 0.1` in a notebook cell.

Thresholding is the highest-leverage choice in the pipeline: it converts a continuous residual into the mask Dice is computed against, so it determines every headline number. This spec centralizes it in one module, makes the strategy swappable by config alone, and — critically — requires the operating point to be **selected by a sweep on the validation split**, not asserted. The spec's own "Notes / deviations" section flags that `planning/02-architecture.md` mislabels a 95th-percentile threshold as "matching the prior work"; it does not (the prior work was absolute at 0.1), and percentile vs absolute measure materially different things. The sweep resolves that empirically.

Intended outcome: `ReconstructionEngine` becomes the only producer of anomaly masks, `recon/` the only place binarization happens, and `artifacts/results/recon/<run_id>/<volume_id>.pt` the single artifact the Spec 004 eval harness reads — so the harness never calls `model.forward()`.

### Decisions taken with the user

1. **`ReconResult` is per-volume at `(1, D, 128, 128)`** — chunk for the forward pass, reassemble to full depth (D=160 for BraTS: 155 slices zero-padded to 10×16) before thresholding. `(1,16,128,128)` remains the *model-boundary* invariant, not the artifact shape. This matches the spec's one-file-per-`volume_id` path and how `run_slice.py` already reassembles. The `ReconResult` docstring must be corrected to say so.
2. **The sweep gets its own entry point** — `scripts/run_sweep.py` + `make sweep`, not a flag on Spec 004's `run_eval.py`. Keeping tuning physically separate from test-split scoring makes the no-leakage rule structural rather than a comment.
3. **The sweep is manual-invoke, cluster-only.** All code lands with unit tests against synthetic tensors and stub models; nothing heavy runs on the laptop. The chosen operating point is written back into `configs/threshold/` by hand after the user reports the cluster numbers.

---

## What already exists (reuse, don't rewrite)

| Thing | Path | Note |
|---|---|---|
| `ReconResult`, `ThresholdStrategy` Protocol | `src/mri_ad/recon/types.py` | Written; docstring needs the shape fix |
| `FixedPercentileThreshold` | `src/mri_ad/recon/threshold.py` | Exists but not exact (see R1) |
| Chunk/reassemble helpers | `src/mri_ad/data/chunking.py` | `pad_depth`, `chunk_volume`, `_CHUNK_DEPTH = 16` |
| Chunk-level datasets | `src/mri_ad/data/datasets.py` | `BraTSDataset`; map index→id via `dataset.volume_ids[volume_index]` |
| Val/test holdout | `src/mri_ad/data/split.py` | `SplitContract.build/save/load/verify`; BraTS `val` = 0.2 |
| `MetricsComputer.dice` | `src/mri_ad/eval/metrics.py` | The only working metric (iou/psnr/ssim raise `NotImplementedError("Spec 004")`) — the sweep is Dice-only, which is exactly what acceptance #5 asks for |
| `{target, params}` factory | `scripts/run_slice.py:33` `_instantiate` | **Hoist to shared util** — this is acceptance #4's mechanism |
| Chunk loop + reassembly | `scripts/run_slice.py:101-124` | This block is what `ReconstructionEngine.run` absorbs |
| Provenance spine | `src/mri_ad/utils/{run_logger,seed,device}.py` | `RunLogger` context manager; needs a `run_id` accessor |
| Model interface | `src/mri_ad/models/base.py`, `registry.py` | `AnomalyDetectionModel`; `build_default_registry()` |
| `ReconError` | `src/mri_ad/exceptions.py` | Already defined |
| Sweep axes | `configs/threshold/percentile.yaml:sweep` | `percentiles: [90, 92.5, 95, 97.5, 99]`, `absolute: [0.05, 0.1, 0.15, 0.2]` — already present |

Boundary rule from `tests/test_model_boundary.py`: `models/` may never import `recon/`. Dependency arrow is recon → models only.

---

## Files to create / modify

### New

| File | Contents |
|---|---|
| `src/mri_ad/recon/engine.py` | `ReconstructionEngine` |
| `src/mri_ad/recon/io.py` | `save_result`, `load_result`, `result_path` |
| `src/mri_ad/recon/sweep.py` | `SweepPoint`, `build_sweep_strategies`, `run_threshold_sweep`, `select_operating_point` |
| `src/mri_ad/utils/instantiate.py` | `instantiate_from_config(node)` — hoisted from `run_slice.py` |
| `configs/threshold/absolute.yaml` | `AbsoluteThreshold`, `value: 0.1` (the prior work's constant) |
| `configs/threshold/otsu.yaml` | `OtsuThreshold`, `bins: 256` |
| `scripts/run_sweep.py` | Hydra entry point for acceptance #5 |
| `tests/test_recon.py` | Acceptance tests 1–7 |

### Modified

| File | Change |
|---|---|
| `src/mri_ad/recon/threshold.py` | Make percentile exact; add `AbsoluteThreshold`, `OtsuThreshold` |
| `src/mri_ad/recon/types.py` | Fix shape docstring; add `ground_truth: Tensor \| None = None` |
| `src/mri_ad/recon/__init__.py` | Export engine, result types, threshold strategies, io helpers |
| `src/mri_ad/utils/run_logger.py` | Add `run_id` property (`self.run_dir.name`) |
| `src/mri_ad/utils/__init__.py` | Export `instantiate_from_config` |
| `scripts/run_slice.py` | Delete local `_instantiate`; route reconstruction through `ReconstructionEngine` |
| `configs/config.yaml` | Add a `recon:` block (`save_results`, `batch_size`, `results_dir`) |
| `configs/data/default.yaml` | Add `eval_subset: null` (referenced by both experiment configs, undefined in the base — a latent `MissingMandatoryValue`) |
| `Makefile` | `sweep` target + `.PHONY` + help text, marked cluster/manual |
| `progress_report.md` | Append the Spec 003 entry (mandatory) |

---

## Design

### 1. Threshold strategies — `recon/threshold.py`

All three implement the `ThresholdStrategy` Protocol: `__call__(residual: Tensor) -> Tensor`, same shape, values in `{0.0, 1.0}`, same dtype as input.

**`FixedPercentileThreshold(percentile: float = 95.0)`** — current implementation computes `torch.quantile` then `residual > cutoff`, which under ties or duplicate values flags an unpredictable count. Acceptance #3 demands *exactly* `(100-p)%` of voxels ±1. Replace with rank selection:

```
n    = residual.numel()
k    = int(round(n * (100.0 - p) / 100.0))        # exact target count
if k == 0: return zeros_like(residual)
idx  = torch.argsort(residual.flatten(), descending=True, stable=True)[:k]
mask = zeros(n); mask[idx] = 1.0; return mask.view_as(residual)
```

`stable=True` keeps it deterministic; exactly `k` voxels by construction, and it sidesteps `torch.quantile`'s ~16M-element input cap. Sorting 2.6M floats per volume is negligible next to the forward pass. The existing loose test (`abs(mask.mean() - 0.05) < 0.005`) tightens to an exact count assertion.

**`AbsoluteThreshold(value: float)`** — `(residual > value).to(residual.dtype)`. Trivial, but this is the strategy that actually reproduces the prior work.

**`OtsuThreshold(bins: int = 256)`** — pure-torch histogram Otsu (`torch.histc` over `[min, max]` computed from the data), not `skimage.filters.threshold_otsu`. Reasons: no numpy round-trip, deterministic, and it handles **unbounded residuals** correctly — `models/base.py` warns that only UNETR bounds output to `[0,1]` (Sigmoid); UNet/AttUNet do not, so residuals can exceed 1.0. A fixed `[0,1]` binning would silently clip.

> The unbounded-output fact also means `AbsoluteThreshold` at 0.05–0.2 behaves differently per architecture. That is a real finding for the sweep to surface, not a bug to paper over — record it in the sweep output.

### 2. `ReconstructionEngine` — `recon/engine.py`

```python
class ReconstructionEngine:
    def __init__(self, model: AnomalyDetectionModel, config: DictConfig) -> None: ...
    def run(self, volume: Tensor, *, volume_id: str,
            ground_truth: Tensor | None = None) -> ReconResult: ...
    def run_dataset_volume(self, dataset: BraTSDataset, volume_index: int) -> ReconResult: ...
```

*Deviation from the spec's `run(volume) -> ReconResult` signature:* `volume_id` is added as a required keyword. The spec mandates the persistence path `.../<volume_id>.pt`, and `BraTSDataset.__getitem__` deliberately returns no path or subject id (NFR-13, no patient identifiers) — so the id must be threaded in explicitly. Note this in the module docstring.

`run()` behaviour:
1. Validate input is `(1, D, H, W)` float32 with `D % 16 == 0`, `H == W == 128`; raise `ReconError` otherwise. Accept `(D,H,W)` and unsqueeze.
2. Assert `model.training is False` — raise `ReconError` if a model in train mode is handed in (dropout/BN would make reconstructions nondeterministic).
3. `chunk_volume(...)` → `(N,1,16,128,128)`; forward in batches of `cfg.recon.batch_size` under `torch.no_grad()`, on `DeviceManager.get_device(cfg.device)`, moving each output back to CPU immediately.
4. `torch.cat(chunks, dim=1)` along **depth** → `(1,D,128,128)` (the `dim=1` detail from `run_slice.py:119` — getting this wrong silently reassembles along the wrong axis).
5. `residual = (original - reconstruction).abs()`.
6. `anomaly_mask = self._threshold(residual)` — applied **globally over the whole volume**, once, never per chunk. Per-chunk percentile thresholding would flag 5% of *every* chunk including tumour-free ones; that is prior-work bug #2 in a new costume.
7. Return `ReconResult(volume_id, original, reconstruction, residual, anomaly_mask, ground_truth)` with every tensor `.detach().cpu().float().contiguous()`.

`run_dataset_volume()` wraps the chunk-gathering loop: iterate the dataset's chunks for one `volume_index`, `cat` image and label chunks to full depth, resolve `volume_id = dataset.volume_ids[volume_index]`, delegate to `run()`. This is where the "accumulate across all chunks" discipline lives.

The threshold is built once in `__init__` via `instantiate_from_config(config.threshold)` — that single line is what makes acceptance #4 (swap by config, no code change) true.

### 3. `ReconResult` + persistence — `recon/types.py`, `recon/io.py`

Add `ground_truth: Tensor | None = None` as a sixth field. Rationale: the spec's own docstring says `ReconResult` is "the **only** thing the evaluation harness reads", and both Spec 004 and the sweep need the binarized GT (`seg > 0`, nearest-neighbour resized — already handled by `data/transforms.py::_binarize_label`) alongside the residual. Without it, every consumer re-opens the dataset and the "only thing read" claim is false. The four spec-mandated tensors are unchanged; this is an additive, defaulted field.

```python
def result_path(root: Path, run_id: str, volume_id: str) -> Path      # root/<run_id>/<volume_id>.pt
def save_result(result: ReconResult, root: Path, run_id: str) -> Path # mkdir parents, torch.save
def load_result(path: Path) -> ReconResult
```

Serialize as a plain `dict[str, Tensor | str]` rather than pickling the dataclass — a pickled frozen dataclass breaks on any future field rename, and `torch.load(weights_only=True)` refuses arbitrary classes. `load_result` reconstructs the dataclass. Bit-identical round-trip (acceptance #7) follows from storing contiguous float32 CPU tensors and asserting with `torch.equal`, not `allclose`.

The `<run_id>` subdirectory is mandatory (acceptance #6). `run_id` comes from `RunLogger.run_id` — a new property returning `self.run_dir.name` (the existing `%Y%m%dT%H%M%S_%fZ` stamp, already unique per run).

**Disk budget:** 6 tensors × 160×128×128 × 4 B ≈ **63 MB per volume**; the BraTS val split (~74 volumes at 0.2) is ≈ 4.6 GB. Gate persistence behind `cfg.recon.save_results` (default `true`) so a cluster sweep can stream in-memory with `+recon.save_results=false`. `artifacts/results/recon/` is already git-ignored.

### 4. Sweep — `recon/sweep.py` + `scripts/run_sweep.py`

`sweep.py` is **pure and model-free**: it takes an iterable of `ReconResult` (each carrying `residual` + `ground_truth`) and re-thresholds them. One expensive forward pass over the val split, then every sweep point scored offline — and it makes the whole sweep unit-testable on synthetic tensors.

```python
@dataclass(frozen=True)
class SweepPoint:
    strategy: str        # "percentile" | "absolute" | "otsu"
    param: str           # "percentile" | "value" | "-"
    value: float | None
    dice_mean: float
    dice_std: float
    n_volumes: int
    split: str           # always "val"

def build_sweep_strategies(sweep_cfg) -> list[tuple[SweepPoint-key, ThresholdStrategy]]
def run_threshold_sweep(results, sweep_cfg, *, split) -> list[SweepPoint]
def select_operating_point(points) -> SweepPoint   # argmax dice_mean, ties -> lowest value
```

`run_threshold_sweep` raises `ReconError` if `split != "val"` — leakage guard at the function boundary, not just in the script (trap #5: never tune a threshold on test).

`scripts/run_sweep.py` follows the `run_slice.py` template exactly (`seed_everything` → `RunLogger` → device → registry → model → engine):
1. Build or load+`verify` the `SplitContract` at `cfg.data.split.contract_path`, take `contract.brats["val"]` (honouring `cfg.data.eval_subset`).
2. For each val volume: `engine.run_dataset_volume(...)`, optionally `save_result`, append to the in-memory list.
3. `run_threshold_sweep(results, cfg.threshold.sweep, split="val")` — raise a clear `ConfigError` if `cfg.threshold.sweep` is absent (it lives in `configs/threshold/percentile.yaml`, so the script must be run with `threshold=percentile`, which is the default).
4. Write `artifacts/results/threshold_sweep_<model>.csv` (one row per `SweepPoint`) and `sweep_summary.json` (chosen point + model name + n_volumes + split).
5. `run.record(best_strategy=..., best_value=..., best_dice=...)` and print the argmax.

No plot — visualization is Spec 010's. The CSV is the artifact; the curve gets drawn there.

**The chosen operating point is not auto-written into config.** The script reports it; the user reviews the numbers and edits `configs/threshold/*.yaml` by hand, with the justification recorded in `progress_report.md`. An auto-updating config would make the headline number untraceable.

### 5. Makefile

```make
sweep:  ## Spec 003: Dice-vs-threshold sweep on the VALIDATION split (GPU; manual)
	$(PY) scripts/run_sweep.py $(HYDRA_OVERRIDES)
```

Add to `.PHONY`. Comment it as GPU work to be run on the cluster, e.g. `make sweep HYDRA_OVERRIDES="+experiment=cluster model=unetr"`.

---

## Tests — `tests/test_recon.py`

Follow house conventions: `REPO = Path(__file__).resolve().parent.parent`, `pytest.importorskip("monai")`, `tmp_path` for artifact roots, `OmegaConf.create` mini-configs modelled on `tests/test_data.py::_base_cfg`.

**No real model forward passes.** Define a `_StubModel(AnomalyDetectionModel)` that returns `x * 0.9 + noise` (seeded) with a trivial `model_card` and no-op `load_checkpoint`. This exercises every engine code path at full `(1,160,128,128)` shape while staying laptop-safe, and keeps the tests fast and deterministic. One small-UNETR smoke test may reuse `tests/test_slice.py::small_unetr`'s downsized config if a real-module path is wanted, marked `@pytest.mark.slow`.

| # | Acceptance criterion | Test |
|---|---|---|
| 1 | Four tensors, correctly shaped, any registered model | `run()` on stub + one small real model → assert all fields present, `(1,160,128,128)`, float32, `.device.type == "cpu"` |
| 2 | `residual >= 0`, mask ⊆ `{0,1}` | `assert (r >= 0).all()`; `assert set(mask.unique().tolist()) <= {0.0, 1.0}` |
| 3 | Percentile flags exactly `(100-p)%` ±1 | Parametrize p ∈ {90, 95, 99}; assert `int(mask.sum()) == round(n*(100-p)/100)`. Include a residual with **many duplicate values** (`torch.zeros` + a few spikes) — the case the old `>` implementation fails |
| 4 | Strategy swap is config-only | Build engines from each of the three `configs/threshold/*.yaml` via `instantiate_from_config`; assert the strategy type and that masks differ. Zero test-side imports of concrete strategy classes |
| 5 | Sweep produces a Dice curve, argmax on val | `run_threshold_sweep` over hand-built `ReconResult`s where the ground truth is a known cube and the residual is engineered so a *known* percentile wins; assert `select_operating_point` finds it, assert one row per configured sweep point, assert `split="test"` raises `ReconError` |
| 6 | Two models don't collide | Save results for the same `volume_id` under two `run_id`s; assert two distinct files, both loadable, contents differ |
| 7 | Bit-identical round-trip | `save_result` → `load_result` → `torch.equal` on all six tensors + `volume_id` string equality |

Plus: `ReconError` on wrong input shape; `ReconError` when `model.training is True`; a boundary test that no module outside `recon/` calls a threshold (extend `tests/test_model_boundary.py` with a grep for `threshold`/`> 0.1`-style binarization in `eval/`, `classical/`, `viz/`).

---

## Order of work

1. `utils/instantiate.py` + `RunLogger.run_id` + config additions (`recon:` block, `data.eval_subset`) — the shared plumbing.
2. `recon/threshold.py`: exact percentile, `AbsoluteThreshold`, `OtsuThreshold` + the three YAMLs. Tests 3, 4.
3. `recon/types.py` docstring + `ground_truth` field; `recon/io.py`. Tests 6, 7.
4. `recon/engine.py`. Tests 1, 2.
5. Refactor `scripts/run_slice.py` onto the engine — must still produce an identical Dice for a fixed seed/volume (regression guard on the refactor).
6. `recon/sweep.py` + `scripts/run_sweep.py` + Makefile. Test 5.
7. `ruff format/check`, full `pytest`, append `progress_report.md`.

---

## Risks & watch-items

- **R1 — Percentile exactness.** The current `residual > quantile` fails acceptance #3 on tied values. The rank-selection rewrite is the fix; the duplicate-heavy test case is what proves it.
- **R2 — Unbounded reconstructions.** UNet/AttUNet have no output activation, so residuals may exceed 1.0. Affects absolute thresholds at 0.05–0.2 and any fixed-range binning. Otsu must derive its range from the data. Report per-architecture residual ranges in the sweep output.
- **R3 — Refactoring `run_slice.py` changes its number.** The engine applies the threshold globally over the reassembled volume, exactly as `run_slice.py` does today, so Dice should be unchanged. Verify explicitly rather than assume.
- **R4 — Disk.** ~63 MB/volume. `cfg.recon.save_results=false` for large sweeps.
- **R5 — `build_dataloader` consumes a batch at construction** (`data/loaders.py` validates `next(iter(loader))`). The engine iterates `BraTSDataset` directly per volume rather than going through `build_dataloader`, sidestepping this — but don't "helpfully" swap in the dataloader later without accounting for it.
- **R6 — Leakage.** The only defence that matters: `run_threshold_sweep` refuses any split but `val`, and `run_sweep.py` reads only `contract.brats["val"]`.

---

## Verification

**Laptop (safe, no GPU, no data):**
```bash
make lint
python -m pytest tests/test_recon.py -v          # acceptance 1-7
make test                                        # full suite + coverage, no regressions
python -c "from mri_ad.recon import ReconstructionEngine, save_result, load_result"
```
Config composition without touching data:
```bash
python -c "from hydra import compose, initialize; \
  [print(compose('config', overrides=[f'threshold={t}']).threshold.target) \
   for t in ('percentile','absolute','otsu')]"   # proves acceptance #4 end-to-end
```

**Cluster (manual invoke — propose, do not run autonomously):**
```bash
make check-data
make slice  HYDRA_OVERRIDES="+experiment=cluster model=unetr"   # Dice unchanged vs pre-refactor
make sweep  HYDRA_OVERRIDES="+experiment=cluster model=unetr"
```
Then inspect `artifacts/results/threshold_sweep_unetr.csv` (one row per sweep point, `split=val` on every row), `sweep_summary.json`, and `artifacts/runs/<stamp>/run_meta.json` (git SHA, seed, resolved config, best point). Confirm `artifacts/results/recon/<run_id>/` holds one `.pt` per val volume and that a second model run creates a *different* `run_id` directory.

**Done when:** all seven acceptance tests pass, the sweep CSV exists with a defensible argmax on the validation split, the chosen operating point is committed to `configs/threshold/` *with its justification written into `progress_report.md`*, and `run_slice.py` no longer contains any thresholding or reconstruction logic of its own.
