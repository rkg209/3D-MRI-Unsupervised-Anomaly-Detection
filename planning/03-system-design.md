# system-design.md

## 3D MRI Unsupervised Anomaly Detection — System Design

---

## 1. Modules

The codebase is organized as a Python package `src/mri_ad/` with eight sub-packages. Each sub-package maps to exactly one functional responsibility. The dependency rule is strict: a module may only import from modules in the same layer or a lower-numbered layer. `utils/` is the single exception — it is importable from any layer.

---

### 1.1 `src/mri_ad/data/`

| File | Class / Symbol | Role |
|---|---|---|
| `datasets.py` | `OpenBHBDataset` | MONAI Dataset wrapping healthy T1 volumes |
| `datasets.py` | `BraTSDataset` | MONAI Dataset wrapping anomalous volumes + segmentation labels |
| `transforms.py` | `TransformPipeline` | Composed MONAI transform chain; shared by both datasets |
| `transforms.py` | `LabelTransformPipeline` | Extends `TransformPipeline` with label-specific transforms for BraTS |
| `split.py` | `SplitContract` | Deterministic train/val/test split; serializes to `artifacts/split_contract.json` |
| `validation.py` | `DataValidator` | Asserts shape `(B,1,16,128,128)`, dtype `float32`, values `∈ [0,1]` at dataloader init |
| `validation.py` | `DescriptiveValidationError` | Raised with actual shape and value range on any contract violation |
| `loaders.py` | `build_train_loader()` | Factory: returns `DataLoader` over `OpenBHBDataset` with `CacheDataset` wrapping |
| `loaders.py` | `build_eval_loader()` | Factory: returns `DataLoader` over `BraTSDataset` |

**Internal dependencies:** `transforms.py` ← `datasets.py` ← `loaders.py` ← `split.py`. `validation.py` is called inside `loaders.py` at construction time.

**External dependencies:** `monai.data`, `monai.transforms`, `torch`, `nibabel` (via MONAI).

---

### 1.2 `src/mri_ad/models/`

| File | Class / Symbol | Role |
|---|---|---|
| `base.py` | `AnomalyDetectionModel` (ABC) | Abstract base: `forward()`, `load_checkpoint()`, `model_card` property |
| `base.py` | `ModelCard` (dataclass) | Metadata: name, param count, input/output shape, training data, known characteristics |
| `base.py` | `CheckpointError` | Raised when checkpoint loading fails (missing file, shape mismatch, corrupt weights) |
| `unet.py` | `UNetModel` | Wraps `monai.networks.nets.UNet`; configurable channels and strides |
| `attention_unet.py` | `AttentionUNetModel` | Wraps `monai.networks.nets.AttentionUnet` |
| `unetr.py` | `UNETRModel` | Wraps `monai.networks.nets.UNETR`; primary model; img_size `(16,128,128)` |
| `registry.py` | `ModelRegistry` | Singleton; maps string names to `AnomalyDetectionModel` instances; validates ABC compliance at registration |

**Internal dependencies:** `base.py` ← `unet.py`, `attention_unet.py`, `unetr.py` ← `registry.py`.

**External dependencies:** `monai.networks.nets`, `torch`, `omegaconf`.

---

### 1.3 `src/mri_ad/recon/`

| File | Class / Symbol | Role |
|---|---|---|
| `engine.py` | `ReconstructionEngine` | Orchestrates forward pass, residual computation, thresholding; writes `ReconResult` to disk |
| `result.py` | `ReconResult` (dataclass) | Holds `original`, `reconstruction`, `residual`, `anomaly_mask` — all `Tensor(1,16,128,128)` |
| `threshold.py` | `ThresholdStrategy` (Protocol) | Callable protocol: `(residual: Tensor) → binary Tensor` |
| `threshold.py` | `FixedPercentileThreshold` | Thresholds at configured percentile (default: 95th); default for all reported results |
| `threshold.py` | `OtsuThreshold` | Per-volume Otsu on residual histogram via `skimage.filters.threshold_otsu` |
| `threshold.py` | `AdaptiveThreshold` | Configurable fallback; selected via `configs/experiment/*.yaml` |
| `io.py` | `save_recon_result()` | `torch.save` wrapper; writes to `artifacts/results/recon/<volume_id>.pt` |
| `io.py` | `load_recon_result()` | `torch.load` wrapper; validates loaded tensor shapes before returning |

**Internal dependencies:** `result.py` ← `threshold.py` ← `engine.py` ← `io.py`.

**External dependencies:** `torch`, `skimage.filters`, `monai.networks.nets` (via injected model only — no direct import).

---

### 1.4 `src/mri_ad/eval/`

| File | Class / Symbol | Role |
|---|---|---|
| `metrics.py` | `MetricsComputer` | Single source of truth for Dice, IoU, PSNR, SSIM formulas |
| `metrics.py` | `VolumeMetrics` (dataclass) | Per-volume: `dice`, `iou`, `psnr`, `ssim`, `volume_id` |
| `metrics.py` | `AggregateMetrics` (dataclass) | Split-level: mean/std for each metric; list of `VolumeMetrics` |
| `evaluator.py` | `VolumeEvaluator` | Calls `MetricsComputer` on a single `ReconResult` + label tensor |
| `evaluator.py` | `AggregateEvaluator` | Iterates saved `.pt` files in `results_dir`; raises `TypeError` if a model object is passed |
| `report.py` | `ReportGenerator` | Reads only `Path` arguments; writes CSV, LaTeX tables, ROC/PR/Dice-distribution plots |
| `report.py` | `RunMetrics` (dataclass) | Holds one run's `AggregateMetrics` plus run metadata for comparison tables |

**Internal dependencies:** `metrics.py` ← `evaluator.py` ← `report.py`.

**External dependencies:** `torch`, `monai.metrics.SSIMMetric`, `matplotlib`, `pandas`, `scipy`.

---

### 1.5 `src/mri_ad/classical/`

| File | Class / Symbol | Role |
|---|---|---|
| `features.py` | `FeatureExtractor` | Extracts ~40 per-slice features: intensity statistics, GLCM texture, Sobel gradient, radiomic proxies |
| `baseline.py` | `ClassicalBaseline` | `StratifiedKFold` cross-validation over XGBoost or LightGBM; `fit()`, `evaluate()`, `save()` |
| `baseline.py` | `ClassicalMetrics` (dataclass) | `roc_auc_mean/std`, `pr_auc_mean/std`, `n_folds`, `granularity="slice-level"` |

**Internal dependencies:** `features.py` ← `baseline.py`.

**External dependencies:** `numpy`, `skimage`, `pyradiomics`, `xgboost` or `lightgbm`, `sklearn`.

**Explicit non-dependencies:** No imports from `models/`, `recon/`, or `eval/`.

---

### 1.6 `src/mri_ad/synth/`

| File | Class / Symbol | Role |
|---|---|---|
| `fpi.py` | `FPIAnomalyGenerator` | Foreign Patch Interpolation: blends donor patches into a healthy volume at random locations |
| `fpi.py` | `SynthResult` (dataclass) | `corrupted`, `healthy`, `synth_mask` — all `Tensor(1,16,128,128)` |
| `dataset.py` | `AnomalyInformedDataset` | Wraps `OpenBHBDataset`; applies `FPIAnomalyGenerator` on-the-fly; returns `{"corrupted", "healthy", "mask"}` |

**Internal dependencies:** `fpi.py` ← `dataset.py`.

**External dependencies:** `torch`, `monai.data`, `numpy`.

---

### 1.7 `src/mri_ad/viz/`

| File | Class / Symbol | Role |
|---|---|---|
| `viewer.py` | `VolumeViewer` | 5-panel Matplotlib viewer (original, recon, residual, mask, GT overlay); `show_interactive()` for Jupyter, `export_video()` for headless |
| `exporter.py` | `DemoExporter` | Loads saved `ReconResult`; calls `VolumeViewer.export_video()`; triggered by `make demo` |

**Internal dependencies:** `viewer.py` ← `exporter.py`.

**External dependencies:** `matplotlib`, `ipywidgets`, `imageio[ffmpeg]`.

---

### 1.8 `src/mri_ad/utils/`

| File | Class / Symbol | Role |
|---|---|---|
| `seed.py` | `seed_everything(seed: int)` | Sets `random`, `numpy`, `torch`, `cuda` seeds; enforces deterministic cuDNN |
| `logging.py` | `RunLogger` | Writes `run_meta.json` at run start and end; records git SHA, resolved config, seed, timing |
| `config.py` | `ConfigLoader` | Thin Hydra `compose` wrapper; validates required keys before returning `DictConfig` |
| `device.py` | `DeviceManager` | `get_device()` returns `cuda` / `mps` / `cpu`; `move()` transfers model or tensor to device |

**External dependencies:** `hydra-core`, `omegaconf`, `torch`, `numpy`. No imports from any other `src/mri_ad/` sub-package.

---

### 1.9 Entry-Point Scripts

Located at `scripts/` (not inside the package):

| Script | Makefile Target | Hydra `@hydra.main` Config |
|---|---|---|
| `scripts/run_slice.py` | `make slice` | `configs/experiment/slice.yaml` |
| `scripts/run_eval.py` | `make eval` | `configs/experiment/eval.yaml` |
| `scripts/run_train.py` | `make train` | `configs/experiment/train.yaml` |
| `scripts/run_report.py` | `make report` | `configs/experiment/report.yaml` |
| `scripts/run_classical.py` | `make classical` | `configs/experiment/classical.yaml` |
| `scripts/run_demo.py` | `make demo` | `configs/experiment/demo.yaml` |

Each script calls `seed_everything()` and `RunLogger.log_run_start()` as its first two actions, then delegates entirely to the appropriate module layer.

---

## 2. Services

This system has no network services. "Services" in this context means **stateful runtime objects** that are constructed once per process invocation and shared across the call graph. They are not singletons in the global-state sense — they are constructed by the entry-point script and passed down explicitly.

---

### 2.1 `ModelRegistry` — Model Lookup Service

**Lifecycle:** Constructed once per process. Populated by reading all `configs/models/*.yaml` files at startup.

**State:** A `dict[str, AnomalyDetectionModel]` mapping model names to instantiated (but not yet checkpoint-loaded) model objects.

**Operations:**
- `register(name, cls, config)` — validates ABC compliance; raises `TypeError` if `cls` does not implement `AnomalyDetectionModel`
- `get(name) → AnomalyDetectionModel` — raises `KeyError` with available names if not found
- `list_available() → list[str]` — used by CLI help text

**Who constructs it:** The entry-point script, immediately after `seed_everything()`.

---

### 2.2 `ReconstructionEngine` — Inference Service

**Lifecycle:** Constructed once per process. Holds a reference to a loaded model and a `ThresholdStrategy`.

**State:**
- `model: AnomalyDetectionModel` — checkpoint already loaded, moved to device
- `threshold_strategy: ThresholdStrategy` — injected from config
- `device: torch.device` — from `DeviceManager.get_device()`

**Operations:**
- `run(volume: Tensor) → ReconResult` — single-volume inference; moves input to device, runs forward pass, computes residual, applies threshold, moves result back to CPU
- `run_batch(loader: DataLoader) → Iterator[ReconResult]` — yields one `ReconResult` per volume; calls `save_recon_result()` after each

**Who constructs it:** `run_eval.py` and `run_slice.py` entry points.

---

### 2.3 `AggregateEvaluator` — Metrics Aggregation Service

**Lifecycle:** Constructed once per process. Stateless beyond its config.

**State:** `results_dir: Path`, `labels_dir: Path`, `metrics_computer: MetricsComputer`.

**Operations:**
- `evaluate_split() → AggregateMetrics` — iterates `.pt` files in `results_dir`; loads each `ReconResult`; loads corresponding label tensor; calls `VolumeEvaluator.evaluate()`; aggregates
- `write_csv(metrics: AggregateMetrics, path: Path)` — writes `per_volume.csv`
- `write_json(metrics: AggregateMetrics, path: Path)` — writes `aggregate.json`

**Who constructs it:** `run_eval.py` (after inference completes) and `run_report.py` (reads pre-saved files only).

---

### 2.4 `RunLogger` — Audit Service

**Lifecycle:** Constructed once per process. Writes to a timestamped directory under `artifacts/runs/`.

**State:** `run_dir: Path` (created at construction), `start_time: datetime`, `config_snapshot: DictConfig`.

**Operations:**
- `log_run_start(config)` — writes `run_meta.json` with git SHA, resolved config, seed, hostname, start time
- `log_run_end(metrics: dict)` — appends `end_time` and final metrics to `run_meta.json`
- `log_checkpoint(step: int, loss: float)` — appends to `training_log.jsonl` during training

**Who constructs it:** Every entry-point script, immediately after `seed_everything()`.

---

### 2.5 `ClassicalBaseline` — Classifier Service

**Lifecycle:** Constructed once per `run_classical.py` invocation. Holds fitted model state after `fit()`.

**State:** `model` (XGBoost or LightGBM estimator), `config: DictConfig`, `is_fitted: bool`.

**Operations:**
- `fit(X, y)` — runs `StratifiedKFold` cross-validation; stores fold results
- `evaluate(X, y) → ClassicalMetrics` — raises `RuntimeError` if called before `fit()`
- `save(path)` — serializes fitted model via `joblib.dump`

**Who constructs it:** `run_classical.py`.

---

## 3. Internal Workflows

Each workflow corresponds to one Makefile target. Workflows are described as ordered steps with the responsible class named at each step.

---

### 3.1 Workflow: `make slice`

**Purpose:** Reconstruct one volume, produce a figure. Fast feedback loop. GPU optional.

```
Step 1  seed_everything(cfg.seed)                          [utils/seed.py]
Step 2  RunLogger.log_run_start(cfg)                       [utils/logging.py]
Step 3  ModelRegistry.get(cfg.model.name)                  [models/registry.py]
Step 4  model.load_checkpoint(cfg.checkpoints.path)        [models/unetr.py]
Step 5  DeviceManager.move(model, device)                  [utils/device.py]
Step 6  BraTSDataset(cfg, split="test")                    [data/datasets.py]
        → DataValidator asserts contract                   [data/validation.py]
Step 7  volume, label = dataset[cfg.slice.volume_index]    [data/datasets.py]
Step 8  ReconstructionEngine.run(volume)                   [recon/engine.py]
        → model.forward(volume)                            [models/unetr.py]
        → residual = abs(original - reconstruction)
        → FixedPercentileThreshold(residual)               [recon/threshold.py]
        → returns ReconResult
Step 9  VolumeViewer(result, label).export_video(...)      [viz/viewer.py]
        OR show_interactive() if in Jupyter
Step 10 RunLogger.log_run_end({"volume_id": ...})          [utils/logging.py]
```

**Output:** One figure saved to `artifacts/results/figures/slice_<volume_id>.png`.

---

### 3.2 Workflow: `make eval`

**Purpose:** Full test-split evaluation from saved checkpoints. Primary result-generation workflow.

```
Step 1  seed_everything(cfg.seed)                          [utils/seed.py]
Step 2  RunLogger.log_run_start(cfg)                       [utils/logging.py]
Step 3  ModelRegistry.get(cfg.model.name)                  [models/registry.py]
Step 4  model.load_checkpoint(cfg.checkpoints.path)        [models/unetr.py]
Step 5  DeviceManager.move(model, device)                  [utils/device.py]
Step 6  build_eval_loader(cfg)                             [data/loaders.py]
        → BraTSDataset with SplitContract(split="test")   [data/split.py]
        → DataValidator asserts contract on first batch    [data/validation.py]
Step 7  ReconstructionEngine.run_batch(loader)             [recon/engine.py]
        For each volume:
          → model.forward(volume)
          → compute residual
          → apply ThresholdStrategy
          → save_recon_result(result, artifacts/results/recon/)  [recon/io.py]
Step 8  AggregateEvaluator.evaluate_split(               [eval/evaluator.py]
            results_dir=artifacts/results/recon/,
            labels_dir=artifacts/results/labels/)
        For each saved ReconResult:
          → load_recon_result()                            [recon/io.py]
          → VolumeEvaluator.evaluate(result, label)        [eval/evaluator.py]
          → MetricsComputer.dice/iou/psnr/ssim             [eval/metrics.py]
Step 9  AggregateEvaluator.write_csv(metrics, ...)        [eval/evaluator.py]
        AggregateEvaluator.write_json(metrics, ...)
Step 10 RunLogger.log_run_end(aggregate_metrics.to_dict()) [utils/logging.py]
```

**Output:** `artifacts/runs/<timestamp>/metrics/per_volume.csv`, `aggregate.json`.

---

### 3.3 Workflow: `make train`

**Purpose:** Synthetic-anomaly fine-tuning of UNETR. Manual invocation only.

```
Step 1  seed_everything(cfg.seed)                          [utils/seed.py]
Step 2  RunLogger.log_run_start(cfg)                       [utils/logging.py]
Step 3  ModelRegistry.get("unetr")                         [models/registry.py]
Step 4  model.load_checkpoint(cfg.checkpoints.pretrained)  [models/unetr.py]
        → freeze encoder weights per cfg.training.freeze_encoder
Step 5  DeviceManager.move(model, device)                  [utils/device.py]
Step 6  AnomalyInformedDataset(cfg)                        [synth/dataset.py]
        → wraps OpenBHBDataset                             [data/datasets.py]
        → FPIAnomalyGenerator constructed with cfg.synth   [synth/fpi.py]
Step 7  build_train_loader(cfg) over AnomalyInformedDataset [data/loaders.py]
Step 8  Training loop (inside run_train.py):
        optimizer = AdamW(model.parameters(), lr=cfg.training.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
        loss_fn = MSELoss() + SSIMLoss() weighted by cfg.training.loss_weights
        For each epoch:
          For each batch {"corrupted", "healthy", "mask"}:
            → FPIAnomalyGenerator.generate(volume, donor)  [synth/fpi.py]
              (on-the-fly, inside AnomalyInformedDataset.__getitem__)
            → pred = model.forward(corrupted)
            → loss = loss_fn(pred, healthy)
            → loss.backward(); optimizer.step()
            → RunLogger.log_checkpoint(step, loss)         [utils/logging.py]
          scheduler.step()
Step 9  model.save_checkpoint(cfg.checkpoints.output)      [models/unetr.py]
Step 10 RunLogger.log_run_end({"final_loss": ...})         [utils/logging.py]
```

**Output:** `artifacts/checkpoints/unetr_synth.pt`, `artifacts/runs/synth/run_meta.json`.

---

### 3.4 Workflow: `make report`

**Purpose:** Regenerate all tables, plots, and README scorecard from saved artifacts. No GPU, no model.

```
Step 1  RunLogger.log_run_start(cfg)                       [utils/logging.py]
        (no seed needed — pure file transformation)
Step 2  ReportGenerator.generate_table(                    [eval/report.py]
            metrics=load aggregate.json,
            output_path=artifacts/results/tables/)
        → writes per-model CSV
        → writes LaTeX-formatted comparison table
Step 3  ReportGenerator.generate_plots(                    [eval/report.py]
            metrics_dir=artifacts/runs/*/metrics/,
            output_path=artifacts/results/figures/)
        → ROC curve (DL models + classical baseline overlaid)
        → PR curve
        → Dice distribution histogram per model
Step 4  ReportGenerator.generate_comparison_table(         [eval/report.py]
            runs=load all run_meta.json files)
        → Architecture × loss-function matrix
Step 5  DemoExporter.export(                               [viz/exporter.py]
            results_dir=artifacts/results/recon/,
            volume_id=cfg.report.demo_volume_id,
            output_path=artifacts/demo/demo_video.mp4)
Step 6  Inject scorecard into README.md between markers    [run_report.py]
        <!-- SCORECARD_START --> ... <!-- SCORECARD_END -->
Step 7  RunLogger.log_run_end({})                          [utils/logging.py]
```

**Output:** All figures, tables, demo video, updated README scorecard. Runs in < 2 minutes, no GPU.

---

### 3.5 Workflow: `make classical`

**Purpose:** Train and evaluate the classical ML baseline.

```
Step 1  seed_everything(cfg.seed)                          [utils/seed.py]
Step 2  RunLogger.log_run_start(cfg)                       [utils/logging.py]
Step 3  build_eval_loader(cfg) with SplitContract          [data/loaders.py]
        (same split as DL pipeline — enforced by SplitContract)
Step 4  For each volume in BraTSDataset:
          FeatureExtractor.extract(volume)                 [classical/features.py]
          → returns np.ndarray shape (16, ~40)
          → slice label: 1 if any tumor voxel in slice, else 0
        Accumulate X: (n_volumes*16, ~40), y: (n_volumes*16,)
Step 5  ClassicalBaseline.fit(X_train, y_train)            [classical/baseline.py]
        → StratifiedKFold(n_splits=cfg.classical.k_folds)
        → XGBoost or LightGBM per cfg.classical.classifier
Step 6  ClassicalBaseline.evaluate(X_test, y_test)         [classical/baseline.py]
        → ClassicalMetrics(roc_auc_mean, roc_auc_std, ...)
Step 7  ClassicalBaseline.save(artifacts/classical/model.joblib)
        write ClassicalMetrics to artifacts/classical/metrics.json
Step 8  RunLogger.log_run_end(metrics.to_dict())           [utils/logging.py]
```

**Output:** `artifacts/classical/metrics.json`, `model.joblib`.

---

## 4. Event Flows

"Events" in this system are not message-queue events. They are **control-flow transitions** between components — points where one component finishes and hands a result to the next. Each event is described with its trigger, payload, and consumer.

---

### 4.1 Event: `DataBatchReady`

**Trigger:** `DataLoader.__iter__` yields a batch.
**Producer:** `build_eval_loader()` or `build_train_loader()`
**Payload:** `dict{"image": Tensor(B,1,16,128,128), "label": Tensor(B,1,16,128,128)}`
**Consumer:** `ReconstructionEngine.run()` (eval) or training loop (train)
**Validation gate:** `DataValidator` fires synchronously inside the loader's collate step. If the batch fails validation, `DescriptiveValidationError` is raised before the payload reaches any consumer. The error message includes actual shape and value range.

---

### 4.2 Event: `ReconResultReady`

**Trigger:** `ReconstructionEngine.run()` completes for one volume.
**Producer:** `ReconstructionEngine`
**Payload:** `ReconResult(original, reconstruction, residual, anomaly_mask)` — all CPU tensors, shape `(1,16,128,128)`
**Consumers (two paths, mutually exclusive):**
- **Eval path:** `save_recon_result()` writes `.pt` file to disk → `AggregateEvaluator` reads it later
- **Slice path:** `VolumeViewer` consumes directly in memory; no disk write of the `.pt` file

**Invariant:** The `ReconResult` is always moved to CPU before being handed to any consumer. GPU memory is freed immediately after `run()` returns.

---

### 4.3 Event: `MetricsReady`

**Trigger:** `AggregateEvaluator.evaluate_split()` completes.
**Producer:** `AggregateEvaluator`
**Payload:** `AggregateMetrics` object + written CSV/JSON files
**Consumers:**
- `RunLogger.log_run_end()` — records metrics in `run_meta.json`
- `ReportGenerator` — reads the JSON/CSV files (not the in-memory object) during `make report`

**Decoupling note:** `ReportGenerator` never receives an `AggregateMetrics` object directly. It reads files. This is the Boundary 4 enforcement described in the architecture document.

---

### 4.4 Event: `SynthBatchReady`

**Trigger:** `AnomalyInformedDataset.__getitem__` returns.
**Producer:** `AnomalyInformedDataset` (which internally calls `FPIAnomalyGenerator.generate()`)
**Payload:** `dict{"corrupted": Tensor(1,16,128,128), "healthy": Tensor(1,16,128,128), "mask": Tensor(1,16,128,128)}`
**Consumer:** Training loop in `run_train.py`

**Timing:** FPI generation is on-the-fly per `__getitem__` call. The donor volume is selected randomly from the same dataset using a seeded RNG. This means the synthetic anomaly seen at epoch N differs from epoch N+1 (data augmentation effect), but the sequence is fully reproducible given the global seed.

---

### 4.5 Event: `CheckpointSaved`

**Trigger:** `model.save_checkpoint()` completes at end of training.
**Producer:** `UNETRModel.save_checkpoint()`
**Payload:** `.pt` file at `artifacts/checkpoints/unetr_synth.pt`
**Consumer:** Next `make eval` invocation — `ModelRegistry` loads this checkpoint via `cfg.checkpoints.path`

**Decoupling note:** Training and evaluation are separate process invocations. The checkpoint file on disk is the only coupling between them.

---

### 4.6 Event: `RunStartLogged` / `RunEndLogged`

**Trigger:** `RunLogger.log_run_start()` / `RunLogger.log_run_end()`
**Producer:** Every entry-point script
**Payload:** `run_meta.json` written to `artifacts/runs/<timestamp>-<model>-<loss>/`
**Consumer:** `ReportGenerator.generate_comparison_table()` — reads all `run_meta.json` files to assemble the architecture × loss matrix

---

## 5. State Transitions

The system has no persistent runtime state (no database, no server). State transitions describe how **artifact files on disk** evolve across workflow invocations, and how **in-process objects** transition through their lifecycle within a single invocation.

---

### 5.1 Artifact State Machine

```
INITIAL STATE: No artifacts on disk
        │
        ▼ make train (or: place pre-trained checkpoint manually)
CHECKPOINT_AVAILABLE
  artifacts/checkpoints/unetr_mse_ssim.pt  (or unetr_synth.pt)
        │
        ▼ make eval
RECON_RESULTS_AVAILABLE
  artifacts/results/recon/*.pt
  artifacts/runs/<timestamp>/metrics/per_volume.csv
  artifacts/runs/<timestamp>/metrics/aggregate.json
  artifacts/runs/<timestamp>/run_meta.json
        │
        ├──────────────────────────────────────────┐
        ▼ make classical                           ▼ make report
CLASSICAL_METRICS_AVAILABLE              REPORT_COMPLETE
  artifacts/classical/metrics.json         artifacts/results/figures/*.png
  artifacts/classical/model.joblib         artifacts/results/tables/*.csv
        │                                  artifacts/results/tables/*.tex
        └──────────────────────────────────artifacts/demo/demo_video.mp4
                                           README.md (scorecard updated)
```

**Transition guards:**
- `make eval` requires: checkpoint file exists at `cfg.checkpoints.path`. Fails with `CheckpointError` otherwise.
- `make report` requires: at least one `aggregate.json` in `artifacts/runs/*/metrics/`. Fails with `FileNotFoundError` with a message directing the user to run `make eval` first.
- `make classical` requires: BraTS data accessible at `cfg.data.root`. Independent of DL pipeline state.
- `make slice` requires: checkpoint file exists. Independent of eval state.

---

###