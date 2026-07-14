# architecture.md

## 3D MRI Unsupervised Anomaly Detection — Architecture Document

---

## 1. System Overview

This system is a **local research codebase** — not a service, not a deployed application. It is a Python monorepo that implements a reproducible comparative study of unsupervised anomaly detection in 3D brain MRI. There is no server, no API, no database, and no container orchestration. The "system" is a set of Python modules wired together by a Makefile, driven by Hydra configuration, and operated by a single engineer (or a Claude Code agent acting under that engineer's supervision).

The system has four operational modes:

| Mode | Trigger | GPU Required | Description |
|---|---|---|---|
| **Slice** | `make slice` | Optional | Load one checkpoint → reconstruct one volume → score → save figure |
| **Eval** | `make eval` | Yes (preferred) | Full test-split evaluation from saved checkpoints |
| **Train** | `make train` (manual only) | Yes | Synthetic-anomaly fine-tuning from UNETR checkpoint |
| **Report** | `make report` | No | Regenerate all tables, plots, README scorecard from saved artifacts |

The system processes 3D brain MRI volumes of shape `(1, 16, 128, 128)` — one channel, 16 depth slices, 128×128 in-plane — through a reconstruction pipeline. The anomaly signal is the residual between the original volume and its reconstruction. Detection quality is measured by Dice and IoU against BraTS ground-truth segmentation maps.

**The single architectural invariant:** every design decision preserves detection separability. Reconstruction fidelity (PSNR, SSIM) is measured only to explain failure modes, never to optimize.

---

## 2. Component Architecture

The system is organized into eight functional components under `src/mri_ad/`. Each component has a single responsibility, a defined interface, and no circular dependencies. The dependency graph is strictly layered.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                │
│   Makefile targets → Python CLI scripts (Hydra @hydra.main)         │
└────────────────────────┬────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────────┐
         ▼               ▼                   ▼
┌──────────────┐  ┌─────────────┐   ┌──────────────────┐
│   data/      │  │  models/    │   │    synth/         │
│  (Layer 1)   │  │  (Layer 1)  │   │   (Layer 1)       │
└──────┬───────┘  └──────┬──────┘   └────────┬─────────┘
       │                 │                   │
       └────────┬────────┘                   │
                ▼                            │
       ┌──────────────────┐                  │
       │     recon/       │◄─────────────────┘
       │    (Layer 2)     │
       └────────┬─────────┘
                │
       ┌────────┴──────────┐
       ▼                   ▼
┌─────────────┐   ┌──────────────────┐
│   eval/     │   │   classical/     │
│  (Layer 3)  │   │   (Layer 3)      │
└──────┬──────┘   └────────┬─────────┘
       │                   │
       └─────────┬─────────┘
                 ▼
       ┌──────────────────┐
       │      viz/        │
       │    (Layer 4)     │
       └──────────────────┘

       ┌──────────────────┐   (horizontal — used by all layers)
       │     utils/       │
       └──────────────────┘
```

### 2.1 `data/` — Data Layer

**Responsibility:** Load, validate, preprocess, and serve MRI volumes as PyTorch tensors.

**Key classes:**

```
OpenBHBDataset(monai.data.Dataset)
    - Wraps healthy T1 volumes for training
    - Applies the standard transform chain
    - Returns: {"image": Tensor(1,16,128,128)}

BraTSDataset(monai.data.Dataset)
    - Wraps anomalous volumes + segmentation maps for evaluation
    - Returns: {"image": Tensor(1,16,128,128), "label": Tensor(1,16,128,128)}

SplitContract
    - Deterministic train/val/test split given seed
    - Serializes split indices to artifacts/split_contract.json
    - Raises on any attempt to use a different split without explicit override

DataValidator
    - Called at dataloader init
    - Asserts shape == (B,1,16,128,128), dtype == float32, values in [0,1]
    - Raises DescriptiveValidationError on any violation

TransformPipeline
    - Composed MONAI transforms: LoadImage → EnsureChannelFirst →
      ScaleIntensityRange(a_min, a_max, b_min=0, b_max=1) →
      CropForeground → SpatialPad → DivisiblePad →
      RandSpatialCrop(roi_size=(1,16,128,128))
    - Identical pipeline for both datasets; BraTS adds label transforms
```

**Interface contract:** Every dataloader yields dicts with string keys `"image"` and (for BraTS) `"label"`. Shape and dtype are guaranteed by `DataValidator`. No raw file paths, patient IDs, or metadata escape this layer.

### 2.2 `models/` — Model Registry

**Responsibility:** Expose UNet, Attention-UNet, and UNETR behind a single interface. Load checkpoints. Provide model cards.

**Uniform interface (abstract base class):**

```python
class AnomalyDetectionModel(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Input: (B,1,16,128,128) → Output: (B,1,16,128,128)"""

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Load weights from path; raise CheckpointError on failure."""

    @property
    @abstractmethod
    def model_card(self) -> ModelCard:
        """Return ModelCard(name, param_count, input_shape, output_shape,
                            training_data, known_characteristics)"""
```

**Concrete implementations:**

```
UNetModel(AnomalyDetectionModel)
    - Wraps monai.networks.nets.UNet
    - Config: channels, strides, num_res_units

AttentionUNetModel(AnomalyDetectionModel)
    - Wraps monai.networks.nets.AttentionUnet
    - Config: channels, strides

UNETRModel(AnomalyDetectionModel)
    - Wraps monai.networks.nets.UNETR
    - Config: img_size=(16,128,128), feature_size, hidden_size,
              mlp_dim, num_heads, pos_embed, norm_name
    - This is the primary model; checkpoint is the existing trained weight

ModelRegistry
    - Singleton; populated from configs/models/*.yaml
    - get(name: str) -> AnomalyDetectionModel
    - register(name: str, cls: type, config: DictConfig) -> None
```

**Extension rule:** Adding a new model requires only: a new class implementing `AnomalyDetectionModel`, a new `configs/models/<name>.yaml`, and a checkpoint path. Zero changes to `recon/`, `eval/`, or any other layer.

### 2.3 `recon/` — Reconstruction & Anomaly-Map Engine

**Responsibility:** Given a model and a volume, produce reconstruction, residual map, and thresholded anomaly mask. Centralize all thresholding logic.

**Key classes:**

```
ReconstructionEngine
    - __init__(model: AnomalyDetectionModel, config: DictConfig)
    - run(volume: Tensor) -> ReconResult

ReconResult(dataclass)
    - original: Tensor(1,16,128,128)
    - reconstruction: Tensor(1,16,128,128)
    - residual: Tensor(1,16,128,128)      # abs(original - reconstruction)
    - anomaly_mask: Tensor(1,16,128,128)  # binary, thresholded

ThresholdStrategy(Protocol)
    - __call__(residual: Tensor) -> Tensor  # returns binary mask

FixedPercentileThreshold(ThresholdStrategy)
    - Threshold at config.threshold.percentile (default: 95th)

OtsuThreshold(ThresholdStrategy)
    - Per-volume Otsu on the residual histogram

AdaptiveThreshold(ThresholdStrategy)
    - Configurable; selected via configs/experiment/*.yaml
```

**Thresholding is the only place in the codebase where a residual becomes a binary mask.** No other module performs thresholding. The strategy is injected via config; the default for all reported results is `FixedPercentileThreshold` at the 95th percentile (matching the prior work).

### 2.4 `eval/` — Evaluation Harness

**Responsibility:** Compute metrics from saved `ReconResult` objects. Generate tables and plots. Never re-run inference.

**Key classes:**

```
MetricsComputer
    - dice(pred: Tensor, gt: Tensor, eps: float = 1e-6) -> float
      # 2*|pred∩gt| / (|pred| + |gt| + eps)
    - iou(pred: Tensor, gt: Tensor, eps: float = 1e-6) -> float
      # |pred∩gt| / (|pred∪gt| + eps)
    - psnr(recon: Tensor, orig: Tensor) -> float
      # 20*log10(1.0 / sqrt(mse)); range [0,1] assumed
    - ssim(recon: Tensor, orig: Tensor) -> float
      # monai.metrics.SSIMMetric

VolumeEvaluator
    - evaluate(result: ReconResult, label: Tensor) -> VolumeMetrics
    - Returns VolumeMetrics(dice, iou, psnr, ssim, volume_id)

AggregateEvaluator
    - evaluate_split(results_dir: Path, labels_dir: Path) -> AggregateMetrics
    - Loads saved ReconResult objects; never calls model.forward()
    - Returns AggregateMetrics(mean_dice, std_dice, mean_iou, std_iou, ...)

ReportGenerator
    - generate_table(metrics: AggregateMetrics, output_path: Path) -> None
      # Writes CSV + LaTeX-formatted table
    - generate_plots(metrics_dir: Path, output_path: Path) -> None
      # ROC curve, PR curve, Dice distribution histogram
    - generate_comparison_table(runs: List[RunMetrics]) -> None
      # Architecture × loss matrix
```

**Metric definitions are canonical and defined once.** `MetricsComputer` is the single source of truth for Dice, IoU, PSNR, and SSIM formulas. No other module computes these.

### 2.5 `classical/` — Classical-ML Baseline

**Responsibility:** Extract features from MRI volumes, train a gradient-boosting classifier, evaluate with cross-validation.

**Operating granularity decision (resolved here, not deferred):** **Slice-level classification.** Each 2D slice (128×128) is one sample. Label: 1 if the corresponding BraTS segmentation slice contains any tumor voxel, 0 otherwise. This choice produces clean ROC-AUC curves comparable to standard binary classification benchmarks, avoids the complexity of supervoxel segmentation, and is honest about the granularity gap vs. voxel-level Dice. The gap is explicitly documented in the comparison narrative.

```
FeatureExtractor
    - extract(volume: Tensor) -> np.ndarray  # shape (16, n_features)
    - Features per slice:
        Intensity: mean, std, skewness, kurtosis, percentiles (5,25,50,75,95)
        Texture (GLCM via skimage): contrast, dissimilarity, homogeneity,
                                    energy, correlation (4 angles, averaged)
        Gradient: mean gradient magnitude (Sobel)
        Radiomic proxies: run-length features (via PyRadiomics on 2D slice)
    - Total: ~40 features per slice

ClassicalBaseline
    - __init__(config: DictConfig)
    - fit(X: np.ndarray, y: np.ndarray) -> None
      # StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=seed)
      # XGBoost or LightGBM; choice in configs/classical/classifier.yaml
    - evaluate(X: np.ndarray, y: np.ndarray) -> ClassicalMetrics
      # Returns mean±std ROC-AUC, PR-AUC across folds
    - save(path: Path) -> None  # Persists fitted model

ClassicalMetrics(dataclass)
    - roc_auc_mean: float
    - roc_auc_std: float
    - pr_auc_mean: float
    - pr_auc_std: float
    - n_folds: int
    - granularity: str  # always "slice-level"
```

### 2.6 `synth/` — Synthetic-Anomaly Module

**Responsibility:** Generate plausible 3D synthetic lesions on healthy volumes for anomaly-informed training.

**Generation technique (resolved here):** **Foreign Patch Interpolation (FPI)** — a technique that blends patches from different healthy volumes at random locations with random intensity scaling, producing synthetic anomalies that are spatially coherent but intensity-inconsistent with surrounding tissue. FPI is chosen over Poisson blending (requires mask boundary computation, slower) and 3D CutPaste (produces too-sharp boundaries that are trivially detectable). FPI is documented in the Spec 009 plan.

```
FPIAnomalyGenerator
    - __init__(config: DictConfig)
    - generate(volume: Tensor, donor_volume: Tensor) -> SynthResult

SynthResult(dataclass)
    - corrupted: Tensor(1,16,128,128)   # input to model during training
    - healthy: Tensor(1,16,128,128)     # reconstruction target
    - synth_mask: Tensor(1,16,128,128)  # where corruption was applied

FPIConfig (in configs/synth/fpi.yaml)
    - patch_size_range: [8, 32]         # voxels, cubic
    - n_patches_range: [1, 4]
    - alpha_range: [0.3, 0.7]           # interpolation weight
    - seed: inherited from global seed

AnomalyInformedDataset(monai.data.Dataset)
    - Wraps OpenBHB; applies FPIAnomalyGenerator on-the-fly
    - Returns: {"corrupted": Tensor, "healthy": Tensor, "mask": Tensor}
```

### 2.7 `viz/` — Visualization & Demo Export

**Responsibility:** Render multi-panel views of reconstruction results; export demo video.

**Viewer technology decision (resolved here):** **Matplotlib-based interactive viewer** running in a Jupyter notebook, with a separate headless rendering path for demo video export. NiiVue requires a browser server; Gradio adds a web-app dependency. Matplotlib with `ipywidgets` sliders for slice navigation is sufficient for the portfolio use case, runs locally without a server, and the headless path (Matplotlib `Agg` backend + `imageio`/`ffmpeg`) produces the demo video reproducibly from a script.

```
VolumeViewer
    - __init__(result: ReconResult, label: Tensor)
    - show_interactive() -> None
      # Jupyter widget: 5-panel (original, recon, residual, mask, GT overlay)
      # Slider for depth-slice navigation
    - export_video(output_path: Path, fps: int = 4) -> None
      # Renders all 16 depth slices × 5 panels → MP4 via imageio[ffmpeg]

DemoExporter
    - export(results_dir: Path, volume_id: str, output_path: Path) -> None
      # Loads saved ReconResult, calls VolumeViewer.export_video
      # Triggered by: make demo or make report
```

### 2.8 `utils/` — Cross-Cutting Utilities

**Responsibility:** Seeds, logging, config loading, registry helpers. Used by all layers; depends on nothing in `src/mri_ad/`.

```
seed_everything(seed: int) -> None
    - Sets: random.seed, np.random.seed, torch.manual_seed,
            torch.cuda.manual_seed_all
    - Sets: torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    - Called once at the top of every entry-point script

RunLogger
    - log_run_start(config: DictConfig) -> None
      # Writes to artifacts/runs/<timestamp>/run_meta.json:
      #   git_sha, resolved_config, seed, start_time, hostname
    - log_run_end(metrics: dict) -> None
      # Appends end_time, metrics to run_meta.json

ConfigLoader
    - Thin wrapper around Hydra compose API
    - Validates required keys are present before returning config

DeviceManager
    - get_device() -> torch.device
      # Returns cuda if available, mps if on Apple Silicon, else cpu
    - move(model_or_tensor, device) -> same type
```

---

## 3. Data Flow

### 3.1 Evaluation Flow (Primary — no training)

```
DISK                    DATA LAYER              RECON ENGINE           EVAL HARNESS
─────                   ──────────              ────────────           ────────────

checkpoints/            ModelRegistry           ReconstructionEngine   MetricsComputer
  unetr_mse_ssim.pt ──► .get("unetr")         .run(volume)           .dice(mask, gt)
                         .load_checkpoint()                            .iou(mask, gt)
                              │                      │                      │
data/brats/             BraTSDataset                 │                      │
  *.nii.gz ──────────► .transform_pipeline          │                      │
                         DataValidator               │                      │
                              │                      │                      │
                              ▼                      ▼                      ▼
                         {"image": (B,1,16,128,128)  ReconResult{          VolumeMetrics{
                          "label": (B,1,16,128,128)} original,             dice, iou,
                                    │                reconstruction,        psnr, ssim}
                                    │                residual,                  │
                                    └──────────────► anomaly_mask}              │
                                                                               ▼
                                                                    artifacts/
                                                                      metrics/
                                                                        per_volume.csv
                                                                        aggregate.json
```

### 3.2 Report Flow (No GPU, No Model)

```
artifacts/              ReportGenerator         README.md
  metrics/              ────────────────        ─────────
    aggregate.json ───► .generate_table()  ───► Scorecard section
    per_volume.csv ───► .generate_plots()  ───► Figures embedded
    runs/*/            .generate_comparison     (make report writes
      run_meta.json      _table()               directly to README
                                                between markers)
```

### 3.3 Training Flow (Spec 009 only — manual invoke)

```
data/openbhb/           AnomalyInformedDataset  UNETR Fine-tune        artifacts/
  *.nii.gz ──────────► FPIAnomalyGenerator     ────────────────        ──────────
                         .generate(vol, donor)   optimizer: AdamW       checkpoints/
checkpoints/                  │                  loss: MSE+SSIM           unetr_synth.pt
  unetr_mse_ssim.pt ──► UNETRModel             scheduler: CosineAnneal  runs/synth/
  (frozen backbone,          │                  input: corrupted           run_meta.json
   fine-tune decoder)        ▼                  target: healthy
                         {"corrupted": Tensor                           → then: make eval
                          "healthy": Tensor                               (same test split)
                          "mask": Tensor}
```

### 3.4 Classical Baseline Flow

```
data/brats/             FeatureExtractor        ClassicalBaseline       artifacts/
  *.nii.gz ──────────► .extract(volume)        ────────────────        ──────────
                         per-slice features      StratifiedKFold         classical/
                         shape: (16, ~40)        XGBoost/LightGBM          metrics.json
                              │                  .fit(X_train, y_train)    roc_curve.png
                              │                  .evaluate(X_test, y_test) pr_curve.png
                              ▼
                         X: (n_slices, ~40)
                         y: (n_slices,)  # 1 if tumor present in slice
```

### 3.5 Volume Shape Contract (Enforced Throughout)

```
Raw NIfTI on disk:     Variable shape, variable intensity range
                              │
                              ▼ TransformPipeline
After preprocessing:   (1, 16, 128, 128)  float32  values ∈ [0, 1]
                              │
                              ▼ DataValidator (asserts at dataloader init)
Model input:           (B, 1, 16, 128, 128)  on device
                              │
                              ▼ model.forward()
Model output:          (B, 1, 16, 128, 128)  on device
                              │
                              ▼ ReconstructionEngine
ReconResult tensors:   (1, 16, 128, 128)  float32  cpu  values ∈ [0, 1]
```

The shape `(1, 16, 128, 128)` is the **system-wide invariant**. Any tensor that deviates from this shape at a component boundary is a bug, not a configuration option.

---

## 4. Service Boundaries

This system has no services in the network sense. "Service boundaries" here means **module interface contracts** — the points where one component hands off to another and what guarantees are made at each handoff.

### Boundary 1: Data Layer → Everything Else

**Contract:** Any consumer of the data layer receives only:
- `Tensor` objects of shape `(B, 1, 16, 128, 128)`, dtype `float32`, values in `[0, 1]`
- Dict keys `"image"` and `"label"` (BraTS only)
- No file paths, no patient identifiers, no raw metadata

**Enforcement:** `DataValidator` raises `DescriptiveValidationError` at dataloader initialization if any batch violates this contract. The error message includes the actual shape and value range observed.

### Boundary 2: Model Registry → Reconstruction Engine

**Contract:** `ModelRegistry.get(name)` returns an object satisfying `AnomalyDetectionModel`. The engine calls only `model.forward(x)` and `model.load_checkpoint(path)`. The engine never inspects model internals, never accesses model weights directly, and never calls optimizer methods.

**Enforcement:** `AnomalyDetectionModel` is an ABC. Any class that does not implement `forward`, `load_checkpoint`, and `model_card` cannot be registered. The registry validates the interface at registration time.

### Boundary 3: Reconstruction Engine → Evaluation Harness

**Contract:** The engine writes `ReconResult` objects to disk as `.pt` files (via `torch.save`). The evaluation harness loads these files; it never calls `model.forward()`. This boundary is the **separation between inference and evaluation** — they can run independently, on different machines, at different times.

**Enforcement:** `AggregateEvaluator` takes a `results_dir: Path` as input, not a model. If a model object is passed, it raises `TypeError`. This prevents accidental re-inference during report generation.

### Boundary 4: Evaluation Harness → Report Generator

**Contract:** The harness writes structured JSON and CSV files to `artifacts/metrics/`. The report generator reads only these files. It never imports from `eval/` at runtime — it is a pure file-to-file transformation.

**Enforcement:** `ReportGenerator` takes only `Path` arguments. It has no imports from `src/mri_ad/eval/`. This ensures `make report` has zero ML dependencies and runs in under 2 minutes with no GPU.

### Boundary 5: Classical Baseline ↔ DL Pipeline

**Contract:** The classical baseline and the DL pipeline share only the test split (via `SplitContract`) and the BraTS data (via `BraTSDataset`). They share no model objects, no feature representations, and no intermediate computations. The comparison is performed by `ReportGenerator` reading both pipelines' saved metric files.

**Enforcement:** `classical/` has no imports from `models/` or `recon/`. `eval/` has no imports from `classical/`. The comparison table is assembled from JSON files only.

---

## 5. Deployment Architecture

The system deploys to exactly two environments. There is no staging environment, no container, and no cloud service.

### Environment A: College GPU Cluster

```
┌─────────────────────────────────────────────────────────┐
│                   GPU Cluster Node                       │
│                                                         │
│  OS: Linux (SLURM-managed or interactive)               │
│  GPU: NVIDIA (CUDA 11.8+); minimum 16 GB VRAM           │
│       (UNETR at batch_size=2 on (1,16,128,128) ≈ 8 GB) │
│  Python: 3.11 via conda or venv                         │
│                                                         │
│  Paths (environment-specific, Hydra override):          │
│    data.root: /scratch/<user>/mri-data/                 │
│    checkpoints.root: /scratch/<user>/checkpoints/       │
│    artifacts.root: /scratch/<user>/artifacts/           │
│                                                         │
│  Supported targets:                                     │
│    make slice   ✓                                       │
│    make train   ✓  (manual only, SLURM job or srun)     │
│    make eval    ✓  (full test split)                    │
│    make report  ✓  (no GPU needed)                      │
│    make test    ✓                                       │
└─────────────────────────────────────────────────────────┘
```

**Cluster-specific config override** (`configs/experiment/cluster.yaml`):
```yaml
data:
  root: /scratch/${oc.env:USER}/mri-data
checkpoints:
  root: /scratch/${oc.env:USER}/checkpoints
artifacts:
  root: /scratch/${oc.env:USER}/artifacts
training:
  batch_size: 2
  num_workers: 8
```

Invocation: `make train HYDRA_OVERRIDES="+experiment=cluster"`

### Environment B: Developer Laptop

```
┌─────────────────────────────────────────────────────────┐
│                   Developer Laptop                       │
│                                                         │
│  OS: macOS or Linux                                     │
│  GPU: Optional (MPS on Apple Silicon, or CPU fallback)  │
│  Python: 3.11 via conda or venv                         │
│                                                         │
│  Paths (environment-specific, Hydra override):          │
│    data.root: ~/data/mri/                               │
│    checkpoints.root: ~/checkpoints/mri/                 │
│    artifacts.root: ./artifacts/                         │
│                                                         │
│  Supported targets:                                     │
│    make slice   ✓  (CPU or MPS)                         │
│    make train   ✗  (not recommended; too slow)          │
│    make eval    ✓  (reduced subset via config flag)     │
│    make report  ✓  (no GPU needed)                      │
│    make test    ✓  (CPU, synthetic tensors)             │
└─────────────────────────────────────────────────────────┘
```

**Laptop-specific config override** (`configs/experiment/laptop.yaml`):
```yaml
data:
  root: ~/data/mri
  eval_subset: 20          # evaluate on 20 volumes for local verification
checkpoints:
  root: ~/checkpoints/mri
artifacts:
  root: ./artifacts
training:
  batch_size: 1
  num_workers: 2
```

### Installation (Both Environments)

```bash
git clone https://github.com/<user>/mri-anomaly-detection.git
cd mri-anomaly-detection
pip install -e ".[dev]"          # installs all deps from pyproject.toml
# Place data at configured path, place checkpoints at configured path
make test                        # verify installation
make slice                       # verify end-to-end pipeline
```

No Docker, no Kubernetes, no cloud provider. The `pyproject.toml` is the sole dependency specification.

### Artifact Persistence

```
artifacts/                        # committed to git if small; gitignored if large
  runs/
    <timestamp>-<model>-<loss>/
      run_meta.json               # git SHA, config, seed, timing
      metrics/
        per_volume.csv
        aggregate.json
  results/
    recon/                        # saved ReconResult .pt files (gitignored)
    figures/                      # PNG plots (committed)
    tables/                       # CSV + LaTeX tables (committed)
  demo/
    demo_video.mp4                # committed as GitHub release asset
```

Files in `artifacts/results/recon/` (the `.pt` files) are gitignored due to size. Figures, tables, and the demo video are committed or released. The `run_meta.json` files are always committed — they are the reproducibility audit trail.

---

## 6. Scaling Strategy

This is a single-researcher project with fixed compute. "Scaling" means handling the actual data volumes efficiently, not horizontal scaling.

### Data Scale

- **OpenBHB training set:** ~3,000 healthy volumes (prior work used the validation subset, ~300 volumes, due to compute). The data layer uses `monai.data.CacheDataset` with `cache_rate=1.0` on the cluster (fits in RAM after preprocessing) and `cache_rate=0.0` on the laptop (stream from disk).
- **BraTS test set:** ~100 volumes used for evaluation. Fits entirely in GPU memory for batch evaluation.
- **Volume shape:** `(1, 16, 128, 128)` = 262,144 voxels per volume. At float32, one volume is 1 MB. A batch of 2 is 2 MB — negligible.

### Compute Scale

- **Inference:** Single-volume inference (forward pass + residual computation) targets < 5 seconds on GPU (NFR-9). UNETR at `(1, 16, 128, 128)` is well within this on any modern GPU.
- **Full eval:** 100 volumes × < 5 seconds = < 9 minutes inference + metric computation. Well within