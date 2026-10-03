# database-design.md

## 3D MRI Unsupervised Anomaly Detection — Database Design

---

## 1. Preamble: Why This Document Exists Despite Having No Database

This system has no database server, no ORM, and no SQL. The architecture document states this explicitly: "There is no server, no API, no database." Nevertheless, a database design document is warranted because the system does have **persistent structured state** — it is simply stored as files on disk rather than in a relational engine. Every concept that would normally live in a database table (entities, relationships, ownership, integrity constraints, access patterns) exists here in a different physical form.

This document makes those implicit structures explicit. It answers:

- What are the logical entities, and what are their attributes?
- What are the relationships between entities, and what are their cardinalities?
- Which module owns each entity (i.e., which module is the sole writer)?
- How are referential integrity and uniqueness enforced without a database engine?
- How would this design map to a relational schema if the system were ever promoted to a multi-user service?

The design is presented in two registers: the **as-built file-based design** (what actually exists) and the **relational projection** (what it would look like as normalized SQL, for clarity and future-proofing).

---

## 2. Entity Inventory

The system has eight logical entities. Each entity is a distinct, identifiable thing with a stable identity and a set of attributes that are read or written by one or more modules.

| # | Entity | Physical Form | Owner Module | Identity Key |
|---|---|---|---|---|
| E1 | `Run` | `artifacts/runs/<run_id>/run_meta.json` | `utils/` | `run_id` (timestamp + model + loss) |
| E2 | `Checkpoint` | `artifacts/checkpoints/<name>.pt` | `models/` | `checkpoint_name` |
| E3 | `SplitContract` | `artifacts/split_contract.json` | `data/` | `(dataset_name, seed)` |
| E4 | `Volume` | NIfTI file on disk; identity tracked by index | `data/` | `(dataset_name, volume_index)` |
| E5 | `ReconResult` | `artifacts/results/recon/<volume_id>.pt` | `recon/` | `(run_id, volume_id)` |
| E6 | `VolumeMetrics` | Row in `artifacts/runs/<run_id>/metrics/per_volume.csv` | `eval/` | `(run_id, volume_id)` |
| E7 | `AggregateMetrics` | `artifacts/runs/<run_id>/metrics/aggregate.json` | `eval/` | `run_id` |
| E8 | `ClassicalMetrics` | `artifacts/classical/metrics.json` | `classical/` | `(seed, k_folds, classifier)` |

---

## 3. Entity Definitions

### E1 — `Run`

A `Run` is one complete invocation of any entry-point script. It is the audit trail entity. Every other entity that is produced during a process invocation is associated with exactly one `Run`.

**Attributes:**

| Attribute | Type | Nullable | Description |
|---|---|---|---|
| `run_id` | `string` | No | `<ISO8601_timestamp>-<model_name>-<loss_name>`; globally unique by construction |
| `run_type` | `enum` | No | One of: `slice`, `eval`, `train`, `report`, `classical`, `demo` |
| `git_sha` | `string(40)` | No | Full SHA of HEAD at invocation time; `"dirty"` suffix if working tree modified |
| `seed` | `integer` | No | Global random seed used for this run |
| `hostname` | `string` | No | Machine hostname; distinguishes cluster vs. laptop runs |
| `start_time` | `datetime` | No | ISO 8601 UTC |
| `end_time` | `datetime` | Yes | Null until `log_run_end()` is called; null indicates a crashed run |
| `model_name` | `string` | Yes | Null for `report` and `classical` runs |
| `loss_name` | `string` | Yes | Null for non-training runs |
| `resolved_config` | `json` | No | Full Hydra-resolved config snapshot; enables exact reproduction |
| `final_metrics` | `json` | Yes | Summary metrics dict appended by `log_run_end()`; null for crashed runs |

**Physical form:** `artifacts/runs/<run_id>/run_meta.json`

**Integrity rules:**
- `run_id` is constructed to be unique: the timestamp component has millisecond precision, and concurrent invocations on the same machine are not supported (single-engineer system).
- A `run_meta.json` with a null `end_time` is a sentinel for a crashed or interrupted run. `ReportGenerator` skips such runs when assembling comparison tables and emits a warning.
- `git_sha` is recorded but not enforced as a foreign key. The system does not prevent running with a dirty working tree; it only records the fact.

---

### E2 — `Checkpoint`

A `Checkpoint` is a saved model weight file. It is the coupling point between the `train` workflow and all subsequent workflows.

**Attributes:**

| Attribute | Type | Nullable | Description |
|---|---|---|---|
| `checkpoint_name` | `string` | No | Filename without extension, e.g., `unetr_mse_ssim`, `unetr_synth` |
| `model_architecture` | `string` | No | One of: `unet`, `attention_unet`, `unetr` |
| `loss_combination` | `string` | No | e.g., `mse_ssim`, `mse_only`, `ssim_only` |
| `produced_by_run_id` | `string` | Yes | `run_id` of the training run; null for externally sourced checkpoints (the pre-trained UNETR weight) |
| `file_path` | `path` | No | Absolute path on the current machine; environment-specific |
| `file_size_bytes` | `integer` | No | Recorded at save time; used to detect truncated files |
| `param_count` | `integer` | No | Number of trainable parameters; recorded in `ModelCard` |
| `training_data` | `string` | No | Dataset name used for training, e.g., `openbhb_validation_subset` |

**Physical form:** The `.pt` file itself carries weights only. The metadata attributes above are recorded in the `run_meta.json` of the producing run (for trained checkpoints) or in `configs/models/<name>.yaml` (for externally sourced checkpoints).

**Integrity rules:**
- `ModelRegistry` validates that a checkpoint file exists and is loadable before returning a model object. A missing or corrupt file raises `CheckpointError` immediately — it does not propagate as a null.
- `file_size_bytes` is checked at load time against the recorded value. A mismatch raises `CheckpointError` with a message distinguishing "file not found" from "file truncated."
- There is no foreign key enforcement between `Checkpoint.produced_by_run_id` and `Run.run_id`. The relationship is advisory. If the producing run's `run_meta.json` is deleted, the checkpoint remains usable.

---

### E3 — `SplitContract`

A `SplitContract` is the deterministic assignment of volume indices to train, validation, and test splits. It is the referential integrity anchor for all comparisons: every model and the classical baseline must use the same test split.

**Attributes:**

| Attribute | Type | Nullable | Description |
|---|---|---|---|
| `dataset_name` | `string` | No | e.g., `brats_2021` |
| `seed` | `integer` | No | The seed used to generate the split |
| `total_volumes` | `integer` | No | Total number of volumes in the dataset at split time |
| `train_indices` | `integer[]` | No | Sorted list of volume indices assigned to train |
| `val_indices` | `integer[]` | No | Sorted list of volume indices assigned to validation |
| `test_indices` | `integer[]` | No | Sorted list of volume indices assigned to test |
| `created_at` | `datetime` | No | ISO 8601 UTC; set once, never updated |
| `schema_version` | `integer` | No | Incremented if the split format changes; current: `1` |

**Physical form:** `artifacts/split_contract.json`

**Integrity rules:**
- `SplitContract` is **write-once**. After the file is created, any attempt to create a new split with a different seed or different `total_volumes` raises `SplitContractViolationError` unless the caller passes an explicit `--override-split` flag. This is the strongest integrity guarantee in the system.
- The sets `train_indices`, `val_indices`, `test_indices` must be disjoint and their union must equal `range(total_volumes)`. This is validated at creation time and at load time.
- `SplitContract` is committed to git. Any change to the file is visible in the diff and must be accompanied by a commit message explaining why the split was regenerated.
- Both `BraTSDataset` and `ClassicalBaseline` consume the same `SplitContract`. This is the sole mechanism ensuring the classical baseline and DL pipeline are evaluated on identical volumes.

---

### E4 — `Volume`

A `Volume` is one preprocessed MRI volume. It is not stored as a discrete artifact — it lives in the source NIfTI files and is materialized as a tensor by the data layer on demand. However, it has a stable logical identity that is referenced by downstream entities.

**Attributes:**

| Attribute | Type | Nullable | Description |
|---|---|---|---|
| `dataset_name` | `string` | No | `openbhb` or `brats_2021` |
| `volume_index` | `integer` | No | Position in the dataset's file list; stable given a fixed data directory |
| `split_assignment` | `enum` | No | `train`, `val`, or `test`; derived from `SplitContract` |
| `source_path` | `path` | No | Absolute path to the source NIfTI file |
| `has_label` | `boolean` | No | True for BraTS volumes; false for OpenBHB |
| `preprocessed_shape` | `tuple` | No | Always `(1, 16, 128, 128)` after `TransformPipeline`; recorded here for documentation |

**Physical form:** No discrete artifact. Identity is `(dataset_name, volume_index)`. The `source_path` is the physical anchor.

**Integrity rules:**
- `DataValidator` enforces that every materialized volume has shape `(B, 1, 16, 128, 128)`, dtype `float32`, and values in `[0, 1]`. A violation raises `DescriptiveValidationError` before the tensor reaches any consumer.
- `volume_index` is stable only if the data directory contents do not change. The `SplitContract` records `total_volumes`; if the data directory has a different count at load time, `SplitContract` raises an error rather than silently using a misaligned split.
- No patient identifiers, file paths, or metadata escape the `data/` layer. Downstream entities reference volumes only by `(dataset_name, volume_index)`.

---

### E5 — `ReconResult`

A `ReconResult` is the output of one forward pass through one model on one volume. It is the central artifact of the eval pipeline.

**Attributes:**

| Attribute | Type | Nullable | Description |
|---|---|---|---|
| `run_id` | `string` | No | FK → `Run.run_id` |
| `volume_id` | `string` | No | `<dataset_name>_<volume_index>`; FK → `Volume` identity |
| `model_name` | `string` | No | FK → `Checkpoint.model_architecture` |
| `checkpoint_name` | `string` | No | FK → `Checkpoint.checkpoint_name` |
| `threshold_strategy` | `string` | No | e.g., `fixed_percentile_95`, `otsu` |
| `original` | `Tensor(1,16,128,128)` | No | The preprocessed input volume |
| `reconstruction` | `Tensor(1,16,128,128)` | No | Model output |
| `residual` | `Tensor(1,16,128,128)` | No | `abs(original - reconstruction)` |
| `anomaly_mask` | `Tensor(1,16,128,128)` | No | Binary mask after thresholding |

**Physical form:** `artifacts/results/recon/<volume_id>.pt` — a `torch.save` dict containing all tensor attributes plus a metadata sub-dict with the non-tensor attributes.

**Integrity rules:**
- `load_recon_result()` validates tensor shapes immediately after loading. A shape mismatch raises `ReconResultCorruptError` with the expected and actual shapes.
- The metadata sub-dict inside the `.pt` file records `run_id`, `volume_id`, `model_name`, `checkpoint_name`, and `threshold_strategy`. These are written by `save_recon_result()` and validated by `load_recon_result()`.
- `AggregateEvaluator` matches each `.pt` file to its corresponding label tensor using `volume_id`. If no label is found for a given `volume_id`, the volume is skipped and a warning is logged — it is not a fatal error, because partial eval results are valid.
- The `.pt` files are gitignored due to size. Their existence is implied by the `run_meta.json` of the producing run, which records the count of volumes processed.

**Note on the current-run limitation:** The current design writes all `ReconResult` files for a given run into a single flat directory `artifacts/results/recon/`. This means that running `make eval` twice with different models will overwrite the first run's results if the `volume_id` values collide. The correct fix — namespacing by `run_id` — is noted in Section 7 (Schema Evolution).

---

### E6 — `VolumeMetrics`

A `VolumeMetrics` record holds the evaluation metrics for one volume in one run.

**Attributes:**

| Attribute | Type | Nullable | Description |
|---|---|---|---|
| `run_id` | `string` | No | FK → `Run.run_id` |
| `volume_id` | `string` | No | FK → `Volume` identity |
| `dice` | `float` | No | Dice coefficient; formula defined in `MetricsComputer` |
| `iou` | `float` | No | Intersection over Union; formula defined in `MetricsComputer` |
| `psnr` | `float` | No | Peak Signal-to-Noise Ratio; range `[0, ∞)` |
| `ssim` | `float` | No | Structural Similarity Index; range `[0, 1]` |

**Physical form:** One row per volume in `artifacts/runs/<run_id>/metrics/per_volume.csv`. The CSV has a header row; columns are in the order listed above.

**Integrity rules:**
- `dice` and `iou` are in `[0, 1]`. Values outside this range indicate a bug in `MetricsComputer` and are caught by a post-computation assertion.
- `psnr` is unbounded above but is expected to be in `[0, 60]` for typical MRI reconstructions. Values above 60 are logged as warnings (may indicate a trivial reconstruction).
- Each `(run_id, volume_id)` pair is unique within a CSV file. Duplicate rows indicate a bug in `AggregateEvaluator` and are caught by a post-write deduplication check.

---

### E7 — `AggregateMetrics`

An `AggregateMetrics` record summarizes one complete eval run across all test volumes.

**Attributes:**

| Attribute | Type | Nullable | Description |
|---|---|---|---|
| `run_id` | `string` | No | FK → `Run.run_id`; PK for this entity |
| `n_volumes` | `integer` | No | Number of volumes evaluated |
| `mean_dice` | `float` | No | Mean Dice across all volumes |
| `std_dice` | `float` | No | Standard deviation of Dice |
| `mean_iou` | `float` | No | Mean IoU |
| `std_iou` | `float` | No | Standard deviation of IoU |
| `mean_psnr` | `float` | No | Mean PSNR |
| `std_psnr` | `float` | No | Standard deviation of PSNR |
| `mean_ssim` | `float` | No | Mean SSIM |
| `std_ssim` | `float` | No | Standard deviation of SSIM |
| `threshold_strategy` | `string` | No | Threshold strategy used; must match all constituent `ReconResult` records |
| `model_name` | `string` | No | Model architecture name |
| `checkpoint_name` | `string` | No | Checkpoint used |

**Physical form:** `artifacts/runs/<run_id>/metrics/aggregate.json`

**Integrity rules:**
- `n_volumes` must equal the number of rows in the corresponding `per_volume.csv`. A mismatch raises `MetricsIntegrityError`.
- `mean_dice` must equal `mean(per_volume.csv.dice)` to within floating-point tolerance (`1e-6`). This is checked by `AggregateEvaluator` before writing.
- `ReportGenerator` reads `aggregate.json` files from multiple runs to build the comparison table. It validates that all runs being compared used the same `SplitContract` (by checking the `seed` field in the corresponding `run_meta.json`). Runs with different seeds are excluded from the comparison table with a warning.

---

### E8 — `ClassicalMetrics`

A `ClassicalMetrics` record holds the cross-validation results for the classical ML baseline.

**Attributes:**

| Attribute | Type | Nullable | Description |
|---|---|---|---|
| `seed` | `integer` | No | Global seed; part of composite PK |
| `k_folds` | `integer` | No | Number of CV folds; part of composite PK |
| `classifier` | `string` | No | `xgboost` or `lightgbm`; part of composite PK |
| `granularity` | `string` | No | Always `slice-level`; recorded for documentation |
| `n_samples` | `integer` | No | Total number of slices used (n_volumes × 16) |
| `n_positive` | `integer` | No | Number of slices with tumor present |
| `roc_auc_mean` | `float` | No | Mean ROC-AUC across folds |
| `roc_auc_std` | `float` | No | Standard deviation of ROC-AUC |
| `pr_auc_mean` | `float` | No | Mean PR-AUC across folds |
| `pr_auc_std` | `float` | No | Standard deviation of PR-AUC |
| `per_fold_roc_auc` | `float[]` | No | Individual fold scores; enables recomputing mean/std |

**Physical form:** `artifacts/classical/metrics.json`

**Integrity rules:**
- `roc_auc_mean` must equal `mean(per_fold_roc_auc)` to within `1e-6`. Checked at write time.
- `n_positive / n_samples` is the class balance. If this ratio is below 0.05 or above 0.95, a warning is logged — extreme imbalance would make ROC-AUC misleading.
- `ClassicalMetrics` is compared to `AggregateMetrics` only by `ReportGenerator`, which reads both as JSON files. The comparison is narrative (different granularities, different metric spaces) and is documented as such in the generated report.

---

## 4. Relationships

### 4.1 Relationship Map

```
Run ──────────────────────────────────────────────────────────────────┐
 │                                                                    │
 │ produces (0..1)                                                    │
 ▼                                                                    │
Checkpoint ◄──────────────────────────────────────────────────────── │
 │                                                                    │
 │ used_by (0..*)                                                     │
 ▼                                                                    │
Run (eval run) ──── processes ──► Volume (many-to-many via ReconResult)
 │                                    │
 │ produces (1..*)                    │ has_label (0..1)
 ▼                                    │
ReconResult ◄────────────────────────┘
 │
 │ evaluated_by (1..1)
 ▼
VolumeMetrics
 │
 │ aggregated_into (many-to-one)
 ▼
AggregateMetrics
 │
 │ belongs_to (1..1)
 ▼
Run

SplitContract ──── governs ──► Volume (partitions all volumes into splits)
     │
     │ referenced_by
     ├──► Run (all eval and classical runs)
     └──► ClassicalMetrics (via run seed)
```

### 4.2 Relationship Definitions

**`Run` → `Checkpoint` (produces, 0..1):**
A training run produces exactly one checkpoint. An eval, slice, report, or classical run produces no checkpoint. A checkpoint may exist without a producing run record (externally sourced pre-trained weights).

**`Checkpoint` → `Run` (used_by, 0..*):**
A checkpoint may be used by zero or more eval/slice runs. The relationship is recorded in `Run.resolved_config.checkpoints.path`, not as a discrete join table.

**`Run` × `Volume` → `ReconResult` (many-to-many, resolved):**
One eval run processes many volumes. One volume may be processed by many runs (different models, different checkpoints). The `ReconResult` is the resolution entity, with `(run_id, volume_id)` as its composite key.

**`ReconResult` → `VolumeMetrics` (one-to-one):**
Each `ReconResult` produces exactly one `VolumeMetrics` record. The relationship is enforced by `VolumeEvaluator`, which takes a `ReconResult` and a label tensor and returns a `VolumeMetrics`.

**`VolumeMetrics` → `AggregateMetrics` (many-to-one):**
All `VolumeMetrics` records for a given `run_id` aggregate into one `AggregateMetrics` record. The cardinality is `n_test_volumes`-to-one.

**`SplitContract` → `Volume` (governs, one-to-many):**
One `SplitContract` partitions all volumes in a dataset into train/val/test. Every `Volume` belongs to exactly one split under a given `SplitContract`. The `SplitContract` is the referential integrity anchor for all comparisons.

**`SplitContract` → `Run` (referenced_by, one-to-many):**
All eval and classical runs reference the same `SplitContract`. The `seed` field in `Run.resolved_config` is the foreign key. `ReportGenerator` validates this when assembling comparison tables.

---

## 5. Data Ownership

Data ownership defines which module is the **sole writer** of each entity. All other modules are readers. Write conflicts are prevented by design (single-process, single-engineer system), but ownership is documented to make the invariants explicit and to guide any future refactoring.

| Entity | Owner (Sole Writer) | Readers |
|---|---|---|
| `Run` | `utils/logging.py` (`RunLogger`) | `eval/report.py` (`ReportGenerator`) |
| `Checkpoint` | `models/` (save) or external (pre-trained) | `models/registry.py`, `recon/engine.py` |
| `SplitContract` | `data/split.py` (`SplitContract`) | `data/loaders.py`, `classical/baseline.py` |
| `Volume` (identity) | `data/` (no write; identity is derived from filesystem) | All modules via `data/loaders.py` |
| `ReconResult` | `recon/io.py` (`save_recon_result`) | `eval/evaluator.py`, `viz/viewer.py` |
| `VolumeMetrics` | `eval/evaluator.py` (`AggregateEvaluator`) | `eval/report.py` |
| `AggregateMetrics` | `eval/evaluator.py` (`AggregateEvaluator`) | `eval/report.py` |
| `ClassicalMetrics` | `classical/baseline.py` (`ClassicalBaseline`) | `eval/report.py` |

**Ownership rules:**

1. **No module writes an entity it does not own.** `ReportGenerator` reads `aggregate.json` but never writes it. `recon/engine.py` calls `save_recon_result()` but never writes `VolumeMetrics`.

2. **Readers receive entities through defined interfaces, not by reading the owner's internal state.** `eval/evaluator.py` reads `ReconResult` via `load_recon_result()`, not by importing from `recon/engine.py`. `ReportGenerator` reads `aggregate.json` from disk, not by calling `AggregateEvaluator` methods.

3. **The `data/` layer owns the `Volume` identity but does not own any downstream artifact.** Once a tensor leaves the data layer, it is owned by the receiving module. The data layer never reads `ReconResult`, `VolumeMetrics`, or any other downstream entity.

4. **`utils/` owns `Run` records exclusively.** No other module writes to `run_meta.json`. Modules that want to record information in the run log call `RunLogger` methods — they do not write to the file directly.

---

## 6. Multi-Tenancy

**This system is single-tenant by design.** There is one researcher, one machine (or one cluster node) at a time, one active experiment at a time. Multi-tenancy — the ability to serve multiple isolated users or experiments simultaneously — is not a requirement and is not implemented.

However, the design does handle two scenarios that resemble multi-tenancy concerns:

### 6.1 Multiple Environments (Cluster vs. Laptop)

The system runs in two environments with different filesystem paths. This is handled by Hydra config overrides (`+experiment=cluster` or `+experiment=laptop`), not by any runtime isolation mechanism. The two environments share no state — they have separate `artifacts/` directories and separate data roots. There is no synchronization between them.

**Implication for the schema:** All `path` attributes in entity definitions are environment-specific. A `run_meta.json` written on the cluster contains cluster-absolute paths. If that file is copied to the laptop, the paths are invalid. The `run_id` and all non-path attributes remain valid. This is acceptable: `run_meta.json` files are committed to git for their metadata, not for their paths.

### 6.2 Multiple Concurrent Runs (Not Supported)

The system does not support concurrent invocations. Two simultaneous `make eval` calls would write to the same `artifacts/results/recon/` directory and produce undefined behavior. This is not a concern for a single-engineer system, but it is documented here as a known limitation.

**The `run_id` timestamp is the only isolation mechanism.** Each run writes its metrics to a timestamped directory (`artifacts/runs/<run_id>/`), so metric files from different runs do not collide. However, `ReconResult` `.pt` files are written to a shared flat directory, which is the collision point identified in Section 3 (E5).

### 6.3 If Multi-Tenancy Were Required

If this system were promoted to a shared research platform serving multiple users or concurrent experiments, the following changes would be required:

1. **Namespace `ReconResult` files by `run_id`:** `artifacts/results/recon/<run_id>/<volume_id>.pt` instead of the current flat layout.
2. **Add a `user_id` attribute to `Run`:** Enables per-user filtering in `ReportGenerator`.
3. **Replace file-based locking with a database:** SQLite would be sufficient for a small team; PostgreSQL for a larger deployment.
4. **Promote `SplitContract` to a versioned, named entity:** Multiple users might want different splits for different experiments. The current write-once, single-file design would need to become a keyed lookup.
5. **Add a `project_id` or `experiment_id` grouping entity:** To isolate one researcher's runs from another's in the comparison tables.

None of these changes are planned. They are documented here to make the implicit single-tenancy assumption explicit and to provide a migration path if requirements change.

---

## 7. Relational Projection

The following is the normalized relational schema that the file-based design implicitly implements. It is provided for clarity and as a reference for any future migration to a database-backed system. The schema is in third normal form (3NF).

```sql
-- E1: Run
CREATE TABLE run (
    run_id              TEXT        PRIMARY KEY,
    run_type            TEXT        NOT NULL CHECK (run_type IN
                                        ('slice','eval','train','report','classical','demo')),
    git_sha             TEXT        NOT NULL,
    seed                INTEGER     NOT NULL,
    hostname            TEXT        NOT NULL,
    start_time          TIMESTAMPTZ NOT NULL,
    end_time            TIMESTAMPTZ,               -- NULL = crashed/in-progress
    model_name          TEXT,                      -- NULL for report/classical runs
    loss_name           TEXT,                      -- NULL for non-training runs
    resolved_config     JSONB       NOT NULL,
    final_metrics       JSONB                      -- NULL = crashed/in-progress
);

-- E2: Checkpoint
CREATE TABLE checkpoint (
    checkpoint_name     TEXT        PRIMARY KEY,
    model_architecture  TEXT        NOT NULL CHECK (model_architecture IN
                                        ('unet','attention_unet','unetr')),
    loss_combination    TEXT        NOT NULL,
    produced_by_run_id  TEXT        REFERENCES run(run_id),  -- NULL = external
    file_path           TEXT        NOT NULL,
    file_size_bytes     BIGINT      NOT NULL,
    param_count         INTEGER     NOT NULL,
    training_data       TEXT        NOT NULL
);

-- E3: SplitContract
CREATE TABLE split_contract (
    dataset_name        TEXT        NOT NULL,
    seed                INTEGER     NOT NULL,
    total_volumes       INTEGER     NOT NULL,
    train_indices       INTEGER[]   NOT NULL,
    val_indices         INTEGER[]   NOT NULL,
    test_indices        INTEGER[]   NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL,
    schema_version      INTEGER     NOT NULL DEFAULT 1,
    PRIMARY KEY (dataset_name, seed)
);

-- E4: Volume (logical entity; no physical table in current system)
CREATE TABLE volume (
    dataset_name        TEXT        NOT NULL,
    volume_index        INTEGER     NOT NULL,
    split_assignment    TEXT        NOT NULL CHECK (split_assignment IN
                                        ('train','val','test')),
    source_path         TEXT        NOT NULL,
    has_label           BOOLEAN     NOT NULL,
    PRIMARY KEY (dataset_name, volume_index),
    FOREIGN KEY (dataset_name, split_assignment)
        REFERENCES split_contract(dataset_name, seed)
        -- Note: seed would need to be carried here in a full normalization;
        -- simplified for readability
);

-- E5: ReconResult (tensor data stored externally; metadata here)
CREATE TABLE recon_result (
    run_id              TEXT        NOT NULL REFERENCES run(run_id),
    volume_id           TEXT        NOT NULL,      -- '<dataset_name>_<volume_index>'
    model_name          TEXT        NOT NULL,
    checkpoint_name     TEXT        NOT NULL REFERENCES checkpoint(checkpoint_name),
    threshold_strategy  TEXT        NOT NULL,
    artifact_path       TEXT        NOT NULL,      -- path to .pt