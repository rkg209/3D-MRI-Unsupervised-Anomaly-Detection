```sql
-- =============================================================================
-- database-schema.sql
-- 3D MRI Unsupervised Anomaly Detection — Research Artifact Store
--
-- Purpose:
--   Relational schema for persisting run metadata, per-volume metrics,
--   aggregate metrics, split contracts, model registry entries, and
--   classical-baseline results produced by the mri_ad pipeline.
--
--   This schema replaces / complements the ad-hoc JSON/CSV artifact files
--   described in the architecture document. The flat-file artifacts remain
--   the primary interchange format between pipeline stages; this database
--   is the queryable audit trail and reporting backend.
--
-- Engine assumptions:
--   PostgreSQL 15+. JSONB, gen_random_uuid(), and CHECK constraints are used
--   freely. To port to SQLite, replace JSONB → TEXT, UUID → TEXT,
--   TIMESTAMPTZ → TEXT, and drop the GIN indexes.
--
-- Normalization target: 3NF throughout. Denormalized summary columns are
--   added only where query performance on the reporting path demands it,
--   and are marked with a comment explaining the trade-off.
--
-- =============================================================================
-- MIGRATION STRUCTURE (intended workflow)
-- -----------------------------------------------------------------------------
-- Migrations live in migrations/ at the repo root, managed by a tool such as
-- Flyway or golang-migrate. Naming convention:
--
--   migrations/
--     V001__initial_schema.sql          ← this file, split into one migration
--     V002__add_ssim_to_volume_metrics.sql
--     V003__classical_baseline_tables.sql
--     ...
--
-- Rules:
--   1. Every ALTER TABLE, CREATE INDEX, or DROP is a new versioned migration.
--   2. Migrations are append-only; no migration file is ever edited after merge.
--   3. The schema_migrations table (created by the migration tool) tracks
--      applied versions. Do not create it manually.
--   4. Down-migrations are provided as V<n>__<desc>__down.sql siblings.
--   5. The CI pipeline runs all pending migrations against a fresh PostgreSQL
--      instance before any test suite execution.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 0. EXTENSIONS
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- GIN on scalar columns


-- ---------------------------------------------------------------------------
-- 1. ENUMERATIONS
-- ---------------------------------------------------------------------------

-- The four operational modes that can produce a run.
CREATE TYPE run_mode AS ENUM (
    'slice',
    'eval',
    'train',
    'report',
    'classical',
    'demo'
);

-- Lifecycle state of a run within a single process invocation.
CREATE TYPE run_status AS ENUM (
    'started',
    'completed',
    'failed'
);

-- Which dataset a volume belongs to.
CREATE TYPE dataset_name AS ENUM (
    'openbhb',
    'brats'
);

-- Which split a volume was assigned to by SplitContract.
CREATE TYPE split_name AS ENUM (
    'train',
    'val',
    'test'
);

-- Thresholding strategy used to produce an anomaly mask.
CREATE TYPE threshold_strategy AS ENUM (
    'fixed_percentile',
    'otsu',
    'adaptive'
);

-- Granularity at which classical-baseline labels are defined.
CREATE TYPE classification_granularity AS ENUM (
    'slice_level',
    'volume_level'
);


-- ---------------------------------------------------------------------------
-- 2. SPLIT CONTRACTS
-- ---------------------------------------------------------------------------
-- Represents one serialized SplitContract. Each contract is identified by
-- its seed and a content hash of the resulting index lists, ensuring that
-- two contracts produced with the same seed but different data are distinct.

CREATE TABLE split_contracts (
    split_contract_id   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    seed                INTEGER         NOT NULL,
    -- SHA-256 of the canonical JSON representation of {train, val, test} index lists.
    content_hash        CHAR(64)        NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    -- Serialized index lists for full reproducibility audit.
    train_indices       JSONB           NOT NULL,  -- array of integers
    val_indices         JSONB           NOT NULL,
    test_indices        JSONB           NOT NULL,
    -- Human-readable counts (denormalized for fast reporting queries).
    n_train             INTEGER         NOT NULL CHECK (n_train >= 0),
    n_val               INTEGER         NOT NULL CHECK (n_val >= 0),
    n_test              INTEGER         NOT NULL CHECK (n_test >= 0),

    CONSTRAINT uq_split_contract_hash UNIQUE (content_hash)
);

COMMENT ON TABLE split_contracts IS
    'Immutable record of each SplitContract produced by data/split.py. '
    'The content_hash enforces that the same logical split is never registered twice.';

CREATE INDEX idx_split_contracts_seed ON split_contracts (seed);


-- ---------------------------------------------------------------------------
-- 3. MODEL REGISTRY
-- ---------------------------------------------------------------------------
-- One row per registered model variant. Mirrors ModelRegistry / ModelCard.

CREATE TABLE model_registry (
    model_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name          VARCHAR(128)    NOT NULL,
    -- Semantic version or git-tag of the model config at registration time.
    model_version       VARCHAR(64)     NOT NULL,
    architecture        VARCHAR(64)     NOT NULL,   -- 'unet', 'attention_unet', 'unetr'
    param_count         BIGINT          CHECK (param_count > 0),
    -- Shape stored as a JSONB array, e.g. [1, 16, 128, 128].
    input_shape         JSONB           NOT NULL,
    output_shape        JSONB           NOT NULL,
    training_dataset    VARCHAR(128),               -- e.g. 'openbhb_validation_subset'
    -- Full ModelCard fields serialized for forward-compatibility.
    model_card          JSONB           NOT NULL DEFAULT '{}',
    -- Path to the canonical checkpoint file (relative to checkpoints.root).
    checkpoint_path     TEXT,
    -- SHA-256 of the checkpoint file bytes at registration time.
    checkpoint_hash     CHAR(64),
    registered_at       TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_model_name_version UNIQUE (model_name, model_version)
);

COMMENT ON TABLE model_registry IS
    'Mirrors the in-process ModelRegistry singleton. One row per model variant. '
    'checkpoint_hash allows detection of checkpoint file tampering between runs.';

CREATE INDEX idx_model_registry_name ON model_registry (model_name);


-- ---------------------------------------------------------------------------
-- 4. RUNS
-- ---------------------------------------------------------------------------
-- One row per process invocation (one Makefile target execution).
-- Corresponds to one run_meta.json file written by RunLogger.

CREATE TABLE runs (
    run_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Matches the <timestamp>-<model>-<loss> directory name in artifacts/runs/.
    run_dir_name        VARCHAR(256)    NOT NULL,
    mode                run_mode        NOT NULL,
    status              run_status      NOT NULL DEFAULT 'started',
    -- Git SHA of the repo at invocation time.
    git_sha             CHAR(40)        NOT NULL,
    seed                INTEGER         NOT NULL,
    hostname            VARCHAR(256),
    started_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    ended_at            TIMESTAMPTZ,
    -- Wall-clock duration in seconds (denormalized; derivable from timestamps).
    duration_seconds    NUMERIC(10, 3)  CHECK (duration_seconds >= 0),
    -- Full resolved Hydra config snapshot.
    resolved_config     JSONB           NOT NULL DEFAULT '{}',
    -- Any error message if status = 'failed'.
    error_message       TEXT,

    -- FK to the model used in this run (NULL for report/demo runs).
    model_id            UUID            REFERENCES model_registry (model_id)
                                            ON DELETE SET NULL,
    -- FK to the split contract used (NULL for train/report runs).
    split_contract_id   UUID            REFERENCES split_contracts (split_contract_id)
                                            ON DELETE SET NULL,

    CONSTRAINT uq_run_dir_name UNIQUE (run_dir_name),
    CONSTRAINT chk_run_ended_after_started
        CHECK (ended_at IS NULL OR ended_at >= started_at)
);

COMMENT ON TABLE runs IS
    'One row per pipeline invocation. The audit trail equivalent of run_meta.json. '
    'status transitions: started → completed | failed.';

CREATE INDEX idx_runs_mode        ON runs (mode);
CREATE INDEX idx_runs_status      ON runs (status);
CREATE INDEX idx_runs_started_at  ON runs (started_at DESC);
CREATE INDEX idx_runs_model_id    ON runs (model_id);
CREATE INDEX idx_runs_git_sha     ON runs (git_sha);


-- ---------------------------------------------------------------------------
-- 5. VOLUMES
-- ---------------------------------------------------------------------------
-- One row per unique MRI volume file across both datasets.
-- Volume identity is the content hash of the NIfTI file, not the filename,
-- so renamed or moved files are still recognized as the same volume.

CREATE TABLE volumes (
    volume_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    -- SHA-256 of the raw NIfTI file bytes.
    file_hash           CHAR(64)        NOT NULL,
    -- Original filename (basename only; no absolute path stored).
    original_filename   VARCHAR(512)    NOT NULL,
    dataset             dataset_name    NOT NULL,
    -- True if this volume has a corresponding BraTS segmentation label.
    has_label           BOOLEAN         NOT NULL DEFAULT FALSE,
    -- Voxel shape after preprocessing: always (1, 16, 128, 128) post-pipeline,
    -- but we store the raw shape for provenance.
    raw_shape           JSONB,          -- e.g. [1, 155, 240, 240]
    registered_at       TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_volume_file_hash UNIQUE (file_hash)
);

COMMENT ON TABLE volumes IS
    'Registry of every MRI volume file encountered by the pipeline. '
    'file_hash is the stable identity; filenames are treated as mutable metadata.';

CREATE INDEX idx_volumes_dataset  ON volumes (dataset);
CREATE INDEX idx_volumes_has_label ON volumes (has_label);


-- ---------------------------------------------------------------------------
-- 6. SPLIT ASSIGNMENTS
-- ---------------------------------------------------------------------------
-- Many-to-many: one volume can appear in multiple split contracts
-- (e.g., if the seed changes), but within one contract each volume
-- appears in exactly one split.

CREATE TABLE split_assignments (
    split_assignment_id UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    split_contract_id   UUID            NOT NULL
                            REFERENCES split_contracts (split_contract_id)
                            ON DELETE CASCADE,
    volume_id           UUID            NOT NULL
                            REFERENCES volumes (volume_id)
                            ON DELETE CASCADE,
    split               split_name      NOT NULL,

    CONSTRAINT uq_split_assignment UNIQUE (split_contract_id, volume_id)
);

COMMENT ON TABLE split_assignments IS
    'Resolves the many-to-many relationship between split contracts and volumes. '
    'Enforces that each volume appears in exactly one split per contract.';

CREATE INDEX idx_split_assignments_contract ON split_assignments (split_contract_id);
CREATE INDEX idx_split_assignments_volume   ON split_assignments (volume_id);
CREATE INDEX idx_split_assignments_split    ON split_assignments (split);


-- ---------------------------------------------------------------------------
-- 7. RECONSTRUCTION RESULTS
-- ---------------------------------------------------------------------------
-- One row per (run, volume) pair where inference was performed.
-- Tensor data is NOT stored in the database; only metadata and scalar
-- summaries are stored. The .pt file path is the pointer to the binary.

CREATE TABLE reconstruction_results (
    recon_result_id     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID            NOT NULL
                            REFERENCES runs (run_id)
                            ON DELETE CASCADE,
    volume_id           UUID            NOT NULL
                            REFERENCES volumes (volume_id)
                            ON DELETE RESTRICT,
    -- Relative path to the .pt file under artifacts/results/recon/.
    artifact_path       TEXT            NOT NULL,
    -- SHA-256 of the .pt file for integrity verification.
    artifact_hash       CHAR(64),
    threshold_strategy  threshold_strategy  NOT NULL,
    -- The percentile value used (relevant only for fixed_percentile strategy).
    threshold_percentile NUMERIC(5, 2)  CHECK (
                            threshold_percentile IS NULL
                            OR (threshold_percentile > 0 AND threshold_percentile < 100)
                        ),
    -- Scalar summaries of the residual tensor (not the mask).
    residual_min        REAL,
    residual_max        REAL,
    residual_mean       REAL,
    residual_std        REAL,
    -- Fraction of voxels set to 1 in the anomaly mask.
    mask_positive_rate  REAL            CHECK (
                            mask_positive_rate IS NULL
                            OR (mask_positive_rate >= 0 AND mask_positive_rate <= 1)
                        ),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_recon_result_run_volume UNIQUE (run_id, volume_id)
);

COMMENT ON TABLE reconstruction_results IS
    'Metadata for each ReconResult produced by ReconstructionEngine. '
    'Binary tensor data lives in the .pt file referenced by artifact_path; '
    'this table stores only scalar summaries and provenance.';

CREATE INDEX idx_recon_results_run_id    ON reconstruction_results (run_id);
CREATE INDEX idx_recon_results_volume_id ON reconstruction_results (volume_id);


-- ---------------------------------------------------------------------------
-- 8. VOLUME METRICS
-- ---------------------------------------------------------------------------
-- One row per (reconstruction_result, label) evaluation.
-- Corresponds to VolumeMetrics produced by VolumeEvaluator.

CREATE TABLE volume_metrics (
    volume_metric_id    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    recon_result_id     UUID            NOT NULL
                            REFERENCES reconstruction_results (recon_result_id)
                            ON DELETE CASCADE,
    run_id              UUID            NOT NULL
                            REFERENCES runs (run_id)
                            ON DELETE CASCADE,
    volume_id           UUID            NOT NULL
                            REFERENCES volumes (volume_id)
                            ON DELETE RESTRICT,
    -- Detection metrics (primary — against BraTS ground-truth mask).
    dice                REAL            NOT NULL CHECK (dice >= 0 AND dice <= 1),
    iou                 REAL            NOT NULL CHECK (iou  >= 0 AND iou  <= 1),
    -- Reconstruction fidelity metrics (secondary — explain failure modes only).
    psnr                REAL            CHECK (psnr >= 0),
    ssim                REAL            CHECK (ssim >= -1 AND ssim <= 1),
    computed_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_volume_metric_recon UNIQUE (recon_result_id)
);

COMMENT ON TABLE volume_metrics IS
    'Per-volume evaluation results from VolumeEvaluator / MetricsComputer. '
    'dice and iou are the primary metrics; psnr and ssim are diagnostic only. '
    'run_id and volume_id are denormalized from recon_result_id for query convenience.';

CREATE INDEX idx_volume_metrics_run_id    ON volume_metrics (run_id);
CREATE INDEX idx_volume_metrics_volume_id ON volume_metrics (volume_id);
CREATE INDEX idx_volume_metrics_dice      ON volume_metrics (dice);
CREATE INDEX idx_volume_metrics_iou       ON volume_metrics (iou);


-- ---------------------------------------------------------------------------
-- 9. AGGREGATE METRICS
-- ---------------------------------------------------------------------------
-- One row per run summarizing all per-volume metrics for that run.
-- Corresponds to AggregateMetrics produced by AggregateEvaluator.
-- These values are derivable from volume_metrics but are stored here
-- to make the reporting path a single-row lookup per run.

CREATE TABLE aggregate_metrics (
    aggregate_metric_id UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID            NOT NULL
                            REFERENCES runs (run_id)
                            ON DELETE CASCADE,
    -- Number of volumes evaluated.
    n_volumes           INTEGER         NOT NULL CHECK (n_volumes > 0),
    -- Detection metrics.
    dice_mean           REAL            NOT NULL CHECK (dice_mean >= 0 AND dice_mean <= 1),
    dice_std            REAL            NOT NULL CHECK (dice_std  >= 0),
    iou_mean            REAL            NOT NULL CHECK (iou_mean  >= 0 AND iou_mean  <= 1),
    iou_std             REAL            NOT NULL CHECK (iou_std   >= 0),
    -- Reconstruction fidelity metrics.
    psnr_mean           REAL,
    psnr_std            REAL            CHECK (psnr_std IS NULL OR psnr_std >= 0),
    ssim_mean           REAL,
    ssim_std            REAL            CHECK (ssim_std IS NULL OR ssim_std >= 0),
    computed_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_aggregate_metrics_run UNIQUE (run_id)
);

COMMENT ON TABLE aggregate_metrics IS
    'Split-level summary metrics per run. Denormalized from volume_metrics for '
    'fast reporting queries. Values must be consistent with the corresponding '
    'volume_metrics rows; a CHECK CONSTRAINT cannot enforce cross-table consistency, '
    'so the application layer (AggregateEvaluator) is responsible for correctness.';

CREATE INDEX idx_aggregate_metrics_dice_mean ON aggregate_metrics (dice_mean DESC);
CREATE INDEX idx_aggregate_metrics_iou_mean  ON aggregate_metrics (iou_mean  DESC);


-- ---------------------------------------------------------------------------
-- 10. TRAINING CHECKPOINTS
-- ---------------------------------------------------------------------------
-- One row per checkpoint file saved during or after a training run.
-- A training run may save multiple intermediate checkpoints (one per epoch)
-- plus a final checkpoint.

CREATE TABLE training_checkpoints (
    checkpoint_id       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID            NOT NULL
                            REFERENCES runs (run_id)
                            ON DELETE CASCADE,
    model_id            UUID            NOT NULL
                            REFERENCES model_registry (model_id)
                            ON DELETE RESTRICT,
    -- Epoch number at which this checkpoint was saved (NULL = final).
    epoch               INTEGER         CHECK (epoch IS NULL OR epoch >= 0),
    is_final            BOOLEAN         NOT NULL DEFAULT FALSE,
    -- Relative path under artifacts/checkpoints/.
    artifact_path       TEXT            NOT NULL,
    artifact_hash       CHAR(64),
    -- Training loss at this checkpoint.
    train_loss          REAL,
    val_loss            REAL,
    saved_at            TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_checkpoint_run_epoch UNIQUE (run_id, epoch),
    CONSTRAINT chk_final_epoch_null
        CHECK (NOT (is_final = TRUE AND epoch IS NOT NULL))
);

COMMENT ON TABLE training_checkpoints IS
    'Records every checkpoint file emitted by run_train.py. '
    'is_final = TRUE marks the checkpoint consumed by subsequent make eval runs. '
    'The UNIQUE constraint on (run_id, epoch) prevents duplicate epoch saves.';

CREATE INDEX idx_training_checkpoints_run_id   ON training_checkpoints (run_id);
CREATE INDEX idx_training_checkpoints_model_id ON training_checkpoints (model_id);
CREATE INDEX idx_training_checkpoints_is_final ON training_checkpoints (is_final)
    WHERE is_final = TRUE;


-- ---------------------------------------------------------------------------
-- 11. TRAINING LOG ENTRIES
-- ---------------------------------------------------------------------------
-- One row per step logged by RunLogger.log_checkpoint().
-- Corresponds to lines in training_log.jsonl.

CREATE TABLE training_log_entries (
    log_entry_id        UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID            NOT NULL
                            REFERENCES runs (run_id)
                            ON DELETE CASCADE,
    global_step         INTEGER         NOT NULL CHECK (global_step >= 0),
    epoch               INTEGER         NOT NULL CHECK (epoch >= 0),
    batch_index         INTEGER         NOT NULL CHECK (batch_index >= 0),
    train_loss          REAL            NOT NULL,
    -- Component losses (NULL if not separately tracked).
    mse_loss            REAL,
    ssim_loss           REAL,
    learning_rate       REAL,
    logged_at           TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_training_log_run_step UNIQUE (run_id, global_step)
);

COMMENT ON TABLE training_log_entries IS
    'Fine-grained training loss log. Mirrors training_log.jsonl. '
    'Intended for loss-curve plotting in make report. '
    'High write volume during training; consider partitioning by run_id '
    'if training runs become very long.';

CREATE INDEX idx_training_log_run_id ON training_log_entries (run_id, global_step);


-- ---------------------------------------------------------------------------
-- 12. CLASSICAL BASELINE RESULTS
-- ---------------------------------------------------------------------------
-- One row per cross-validation fold result, plus one summary row per run.

CREATE TABLE classical_cv_folds (
    fold_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID            NOT NULL
                            REFERENCES runs (run_id)
                            ON DELETE CASCADE,
    fold_index          INTEGER         NOT NULL CHECK (fold_index >= 0),
    n_folds             INTEGER         NOT NULL CHECK (n_folds > 0),
    granularity         classification_granularity  NOT NULL,
    -- Classifier type used in this fold.
    classifier_type     VARCHAR(64)     NOT NULL,   -- 'xgboost', 'lightgbm'
    -- Counts for this fold's test partition.
    n_samples_train     INTEGER         NOT NULL CHECK (n_samples_train > 0),
    n_samples_test      INTEGER         NOT NULL CHECK (n_samples_test  > 0),
    n_positive_test     INTEGER         NOT NULL CHECK (n_positive_test >= 0),
    -- Fold-level metrics.
    roc_auc             REAL            NOT NULL CHECK (roc_auc >= 0 AND roc_auc <= 1),
    pr_auc              REAL            NOT NULL CHECK (pr_auc  >= 0 AND pr_auc  <= 1),
    computed_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_classical_fold_run_index UNIQUE (run_id, fold_index)
);

COMMENT ON TABLE classical_cv_folds IS
    'Per-fold results from ClassicalBaseline.fit() / evaluate(). '
    'Granularity is always slice_level per the architecture decision in classical/.';

CREATE INDEX idx_classical_cv_folds_run_id ON classical_cv_folds (run_id);


CREATE TABLE classical_aggregate_metrics (
    classical_agg_id    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID            NOT NULL
                            REFERENCES runs (run_id)
                            ON DELETE CASCADE,
    granularity         classification_granularity  NOT NULL,
    classifier_type     VARCHAR(64)     NOT NULL,
    n_folds             INTEGER         NOT NULL CHECK (n_folds > 0),
    roc_auc_mean        REAL            NOT NULL CHECK (roc_auc_mean >= 0 AND roc_auc_mean <= 1),
    roc_auc_std         REAL            NOT NULL CHECK (roc_auc_std  >= 0),
    pr_auc_mean         REAL            NOT NULL CHECK (pr_auc_mean  >= 0 AND pr_auc_mean  <= 1),
    pr_auc_std          REAL            NOT NULL CHECK (pr_auc_std   >= 0),
    -- Path to the serialized model artifact.
    model_artifact_path TEXT,
    model_artifact_hash CHAR(64),
    computed_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_classical_agg_run UNIQUE (run_id)
);

COMMENT ON TABLE classical_aggregate_metrics IS
    'Summary of ClassicalMetrics produced by ClassicalBaseline.evaluate(). '
    'Mirrors artifacts/classical/metrics.json. '
    'Joined with aggregate_metrics on run_id for the comparison table in make report.';

CREATE INDEX idx_classical_agg_roc_auc ON classical_aggregate_metrics (roc_auc_mean DESC);


-- ---------------------------------------------------------------------------
-- 13. FEATURE EXTRACTION METADATA
-- ---------------------------------------------------------------------------
-- Records which feature set was used for a classical run.
-- Feature values themselves are NOT stored (too large; live in memory only).

CREATE TABLE feature_extraction_configs (
    feature_config_id   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID            NOT NULL
                            REFERENCES runs (run_id)
                            ON DELETE CASCADE,
    -- Total number of features extracted per slice.
    n_features          INTEGER         NOT NULL CHECK (n_features > 0),
    -- Feature group flags.
    includes_intensity  BOOLEAN         NOT NULL DEFAULT TRUE,
    includes_glcm       BOOLEAN         NOT NULL DEFAULT TRUE,
    includes_gradient   BOOLEAN         NOT NULL DEFAULT TRUE,
    includes_radiomic   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Full feature name list for reproducibility.
    feature_names       JSONB           NOT NULL DEFAULT '[]',
    -- Config snapshot for FeatureExtractor.
    extractor_config    JSONB           NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_feature_config_run UNIQUE (run_id)
);

COMMENT ON TABLE feature_extraction_configs IS
    'Records the feature engineering configuration used by FeatureExtractor '
    'for each classical baseline run. Enables reproducibility audits when '
    'feature sets change between experiments.';


-- ---------------------------------------------------------------------------
-- 14. REPORT ARTIFACTS
-- ---------------------------------------------------------------------------
-- Tracks every file written by ReportGenerator and DemoExporter.

CREATE TABLE report_artifacts (
    report_artifact_id  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID            NOT NULL
                            REFERENCES runs (run_id)
                            ON DELETE CASCADE,
    -- Relative path under artifacts/.
    artifact_path       TEXT            NOT NULL,
    artifact_type       VARCHAR(64)     NOT NULL,  -- 'figure', 'table_csv', 'table_tex',
                                                   -- 'demo_video', 'readme_scorecard'
    -- SHA-256 of the file at write time.
    artifact_hash       CHAR(64),
    -- Which run(s) contributed data to this artifact (for multi-run comparison tables).
    source_run_ids      JSONB           NOT NULL DEFAULT '[]',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now()
);

COMMENT ON TABLE report_artifacts IS
    'Provenance record for every file written by ReportGenerator / DemoExporter. '
    'source_run_ids captures which eval/classical runs fed into a comparison table, '
    'enabling traceability from a figure back to the exact runs that produced it.';

CREATE INDEX idx_report_artifacts_run_id ON report_artifacts (run_id);
CREATE INDEX idx_report_artifacts_type   ON report_artifacts (artifact_type);


-- ---------------------------------------------------------------------------
-- 15. VIEWS — Reporting Convenience
-- ---------------------------------------------------------------------------

-- 15.1 Comparison table view: one row per eval run with model and aggregate metrics.
--      Mirrors the architecture × loss matrix produced by generate_comparison_table().
CREATE VIEW v_model_comparison AS
SELECT
    r.run_id,
    r.run_dir_name,
    r.git_sha,
    r.seed,
    r.started_at,
    r.ended_at,
    r.duration_seconds,
    mr.model_name,
    mr.architecture,
    mr.model_version,
    mr.checkpoint_path,
    am.n_volumes,
    am.dice_mean,
    am.dice_std,
    am.iou_mean,
    am.iou_std,
    am.psnr_mean,
    am.ssim_mean,
    -- Resolved config fields surfaced for the comparison table.
    (r.resolved_config -> 'training' ->> 'loss')        AS loss_function,
    (r.resolved_config -> 'threshold' ->> 'strategy')   AS threshold_strategy,
    (r.resolved_config -> 'threshold' ->> 'percentile') AS threshold_percentile
FROM runs r
JOIN model_registry     mr ON mr.model_id = r.model_id
JOIN aggregate_metrics  am ON am.run_id   = r.run_id
WHERE r.mode   = 'eval'
  AND r.status = 'completed';

COMMENT ON VIEW v_model_comparison IS
    'Flat view for the architecture × loss comparison table. '
    'Consumed by ReportGenerator.generate_comparison_table() via direct SQL.';


-- 15.2 Per-volume detail view: joins volume_metrics with volume and run metadata.
CREATE VIEW v_volume_metrics_detail AS
SELECT
    vm.volume_metric_id,
    r.run_id,
    r.run_dir_name,
    mr.model_name,
    mr.architecture,
    v.volume_id,
    v.original_filename,
    v.dataset,
    sa.split,
    vm.dice,
    vm.iou,
    vm.psnr,
    vm.ssim,
    rr.threshold_strategy,
    rr.threshold_percentile,
    rr.mask_positive_rate,
    vm.computed_at
FROM volume_metrics     vm
JOIN reconstruction_results rr ON rr.recon_result_id = vm.recon_result_id
JOIN runs               r  ON r.run_id    = vm.run_id
JOIN model_registry     mr ON mr.model_id = r.model_id
JOIN volumes            v  ON v.volume_id = vm.volume_id
LEFT JOIN split_assignments sa
    ON sa.volume_id = vm.volume_id
   AND sa.split_contract_id = r.split_contract_id;

COMMENT ON VIEW v_volume_metrics_detail IS
    'Full per-volume metrics with run and model context. '
    'Used for Dice distribution histograms and per-volume CSV export.';


-- 15.3 Classical vs DL comparison view.
CREATE VIEW v_classical_vs_dl AS
SELECT
    'dl'                        AS pipeline,
    r.run_id,
    mr.model_name               AS model_or_classifier,
    mr.architecture             AS architecture_or_type,
    am.dice_mean                AS primary_metric_mean,
    am.dice_std                 AS primary_metric_std,
    'dice'                      AS primary_metric_name,
    am.n_volumes                AS n_samples,
    r.started_at
FROM runs r
JOIN model_registry    mr ON mr.model_id = r.model_id
JOIN aggregate_metrics am ON am.run_id   = r.run_id
WHERE r.mode = 'eval' AND r.status = 'completed'

UNION ALL

SELECT
    'classical'                 AS pipeline,
    r.run_id,
    cam.classifier_type         AS model_or_classifier,
    CAST(cam.granularity AS TEXT) AS architecture_or_type,
    cam.roc_auc_mean            AS primary_metric_mean,
    cam.roc_auc_std             AS primary_metric_std,
    'roc_auc'                   AS primary_metric_name,
    cam.n_folds                 AS n_samples,
    r.started_at