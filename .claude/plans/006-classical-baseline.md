# Plan · Spec 006 — Classical-ML baseline

**Spec:** [`specs/006-classical-baseline.md`](../../specs/006-classical-baseline.md) (approved)
**Depends on:** 001 (data layer), 004 (eval harness — for artifact conventions only, never imported)
**Fresh compute:** none. CPU only. Feature extraction is ~10–20 CPU-minutes and is **user-invoked**.

---

## Context

The prior body of work is deep-learning-only. Spec 006 adds the honest control: engineered
radiomic/texture features into a gradient-boosting classifier. If hand-crafted features plus
XGBoost match the reconstruction pipeline, that is a finding worth reporting (Spec 007 owns the
narrative). Today `src/mri_ad/classical/` is a stub `__init__.py`, `configs/classical/` is empty,
and the `make classical` target already points at a `scripts/run_classical.py` that does not exist.

Three decisions were locked before planning:

| Decision | Choice | Why |
|---|---|---|
| CV pool | `contract.brats["test"]` only | Same subject population the DL pipeline is scored on → Spec 007 comparison is apples-to-apples (acceptance 5). Caveat to document: classical is *supervised* on held-in folds; the DL models never see labels. |
| Classifier | XGBoost, `tree_method=hist`, pinned `n_jobs` | Listed first in the spec, already a hard dep, deterministic when pinned. |
| Features | Cached to `.npz` with a provenance header | Extraction is expensive; a header mismatch is a **loud failure**, never silent reuse. |

Granularity is **slice-level** and is not a config switch: one 2D axial 128×128 slice = one sample,
label = 1 iff the binarized seg slice holds a tumor voxel. It does **not** localize, so ROC-AUC here
is not comparable to voxel-level Dice. That caveat is stamped into every artifact (acceptance 6).

---

## Files

### New

| Path | Purpose |
|---|---|
| `configs/classical/default.yaml` | every hyperparameter, feature knob, cache/report path |
| `src/mri_ad/classical/runlength.py` | GLRLM — skimage ships none, hand-rolled, own tests |
| `src/mri_ad/classical/features.py` | `FeatureConfig`, `FeatureExtractor`, `feature_names` |
| `src/mri_ad/classical/dataset.py` | `SliceDataset`, `slice_labels`, `build_slice_dataset` |
| `src/mri_ad/classical/cache.py` | `FeatureCacheHeader`, `FeatureCache` |
| `src/mri_ad/classical/metrics.py` | `GRANULARITY`, `FoldMetrics`, `ClassicalMetrics`, `aggregate_folds` |
| `src/mri_ad/classical/classifier.py` | `build_xgboost` — the **only** file importing `xgboost` |
| `src/mri_ad/classical/baseline.py` | `ClassicalBaseline.make_splits` / `.cross_validate` |
| `src/mri_ad/classical/report.py` | `ClassicalReportGenerator` (re-implemented; may not import `eval/`) |
| `scripts/run_classical.py` | the `make classical` entry point |
| `tests/test_classical.py` | acceptance 1–6 |
| `tests/test_classical_cache.py` | cache provenance / loud-failure |
| `tests/test_classical_boundary.py` | acceptance 7 |

### Modified

- `src/mri_ad/exceptions.py` — add `ClassicalError`, `FeatureCacheError(ClassicalError)`.
- `src/mri_ad/data/split.py` — add `SplitContract.content_hash()` and
  `resolve_contract(cfg, *, build_if_missing)` (see R4).
- `src/mri_ad/data/datasets.py` + `data/__init__.py` — add public
  `load_preprocessed_volume(cfg, volume_id)` returning the *whole* volume, not chunks.
- `configs/config.yaml` — `defaults:` gains `- classical: default`.
- `configs/experiment/laptop.yaml` — `classical: {subject_limit: 10}` for smoke runs.
- `tests/test_scaffold.py` — new exceptions in the hierarchy test; new config in `test_configs_parse`.
- `README.md` (granularity + PyRadiomics substitution paragraph), `progress_report.md` (append).
- `Makefile` needs **no** change — `classical: $(PY) scripts/run_classical.py $(HYDRA_OVERRIDES)` exists.

---

## Feature set — 44 features/slice, scikit-image only

`feature_names(config)` is the single source of truth for column order; it is stamped into the
cache header, the JSON, and the importance CSV.

- **First-order (15):** mean, std, min, max, range, skew, kurtosis, p05/25/50/75/95, IQR, energy,
  entropy. `scipy.stats` skew/kurtosis with an explicit `std == 0 → 0.0` guard.
- **GLCM (12):** `graycomatrix(distances=[1], angles=[0,45,90,135]°, levels=32, symmetric=True,
  normed=True)` on the slice quantized to uint8. Six props (contrast, dissimilarity, homogeneity,
  energy, correlation, ASM) × {mean, std over angles}.
- **Gradient (5):** `skimage.filters.sobel` → mean, std, max, p90, energy.
- **GLRLM (11):** SRE, LRE, GLN, RLN, RP, LGRE, HGRE, SRLGE, SRHGE, LRLGE, LRHGE, averaged over
  4 directions. See `runlength.py` below.
- **Context (1):** `fg_fraction`. `include_slice_index` defaults **false** — z-position is
  anatomical prior, not texture, and inflates AUC for free.

**PyRadiomics substitution** (permitted by the spec) is recorded in `features.py`'s docstring, in
the config, and in the JSON as `features.library: "scikit-image"` / `pyradiomics_used: false` /
`substitution_note`. The `[radiomics]` extra stays optional and unused.

---

## Interfaces

### `classical/runlength.py` — highest-risk novel code, isolated on purpose
```python
def run_length_matrix(quantized, *, levels, angle_deg) -> np.ndarray
def glrlm_features(matrix) -> dict[str, float]          # 11 descriptors, every denominator guarded
def run_length_features(image, *, levels, angles_deg) -> dict[str, float]
```
Directions reduce to 1D lines (rows / columns / `np.diagonal` of the array and its `fliplr`); runs
come from `np.flatnonzero(np.diff(line))` — no Python loop over pixels. Galloway 1975 +
Chu/Dasarathy-Holder formulas cited in the module docstring so a reviewer can check them.

### `classical/features.py`
```python
@dataclass(frozen=True)
class FeatureConfig:
    ...                                   # every knob that changes a feature value
    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> FeatureConfig   # OmegaConf -> plain tuples
    def content_hash(self) -> str          # sha256 of canonical JSON; keys the cache

class FeatureExtractor:
    def extract(self, volume) -> np.ndarray          # (D,H,W) or (1,D,H,W) in [0,1] -> (D, F) float32
    def extract_slice(self, image) -> np.ndarray     # (F,)
    @property
    def sanitized_count(self) -> int                 # non-finite values replaced, REPORTED not hidden
```
GLCM `correlation` is NaN on a constant slice and skew/kurtosis are undefined there, so every value
passes an explicit non-finite → `0.0` sanitizer whose hit count lands in the JSON. A silent NaN
sweep is exactly what acceptance test 1 exists to catch.

### `data/datasets.py` (new public helper)
```python
def load_preprocessed_volume(cfg, volume_id) -> dict[str, Tensor]
    # {"image": (1,D,128,128) in [0,1], "label": (1,D,128,128) in {0,1}}
```
Runs the **same** `_LoadBraTSd` + `build_transforms(cfg, dataset="brats")` pipeline `BraTSDataset`
uses (trap #3), but returns the whole volume rather than depth-padded chunks — the classical
baseline must never see chunking's zero-padded tail slices, which would enter the pool as
fabricated negatives. No path or identifier survives into the returned dict.

### `classical/dataset.py`
```python
@dataclass(frozen=True)
class SliceDataset:
    X, y, groups, slice_index, feature_names        # groups[i] = BraTS subject folder name

def slice_labels(label_volume, *, min_tumor_voxels=1) -> np.ndarray   # (D,) uint8
def build_slice_dataset(cfg, subject_ids, extractor, *, progress=True)
    -> tuple[SliceDataset, dict[str, int]]
```
The seg is **already** binarized and nearest-resized by `build_transforms` — never re-binarize or
re-resize here (trap #1). Enforces `(H,W) == (cfg.shape.height, cfg.shape.width)` per volume. Slices
below `foreground_eps` are dropped when `drop_empty_slices` — pure-background slices are trivially
negative and would flatter both AUCs; the returned counter records every drop.

### `classical/metrics.py` — `ClassicalMetrics` lives *here*, not in `eval/` (acceptance 7)
```python
GRANULARITY = "slice-level"
GRANULARITY_NOTE = "...does NOT localize; ROC-AUC is NOT comparable to voxel-level Dice..."

@dataclass(frozen=True)
class FoldMetrics:
    fold, n_train, n_test, n_train_subjects, n_test_subjects, n_test_positive,
    positive_rate, roc_auc, pr_auc, pr_auc_prevalence_baseline,
    roc_auc_majority_baseline, scale_pos_weight

@dataclass(frozen=True)
class ClassicalMetrics:
    granularity, n_folds, n_samples, n_subjects, n_features, n_positive, positive_rate,
    roc_auc_mean/std, pr_auc_mean/std, pr_auc_prevalence_baseline_mean,
    roc_auc_majority_baseline, folds, feature_names

def binary_scores(y_true, y_score) -> tuple[float, float]
def aggregate_folds(...) -> ClassicalMetrics
```
PR-AUC is `average_precision_score` (step-wise), **not** trapezoid-under-PR, which is optimistically
biased — documented, not implicit. A single-class fold raises `ClassicalError` rather than
returning NaN. `aggregate_folds` uses `statistics.pstdev`, matching `eval/metrics.aggregate`'s
convention — reimplemented because `classical/` may not import `eval/`.

### `classical/classifier.py` / `classical/baseline.py`
```python
def build_xgboost(params, *, seed, scale_pos_weight)     # module-scope `import xgboost`, imported lazily

class ClassicalBaseline:
    def __init__(self, cfg, *, estimator_factory=None)   # injection = tests need no xgboost
    def make_splits(self, data) -> list[tuple[np.ndarray, np.ndarray]]
    def cross_validate(self, data) -> ClassicalMetrics
    @property
    def feature_importance(self) -> dict[str, float]     # mean gain across folds
```
`StratifiedGroupKFold(n_splits=cfg.classical.split.cv.n_splits, shuffle=True,
random_state=cfg.seed).split(X, y, groups=subject_ids)`. **Grouping by subject is not configurable**
— adjacent slices of one brain are near-duplicates and an ungrouped split inflates AUC to a fake
number. `make_splits` asserts train/test subject-set disjointness before returning.
`scale_pos_weight="auto"` is recomputed as `n_neg/n_pos` **within each training fold**; it never
sees the test fold, and no resampling is applied anywhere.

### `classical/report.py`
`ClassicalReportGenerator(metrics_dir, figures_dir)` mirrors `eval/report.py`'s discipline (plain
mappings in, lazy matplotlib) but is an independent implementation — the module docstring states the
duplication is intentional and names acceptance 7 as the cause. Writers:
`write_per_fold`, `write_metrics_json`, `write_summary_markdown`, `write_feature_importance`,
`plot_class_balance`, `plot_fold_auc`. Every one stamps `granularity`: a column on every CSV row, a
top-level JSON key plus `granularity_note`, the markdown H1 plus a bolded caveat, and a matplotlib
`suptitle`.

### `scripts/run_classical.py`
```
seed_everything(cfg.seed, cfg.deterministic)
with RunLogger(cfg) as run:
    contract  = resolve_contract(cfg, build_if_missing=cfg.classical.split.build_contract_if_missing)
    pool      = list(contract.brats["test"])       # acceptance 5: this list, nothing else
    if cfg.classical.subject_limit: pool = pool[:n]          # JSON records complete=False
    extractor = FeatureExtractor(FeatureConfig.from_cfg(cfg))
    header    = FeatureCacheHeader(...)            # split hash + seed + feature hash + names
    data      = FeatureCache(dir).load(header) or build_slice_dataset(...) -> save
    metrics   = ClassicalBaseline(cfg).cross_validate(data)
    ClassicalReportGenerator(...).write_* / plot_*
    run.record(roc_auc_mean=..., pr_auc_mean=..., positive_rate=..., granularity="slice-level")
```
Module-level imports restricted to hydra/omegaconf, `mri_ad.classical.*`, `mri_ad.data.split`,
`mri_ad.exceptions`, `mri_ad.utils.{run_logger,seed}` — never `models`/`recon`/`eval`.

---

## `configs/classical/default.yaml` (shape)

```yaml
granularity: slice-level          # not a switch — see the header comment (does not localize)
subject_limit: null

split:
  source: brats_test              # acceptance 5 — the identical SplitContract split
  build_contract_if_missing: true # nothing in the repo writes split_contract.json yet (R4)
  cv: {scheme: StratifiedGroupKFold, n_splits: 5, shuffle: true}   # random_state IS cfg.seed

features:
  library: skimage
  label_threshold_voxels: 1
  drop_empty_slices: true
  foreground_eps: 1.0e-6
  include_slice_index: false
  percentiles: [5, 25, 50, 75, 95]
  glcm: {levels: 32, distances: [1], angles_deg: [0,45,90,135], symmetric: true, normed: true,
         props: [contrast, dissimilarity, homogeneity, energy, correlation, ASM]}
  runlength: {levels: 16, angles_deg: [0, 45, 90, 135]}
  gradient: {percentile: 90}

cache: {enabled: true, refresh: false, dir: ${paths.artifact_root}/classical/features}

classifier:
  name: xgboost
  scale_pos_weight: auto          # n_neg/n_pos of the TRAIN fold only
  params: {n_estimators: 400, max_depth: 5, learning_rate: 0.05, subsample: 0.8,
           colsample_bytree: 0.8, min_child_weight: 5, reg_lambda: 1.0,
           tree_method: hist,     # deterministic; exact/approx are not pinned across builds
           n_jobs: 4,             # pinned: -1 makes tree construction machine-dependent
           eval_metric: logloss}

report: {metrics_dir: ${paths.artifact_root}/classical/metrics,
         figures_dir: ${paths.artifact_root}/classical/figures}
```
The file header comment carries the granularity caveat and the PyRadiomics substitution note in full.

---

## Artifacts

```
artifacts/classical/
  features/slice-level_<feathash12>_<splithash12>.npz   # X,y,groups,slice_index,feature_names,meta
  metrics/classical_metrics.json
  metrics/per_fold.csv              # granularity,fold,n_train,n_test,...,roc_auc,pr_auc,
                                    # pr_auc_prevalence_baseline,roc_auc_majority_baseline,scale_pos_weight
  metrics/feature_importance.csv    # granularity,rank,feature,mean_gain
  metrics/summary.md
  figures/{class_balance,fold_auc}.png
artifacts/runs/<stamp>/run_meta.json                     # RunLogger, unchanged
```

`classical_metrics.json` top-level keys: `spec`, `granularity`, `granularity_note`, `run_id`, `seed`,
`complete`, `subject_limit`, `split{source,split_seed,split_hash,n_subjects,cv{...}}`,
`features{n_features,names,config_hash,library,pyradiomics_used,substitution_note,
n_sanitized_values,cache_path}`, `counts{n_volumes,n_slices_seen,n_slices_dropped_empty,n_samples}`,
`class_balance{n_positive,n_negative,positive_rate,per_fold_positive_rate}`,
`headline{roc_auc_mean,roc_auc_std,pr_auc_mean,pr_auc_std}`,
`baselines{roc_auc_majority: 0.5, pr_auc_prevalence_mean, note}`,
`classifier{name,params,scale_pos_weight_per_fold}`, `folds[]`.

The `baselines.note` reads: *a majority-class classifier scores ROC-AUC 0.5 by construction; the
honest PR-AUC floor is the positive rate — read PR-AUC against that floor, never against 0.*

---

## Tests → acceptance mapping

All synthetic: numpy-RNG `(1, D, 32, 32)` volumes and a recording stub estimator. No real data, no
forward pass, no xgboost except two `importorskip` tests. (House rule + the "no heavy local runs"
constraint.)

| Acc. | Tests |
|---|---|
| **1** | `test_extract_returns_n_slices_by_n_features` · `test_extracted_matrix_is_finite_and_non_nan` · `test_constant_and_all_zero_slices_stay_finite` · `test_feature_names_are_unique_and_match_matrix_width` · `test_sanitized_count_is_reported_not_hidden` · GLRLM: `test_glrlm_of_a_uniform_image_is_one_long_run`, `test_run_length_matrix_row_sums_match_pixel_count`, `test_glrlm_features_of_an_empty_slice_are_zero_not_nan` |
| **2** | `test_cross_validate_returns_one_FoldMetrics_per_fold` · `test_roc_and_pr_auc_reported_as_mean_and_std` (std == `pstdev` of fold values) · `test_single_class_fold_raises_rather_than_returning_nan` |
| **3** | `test_no_subject_appears_in_both_train_and_test_of_any_fold` · `test_every_sample_is_tested_exactly_once` · `test_splits_are_identical_for_the_same_seed_and_differ_for_another` · `test_ungrouped_kfold_would_split_a_subject` (documents *why*, with near-duplicate synthetic slices) |
| **4** | `test_class_balance_fields_are_populated` · `test_pr_auc_prevalence_baseline_equals_positive_rate` · `test_roc_auc_majority_baseline_is_reported_as_half` · `test_scale_pos_weight_is_computed_from_the_train_fold_only` |
| **5** | `test_cv_pool_is_exactly_contract_brats_test` · `test_no_val_or_openbhb_subject_enters_the_pool` · `test_cache_header_records_the_split_hash_and_seed` |
| **6** | `test_granularity_is_slice_level_in_ClassicalMetrics` · `test_every_generated_artifact_records_granularity` (parametrized over produced files, so a new artifact type cannot skip it) |
| **7** | `tests/test_classical_boundary.py`: AST walk over `src/mri_ad/classical/**/*.py` (mirrors `tests/test_model_boundary.py`) · subprocess `sys.modules` leak check · `run_classical.py` module-level import check · `test_importing_classical_does_not_require_xgboost` |
| cache | `tests/test_classical_cache.py`: round-trip; `FeatureCacheError` naming the offending field for each of split-hash / feature-hash / granularity / feature-names / schema-version mismatch; absent cache → `None`; interrupted save leaves no partial `.npz` |

Note: `tests/test_model_boundary.py`'s `NO_THRESHOLD_DIRS` already scans `classical/` for
`residual <op> number` — no feature module may threshold a residual.

---

## Ordered steps

1. `exceptions.py`: `ClassicalError`, `FeatureCacheError`; update `test_scaffold.py`.
2. `configs/classical/default.yaml` + `configs/config.yaml` defaults + `test_configs_parse` param.
3. `classical/runlength.py` + tests. Pure numpy, no config — the riskiest new code, done first.
4. `classical/features.py` + acceptance-1 tests (`len(feature_names) == 44`; `content_hash` changes
   when any knob changes).
5. `data/split.py`: `content_hash` + `resolve_contract` + tests (build-if-missing, verify-on-mismatch).
6. `data/datasets.py`: `load_preprocessed_volume` + export.
7. `classical/dataset.py` + label / empty-slice tests (monkeypatch the loader — no I/O).
8. `classical/cache.py` + `tests/test_classical_cache.py`.
9. `classical/metrics.py` + acceptance-2/4 metric tests.
10. `classical/classifier.py`, `classical/baseline.py` + acceptance-2/3/4 CV tests (stub estimator).
11. `classical/report.py` + acceptance-6 artifact tests.
12. `scripts/run_classical.py`, `classical/__init__.py` exports, `tests/test_classical_boundary.py`;
    then `configs/experiment/laptop.yaml`, `README.md`, `progress_report.md`.

---

## Verification

- `make lint` — ruff format + check (line-length 100, `ANN`/`D` enforced on non-test code).
- `make test` — full pytest; every acceptance test above runs on synthetic tensors in seconds and
  requires neither `data/` nor `checkpoints/`.
- `python -c "from hydra import compose, initialize"` config-composition check is covered by
  `test_scaffold.py::test_configs_parse`.
- **Proposed, not run by the agent:** `make classical HYDRA_OVERRIDES="+experiment=laptop"` for a
  10-subject smoke run, then `make classical` for the real run. Both need `make check-data` to pass
  first. Verify afterwards that `artifacts/classical/metrics/classical_metrics.json` exists, its
  `split.split_hash` matches `artifacts/split_contract.json`, `complete: true`, and that
  `roc_auc_mean` is traceable to the `per_fold.csv` rows.

---

## Risks

**R1 · Determinism.** Three hazards. (a) XGBoost with `n_jobs=-1` or an unpinned `tree_method`
produces machine-dependent trees → `tree_method: hist` and a positive integer `n_jobs`, both
asserted by a test that reads the config. (b) `StratifiedGroupKFold` depends on `random_state`
*and* on the input order of `groups` → the pool comes from `contract.brats["test"]` in that exact
order, never `os.listdir` order. (c) A rebuilt contract could silently re-partition → see R4.

**R2 · Class imbalance.** Positive rate is likely 10–20%. Guarded three ways: PR-AUC is always
emitted next to its prevalence floor; `scale_pos_weight` is computed per training fold so test-fold
prevalence never leaks into training (asserted); a single-class fold raises rather than yielding
NaN. **No resampling anywhere** — SMOTE over near-duplicate adjacent slices would manufacture the
exact leak acceptance 3 exists to prevent.

**R3 · Cost.** ~295 subjects × ~155 slices ≈ 46k rows × 44 float32 ≈ 8 MB — the matrix is trivial.
Extraction is the cost: ~10–25 ms/slice → roughly 10–20 CPU-minutes plus ~295 nibabel loads. Peak
RSS is one volume (~40 MB) since each is released after extraction. Mitigated by the disk cache and
`subject_limit: 10` in the laptop experiment. **User-invoked only; never launched from an agent turn.**

**R4 · No `split_contract.json` exists on disk.** Nothing in the repo has ever written one;
`run_recon.py:47` and `run_sweep.py:57` both dead-end on it today. Spec 006 must not invent its own
pool. Resolution: `resolve_contract(cfg, build_if_missing=…)` builds from sorted directory listings
at `cfg.seed`, saves to the canonical `cfg.data.split.contract_path`, and thereafter loads **and
verifies** — a data directory that has gained or lost subjects is a loud
`SplitContractViolationError`, never a quiet re-partition. The contract's `content_hash` is embedded
in the feature cache and the metrics JSON. *This is a small, deliberate scope extension into Spec
001's module; `run_recon.py`/`run_sweep.py` can adopt it later without churn.*

**R5 · GLRLM has no library implementation.** skimage ships `graycomatrix` only; PyRadiomics would
provide GLRLM but is the fragile dep the spec permits substituting away from. Isolated in its own
module with its own correctness tests and cited formulas, so a disagreement with PyRadiomics is a
documented, testable delta rather than a black box.

**R6 · `xgboost`/`lightgbm` fail to import on this machine** (missing `libomp.dylib`). Every
classical module except `classifier.py` stays importable without it; `classifier.py` is imported
lazily from inside `ClassicalBaseline`, and a boundary test asserts `import mri_ad.classical`
succeeds with `xgboost` poisoned in `sys.modules`. User fix: `brew install libomp` (README line).

**R7 · Label erosion by the seg resize.** `build_transforms` nearest-resizes the seg 240→128; a
few-voxel tumor on one slice can vanish, turning a positive into a negative.
`label_threshold_voxels` is config-driven and `n_positive` is recorded, so the effect is visible; a
note goes in the summary markdown. Inherent to sharing the DL pipeline's preprocessing (trap #3) —
the right trade.

**R8 · Duplicated report code.** `classical/report.py` re-implements part of `eval/report.py`
because acceptance 7 forbids the import. Drift risk mitigated by a module docstring naming the cause
and by `aggregate_folds` documenting that it matches `eval/metrics.aggregate`'s `pstdev` convention.
