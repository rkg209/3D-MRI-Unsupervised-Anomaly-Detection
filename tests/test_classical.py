"""Spec 006 acceptance tests. All synthetic — no real data, no forward pass, no xgboost import
except where explicitly marked ``importorskip`` (R6: xgboost may fail to import on this machine).
"""

from __future__ import annotations

import numpy as np
import pytest

from mri_ad.classical.baseline import ClassicalBaseline
from mri_ad.classical.dataset import SliceDataset
from mri_ad.classical.features import FeatureConfig, FeatureExtractor, feature_names
from mri_ad.classical.metrics import GRANULARITY, FoldMetrics, aggregate_folds, binary_scores
from mri_ad.classical.report import ClassicalReportGenerator
from mri_ad.classical.runlength import (
    GLRLM_FEATURE_NAMES,
    glrlm_features,
    run_length_features,
    run_length_matrix,
)
from mri_ad.exceptions import ClassicalError

RNG = np.random.default_rng(0)


def _random_slice(size: int = 32) -> np.ndarray:
    return RNG.random((size, size)).astype(np.float32)


# ── Acceptance 1: GLRLM ──────────────────────────────────────────────────────────────────────
def test_glrlm_of_a_uniform_image_is_one_long_run() -> None:
    image = np.full((8, 8), 0.5, dtype=np.float32)
    matrix = run_length_matrix(image.astype(np.int64), levels=4, angle_deg=0)
    # a constant row of length 8 quantizes to gray level 0 (see _quantize), one run of length 8
    # per row, 8 rows -> matrix[0, 7] == 8, everything else zero.
    assert matrix[0, -1] == 8
    assert matrix.sum() == 8


def test_run_length_matrix_row_sums_match_pixel_count() -> None:
    image = RNG.integers(0, 4, size=(10, 12)).astype(np.int64)
    for angle in (0, 45, 90, 135):
        matrix = run_length_matrix(image, levels=4, angle_deg=angle)
        run_len = np.arange(1, matrix.shape[1] + 1)
        total_pixels_from_matrix = float((matrix * run_len).sum())
        assert total_pixels_from_matrix == pytest.approx(image.size)


def test_glrlm_features_of_an_empty_slice_are_zero_not_nan() -> None:
    matrix = np.zeros((0, 0))
    features = glrlm_features(matrix)
    assert set(features) == set(GLRLM_FEATURE_NAMES)
    assert all(np.isfinite(v) and v == 0.0 for v in features.values())


def test_run_length_features_averages_over_angles_and_is_finite() -> None:
    image = RNG.random((16, 16)).astype(np.float32)
    features = run_length_features(image, levels=16, angles_deg=[0, 45, 90, 135])
    assert set(features) == set(GLRLM_FEATURE_NAMES)
    assert all(np.isfinite(v) for v in features.values())


# ── Acceptance 1: FeatureExtractor ───────────────────────────────────────────────────────────
def test_extract_returns_n_slices_by_n_features() -> None:
    config = FeatureConfig()
    volume = RNG.random((1, 6, 32, 32)).astype(np.float32)
    extractor = FeatureExtractor(config)
    matrix = extractor.extract(volume)
    assert matrix.shape == (6, len(feature_names(config)))


def test_feature_count_is_44_by_default() -> None:
    assert len(feature_names(FeatureConfig())) == 44


def test_include_slice_index_adds_a_45th_feature() -> None:
    config = FeatureConfig(include_slice_index=True)
    assert len(feature_names(config)) == 45
    assert "slice_index" in feature_names(config)


def test_extracted_matrix_is_finite_and_non_nan() -> None:
    extractor = FeatureExtractor(FeatureConfig())
    volume = RNG.random((4, 32, 32)).astype(np.float32)
    matrix = extractor.extract(volume)
    assert np.all(np.isfinite(matrix))


def test_constant_and_all_zero_slices_stay_finite() -> None:
    extractor = FeatureExtractor(FeatureConfig())
    constant = np.full((2, 32, 32), 0.3, dtype=np.float32)
    zeros = np.zeros((2, 32, 32), dtype=np.float32)
    assert np.all(np.isfinite(extractor.extract(constant)))
    assert np.all(np.isfinite(extractor.extract(zeros)))


def test_feature_names_are_unique_and_match_matrix_width() -> None:
    config = FeatureConfig()
    names = feature_names(config)
    assert len(names) == len(set(names))
    matrix = FeatureExtractor(config).extract(RNG.random((3, 32, 32)).astype(np.float32))
    assert matrix.shape[1] == len(names)


def test_sanitized_count_is_reported_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a non-finite value through the pipeline and confirm it is counted, not swept away."""
    import mri_ad.classical.features as features_mod

    real_glcm = features_mod._glcm

    def _poisoned_glcm(image: np.ndarray, config: FeatureConfig) -> dict:
        values = real_glcm(image, config)
        values["glcm_correlation_mean"] = float("nan")
        return values

    monkeypatch.setattr(features_mod, "_glcm", _poisoned_glcm)
    extractor = FeatureExtractor(FeatureConfig())
    extractor.extract(RNG.random((3, 16, 16)).astype(np.float32))
    assert extractor.sanitized_count == 3  # one poisoned value per slice, 3 slices


def test_content_hash_changes_when_a_knob_changes() -> None:
    base = FeatureConfig()
    changed = FeatureConfig(glcm_levels=64)
    assert base.content_hash() != changed.content_hash()


def test_content_hash_is_stable_for_identical_config() -> None:
    assert FeatureConfig().content_hash() == FeatureConfig().content_hash()


# ── classical/dataset.py: slice_labels + build_slice_dataset (monkeypatched loader, no I/O) ───
def _volume(depth: int, *, tumor_slices: set[int], size: int = 16) -> dict:
    image = RNG.random((1, depth, size, size)).astype(np.float32)
    label = np.zeros((1, depth, size, size), dtype=np.float32)
    for d in tumor_slices:
        label[0, d, 0, 0] = 1.0
    return {"image": image, "label": label}


def test_slice_labels_flags_slices_with_at_least_min_tumor_voxels() -> None:
    from mri_ad.classical.dataset import slice_labels

    label = np.zeros((1, 4, 8, 8), dtype=np.float32)
    label[0, 1, :2, :2] = 1.0  # 4 voxels
    labels = slice_labels(label, min_tumor_voxels=3)
    assert labels.tolist() == [0, 1, 0, 0]


def test_build_slice_dataset_shapes_and_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    import mri_ad.classical.dataset as dataset_mod

    volumes = {
        "sub-0": _volume(4, tumor_slices={1}),
        "sub-1": _volume(4, tumor_slices=set()),
    }
    monkeypatch.setattr(dataset_mod, "load_preprocessed_volume", lambda cfg, sid: volumes[sid])

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "shape": {"height": 16, "width": 16},
            "classical": {
                "features": {
                    "label_threshold_voxels": 1,
                    "drop_empty_slices": False,
                    "foreground_eps": 1.0e-6,
                }
            },
        }
    )
    extractor = FeatureExtractor(FeatureConfig())
    data, counters = dataset_mod.build_slice_dataset(
        cfg, ["sub-0", "sub-1"], extractor, progress=False
    )

    assert data.X.shape == (8, len(feature_names(FeatureConfig())))
    assert data.y.shape == (8,)
    assert counters["n_volumes"] == 2
    assert counters["n_samples"] == 8
    assert set(data.groups.tolist()) == {"sub-0", "sub-1"}
    mask = (data.groups == "sub-0") & (data.slice_index == 1)
    assert data.y[mask].tolist() == [1]


# ── Acceptance 2/4: classical/metrics.py ─────────────────────────────────────────────────────
def _fold(fold: int, roc_auc: float, pr_auc: float, positive_rate: float = 0.2) -> FoldMetrics:
    return FoldMetrics(
        fold=fold,
        n_train=80,
        n_test=20,
        n_train_subjects=8,
        n_test_subjects=2,
        n_test_positive=int(20 * positive_rate),
        positive_rate=positive_rate,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        pr_auc_prevalence_baseline=positive_rate,
        roc_auc_majority_baseline=0.5,
        scale_pos_weight=(1 - positive_rate) / positive_rate,
    )


def test_binary_scores_returns_roc_and_pr_auc() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.4, 0.8, 0.9])
    roc_auc, pr_auc = binary_scores(y_true, y_score)
    assert 0.0 <= roc_auc <= 1.0
    assert 0.0 <= pr_auc <= 1.0


def test_binary_scores_raises_on_single_class_fold() -> None:
    with pytest.raises(ClassicalError):
        binary_scores(np.zeros(5), np.random.default_rng(0).random(5))


def test_aggregate_folds_reports_mean_and_pstdev_of_roc_and_pr_auc() -> None:
    folds = [_fold(0, 0.6, 0.3), _fold(1, 0.8, 0.5), _fold(2, 0.7, 0.4)]
    import statistics

    metrics = aggregate_folds(
        folds,
        n_samples=300,
        n_subjects=30,
        n_features=44,
        n_positive=60,
        positive_rate=0.2,
        feature_names=[f"f{i}" for i in range(44)],
    )
    assert metrics.granularity == GRANULARITY
    assert metrics.n_folds == 3
    assert metrics.roc_auc_mean == pytest.approx(statistics.fmean([0.6, 0.8, 0.7]))
    assert metrics.roc_auc_std == pytest.approx(statistics.pstdev([0.6, 0.8, 0.7]))
    assert metrics.pr_auc_mean == pytest.approx(statistics.fmean([0.3, 0.5, 0.4]))
    assert metrics.pr_auc_std == pytest.approx(statistics.pstdev([0.3, 0.5, 0.4]))


def test_class_balance_fields_are_populated() -> None:
    folds = [_fold(0, 0.6, 0.3, positive_rate=0.15)]
    metrics = aggregate_folds(
        folds,
        n_samples=100,
        n_subjects=10,
        n_features=44,
        n_positive=15,
        positive_rate=0.15,
        feature_names=[f"f{i}" for i in range(44)],
    )
    assert metrics.n_positive == 15
    assert metrics.positive_rate == 0.15


def test_pr_auc_prevalence_baseline_equals_positive_rate() -> None:
    folds = [_fold(0, 0.6, 0.3, positive_rate=0.25)]
    metrics = aggregate_folds(
        folds,
        n_samples=100,
        n_subjects=10,
        n_features=44,
        n_positive=25,
        positive_rate=0.25,
        feature_names=[f"f{i}" for i in range(44)],
    )
    assert metrics.pr_auc_prevalence_baseline_mean == pytest.approx(0.25)


def test_roc_auc_majority_baseline_is_reported_as_half() -> None:
    folds = [_fold(0, 0.6, 0.3)]
    metrics = aggregate_folds(
        folds,
        n_samples=100,
        n_subjects=10,
        n_features=44,
        n_positive=20,
        positive_rate=0.2,
        feature_names=[f"f{i}" for i in range(44)],
    )
    assert metrics.roc_auc_majority_baseline == pytest.approx(0.5)


def test_granularity_is_slice_level_in_ClassicalMetrics() -> None:
    folds = [_fold(0, 0.6, 0.3)]
    metrics = aggregate_folds(
        folds,
        n_samples=100,
        n_subjects=10,
        n_features=44,
        n_positive=20,
        positive_rate=0.2,
        feature_names=[f"f{i}" for i in range(44)],
    )
    assert metrics.granularity == "slice-level"


# ── Acceptance 2/3/4: classical/baseline.py cross-validation (stub estimator, no xgboost) ─────
class _StubEstimator:
    """Deterministic stand-in for XGBClassifier: score = normalized feature 0."""

    def __init__(self, scale_pos_weight: float) -> None:
        self.scale_pos_weight = scale_pos_weight
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _StubEstimator:
        del y
        self.feature_importances_ = np.full(X.shape[1], 1.0 / X.shape[1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        col = X[:, 0]
        span = col.max() - col.min()
        scores = (col - col.min()) / (span + 1e-9)
        return np.stack([1 - scores, scores], axis=1)


def _synthetic_slice_dataset(
    n_subjects: int = 12, per_subject: int = 20, n_features: int = 6, positive_rate: float = 0.2
) -> SliceDataset:
    rng = np.random.default_rng(1)
    x_rows, y_rows, group_rows, slice_rows = [], [], [], []
    for s in range(n_subjects):
        subject_n = per_subject + int(rng.integers(-3, 4))  # vary group size, breaks tie-symmetry
        n_pos = max(1, int(round(subject_n * positive_rate)))
        labels = np.array([1] * n_pos + [0] * (subject_n - n_pos))
        rng.shuffle(labels)
        for i, label in enumerate(labels):
            feat = rng.random(n_features)
            feat[0] = feat[0] + (2.0 if label else 0.0)  # feature 0 correlates with label
            x_rows.append(feat)
            y_rows.append(label)
            group_rows.append(f"sub-{s}")
            slice_rows.append(i)
    return SliceDataset(
        X=np.stack(x_rows).astype(np.float32),
        y=np.array(y_rows, dtype=np.uint8),
        groups=np.array(group_rows, dtype=object),
        slice_index=np.array(slice_rows, dtype=np.int64),
        feature_names=[f"f{i}" for i in range(n_features)],
    )


def _cfg_with_splits(n_splits: int = 3, *, scale_pos_weight: str | float = "auto"):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "seed": 0,
            "classical": {
                "split": {"cv": {"n_splits": n_splits, "shuffle": True}},
                "classifier": {"scale_pos_weight": scale_pos_weight, "params": {}},
            },
        }
    )


def _stub_baseline(n_splits: int = 3, **kwargs) -> ClassicalBaseline:
    cfg = _cfg_with_splits(n_splits, **kwargs)
    return ClassicalBaseline(cfg, estimator_factory=_StubEstimator)


def test_cross_validate_returns_one_FoldMetrics_per_fold() -> None:
    data = _synthetic_slice_dataset()
    baseline = _stub_baseline(n_splits=3)
    metrics = baseline.cross_validate(data)
    assert metrics.n_folds == 3
    assert len(metrics.folds) == 3


def test_no_subject_appears_in_both_train_and_test_of_any_fold() -> None:
    data = _synthetic_slice_dataset()
    baseline = _stub_baseline(n_splits=3)
    for train_idx, test_idx in baseline.make_splits(data):
        train_subjects = set(data.groups[train_idx].tolist())
        test_subjects = set(data.groups[test_idx].tolist())
        assert train_subjects.isdisjoint(test_subjects)


def test_every_sample_is_tested_exactly_once() -> None:
    data = _synthetic_slice_dataset()
    baseline = _stub_baseline(n_splits=3)
    seen = np.zeros(len(data.y), dtype=int)
    for _, test_idx in baseline.make_splits(data):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_splits_are_identical_for_the_same_seed_and_differ_for_another() -> None:
    data = _synthetic_slice_dataset()
    cfg_a = _cfg_with_splits(3)
    cfg_b = _cfg_with_splits(3)
    cfg_c = _cfg_with_splits(3)
    cfg_c.seed = 999

    splits_a = ClassicalBaseline(cfg_a, estimator_factory=_StubEstimator).make_splits(data)
    splits_b = ClassicalBaseline(cfg_b, estimator_factory=_StubEstimator).make_splits(data)
    splits_c = ClassicalBaseline(cfg_c, estimator_factory=_StubEstimator).make_splits(data)

    for (tr_a, te_a), (tr_b, te_b) in zip(splits_a, splits_b, strict=True):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(te_a, te_b)
    assert any(
        not np.array_equal(te_a, te_c)
        for (_, te_a), (_, te_c) in zip(splits_a, splits_c, strict=True)
    )


def test_ungrouped_kfold_would_split_a_subject() -> None:
    """Documents WHY grouping matters: near-duplicate slices of one subject leak un-grouped."""
    from sklearn.model_selection import StratifiedKFold

    data = _synthetic_slice_dataset(n_subjects=4, per_subject=20)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    leaked = False
    for train_idx, test_idx in skf.split(data.X, data.y):
        train_subjects = set(data.groups[train_idx].tolist())
        test_subjects = set(data.groups[test_idx].tolist())
        if train_subjects & test_subjects:
            leaked = True
            break
    assert leaked  # ungrouped KFold leaks a subject across train/test on this data


def test_class_balance_and_baselines_from_cross_validate() -> None:
    data = _synthetic_slice_dataset(positive_rate=0.2)
    baseline = _stub_baseline(n_splits=3)
    metrics = baseline.cross_validate(data)
    assert metrics.n_positive == int(data.y.sum())
    assert metrics.positive_rate == pytest.approx(float(data.y.mean()))
    for fold in metrics.folds:
        assert fold.pr_auc_prevalence_baseline == pytest.approx(fold.positive_rate)
        assert fold.roc_auc_majority_baseline == pytest.approx(0.5)


def test_scale_pos_weight_is_computed_from_the_train_fold_only() -> None:
    data = _synthetic_slice_dataset(positive_rate=0.2)
    baseline = _stub_baseline(n_splits=3)
    splits = baseline.make_splits(data)
    metrics = baseline.cross_validate(data)

    for fold, (train_idx, _) in zip(metrics.folds, splits, strict=True):
        y_train = data.y[train_idx]
        n_pos, n_neg = int(y_train.sum()), int(len(y_train) - y_train.sum())
        expected = n_neg / n_pos
        assert fold.scale_pos_weight == pytest.approx(expected)


def test_single_class_fold_raises_rather_than_returning_nan() -> None:
    rng = np.random.default_rng(2)
    x = rng.random((10, 4)).astype(np.float32)
    y = np.zeros(10, dtype=np.uint8)  # every label is 0 -> no fold can be stratified
    data = SliceDataset(
        X=x,
        y=y,
        groups=np.array([f"sub-{i}" for i in range(10)], dtype=object),
        slice_index=np.arange(10),
        feature_names=["f0", "f1", "f2", "f3"],
    )
    baseline = _stub_baseline(n_splits=3)
    with pytest.raises(ClassicalError):
        baseline.cross_validate(data)


def test_feature_importance_raises_before_cross_validate() -> None:
    baseline = _stub_baseline(n_splits=3)
    with pytest.raises(ClassicalError):
        _ = baseline.feature_importance


def test_feature_importance_is_mean_gain_across_folds() -> None:
    data = _synthetic_slice_dataset()
    baseline = _stub_baseline(n_splits=3)
    baseline.cross_validate(data)
    importance = baseline.feature_importance
    assert set(importance) == set(data.feature_names)
    assert all(v > 0 for v in importance.values())


def test_importing_classical_does_not_require_xgboost(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "xgboost", None)  # poison the import
    import importlib

    import mri_ad.classical as classical_mod

    importlib.reload(classical_mod)  # must not raise even with xgboost poisoned


# ── Acceptance 6: every generated artifact records granularity ────────────────────────────────
def _sample_metrics() -> object:
    folds = (_fold(0, 0.6, 0.3, positive_rate=0.2), _fold(1, 0.7, 0.4, positive_rate=0.2))
    return aggregate_folds(
        folds,
        n_samples=200,
        n_subjects=20,
        n_features=4,
        n_positive=40,
        positive_rate=0.2,
        feature_names=["f0", "f1", "f2", "f3"],
    )


def _write_full_report(tmp_path) -> ClassicalReportGenerator:
    gen = ClassicalReportGenerator(tmp_path / "metrics", tmp_path / "figures")
    metrics = _sample_metrics()
    gen.write_per_fold(metrics)
    gen.write_metrics_json(
        metrics,
        run_id="run-1",
        seed=42,
        complete=True,
        subject_limit=None,
        split_info={"source": "brats_test", "split_hash": "x" * 12},
        features_info={"n_features": 4, "library": "scikit-image", "pyradiomics_used": False},
        counts_info={"n_samples": 200},
        classifier_info={"name": "xgboost"},
    )
    gen.write_feature_importance({"f0": 0.4, "f1": 0.3, "f2": 0.2, "f3": 0.1})
    gen.write_summary_markdown()
    gen.plot_class_balance({"n_positive": 40, "n_negative": 160})
    gen.plot_fold_auc(metrics)
    return gen


def test_granularity_is_slice_level_end_to_end() -> None:
    metrics = _sample_metrics()
    assert metrics.granularity == "slice-level"


@pytest.mark.parametrize(
    "check",
    [
        "per_fold_csv",
        "metrics_json_top_level",
        "metrics_json_note",
        "feature_importance_csv",
        "summary_markdown",
    ],
)
def test_every_generated_artifact_records_granularity(tmp_path, check: str) -> None:
    gen = _write_full_report(tmp_path)

    if check == "per_fold_csv":
        import csv

        with gen._per_fold_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert all(row["granularity"] == "slice-level" for row in rows)
    elif check == "metrics_json_top_level":
        payload = gen.read_metrics_json()
        assert payload["granularity"] == "slice-level"
    elif check == "metrics_json_note":
        payload = gen.read_metrics_json()
        assert "does not localize" in payload["granularity_note"].lower() or (
            "not comparable" in payload["granularity_note"].lower()
        )
    elif check == "feature_importance_csv":
        import csv

        with gen._feature_importance_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert all(row["granularity"] == "slice-level" for row in rows)
    elif check == "summary_markdown":
        text = gen._summary_path.read_text()
        assert "slice-level" in text
