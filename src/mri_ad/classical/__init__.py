"""Classical-ML baseline: radiomic features -> gradient boosting (Spec 006).

Importing this package must never require ``xgboost`` (R6: it fails to import on some machines
missing ``libomp``) — ``classifier.py`` is the one module that touches it, and it is imported
lazily from inside :class:`~mri_ad.classical.baseline.ClassicalBaseline`, never at module scope
here or anywhere else in this package.
"""

from __future__ import annotations

from mri_ad.classical.baseline import ClassicalBaseline
from mri_ad.classical.cache import FeatureCache, FeatureCacheHeader
from mri_ad.classical.dataset import SliceDataset, build_slice_dataset, slice_labels
from mri_ad.classical.features import FeatureConfig, FeatureExtractor, feature_names
from mri_ad.classical.metrics import (
    GRANULARITY,
    GRANULARITY_NOTE,
    ClassicalMetrics,
    FoldMetrics,
    aggregate_folds,
    binary_scores,
)
from mri_ad.classical.report import ClassicalReportGenerator

__all__ = [
    "GRANULARITY",
    "GRANULARITY_NOTE",
    "ClassicalBaseline",
    "ClassicalMetrics",
    "ClassicalReportGenerator",
    "FeatureCache",
    "FeatureCacheHeader",
    "FeatureConfig",
    "FeatureExtractor",
    "FoldMetrics",
    "SliceDataset",
    "aggregate_folds",
    "binary_scores",
    "build_slice_dataset",
    "feature_names",
    "slice_labels",
]
