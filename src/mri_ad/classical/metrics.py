"""Cross-validated classification metrics for the classical baseline (Spec 006, acceptance 6/7).

Lives here, not in ``eval/metrics.py`` — acceptance test 7 forbids ``classical/`` from importing
``eval/``. :func:`aggregate_folds` reimplements ``eval/metrics.aggregate``'s ``statistics.pstdev``
convention rather than importing it, and that duplication is intentional (see R8 in the plan).

**Granularity.** This baseline operates on 2D axial slices, not 3D volumes: it does not localize
an anomaly, only classify a slice as tumor-bearing or not. ``GRANULARITY_NOTE`` is stamped into
every artifact this module or ``report.py`` produces so a reader never mistakes this ROC-AUC for
something comparable to voxel-level Dice.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mri_ad.exceptions import ClassicalError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

GRANULARITY = "slice-level"
GRANULARITY_NOTE = (
    "Operating granularity is slice-level: each 2D axial slice is one sample. This does NOT "
    "localize an anomaly, so ROC-AUC here is NOT comparable to voxel-level Dice."
)


@dataclass(frozen=True)
class FoldMetrics:
    """Metrics for one cross-validation fold."""

    fold: int
    n_train: int
    n_test: int
    n_train_subjects: int
    n_test_subjects: int
    n_test_positive: int
    positive_rate: float
    roc_auc: float
    pr_auc: float
    pr_auc_prevalence_baseline: float
    roc_auc_majority_baseline: float
    scale_pos_weight: float


@dataclass(frozen=True)
class ClassicalMetrics:
    """Aggregate metrics across every fold, plus the provenance needed to reproduce them."""

    granularity: str
    n_folds: int
    n_samples: int
    n_subjects: int
    n_features: int
    n_positive: int
    positive_rate: float
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    pr_auc_prevalence_baseline_mean: float
    roc_auc_majority_baseline: float
    folds: tuple[FoldMetrics, ...]
    feature_names: tuple[str, ...]


def binary_scores(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """``(roc_auc, pr_auc)`` for one fold. Raises :class:`ClassicalError` on a single-class fold.

    PR-AUC is ``sklearn.metrics.average_precision_score`` (step-wise), never the trapezoidal
    rule under the PR curve — the trapezoid is optimistically biased for a stepped curve.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    unique = set(int(v) for v in y_true)
    if len(unique) < 2:
        raise ClassicalError(
            f"binary_scores: fold has a single class ({unique}) — ROC-AUC/PR-AUC are undefined. "
            "This is a stratification failure, not a value to silently report as NaN."
        )
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    return roc_auc, pr_auc


def aggregate_folds(
    folds: Sequence[FoldMetrics],
    *,
    n_samples: int,
    n_subjects: int,
    n_features: int,
    n_positive: int,
    positive_rate: float,
    feature_names: Sequence[str],
) -> ClassicalMetrics:
    """Mean +/- population-std across ``folds`` (matches ``eval/metrics.aggregate``)."""
    if not folds:
        raise ClassicalError("aggregate_folds received zero folds.")

    def _mean_std(values: list[float]) -> tuple[float, float]:
        return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0

    roc_auc_mean, roc_auc_std = _mean_std([f.roc_auc for f in folds])
    pr_auc_mean, pr_auc_std = _mean_std([f.pr_auc for f in folds])
    pr_auc_prevalence_baseline_mean = statistics.fmean(
        [f.pr_auc_prevalence_baseline for f in folds]
    )
    roc_auc_majority_baseline = statistics.fmean([f.roc_auc_majority_baseline for f in folds])

    return ClassicalMetrics(
        granularity=GRANULARITY,
        n_folds=len(folds),
        n_samples=n_samples,
        n_subjects=n_subjects,
        n_features=n_features,
        n_positive=n_positive,
        positive_rate=positive_rate,
        roc_auc_mean=roc_auc_mean,
        roc_auc_std=roc_auc_std,
        pr_auc_mean=pr_auc_mean,
        pr_auc_std=pr_auc_std,
        pr_auc_prevalence_baseline_mean=pr_auc_prevalence_baseline_mean,
        roc_auc_majority_baseline=roc_auc_majority_baseline,
        folds=tuple(folds),
        feature_names=tuple(feature_names),
    )


__all__ = [
    "GRANULARITY",
    "GRANULARITY_NOTE",
    "ClassicalMetrics",
    "FoldMetrics",
    "aggregate_folds",
    "binary_scores",
]
