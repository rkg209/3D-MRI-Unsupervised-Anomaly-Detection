"""Grouped cross-validation for the classical baseline (Spec 006, acceptance 2/3/4).

``StratifiedGroupKFold`` grouped by BraTS subject is **not** a config switch: adjacent slices of
one brain are near-duplicates, and an ungrouped split leaks a subject across train/test and
inflates AUC to a fake number (the single easiest way to get a fake result on this baseline).
``scale_pos_weight="auto"`` is recomputed from the training fold only, every fold — the test
fold's class balance must never leak into training.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from mri_ad.classical.dataset import SliceDataset
from mri_ad.classical.metrics import (
    GRANULARITY,
    ClassicalMetrics,
    FoldMetrics,
    aggregate_folds,
    binary_scores,
)
from mri_ad.exceptions import ClassicalError

EstimatorFactory = Callable[..., Any]


@dataclass(frozen=True)
class OofPrediction:
    """One held-out slice's prediction from cross-validation (Spec 007 acceptance 1/3).

    Retained so the classical baseline's ROC/PR curve can be recomputed by the same shared
    curve function every other paradigm's column uses (``eval/curves.py``) — 006 only ever
    published scalar per-fold means, which cannot render a curve.
    """

    granularity: str
    fold: int
    volume_id: str
    slice_index: int
    y_true: int
    y_score: float


class ClassicalBaseline:
    """Owns the grouped CV split, per-fold training/scoring, and feature-importance bookkeeping."""

    def __init__(
        self, cfg: DictConfig, *, estimator_factory: EstimatorFactory | None = None
    ) -> None:
        """``estimator_factory(scale_pos_weight=...)`` overrides the default XGBoost builder.

        Tests inject a stub factory so they never need a working ``xgboost`` import (R6).
        """
        self.cfg = cfg
        self._estimator_factory = estimator_factory
        self._fold_importances: list[dict[str, float]] = []
        self._oof_predictions: list[OofPrediction] = []

    def make_splits(self, data: SliceDataset) -> list[tuple[np.ndarray, np.ndarray]]:
        """``StratifiedGroupKFold`` splits, grouped by subject. Disjointness is asserted."""
        from sklearn.model_selection import StratifiedGroupKFold

        n_splits = int(self.cfg.classical.split.cv.n_splits)
        shuffle = bool(self.cfg.classical.split.cv.shuffle)
        seed = int(self.cfg.seed)

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        splits = list(splitter.split(data.X, data.y, groups=data.groups))

        for fold, (train_idx, test_idx) in enumerate(splits):
            train_subjects = set(data.groups[train_idx].tolist())
            test_subjects = set(data.groups[test_idx].tolist())
            if train_subjects & test_subjects:
                raise ClassicalError(
                    f"Fold {fold}: subject(s) {train_subjects & test_subjects} appear in both "
                    "train and test — grouped split invariant violated."
                )
        return splits

    def _build_estimator(self, *, scale_pos_weight: float) -> Any:
        if self._estimator_factory is not None:
            return self._estimator_factory(scale_pos_weight=scale_pos_weight)

        from mri_ad.classical.classifier import build_xgboost

        params = OmegaConf.to_container(self.cfg.classical.classifier.params, resolve=True)
        return build_xgboost(params, seed=int(self.cfg.seed), scale_pos_weight=scale_pos_weight)

    def _resolve_scale_pos_weight(self, y_train: np.ndarray) -> float:
        configured = self.cfg.classical.classifier.scale_pos_weight
        if configured != "auto":
            return float(configured)
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        if n_pos == 0:
            raise ClassicalError("Training fold has zero positive samples — cannot fit.")
        return n_neg / n_pos

    def cross_validate(self, data: SliceDataset) -> ClassicalMetrics:
        """Fit + score every fold. Returns aggregate metrics; one :class:`FoldMetrics` per fold."""
        splits = self.make_splits(data)
        self._fold_importances = []
        self._oof_predictions = []
        folds: list[FoldMetrics] = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            y_train, y_test = data.y[train_idx], data.y[test_idx]
            scale_pos_weight = self._resolve_scale_pos_weight(y_train)

            estimator = self._build_estimator(scale_pos_weight=scale_pos_weight)
            estimator.fit(data.X[train_idx], y_train)
            scores = np.asarray(estimator.predict_proba(data.X[test_idx]))[:, 1]
            roc_auc, pr_auc = binary_scores(y_test, scores)

            test_groups = data.groups[test_idx]
            test_slice_index = data.slice_index[test_idx]
            for i, _idx in enumerate(test_idx):
                self._oof_predictions.append(
                    OofPrediction(
                        granularity=GRANULARITY,
                        fold=fold,
                        volume_id=str(test_groups[i]),
                        slice_index=int(test_slice_index[i]),
                        y_true=int(y_test[i]),
                        y_score=float(scores[i]),
                    )
                )

            positive_rate = float(y_test.mean())
            folds.append(
                FoldMetrics(
                    fold=fold,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    n_train_subjects=len(set(data.groups[train_idx].tolist())),
                    n_test_subjects=len(set(data.groups[test_idx].tolist())),
                    n_test_positive=int(y_test.sum()),
                    positive_rate=positive_rate,
                    roc_auc=roc_auc,
                    pr_auc=pr_auc,
                    pr_auc_prevalence_baseline=positive_rate,
                    roc_auc_majority_baseline=0.5,
                    scale_pos_weight=scale_pos_weight,
                )
            )
            importances = getattr(estimator, "feature_importances_", None)
            if importances is not None:
                self._fold_importances.append(
                    dict(zip(data.feature_names, (float(v) for v in importances), strict=True))
                )

        return aggregate_folds(
            folds,
            n_samples=len(data.y),
            n_subjects=len(set(data.groups.tolist())),
            n_features=data.X.shape[1],
            n_positive=int(data.y.sum()),
            positive_rate=float(data.y.mean()),
            feature_names=data.feature_names,
        )

    @property
    def feature_importance(self) -> dict[str, float]:
        """Mean gain per feature across folds. Raises if accessed before :meth:`cross_validate`."""
        if not self._fold_importances:
            raise ClassicalError("feature_importance accessed before cross_validate() ran.")
        names = self._fold_importances[0].keys()
        return {
            name: statistics.fmean(fold.get(name, 0.0) for fold in self._fold_importances)
            for name in names
        }

    @property
    def oof_predictions(self) -> list[OofPrediction]:
        """Out-of-fold predictions, one per held-out slice. Raises before :meth:`cross_validate`."""
        if not self._oof_predictions:
            raise ClassicalError("oof_predictions accessed before cross_validate() ran.")
        return list(self._oof_predictions)


__all__ = ["ClassicalBaseline", "OofPrediction"]
