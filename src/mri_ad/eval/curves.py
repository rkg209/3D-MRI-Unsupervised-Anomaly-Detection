"""Report-safe, pure-Python ROC/PR curves (Spec 007).

No ``sklearn``/``scipy``/``numpy`` — this module lives in the report-safe subgraph alongside
``eval/report.py`` and ``eval/matrix.py`` (``tests/test_eval_boundary.py`` AST-walks for exactly
this). Two implementation details are load-bearing, not stylistic:

**Ties.** Scores are sorted descending and every row sharing a score is consumed as one group
before a curve point is emitted — a threshold that cannot be realized (because it would have to
split a group of identical scores) must never appear as an achievable operating point, and naive
per-row emission inflates AUC. Zero-flagged-voxel slices form one large tie group, so this is the
common case here, not an edge case.

**PR-AUC** is the step-wise sum ``sum((R_i - R_{i-1}) * P_i)`` — matching
``sklearn.metrics.average_precision_score`` — never the trapezoidal rule under the PR curve, which
is optimistically biased for a stepped curve and would silently disagree with the classical
baseline's published numbers (``classical/metrics.py::binary_scores``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mri_ad.exceptions import EvalError


@dataclass(frozen=True)
class CurvePoint:
    """One realized operating point on a curve."""

    threshold: float
    x: float  # FPR (ROC) or recall (PR)
    y: float  # TPR (ROC) or precision (PR)


@dataclass(frozen=True)
class BinaryCurves:
    """ROC + PR curves and their AUCs for one binary-scored sample set."""

    roc: tuple[CurvePoint, ...]
    pr: tuple[CurvePoint, ...]
    roc_auc: float
    pr_auc: float
    n: int
    n_positive: int
    prevalence: float
    n_tied_score_groups: int


def _grouped_descending(
    y_true: Sequence[int], y_score: Sequence[float]
) -> list[tuple[float, int, int]]:
    """``[(score, n_positive_in_group, n_total_in_group), ...]`` sorted by score descending."""
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    groups: list[tuple[float, int, int]] = []
    i = 0
    while i < len(order):
        j = i
        score = y_score[order[i]]
        n_pos = 0
        n_total = 0
        while j < len(order) and y_score[order[j]] == score:
            n_total += 1
            n_pos += int(y_true[order[j]])
            j += 1
        groups.append((score, n_pos, n_total))
        i = j
    return groups


def _check_binary(y_true: Sequence[int]) -> tuple[int, int]:
    n = len(y_true)
    n_positive = sum(int(v) for v in y_true)
    if n == 0:
        raise EvalError("binary_curves: received zero samples.")
    unique = {int(v) for v in y_true}
    if len(unique) < 2:
        raise EvalError(
            f"binary_curves: single-class input ({unique}) — ROC-AUC/PR-AUC are undefined."
        )
    return n, n_positive


def binary_curves(y_true: Sequence[int], y_score: Sequence[float]) -> BinaryCurves:
    """Compute ROC + PR curves, their AUCs, and ties diagnostics. Raises on single-class input."""
    n, n_positive = _check_binary(y_true)
    n_negative = n - n_positive
    groups = _grouped_descending(y_true, y_score)

    roc_points: list[CurvePoint] = []
    pr_points: list[CurvePoint] = []
    tp = 0
    fp = 0
    roc_auc = 0.0
    pr_auc = 0.0
    prev_recall = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for score, group_pos, group_total in groups:
        group_neg = group_total - group_pos
        tp += group_pos
        fp += group_neg

        tpr = tp / n_positive
        fpr = fp / n_negative if n_negative else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tpr

        # Trapezoid under ROC is correct there (a straight interpolation between two realized
        # points on a continuous-score ROC curve is exact); PR-AUC uses the step rule instead —
        # see the module docstring.
        roc_auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        pr_auc += (recall - prev_recall) * precision

        roc_points.append(CurvePoint(threshold=score, x=fpr, y=tpr))
        pr_points.append(CurvePoint(threshold=score, x=recall, y=precision))

        prev_fpr, prev_tpr, prev_recall = fpr, tpr, recall

    return BinaryCurves(
        roc=tuple(roc_points),
        pr=tuple(pr_points),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        n=n,
        n_positive=n_positive,
        prevalence=n_positive / n,
        n_tied_score_groups=len(groups),
    )


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """ROC-AUC only, via :func:`binary_curves`."""
    return binary_curves(y_true, y_score).roc_auc


def average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """PR-AUC (average precision) only, via :func:`binary_curves`."""
    return binary_curves(y_true, y_score).pr_auc


__all__ = ["BinaryCurves", "CurvePoint", "average_precision", "binary_curves", "roc_auc"]
