"""Spec 007: ``eval/curves.py`` — ROC/PR correctness, ties, sklearn parity.

No ``monai``/``torch`` import needed here — pure Python + sklearn-in-tests-only (the boundary
test only constrains ``src/``, per the plan).
"""

from __future__ import annotations

import random

import pytest

from mri_ad.eval.curves import average_precision, binary_curves, roc_auc
from mri_ad.exceptions import EvalError


def test_roc_auc_matches_sklearn_on_random_scores() -> None:
    from sklearn.metrics import roc_auc_score

    rng = random.Random(0)
    y_true = [rng.randint(0, 1) for _ in range(200)]
    if len(set(y_true)) < 2:
        y_true[0] = 0
        y_true[1] = 1
    y_score = [rng.random() for _ in range(200)]

    ours = roc_auc(y_true, y_score)
    theirs = float(roc_auc_score(y_true, y_score))
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_average_precision_matches_sklearn_including_ties() -> None:
    from sklearn.metrics import average_precision_score

    y_true = [1, 0, 1, 0, 1, 0, 0, 1]
    y_score = [0.9, 0.9, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]  # deliberate tie groups

    ours = average_precision(y_true, y_score)
    theirs = float(average_precision_score(y_true, y_score))
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_all_zero_scores_give_auc_half_not_one() -> None:
    y_true = [0, 1, 0, 1, 0, 1]
    y_score = [0.0] * 6  # one giant tie group -> every threshold ties
    curves = binary_curves(y_true, y_score)
    assert curves.roc_auc == pytest.approx(0.5)
    assert curves.n_tied_score_groups == 1


def test_single_class_input_raises() -> None:
    with pytest.raises(EvalError):
        binary_curves([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(EvalError):
        binary_curves([0, 0, 0], [0.1, 0.2, 0.3])


def test_prevalence_and_n_are_reported() -> None:
    y_true = [1, 0, 0, 0]
    curves = binary_curves(y_true, [0.9, 0.1, 0.2, 0.3])
    assert curves.n == 4
    assert curves.n_positive == 1
    assert curves.prevalence == pytest.approx(0.25)


def test_binary_curves_raises_on_empty_input() -> None:
    with pytest.raises(EvalError):
        binary_curves([], [])
