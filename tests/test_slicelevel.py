"""Spec 007: ``eval/slicelevel.py`` — the DL -> slice-level reduction.

Synthetic tensors only, no real volumes, no model forward pass — mirrors ``tests/test_recon.py``'s
house convention. ``(1, 8, 16, 16)``-sized results, per the standing "nothing in this suite runs a
model or touches a real volume" rule.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("monai")

from mri_ad.classical.dataset import slice_labels  # noqa: E402
from mri_ad.eval.slicelevel import (  # noqa: E402
    reduce_result,
    reduce_results_dir,
    tune_slice_threshold,
    write_slice_scores,
)
from mri_ad.exceptions import EvalError  # noqa: E402
from mri_ad.recon.io import save_result  # noqa: E402
from mri_ad.recon.types import ReconResult  # noqa: E402

FOREGROUND_EPS = 1e-6


def _result(volume_id: str = "sub-0", *, with_ground_truth: bool = True) -> ReconResult:
    shape = (1, 8, 16, 16)
    original = torch.zeros(shape)
    original[0, 2:6] = 0.5  # slices 0,1,6,7 are "background" (zero-padded tail lookalike)

    mask = torch.zeros(shape)
    mask[0, 3, :4, :4] = 1.0  # slice 3 has some flagged voxels

    ground_truth = None
    if with_ground_truth:
        ground_truth = torch.zeros(shape)
        ground_truth[0, 3, :2, :2] = 1.0  # slice 3 is tumor-bearing

    return ReconResult(
        volume_id=volume_id,
        original=original,
        reconstruction=original.clone(),
        residual=torch.zeros(shape),
        anomaly_mask=mask,
        ground_truth=ground_truth,
    )


# ── reduction correctness ───────────────────────────────────────────────────────────────────────
def test_padded_tail_slices_are_dropped_by_the_foreground_rule() -> None:
    result = _result()
    rows = reduce_result(result, foreground_eps=FOREGROUND_EPS)
    surviving = {r.slice_index for r in rows}
    assert surviving == {2, 3, 4, 5}  # slices 0,1,6,7 have zero foreground -> dropped


def test_flagged_fraction_and_label_are_correct_on_the_surviving_slice() -> None:
    result = _result()
    rows = reduce_result(result, foreground_eps=FOREGROUND_EPS)
    row3 = next(r for r in rows if r.slice_index == 3)
    assert row3.flagged_voxels == 16  # 4x4
    assert row3.n_voxels == 256  # 16x16
    assert row3.flagged_fraction == pytest.approx(16 / 256)
    assert row3.label == 1

    row2 = next(r for r in rows if r.slice_index == 2)
    assert row2.flagged_voxels == 0
    assert row2.label == 0


def test_result_without_ground_truth_raises() -> None:
    result = _result(with_ground_truth=False)
    with pytest.raises(EvalError):
        reduce_result(result, foreground_eps=FOREGROUND_EPS)


def test_label_rule_agrees_with_classical_slice_labels() -> None:
    """The eval-side label restatement must numerically match classical's implementation."""
    result = _result()
    rows = reduce_result(result, foreground_eps=FOREGROUND_EPS)
    labels_by_slice = {r.slice_index: r.label for r in rows}

    classical_labels = slice_labels(result.ground_truth.numpy(), min_tumor_voxels=1)
    for slice_index, label in labels_by_slice.items():
        assert label == int(classical_labels[slice_index])


def test_reduce_results_dir_streams_every_saved_result(tmp_path) -> None:
    root = tmp_path / "results"
    save_result(_result("sub-0"), root, "run-a")
    save_result(_result("sub-1"), root, "run-a")

    rows = reduce_results_dir(root / "run-a", foreground_eps=FOREGROUND_EPS)
    volume_ids = {r.volume_id for r in rows}
    assert volume_ids == {"sub-0", "sub-1"}


def test_write_slice_scores_round_trips_via_csv(tmp_path) -> None:
    rows = reduce_result(_result(), foreground_eps=FOREGROUND_EPS)
    path = write_slice_scores(rows, tmp_path / "slice_scores.csv")
    import csv as csv_module

    with path.open() as fh:
        read_rows = list(csv_module.DictReader(fh))
    assert len(read_rows) == len(rows)
    assert set(read_rows[0].keys()) == {
        "volume_id",
        "slice_index",
        "flagged_voxels",
        "n_voxels",
        "flagged_fraction",
        "foreground_fraction",
        "label",
    }


# ── val-only threshold tuner ─────────────────────────────────────────────────────────────────────
def test_tuner_refuses_a_test_split() -> None:
    rows = reduce_result(_result(), foreground_eps=FOREGROUND_EPS)
    with pytest.raises(EvalError, match="test"):
        tune_slice_threshold(rows, [0.01, 0.1], split="test")


def test_tuner_scores_precision_recall_f1_on_the_val_split() -> None:
    rows = reduce_result(_result(), foreground_eps=FOREGROUND_EPS)
    points = tune_slice_threshold(rows, [0.01, 0.5], split="val")
    assert all(p.split == "val" for p in points)
    assert all(0.0 <= p.precision <= 1.0 for p in points)
    assert all(0.0 <= p.recall <= 1.0 for p in points)


def test_tuner_raises_on_empty_rows() -> None:
    with pytest.raises(EvalError):
        tune_slice_threshold([], [0.1], split="val")
