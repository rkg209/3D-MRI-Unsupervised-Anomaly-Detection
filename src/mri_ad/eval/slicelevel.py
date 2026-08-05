"""DL voxel-mask -> slice-level reduction (Spec 007). Imports ``torch`` — NOT report-safe.

Gives each DL model's voxel-level anomaly mask a slice-level score directly comparable to the
classical baseline's slice-level ROC-AUC (``classical/metrics.py::binary_scores``). The foreground
drop rule here is a **deliberate, exact duplication** of
``classical/dataset.py::build_slice_dataset`` — same ``foreground_eps``, applied to
``ReconResult.original`` rather than the classical pipeline's ``load_preprocessed_volume`` output,
but both pipelines share the identical preprocessing upstream, so the surviving
``(volume_id, slice_index)`` key sets are provably the same set (see the plan's "same-sample
assertion" section). This is what makes chunking's zero-padded tail slices (155 -> 160) fall out
automatically: a zero-padded slice has foreground fraction 0 and is dropped by both sides.

The label rule (``label = 1 iff slice holds >= min_tumor_voxels tumor voxels``) restates
``classical/dataset.py::slice_labels`` rather than importing it — ``eval/`` must not import
``classical/`` (mirror of Spec 006 acceptance 7, kept for symmetry). ``tests/test_slicelevel.py``
pins numeric agreement between the two independent implementations.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from mri_ad.eval.loader import iter_results
from mri_ad.exceptions import EvalError
from mri_ad.recon.types import ReconResult

SLICE_SCORE_FIELDS = (
    "volume_id",
    "slice_index",
    "flagged_voxels",
    "n_voxels",
    "flagged_fraction",
    "foreground_fraction",
    "label",
)


@dataclass(frozen=True)
class SliceScore:
    """One surviving slice's flagged-voxel reduction + label."""

    volume_id: str
    slice_index: int
    flagged_voxels: int
    n_voxels: int
    flagged_fraction: float
    foreground_fraction: float
    label: int


def reduce_result(
    result: ReconResult, *, foreground_eps: float, min_tumor_voxels: int = 1
) -> list[SliceScore]:
    """Reduce one volume's ``anomaly_mask`` to one :class:`SliceScore` per surviving slice.

    Requires ``result.ground_truth is not None`` — an OpenBHB result carries no label and must
    never silently become an all-negative column in the comparison. Drops a slice whose
    foreground fraction ``<= foreground_eps`` (identical rule to
    ``classical/dataset.py::build_slice_dataset``, applied to ``original`` instead of the
    classical pipeline's own load — see the module docstring).
    """
    if result.ground_truth is None:
        raise EvalError(
            f"reduce_result: {result.volume_id!r} has no ground_truth — the slice-level "
            "reduction requires a BraTS ReconResult, not an OpenBHB one."
        )

    original = result.original[0]  # (D, H, W)
    mask = result.anomaly_mask[0]  # (D, H, W)
    ground_truth = result.ground_truth[0]  # (D, H, W)
    depth = original.shape[0]

    rows: list[SliceScore] = []
    for d in range(depth):
        foreground_fraction = float((original[d] > foreground_eps).float().mean().item())
        if foreground_fraction <= foreground_eps:
            continue

        flagged_voxels = int(mask[d].sum().item())
        n_voxels = int(mask[d].numel())
        tumor_voxels = int((ground_truth[d] > 0).sum().item())
        label = int(tumor_voxels >= min_tumor_voxels)

        rows.append(
            SliceScore(
                volume_id=result.volume_id,
                slice_index=d,
                flagged_voxels=flagged_voxels,
                n_voxels=n_voxels,
                flagged_fraction=flagged_voxels / n_voxels,
                foreground_fraction=foreground_fraction,
                label=label,
            )
        )
    return rows


def reduce_results_dir(
    results_dir: Path, *, foreground_eps: float, min_tumor_voxels: int = 1
) -> list[SliceScore]:
    """Reduce every ``ReconResult`` under ``results_dir`` (streamed, one volume at a time)."""
    rows: list[SliceScore] = []
    for result in iter_results(results_dir):
        rows.extend(
            reduce_result(result, foreground_eps=foreground_eps, min_tumor_voxels=min_tumor_voxels)
        )
    return rows


def write_slice_scores(rows: Sequence[SliceScore], path: Path) -> Path:
    """Write ``slice_scores.csv`` — one row per surviving slice."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SLICE_SCORE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


@dataclass(frozen=True)
class ThresholdPoint:
    """One scored slice-level decision threshold, always from the ``val`` split."""

    threshold: float
    precision: float
    recall: float
    f1: float
    split: str


def tune_slice_threshold(
    rows: Sequence[SliceScore], grid: Sequence[float], *, split: str
) -> list[ThresholdPoint]:
    """Score every ``grid`` threshold's precision/recall/F1 on ``flagged_fraction``.

    Raises :class:`EvalError` for any ``split != "val"`` — trap #5 (never tune a threshold on
    the test split), enforced at the function boundary, mirroring
    ``recon/sweep.py::run_threshold_sweep``.
    """
    if split != "val":
        raise EvalError(
            f"tune_slice_threshold refuses split={split!r}: the slice decision threshold must "
            "be tuned on 'val' only, never 'test' (trap #5)."
        )
    if not rows:
        raise EvalError("tune_slice_threshold received an empty row list.")

    points: list[ThresholdPoint] = []
    for threshold in grid:
        tp = fp = fn = 0
        for row in rows:
            predicted = row.flagged_fraction >= threshold
            if predicted and row.label == 1:
                tp += 1
            elif predicted and row.label == 0:
                fp += 1
            elif not predicted and row.label == 1:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        points.append(
            ThresholdPoint(
                threshold=float(threshold), precision=precision, recall=recall, f1=f1, split=split
            )
        )
    return points


__all__ = [
    "SLICE_SCORE_FIELDS",
    "SliceScore",
    "ThresholdPoint",
    "reduce_result",
    "reduce_results_dir",
    "tune_slice_threshold",
    "write_slice_scores",
]
