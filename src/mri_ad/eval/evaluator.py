"""Scores saved ``ReconResult``s against ground truth (Spec 004).

``VolumeEvaluator`` scores one volume; ``AggregateEvaluator`` walks a results directory and
aggregates. Neither ever constructs or calls a model — they take **paths**, never a model
instance (acceptance test 3), which is what makes ``make eval`` (normal mode) forward-pass-free
(acceptance test 4).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from mri_ad.eval.loader import iter_results
from mri_ad.eval.metrics import AggregateMetrics, MetricsComputer, VolumeMetrics, aggregate
from mri_ad.exceptions import EvalError
from mri_ad.recon.types import ReconResult


class VolumeEvaluator:
    """Scores a single :class:`ReconResult` against its ground truth."""

    def __init__(self, metrics: type[MetricsComputer] = MetricsComputer) -> None:
        """Store the metrics implementation (swappable for tests, never for production use)."""
        self.metrics = metrics

    def evaluate(self, result: ReconResult, label: Tensor | None = None) -> VolumeMetrics:
        """Score ``result``. ``label`` overrides ``result.ground_truth`` when given (D-B)."""
        gt = label if label is not None else result.ground_truth
        if gt is None:
            raise EvalError(
                f"ReconResult {result.volume_id!r} has no ground truth and none was supplied "
                "via `label` — the evaluation harness cannot score anomaly detection without it."
            )
        if gt.shape != result.anomaly_mask.shape:
            raise EvalError(
                f"ReconResult {result.volume_id!r}: ground_truth shape {tuple(gt.shape)} != "
                f"anomaly_mask shape {tuple(result.anomaly_mask.shape)}."
            )
        uniques = set(gt.unique().tolist())
        if not uniques <= {0.0, 1.0}:
            raise EvalError(
                f"ReconResult {result.volume_id!r}: ground_truth is not binarized (values "
                f"{sorted(uniques)}). Trap #1 — binarize with (seg > 0) before evaluation; "
                "never feed a raw {0,1,2,4} BraTS map into Dice/IoU."
            )
        mask_uniques = set(result.anomaly_mask.unique().tolist())
        if not mask_uniques <= {0.0, 1.0}:
            raise EvalError(
                f"ReconResult {result.volume_id!r}: anomaly_mask is not binary (values "
                f"{sorted(mask_uniques)}). Thresholding (recon/) must always produce a {{0,1}} "
                "mask, never a soft/probability map."
            )

        dice = self.metrics.dice(result.anomaly_mask, gt)
        iou = self.metrics.iou(result.anomaly_mask, gt)
        psnr = self.metrics.psnr(result.reconstruction, result.original)
        ssim = self.metrics.ssim(result.reconstruction, result.original)

        return VolumeMetrics(volume_id=result.volume_id, dice=dice, iou=iou, psnr=psnr, ssim=ssim)


class AggregateEvaluator:
    """Walks a results directory and scores every volume in it."""

    def __init__(self, volume_evaluator: VolumeEvaluator | None = None) -> None:
        """Store the (or a default) :class:`VolumeEvaluator`."""
        self.volume_evaluator = volume_evaluator or VolumeEvaluator()

    def evaluate_volumes(
        self, results_dir: Path, labels_dir: Path | None = None
    ) -> list[VolumeMetrics]:
        """Score every ``ReconResult`` under ``results_dir``, streamed one at a time."""
        _require_path(results_dir, "evaluate_volumes")
        rows: list[VolumeMetrics] = []
        for result in iter_results(results_dir):
            label = _load_label_override(labels_dir, result.volume_id) if labels_dir else None
            rows.append(self.volume_evaluator.evaluate(result, label=label))
        return rows

    def evaluate_split(self, results_dir: Path, labels_dir: Path | None = None) -> AggregateMetrics:
        """Score every volume under ``results_dir`` and return the mean +/- std summary."""
        _require_path(results_dir, "evaluate_split")
        return aggregate(self.evaluate_volumes(results_dir, labels_dir=labels_dir))


def _require_path(results_dir: object, caller: str) -> None:
    if not isinstance(results_dir, str | Path):
        raise TypeError(
            f"{caller} expects a results directory Path, got {type(results_dir).__name__}. "
            "The evaluation harness scores saved ReconResults and never re-runs forward()."
        )


def _load_label_override(labels_dir: Path, volume_id: str) -> Tensor:
    path = Path(labels_dir) / f"{volume_id}.pt"
    if not path.is_file():
        raise EvalError(f"No label override file at {path} for volume {volume_id!r}.")
    # map_location="cpu": every ReconResult tensor is CPU-only by construction (recon/io.py), so
    # a label override saved from a CUDA session must not silently land on GPU (or fail to
    # deserialize at all on a CPU-only box).
    obj = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(obj, Tensor):
        raise EvalError(f"Label override at {path} is a {type(obj).__name__}, expected a Tensor.")
    return obj


__all__ = ["AggregateEvaluator", "VolumeEvaluator"]
