"""Canonical metric definitions — the single source of truth (Spec 004, NFR-5).

No other module may compute Dice, IoU, PSNR, or SSIM. If you find yourself writing
``2 * intersection / ...`` anywhere else, import from here instead.

**Read this before touching anything below.** The prior work's metric code was wrong in ways
that invalidated its headline numbers, and the bugs are subtle enough to reintroduce by accident:

1. The ground-truth segmentation MUST be binarized (``seg > 0``). BraTS labels are ``{0,1,2,4}``.
   The prior code fed the raw multi-class map straight into the Dice numerator, so
   ``(gt * pred).sum()`` was weighted by *label magnitude* — tumor core (4) counted 4x as much
   as edema (1). That is not a Dice coefficient.
2. The segmentation MUST be resized with **nearest-neighbour** interpolation. The prior code used
   trilinear, producing fractional "labels" like 0.37.
3. Scores MUST be accumulated across **all** subjects and **all** depth-chunks. The prior code
   re-initialized its score list inside the per-subject loop and only ever scored chunk 4 of 8,
   so every reported "average" was one subject, one eighth of a volume.
4. PSNR and SSIM are **explanatory context only** (NFR-6). They are never an optimization target
   and never a headline number. Higher fidelity made detection *worse* — that is the central
   finding, and reporting PSNR as if it were a win would invert the project's thesis.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from torch import Tensor

from mri_ad.exceptions import EvalError

EPS: float = 1e-6
_SSIM_WIN_SIZE: int = 11
# 10*log10(1/EPS) — the ceiling PSNR hits once mse is clamped to EPS. Documented, not magic:
# an unclamped identical-volume mse of 0.0 would return inf, which poisons fmean/pstdev and
# is not valid JSON.
PSNR_CEILING_DB: float = 10.0 * math.log10(1.0 / EPS)


def _require_matching_shape(a: Tensor, b: Tensor, *, caller: str) -> None:
    """Guard against silent broadcasting (e.g. a size-1 trailing dim inflating Dice past 1.0)."""
    if a.shape != b.shape:
        raise EvalError(f"{caller}: shape mismatch, a={tuple(a.shape)} b={tuple(b.shape)}.")


@dataclass(frozen=True)
class VolumeMetrics:
    """Metrics for a single volume."""

    volume_id: str
    dice: float
    iou: float
    psnr: float  # context only
    ssim: float  # context only


@dataclass(frozen=True)
class AggregateMetrics:
    """Mean +/- std across the test split."""

    n_volumes: int
    dice_mean: float
    dice_std: float
    iou_mean: float
    iou_std: float
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float


class MetricsComputer:
    """Canonical implementations. Defined once, used everywhere."""

    @staticmethod
    def dice(pred: Tensor, gt: Tensor, eps: float = EPS) -> float:
        """Dice coefficient: ``2|pred & gt| / (|pred| + |gt| + eps)``.

        Args:
            pred: Binary anomaly mask.
            gt: Binary ground truth. **Must already be binarized** — pass ``(seg > 0)``.
            eps: Numerical-stability term; also defines the empty-mask convention
                (both empty -> 1.0, one empty -> 0.0).

        Note:
            ``gt`` is used **as given** — this method does not binarize it. Passing a raw
            ``{0,1,2,4}`` BraTS map here weights the numerator by label magnitude and yields an
            invalid Dice; that is the prior work's bug and the reason the docstring demands
            ``(seg > 0)``. See the module header.
        """
        _require_matching_shape(pred, gt, caller="dice")
        pred = pred.flatten().float()
        gt = gt.flatten().float()
        denom = pred.sum() + gt.sum()
        if denom == 0:  # both masks empty -> perfect agreement by convention
            return 1.0
        intersection = (pred * gt).sum()
        return float((2.0 * intersection / (denom + eps)).item())

    @staticmethod
    def iou(pred: Tensor, gt: Tensor, eps: float = EPS) -> float:
        """IoU: ``|pred & gt| / (|pred | gt| + eps)``.

        Mirrors :meth:`dice` exactly: no internal binarization (callers pass ``seg > 0``),
        ``union == 0`` -> 1.0 by convention (both masks empty), else ``inter / (union + eps)``.
        """
        _require_matching_shape(pred, gt, caller="iou")
        pred = pred.flatten().float()
        gt = gt.flatten().float()
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() - intersection
        if union == 0:  # both masks empty -> perfect agreement by convention
            return 1.0
        return float((intersection / (union + eps)).item())

    @staticmethod
    def psnr(recon: Tensor, orig: Tensor) -> float:
        """PSNR in dB, assuming a ``[0, 1]`` data range.

        **Context only — never a target.** Whole-volume MSE, clamped at ``EPS`` before the log
        so an identical-volume MSE of 0.0
        returns a finite ``PSNR_CEILING_DB`` (~60 dB) instead of ``inf`` — which would poison
        ``statistics.fmean``/``pstdev`` in :func:`aggregate` and is not valid JSON. Reconstructions
        are **not** clamped to ``[0, 1]`` before this — UNet/AttUNet are unbounded, and clamping
        would silently flatter them by hiding out-of-range error.
        """
        _require_matching_shape(recon, orig, caller="psnr")
        mse = float(((recon.float() - orig.float()) ** 2).mean().item())
        mse = max(mse, EPS)
        return 10.0 * math.log10(1.0 / mse)

    @staticmethod
    def ssim(recon: Tensor, orig: Tensor) -> float:
        """SSIM via ``monai.metrics.SSIMMetric``.

        **Context only — never a target.** Expects the ``(1, D, 128, 128)`` shape contract;
        adds a batch dim -> ``(1, 1, D, H, W)``.
        Every spatial dim must be >= the metric's window size (11) or MONAI raises an opaque
        conv error — this raises :class:`~mri_ad.exceptions.EvalError` naming the observed shape
        instead. Unit tests must use ``D >= 11``.
        """
        _require_matching_shape(recon, orig, caller="ssim")
        spatial = recon.shape[-3:]
        if any(s < _SSIM_WIN_SIZE for s in spatial):
            raise EvalError(
                f"ssim: every spatial dim must be >= {_SSIM_WIN_SIZE} (SSIMMetric's window "
                f"size), got spatial shape {tuple(spatial)}."
            )
        metric = _ssim_metric()
        value = metric(recon.float().unsqueeze(0), orig.float().unsqueeze(0))
        # MONAI's CumulativeIterationMetric buffers every call's result internally (needed for
        # its own .aggregate()). We only ever read the per-call return value, never call
        # .aggregate() on this shared instance — but leaving the buffer unreset would grow it
        # unboundedly across a 250-volume run and silently corrupt any future .aggregate() call.
        metric.reset()
        return float(value.mean().item())


@lru_cache(maxsize=1)
def _ssim_metric() -> object:
    """Build (once) the MONAI SSIM metric.

    Function-local import keeps ``monai`` off callers that only need Dice/IoU.
    """
    from monai.metrics import SSIMMetric

    return SSIMMetric(spatial_dims=3, data_range=1.0)


def aggregate(volumes: Sequence[VolumeMetrics]) -> AggregateMetrics:
    """Mean +/- population-std across ``volumes``.

    Matches ``recon/sweep.py``'s convention.
    """
    if not volumes:
        raise EvalError("aggregate() received an empty sequence of VolumeMetrics.")

    def _mean_std(values: list[float]) -> tuple[float, float]:
        return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0

    dice_mean, dice_std = _mean_std([v.dice for v in volumes])
    iou_mean, iou_std = _mean_std([v.iou for v in volumes])
    psnr_mean, psnr_std = _mean_std([v.psnr for v in volumes])
    ssim_mean, ssim_std = _mean_std([v.ssim for v in volumes])

    return AggregateMetrics(
        n_volumes=len(volumes),
        dice_mean=dice_mean,
        dice_std=dice_std,
        iou_mean=iou_mean,
        iou_std=iou_std,
        psnr_mean=psnr_mean,
        psnr_std=psnr_std,
        ssim_mean=ssim_mean,
        ssim_std=ssim_std,
    )


class LegacyMetricsComputer:
    """Bug-compatible reproductions of ``legacy/metric-uad.ipynb``. NEVER publish these numbers.

    Exists here (not in ``eval/legacy.py``) because NFR-5 says metrics are defined *only* in
    ``metrics.py`` — it does not say only one class may live there. Keeping the bug-compat
    formulas next to the correct ones makes the diff between them impossible to miss.
    """

    @staticmethod
    def dice_bugcompat(pred: Tensor, gt_raw: Tensor) -> float:
        """Prior-work Dice.

        Binarizes only ``pred`` (a no-op, kept to mirror the source), uses ``gt_raw``
        raw/fractional, and has **no eps** in the denominator — a zero denominator matches
        numpy's warn-and-``nan`` behaviour rather than the corrected 1.0 convention.
        """
        pred = (pred.flatten().float() > 0.5).float()
        mask = gt_raw.flatten().float()
        denom = pred.sum() + mask.sum()
        if denom == 0:
            return float("nan")
        intersection = (pred * mask).sum()
        return float((2.0 * intersection / denom).item())

    @staticmethod
    def iou_bugcompat(pred: Tensor, gt_raw: Tensor) -> float:
        """Prior-work IoU: binarizes neither input, adds a fixed ``1e-6``."""
        pred = pred.flatten().float()
        mask = gt_raw.flatten().float()
        intersection = (pred * mask).sum()
        union = pred.sum() + mask.sum() - intersection
        return float((intersection / (union + 1e-6)).item())


__all__ = [
    "EPS",
    "PSNR_CEILING_DB",
    "AggregateMetrics",
    "LegacyMetricsComputer",
    "MetricsComputer",
    "VolumeMetrics",
    "aggregate",
]
