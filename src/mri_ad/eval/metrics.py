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

from dataclasses import dataclass

from torch import Tensor

EPS: float = 1e-6


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
        """
        raise NotImplementedError("Spec 004")

    @staticmethod
    def iou(pred: Tensor, gt: Tensor, eps: float = EPS) -> float:
        """IoU: ``|pred & gt| / (|pred | gt| + eps)``."""
        raise NotImplementedError("Spec 004")

    @staticmethod
    def psnr(recon: Tensor, orig: Tensor) -> float:
        """PSNR in dB, assuming a ``[0, 1]`` data range. **Context only — never a target.**"""
        raise NotImplementedError("Spec 004")

    @staticmethod
    def ssim(recon: Tensor, orig: Tensor) -> float:
        """SSIM via ``monai.metrics.SSIMMetric``. **Context only — never a target.**"""
        raise NotImplementedError("Spec 004")
