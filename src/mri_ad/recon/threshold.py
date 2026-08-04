"""Residual -> binary mask thresholding strategies (Spec 003).

This is the **only** place a residual is binarized (``recon/`` owns thresholding). Three
strategies conform to the :class:`~mri_ad.recon.types.ThresholdStrategy` protocol and are
injected via ``configs/threshold/*.yaml`` — swapping one for another is a config change, never
a code change (acceptance test 4).
"""

from __future__ import annotations

import torch
from torch import Tensor


class FixedPercentileThreshold:
    """Flag the top ``(100 - percentile)%`` of residual voxels as anomalous.

    Fixes the flagged-voxel count *exactly* by rank selection rather than ``residual > quantile``
    — the latter flags an unpredictable count under tied/duplicate values (e.g. large regions of
    zero residual), which fails acceptance test 3. Sorting ~2.6M floats per volume is negligible
    next to a model forward pass.
    """

    def __init__(self, percentile: float = 95.0) -> None:
        """Store the percentile cut (e.g. ``95.0`` flags the top 5% of residual voxels)."""
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}.")
        self.percentile = float(percentile)

    def __call__(self, residual: Tensor) -> Tensor:
        """Return a ``{0., 1.}`` mask, same shape as ``residual``, over the whole volume."""
        flat = residual.flatten().float()
        n = flat.numel()
        k = int(round(n * (100.0 - self.percentile) / 100.0))
        mask = torch.zeros(n, dtype=residual.dtype)
        if k > 0:
            idx = torch.argsort(flat, descending=True, stable=True)[:k]
            mask[idx] = 1.0
        return mask.view_as(residual)


class AbsoluteThreshold:
    """Flag every voxel with residual strictly greater than a fixed value.

    The strategy that actually reproduces the prior work's hardcoded ``residual > 0.1``. Unlike
    :class:`FixedPercentileThreshold`, the flagged-voxel *count* varies with how badly the model
    reconstructs a given volume — which is the actual anomaly signal.
    """

    def __init__(self, value: float = 0.1) -> None:
        """Store the absolute cutoff."""
        self.value = float(value)

    def __call__(self, residual: Tensor) -> Tensor:
        """Return a ``{0., 1.}`` mask, same shape as ``residual``."""
        return (residual > self.value).to(residual.dtype)


class OtsuThreshold:
    """Otsu's method: pick the cutoff that maximizes inter-class variance of the histogram.

    Implemented in pure torch (``torch.histc`` over the data's own ``[min, max]``) rather than
    ``skimage.filters.threshold_otsu``, so it handles **unbounded residuals** correctly. Only
    UNETR bounds its reconstruction output to ``[0, 1]`` (final Sigmoid); UNet/AttUNet do not, so
    residuals can exceed 1.0 and a fixed ``[0, 1]`` binning would silently clip them.
    """

    def __init__(self, bins: int = 256) -> None:
        """Store the histogram resolution."""
        if bins < 2:
            raise ValueError(f"bins must be >= 2, got {bins}.")
        self.bins = int(bins)

    def __call__(self, residual: Tensor) -> Tensor:
        """Return a ``{0., 1.}`` mask, same shape as ``residual``."""
        flat = residual.flatten().float()
        lo, hi = float(flat.min()), float(flat.max())
        if hi <= lo:  # constant residual: nothing to separate
            return torch.zeros_like(residual)

        hist = torch.histc(flat, bins=self.bins, min=lo, max=hi).float()
        hist = hist / hist.sum()

        bin_width = (hi - lo) / self.bins
        centers = lo + bin_width * (torch.arange(self.bins, dtype=torch.float32) + 0.5)

        weight_bg = torch.cumsum(hist, dim=0)
        weight_fg = 1.0 - weight_bg

        mean_cum = torch.cumsum(hist * centers, dim=0)
        total_mean = mean_cum[-1]
        mean_bg = mean_cum / weight_bg.clamp_min(1e-12)
        mean_fg = (total_mean - mean_cum) / weight_fg.clamp_min(1e-12)

        between_class_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        # Bins where one class is empty can't be a valid split point.
        valid = (weight_bg > 0) & (weight_fg > 0)
        between_class_var = torch.where(
            valid, between_class_var, torch.full_like(between_class_var, float("-inf"))
        )

        best_bin = int(torch.argmax(between_class_var))
        cutoff = float(centers[best_bin])
        return (residual > cutoff).to(residual.dtype)


__all__ = ["AbsoluteThreshold", "FixedPercentileThreshold", "OtsuThreshold"]
