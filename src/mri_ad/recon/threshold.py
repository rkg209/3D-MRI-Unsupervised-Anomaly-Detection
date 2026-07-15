"""Residual -> binary mask thresholding (Spec 000 slice; strategies + sweep land in Spec 003).

This is the **only** place a residual is binarized (recon/ owns thresholding).
``FixedPercentileThreshold`` is the hardcoded operating point for the vertical slice: flag the top
``(100 - percentile)%`` of residual voxels. It fixes the flagged-voxel fraction by construction —
a *deliberate* deviation from the prior work's absolute ``residual > 0.1``. See
``configs/threshold/percentile.yaml``.
"""

from __future__ import annotations

import torch
from torch import Tensor


class FixedPercentileThreshold:
    """Flag the top ``(100 - percentile)%`` of residual voxels as anomalous.

    Conforms to the :class:`~mri_ad.recon.types.ThresholdStrategy` protocol.
    """

    def __init__(self, percentile: float = 95.0) -> None:
        """Store the percentile cut (e.g. ``95.0`` flags the top 5% of residual voxels)."""
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}.")
        self.percentile = float(percentile)

    def __call__(self, residual: Tensor) -> Tensor:
        """Return a ``{0., 1.}`` mask, same shape as ``residual``, over the whole volume."""
        cutoff = torch.quantile(residual.flatten().float(), self.percentile / 100.0)
        return (residual > cutoff).to(residual.dtype)
