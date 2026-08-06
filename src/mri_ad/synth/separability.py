"""Oracle-optimistic intensity separability of the synthetic mask (Spec 009 acceptance 2).

If a global intensity threshold can already recover ``synth_mask`` from ``corrupted``, FPI
corruption is a trivial shortcut: the model (or, worse, the eval harness) could learn "flag the
brightest/darkest blob" instead of anything about anatomy. :func:`best_intensity_dice` is
deliberately generous to that shortcut (sweeps every quantile threshold, both polarities, and
reports the best) so that failing to beat it is a real, structural finding — not a fixture
artefact. Shared by the acceptance test and the real-data pre-flight guard in
``scripts/run_train.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-6


def _dice(pred: Tensor, gt: Tensor) -> float:
    pred = pred.flatten().float()
    gt = gt.flatten().float()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    intersection = (pred * gt).sum()
    return float((2.0 * intersection / (denom + _EPS)).item())


def best_intensity_dice(
    corrupted: Tensor,
    synth_mask: Tensor,
    *,
    foreground_threshold: float,
    n_thresholds: int = 256,
) -> float:
    """Best Dice any single intensity threshold (either polarity) achieves against the mask.

    Restricted to foreground voxels (``corrupted > foreground_threshold``) so a trivial
    background/foreground split cannot inflate the score. Thresholds are swept as quantiles of
    the foreground intensity distribution; for each, both ``corrupted > t`` and ``corrupted < t``
    are scored and the best Dice over all thresholds and polarities is returned — an oracle that
    picks the winner using the ground-truth mask it should not have access to. If this oracle
    cannot reach a high Dice, no simpler classifier can either.
    """
    foreground = corrupted > foreground_threshold
    gt = synth_mask > 0

    values = corrupted[foreground]
    if values.numel() == 0:
        return 0.0

    quantiles = torch.linspace(0.0, 1.0, n_thresholds)
    thresholds = torch.quantile(values.float(), quantiles)

    best = 0.0
    for t in thresholds.tolist():
        above = (corrupted > t) & foreground
        below = (corrupted < t) & foreground
        best = max(best, _dice(above, gt), _dice(below, gt))
    return best


__all__ = ["best_intensity_dice"]
