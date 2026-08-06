"""Tests for ``SSIMMSELoss`` (Spec 009 prerequisite, R11).

Guards against MONAI's ``SSIMLoss`` signature drifting across versions (``data_range`` as a
constructor arg vs a forward arg) silently producing a wrong objective — the only defense is
behavioral: identical inputs score ~0, and moving ``pred`` toward ``target`` monotonically lowers
the loss.
"""

from __future__ import annotations

import torch

from mri_ad.losses import SSIMMSELoss


def _volume(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 1, 16, 32, 32), generator=g)


def test_identical_inputs_score_near_zero() -> None:
    x = _volume(0)
    loss = SSIMMSELoss()
    value = loss(x, x)
    assert value.item() < 1e-5


def test_monotone_decrease_as_pred_approaches_target() -> None:
    target = _volume(1)
    noise = _volume(2)
    loss = SSIMMSELoss()

    losses = []
    for alpha in (1.0, 0.75, 0.5, 0.25, 0.0):
        pred = alpha * noise + (1 - alpha) * target
        losses.append(loss(pred, target).item())

    for earlier, later in zip(losses, losses[1:], strict=False):
        assert later < earlier


def test_weights_scale_each_component() -> None:
    pred = _volume(3)
    target = _volume(4)

    combined = SSIMMSELoss(ssim_weight=1.0, mse_weight=1.0)(pred, target)
    mse_only = SSIMMSELoss(ssim_weight=0.0, mse_weight=1.0)(pred, target)
    ssim_only = SSIMMSELoss(ssim_weight=1.0, mse_weight=0.0)(pred, target)

    assert torch.isclose(combined, mse_only + ssim_only, atol=1e-5)


def test_output_is_scalar() -> None:
    pred = _volume(5)
    target = _volume(6)
    value = SSIMMSELoss()(pred, target)
    assert value.dim() == 0
