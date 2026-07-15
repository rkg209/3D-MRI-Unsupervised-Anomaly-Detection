"""Loud batch validation (Spec 001).

A shape or intensity regression must fail at dataloader init, not silently corrupt a Dice score
four specs later. ``DataValidator`` is the one place that contract is checked; the error message
always names the *observed* shape and range, never just the expected one.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from mri_ad import VOLUME_SHAPE
from mri_ad.exceptions import DescriptiveValidationError


class DataValidator:
    """Validates that image batches obey the shape invariant, dtype, and value range."""

    @staticmethod
    def validate_batch(batch: Tensor) -> None:
        """Assert ``batch`` is ``(B, *VOLUME_SHAPE)`` float32 in ``[0, 1]``.

        Raises:
            DescriptiveValidationError: naming the observed shape and ``[min, max]`` range.
        """
        expected_item_shape = VOLUME_SHAPE
        if batch.ndim != 5 or tuple(batch.shape[1:]) != expected_item_shape:
            raise DescriptiveValidationError(
                f"Expected batch shape (B, {', '.join(map(str, expected_item_shape))}), "
                f"got {tuple(batch.shape)}."
            )
        if batch.dtype != torch.float32:
            raise DescriptiveValidationError(
                f"Expected dtype torch.float32, got {batch.dtype} (shape {tuple(batch.shape)})."
            )
        vmin, vmax = float(batch.min()), float(batch.max())
        if vmin < 0.0 or vmax > 1.0:
            raise DescriptiveValidationError(
                f"Expected values in [0, 1], observed range [{vmin}, {vmax}] "
                f"(shape {tuple(batch.shape)})."
            )

    @staticmethod
    def wrap_loader(loader: DataLoader) -> Iterator:
        """Yield from ``loader``, validating the image tensor of the first batch it produces."""
        for i, batch in enumerate(loader):
            image = batch["image"] if isinstance(batch, dict) else batch
            if i == 0:
                DataValidator.validate_batch(image)
            yield batch


__all__ = ["DataValidator"]
