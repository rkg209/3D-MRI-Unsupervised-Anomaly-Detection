"""Loss functions. Only MSE, SSIM, SSIM+MSE, and multi-scale MSE have ever been run."""

from __future__ import annotations

import torch.nn.functional as F
from monai.losses import SSIMLoss
from torch import Tensor, nn


class SSIMMSELoss(nn.Module):
    """SSIM + MSE, unweighted sum by default — the loss behind the best prior result.

    Recovered verbatim from ``legacy/attUNET/3d-attentionunet.ipynb``::

        return (1 - ssim(pred, target, data_range=1.0)) + F.mse_loss(pred, target)

    ``monai.losses.SSIMLoss`` already returns ``1 - ssim``, so this is a weighted sum of that
    and ``F.mse_loss``. Matches ``configs/loss/mse_ssim.yaml``.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        ssim_weight: float = 1.0,
        mse_weight: float = 1.0,
        spatial_dims: int = 3,
    ) -> None:
        """Build the SSIM component once; MSE has no state."""
        super().__init__()
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self._ssim = SSIMLoss(spatial_dims=spatial_dims, data_range=data_range)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """``(B,1,16,128,128) -> ()``: ``ssim_weight * (1-ssim) + mse_weight * mse``."""
        return self.ssim_weight * self._ssim(pred, target) + self.mse_weight * F.mse_loss(
            pred, target
        )


__all__ = ["SSIMMSELoss"]
