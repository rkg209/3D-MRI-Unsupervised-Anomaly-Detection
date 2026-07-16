"""MONAI AttentionUnet — kept as a prior-work reference row (Spec 002).

Not a headline model (see CLAUDE.md D3): the prior work already showed UNet/AttUNet reconstruct
tumors too faithfully to flag them. Kept for the comparison table, not as a target of optimization.
"""

from __future__ import annotations

from pathlib import Path

from monai.networks.nets import AttentionUnet
from torch import Tensor

from mri_ad import VOLUME_SHAPE
from mri_ad.models._checkpoint import load_checked_state_dict
from mri_ad.models.base import AnomalyDetectionModel, ModelCard


class AttentionUNetModel(AnomalyDetectionModel):
    """Wraps ``monai.networks.nets.AttentionUnet``. Output is unbounded."""

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        kernel_size: int = 3,
    ) -> None:
        """Build the wrapped AttentionUnet. Defaults match configs/model/attention_unet.yaml."""
        super().__init__()
        self.net = AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(channels),
            strides=tuple(strides),
            kernel_size=kernel_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct ``x`` (``(B, 1, 16, 128, 128)``). Output is unbounded."""
        return self.net(x)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load weights into the inner ``net`` (bare, unprefixed legacy keys).

        The trained weights were saved from a bare ``monai.networks.nets.AttentionUnet``, so
        loading into ``self`` (prefix ``net.``) would fail ``strict=True``. Delegate to the net.
        """
        load_checked_state_dict(self.net, path)

    @property
    def model_card(self) -> ModelCard:
        """Provenance and shape contract for this architecture."""
        return ModelCard(
            name="attention_unet",
            param_count=sum(p.numel() for p in self.parameters()),
            input_shape=VOLUME_SHAPE,
            output_shape=VOLUME_SHAPE,
            training_data="OpenBHB healthy T1 (reconstruction)",
            training_loss="SSIM + MSE",
            output_activation=None,
            known_characteristics=(
                "Reference row (prior-work paradigm). Attention gates do not resolve the "
                "reconstruct-the-tumor failure mode any better than plain UNet."
            ),
        )
