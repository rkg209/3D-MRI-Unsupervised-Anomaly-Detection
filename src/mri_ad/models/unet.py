"""MONAI UNet — kept as a prior-work reference row (Spec 002).

Not a headline model (see CLAUDE.md D3): the prior work already showed UNet/AttUNet reconstruct
tumors too faithfully to flag them. Kept for the comparison table, not as a target of optimization.
"""

from __future__ import annotations

from pathlib import Path

from monai.networks.nets import UNet
from torch import Tensor

from mri_ad import VOLUME_SHAPE
from mri_ad.models._checkpoint import load_checked_state_dict
from mri_ad.models.base import AnomalyDetectionModel, ModelCard


class UNetModel(AnomalyDetectionModel):
    """Wraps ``monai.networks.nets.UNet``. Output is unbounded — no final activation."""

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        norm: str = "BATCH",
        dropout: float = 0.2,
    ) -> None:
        """Build the wrapped ``UNet``. Defaults match ``configs/model/unet.yaml``."""
        super().__init__()
        self.net = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(channels),
            strides=tuple(strides),
            num_res_units=num_res_units,
            norm=norm,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct ``x`` (``(B, 1, 16, 128, 128)``). Output is unbounded."""
        return self.net(x)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load weights into the inner ``net`` (bare, unprefixed legacy keys).

        The trained weights were saved from a bare ``monai.networks.nets.UNet``, so loading
        into ``self`` (prefix ``net.``) would fail ``strict=True``. Delegate to the inner net.
        """
        load_checked_state_dict(self.net, path)

    @property
    def model_card(self) -> ModelCard:
        """Provenance and shape contract for this architecture."""
        return ModelCard(
            name="unet",
            param_count=sum(p.numel() for p in self.parameters()),
            input_shape=VOLUME_SHAPE,
            output_shape=VOLUME_SHAPE,
            training_data="OpenBHB healthy T1 (reconstruction)",
            training_loss="MSE",
            output_activation=None,
            known_characteristics=(
                "Reference row (prior-work paradigm). Reconstructs tumor tissue too faithfully "
                "to flag it well — the central failure this project studies."
            ),
        )
