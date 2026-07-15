"""UNETR reconstruction model (Spec 000 slice; registry lands in Spec 002).

Ported from ``legacy/GUI/utilities/utility.py`` (``UNETR_Reconstruction``). This is deliberately
**not** :class:`monai.networks.nets.UNETR`: the trained weights use a plain ``nn.Conv3d`` output
head (key ``out.weight``) and a final ``Sigmoid``, where MONAI's ``UNETR`` uses ``UnetOutBlock``
(key ``out.conv.conv.weight``) and no activation. Loading the weights into MONAI's class would
silently mismatch keys — the exact "load partially, score garbage" failure this project guards
against (known trap #4, spec 002).

It is still MONAI-backed (D6): it composes ``monai.networks.blocks.Unetr*`` and
``monai.networks.nets.ViT``. The module/attribute names below are kept byte-identical to the
legacy class so the checkpoint ``state_dict`` keys match under ``strict=True``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets import ViT
from torch import Tensor, nn

from mri_ad import VOLUME_SHAPE
from mri_ad.exceptions import CheckpointError
from mri_ad.models.base import AnomalyDetectionModel, ModelCard

# First bytes of a Git-LFS pointer stub. The inherited legacy .pth files are 133-byte stubs,
# not weights; loading one must fail loudly rather than degrade to random init.
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


class UNETRReconstruction(AnomalyDetectionModel):
    """ViT-backboned UNETR trained to reconstruct healthy MRI; output bounded to ``[0, 1]``."""

    def __init__(
        self,
        in_channels: int = 1,
        img_size: tuple[int, int, int] = (16, 128, 128),
        feature_size: int = 32,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        patch_size: tuple[int, int, int] = (16, 16, 16),
        norm_name: tuple | str = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """Build the ViT encoder + UNETR decoder. Defaults match ``configs/model/unetr.yaml``."""
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )

        img_size = tuple(img_size)
        patch_size = tuple(patch_size)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.feat_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=False,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = nn.Conv3d(feature_size, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _proj_feat(self, x: Tensor) -> Tensor:
        """Reshape a ViT token sequence back to a 3D feature grid."""
        x = x.view(
            x.size(0), self.feat_size[0], self.feat_size[1], self.feat_size[2], self.hidden_size
        )
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct ``x`` (``(B, 1, 16, 128, 128)``). Output is bounded to ``[0, 1]``."""
        hidden, hidden_states = self.vit(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self._proj_feat(hidden_states[3]))
        enc3 = self.encoder3(self._proj_feat(hidden_states[6]))
        enc4 = self.encoder4(self._proj_feat(hidden_states[9]))
        dec4 = self._proj_feat(hidden)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.sigmoid(self.out(out))

    def load_checkpoint(self, path: Path | str) -> None:
        """Load weights with ``strict=True``, handling both on-disk layouts.

        Accepts a bare ``state_dict`` and a ``{"model_state_dict": ...}`` wrapper — both exist
        among the real weights. Raises :class:`CheckpointError` (never falls back to
        ``strict=False``) on a missing file, a Git-LFS stub, or a key mismatch.
        """
        path = Path(path)
        if not path.is_file():
            raise CheckpointError(f"Checkpoint not found: {path.name}")
        with path.open("rb") as fh:
            if fh.read(len(_LFS_MAGIC)) == _LFS_MAGIC:
                raise CheckpointError(
                    f"Checkpoint {path.name} is a Git-LFS pointer stub, not real weights "
                    f"({path.stat().st_size} bytes). Fetch the real file (see README)."
                )
        loaded = torch.load(path, map_location="cpu")
        state = (
            loaded["model_state_dict"]
            if isinstance(loaded, dict) and "model_state_dict" in loaded
            else loaded
        )
        try:
            self.load_state_dict(state, strict=True)
        except RuntimeError as err:
            raise CheckpointError(f"State-dict mismatch loading {path.name}: {err}") from err

    @property
    def model_card(self) -> ModelCard:
        """Provenance and shape contract for this architecture."""
        return ModelCard(
            name="unetr",
            param_count=sum(p.numel() for p in self.parameters()),
            input_shape=VOLUME_SHAPE,
            output_shape=VOLUME_SHAPE,
            training_data="OpenBHB healthy T1 (reconstruction)",
            training_loss="MSE + SSIM (with augmentation)",
            output_activation="sigmoid",
            known_characteristics=(
                "Patch-based ViT features constrain reconstruction of unseen tumor tissue "
                "better than UNet/AttUNet, giving the best detection separability in prior work."
            ),
        )
