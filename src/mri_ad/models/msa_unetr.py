"""Multi-scale attention UNETR variant (Spec 012, stretch).

Subclasses :class:`~mri_ad.models.unetr.UNETRReconstruction` rather than copy-pasting it, so every
shared submodule keeps an identical attribute name and therefore an identical ``state_dict`` key.
That makes the warm-start in :meth:`MultiScaleAttentionUNETR.load_from_unetr` a pure key
partition (shared keys load from the plain-UNETR checkpoint; the four new gate prefixes stay at
their zero-init identity) rather than a hand-maintained key remap.

Gated at all four UNETR skip connections, each gate optionally conditioned on the deepest ViT
feature (``multiscale_context``) — the two-part "multi-scale" axis from the spec (D5): gating
*and* global-context conditioning at every scale, not just uniform ViT features.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
from torch import Tensor

from mri_ad import VOLUME_SHAPE
from mri_ad.exceptions import CheckpointError
from mri_ad.models._checkpoint import LFS_MAGIC, load_checked_state_dict
from mri_ad.models.attention_gate import MultiScaleAttentionGate
from mri_ad.models.base import ModelCard
from mri_ad.models.unetr import UNETRReconstruction

# scale -> (f_g channels, f_x channels), read off the shapes UNETRReconstruction.forward already
# produces: g is the decoder's pre-upsample input at that step, x is the skip it will be
# concatenated with. See src/mri_ad/models/unetr.py:165-177.
_SCALE_CHANNELS = {
    4: ("hidden_size", "feature_size_8"),
    3: ("feature_size_8", "feature_size_4"),
    2: ("feature_size_4", "feature_size_2"),
    1: ("feature_size_2", "feature_size_1"),
}


class MultiScaleAttentionUNETR(UNETRReconstruction):
    """UNETR with multi-scale attention gates on its four skip connections.

    Bitwise identical to plain :class:`UNETRReconstruction` at construction time (the gate's
    ``psi`` conv is zero-initialized, D3) — see
    ``tests/test_msa_unetr.py::test_identity_at_init_matches_plain_unetr``.
    """

    GATE_PREFIXES: ClassVar[tuple[str, ...]] = ("gate1.", "gate2.", "gate3.", "gate4.")

    def __init__(
        self,
        *,
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
        gate_reduction: int = 2,
        multiscale_context: bool = True,
        gated_scales: tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        """Build plain UNETR, then attach gates. ``**unetr_kwargs`` reproduces every shared arg."""
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            patch_size=patch_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
        )
        self.gate_reduction = gate_reduction
        self.multiscale_context = multiscale_context
        self.gated_scales = tuple(gated_scales)

        channel_of = {
            "hidden_size": hidden_size,
            "feature_size_8": feature_size * 8,
            "feature_size_4": feature_size * 4,
            "feature_size_2": feature_size * 2,
            "feature_size_1": feature_size,
        }
        context_channels = hidden_size if multiscale_context else None

        for scale, (g_key, x_key) in _SCALE_CHANNELS.items():
            gate = None
            if scale in self.gated_scales:
                f_x = channel_of[x_key]
                gate = MultiScaleAttentionGate(
                    spatial_dims=3,
                    f_g=channel_of[g_key],
                    f_x=f_x,
                    f_int=max(f_x // gate_reduction, 1),
                    f_context=context_channels,
                    norm_name=norm_name,
                )
            setattr(self, f"gate{scale}", gate)

    def _gate(self, scale: int, *, g: Tensor, x: Tensor, context: Tensor | None) -> Tensor:
        gate = getattr(self, f"gate{scale}")
        if gate is None:
            return x
        return gate(g, x, context)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct ``x`` (``(B, 1, 16, 128, 128)``). Output is bounded to ``[0, 1]``."""
        hidden, hidden_states = self.vit(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self._proj_feat(hidden_states[3]))
        enc3 = self.encoder3(self._proj_feat(hidden_states[6]))
        enc4 = self.encoder4(self._proj_feat(hidden_states[9]))
        dec4 = self._proj_feat(hidden)
        ctx = dec4 if self.multiscale_context else None

        dec3 = self.decoder5(dec4, self._gate(4, g=dec4, x=enc4, context=ctx))
        dec2 = self.decoder4(dec3, self._gate(3, g=dec3, x=enc3, context=ctx))
        dec1 = self.decoder3(dec2, self._gate(2, g=dec2, x=enc2, context=ctx))
        out = self.decoder2(dec1, self._gate(1, g=dec1, x=enc1, context=ctx))
        return self.sigmoid(self.out(out))

    def load_checkpoint(self, path: Path | str) -> None:
        """Load an ``msa_unetr`` checkpoint (its own, trained gates included). ``strict=True``."""
        load_checked_state_dict(self, path)

    def load_from_unetr(self, path: Path | str) -> None:
        """Warm-start the shared UNETR submodules from a plain ``unetr`` checkpoint (D4).

        The gates keep their zero-init identity — they are not in the checkpoint being loaded.
        Raises :class:`CheckpointError` unless the checkpoint's key set is *exactly* this model's
        key set minus the four gate prefixes; never ``strict=False``. A checkpoint that already
        contains gate keys (i.e. is itself an ``msa_unetr`` checkpoint) also raises — it belongs
        in :meth:`load_checkpoint`, not here.
        """
        path = Path(path)
        if not path.is_file():
            raise CheckpointError(f"Checkpoint not found: {path.name}")
        with path.open("rb") as fh:
            if fh.read(len(LFS_MAGIC)) == LFS_MAGIC:
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

        self_keys = set(self.state_dict().keys())
        gate_keys = {key for key in self_keys if key.startswith(self.GATE_PREFIXES)}
        expected_ckpt_keys = self_keys - gate_keys
        ckpt_keys = set(state.keys())
        if ckpt_keys != expected_ckpt_keys:
            missing = sorted(expected_ckpt_keys - ckpt_keys)
            unexpected = sorted(ckpt_keys - expected_ckpt_keys)
            raise CheckpointError(
                f"load_from_unetr key mismatch loading {path.name}: missing={missing} "
                f"unexpected={unexpected}. A checkpoint that includes gate keys is an msa_unetr "
                "checkpoint — load it with load_checkpoint(), not load_from_unetr()."
            )

        own_state = self.state_dict()
        own_state.update(state)
        self.load_state_dict(own_state, strict=True)

    @property
    def model_card(self) -> ModelCard:
        """Provenance and shape contract for this architecture."""
        return ModelCard(
            name="msa_unetr",
            param_count=sum(p.numel() for p in self.parameters()),
            input_shape=VOLUME_SHAPE,
            output_shape=VOLUME_SHAPE,
            training_data=(
                "OpenBHB healthy T1, FPI synthetic-anomaly restoration (Spec 009 objective)"
            ),
            training_loss="MSE + SSIM (fine-tune from plain UNETR)",
            output_activation="sigmoid",
            known_characteristics=(
                "STRETCH (Spec 012). Bitwise identical to plain UNETR at initialization "
                "(zero-init gate psi conv, D3): any scoring difference is attributable to the "
                "gates, not to a broken init. GPU training is gated on Spec 005 showing the "
                "fidelity/detection anti-correlation is scale-dependent; as of this build the "
                "gate is closed and this row renders n/a in the 005 matrix."
            ),
        )
