"""Multi-scale attention gate (Spec 012, stretch).

A single gate applied to one UNETR skip connection, optionally conditioned on a shared global
context feature (the deepest ViT token grid) so every scale attends with knowledge of the whole
volume, not just its local encoder/decoder pair — the "multi-scale" half of the spec's hypothesis.

Uses ``instance`` normalization (matching ``UnetrBasicBlock``'s own ``norm_name`` convention),
never MONAI ``AttentionBlock``'s hardcoded ``BatchNorm``: batch statistics over a batch of two
3D volumes are noise, and eval-mode running stats would diverge from train-mode stats — a silent
train/eval skew this project's determinism rules forbid.
"""

from __future__ import annotations

import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.utils import get_norm_layer
from torch import Tensor, nn


def _projection(
    in_channels: int, out_channels: int, spatial_dims: int, norm_name: tuple | str
) -> nn.Sequential:
    return nn.Sequential(
        Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        ),
        get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels),
    )


class MultiScaleAttentionGate(nn.Module):
    """Additive attention gate on one skip connection, optionally conditioned on global context.

    ``gate = 2 * sigmoid(psi(relu(W_g(up(g)) + W_x(x) [+ W_c(up(context))])))``
    ``out  = x * gate``

    ``g`` (the decoder's pre-upsample feature) is coarser than ``x`` (the skip connection it will
    eventually be concatenated with inside the ``UnetrUpBlock`` that consumes this gate's output),
    so ``g`` is trilinearly upsampled to ``x``'s spatial shape before projection — mirroring how
    MONAI's own ``AttentionLayer`` upsamples its gating signal before ``AttentionBlock``.

    ``psi`` is zero-initialized (weight *and* bias), so ``psi(...)`` is exactly ``0`` for *any*
    input — the gate evaluates to exactly ``1.0`` and ``out`` is bitwise identical to ``x`` at
    construction time, regardless of how ``W_g``/``W_x``/``W_c`` are initialized.
    """

    def __init__(
        self,
        *,
        spatial_dims: int = 3,
        f_g: int,
        f_x: int,
        f_int: int,
        f_context: int | None = None,
        norm_name: tuple | str = "instance",
    ) -> None:
        """Build the gate. ``f_context`` declares (and requires) the context branch."""
        super().__init__()
        self.f_context = f_context
        self.W_g = _projection(f_g, f_int, spatial_dims, norm_name)
        self.W_x = _projection(f_x, f_int, spatial_dims, norm_name)
        self.W_c = (
            _projection(f_context, f_int, spatial_dims, norm_name)
            if f_context is not None
            else None
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi = Convolution(
            spatial_dims=spatial_dims,
            in_channels=f_int,
            out_channels=1,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )
        nn.init.zeros_(self.psi.conv.weight)
        nn.init.zeros_(self.psi.conv.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: Tensor, x: Tensor, context: Tensor | None = None) -> Tensor:
        """Gate ``x`` using signal ``g``, optionally conditioned on ``context``.

        Raises ``ValueError`` if the presence of ``context`` disagrees with whether this gate was
        built with a context branch — never silently ignores an argument.
        """
        if (self.W_c is None) != (context is None):
            got = "a tensor" if context is not None else "None"
            raise ValueError(
                "MultiScaleAttentionGate context mismatch: "
                f"gate built with f_context={self.f_context!r} but forward() received "
                f"context={got}."
            )
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="trilinear", align_corners=False)
        summed = self.W_g(g) + self.W_x(x)
        if self.W_c is not None:
            context = F.interpolate(
                context, size=x.shape[2:], mode="trilinear", align_corners=False
            )
            summed = summed + self.W_c(context)
        gate = 2 * self.sigmoid(self.psi(self.relu(summed)))
        return x * gate
