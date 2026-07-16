"""AnoDDPM — the diffusion paradigm (Spec 002 shape/interface only; Spec 013 owns training).

Spec 002 registers this model and proves its ``(B,1,16,128,128) -> same shape`` contract on a
random-init net. Spec 013 owns the real noise schedule, ``t_noise`` tuning, sampler choice, and
training. Until Spec 013 lands a checkpoint, :meth:`load_checkpoint` always raises
:class:`CheckpointError` — there is nothing to load.

``forward`` implements the AnoDDPM partial-noise-then-denoise recipe: noise the input up to
``t_noise`` (not all the way to pure noise), then run the reverse process back to ``t=0``. The
idea (tested for real in Spec 013) is that a generative prior trained only on healthy anatomy
regenerates a healthy estimate in place of a tumor, rather than reconstructing the tumor itself.
"""

from __future__ import annotations

from pathlib import Path

import torch
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from torch import Tensor

from mri_ad import VOLUME_SHAPE
from mri_ad.models._checkpoint import load_checked_state_dict
from mri_ad.models.base import AnomalyDetectionModel, ModelCard


class DiffusionADModel(AnomalyDetectionModel):
    """Wraps ``monai.networks.nets.DiffusionModelUNet`` + ``DDPMScheduler``.

    Output range is **not** ``[0, 1]`` — it follows whatever intensity range the scheduler's
    sample space uses (matches the input's own range, since ``add_noise``/``step`` operate
    directly on it). Downstream code (``recon/``) must not assume a bounded output.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: tuple[int, ...] = (1, 1, 2),
        channels: tuple[int, ...] = (32, 64, 64),
        attention_levels: tuple[bool, ...] = (False, False, True),
        norm_num_groups: int = 32,
        num_head_channels: int = 8,
        num_train_timesteps: int = 1000,
        t_noise: int = 250,
    ) -> None:
        """Build the wrapped denoising net + scheduler. Values are provisional (Spec 013 finalizes).

        ``channels`` entries must each be a multiple of ``norm_num_groups`` (a
        ``DiffusionModelUNet`` constraint) — the default 32/32 pairing holds for the real config;
        tests downsize both together.
        """
        super().__init__()
        self.net = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=tuple(num_res_blocks),
            channels=tuple(channels),
            attention_levels=tuple(attention_levels),
            norm_num_groups=norm_num_groups,
            num_head_channels=num_head_channels,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        self.t_noise = t_noise

    def forward(self, x: Tensor) -> Tensor:
        """Partial-noise ``x`` to ``t_noise``, then denoise to a healthy estimate at ``t=0``."""
        batch = x.shape[0]
        t_noise = torch.full((batch,), self.t_noise, device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        sample = self.scheduler.add_noise(original_samples=x, noise=noise, timesteps=t_noise)
        for t in reversed(range(self.t_noise + 1)):
            timesteps = torch.full((batch,), t, device=x.device, dtype=torch.long)
            model_output = self.net(sample, timesteps)
            sample, _ = self.scheduler.step(model_output, t, sample)
        return sample

    def load_checkpoint(self, path: Path | str) -> None:
        """Always raises until Spec 013 trains weights — there is no checkpoint yet.

        Delegates into the inner ``net`` (wrapper pattern, decision #2) so the loader is ready
        the moment Spec 013 lands a real checkpoint.
        """
        load_checked_state_dict(self.net, path)

    @property
    def model_card(self) -> ModelCard:
        """Provenance and shape contract for this architecture."""
        return ModelCard(
            name="diffusion",
            param_count=sum(p.numel() for p in self.parameters()),
            input_shape=VOLUME_SHAPE,
            output_shape=VOLUME_SHAPE,
            training_data="OpenBHB healthy T1 (untrained placeholder — Spec 013)",
            training_loss="DDPM denoising (untrained — Spec 013)",
            output_activation=None,
            known_characteristics=(
                "AnoDDPM headline paradigm (Spec 013). Output is not bounded to [0,1]; tests "
                "whether a generative prior resists rebuilding the tumor the way UNet/UNETR do."
            ),
        )
