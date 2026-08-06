"""AnoDDPM — the diffusion paradigm (Spec 013).

``forward`` implements the AnoDDPM partial-noise-then-denoise recipe: noise the input up to
``t_noise`` (not all the way to pure noise), then run the reverse process back to ``t=0``. The
idea is that a generative prior trained only on healthy anatomy regenerates a healthy estimate
in place of a tumor, rather than reconstructing the tumor itself — the sharpest test of THE
CENTRAL DOMAIN FACT (CLAUDE.md). Whether it actually does is the deliverable, not an assumption.

Two schedulers, one net (D-2): training (``objectives.ddpm_step``) always uses the ``DDPMScheduler``
sequential formulation; ``forward`` samples with a strided ``DDIMScheduler`` (``eta=0``, fully
deterministic) when ``sampler="ddim"`` — about 5x fewer network calls than the sequential DDPM
reverse process, which matters because eval means one forward pass per depth-chunk per volume
per sweep point. ``sampler="ddpm"`` is kept for tests/debugging: it walks every integer timestep
down to 0 sequentially.

Deviation from the AnoDDPM paper (D-1, documented in ``known_characteristics``): Gaussian noise
only, not simplex/low-frequency noise. MONAI ships no simplex-noise sampler and hand-rolling one
fights CLAUDE.md D6 (adopt MONAI, don't hand-roll) for a one-off ablation outside this spec's
scope. A natural future extension, not a silent gap.
"""

from __future__ import annotations

from pathlib import Path

import torch
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler
from torch import Tensor

from mri_ad import VOLUME_SHAPE
from mri_ad.models._checkpoint import load_checked_state_dict
from mri_ad.models.base import AnomalyDetectionModel, ModelCard


class DiffusionADModel(AnomalyDetectionModel):
    """Wraps ``monai.networks.nets.DiffusionModelUNet`` + a DDPM/DDIM scheduler pair.

    ``forward`` accepts and returns ``[0, 1]`` like every other model in the registry (D-4): the
    rescale to the ``[-1, 1]`` range the noise schedule and ``clip_sample`` assume is confined
    entirely inside this class, so ``recon/`` never has to special-case the diffusion row.
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
        sampler: str = "ddim",
        num_inference_steps: int = 50,
        sample_seed: int = 0,
    ) -> None:
        """Build the wrapped denoising net + scheduler pair.

        ``channels`` entries must each be a multiple of ``norm_num_groups`` (a
        ``DiffusionModelUNet`` constraint) — tests downsize both together (e.g. ``channels=(4,
        8)``, ``norm_num_groups=4``). ``sampler`` selects the reverse process ``forward`` uses:
        ``"ddim"`` (default, D-2) is strided and deterministic at ``eta=0``; ``"ddpm"`` walks
        every timestep down to 0 sequentially.
        """
        super().__init__()
        if any(c % norm_num_groups != 0 for c in channels):
            raise ValueError(
                f"channels {tuple(channels)} must each be a multiple of norm_num_groups "
                f"({norm_num_groups}) — a DiffusionModelUNet constraint."
            )
        if sampler not in ("ddim", "ddpm"):
            raise ValueError(f"sampler must be 'ddim' or 'ddpm', got {sampler!r}.")

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
        # Training objective (D-7's denoising loss) always uses the plain sequential DDPM
        # formulation — `objectives.ddpm_step` reads this scheduler, never `infer_scheduler`.
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        self.infer_scheduler = (
            DDIMScheduler(num_train_timesteps=num_train_timesteps)
            if sampler == "ddim"
            else DDPMScheduler(num_train_timesteps=num_train_timesteps)
        )
        self.sampler = sampler
        self.num_train_timesteps = num_train_timesteps
        self.t_noise = t_noise
        self.num_inference_steps = num_inference_steps
        self.sample_seed = sample_seed

    def _reverse_timesteps(self, device: torch.device) -> list[int]:
        """Descending timesteps to walk from ``t_noise`` down to (and including) 0."""
        if self.sampler == "ddim":
            self.infer_scheduler.set_timesteps(self.num_inference_steps, device=device)
            all_steps = [int(t) for t in self.infer_scheduler.timesteps.tolist()]
            steps = [t for t in all_steps if t <= self.t_noise]
            # `t_noise` may fall strictly between two strided steps; always denoise starting
            # from exactly the timestep we noised to, or the residual reflects the wrong SNR.
            if not steps or steps[0] != self.t_noise:
                steps = [self.t_noise, *steps]
            return steps
        return list(range(self.t_noise, -1, -1))

    def forward(self, x: Tensor) -> Tensor:
        """Partial-noise ``x`` to ``t_noise``, then denoise to a healthy estimate at ``t=0``.

        ``x`` and the return value are both ``[0, 1]`` (D-4); the noise schedule operates in
        ``[-1, 1]`` internally. The one stochastic op — the initial noise draw — is seeded from
        ``sample_seed`` (D-6), so with the deterministic DDIM step (``eta=0``) the whole reverse
        process is bitwise reproducible across calls.
        """
        batch = x.shape[0]
        x_pm1 = x * 2.0 - 1.0

        # A CPU generator, regardless of `x`'s device: torch.randn(..., generator=g) requires g's
        # device to match the created tensor's, and CPU is the one device every run has.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.sample_seed)
        noise = torch.randn(x_pm1.shape, generator=generator, dtype=x_pm1.dtype).to(x.device)

        t_noise_batch = torch.full((batch,), self.t_noise, device=x.device, dtype=torch.long)
        sample = self.infer_scheduler.add_noise(
            original_samples=x_pm1, noise=noise, timesteps=t_noise_batch
        )

        for t in self._reverse_timesteps(x.device):
            timesteps = torch.full((batch,), t, device=x.device, dtype=torch.long)
            model_output = self.net(sample, timesteps)
            step_kwargs = {"eta": 0.0} if self.sampler == "ddim" else {}
            sample, _ = self.infer_scheduler.step(model_output, t, sample, **step_kwargs)

        return ((sample.clamp(-1.0, 1.0) + 1.0) / 2.0).to(x.dtype)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load weights with ``strict=True`` into the whole wrapper (D-5), not just ``net``.

        ``CheckpointWriter`` saves ``model.state_dict()`` of the wrapper, so on-disk keys are
        ``net.*`` (the scheduler pair contributes nothing — ``DDPMScheduler``/``DDIMScheduler``
        hold their alphas/betas as plain attributes, not buffers, so ``state_dict()`` is empty).
        Loading into ``self.net`` would raise on the ``net.``-prefixed keys; matches the
        ``unetr``/``msa_unetr`` precedent of loading into ``self``.
        """
        load_checked_state_dict(self, path)

    @property
    def model_card(self) -> ModelCard:
        """Provenance and shape contract for this architecture."""
        return ModelCard(
            name="diffusion",
            param_count=sum(p.numel() for p in self.parameters()),
            input_shape=VOLUME_SHAPE,
            output_shape=VOLUME_SHAPE,
            training_data="OpenBHB healthy T1, from scratch (Spec 013)",
            training_loss="DDPM denoising: MSE(predicted noise, true noise), t ~ U[0, T)",
            output_activation=None,
            known_characteristics=(
                "AnoDDPM headline paradigm (Spec 013): partial-noise-to-t_noise then a "
                "deterministic strided DDIM reverse process (D-2), t_noise val-tuned via "
                "run_tnoise_sweep.py. Gaussian noise only, not the published AnoDDPM simplex "
                "noise (D-1, documented deviation, future extension). Output is rescaled back "
                "to [0,1] (D-4) so it is directly comparable to every other model's residual."
            ),
        )
