"""Training objectives (Spec 009) — the ``StepFn`` implementations ``Trainer.fit()`` runs.

Kept separate from ``loop.py`` so Spec 013's from-scratch DDPM loop adds a sibling
``ddpm_step`` here and reuses everything in ``loop.py``/``checkpointing.py``/``splits.py``
unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def reconstruction_step(
    model: nn.Module, batch: dict[str, Tensor], device: torch.device, *, criterion: nn.Module
) -> Tensor:
    """``criterion(model(corrupted), healthy)`` — the Spec 009 restoration objective.

    ``criterion`` is bound via ``functools.partial`` at construction time so this still matches
    the ``StepFn`` signature (``(model, batch, device) -> Tensor``).
    """
    return criterion(model(batch["corrupted"].to(device)), batch["healthy"].to(device))


def ddpm_step(
    model: nn.Module, batch: dict[str, Tensor], device: torch.device, *, criterion: nn.Module
) -> Tensor:
    """DDPM denoising loss (Spec 013, D-7): ``criterion(predicted_noise, true_noise)``, `t~U[0,T)`.

    From-scratch objective for :class:`~mri_ad.models.diffusion.DiffusionADModel` — there is no
    corruption to invert, so this reads ``batch["image"]`` (``OpenBHBDataset``'s key), not
    :func:`reconstruction_step`'s ``"corrupted"``/``"healthy"``. Rescales to ``[-1, 1]`` to match
    the noise schedule's convention (D-4, mirroring ``DiffusionADModel.forward``), draws ``t`` and
    the noise, and calls the wrapped net directly — bypassing ``model.forward``'s reverse-process
    loop, which has no role during training.
    """
    x = batch["image"].to(device) * 2.0 - 1.0

    scheduler = model.scheduler
    batch_size = x.shape[0]
    timesteps = torch.randint(
        0, scheduler.num_train_timesteps, (batch_size,), device=device, dtype=torch.long
    )
    noise = torch.randn_like(x)
    noisy = scheduler.add_noise(original_samples=x, noise=noise, timesteps=timesteps)
    predicted_noise = model.net(noisy, timesteps)
    return criterion(predicted_noise, noise)


__all__ = ["ddpm_step", "reconstruction_step"]
