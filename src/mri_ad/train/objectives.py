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


__all__ = ["reconstruction_step"]
