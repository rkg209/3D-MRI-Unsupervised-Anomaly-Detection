"""The training loop (Spec 009): fine-tune UNETR to restore FPI-corrupted volumes.

Peer of ``data``/``models``/``synth`` (D6) — this package never imports ``mri_ad.recon`` or
``mri_ad.eval``. Spec 013's from-scratch DDPM loop reuses everything here but the ``StepFn``.
"""

from __future__ import annotations

from mri_ad.train.checkpointing import CheckpointWriter
from mri_ad.train.loop import EpochRecord, StepFn, Trainer, TrainSummary
from mri_ad.train.objectives import ddpm_step, reconstruction_step
from mri_ad.train.splits import TrainingIds, resolve_training_ids

__all__ = [
    "CheckpointWriter",
    "EpochRecord",
    "StepFn",
    "Trainer",
    "TrainSummary",
    "TrainingIds",
    "ddpm_step",
    "reconstruction_step",
    "resolve_training_ids",
]
