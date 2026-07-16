"""Model registry: UNet / Attention-UNet / UNETR / Diffusion behind one interface (Spec 002)."""

from __future__ import annotations

from mri_ad.models.attention_unet import AttentionUNetModel
from mri_ad.models.base import AnomalyDetectionModel, ModelCard
from mri_ad.models.diffusion import DiffusionADModel
from mri_ad.models.registry import ModelRegistry, build_default_registry
from mri_ad.models.unet import UNetModel
from mri_ad.models.unetr import UNETRReconstruction

__all__ = [
    "AnomalyDetectionModel",
    "ModelCard",
    "ModelRegistry",
    "build_default_registry",
    "UNetModel",
    "AttentionUNetModel",
    "UNETRReconstruction",
    "DiffusionADModel",
]
