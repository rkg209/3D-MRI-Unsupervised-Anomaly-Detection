"""The one interface every model conforms to (Spec 002).

Adding a model must require **only**: a class implementing :class:`AnomalyDetectionModel`,
a ``configs/model/<name>.yaml``, and a checkpoint path. Zero changes to ``recon/``, ``eval/``,
or any other layer (FR-14 / NFR-15).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor, nn


@dataclass(frozen=True)
class ModelCard:
    """Provenance and shape contract for one architecture."""

    name: str
    param_count: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    training_data: str
    training_loss: str
    output_activation: str | None
    known_characteristics: str


class AnomalyDetectionModel(nn.Module, ABC):
    """Uniform interface over UNet / Attention-UNet / UNETR."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct a batch.

        Args:
            x: ``(B, 1, 16, 128, 128)``, float32, values in ``[0, 1]``.

        Returns:
            The reconstruction, same shape as ``x``.

        Note:
            Only UNETR bounds its output to ``[0, 1]`` (it ends in a Sigmoid). UNet and
            Attention-UNet do not, so downstream code must not assume a bounded output.
        """

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Load weights from ``path``.

        Must handle **both** checkpoint layouts found in the wild — a dict with a
        ``"model_state_dict"`` key, and a bare ``state_dict`` — and must load with
        ``strict=True``. Raise :class:`~mri_ad.exceptions.CheckpointError` on a missing file,
        a Git-LFS pointer stub, or a key mismatch. Never silently degrade to ``strict=False``.
        """

    @property
    @abstractmethod
    def model_card(self) -> ModelCard:
        """Return this architecture's model card."""
