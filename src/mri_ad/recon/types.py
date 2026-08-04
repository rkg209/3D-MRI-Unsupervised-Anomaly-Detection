"""The reconstruction engine's output contract (Spec 003)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass(frozen=True)
class ReconResult:
    """Everything the engine produces for one volume.

    All tensors are ``(1, D, 128, 128)``, float32, on CPU, where ``D`` is the volume's full
    (chunk-padded) depth — 160 for BraTS (155 slices zero-padded to 10x16). ``(1,16,128,128)``
    remains the *model-boundary* invariant (one chunk, one forward pass); this is the
    already-reassembled, full-depth artifact. This object is what gets persisted to
    ``artifacts/results/recon/<run_id>/<volume_id>.pt`` and is the **only** thing the evaluation
    harness reads — the harness never calls ``model.forward()``.
    """

    volume_id: str
    original: Tensor
    reconstruction: Tensor
    residual: Tensor  # abs(original - reconstruction)
    anomaly_mask: Tensor  # binary
    ground_truth: Tensor | None = None  # binarized (seg > 0), nearest-resized; None for OpenBHB


class ThresholdStrategy(Protocol):
    """Turns a continuous residual map into a binary anomaly mask.

    This is the **only** place in the codebase where thresholding happens. No other module
    may binarize a residual. The strategy is injected via ``configs/threshold/``.
    """

    def __call__(self, residual: Tensor) -> Tensor:
        """Return a binary mask the same shape as ``residual``."""
        ...
