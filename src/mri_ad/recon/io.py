"""Persistence for :class:`~mri_ad.recon.types.ReconResult` (Spec 003).

Serialized as a plain ``dict[str, Tensor | str]`` rather than pickling the frozen dataclass
directly — a pickled dataclass breaks on any future field rename, and
``torch.load(weights_only=True)`` refuses arbitrary classes. ``load_result`` reconstructs the
dataclass from the dict.

The ``<run_id>`` subdirectory is mandatory (acceptance test 6): a flat directory would let two
models' results collide on ``volume_id`` and silently overwrite each other.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mri_ad.recon.types import ReconResult

_TENSOR_FIELDS = ("original", "reconstruction", "residual", "anomaly_mask")


def result_path(root: Path, run_id: str, volume_id: str) -> Path:
    """Return ``root/<run_id>/<volume_id>.pt`` — the canonical persistence path."""
    return Path(root) / run_id / f"{volume_id}.pt"


def save_result(result: ReconResult, root: Path, run_id: str) -> Path:
    """Persist ``result`` under ``root/<run_id>/<volume_id>.pt``; returns the written path."""
    path = result_path(root, run_id, result.volume_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, torch.Tensor | str] = {"volume_id": result.volume_id}
    for field in _TENSOR_FIELDS:
        payload[field] = getattr(result, field).detach().cpu().float().contiguous()
    if result.ground_truth is not None:
        payload["ground_truth"] = result.ground_truth.detach().cpu().float().contiguous()

    torch.save(payload, path)
    return path


def load_result(path: Path) -> ReconResult:
    """Load a :class:`ReconResult` previously written by :func:`save_result`."""
    payload = torch.load(Path(path), weights_only=True)
    return ReconResult(
        volume_id=str(payload["volume_id"]),
        original=payload["original"],
        reconstruction=payload["reconstruction"],
        residual=payload["residual"],
        anomaly_mask=payload["anomaly_mask"],
        ground_truth=payload.get("ground_truth"),
    )


__all__ = ["load_result", "result_path", "save_result"]
