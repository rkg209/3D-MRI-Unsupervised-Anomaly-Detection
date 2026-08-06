"""Checkpoint writing (Spec 009).

Writes exactly the dict layout ``mri_ad.models._checkpoint.load_checked_state_dict`` already
knows how to unwrap (``{"model_state_dict": ...}``), plus provenance fields for
``run_meta.json``/manual inspection. Per NFR-13, ``meta`` must never carry an on-disk path or
patient identifier — only the git SHA, seed, split hash, and resolved hyperparameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


class CheckpointWriter:
    """Writes ``path`` on each :meth:`save`, overwriting the previous best."""

    def __init__(self, path: Path, *, meta: Mapping[str, Any]) -> None:
        """Store the destination ``path`` and static provenance ``meta`` (git SHA, seed, ...)."""
        self.path = Path(path)
        self.meta = dict(meta)

    def save(self, model: nn.Module, *, epoch: int, metrics: Mapping[str, float]) -> Path:
        """Write ``path``: ``model_state_dict`` plus ``epoch``/``metrics``/the static ``meta``."""
        selection_key = self.meta.get("selection_key")
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "selection_key": selection_key,
            "selection_value": metrics.get(selection_key) if selection_key else None,
            "val_loss": metrics.get("val_loss"),
            "git_sha": self.meta.get("git_sha"),
            "seed": self.meta.get("seed"),
            "split_hash": self.meta.get("split_hash"),
            "synth": self.meta.get("synth", {}),
            "train": self.meta.get("train", {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.path)
        return self.path


__all__ = ["CheckpointWriter"]
