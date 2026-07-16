"""Shared checkpoint-loading helper (Spec 002).

Single source of truth for the dual-layout + LFS-stub + ``strict=True`` logic. Every model's
``load_checkpoint`` routes through :func:`load_checked_state_dict` so the "detect a stub, refuse
a partial load" behaviour lives in exactly one place instead of being re-derived per model.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mri_ad.exceptions import CheckpointError

# First bytes of a Git-LFS pointer stub. The inherited legacy .pth files are 133-byte stubs,
# not weights; loading one must fail loudly rather than degrade to random init.
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def load_checked_state_dict(module: nn.Module, path: Path | str) -> None:
    """Load weights from ``path`` into ``module`` with ``strict=True``.

    Accepts a bare ``state_dict`` and a ``{"model_state_dict": ...}`` wrapper — both layouts
    exist among the real weights. Raises :class:`CheckpointError` (never falls back to
    ``strict=False``) on a missing file, a Git-LFS stub, or a key mismatch.

    ``module`` is the target to load *into* — for wrapper models this is the inner net, not the
    wrapper itself, so bare (unprefixed) legacy keys line up (Spec 002 design decision #2).
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path.name}")
    with path.open("rb") as fh:
        if fh.read(len(LFS_MAGIC)) == LFS_MAGIC:
            raise CheckpointError(
                f"Checkpoint {path.name} is a Git-LFS pointer stub, not real weights "
                f"({path.stat().st_size} bytes). Fetch the real file (see README)."
            )
    loaded = torch.load(path, map_location="cpu")
    state = (
        loaded["model_state_dict"]
        if isinstance(loaded, dict) and "model_state_dict" in loaded
        else loaded
    )
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as err:
        raise CheckpointError(f"State-dict mismatch loading {path.name}: {err}") from err
