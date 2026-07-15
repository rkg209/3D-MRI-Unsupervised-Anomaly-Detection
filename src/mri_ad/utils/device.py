"""Device selection (Spec 000 spine).

A single choke point for `cuda -> mps -> cpu`, so no module hand-rolls its own
`torch.cuda.is_available()` dance and drifts.
"""

from __future__ import annotations

import torch


class DeviceManager:
    """Resolve the compute device once, consistently, for every entry point."""

    @staticmethod
    def get_device(preference: str = "auto") -> torch.device:
        """Return the device to run on.

        Args:
            preference: ``"auto"`` picks ``cuda -> mps -> cpu`` by availability. Any other value
                (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``) is honoured verbatim so a run can be pinned.

        Returns:
            The resolved :class:`torch.device`.
        """
        if preference and preference != "auto":
            return torch.device(preference)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
