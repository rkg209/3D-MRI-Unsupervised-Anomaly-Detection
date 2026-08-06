"""Depth-range policies for the zero-padded volume (Spec 010, D-C).

BraTS volumes are 155 real slices zero-padded to 160 (``recon/types.py:14``). A literal
``all``-depth video opens and closes on black. ``nonzero`` (the default) trims to the slices
that actually carry signal; ``gt_window`` centres on the lesion; ``explicit`` is a manual
override. All policies fall back toward ``all`` rather than ever returning an empty frame list.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from mri_ad.exceptions import VizError
from mri_ad.viz.panels import _as_dhw


@dataclass(frozen=True)
class DepthSpec:
    """Which depth indices to render."""

    policy: str = "nonzero"  # all | nonzero | gt_window | explicit
    window: int = 40
    min_intensity: float = 0.0
    explicit: tuple[int, int] | None = None
    max_frames: int = 160


def _all_depths(depth: int) -> list[int]:
    return list(range(depth))


def _nonzero_depths(original: Tensor, min_intensity: float) -> list[int]:
    dhw = _as_dhw(original, "original")
    depth = dhw.shape[0]
    return [d for d in range(depth) if float(dhw[d].max()) > min_intensity]


def _gt_window_depths(label: Tensor | None, window: int, depth: int) -> list[int]:
    if label is None:
        return []
    dhw = _as_dhw(label, "ground_truth")
    areas = dhw.sum(dim=(-1, -2))
    if float(areas.max()) <= 0.0:
        return []
    center = int(areas.argmax().item())
    lo = max(0, center - window // 2)
    hi = min(depth, lo + window)
    lo = max(0, hi - window)
    return list(range(lo, hi))


def _explicit_depths(explicit: tuple[int, int] | None, depth: int) -> list[int]:
    if explicit is None:
        return []
    start, stop = explicit
    start = max(0, start)
    stop = min(depth - 1, stop)
    if start > stop:
        return []
    return list(range(start, stop + 1))


def _subsample(depths: list[int], max_frames: int) -> list[int]:
    if len(depths) <= max_frames or max_frames <= 0:
        return depths
    stride = len(depths) / max_frames
    picked = sorted({depths[int(i * stride)] for i in range(max_frames)})
    return picked


def select_depths(original: Tensor, label: Tensor | None, spec: DepthSpec) -> list[int]:
    """Return the sorted, unique, subsampled list of depth indices to render for ``spec.policy``.

    Falls back ``gt_window -> nonzero -> all`` whenever a policy would otherwise yield fewer than
    2 depths (keeps tiny synthetic test tensors and edge-case volumes working).
    """
    dhw = _as_dhw(original, "original")
    depth = dhw.shape[0]

    if spec.policy == "explicit":
        depths = _explicit_depths(spec.explicit, depth)
    elif spec.policy == "gt_window":
        depths = _gt_window_depths(label, spec.window, depth)
    elif spec.policy == "nonzero":
        depths = _nonzero_depths(original, spec.min_intensity)
    elif spec.policy == "all":
        depths = _all_depths(depth)
    else:
        raise VizError(f"Unknown depth policy {spec.policy!r}.")

    if len(depths) < 2 and spec.policy == "gt_window":
        depths = _nonzero_depths(original, spec.min_intensity)
    if len(depths) < 2:
        depths = _all_depths(depth)

    depths = sorted(set(depths))
    return _subsample(depths, spec.max_frames)


__all__ = ["DepthSpec", "select_depths"]
