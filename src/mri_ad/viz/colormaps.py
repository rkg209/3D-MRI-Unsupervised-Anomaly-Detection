"""Guarded normalization + the ported ``cm.hot`` recipe + overlay compositing (Spec 010).

All colour math happens here, in numpy, never in matplotlib — every panel this module produces
is a pure ``(H, W, 3)`` uint8 array *before* it ever reaches ``imshow``. This is what makes frame
reproducibility (acceptance 3) and the pixel-equality privacy guard (acceptance 5) assertable
without rendering anything.

Ports ``legacy/GUI/tk_app.py:88-101``'s ``cm.hot(normalize)`` heatmap recipe, minus its defects:
the legacy code divides by a bare ``np.max`` (line 33) and an unguarded ``max - min`` (line 91),
both of which are guaranteed to crash or misbehave on BraTS's zero-padded slices — not a
hypothetical edge case, five real slices per volume.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from matplotlib import cm

EPS: Final = 1e-8


def normalize_slice(
    a: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    eps: float = EPS,
) -> np.ndarray:
    """Min-max normalize ``a`` to ``[0, 1]``, guarding the degenerate range.

    ``vmin``/``vmax`` default to ``a.min()``/``a.max()``. If the resulting range is ``<= eps``
    (an all-zero pad slice, a constant-value slice, or an all-``nan`` slice), no division is ever
    executed — the guard short-circuits to ``zeros_like(a)`` instead. This is a *guard*, not an
    epsilon fudge: a bare epsilon still divides (producing garbage or ``nan``) for a constant
    slice, it only avoids a literal ``ZeroDivisionError``.
    """
    lo = a.min() if vmin is None else vmin
    hi = a.max() if vmax is None else vmax
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= eps:
        return np.zeros_like(a, dtype=np.float64)
    out = (a.astype(np.float64) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def hot_rgb(
    a: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    eps: float = EPS,
) -> np.ndarray:
    """``(H, W)`` -> ``(H, W, 3)`` uint8 via ``matplotlib.cm.hot`` on a normalized slice."""
    normed = normalize_slice(a, vmin=vmin, vmax=vmax, eps=eps)
    rgba = cm.hot(normed)
    return (rgba[..., :3] * 255).astype(np.uint8)


def gray_rgb(
    a: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    eps: float = EPS,
) -> np.ndarray:
    """``(H, W)`` -> ``(H, W, 3)`` uint8 grayscale on a normalized slice."""
    normed = normalize_slice(a, vmin=vmin, vmax=vmax, eps=eps)
    gray = (normed * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def overlay_rgb(
    base: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Alpha-blend ``color`` onto ``base`` (``(H, W, 3)`` uint8) wherever ``mask`` is truthy."""
    out = base.astype(np.float64).copy()
    hit = mask.astype(bool)
    tint = np.asarray(color, dtype=np.float64)
    out[hit] = (1.0 - alpha) * out[hit] + alpha * tint
    return np.clip(out, 0, 255).astype(np.uint8)


__all__ = ["EPS", "gray_rgb", "hot_rgb", "normalize_slice", "overlay_rgb"]
