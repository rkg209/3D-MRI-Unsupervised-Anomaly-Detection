"""One depth -> one RGB frame (Spec 010).

``FrameRenderer`` creates its ``Figure`` + ``FigureCanvasAgg`` exactly once and reuses it across
every depth — 160 frames therefore cost one figure, not 160. Axes use ``fig.subplots_adjust``
with fixed constants, never ``tight_layout``/``bbox_inches="tight"``, whose output size varies
with title length and would break a video encoder mid-stream.

No library code here sets a global matplotlib backend (``matplotlib.use``) — ``FigureCanvasAgg``
is instantiated directly, so importing this module never hijacks a caller's (e.g. a notebook's)
backend.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import Tensor

from mri_ad.recon.types import ReconResult
from mri_ad.viz.panels import RenderSpec, _as_dhw, build_panels

_SUBPLOT_ADJUST = {"left": 0.01, "right": 0.99, "top": 0.85, "bottom": 0.02, "wspace": 0.05}


class FrameRenderer:
    """Renders one composite ``(H, W, 3)`` uint8 frame per depth, reusing one ``Figure``."""

    def __init__(self, spec: RenderSpec) -> None:
        """Build the persistent figure/axes/image artists for ``spec``."""
        self.spec = spec
        n = len(spec.panels)
        self._fig = Figure(figsize=spec.figsize, dpi=spec.dpi)
        self._canvas = FigureCanvasAgg(self._fig)
        # squeeze=False: subplots(1, 1) returns a bare Axes (not iterable) when n == 1.
        self._axes = self._fig.subplots(1, n, squeeze=False)[0]
        self._fig.subplots_adjust(**_SUBPLOT_ADJUST)
        self._images = []
        for ax, panel in zip(self._axes, spec.panels, strict=True):
            im = ax.imshow(np.zeros((1, 1, 3), dtype=np.uint8))
            ax.set_title(panel.title)
            ax.axis("off")
            self._images.append(im)
        self._suptitle = self._fig.suptitle("")

    def render(self, result: ReconResult, label: Tensor | None, depth: int) -> np.ndarray:
        """Render the frame at ``depth`` and return it as an ``(H, W, 3)`` uint8 array."""
        original = _as_dhw(result.original, "original")
        n_depths = original.shape[0]
        residual_vmax = None
        if self.spec.residual_scale == "per_volume":
            residual = _as_dhw(result.residual, "residual")
            residual_vmax = float(residual.max())

        panels = build_panels(result, label, depth, self.spec, residual_vmax=residual_vmax)
        for im, (title, rgb) in zip(self._images, panels, strict=True):
            im.set_data(rgb)
            # set_data alone does not update the image's extent/axes limits, which stay pinned
            # to whatever shape was passed to the first imshow() call — harmless for the square
            # placeholder + square panels this ships with, but silently aspect-distorts any
            # non-square panel. Keep the extent honest on every frame.
            im.set_extent((-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
            im.axes.set_title(title)

        prefix = f"{result.volume_id}  ·  " if self.spec.annotate_volume_id else ""
        self._suptitle.set_text(f"{prefix}depth {depth:03d} / {n_depths}  ·  {self.spec.caption}")

        self._canvas.draw()
        buf = np.asarray(self._canvas.buffer_rgba())
        return buf[..., :3].copy()

    def close(self) -> None:
        """Drop references to the figure/canvas/artists so they can be garbage collected.

        The figure was created via ``Figure()`` directly (never ``plt.figure()``), so it was
        never registered with pyplot's global figure manager — there is nothing for
        ``pyplot.close`` to do, and importing pyplot here would be pointless module weight.
        """
        self._images = []
        self._axes = None
        self._canvas = None
        self._fig = None

    def __enter__(self) -> FrameRenderer:
        """Context-manager entry; returns ``self``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Context-manager exit; closes the figure."""
        self.close()


def iter_frames(
    result: ReconResult, label: Tensor | None, depths: Sequence[int], spec: RenderSpec
) -> Iterator[np.ndarray]:
    """Yield one ``(H, W, 3)`` uint8 frame per depth in ``depths``; streams, never materializes."""
    with FrameRenderer(spec) as renderer:
        for d in depths:
            yield renderer.render(result, label, d)


def frames_digest(frames: Iterable[np.ndarray]) -> str:
    """Streaming sha256 over each frame's raw bytes, in order."""
    h = hashlib.sha256()
    for frame in frames:
        h.update(np.ascontiguousarray(frame).tobytes())
    return h.hexdigest()


__all__ = ["FrameRenderer", "frames_digest", "iter_frames"]
