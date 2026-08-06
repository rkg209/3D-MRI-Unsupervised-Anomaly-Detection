"""``VolumeViewer`` — the spec's interactive + export contract object (Spec 010).

Reproducibility is scoped to a fixed environment: frames are a pure function of
``(tensors, RenderSpec, DepthSpec)`` with no RNG, no wall-clock, and no path-derived content, but
matplotlib/FreeType version drift can still shift individual pixels across environments. Compare
digests *within* one environment/run, never across a library upgrade.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch import Tensor

from mri_ad.exceptions import VizError
from mri_ad.recon.types import ReconResult
from mri_ad.viz.depth import DepthSpec, select_depths
from mri_ad.viz.frames import FrameRenderer, frames_digest, iter_frames
from mri_ad.viz.panels import RenderSpec
from mri_ad.viz.video import VideoSpec, write_video


class VolumeViewer:
    """Synchronized 5-panel viewer over one :class:`ReconResult`."""

    def __init__(
        self,
        result: ReconResult,
        label: Tensor | None = None,
        spec: RenderSpec | None = None,
        depth: DepthSpec | None = None,
    ) -> None:
        """Store the result and specs. ``label`` defaults to ``result.ground_truth`` when None."""
        self.result = result
        self.label = label if label is not None else result.ground_truth
        self.spec = spec or _default_render_spec()
        self.depth_spec = depth or DepthSpec()

    @property
    def depths(self) -> list[int]:
        """The depth indices this viewer will render, per ``self.depth_spec``."""
        return select_depths(self.result.original, self.label, self.depth_spec)

    def frame(self, depth: int) -> Any:
        """Render one composite ``(H, W, 3)`` uint8 frame at ``depth``. Shared by both paths."""
        with FrameRenderer(self.spec) as renderer:
            return renderer.render(self.result, self.label, depth)

    def show_interactive(self) -> Any:
        """Return an ``ipywidgets`` slider + image wired to :meth:`frame`.

        Imports ``ipywidgets`` inside the function body — it is an optional extra
        (``pip install -e '.[viz]'``), deliberately absent from the headless export path and
        from ``dev``, so a missing install raises :class:`VizError` rather than crashing import
        of this module for everyone else.
        """
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise VizError(
                "The interactive viewer needs the optional viz extra: pip install -e '.[viz]'"
            ) from exc

        import io

        from PIL import Image

        depths = self.depths

        def _to_png_bytes(depth: int) -> bytes:
            arr = self.frame(depth)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return buf.getvalue()

        image = widgets.Image(value=_to_png_bytes(depths[0]), format="png")
        slider = widgets.IntSlider(min=depths[0], max=depths[-1], value=depths[0])

        def _on_change(change: dict) -> None:
            if change["name"] == "value":
                image.value = _to_png_bytes(change["new"])

        slider.observe(_on_change, names="value")
        return widgets.VBox([slider, image])

    def export_video(self, path: Path, fps: int = 4, video: VideoSpec | None = None) -> Path:
        """Export the full depth sequence to ``path`` and write a ``<path>.frames.json`` sidecar."""
        video = video or VideoSpec()
        depths = self.depths
        n, digest = write_video(
            iter_frames(self.result, self.label, depths, self.spec), Path(path), fps, video
        )

        frame_shape = list(self.frame(depths[0]).shape)
        sidecar = Path(str(path) + ".frames.json")
        sidecar.write_text(
            json.dumps(
                {"n_frames": n, "fps": fps, "frame_shape": frame_shape, "sha256": digest},
                indent=2,
            )
        )
        return Path(path)


def _default_render_spec() -> RenderSpec:
    from mri_ad.viz.panels import PanelSpec

    panels = (
        PanelSpec(key="original", title="original", cmap="gray"),
        PanelSpec(key="reconstruction", title="reconstruction", cmap="gray"),
        PanelSpec(key="residual", title="residual", cmap="hot"),
        PanelSpec(key="anomaly_mask", title="anomaly mask", cmap="gray"),
        PanelSpec(key="ground_truth", title="ground truth", cmap="gray", overlay="original"),
    )
    return RenderSpec(
        panels=panels,
        figsize=(15.0, 3.2),
        dpi=100,
        caption="saved reconstruction · no live inference",
    )


__all__ = ["VolumeViewer", "frames_digest"]
