"""3D viewer and demo-video export (Spec 010)."""

from __future__ import annotations

from mri_ad.viz.colormaps import hot_rgb, normalize_slice
from mri_ad.viz.depth import DepthSpec
from mri_ad.viz.exporter import DemoExport, DemoExporter
from mri_ad.viz.panels import RenderSpec
from mri_ad.viz.video import VideoSpec
from mri_ad.viz.viewer import VolumeViewer

__all__ = [
    "DemoExport",
    "DemoExporter",
    "DepthSpec",
    "RenderSpec",
    "VideoSpec",
    "VolumeViewer",
    "hot_rgb",
    "normalize_slice",
]
