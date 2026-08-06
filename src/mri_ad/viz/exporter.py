"""``DemoExporter`` — the object behind ``make demo`` (Spec 010).

Picks a volume (defaulting to the largest-ground-truth-area volume — a demo should show the
method working on a visible lesion), loads it through ``eval.loader``, and writes a video via
:class:`~mri_ad.viz.viewer.VolumeViewer`. Never imports a model or the reconstruction engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mri_ad.eval.loader import iter_results, load_by_volume_id
from mri_ad.exceptions import ArtifactError, VizError
from mri_ad.viz.depth import DepthSpec
from mri_ad.viz.panels import RenderSpec
from mri_ad.viz.video import VideoSpec
from mri_ad.viz.viewer import VolumeViewer


@dataclass(frozen=True)
class DemoExport:
    """The record of one completed demo export."""

    path: Path
    volume_id: str
    n_frames: int
    depths: list[int]
    frames_digest: str
    fps: int


class DemoExporter:
    """Builds and exports the demo video from a saved results directory."""

    def __init__(
        self,
        render: RenderSpec,
        depth: DepthSpec,
        video: VideoSpec,
        fps: int = 4,
    ) -> None:
        """Store the render/depth/video specs and default fps."""
        self.render = render
        self.depth = depth
        self.video = video
        self.fps = fps

    @classmethod
    def from_cfg(cls, cfg: object) -> DemoExporter:
        """Build a :class:`DemoExporter` from the ``viz`` Hydra config node."""
        render = RenderSpec.from_cfg(cfg)
        depth = DepthSpec(
            policy=str(cfg.depth.policy),
            window=int(cfg.depth.window),
            min_intensity=float(cfg.depth.min_intensity),
            explicit=tuple(cfg.depth.explicit) if cfg.depth.explicit is not None else None,
            max_frames=int(cfg.depth.max_frames),
        )
        video = VideoSpec(
            codec=str(cfg.video.codec),
            quality=int(cfg.video.quality),
            pixelformat=str(cfg.video.pixelformat),
            macro_block_size=int(cfg.video.macro_block_size),
        )
        return cls(render=render, depth=depth, video=video, fps=int(cfg.fps))

    def pick_volume(self, results_dir: Path) -> str:
        """Return the volume_id with the largest ground-truth area (ties -> lexicographic min).

        Streams via ``iter_results`` rather than loading everything into memory at once.
        """
        best_id: str | None = None
        best_area = -1.0
        for result in iter_results(results_dir):
            if result.ground_truth is None:
                continue
            area = float(result.ground_truth.sum())
            if area > best_area or (
                area == best_area and (best_id is None or result.volume_id < best_id)
            ):
                best_area = area
                best_id = result.volume_id
        if best_id is None:
            raise ArtifactError(
                f"No volume with ground truth found under {Path(results_dir).name}. The demo is "
                "a BraTS artifact and needs ground truth to render the fifth panel."
            )
        return best_id

    def export(self, results_dir: Path, volume_id: str | None, output_path: Path) -> DemoExport:
        """Export the demo video for ``volume_id`` (or the auto-picked one) to ``output_path``."""
        results_dir = Path(results_dir)
        volume_id = volume_id or self.pick_volume(results_dir)
        result = load_by_volume_id(results_dir, volume_id)
        if result.ground_truth is None:
            raise ArtifactError(
                f"ReconResult {volume_id!r} has no ground_truth. The demo is a BraTS artifact "
                "and would otherwise ship a silently black ground-truth panel."
            )

        viewer = VolumeViewer(result, spec=self.render, depth=self.depth)
        depths = viewer.depths
        viewer.export_video(Path(output_path), fps=self.fps, video=self.video)

        import json

        sidecar = json.loads(Path(str(output_path) + ".frames.json").read_text())
        if sidecar["n_frames"] != len(depths):
            raise VizError(
                f"Exported {sidecar['n_frames']} frames but {len(depths)} depths were selected "
                f"for {volume_id!r} — the video and the recorded depth list have diverged."
            )
        return DemoExport(
            path=Path(output_path),
            volume_id=volume_id,
            n_frames=sidecar["n_frames"],
            depths=depths,
            frames_digest=sidecar["sha256"],
            fps=self.fps,
        )


__all__ = ["DemoExport", "DemoExporter"]
