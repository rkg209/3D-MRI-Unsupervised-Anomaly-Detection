"""Frames -> a video file (Spec 010).

Dispatches on suffix: ``.mp4`` needs ffmpeg (via ``imageio``); ``.gif``/``.webp`` use the pillow
plugin and need no ffmpeg at all — the escape hatch that lets the whole test suite exercise a
real export path on any machine, ffmpeg or not. Frames are streamed straight to the writer and
hashed in the same pass, so nothing accumulates in memory (160 x 1500x320x3 uint8 frames would be
~230 MB if materialized). ``imageio`` is imported function-locally so importing ``mri_ad.viz``
never pulls in ffmpeg as a side effect.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mri_ad.exceptions import VizError


@dataclass(frozen=True)
class VideoSpec:
    """Encoding parameters for ``write_video``."""

    codec: str = "libx264"
    quality: int = 8
    pixelformat: str = "yuv420p"
    macro_block_size: int = 1


def write_video(
    frames: Iterable[np.ndarray], path: Path, fps: int, video: VideoSpec
) -> tuple[int, str]:
    """Stream ``frames`` to ``path`` (mp4 via ffmpeg, gif/webp via pillow). Returns ``(n, digest)``.

    Raises :class:`VizError` (naming ``pip install "imageio[ffmpeg]"``) if an ``.mp4`` is
    requested and ffmpeg is missing or unusable — never a bare ``imageio``/``OSError``.
    """
    import hashlib

    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".mp4":
        writer_kwargs = {
            "fps": fps,
            "codec": video.codec,
            "quality": video.quality,
            "pixelformat": video.pixelformat,
            "macro_block_size": video.macro_block_size,
        }
    elif suffix in (".gif", ".webp"):
        # imageio's pillow plugin deprecated `fps` in favor of `duration` (ms/frame).
        writer_kwargs = {"duration": 1000.0 / fps}
    else:
        raise VizError(f"Unsupported video suffix {suffix!r}; expected .mp4, .gif, or .webp.")

    h = hashlib.sha256()
    n = 0
    try:
        with imageio.get_writer(str(path), **writer_kwargs) as writer:
            for frame in frames:
                arr = np.ascontiguousarray(frame)
                writer.append_data(arr)
                h.update(arr.tobytes())
                n += 1
    except Exception as exc:
        # Broad on purpose: a missing/unusable ffmpeg surfaces through imageio as anything from
        # OSError to a plugin-specific TypeError (observed: imageio's legacy get_writer silently
        # falling back to a TIFF plugin for a .mp4 path, then raising TypeError on append_data).
        # Whatever it is, don't leave a corrupt/partial file behind wearing the right extension.
        path.unlink(missing_ok=True)
        raise VizError(
            f"Video export to {path.name} failed ({exc}). If this is an .mp4 export, install a "
            'working ffmpeg via `pip install "imageio[ffmpeg]"`, or export a .gif instead.'
        ) from exc

    return n, h.hexdigest()


__all__ = ["VideoSpec", "write_video"]
