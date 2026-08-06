"""The 5-panel contract (Spec 010).

``build_panels`` is deliberately unit-testable with no rendering at all: it returns five
``(title, (H, W, 3) uint8)`` pairs, all sliced at the same ``depth`` — this is what makes "one
slider drives all five panels" (acceptance 1) a property of the data, not of the figure code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from mri_ad.exceptions import VizError
from mri_ad.recon.types import ReconResult
from mri_ad.viz.colormaps import gray_rgb, hot_rgb, overlay_rgb


@dataclass(frozen=True)
class PanelSpec:
    """One panel's static rendering config."""

    key: str
    title: str
    cmap: str
    overlay: str | None = None


@dataclass(frozen=True)
class RenderSpec:
    """Everything needed to render one frame, independent of any file path."""

    panels: tuple[PanelSpec, ...]
    figsize: tuple[float, float]
    dpi: int
    caption: str
    annotate_volume_id: bool = False
    residual_scale: str = "per_volume"  # per_volume | per_slice
    overlay_color: tuple[int, int, int] = (255, 0, 0)
    overlay_alpha: float = 0.45
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Validate ``residual_scale`` at construction.

        A config typo must fail loudly, not silently fall through to the per-slice behaviour
        D-D exists to avoid.
        """
        if self.residual_scale not in ("per_volume", "per_slice"):
            raise VizError(
                f"viz.normalize.residual_scale must be 'per_volume' or 'per_slice', got "
                f"{self.residual_scale!r}."
            )

    @classmethod
    def from_cfg(cls, cfg: object) -> RenderSpec:
        """Build a :class:`RenderSpec` from the ``viz`` Hydra config node.

        Validates that ``figsize x dpi`` is an even integer pair in both dimensions (R2):
        libx264 + yuv420p requires even-dimensioned frames, and a silent imageio resize would
        make the encoded video no longer match the hashed, tested frame array.
        """
        panels = tuple(
            PanelSpec(key=p["key"], title=p["title"], cmap=p["cmap"], overlay=p.get("overlay"))
            for p in cfg.panels
        )
        dpi = int(cfg.dpi)
        figsize = (float(cfg.figsize[0]), float(cfg.figsize[1]))
        px_w, px_h = figsize[0] * dpi, figsize[1] * dpi
        for name, px in (("width", px_w), ("height", px_h)):
            if abs(px - round(px)) > 1e-6 or int(round(px)) % 2 != 0:
                raise VizError(
                    f"viz.figsize x viz.dpi must yield an even integer pixel {name} "
                    f"(got {px}); libx264/yuv420p requires even frame dimensions."
                )
        return cls(
            panels=panels,
            figsize=figsize,
            dpi=dpi,
            caption=str(cfg.caption),
            annotate_volume_id=bool(cfg.annotate_volume_id),
            residual_scale=str(cfg.normalize.residual_scale),
            overlay_color=tuple(int(c) for c in cfg.overlay.color),
            overlay_alpha=float(cfg.overlay.alpha),
            eps=float(cfg.normalize.eps),
        )


def _as_dhw(t: Tensor, name: str) -> Tensor:
    """Squeeze ``(1, D, H, W) -> (D, H, W)``; a bare ``(D, H, W)`` passes through unchanged."""
    if t.dim() == 4 and t.shape[0] == 1:
        return t.squeeze(0)
    if t.dim() == 3:
        return t
    raise VizError(f"{name} has unexpected shape {tuple(t.shape)}; expected (1,D,H,W) or (D,H,W).")


def build_panels(
    result: ReconResult,
    label: Tensor | None,
    depth: int,
    spec: RenderSpec,
    *,
    residual_vmax: float | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Render all five panels at ``depth``, returning ``(title, rgb)`` pairs in ``spec`` order."""
    original = _as_dhw(result.original, "original")
    reconstruction = _as_dhw(result.reconstruction, "reconstruction")
    residual = _as_dhw(result.residual, "residual")
    mask = _as_dhw(result.anomaly_mask, "anomaly_mask")
    gt = label if label is not None else result.ground_truth
    if gt is not None:
        gt = _as_dhw(gt, "ground_truth")
        if gt.shape != original.shape:
            raise VizError(
                f"ReconResult {result.volume_id!r}: ground_truth shape {tuple(gt.shape)} != "
                f"original shape {tuple(original.shape)}."
            )

    if depth < 0 or depth >= original.shape[0]:
        raise VizError(f"depth {depth} out of range for volume of depth {original.shape[0]}.")

    original_slice = original[depth].numpy()
    reconstruction_slice = reconstruction[depth].numpy()
    residual_slice = residual[depth].numpy()
    mask_slice = mask[depth].numpy()

    for name, arr in (
        ("original", original_slice),
        ("reconstruction", reconstruction_slice),
        ("residual", residual_slice),
        ("anomaly_mask", mask_slice),
    ):
        if not np.isfinite(arr).all():
            raise VizError(
                f"ReconResult {result.volume_id!r}: {name} at depth {depth} contains "
                "NaN/inf — refusing to render a plausible-looking frame over garbage."
            )

    panels: list[tuple[str, np.ndarray]] = []
    for p in spec.panels:
        if p.key == "original":
            rgb = gray_rgb(original_slice, vmin=0.0, vmax=1.0, eps=spec.eps)
        elif p.key == "reconstruction":
            rgb = gray_rgb(reconstruction_slice, vmin=0.0, vmax=1.0, eps=spec.eps)
        elif p.key == "residual":
            if spec.residual_scale == "per_volume" and residual_vmax is None:
                raise VizError(
                    "residual_scale='per_volume' requires residual_vmax (the volume-wide max) "
                    "to be supplied — pass it explicitly, or use residual_scale='per_slice'."
                )
            vmax = residual_vmax if spec.residual_scale == "per_volume" else None
            rgb = hot_rgb(residual_slice, vmin=0.0, vmax=vmax, eps=spec.eps)
        elif p.key == "anomaly_mask":
            rgb = gray_rgb(mask_slice, vmin=0.0, vmax=1.0, eps=spec.eps)
        elif p.key == "ground_truth":
            if gt is None:
                rgb = np.zeros((*original_slice.shape, 3), dtype=np.uint8)
                panels.append(("ground truth (unavailable)", rgb))
                continue
            base = gray_rgb(original_slice, vmin=0.0, vmax=1.0, eps=spec.eps)
            rgb = overlay_rgb(base, gt[depth].numpy(), spec.overlay_color, spec.overlay_alpha)
        else:
            raise VizError(f"Unknown panel key {p.key!r} in viz config.")
        panels.append((p.title, rgb))

    return panels


__all__ = ["PanelSpec", "RenderSpec", "build_panels"]
