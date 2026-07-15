"""The one shared preprocessing pipeline (Spec 001).

The prior work applied *different* preprocessing to OpenBHB (training) and BraTS (evaluation) —
a bug this spec exists to close. ``build_transforms`` is the single function both datasets call;
only the source-specific orient/crop step (selected by ``dataset``) differs. Everything after that
— resize, normalize, label binarization — is byte-identical MONAI dict-transform code.

Pipeline: orient to ``(D,H,W)`` -> crop (OpenBHB only) -> trilinear resize H,W to 128 (image) /
nearest resize H,W to 128 (label, binarized first) -> min-max normalize to ``[0,1]`` (image only,
eps 1e-8). Depth padding + chunking happen downstream in ``chunking.py`` / ``datasets.py`` — this
module returns one preprocessed ``(1,D,H,W)`` volume per key, not chunks.

Determinism (NFR-1): the eval/validation path never contains a random transform. A
``RandSpatialCrop``-style augmentation is permitted only when ``training=True``, and none is
wired in yet — that is an extension point for the training loop (Spec 003), not exercised here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from monai.transforms import Compose, Lambdad, MapTransform, Resized
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

DatasetName = Literal["openbhb", "brats"]


class _OrientOpenBHBd(MapTransform):
    """OpenBHB ``.npy`` volume, already ``(D,H,W)``-ordered once squeezed -> crop to the box."""

    def __init__(self, keys: list[str], crop: dict[str, list[int]]) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.crop = crop

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        z0, z1 = self.crop["z"]
        y0, y1 = self.crop["y"]
        x0, x1 = self.crop["x"]
        for key in self.keys:
            vol = _as_array(d[key]).squeeze()
            vol = vol[z0:z1, y0:y1, x0:x1]
            d[key] = torch.from_numpy(np.ascontiguousarray(vol))
        return d


class _OrientBraTSd(MapTransform):
    """BraTS native ``(H,W,D)`` volume -> permute to ``(D,H,W)``. No crop (whole volume kept)."""

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            vol = _as_array(d[key])
            vol = np.transpose(vol, (2, 0, 1))
            d[key] = torch.from_numpy(np.ascontiguousarray(vol))
        return d


def _as_array(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _add_channel(x: Tensor) -> Tensor:
    return x.unsqueeze(0) if x.ndim == 3 else x


def _binarize_label(x: Tensor) -> Tensor:
    """``seg > 0`` binarization, applied BEFORE resize (trap #1 — never trilinear-resize labels)."""
    return (x > 0).float()


def _minmax(eps: float) -> Callable[[Tensor], Tensor]:
    def _apply(x: Tensor) -> Tensor:
        vmin, vmax = x.min(), x.max()
        return (x - vmin) / (vmax - vmin + eps)

    return _apply


def build_transforms(cfg: DictConfig, *, dataset: DatasetName, training: bool = False) -> Compose:
    """Return the shared MONAI ``Compose`` for ``dataset ∈ {"openbhb", "brats"}``.

    Input: a dict with a raw ``"image"`` array/tensor (native orientation) and, for BraTS, a raw
    ``"label"`` array/tensor. Output: ``{"image": (1,D,H,W) float32 in [0,1], ["label": (1,D,H,W)
    in {0.,1.}]}``. Depth is untouched here (no padding, no chunking).
    """
    height = int(cfg.shape.height)
    width = int(cfg.shape.width)
    eps = float(cfg.data.preprocess.eps)
    resize_mode = str(cfg.data.preprocess.resize_mode)

    if dataset == "openbhb":
        crop = OmegaConf.to_container(cfg.data.openbhb.crop, resolve=True)
        orient: MapTransform = _OrientOpenBHBd(keys=["image"], crop=crop)  # type: ignore[arg-type]
        has_label = False
    elif dataset == "brats":
        orient = _OrientBraTSd(keys=["image", "label"])
        has_label = True
    else:
        raise ValueError(f"Unknown dataset {dataset!r}; expected 'openbhb' or 'brats'.")

    keys = ["image", "label"] if has_label else ["image"]
    steps: list = [orient, Lambdad(keys=keys, func=_add_channel)]

    if has_label:
        steps.append(Lambdad(keys=["label"], func=_binarize_label))
        steps.append(Resized(keys=["label"], spatial_size=(-1, height, width), mode="nearest"))

    steps.append(
        Resized(
            keys=["image"],
            spatial_size=(-1, height, width),
            mode=resize_mode,
            align_corners=False if resize_mode != "nearest" else None,
        )
    )
    steps.append(Lambdad(keys=["image"], func=_minmax(eps)))

    # training=True is an extension point for Spec 003's training-only augmentation variant; no
    # random transform is wired in here, so the eval/training compositions are currently identical
    # apart from this flag being available to future callers (NFR-1: never randomize eval/val).
    del training

    return Compose(steps)


__all__ = ["DatasetName", "build_transforms"]
