"""Chunk-level datasets (Spec 001).

`OpenBHBDataset` and `BraTSDataset` enumerate subjects, run each volume through the ONE shared
preprocessing pipeline (`transforms.build_transforms`), and present chunk-level items: each
`__getitem__` returns one `(1,16,128,128)` chunk plus non-leaking `volume_index`/`chunk_index`
integers, so evaluation can accumulate per-volume across ALL chunks (kills the prior work's
"only 1 of 8 chunks scored" bug). No file path or patient identifier is ever returned (NFR-13) —
paths are consumed by the load step and never survive into the cached, transformed dict.

Per-volume preprocessing is cached via `monai.data.CacheDataset` (`cache_rate` from
`cfg.data.loader.cache_rate`) so repeated chunk access does not re-read/re-transform a volume
from disk on every `__getitem__`.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path

import nibabel as nib
import numpy as np
from monai.data import CacheDataset
from monai.transforms import Compose
from omegaconf import DictConfig
from torch.utils.data import Dataset

from mri_ad.data.chunking import chunk_volume, pad_depth
from mri_ad.data.transforms import build_transforms
from mri_ad.exceptions import DataError


class _LoadOpenBHBd:
    """`{"image_path": Path} -> {"image": ndarray}`. The only place an OpenBHB path is touched."""

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        d["image"] = np.load(str(d.pop("image_path")))
        return d


class _LoadBraTSd:
    """`{"image_path", "label_path": Path} -> {"image", "label": ndarray}`."""

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        d["image"] = np.asarray(nib.load(str(d.pop("image_path"))).get_fdata(), dtype=np.float32)
        d["label"] = np.asarray(nib.load(str(d.pop("label_path"))).get_fdata(), dtype=np.float32)
        return d


def _find_modality(subject_dir: Path, volume_id: str, suffix: str) -> Path:
    """Find ``<id>_<suffix>.nii(.gz)`` inside a BraTS subject directory."""
    for ext in (".nii.gz", ".nii"):
        candidate = subject_dir / f"{volume_id}_{suffix}{ext}"
        if candidate.is_file():
            return candidate
    raise DataError(f"Missing {suffix!r} volume for subject {volume_id}.")


class OpenBHBDataset(Dataset):
    """Healthy T1 volumes, chunk-level: ``{"image", "volume_index", "chunk_index"}``."""

    def __init__(self, cfg: DictConfig, volume_ids: list[str]) -> None:
        """Enumerate ``volume_ids`` (filenames under ``cfg.data.openbhb.dir``) chunk-level."""
        self.cfg = cfg
        self.dir = Path(str(cfg.data.openbhb.dir))
        self.volume_ids = list(volume_ids)
        self.chunk_depth = int(cfg.data.preprocess.chunk_depth)

        crop = cfg.data.openbhb.crop
        crop_depth = int(crop.z[1]) - int(crop.z[0])
        self._chunks_per_volume = math.ceil(crop_depth / self.chunk_depth)

        cache_items = [{"image_path": self.dir / vid} for vid in self.volume_ids]
        transform = Compose([_LoadOpenBHBd(), build_transforms(cfg, dataset="openbhb")])
        self._volumes = CacheDataset(
            data=cache_items, transform=transform, cache_rate=float(cfg.data.loader.cache_rate)
        )

    def __len__(self) -> int:
        """Total chunk count: ``len(volume_ids) * chunks_per_volume``."""
        return len(self.volume_ids) * self._chunks_per_volume

    def __getitem__(self, index: int) -> dict:
        """Return one chunk: ``{"image", "volume_index", "chunk_index"}``."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        vol_idx, chunk_idx = divmod(index, self._chunks_per_volume)
        volume = self._volumes[vol_idx]
        image = chunk_volume(
            pad_depth(volume["image"].squeeze(0), chunk_depth=self.chunk_depth),
            chunk_depth=self.chunk_depth,
        )[chunk_idx]
        return {"image": image, "volume_index": vol_idx, "chunk_index": chunk_idx}


class BraTSDataset(Dataset):
    """Tumor T2 volumes + segmentation, chunk-level. Whole depth covered (padded, never dropped)."""

    def __init__(self, cfg: DictConfig, volume_ids: list[str]) -> None:
        """Enumerate ``volume_ids`` (subject folders under ``cfg.data.brats.dir``) chunk-level."""
        self.cfg = cfg
        self.dir = Path(str(cfg.data.brats.dir))
        self.modality = str(cfg.data.brats.modality)
        self.volume_ids = list(volume_ids)
        self.chunk_depth = int(cfg.data.preprocess.chunk_depth)

        self._chunk_counts = [self._count_chunks(vid) for vid in self.volume_ids]
        self._offsets = list(accumulate([0, *self._chunk_counts]))

        cache_items = [
            {
                "image_path": _find_modality(self.dir / vid, vid, self.modality),
                "label_path": _find_modality(self.dir / vid, vid, "seg"),
            }
            for vid in self.volume_ids
        ]
        transform = Compose([_LoadBraTSd(), build_transforms(cfg, dataset="brats")])
        self._volumes = CacheDataset(
            data=cache_items, transform=transform, cache_rate=float(cfg.data.loader.cache_rate)
        )

    def _count_chunks(self, volume_id: str) -> int:
        image_path = _find_modality(self.dir / volume_id, volume_id, self.modality)
        depth = int(nib.load(str(image_path)).shape[-1])
        return math.ceil(depth / self.chunk_depth)

    def __len__(self) -> int:
        """Total chunk count across all subjects (whole-depth padded, never truncated)."""
        return self._offsets[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        vol_idx = bisect_right(self._offsets, index) - 1
        return vol_idx, index - self._offsets[vol_idx]

    def __getitem__(self, index: int) -> dict:
        """Return one chunk: ``{"image", "label", "volume_index", "chunk_index"}``."""
        vol_idx, chunk_idx = self._locate(index)
        volume = self._volumes[vol_idx]
        image = chunk_volume(
            pad_depth(volume["image"].squeeze(0), chunk_depth=self.chunk_depth),
            chunk_depth=self.chunk_depth,
        )[chunk_idx]
        label = chunk_volume(
            pad_depth(volume["label"].squeeze(0), chunk_depth=self.chunk_depth),
            chunk_depth=self.chunk_depth,
        )[chunk_idx]
        return {"image": image, "label": label, "volume_index": vol_idx, "chunk_index": chunk_idx}


def load_preprocessed_volume(cfg: DictConfig, volume_id: str) -> dict:
    """Run one BraTS subject through the shared pipeline and return the **whole** volume.

    Unlike :class:`BraTSDataset`, this never pads or chunks depth — Spec 006's classical
    baseline must never see chunking's zero-padded tail slices, which would enter the slice
    pool as fabricated negatives. Returns ``{"image": (1,D,128,128) in [0,1], "label":
    (1,D,128,128) in {0,1}}``. No path or identifier survives into the returned dict.
    """
    subject_dir = Path(str(cfg.data.brats.dir)) / volume_id
    modality = str(cfg.data.brats.modality)
    data = {
        "image_path": _find_modality(subject_dir, volume_id, modality),
        "label_path": _find_modality(subject_dir, volume_id, "seg"),
    }
    transform = Compose([_LoadBraTSd(), build_transforms(cfg, dataset="brats")])
    result = transform(data)
    return {"image": result["image"], "label": result["label"]}


__all__ = ["BraTSDataset", "OpenBHBDataset", "load_preprocessed_volume"]
