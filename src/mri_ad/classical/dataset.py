"""Slice-level dataset assembly for the classical baseline (Spec 006).

Turns a list of BraTS subject ids into a flat, slice-level ``(X, y, groups)`` table.
:func:`build_slice_dataset` calls :func:`mri_ad.data.datasets.load_preprocessed_volume` (never
``BraTSDataset``) so the classical baseline never sees chunking's zero-padded tail slices — those
would enter the slice pool as fabricated negatives (trap-adjacent, Spec 006 R7/plan). The seg is
already binarized and nearest-resized by ``build_transforms`` (trap #1) — this module never
re-binarizes or re-resizes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from mri_ad.classical.features import FeatureExtractor, feature_names
from mri_ad.data.datasets import load_preprocessed_volume
from mri_ad.exceptions import ClassicalError


@dataclass(frozen=True)
class SliceDataset:
    """A flat, slice-level table ready for cross-validation."""

    X: np.ndarray  # (N, F) float32
    y: np.ndarray  # (N,) uint8
    groups: np.ndarray  # (N,) object — BraTS subject folder name per row
    slice_index: np.ndarray  # (N,) int64 — depth index within its source volume
    feature_names: list[str]


def slice_labels(label_volume: np.ndarray, *, min_tumor_voxels: int = 1) -> np.ndarray:
    """Per-slice binary label: 1 iff the slice holds >= ``min_tumor_voxels`` tumor voxels.

    ``label_volume`` is ``(1,D,H,W)`` or ``(D,H,W)`` and is assumed **already** binarized
    (``seg > 0``) by the shared preprocessing pipeline (trap #1) — this never re-binarizes.
    """
    arr = np.asarray(label_volume)
    if arr.ndim == 4:
        arr = arr[0]
    voxel_counts = (arr > 0).reshape(arr.shape[0], -1).sum(axis=1)
    return (voxel_counts >= min_tumor_voxels).astype(np.uint8)


def build_slice_dataset(
    cfg: DictConfig,
    subject_ids: list[str],
    extractor: FeatureExtractor,
    *,
    progress: bool = True,
) -> tuple[SliceDataset, dict[str, int]]:
    """Extract features + labels for every slice of every subject in ``subject_ids``.

    Returns the assembled :class:`SliceDataset` plus a counter dict (``n_volumes``,
    ``n_slices_seen``, ``n_slices_dropped_empty``, ``n_samples``) — every drop is recorded, never
    silent. Raises :class:`ClassicalError` if a volume's ``(H, W)`` disagrees with
    ``cfg.shape.height/width`` or if the resulting sample pool is empty.
    """
    height, width = int(cfg.shape.height), int(cfg.shape.width)
    min_tumor_voxels = int(cfg.classical.features.label_threshold_voxels)
    drop_empty = bool(cfg.classical.features.drop_empty_slices)
    foreground_eps = float(cfg.classical.features.foreground_eps)

    iterator: list[str] = subject_ids
    if progress:
        from tqdm import tqdm

        iterator = tqdm(subject_ids, desc="classical: extracting features")

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    group_rows: list[str] = []
    slice_idx_rows: list[int] = []
    counters = {"n_volumes": 0, "n_slices_seen": 0, "n_slices_dropped_empty": 0, "n_samples": 0}

    for subject_id in iterator:
        volume = load_preprocessed_volume(cfg, subject_id)
        image = np.asarray(volume["image"])  # (1,D,H,W)
        label = np.asarray(volume["label"])  # (1,D,H,W)
        if tuple(image.shape[-2:]) != (height, width):
            raise ClassicalError(
                f"{subject_id}: expected (H,W)=({height},{width}), got {tuple(image.shape[-2:])}."
            )
        counters["n_volumes"] += 1

        labels = slice_labels(label, min_tumor_voxels=min_tumor_voxels)
        features = extractor.extract(image)
        depth = features.shape[0]
        counters["n_slices_seen"] += depth

        image_slices = image[0] if image.ndim == 4 else image  # (D,H,W)
        for d in range(depth):
            fg_fraction = float(np.mean(image_slices[d] > foreground_eps))
            if drop_empty and fg_fraction <= foreground_eps:
                counters["n_slices_dropped_empty"] += 1
                continue
            x_rows.append(features[d])
            y_rows.append(int(labels[d]))
            group_rows.append(subject_id)
            slice_idx_rows.append(d)

    if not x_rows:
        raise ClassicalError("build_slice_dataset produced zero samples for the given pool.")

    counters["n_samples"] = len(x_rows)
    data = SliceDataset(
        X=np.stack(x_rows, axis=0).astype(np.float32),
        y=np.array(y_rows, dtype=np.uint8),
        groups=np.array(group_rows, dtype=object),
        slice_index=np.array(slice_idx_rows, dtype=np.int64),
        feature_names=feature_names(extractor.config),
    )
    return data, counters


__all__ = ["SliceDataset", "build_slice_dataset", "slice_labels"]
