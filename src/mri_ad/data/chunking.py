"""Deterministic depth-chunking (Spec 001).

Ported verbatim-in-logic from the Spec 000 provisional ``slice_io._pad_depth``/``_chunk`` before
that module is deleted. Depth is zero-padded to the next multiple of the chunk depth so the
*whole* volume is covered — the prior work silently dropped the remainder (155 -> 128, discarding
27 slices). Chunks are non-overlapping and never randomized (NFR-1); a `RandSpatialCrop`-style
augmentation belongs only on the opt-in training transform variant, never here.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

_CHUNK_DEPTH = 16


def pad_depth(vol: Tensor, *, chunk_depth: int = _CHUNK_DEPTH) -> Tensor:
    """Zero-pad depth (dim 0 of ``(D, H, W)``) up to the next multiple of ``chunk_depth``."""
    depth = vol.shape[0]
    target = math.ceil(depth / chunk_depth) * chunk_depth
    if target == depth:
        return vol
    return F.pad(vol, (0, 0, 0, 0, 0, target - depth))


def chunk_volume(vol: Tensor, *, chunk_depth: int = _CHUNK_DEPTH) -> Tensor:
    """Split a padded ``(D, H, W)`` volume into ``(N, 1, chunk_depth, H, W)`` chunks."""
    depth = vol.shape[0]
    if depth % chunk_depth != 0:
        raise ValueError(
            f"chunk_volume expects depth padded to a multiple of {chunk_depth}, got {depth}."
        )
    n = depth // chunk_depth
    return vol.reshape(n, 1, chunk_depth, vol.shape[1], vol.shape[2])


@dataclass(frozen=True)
class ChunkItem:
    """One flattened depth chunk with its non-leaking provenance indices."""

    image: Tensor
    volume_index: int
    chunk_index: int
    label: Tensor | None = None


def flatten_chunks(
    volumes: list[dict[str, Tensor]],
    *,
    chunk_depth: int = _CHUNK_DEPTH,
) -> Iterator[ChunkItem]:
    """Flatten a list of ``{"image": (D,H,W), ["label": (D,H,W)]}`` volumes into chunk items.

    Each item carries an integer ``volume_index`` (position in ``volumes``) and ``chunk_index``
    (position within that volume) so evaluation can accumulate per-volume across *all* chunks
    (kills the prior work's "only 1 of 8 chunks scored" bug). No path or subject id is carried.
    """
    for volume_index, volume in enumerate(volumes):
        image_chunks = chunk_volume(
            pad_depth(volume["image"], chunk_depth=chunk_depth), chunk_depth=chunk_depth
        )
        label_chunks = None
        if "label" in volume:
            label_chunks = chunk_volume(
                pad_depth(volume["label"], chunk_depth=chunk_depth), chunk_depth=chunk_depth
            )
        for chunk_index in range(image_chunks.shape[0]):
            yield ChunkItem(
                image=image_chunks[chunk_index],
                volume_index=volume_index,
                chunk_index=chunk_index,
                label=label_chunks[chunk_index] if label_chunks is not None else None,
            )


__all__ = ["ChunkItem", "chunk_volume", "flatten_chunks", "pad_depth"]
