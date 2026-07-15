"""Deterministic dataloader construction (Spec 001, `monai-patterns` house style).

`build_dataloader` is the one place a `DataLoader` gets built: seeded generator,
`monai.data.utils.worker_init_fn` for reproducible multi-worker shuffling, `shuffle=False` on
every split except `train`, and a `DataValidator` check on the first batch at init so a
shape/dtype/range regression fails immediately instead of surfacing as a corrupted Dice later.
"""

from __future__ import annotations

import monai.data.utils
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from mri_ad.data.validation import DataValidator


def build_dataloader(cfg: DictConfig, dataset: Dataset, *, split: str) -> DataLoader:
    """Build a seeded, deterministic `DataLoader` and validate its first batch.

    Args:
        cfg: Resolved Hydra config (uses `cfg.data.loader.*` and `cfg.seed`).
        dataset: A chunk-level dataset (`OpenBHBDataset`/`BraTSDataset`).
        split: The split name (`"train"`, `"val"`, or `"test"`). Only `"train"` shuffles.

    Returns:
        The constructed `DataLoader`, already validated.
    """
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.loader.batch_size),
        num_workers=int(cfg.data.loader.num_workers),
        shuffle=split == "train",
        generator=torch.Generator().manual_seed(int(cfg.seed)),
        worker_init_fn=monai.data.utils.worker_init_fn,
    )

    first_batch = next(iter(loader))
    image = first_batch["image"] if isinstance(first_batch, dict) else first_batch
    DataValidator.validate_batch(image)

    return loader


__all__ = ["build_dataloader"]
