#!/usr/bin/env python
"""Spec 000 vertical slice: one checkpoint, one BraTS volume -> Dice + figure.

The thinnest end-to-end path through the whole stack, wrapped in the reproducibility spine.
Run with ``make slice`` (see ``configs/config.yaml`` for the operating point). Requires real
weights + BraTS data — run ``make check-data`` first.

Everything provisional here (single-volume loader, single hardcoded threshold, single metric)
is replaced by Specs 001-004 as their abstractions land.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from mri_ad.data.datasets import BraTSDataset  # noqa: E402
from mri_ad.eval.metrics import MetricsComputer  # noqa: E402
from mri_ad.exceptions import DataError  # noqa: E402
from mri_ad.models import build_default_registry  # noqa: E402
from mri_ad.utils import DeviceManager, RunLogger, seed_everything  # noqa: E402


def _instantiate(node: DictConfig) -> Any:
    """Build a non-model object from a ``{target, params}`` config node (e.g. the threshold)."""
    module_path, _, cls_name = str(node.target).rpartition(".")
    cls = getattr(importlib.import_module(module_path), cls_name)
    params = OmegaConf.to_container(node.get("params", {}), resolve=True) or {}
    return cls(**params)


def _resolve_volume_id(brats_dir: Path, volume_id: str | None) -> str:
    """Return the requested subject folder name, or the first sorted subject if ``None``."""
    subjects = sorted(p.name for p in brats_dir.iterdir() if p.is_dir())
    if not subjects:
        raise DataError(f"No BraTS subject directories found under {brats_dir.name}.")
    if volume_id is None:
        return subjects[0]
    if volume_id not in subjects:
        raise DataError(
            f"BraTS subject {volume_id!r} not found ({len(subjects)} subjects present)."
        )
    return volume_id


def _figure(volume: dict[str, torch.Tensor], volume_id: str, out_dir: Path) -> Path:
    """Save the 5-panel figure at the depth slice with the largest ground-truth area."""
    gt = volume["gt"]
    depth_idx = int(gt.sum(dim=(1, 2)).argmax().item())
    panels = [
        ("original", volume["original"], "gray"),
        ("reconstruction", volume["reconstruction"], "gray"),
        ("residual", volume["residual"], "inferno"),
        ("anomaly mask", volume["mask"], "gray"),
        ("ground truth", gt, "gray"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (title, vol, cmap) in zip(axes, panels, strict=True):
        ax.imshow(vol[depth_idx].cpu().numpy(), cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{volume_id}  ·  depth {depth_idx}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"slice_{volume_id}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load -> reconstruct -> threshold -> Dice for one volume; save figure + run_meta."""
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    with RunLogger(cfg) as run:
        device = DeviceManager.get_device(cfg.device)

        registry = build_default_registry()
        model = registry.get(cfg.model.name)
        model.load_checkpoint(registry.checkpoint_path(cfg.model.name))
        model.to(device).eval()

        threshold = _instantiate(cfg.threshold)

        brats_dir = Path(str(cfg.data.brats.dir))
        if not brats_dir.is_dir():
            raise DataError(f"BraTS directory not present: {brats_dir.name}.")
        volume_id = _resolve_volume_id(brats_dir, cfg.slice.volume_id)
        dataset = BraTSDataset(cfg, [volume_id])

        image_chunks = []
        label_chunks = []
        recon_chunks = []
        residual_chunks = []
        with torch.no_grad():
            for i in range(len(dataset)):
                item = dataset[i]
                chunk = item["image"]
                image_chunks.append(chunk)
                label_chunks.append(item["label"])
                x = chunk.unsqueeze(0).to(device)  # (1, 1, 16, 128, 128)
                recon = model(x).squeeze(0).cpu()  # (1, 16, 128, 128)
                recon_chunks.append(recon)
                residual_chunks.append((chunk - recon).abs())

        # Reassemble ALL chunks to full depth (prior-work bug #2: score the whole volume, not
        # one chunk). Each chunk is (1, 16, 128, 128); cat along depth (dim=1) -> (1, D, 128, 128),
        # drop the channel -> (D, 128, 128).
        original = torch.cat(image_chunks, dim=1).squeeze(0)
        reconstruction = torch.cat(recon_chunks, dim=1).squeeze(0)
        residual = torch.cat(residual_chunks, dim=1).squeeze(0)
        gt = torch.cat(label_chunks, dim=1).squeeze(0)

        mask = threshold(residual)  # over the whole volume

        dice = MetricsComputer.dice(mask, gt)
        assert 0.0 <= dice <= 1.0, f"Dice out of range: {dice}"

        _figure(
            {
                "original": original,
                "reconstruction": reconstruction,
                "residual": residual,
                "mask": mask,
                "gt": gt,
            },
            volume_id,
            Path(str(cfg.paths.artifact_root)) / "figures",
        )

        run.record(dice=dice)
        print(f"{dice:.6f}")


if __name__ == "__main__":
    main()
