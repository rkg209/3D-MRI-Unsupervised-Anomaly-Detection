#!/usr/bin/env python
"""Spec 000 vertical slice: one checkpoint, one BraTS volume -> Dice + figure.

The thinnest end-to-end path through the whole stack, wrapped in the reproducibility spine.
Run with ``make slice`` (see ``configs/config.yaml`` for the operating point). Requires real
weights + BraTS data — run ``make check-data`` first.

Everything provisional here (single-volume loader, single hardcoded threshold, single metric)
is replaced by Specs 001-004 as their abstractions land.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from mri_ad.data.datasets import BraTSDataset  # noqa: E402
from mri_ad.eval.metrics import MetricsComputer  # noqa: E402
from mri_ad.exceptions import DataError  # noqa: E402
from mri_ad.models import build_default_registry  # noqa: E402
from mri_ad.recon import ReconstructionEngine  # noqa: E402
from mri_ad.utils.device import DeviceManager  # noqa: E402
from mri_ad.utils.run_logger import RunLogger  # noqa: E402
from mri_ad.utils.seed import seed_everything  # noqa: E402


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

        brats_dir = Path(str(cfg.data.brats.dir))
        if not brats_dir.is_dir():
            raise DataError(f"BraTS directory not present: {brats_dir.name}.")
        volume_id = _resolve_volume_id(brats_dir, cfg.slice.volume_id)
        dataset = BraTSDataset(cfg, [volume_id])

        engine = ReconstructionEngine(model, cfg)
        result = engine.run_dataset_volume(dataset, volume_index=0)

        # Drop the channel dim (1, D, 128, 128) -> (D, 128, 128) for Dice and the figure.
        original = result.original.squeeze(0)
        reconstruction = result.reconstruction.squeeze(0)
        residual = result.residual.squeeze(0)
        mask = result.anomaly_mask.squeeze(0)
        gt = result.ground_truth.squeeze(0)

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
