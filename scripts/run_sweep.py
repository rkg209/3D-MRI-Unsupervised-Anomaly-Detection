#!/usr/bin/env python
"""Spec 003: Dice-vs-threshold sweep on the VALIDATION split. GPU; manual invoke only.

One forward pass per validation volume (the expensive part), then every configured threshold
operating point is scored offline against the saved residual + ground truth. The chosen point is
reported, never auto-written into config — a human reviews the numbers and edits
``configs/threshold/*.yaml`` by hand, with the justification recorded in ``progress_report.md``.

Run with ``make sweep HYDRA_OVERRIDES="+experiment=cluster model=unetr"``. Requires real
weights + BraTS data — run ``make check-data`` first.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mri_ad.data.datasets import BraTSDataset
from mri_ad.data.split import SplitContract
from mri_ad.exceptions import ConfigError, DataError
from mri_ad.models import build_default_registry
from mri_ad.recon import (
    ReconstructionEngine,
    run_threshold_sweep,
    save_result,
    select_operating_point,
)
from mri_ad.utils import DeviceManager, RunLogger, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Reconstruct every val volume, sweep the threshold, and write the Dice-vs-threshold CSV."""
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    if "sweep" not in cfg.threshold:
        raise ConfigError(
            "cfg.threshold.sweep is absent — run with the default threshold=percentile "
            "(configs/threshold/percentile.yaml), which carries the sweep axes."
        )

    with RunLogger(cfg) as run:
        device = DeviceManager.get_device(cfg.device)

        registry = build_default_registry()
        model = registry.get(cfg.model.name)
        model.load_checkpoint(registry.checkpoint_path(cfg.model.name))
        model.to(device).eval()

        contract_path = Path(str(cfg.data.split.contract_path))
        if not contract_path.is_file():
            raise DataError(
                f"No split contract at {contract_path.name}. Build one before sweeping "
                "(SplitContract.build + .save)."
            )
        contract = SplitContract.load(contract_path)
        val_ids = list(contract.brats["val"])
        subset = cfg.data.get("eval_subset")
        if subset is not None:
            val_ids = val_ids[: int(subset)]

        dataset = BraTSDataset(cfg, val_ids)
        engine = ReconstructionEngine(model, cfg)

        results = []
        for volume_index in range(len(val_ids)):
            result = engine.run_dataset_volume(dataset, volume_index)
            if bool(cfg.recon.save_results):
                save_result(result, Path(str(cfg.recon.results_dir)), run.run_id)
            results.append(result)

        points = run_threshold_sweep(results, cfg.threshold.sweep, split="val")
        best = select_operating_point(points)

        results_dir = Path(str(cfg.paths.artifact_root)) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        csv_path = results_dir / f"threshold_sweep_{cfg.model.name}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "strategy",
                    "param",
                    "value",
                    "dice_mean",
                    "dice_std",
                    "n_volumes",
                    "split",
                ],
            )
            writer.writeheader()
            for p in points:
                writer.writerow(
                    {
                        "strategy": p.strategy,
                        "param": p.param,
                        "value": p.value,
                        "dice_mean": p.dice_mean,
                        "dice_std": p.dice_std,
                        "n_volumes": p.n_volumes,
                        "split": p.split,
                    }
                )

        summary_path = results_dir / "sweep_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "model": cfg.model.name,
                    "n_volumes": best.n_volumes,
                    "split": best.split,
                    "best_strategy": best.strategy,
                    "best_param": best.param,
                    "best_value": best.value,
                    "best_dice_mean": best.dice_mean,
                    "best_dice_std": best.dice_std,
                },
                indent=2,
            )
        )

        run.record(
            best_strategy=best.strategy,
            best_value=best.value,
            best_dice=best.dice_mean,
        )
        print(f"best: {best.strategy}={best.value} dice={best.dice_mean:.4f} (n={best.n_volumes})")


if __name__ == "__main__":
    main()
