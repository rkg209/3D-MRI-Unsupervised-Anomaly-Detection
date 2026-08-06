#!/usr/bin/env python
"""Spec 013 D-10: ``t_noise`` sweep on the BraTS VALIDATION split. GPU; manual invoke only.

For each ``t_noise`` candidate in ``cfg.model.sweep.t_noise_values``, reconstructs every val
volume at that noise level and runs the existing threshold sweep (``recon.sweep``) against it —
reusing ``build_sweep_strategies``/``run_threshold_sweep``/``select_operating_point`` unchanged,
same trap-#5 guard (``run_threshold_sweep`` refuses any split but ``"val"``). Writes one row per
``(t_noise, best_threshold)`` to ``artifacts/results/tnoise_sweep_diffusion.csv``, and the overall
winner to ``artifacts/results/tnoise_selection.json`` — a human then copies that winning
``t_noise`` into ``configs/model/diffusion.yaml`` by hand, same discipline as ``run_sweep.py``'s
threshold (never auto-written into config).

A dedicated script rather than a Hydra multirun over ``run_sweep.py``: that script writes one
fixed filename per model, so a ``t_noise`` multirun would silently overwrite its own results N
times and report only the last (D-10).

Run with ``make tnoise-sweep HYDRA_OVERRIDES="+experiment=cluster model=diffusion"``. Requires
real diffusion weights — run ``make check-data`` and train first (``make train
train=ddpm_scratch``).
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
from mri_ad.recon import ReconstructionEngine, run_threshold_sweep, select_operating_point
from mri_ad.recon.sweep import SweepPoint
from mri_ad.utils.device import DeviceManager
from mri_ad.utils.run_logger import RunLogger
from mri_ad.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Sweep ``t_noise`` x threshold on BraTS val; write the CSV + the winning combination."""
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    if str(cfg.model.name) != "diffusion":
        raise ConfigError(
            f"run_tnoise_sweep.py requires model=diffusion (got {cfg.model.name!r}) — t_noise "
            "is a diffusion-only knob."
        )
    if "sweep" not in cfg.model or "t_noise_values" not in cfg.model.sweep:
        raise ConfigError(
            "cfg.model.sweep.t_noise_values is absent from configs/model/diffusion.yaml."
        )
    if "sweep" not in cfg.threshold:
        raise ConfigError(
            "cfg.threshold.sweep is absent — run with the default threshold=percentile "
            "(configs/threshold/percentile.yaml), which carries the sweep axes."
        )

    with RunLogger(cfg) as run:
        device = DeviceManager.get_device(cfg.device)

        registry = build_default_registry()
        model = registry.get("diffusion")
        model.load_checkpoint(registry.checkpoint_path("diffusion"))
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
        # One engine, reused across t_noise values — only `model.t_noise` changes per iteration,
        # so the net's weights are never rebuilt, just its noising depth.
        engine = ReconstructionEngine(model, cfg)

        rows: list[tuple[int, SweepPoint]] = []
        for t_noise in cfg.model.sweep.t_noise_values:
            model.t_noise = int(t_noise)
            results = [engine.run_dataset_volume(dataset, i) for i in range(len(val_ids))]
            points = run_threshold_sweep(results, cfg.threshold.sweep, split="val")
            rows.append((int(t_noise), select_operating_point(points)))

        winner_t_noise, winner = max(rows, key=lambda row: row[1].dice_mean)

        results_dir = Path(str(cfg.paths.artifact_root)) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        csv_path = results_dir / "tnoise_sweep_diffusion.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "t_noise",
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
            for t_noise, best in rows:
                writer.writerow(
                    {
                        "t_noise": t_noise,
                        "strategy": best.strategy,
                        "param": best.param,
                        "value": best.value,
                        "dice_mean": best.dice_mean,
                        "dice_std": best.dice_std,
                        "n_volumes": best.n_volumes,
                        "split": best.split,
                    }
                )

        selection_path = results_dir / "tnoise_selection.json"
        selection_path.write_text(
            json.dumps(
                {
                    "t_noise": winner_t_noise,
                    "threshold_strategy": winner.strategy,
                    "threshold_param": winner.param,
                    "threshold_value": winner.value,
                    "dice_mean": winner.dice_mean,
                    "dice_std": winner.dice_std,
                    "n_volumes": winner.n_volumes,
                    "split": winner.split,
                },
                indent=2,
            )
        )

        run.record(
            best_t_noise=winner_t_noise,
            best_strategy=winner.strategy,
            best_value=winner.value,
            best_dice=winner.dice_mean,
        )
        print(
            f"best: t_noise={winner_t_noise} {winner.strategy}={winner.value} "
            f"dice={winner.dice_mean:.4f} (n={winner.n_volumes})"
        )


if __name__ == "__main__":
    main()
