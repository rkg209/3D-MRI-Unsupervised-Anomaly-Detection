#!/usr/bin/env python
"""Spec 004 (D-A): reconstruct the TEST split -> saved ``ReconResult``s. GPU; manual invoke only.

``run_sweep.py`` only ever persists **val**-split results (trap #5 — never tune a threshold on
test). Nothing upstream of Spec 004 produces test-split results, and ``run_eval.py`` never
constructs a model — so this script is the one place that turns "test split + checkpoint" into
the ``artifacts/results/recon/<run_id>/*.pt`` files ``make eval`` scores. Structurally a clone of
``run_sweep.py`` minus the threshold sweep.

Run with ``make recon HYDRA_OVERRIDES="+experiment=cluster model=unetr"`` after
``make check-data``. Prints the ``run_id`` — pass it back to ``make eval`` as
``+eval.results_run_id=<run_id>`` (or leave it unset to let ``make eval`` pick the newest run).
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mri_ad.data.datasets import BraTSDataset
from mri_ad.data.split import SplitContract
from mri_ad.eval.loader import MANIFEST_FILENAME
from mri_ad.exceptions import DataError
from mri_ad.models import build_default_registry
from mri_ad.recon import ReconstructionEngine, save_result
from mri_ad.utils.device import DeviceManager
from mri_ad.utils.run_logger import RunLogger
from mri_ad.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Reconstruct every TEST-split BraTS volume and persist a ``ReconResult`` for each."""
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    with RunLogger(cfg) as run:
        device = DeviceManager.get_device(cfg.device)

        registry = build_default_registry()
        model = registry.get(cfg.model.name)
        model.load_checkpoint(registry.checkpoint_path(cfg.model.name))
        model.to(device).eval()

        contract_path = Path(str(cfg.data.split.contract_path))
        if not contract_path.is_file():
            raise DataError(
                f"No split contract at {contract_path.name}. Build one before reconstructing "
                "(SplitContract.build + .save)."
            )
        contract = SplitContract.load(contract_path)
        test_ids = list(contract.brats["test"])
        subset = cfg.data.get("eval_subset")
        if subset is not None:
            test_ids = test_ids[: int(subset)]

        dataset = BraTSDataset(cfg, test_ids)
        engine = ReconstructionEngine(model, cfg)

        results_dir = Path(str(cfg.recon.results_dir))
        for volume_index in range(len(test_ids)):
            with run.timer("inference"):
                result = engine.run_dataset_volume(dataset, volume_index)
            save_result(result, results_dir, run.run_id)

        # run_sweep.py persists val-split results under this same results_dir root with no other
        # record of which split produced a run directory — this manifest is what lets `make eval`
        # refuse to silently score a val-split (or wrong-model/wrong-loss) run as if it belonged
        # to a different cell (Spec 005 D-A: loss added alongside model).
        manifest = {
            "model": str(cfg.model.name),
            "loss": str(cfg.loss.name),
            "split": "test",
            "n_volumes": len(test_ids),
            "split_hash": contract.content_hash(),
        }
        (results_dir / run.run_id / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )

        inference_seconds = run.metrics.get("timings", {}).get("inference", 0.0)
        seconds_per_volume = inference_seconds / len(test_ids) if test_ids else None
        gpu_hours = inference_seconds / 3600 if device.type.startswith("cuda") else None
        run.record(
            model=cfg.model.name,
            loss=cfg.loss.name,
            split="test",
            n_volumes=len(test_ids),
            device=str(device),
            seconds_per_volume=seconds_per_volume,
            gpu_hours=gpu_hours,
        )
        print(run.run_id)


if __name__ == "__main__":
    main()
