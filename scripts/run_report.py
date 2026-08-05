#!/usr/bin/env python
"""Spec 004: regenerate tables/plots from saved metrics. ``make report``. No GPU, no ML imports.

No ``seed_everything`` here — nothing in this script is stochastic, and importing
``mri_ad.utils.seed`` (via the ``mri_ad.utils`` package ``__init__``) would drag ``torch`` in,
breaking the "report-safe subgraph" acceptance test (8). ``RunLogger`` is imported from its own
submodule (``mri_ad.utils.run_logger``, which is itself torch-free) rather than from
``mri_ad.utils``, for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from mri_ad.eval.report import ReportGenerator
from mri_ad.utils.run_logger import RunLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Read ``artifacts/metrics/{per_volume.csv,aggregate.json}`` and regenerate the report."""
    with RunLogger(cfg) as run:
        # Must match run_eval.py's per-cell namespacing (`make eval` writes under
        # metrics_dir/<cell_id>/, figures_dir/<cell_id>/, cell_id = f"{model}__{loss}") — pass
        # model=<name> loss=<name> to report on a different cell than the config default.
        cell_id = f"{cfg.model.name}__{cfg.loss.name}"
        metrics_dir = Path(str(cfg.eval.metrics_dir)) / cell_id
        figures_dir = Path(str(cfg.eval.figures_dir)) / cell_id
        generator = ReportGenerator(metrics_dir, figures_dir)

        # read_aggregate/read_per_volume already raise ArtifactError with a
        # "Run `make eval` first" message when the metrics files are absent.
        aggregate = generator.read_aggregate()
        rows = generator.read_per_volume()

        generator.plot_dice_distribution(rows)
        generator.write_summary_markdown()

        headline = aggregate["headline"]
        run.record(dice_mean=headline["dice_mean"], iou_mean=headline["iou_mean"])
        print(
            f"dice={headline['dice_mean']:.4f} iou={headline['iou_mean']:.4f} "
            f"(n={aggregate['n_volumes']})"
        )


if __name__ == "__main__":
    main()
