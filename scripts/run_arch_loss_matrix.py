#!/usr/bin/env python
"""Spec 005: render the arch x loss matrix from saved ``aggregate.json`` files. `make matrix`.

Reads only ``artifacts/metrics/<cell_id>/aggregate.json`` (Spec 004's harness) — never touches
``checkpoints/``, ``data/``, or a ``.pt`` ``ReconResult``, and never constructs or calls a model.
No GPU, no ``seed_everything`` (nothing here is stochastic, and importing
``mri_ad.utils.seed`` would drag ``torch`` in, breaking the report-safe subgraph guard — same
reasoning as ``run_report.py``). Module-level imports are restricted to hydra/omegaconf,
``mri_ad.eval.matrix``, ``mri_ad.exceptions``, and ``mri_ad.utils.run_logger`` — the same
restriction ``run_report.py`` already lives under (enforced by ``tests/test_eval_boundary.py``).
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from mri_ad.eval.matrix import compute_stats, load_cells, render_csv, render_markdown, write_stats
from mri_ad.utils.run_logger import RunLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Render ``arch_loss_matrix.{csv,md,json}`` from saved ``aggregate.json`` files. No GPU."""
    with RunLogger(cfg) as run:
        cells = load_cells(cfg.matrix, Path(str(cfg.matrix.metrics_root)))
        stats = compute_stats(cells)
        tables_dir = Path(str(cfg.matrix.tables_dir))
        basename = str(cfg.matrix.basename)
        render_csv(cells, tables_dir / f"{basename}.csv")
        render_markdown(cells, stats, tables_dir / f"{basename}.md")
        write_stats(cells, stats, tables_dir / f"{basename}.json")
        run.record(
            n_cells=stats.n_cells,
            n_available=stats.n_available,
            spearman_psnr_dice=stats.spearman_psnr_dice,
        )
    print(f"{stats.n_available}/{stats.n_cells} cells evaluated -> {tables_dir}")


if __name__ == "__main__":
    main()
