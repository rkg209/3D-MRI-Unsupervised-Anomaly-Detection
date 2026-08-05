#!/usr/bin/env python
"""Spec 007: render the headline three-paradigm comparison. ``make paradigm``. No GPU/ML imports.

Report-safe — reads only artifacts already written by ``make classical``
(``classical_metrics.json`` + ``oof_predictions.csv``) and ``make eval`` + ``make slice-scores``
(``aggregate.json`` + ``slice_scores.csv``). Mirrors ``scripts/run_arch_loss_matrix.py``'s
discipline.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from mri_ad.eval.paradigm import (
    assert_same_samples,
    compute_stats,
    load_classical_column,
    load_dl_column,
    plot_curves,
    render_csv,
    render_markdown,
    write_stats,
)
from mri_ad.utils.run_logger import RunLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Assemble the paradigm comparison from saved artifacts and write the table + figure."""
    with RunLogger(cfg) as run:
        paradigm_cfg = OmegaConf.to_container(cfg.paradigm, resolve=True)
        classical_metrics_dir = Path(str(cfg.paradigm.classical_metrics_dir))
        metrics_root = Path(str(cfg.paradigm.metrics_root))

        columns = []
        samples: dict[str, dict] = {}
        split_hashes: dict[str, str | None] = {}

        for col_cfg in paradigm_cfg["columns"]:
            if col_cfg["key"] == "classical":
                column, rows = load_classical_column(paradigm_cfg, classical_metrics_dir)
            else:
                column, rows = load_dl_column(paradigm_cfg, col_cfg, metrics_root)
            columns.append(column)
            samples[column.key] = rows
            split_hashes[column.key] = column.split_hash

        assert_same_samples(samples, split_hashes)
        stats = compute_stats(columns)

        tables_dir = Path(str(cfg.paradigm.tables_dir))
        figures_dir = Path(str(cfg.paradigm.figures_dir))
        basename = str(cfg.paradigm.basename)

        md_path = render_markdown(columns, stats, tables_dir / f"{basename}.md")
        render_csv(columns, tables_dir / f"{basename}.csv")
        write_stats(columns, stats, tables_dir / f"{basename}.json")
        plot_curves(columns, figures_dir / "paradigm_curves.png")

        run.record(
            n_available=stats.n_available,
            best_roc_auc_key=stats.best_roc_auc_key,
            best_dice_key=stats.best_dice_key,
        )
        print(md_path.read_text())


if __name__ == "__main__":
    main()
