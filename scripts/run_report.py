#!/usr/bin/env python
"""Spec 004/005/007: regenerate all tables/plots from saved metrics. ``make report``. No GPU/ML.

A thin orchestrator (Spec 007 R9 — Spec 011 owns full reporting; this only wires in the
sub-reports that already exist so `make report` regenerates their numbers too). Each sub-report is
guarded independently: a missing input degrades to a printed skip-reason, never a crash — running
`make report` on a partially-populated `artifacts/` tree must still finish and show what it could.

No ``seed_everything`` here — nothing in this script is stochastic, and importing
``mri_ad.utils.seed`` (via the ``mri_ad.utils`` package ``__init__``) would drag ``torch`` in,
breaking the "report-safe subgraph" acceptance test (8). ``RunLogger`` is imported from its own
submodule (``mri_ad.utils.run_logger``, which is itself torch-free) rather than from
``mri_ad.utils``, for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from mri_ad.eval.matrix import compute_stats as compute_matrix_stats
from mri_ad.eval.matrix import load_cells
from mri_ad.eval.matrix import render_csv as render_matrix_csv
from mri_ad.eval.matrix import render_markdown as render_matrix_markdown
from mri_ad.eval.matrix import write_stats as write_matrix_stats
from mri_ad.eval.paradigm import assert_same_samples, load_classical_column, load_dl_column
from mri_ad.eval.paradigm import compute_stats as compute_paradigm_stats
from mri_ad.eval.paradigm import plot_curves as plot_paradigm_curves
from mri_ad.eval.paradigm import render_csv as render_paradigm_csv
from mri_ad.eval.paradigm import render_markdown as render_paradigm_markdown
from mri_ad.eval.paradigm import write_stats as write_paradigm_stats
from mri_ad.eval.report import ReportGenerator
from mri_ad.exceptions import ArtifactError
from mri_ad.utils.run_logger import RunLogger


def _report_per_cell(cfg: DictConfig) -> None:
    # Must match run_eval.py's per-cell namespacing (`make eval` writes under
    # metrics_dir/<cell_id>/, figures_dir/<cell_id>/, cell_id = f"{model}__{loss}") — pass
    # model=<name> loss=<name> to report on a different cell than the config default.
    cell_id = f"{cfg.model.name}__{cfg.loss.name}"
    metrics_dir = Path(str(cfg.eval.metrics_dir)) / cell_id
    figures_dir = Path(str(cfg.eval.figures_dir)) / cell_id
    generator = ReportGenerator(metrics_dir, figures_dir)

    aggregate = generator.read_aggregate()
    rows = generator.read_per_volume()

    generator.plot_dice_distribution(rows)
    generator.write_summary_markdown()

    headline = aggregate["headline"]
    print(
        f"[cell {cell_id}] dice={headline['dice_mean']:.4f} iou={headline['iou_mean']:.4f} "
        f"(n={aggregate['n_volumes']})"
    )


def _report_matrix(cfg: DictConfig) -> None:
    cells = load_cells(cfg.matrix, Path(str(cfg.matrix.metrics_root)))
    stats = compute_matrix_stats(cells)
    tables_dir = Path(str(cfg.matrix.tables_dir))
    basename = str(cfg.matrix.basename)
    render_matrix_csv(cells, tables_dir / f"{basename}.csv")
    render_matrix_markdown(cells, stats, tables_dir / f"{basename}.md")
    write_matrix_stats(cells, stats, tables_dir / f"{basename}.json")
    print(f"[matrix] {stats.n_available}/{stats.n_cells} cells evaluated -> {tables_dir}")


def _report_paradigm(cfg: DictConfig) -> None:
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
    stats = compute_paradigm_stats(columns)

    tables_dir = Path(str(cfg.paradigm.tables_dir))
    figures_dir = Path(str(cfg.paradigm.figures_dir))
    basename = str(cfg.paradigm.basename)
    render_paradigm_markdown(columns, stats, tables_dir / f"{basename}.md")
    render_paradigm_csv(columns, tables_dir / f"{basename}.csv")
    write_paradigm_stats(columns, stats, tables_dir / f"{basename}.json")
    plot_paradigm_curves(columns, figures_dir / "paradigm_curves.png")
    print(f"[paradigm] {stats.n_available}/{stats.n_columns} columns available -> {tables_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Regenerate every sub-report from saved artifacts. A missing input skips, never crashes."""
    with RunLogger(cfg):
        for name, report_fn in (
            ("per-cell", _report_per_cell),
            ("arch x loss matrix", _report_matrix),
            ("paradigm comparison", _report_paradigm),
        ):
            try:
                report_fn(cfg)
            except ArtifactError as exc:
                print(f"[skip: {name}] {exc}")


if __name__ == "__main__":
    main()
