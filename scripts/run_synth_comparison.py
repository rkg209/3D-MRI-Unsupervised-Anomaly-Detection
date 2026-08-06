#!/usr/bin/env python
"""Spec 009: render the synthetic-anomaly before/after table. `make synth-table`.

Report-safe: reads only ``artifacts/metrics/<cell_id>/aggregate.json`` (Spec 004) and
``artifacts/runs/<run_id>/run_meta.json`` (Spec 000) — no model, no GPU, no torch/monai import.
Before either eval run exists (the state on a clean checkout), this renders an honest "n/a" table
rather than crashing, mirroring ``run_report.py``/``run_paradigm_comparison.py``'s "independently
guarded" pattern — a missing input prints a skip-reason, it never breaks `make report`.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from mri_ad.eval.before_after import assert_controlled, load_cell, render_csv, render_markdown
from mri_ad.exceptions import ArtifactError
from mri_ad.utils.run_logger import RunLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compare ``cfg.train.init_from`` (before) against ``cfg.train.save_as`` (after)."""
    metrics_root = Path(str(cfg.eval.metrics_dir))
    runs_root = Path(str(cfg.paths.artifact_root)) / "runs"
    tables_dir = Path(str(cfg.paths.artifact_root)) / "tables"

    before_model = str(cfg.train.init_from)
    after_model = str(cfg.train.save_as)
    loss = str(cfg.loss.name)

    with RunLogger(cfg) as run:
        try:
            before = load_cell(metrics_root, runs_root, model=before_model, loss=loss)
            after = load_cell(metrics_root, runs_root, model=after_model, loss=loss)
            assert_controlled(before, after)
        except ArtifactError as exc:
            tables_dir.mkdir(parents=True, exist_ok=True)
            (tables_dir / "synth_before_after.md").write_text(
                f"# Spec 009 — synthetic-anomaly fine-tune: before/after\n\nn/a — {exc}\n"
            )
            run.record(available=False, reason=str(exc))
            print(f"[skip: synth before/after] {exc}")
            return

        tables_dir.mkdir(parents=True, exist_ok=True)
        md_path = tables_dir / "synth_before_after.md"
        md_path.write_text(render_markdown(before, after))
        render_csv(before, after, tables_dir / "synth_before_after.csv")

        run.record(
            available=True,
            before_dice=before.dice_mean,
            after_dice=after.dice_mean,
            delta=after.dice_mean - before.dice_mean,
        )
        print(f"before dice={before.dice_mean:.4f} after dice={after.dice_mean:.4f} -> {md_path}")


if __name__ == "__main__":
    main()
