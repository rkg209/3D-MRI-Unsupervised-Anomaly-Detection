#!/usr/bin/env python
"""Spec 007: reduce one saved DL recon run to slice-level scores. ``make slice-scores``.

Reads persisted ``ReconResult``s (``.pt`` files) and writes ``slice_scores.csv`` — no model is
constructed, no GPU is touched, but this script imports ``torch`` (via ``eval/slicelevel.py``) so
it stays outside the report-safe subgraph on purpose (``run_paradigm_comparison.py`` is the
report-safe consumer of its output).

A **test**-split run writes into ``eval.metrics_dir/<cell_id>/slice_scores.csv`` — the location
``eval/paradigm.py::load_dl_column`` reads. A **val**-split run (from ``make sweep``) writes into
``artifacts/results/slice_threshold_sweep_<cell_id>.csv`` instead, via the threshold tuner — never
into the same path a test-split run would occupy, so a val-tuning run can never masquerade as a
headline number.

Run with ``make slice-scores HYDRA_OVERRIDES="model=unetr loss=mse_ssim
+eval.results_run_id=<run_id>"`` after ``make recon`` (test) or ``make sweep`` (val) has produced
a run directory.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from mri_ad.eval.loader import read_manifest, resolve_results_dir
from mri_ad.eval.slicelevel import (
    reduce_results_dir,
    tune_slice_threshold,
    write_slice_scores,
)
from mri_ad.exceptions import ArtifactError
from mri_ad.utils.run_logger import RunLogger
from mri_ad.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Reduce the resolved results directory's ``ReconResult``s to slice-level scores."""
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    with RunLogger(cfg) as run:
        results_dir = resolve_results_dir(
            Path(str(cfg.recon.results_dir)), run_id=cfg.eval.get("results_run_id")
        )
        manifest = read_manifest(results_dir)
        if manifest is None:
            raise ArtifactError(
                f"{results_dir} has no manifest.json; slice reduction needs to know its split."
            )
        split = str(manifest["split"])
        cell_id = f"{cfg.model.name}__{cfg.loss.name}"
        foreground_eps = float(cfg.classical.features.foreground_eps)

        rows = reduce_results_dir(results_dir, foreground_eps=foreground_eps)

        if split == "test":
            out_path = Path(str(cfg.eval.metrics_dir)) / cell_id / "slice_scores.csv"
            write_slice_scores(rows, out_path)
            run.record(split=split, cell_id=cell_id, n_slices=len(rows), out=str(out_path))
            print(f"{len(rows)} slice scores -> {out_path}")
        elif split == "val":
            grid = OmegaConf.to_container(cfg.paradigm.threshold_grid, resolve=True)
            points = tune_slice_threshold(rows, grid, split="val")
            out_path = (
                Path(str(cfg.paradigm.tables_dir)).parent
                / "results"
                / (f"slice_threshold_sweep_{cell_id}.csv")
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import csv

            with out_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["threshold", "precision", "recall", "f1", "split"])
                for point in points:
                    writer.writerow(
                        [point.threshold, point.precision, point.recall, point.f1, point.split]
                    )
            run.record(split=split, cell_id=cell_id, n_slices=len(rows), out=str(out_path))
            print(f"{len(points)} threshold points -> {out_path}")
        else:
            raise ArtifactError(f"{results_dir}: unrecognized split {split!r} in manifest.json.")


if __name__ == "__main__":
    main()
