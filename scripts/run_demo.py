#!/usr/bin/env python
"""Spec 010: export the demo video from a saved ``ReconResult``. ``make demo``.

No model is constructed, no checkpoint is read, no GPU is touched (acceptance test 2) — this
script reads persisted ``ReconResult`` .pt files exactly like ``make eval`` does. ``ipywidgets``
is never imported: the interactive path lives behind ``VolumeViewer.show_interactive`` and the
optional ``[viz]`` extra.

Run with ``make demo`` after ``make recon`` has populated a results directory, or
``make demo HYDRA_OVERRIDES="+viz.results_run_id=<run_id>"`` to target a specific run.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from mri_ad.eval.loader import read_manifest, resolve_results_dir
from mri_ad.utils.run_logger import RunLogger
from mri_ad.utils.seed import seed_everything
from mri_ad.viz import DemoExporter


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Resolve a results dir -> export the demo video -> record provenance."""
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    with RunLogger(cfg) as run:
        results_root = Path(str(cfg.recon.results_dir))
        results_dir = resolve_results_dir(results_root, run_id=cfg.viz.results_run_id)

        manifest = read_manifest(results_dir)
        if manifest is None:
            print(f"[demo] warning: no manifest under {results_dir.name}; split unknown.")
        else:
            print(f"[demo] run_id={results_dir.name} split={manifest.get('split')!r}")

        exporter = DemoExporter.from_cfg(cfg.viz)
        output_path = Path(str(cfg.viz.output_path))
        volume_id = cfg.viz.volume_id if cfg.viz.volume_id is not None else None

        export = exporter.export(results_dir, volume_id, output_path)

        run.record(
            volume_id=export.volume_id,
            output=str(export.path),
            n_frames=export.n_frames,
            fps=export.fps,
            frames_digest=export.frames_digest,
        )
        print(export.path)


if __name__ == "__main__":
    main()
