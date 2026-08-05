#!/usr/bin/env python
"""Spec 004: score the test-split ``ReconResult``s persisted by ``make recon``. `make eval`.

Normal mode (default) never constructs or calls a model — it reads persisted
:class:`~mri_ad.recon.types.ReconResult` files and scores them via ``mri_ad.eval.evaluator``.
Module-level imports are deliberately restricted to hydra/omegaconf, ``mri_ad.eval.*``,
``mri_ad.exceptions``, and ``mri_ad.utils`` — never ``mri_ad.models`` or ``mri_ad.recon.engine``
(enforced by ``tests/test_eval_boundary.py``'s AST walk). Legacy-compat mode
(``LEGACY_BUG_COMPAT=1`` / ``eval=legacy_compat``) is the one path that touches a model, and it
does so via a **function-local** import inside ``_evaluate_legacy`` — never at module scope.

Run with ``make eval HYDRA_OVERRIDES="+eval.results_run_id=<run_id> model=unetr"`` after
``make recon`` has produced a run directory. ``make eval LEGACY_BUG_COMPAT=1`` reproduces the
prior work's bugs on purpose (never publish that number — see ``mri_ad.eval.legacy``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from mri_ad.eval.evaluator import AggregateEvaluator
from mri_ad.eval.loader import read_manifest, resolve_results_dir
from mri_ad.eval.metrics import aggregate as aggregate_metrics
from mri_ad.eval.report import ReportGenerator
from mri_ad.exceptions import ArtifactError
from mri_ad.utils.run_logger import RunLogger
from mri_ad.utils.seed import seed_everything


def _per_volume_row(volume_metrics: Any) -> dict[str, object]:
    """``VolumeMetrics`` -> the plain mapping ``report.py`` expects (context-only key rename)."""
    d = dataclasses.asdict(volume_metrics)
    return {
        "volume_id": d["volume_id"],
        "dice": d["dice"],
        "iou": d["iou"],
        "psnr_db_context_only": d["psnr"],
        "ssim_context_only": d["ssim"],
    }


def _check_manifest(results_dir: Path, cfg: DictConfig) -> None:
    """Refuse a results directory whose manifest says it isn't a test-split run for this cell.

    The only guard against ``resolve_results_dir``'s newest-mtime fallback silently picking up a
    val-split ``run_sweep.py`` run (trap #5 wearing a new costume). A directory with no manifest
    (hand-built, or written before this check existed) is allowed through as before. The loss
    check (Spec 005 D-A) stops a results dir produced under one loss from being scored as if it
    belonged to a different cell.
    """
    manifest = read_manifest(results_dir)
    if manifest is None:
        return
    if manifest.get("split") != "test":
        raise ArtifactError(
            f"{results_dir} is a {manifest.get('split')!r}-split run (manifest.json), not test. "
            "Pass +eval.results_run_id=<run_id> for a run produced by `make recon`."
        )
    if manifest.get("model") != str(cfg.model.name):
        raise ArtifactError(
            f"{results_dir} was produced by model {manifest.get('model')!r}, not "
            f"{cfg.model.name!r} (manifest.json). Pass +eval.results_run_id=<run_id> for the "
            "right model's `make recon` run, or set model=<the model that produced it>."
        )
    if "loss" not in manifest:
        raise ArtifactError(
            f"{results_dir}/manifest.json predates the (model x loss) cell namespacing and "
            "records no loss. Re-run `make recon` for this cell; a pre-Spec-005 manifest "
            f"cannot be attributed to {cfg.loss.name!r} after the fact."
        )
    if manifest["loss"] != str(cfg.loss.name):
        raise ArtifactError(
            f"{results_dir} was produced with loss {manifest['loss']!r}, not "
            f"{cfg.loss.name!r} (manifest.json). Pass +eval.results_run_id=<run_id> for the "
            "right cell's `make recon` run, or set loss=<the loss that produced it>."
        )


def _evaluate_normal(cfg: DictConfig, *, run_id: str) -> dict[str, float]:
    """Score the resolved results directory. Never builds or calls a model."""
    results_dir = resolve_results_dir(
        Path(str(cfg.recon.results_dir)), run_id=cfg.eval.get("results_run_id")
    )
    _check_manifest(results_dir, cfg)
    labels_dir = cfg.eval.get("labels_dir")
    labels_dir = Path(str(labels_dir)) if labels_dir else None

    evaluator = AggregateEvaluator()
    volumes = evaluator.evaluate_volumes(results_dir, labels_dir=labels_dir)
    agg = aggregate_metrics(volumes)

    rows = [_per_volume_row(v) for v in volumes]
    agg_payload = dataclasses.asdict(agg)
    agg_payload["published_dice"] = float(cfg.eval.legacy.published_dice)

    # Namespaced by cell_id (Spec 005 D-A: f"{model}__{loss}") so `make eval model=unetr
    # loss=mse` and `make eval model=unetr loss=mse_ssim` cannot silently overwrite each other's
    # per_volume.csv/aggregate.json/summary.md — a real risk once the (model x loss) matrix
    # exists.
    cell_id = f"{cfg.model.name}__{cfg.loss.name}"
    metrics_dir = Path(str(cfg.eval.metrics_dir)) / cell_id
    figures_dir = Path(str(cfg.eval.figures_dir)) / cell_id
    generator = ReportGenerator(metrics_dir, figures_dir)
    generator.write_per_volume(rows)
    generator.write_aggregate(
        agg_payload,
        mode="normal",
        model=str(cfg.model.name),
        loss=str(cfg.loss.name),
        split="test",
        n_volumes=agg.n_volumes,
        run_id=run_id,
        results_run_id=results_dir.name,
    )
    generator.plot_dice_distribution(rows)
    generator.write_summary_markdown()

    return {
        "dice_mean": agg.dice_mean,
        "dice_std": agg.dice_std,
        "iou_mean": agg.iou_mean,
        "iou_std": agg.iou_std,
        "n_volumes": agg.n_volumes,
    }


def _evaluate_legacy(cfg: DictConfig, *, run_id: str) -> dict[str, float]:
    """Reproduce ``legacy/metric-uad.ipynb``'s bugs on purpose. NEVER publish this number.

    Imports a model + device only here, function-local — the one path in this script that
    touches weights or BraTS raw volumes directly.
    """
    import json

    from mri_ad.eval.legacy import LegacyCompatConfig, LegacyCompatEvaluator
    from mri_ad.models import build_default_registry
    from mri_ad.utils.device import DeviceManager

    device = DeviceManager.get_device(str(cfg.get("device", "auto")))
    registry = build_default_registry()
    model = registry.get(cfg.model.name)
    model.load_checkpoint(registry.checkpoint_path(cfg.model.name))
    model.to(device).eval()

    legacy_cfg = LegacyCompatConfig(
        subject_count=int(cfg.eval.legacy.subject_count),
        chunk_index=int(cfg.eval.legacy.chunk_index),
        threshold=float(cfg.eval.legacy.threshold),
        crop_size=tuple(cfg.eval.legacy.crop_size),
        sort_subjects=bool(cfg.eval.legacy.sort_subjects),
    )
    evaluator = LegacyCompatEvaluator(model, legacy_cfg, device)
    result = evaluator.evaluate_directory(Path(str(cfg.data.brats.dir)))

    cell_id = f"{cfg.model.name}__{cfg.loss.name}"
    metrics_dir = Path(str(cfg.eval.metrics_dir)) / cell_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(result)
    payload["run_id"] = run_id
    payload["model"] = str(cfg.model.name)
    payload["loss"] = str(cfg.loss.name)
    (metrics_dir / "legacy_compat.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    return {"legacy_dice": result.dice, "legacy_iou": result.iou}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Score saved test-split ``ReconResult``s. LEGACY_BUG_COMPAT=1 for the bug-compat mode."""
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    with RunLogger(cfg) as run:
        if bool(cfg.eval.legacy_compat):
            summary = _evaluate_legacy(cfg, run_id=run.run_id)
        else:
            summary = _evaluate_normal(cfg, run_id=run.run_id)
        run.record(**summary)
    print(summary)


if __name__ == "__main__":
    main()
