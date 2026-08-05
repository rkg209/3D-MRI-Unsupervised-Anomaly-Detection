#!/usr/bin/env python
"""Spec 006: classical-ML baseline — radiomic features -> gradient boosting -> CV. CPU only.

Feature extraction is ~10-20 CPU-minutes and is user-invoked, not autonomous, per CLAUDE.md.
Module-level imports stay inside hydra/omegaconf, ``mri_ad.classical.*``, ``mri_ad.data.split``,
``mri_ad.exceptions``, and ``mri_ad.utils.{run_logger,seed}`` — never ``models``/``recon``/``eval``
(acceptance 7).

Run with ``make classical`` after ``make check-data``. ``+experiment=laptop`` caps the subject
pool via ``classical.subject_limit`` for a fast smoke run.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from mri_ad.classical.baseline import ClassicalBaseline
from mri_ad.classical.cache import FeatureCache, FeatureCacheHeader
from mri_ad.classical.dataset import build_slice_dataset
from mri_ad.classical.features import FeatureConfig, FeatureExtractor, feature_names
from mri_ad.classical.metrics import GRANULARITY
from mri_ad.classical.report import ClassicalReportGenerator
from mri_ad.data.split import resolve_contract
from mri_ad.utils.run_logger import RunLogger
from mri_ad.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Extract slice-level features on the BraTS test split, cross-validate, write the report."""
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    with RunLogger(cfg) as run:
        contract = resolve_contract(
            cfg, build_if_missing=bool(cfg.classical.split.build_contract_if_missing)
        )
        pool = list(contract.brats["test"])  # acceptance 5: this list, and nothing else

        subject_limit = cfg.classical.subject_limit
        complete = subject_limit is None or int(subject_limit) >= len(pool)
        if subject_limit is not None:
            pool = pool[: int(subject_limit)]

        feature_config = FeatureConfig.from_cfg(cfg)
        extractor = FeatureExtractor(feature_config)
        names = tuple(feature_names(feature_config))

        header = FeatureCacheHeader(
            granularity=GRANULARITY,
            split_hash=contract.content_hash(),
            seed=int(cfg.seed),
            feature_hash=feature_config.content_hash(),
            feature_names=names,
        )

        counters: dict[str, int] = {}
        data = None
        cache_enabled = bool(cfg.classical.cache.enabled)
        cache = FeatureCache(Path(str(cfg.classical.cache.dir)))
        if cache_enabled and not bool(cfg.classical.cache.refresh):
            cached = cache.load(header)
            if cached is not None:
                data, counters = cached

        if data is None:
            data, counters = build_slice_dataset(cfg, pool, extractor)
            if cache_enabled:
                cache.save(header, data, counters=counters)

        baseline = ClassicalBaseline(cfg)
        metrics = baseline.cross_validate(data)

        report = ClassicalReportGenerator(
            Path(str(cfg.classical.report.metrics_dir)), Path(str(cfg.classical.report.figures_dir))
        )
        report.write_per_fold(metrics)
        report.write_metrics_json(
            metrics,
            run_id=run.run_id,
            seed=int(cfg.seed),
            complete=complete,
            subject_limit=subject_limit,
            split_info={
                "source": str(cfg.classical.split.source),
                "split_seed": contract.seed,
                "split_hash": contract.content_hash(),
                "n_subjects": len(pool),
                "cv": OmegaConf.to_container(cfg.classical.split.cv, resolve=True),
            },
            features_info={
                "n_features": len(names),
                "names": list(names),
                "config_hash": feature_config.content_hash(),
                "library": feature_config.library,
                "pyradiomics_used": feature_config.pyradiomics_used,
                "substitution_note": (
                    "PyRadiomics is fragile to build on some platforms; this baseline uses "
                    "scikit-image GLCM/first-order/gradient features plus a hand-rolled GLRLM "
                    "instead (skimage ships no run-length equivalent)."
                ),
                "n_sanitized_values": extractor.sanitized_count,
                "cache_path": str(cache._path(header)) if cache_enabled else None,
            },
            counts_info=counters,
            classifier_info={
                "name": str(cfg.classical.classifier.name),
                "params": OmegaConf.to_container(cfg.classical.classifier.params, resolve=True),
                "scale_pos_weight_per_fold": [fold.scale_pos_weight for fold in metrics.folds],
            },
        )
        report.write_feature_importance(baseline.feature_importance)
        report.write_summary_markdown()
        report.plot_class_balance(
            {
                "n_positive": metrics.n_positive,
                "n_negative": metrics.n_samples - metrics.n_positive,
            }
        )
        report.plot_fold_auc(metrics)

        run.record(
            roc_auc_mean=metrics.roc_auc_mean,
            pr_auc_mean=metrics.pr_auc_mean,
            positive_rate=metrics.positive_rate,
            granularity=GRANULARITY,
        )


if __name__ == "__main__":
    main()
