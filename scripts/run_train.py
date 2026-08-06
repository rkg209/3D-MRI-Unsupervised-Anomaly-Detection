#!/usr/bin/env python
"""Spec 009: fine-tune UNETR to restore FPI-corrupted volumes. GPU-spending.

**MANUAL INVOKE ONLY** — the agent must never launch this on its own initiative (CLAUDE.md rule
4). Reachable only via ``make train`` / the ``/train`` skill, which carries
``disable-model-invocation: true``.

Preflight, in order: a CUDA-device guard (refuses an accidental laptop launch unless
``train.allow_cpu=true``), the Spec 009 acceptance-2 real-data separability check (aborts
*before the first optimizer step* if FPI corruption turns out to be trivially separable by a
global intensity threshold), then ``Trainer.fit()``.

Run with ``make train HYDRA_OVERRIDES="+experiment=cluster"`` after ``make check-data``.
"""

from __future__ import annotations

import functools
import importlib
import json
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mri_ad.data.datasets import BraTSDataset, OpenBHBDataset
from mri_ad.data.split import resolve_contract
from mri_ad.eval.metrics import MetricsComputer
from mri_ad.exceptions import ArtifactError, ConfigError
from mri_ad.models import AnomalyDetectionModel, ModelRegistry, build_default_registry
from mri_ad.recon.engine import ReconstructionEngine
from mri_ad.synth.dataset import AnomalyInformedDataset
from mri_ad.synth.separability import best_intensity_dice
from mri_ad.train import CheckpointWriter, Trainer, reconstruction_step, resolve_training_ids
from mri_ad.utils.device import DeviceManager
from mri_ad.utils.instantiate import instantiate_from_config
from mri_ad.utils.run_logger import RunLogger
from mri_ad.utils.seed import seed_everything


def _git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def _require_cuda(cfg: DictConfig) -> torch.device:
    device = DeviceManager.get_device(str(cfg.device))
    if device.type != "cuda" and not bool(cfg.train.allow_cpu):
        raise ConfigError(
            f"run_train.py requires a CUDA device (resolved {device.type!r}); this refuses an "
            "accidental laptop launch (CLAUDE.md rule 4). Set train.allow_cpu=true to override "
            "for local development only."
        )
    return device


def _build_loader(
    dataset: torch.utils.data.Dataset, cfg: DictConfig, *, shuffle: bool, seed: int
) -> torch.utils.data.DataLoader:
    import monai.data.utils

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.data.loader.batch_size),
        num_workers=int(cfg.data.loader.num_workers),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=monai.data.utils.worker_init_fn,
        persistent_workers=False,
    )


def _preflight_separability_check(train_dataset: AnomalyInformedDataset, cfg: DictConfig) -> dict:
    """Acceptance 2, on real data, before the first optimizer step (R5)."""
    n_samples = min(int(cfg.synth.check.n_samples), len(train_dataset))
    foreground_threshold = float(cfg.synth.params.foreground_threshold)
    n_thresholds = int(cfg.synth.check.n_thresholds)

    dices = []
    for index in range(n_samples):
        item = train_dataset[index]
        dices.append(
            best_intensity_dice(
                item["corrupted"],
                item["synth_mask"],
                foreground_threshold=foreground_threshold,
                n_thresholds=n_thresholds,
            )
        )

    report = {
        "n_samples": n_samples,
        "mean_intensity_dice": statistics.fmean(dices) if dices else 0.0,
        "max_intensity_dice": max(dices) if dices else 0.0,
        "max_intensity_dice_bound": float(cfg.synth.check.max_intensity_dice),
    }

    artifact_root = Path(str(cfg.paths.artifact_root))
    out_path = artifact_root / "synth" / "separability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    if bool(cfg.synth.check.abort_on_failure) and (
        report["mean_intensity_dice"] > report["max_intensity_dice_bound"]
        or report["max_intensity_dice"] > report["max_intensity_dice_bound"]
    ):
        raise ArtifactError(
            "Pre-flight separability check failed: FPI corruption is trivially separable by a "
            f"global intensity threshold (mean={report['mean_intensity_dice']:.3f}, "
            f"max={report['max_intensity_dice']:.3f}, bound="
            f"{report['max_intensity_dice_bound']:.3f}). See {out_path}. Aborting before any "
            "GPU-hours are spent (Spec 009 R5)."
        )
    return report


def _make_validate_fn(
    model: nn.Module, cfg: DictConfig, brats_val_ids: list[str]
) -> Callable[[nn.Module, int], float | None]:
    """BraTS-*val* Dice, constructed here (not in ``train/``).

    So ``Trainer`` never imports ``recon``/``eval`` — precedent: ``scripts/run_recon.py``
    already imports both.
    """
    engine = ReconstructionEngine(model, cfg)
    dataset = BraTSDataset(cfg, brats_val_ids)
    n_volumes = min(int(cfg.train.selection.n_val_volumes), len(brats_val_ids))
    every_n = int(cfg.train.selection.dice_every_n_epochs)

    def validate_fn(model: nn.Module, epoch: int) -> float | None:
        if epoch % every_n != 0 or n_volumes == 0:
            return None
        was_training = model.training
        model.eval()
        dices = []
        with torch.no_grad():
            for volume_index in range(n_volumes):
                result = engine.run_dataset_volume(dataset, volume_index)
                dices.append(
                    MetricsComputer.dice(result.anomaly_mask, (result.ground_truth > 0).float())
                )
        model.train(was_training)
        return statistics.fmean(dices) if dices else None

    return validate_fn


def _build_and_warm_start_model(registry: ModelRegistry, cfg: DictConfig) -> AnomalyDetectionModel:
    """Build the model named by ``cfg.train.save_as``/``init_from`` and load its starting weights.

    Spec 012 (STRETCH): ``warm_start_from`` and ``init_from`` are mutually exclusive. The gated
    ``msa_unetr`` variant cannot load a plain-UNETR checkpoint under ``strict=True`` (it has extra
    gate keys), so it warm-starts via ``load_from_unetr`` — a key-partitioned, still-strict load
    (D4) — instead of the ordinary ``load_checkpoint`` path every other fine-tune uses.
    """
    warm_start_from = cfg.train.get("warm_start_from")
    init_from = cfg.train.get("init_from")
    if warm_start_from and init_from:
        raise ConfigError(
            "train.warm_start_from and train.init_from are mutually exclusive "
            f"(got warm_start_from={warm_start_from!r}, init_from={init_from!r})."
        )
    if warm_start_from:
        model = registry.get(str(cfg.train.save_as))
        model.load_from_unetr(registry.checkpoint_path(str(warm_start_from)))
        return model
    model = registry.get(str(init_from))
    model.load_checkpoint(registry.checkpoint_path(str(init_from)))
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Fine-tune ``cfg.train.init_from``, saving under ``cfg.train.save_as``.

    Or warm-start ``cfg.train.warm_start_from`` (Spec 012) on the same FPI corruption objective.
    """
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = _require_cuda(cfg)

    with RunLogger(cfg) as run:
        contract = resolve_contract(cfg, build_if_missing=False)
        ids = resolve_training_ids(cfg, contract)

        generator = instantiate_from_config(cfg.synth)
        train_base = OpenBHBDataset(cfg, list(ids.openbhb_train))
        val_base = OpenBHBDataset(cfg, list(ids.openbhb_val))
        train_dataset = AnomalyInformedDataset(
            train_base,
            generator,
            seed=int(cfg.seed),
            epoch_invariant=False,
            donor_chunk=str(cfg.synth.donor_chunk),
        )
        val_dataset = AnomalyInformedDataset(
            val_base,
            generator,
            seed=int(cfg.seed),
            epoch_invariant=True,
            donor_chunk=str(cfg.synth.donor_chunk),
        )

        _preflight_separability_check(train_dataset, cfg)

        registry = build_default_registry()
        model = _build_and_warm_start_model(registry, cfg)
        model.to(device)

        train_loader = _build_loader(train_dataset, cfg, shuffle=True, seed=int(cfg.seed))
        val_loader = _build_loader(val_dataset, cfg, shuffle=False, seed=int(cfg.seed))

        criterion = instantiate_from_config(cfg.loss)

        # Optimizer/scheduler need `model.parameters()`/`optimizer` as their first positional arg,
        # which `instantiate_from_config`'s `{target, params}` -> `cls(**params)` shape doesn't
        # thread through — so they're built by hand here rather than forcing that shape.
        optimizer_cfg = cfg.train.optimizer
        opt_module, _, opt_cls_name = str(optimizer_cfg.target).rpartition(".")
        optimizer_cls = getattr(importlib.import_module(opt_module), opt_cls_name)
        optimizer = optimizer_cls(
            model.parameters(), **OmegaConf.to_container(optimizer_cfg.params, resolve=True)
        )

        scheduler_cfg = cfg.train.scheduler
        sched_module, _, sched_cls_name = str(scheduler_cfg.target).rpartition(".")
        scheduler_cls = getattr(importlib.import_module(sched_module), sched_cls_name)
        scheduler = scheduler_cls(
            optimizer, **OmegaConf.to_container(scheduler_cfg.params, resolve=True)
        )

        validate_fn = _make_validate_fn(model, cfg, list(ids.brats_select))

        checkpoint_writer = CheckpointWriter(
            Path(str(cfg.train.checkpoint_path)),
            meta={
                "selection_key": str(cfg.train.selection.metric),
                "git_sha": _git_sha(),
                "seed": int(cfg.seed),
                "split_hash": contract.content_hash(),
                "synth": OmegaConf.to_container(cfg.synth.params, resolve=True),
                "train": {"init_from": str(cfg.train.init_from), "save_as": str(cfg.train.save_as)},
            },
        )

        trainer = Trainer(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            step_fn=functools.partial(reconstruction_step, criterion=criterion),
            device=device,
            max_epochs=int(cfg.train.max_epochs),
            grad_clip_norm=float(cfg.train.grad_clip_norm),
            early_stopping_patience=int(cfg.train.early_stopping_patience),
            selection_key=str(cfg.train.selection.metric),
            selection_mode="max" if str(cfg.train.selection.metric) == "val_dice" else "min",
            on_epoch_start=train_dataset.set_epoch,
            validate_fn=validate_fn,
            checkpoint_writer=checkpoint_writer,
        )
        summary = trainer.fit()

        run.record(
            best_epoch=summary.best_epoch,
            best_score=summary.best_score,
            selection_key=summary.selection_key,
            param_l2_delta=summary.param_l2_delta,
            checkpoint_path=str(summary.checkpoint_path.name),
        )
        print(run.run_id)


if __name__ == "__main__":
    main()
