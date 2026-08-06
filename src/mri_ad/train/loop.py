"""The training loop (Spec 009 D6): fit a model, select a checkpoint, prove it moved.

Deliberately hand-rolled (~150 lines) rather than MONAI ``SupervisedTrainer``/Ignite or
Lightning: per-item corruption seeding needs ``dataset.set_epoch(epoch)`` called at an exact
point every epoch, and framework callback ordering makes that awkward to guarantee. ``Trainer``
never imports ``mri_ad.recon`` or ``mri_ad.eval`` — a boundary test enforces this — so Spec 013's
from-scratch DDPM loop can reuse this file with only a new ``StepFn`` (``objectives.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from mri_ad.exceptions import ConfigError, TrainError
from mri_ad.train.checkpointing import CheckpointWriter

StepFn = Callable[[nn.Module, dict[str, Tensor], torch.device], Tensor]
ValidateFn = Callable[[nn.Module, int], "float | None"]
OnEpochStart = Callable[[int], None]


@dataclass(frozen=True)
class EpochRecord:
    """One epoch's bookkeeping — train/val loss, whatever ``validate_fn`` produced, and the LR."""

    epoch: int
    train_loss: float
    val_loss: float
    extra: dict[str, float] = field(default_factory=dict)
    lr: float = 0.0


@dataclass(frozen=True)
class TrainSummary:
    """What :meth:`Trainer.fit` returns: the selected checkpoint and proof training happened."""

    best_epoch: int
    best_score: float
    selection_key: str
    history: tuple[EpochRecord, ...]
    checkpoint_path: Path
    param_l2_delta: float  # L2 distance from the init weights (R6) — 0.0 means nothing trained


def _assert_not_persistent(loader: DataLoader, *, name: str) -> None:
    if getattr(loader, "persistent_workers", False):
        raise ConfigError(
            f"{name}.persistent_workers must be False — with persistent workers, "
            "dataset.set_epoch() never propagates to worker processes and every epoch silently "
            "reuses epoch 0's corruptions (Spec 009 R7)."
        )


class Trainer:
    """Fits ``model`` for up to ``max_epochs``, early-stopping on ``selection_key``."""

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None,
        train_loader: DataLoader,
        val_loader: DataLoader,
        step_fn: StepFn,
        device: torch.device,
        max_epochs: int,
        grad_clip_norm: float,
        early_stopping_patience: int,
        selection_key: str,
        selection_mode: str,
        on_epoch_start: OnEpochStart | None = None,
        validate_fn: ValidateFn | None = None,
        checkpoint_writer: CheckpointWriter,
    ) -> None:
        """Validate loader/selection config; store everything ``fit()`` needs."""
        if selection_mode not in ("min", "max"):
            raise ConfigError(f"selection_mode must be 'min' or 'max', got {selection_mode!r}.")
        if selection_key not in ("val_loss", "val_dice"):
            raise ConfigError(
                f"selection_key must be 'val_loss' or 'val_dice', got {selection_key!r}."
            )
        if selection_key == "val_dice" and validate_fn is None:
            raise ConfigError("selection_key='val_dice' requires a validate_fn.")
        _assert_not_persistent(train_loader, name="train_loader")
        _assert_not_persistent(val_loader, name="val_loader")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.step_fn = step_fn
        self.device = device
        self.max_epochs = int(max_epochs)
        self.grad_clip_norm = float(grad_clip_norm)
        self.early_stopping_patience = int(early_stopping_patience)
        self.selection_key = selection_key
        self.selection_mode = selection_mode
        self.on_epoch_start = on_epoch_start
        self.validate_fn = validate_fn
        self.checkpoint_writer = checkpoint_writer

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        n_batches = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                if train:
                    self.optimizer.zero_grad()
                loss = self.step_fn(self.model, batch, self.device)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
        if n_batches == 0:
            raise TrainError(f"Trainer: {'train' if train else 'val'}_loader produced 0 batches.")
        return total_loss / n_batches

    def fit(self) -> TrainSummary:
        """Run up to ``max_epochs``, writing the best checkpoint as soon as it improves."""
        self.model.to(self.device)
        init_params = [p.detach().clone() for p in self.model.parameters()]

        is_better = (lambda a, b: a > b) if self.selection_mode == "max" else (lambda a, b: a < b)
        best_score = float("-inf") if self.selection_mode == "max" else float("inf")
        best_epoch = -1
        patience = 0
        history: list[EpochRecord] = []

        for epoch in range(self.max_epochs):
            if self.on_epoch_start is not None:
                self.on_epoch_start(epoch)

            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)

            extra: dict[str, float] = {}
            if self.validate_fn is not None:
                dice = self.validate_fn(self.model, epoch)
                if dice is not None:
                    extra["val_dice"] = dice

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            lr = float(self.optimizer.param_groups[0]["lr"])

            history.append(
                EpochRecord(
                    epoch=epoch, train_loss=train_loss, val_loss=val_loss, extra=extra, lr=lr
                )
            )

            current_score = extra.get("val_dice") if self.selection_key == "val_dice" else val_loss
            if current_score is not None and is_better(current_score, best_score):
                best_score = current_score
                best_epoch = epoch
                patience = 0
                self.checkpoint_writer.save(
                    self.model, epoch=epoch, metrics={"val_loss": val_loss, **extra}
                )
            else:
                patience += 1

            if patience >= self.early_stopping_patience:
                break

        if best_epoch == -1:
            raise TrainError(
                "Trainer.fit(): no epoch ever improved the selection score "
                f"({self.selection_key}); no checkpoint was written."
            )

        final_params = list(self.model.parameters())
        param_l2_delta = math.sqrt(
            sum(
                float(((f - i) ** 2).sum().item())
                for f, i in zip(final_params, init_params, strict=True)
            )
        )

        return TrainSummary(
            best_epoch=best_epoch,
            best_score=best_score,
            selection_key=self.selection_key,
            history=tuple(history),
            checkpoint_path=self.checkpoint_writer.path,
            param_l2_delta=param_l2_delta,
        )


__all__ = ["EpochRecord", "OnEpochStart", "StepFn", "Trainer", "TrainSummary", "ValidateFn"]
