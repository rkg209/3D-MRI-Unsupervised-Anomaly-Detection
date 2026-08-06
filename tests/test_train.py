"""Spec 009 acceptance tests for ``train/`` (``loop.py``, ``objectives.py``, ``checkpointing.py``,
``splits.py``).

All tests run on a 3-parameter toy ``nn.Module`` and tiny tensors — no real data, no checkpoints,
no real 3D forward passes (laptop constraint, see CLAUDE.md / the ``no heavy local runs`` memory).
"""

from __future__ import annotations

import ast
import functools
import inspect
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from mri_ad.data.split import SplitContract
from mri_ad.exceptions import ConfigError, SplitContractViolationError, TrainError
from mri_ad.models._checkpoint import load_checked_state_dict
from mri_ad.synth.fpi import FPIAnomalyGenerator
from mri_ad.train.checkpointing import CheckpointWriter
from mri_ad.train.loop import Trainer, TrainSummary
from mri_ad.train.objectives import reconstruction_step
from mri_ad.train.splits import TrainingIds, resolve_training_ids

REPO = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO / "src" / "mri_ad" / "train"


class _ToyModel(nn.Module):
    """A 3-parameter model: cheap enough to `.fit()` in a unit test many times over."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        return self.a * x + self.b + self.c


class _ToyDataset(Dataset):
    """``{"corrupted", "healthy"}`` pairs at a fixed linear relationship the model can learn."""

    def __init__(self, n: int = 8, *, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.corrupted = torch.rand((n, 1, 2, 4, 4), generator=g)
        self.healthy = 2.0 * self.corrupted + 0.1
        self.set_epoch_calls: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.set_epoch_calls.append(epoch)

    def __len__(self) -> int:
        return self.corrupted.shape[0]

    def __getitem__(self, index: int) -> dict:
        return {"corrupted": self.corrupted[index], "healthy": self.healthy[index]}


def _toy_loader(dataset: Dataset, **kwargs: object) -> DataLoader:
    return DataLoader(dataset, batch_size=4, shuffle=False, **kwargs)  # type: ignore[arg-type]


def _toy_trainer(tmp_path: Path, **overrides: object) -> Trainer:
    model = _ToyModel()
    train_loader = _toy_loader(_ToyDataset(seed=1))
    val_loader = _toy_loader(_ToyDataset(seed=2))
    writer = CheckpointWriter(
        tmp_path / "ckpt.pth",
        meta={
            "selection_key": "val_loss",
            "git_sha": "deadbeef",
            "seed": 1,
            "split_hash": "abc123",
            "synth": {},
            "train": {},
        },
    )
    params: dict[str, object] = dict(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.1),
        scheduler=None,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=functools.partial(reconstruction_step, criterion=nn.MSELoss()),
        device=torch.device("cpu"),
        max_epochs=5,
        grad_clip_norm=1.0,
        early_stopping_patience=3,
        selection_key="val_loss",
        selection_mode="min",
        checkpoint_writer=writer,
    )
    params.update(overrides)
    return Trainer(**params)  # type: ignore[arg-type]


# ── Trainer.fit() basics ─────────────────────────────────────────────────────────────────────


def test_fit_returns_a_train_summary_and_writes_a_checkpoint(tmp_path: Path) -> None:
    trainer = _toy_trainer(tmp_path)
    summary = trainer.fit()
    assert isinstance(summary, TrainSummary)
    assert summary.checkpoint_path.is_file()
    assert summary.best_epoch >= 0
    assert summary.selection_key == "val_loss"


def test_parameters_actually_move(tmp_path: Path) -> None:
    trainer = _toy_trainer(tmp_path)
    summary = trainer.fit()
    assert summary.param_l2_delta > 0.0


def test_on_epoch_start_wires_dataset_set_epoch(tmp_path: Path) -> None:
    train_ds = _ToyDataset(seed=1)
    trainer = _toy_trainer(
        tmp_path,
        train_loader=_toy_loader(train_ds),
        on_epoch_start=train_ds.set_epoch,
        max_epochs=3,
        early_stopping_patience=99,
    )
    trainer.fit()
    assert train_ds.set_epoch_calls == [0, 1, 2]


def test_validate_fn_feeds_val_dice_selection(tmp_path: Path) -> None:
    scores = iter([0.1, 0.4, 0.3, 0.5, 0.2])

    def validate_fn(model: nn.Module, epoch: int) -> float:
        del model, epoch
        return next(scores)

    trainer = _toy_trainer(
        tmp_path,
        selection_key="val_dice",
        selection_mode="max",
        validate_fn=validate_fn,
        early_stopping_patience=99,
        max_epochs=5,
    )
    summary = trainer.fit()
    assert summary.selection_key == "val_dice"
    assert summary.best_score == 0.5
    assert summary.best_epoch == 3


def test_validate_fn_returning_none_skips_that_epochs_score(tmp_path: Path) -> None:
    calls = []

    def validate_fn(model: nn.Module, epoch: int) -> float | None:
        del model
        calls.append(epoch)
        return None if epoch % 2 else 0.9 - 0.1 * epoch

    trainer = _toy_trainer(
        tmp_path,
        selection_key="val_dice",
        selection_mode="max",
        validate_fn=validate_fn,
        early_stopping_patience=99,
        max_epochs=4,
    )
    summary = trainer.fit()
    assert calls == [0, 1, 2, 3]
    assert summary.best_epoch == 0  # only epochs 0 and 2 ever produced a score; 0 was higher


def test_early_stopping_halts_before_max_epochs(tmp_path: Path) -> None:
    model = _ToyModel()
    trainer = _toy_trainer(
        tmp_path,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),  # lr=0 -> val_loss never improves
        max_epochs=20,
        early_stopping_patience=2,
    )
    summary = trainer.fit()
    assert len(summary.history) < 20
    assert len(summary.history) == 3  # epoch 0 sets best; epochs 1, 2 fail to beat it -> stop


def test_zero_max_epochs_raises_train_error(tmp_path: Path) -> None:
    trainer = _toy_trainer(tmp_path, max_epochs=0)
    with pytest.raises(TrainError):
        trainer.fit()


def test_empty_loader_raises_train_error(tmp_path: Path) -> None:
    empty_loader = _toy_loader(_ToyDataset(n=0))
    trainer = _toy_trainer(tmp_path, train_loader=empty_loader)
    with pytest.raises(TrainError):
        trainer.fit()


# ── Config validation ────────────────────────────────────────────────────────────────────────


def test_invalid_selection_mode_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _toy_trainer(tmp_path, selection_mode="sideways")


def test_val_dice_selection_without_validate_fn_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _toy_trainer(tmp_path, selection_key="val_dice", selection_mode="max")


def test_persistent_workers_true_raises_config_error(tmp_path: Path) -> None:
    loader = DataLoader(_ToyDataset(), batch_size=4, num_workers=1, persistent_workers=True)
    with pytest.raises(ConfigError):
        _toy_trainer(tmp_path, train_loader=loader)


# ── objectives.py ─────────────────────────────────────────────────────────────────────────────


def test_reconstruction_step_matches_criterion_of_model_output() -> None:
    model = _ToyModel()
    batch = {"corrupted": torch.ones(1, 1, 2, 2), "healthy": torch.ones(1, 1, 2, 2)}
    step = functools.partial(reconstruction_step, criterion=nn.MSELoss())
    loss = step(model, batch, torch.device("cpu"))
    expected = nn.MSELoss()(model(batch["corrupted"]), batch["healthy"])
    assert torch.equal(loss, expected)


# ── checkpointing.py ──────────────────────────────────────────────────────────────────────────


def test_checkpoint_round_trips_through_load_checked_state_dict(tmp_path: Path) -> None:
    model = _ToyModel()
    model.a.data.fill_(3.0)
    writer = CheckpointWriter(
        tmp_path / "ckpt.pth",
        meta={"selection_key": "val_loss", "git_sha": "abc", "seed": 0, "split_hash": "h"},
    )
    path = writer.save(model, epoch=2, metrics={"val_loss": 0.1})

    fresh = _ToyModel()
    load_checked_state_dict(fresh, path)
    assert torch.equal(fresh.a, model.a)


def test_checkpoint_payload_has_no_paths_key(tmp_path: Path) -> None:
    model = _ToyModel()
    writer = CheckpointWriter(
        tmp_path / "ckpt.pth",
        meta={"selection_key": "val_loss", "git_sha": "abc", "seed": 0, "split_hash": "h"},
    )
    path = writer.save(model, epoch=0, metrics={"val_loss": 0.1})
    payload = torch.load(path, map_location="cpu")
    assert "paths" not in payload
    assert set(payload) == {
        "model_state_dict",
        "epoch",
        "selection_key",
        "selection_value",
        "val_loss",
        "git_sha",
        "seed",
        "split_hash",
        "synth",
        "train",
    }


# ── splits.py ─────────────────────────────────────────────────────────────────────────────────


def _contract() -> SplitContract:
    return SplitContract(
        seed=0,
        openbhb={"train": ["a", "b", "c"], "val": ["d"]},
        brats={"val": ["v1"], "test": ["t1", "t2"]},
    )


def test_training_ids_are_openbhb_train_only() -> None:
    ids = resolve_training_ids(OmegaConf.create({}), _contract())
    assert isinstance(ids, TrainingIds)
    assert ids.openbhb_train == ("a", "b", "c")
    assert ids.openbhb_val == ("d",)
    assert ids.brats_select == ("v1",)


def test_resolve_training_ids_rejects_test_overlap() -> None:
    contract = SplitContract(
        seed=0,
        openbhb={"train": ["a"], "val": ["b"]},
        brats={"val": ["t1"], "test": ["t1", "t2"]},  # deliberately overlapping fixture
    )
    with pytest.raises(SplitContractViolationError):
        resolve_training_ids(OmegaConf.create({}), contract)


# ── Boundary: train/ never imports recon/ or eval/ ──────────────────────────────────────────────


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_train_never_imports_recon_or_eval() -> None:
    forbidden = {"mri_ad.recon", "mri_ad.eval"}
    for path in sorted(TRAIN_DIR.rglob("*.py")):
        modules = _imported_modules(path)
        hit = {m for m in modules if any(m == f or m.startswith(f + ".") for f in forbidden)}
        assert not hit, f"{path.name} imports forbidden module(s): {hit}"


# ── Acceptance 3: every FPI generator param is derivable, not hardcoded ─────────────────────────


def test_fpi_config_declares_every_generator_param() -> None:
    signature = inspect.signature(FPIAnomalyGenerator.__init__)
    expected = {name for name in signature.parameters if name != "self"}

    cfg = OmegaConf.load(REPO / "configs" / "synth" / "fpi.yaml")
    declared = set(OmegaConf.to_container(cfg.params, resolve=False))

    assert declared == expected


# ── Acceptance 6: reachable only via /train ──────────────────────────────────────────────────


def test_train_skill_disables_model_invocation() -> None:
    text = (REPO / ".claude" / "skills" / "train" / "SKILL.md").read_text()
    frontmatter = text.split("---")[1]
    assert "disable-model-invocation: true" in frontmatter


def test_makefile_train_target_exists() -> None:
    makefile = (REPO / "Makefile").read_text()
    assert "\ntrain:" in makefile
    assert "run_train.py" in makefile


def test_run_train_refuses_non_cuda_device() -> None:
    import sys

    sys.path.insert(0, str(REPO))
    import scripts.run_train as run_train

    from mri_ad.exceptions import ConfigError as _ConfigError

    cfg = OmegaConf.create({"device": "cpu", "train": {"allow_cpu": False}})
    with pytest.raises(_ConfigError):
        run_train._require_cuda(cfg)  # noqa: SLF001

    cfg_allowed = OmegaConf.create({"device": "cpu", "train": {"allow_cpu": True}})
    device = run_train._require_cuda(cfg_allowed)  # noqa: SLF001
    assert device.type == "cpu"
