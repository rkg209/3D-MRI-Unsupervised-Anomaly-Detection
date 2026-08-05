"""Spec 001 data-layer behaviour tests.

All runnable **without** real data on synthetic `.npy`/`.nii` fixtures written to `tmp_path`.
Locks the acceptance criteria in `specs/001-data-layer.md`: the shared shape/dtype/range
contract, loud validation, the split contract, the one shared pipeline, label binarization, and
determinism.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("monai")

from omegaconf import OmegaConf  # noqa: E402

from mri_ad.data.datasets import (  # noqa: E402
    BraTSDataset,
    OpenBHBDataset,
    load_preprocessed_volume,
)
from mri_ad.data.loaders import build_dataloader  # noqa: E402
from mri_ad.data.split import SplitContract, resolve_contract  # noqa: E402
from mri_ad.data.transforms import build_transforms  # noqa: E402
from mri_ad.data.validation import DataValidator  # noqa: E402
from mri_ad.exceptions import DescriptiveValidationError, SplitContractViolationError  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _make_openbhb_dir(tmp_path: Path, n_subjects: int = 2) -> tuple[Path, list[str]]:
    d = tmp_path / "openbhb"
    d.mkdir()
    ids = []
    rng = np.random.default_rng(0)
    for i in range(n_subjects):
        vid = f"sub-{i}.npy"
        np.save(d / vid, rng.random((1, 1, 182, 218, 182)).astype(np.float32))
        ids.append(vid)
    return d, ids


def _make_brats_dir(
    tmp_path: Path, n_subjects: int = 2, *, depth: int = 155
) -> tuple[Path, list[str]]:
    nib = pytest.importorskip("nibabel")
    d = tmp_path / "brats"
    d.mkdir()
    ids = []
    rng = np.random.default_rng(0)
    for i in range(n_subjects):
        vid = f"BraTS_{i}"
        subject_dir = d / vid
        subject_dir.mkdir()
        img = rng.random((240, 240, depth)).astype(np.float32)
        seg = rng.integers(0, 5, size=(240, 240, depth)).astype(np.float32)
        seg[seg == 3] = 0  # BraTS labels are {0,1,2,4}
        nib.save(nib.Nifti1Image(img, np.eye(4)), str(subject_dir / f"{vid}_t2.nii.gz"))
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(subject_dir / f"{vid}_seg.nii.gz"))
        ids.append(vid)
    return d, ids


def _base_cfg(tmp_path: Path, openbhb_dir: Path, brats_dir: Path):
    return OmegaConf.create(
        {
            "seed": 42,
            "shape": {"height": 128, "width": 128},
            "paths": {"artifact_root": str(tmp_path / "artifacts")},
            "data": {
                "preprocess": {"eps": 1.0e-8, "resize_mode": "trilinear", "chunk_depth": 16},
                "openbhb": {
                    "dir": str(openbhb_dir),
                    "crop": {"z": [50, 130], "y": [20, 160], "x": [20, 196]},
                },
                "brats": {"dir": str(brats_dir), "modality": "t2"},
                "loader": {"cache_rate": 0.0, "batch_size": 4, "num_workers": 0},
            },
        }
    )


# ── Acceptance #1: both dataloaders yield (B,1,16,128,128) float32 in [0,1] ───────────────────
def test_both_dataloaders_yield_the_contracted_batch_shape(tmp_path: Path) -> None:
    openbhb_dir, openbhb_ids = _make_openbhb_dir(tmp_path)
    brats_dir, brats_ids = _make_brats_dir(tmp_path)
    cfg = _base_cfg(tmp_path, openbhb_dir, brats_dir)

    ob_loader = build_dataloader(cfg, OpenBHBDataset(cfg, openbhb_ids), split="val")
    bt_loader = build_dataloader(cfg, BraTSDataset(cfg, brats_ids), split="val")

    for loader in (ob_loader, bt_loader):
        batch = next(iter(loader))
        image = batch["image"]
        assert image.shape[1:] == (1, 16, 128, 128)
        assert image.dtype == torch.float32
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0


# ── Acceptance #2: DataValidator raises with observed shape + range ───────────────────────────
def test_data_validator_raises_descriptive_error_on_malformed_batch() -> None:
    bad = torch.rand(2, 1, 8, 128, 128) * 5.0
    with pytest.raises(DescriptiveValidationError) as excinfo:
        DataValidator.validate_batch(bad)
    message = str(excinfo.value)
    assert "8" in message  # observed depth
    assert "(2, 1, 8, 128, 128)" in message


def test_data_validator_raises_on_out_of_range_values() -> None:
    bad = torch.rand(2, 1, 16, 128, 128) * 5.0 - 1.0
    with pytest.raises(DescriptiveValidationError) as excinfo:
        DataValidator.validate_batch(bad)
    message = str(excinfo.value)
    assert "range" in message.lower()


def test_data_validator_accepts_a_well_formed_batch() -> None:
    good = torch.rand(3, 1, 16, 128, 128)
    DataValidator.validate_batch(good)  # must not raise


# ── Acceptance #3: SplitContract reproducible / seed-sensitive / round-trips ──────────────────
def test_split_contract_reproducible_for_same_seed_different_for_new_seed() -> None:
    openbhb_ids = [f"sub-{i}" for i in range(20)]
    brats_ids = [f"BraTS_{i}" for i in range(20)]

    a = SplitContract.build(42, openbhb_ids, brats_ids)
    b = SplitContract.build(42, openbhb_ids, brats_ids)
    c = SplitContract.build(7, openbhb_ids, brats_ids)

    assert a == b
    assert a != c


def test_split_contract_round_trips_through_disk(tmp_path: Path) -> None:
    openbhb_ids = [f"sub-{i}" for i in range(10)]
    brats_ids = [f"BraTS_{i}" for i in range(10)]
    contract = SplitContract.build(42, openbhb_ids, brats_ids)

    path = tmp_path / "split_contract.json"
    contract.save(path)
    loaded = SplitContract.load(path)

    assert loaded == contract
    assert json.loads(path.read_text())["seed"] == 42


# ── Acceptance #4: disagreeing split raises unless overridden ─────────────────────────────────
def test_split_contract_verify_raises_on_mismatch_and_override_suppresses() -> None:
    openbhb_ids = [f"sub-{i}" for i in range(10)]
    brats_ids = [f"BraTS_{i}" for i in range(10)]
    recorded = SplitContract.build(42, openbhb_ids, brats_ids)
    other = SplitContract.build(7, openbhb_ids, brats_ids)

    with pytest.raises(SplitContractViolationError):
        recorded.verify(other)

    recorded.verify(other, override=True)  # must not raise
    recorded.verify(recorded)  # identical contract never raises


# ── Acceptance #5: train and eval pipelines are the SAME function ─────────────────────────────
def test_openbhb_and_brats_share_identical_output_contract(tmp_path: Path) -> None:
    openbhb_dir, _ = _make_openbhb_dir(tmp_path, n_subjects=1)
    brats_dir, _ = _make_brats_dir(tmp_path, n_subjects=1)
    cfg = _base_cfg(tmp_path, openbhb_dir, brats_dir)

    ob_transform = build_transforms(cfg, dataset="openbhb")
    bt_transform = build_transforms(cfg, dataset="brats")

    rng = np.random.default_rng(1)
    ob_out = ob_transform({"image": rng.random((182, 218, 182)).astype(np.float32)})
    bt_out = bt_transform(
        {
            "image": rng.random((240, 240, 155)).astype(np.float32),
            "label": rng.integers(0, 5, size=(240, 240, 155)).astype(np.float32),
        }
    )

    assert ob_out["image"].dtype == bt_out["image"].dtype == torch.float32
    assert ob_out["image"].shape[0] == bt_out["image"].shape[0] == 1  # channel dim
    assert ob_out["image"].shape[-2:] == bt_out["image"].shape[-2:] == (128, 128)
    for out in (ob_out["image"], bt_out["image"]):
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0


# ── Acceptance #6: BraTS label is binary after preprocessing ──────────────────────────────────
def test_brats_label_is_binary_after_preprocessing(tmp_path: Path) -> None:
    openbhb_dir, _ = _make_openbhb_dir(tmp_path, n_subjects=1)
    brats_dir, _ = _make_brats_dir(tmp_path, n_subjects=1)
    cfg = _base_cfg(tmp_path, openbhb_dir, brats_dir)

    transform = build_transforms(cfg, dataset="brats")
    label = np.zeros((240, 240, 155), dtype=np.float32)
    label[10:20, 10:20, 10:20] = 1.0
    label[30:40, 30:40, 30:40] = 2.0
    label[50:60, 50:60, 50:60] = 4.0

    out = transform({"image": np.random.rand(240, 240, 155).astype(np.float32), "label": label})
    unique = {float(v) for v in torch.unique(out["label"]).tolist()}
    assert unique <= {0.0, 1.0}
    assert unique == {0.0, 1.0}  # the planted foreground must survive as 1.0, not vanish


# ── Acceptance #7: chunking / transforms are deterministic, no Rand* on the eval path ─────────
def test_no_random_transform_on_the_eval_path(tmp_path: Path) -> None:
    openbhb_dir, _ = _make_openbhb_dir(tmp_path, n_subjects=1)
    brats_dir, _ = _make_brats_dir(tmp_path, n_subjects=1)
    cfg = _base_cfg(tmp_path, openbhb_dir, brats_dir)

    for dataset in ("openbhb", "brats"):
        compose = build_transforms(cfg, dataset=dataset)
        for t in compose.transforms:
            assert not type(t).__name__.startswith("Rand"), f"found random transform: {t}"


def test_chunking_is_deterministic_across_runs(tmp_path: Path) -> None:
    openbhb_dir, openbhb_ids = _make_openbhb_dir(tmp_path, n_subjects=1)
    brats_dir, _ = _make_brats_dir(tmp_path, n_subjects=1)
    cfg = _base_cfg(tmp_path, openbhb_dir, brats_dir)

    ds_a = OpenBHBDataset(cfg, openbhb_ids)
    ds_b = OpenBHBDataset(cfg, openbhb_ids)
    for i in range(len(ds_a)):
        assert torch.equal(ds_a[i]["image"], ds_b[i]["image"])


# ── Acceptance #8: privacy — no path / patient-id leaks into returned items ───────────────────
def test_dataset_items_never_leak_a_path_or_scan_suffix(tmp_path: Path) -> None:
    openbhb_dir, openbhb_ids = _make_openbhb_dir(tmp_path, n_subjects=1)
    brats_dir, brats_ids = _make_brats_dir(tmp_path, n_subjects=1)
    cfg = _base_cfg(tmp_path, openbhb_dir, brats_dir)

    ob_item = OpenBHBDataset(cfg, openbhb_ids)[0]
    bt_item = BraTSDataset(cfg, brats_ids)[0]

    for item in (ob_item, bt_item):
        for value in item.values():
            if isinstance(value, torch.Tensor):
                continue
            text = str(value)
            assert ".nii" not in text
            assert ".npy" not in text
            assert str(tmp_path) not in text
        assert isinstance(item["volume_index"], int)
        assert isinstance(item["chunk_index"], int)


# ── Spec 006 R4: content_hash + resolve_contract (build-if-missing / verify-on-mismatch) ──────
def _split_cfg(tmp_path: Path, openbhb_dir: Path, brats_dir: Path, *, seed: int = 42):
    return OmegaConf.create(
        {
            "seed": seed,
            "data": {
                "openbhb": {"dir": str(openbhb_dir)},
                "brats": {"dir": str(brats_dir)},
                "split": {
                    "contract_path": str(tmp_path / "split_contract.json"),
                    "openbhb": {"train": 0.85, "val": 0.15},
                    "brats": {"val": 0.2, "test": 0.8},
                },
            },
        }
    )


def test_content_hash_is_stable_and_seed_sensitive() -> None:
    a = SplitContract.build(42, ["a", "b"], ["x", "y"])
    b = SplitContract.build(42, ["a", "b"], ["x", "y"])
    c = SplitContract.build(7, ["a", "b"], ["x", "y"])
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()


def test_resolve_contract_builds_and_saves_when_missing(tmp_path: Path) -> None:
    openbhb_dir, _ = _make_openbhb_dir(tmp_path, n_subjects=4)
    brats_dir, _ = _make_brats_dir(tmp_path, n_subjects=4)
    cfg = _split_cfg(tmp_path, openbhb_dir, brats_dir)

    contract = resolve_contract(cfg, build_if_missing=True)

    contract_path = Path(cfg.data.split.contract_path)
    assert contract_path.is_file()
    assert SplitContract.load(contract_path) == contract


def test_resolve_contract_raises_when_missing_and_not_build_if_missing(tmp_path: Path) -> None:
    openbhb_dir, _ = _make_openbhb_dir(tmp_path, n_subjects=2)
    brats_dir, _ = _make_brats_dir(tmp_path, n_subjects=2)
    cfg = _split_cfg(tmp_path, openbhb_dir, brats_dir)

    with pytest.raises(SplitContractViolationError):
        resolve_contract(cfg, build_if_missing=False)


def test_resolve_contract_verifies_and_detects_directory_drift(tmp_path: Path) -> None:
    openbhb_dir, _ = _make_openbhb_dir(tmp_path, n_subjects=4)
    brats_dir, _ = _make_brats_dir(tmp_path, n_subjects=4)
    cfg = _split_cfg(tmp_path, openbhb_dir, brats_dir)

    first = resolve_contract(cfg, build_if_missing=True)
    second = resolve_contract(cfg, build_if_missing=False)
    assert first == second  # stable across repeated calls, no re-partition

    # A subject appears (directory drift) -> the recorded contract now disagrees, loudly.
    (brats_dir / "BraTS_extra").mkdir()
    with pytest.raises(SplitContractViolationError):
        resolve_contract(cfg, build_if_missing=False)


def test_load_preprocessed_volume_returns_whole_volume_never_chunked(tmp_path: Path) -> None:
    openbhb_dir, _ = _make_openbhb_dir(tmp_path, n_subjects=1)
    brats_dir, brats_ids = _make_brats_dir(tmp_path, n_subjects=1, depth=40)
    cfg = _base_cfg(tmp_path, openbhb_dir, brats_dir)

    volume = load_preprocessed_volume(cfg, brats_ids[0])

    assert volume["image"].shape == (1, 40, 128, 128)  # whole depth, no padding/chunking to 16
    assert volume["label"].shape == (1, 40, 128, 128)
    assert set(torch.unique(volume["label"]).tolist()) <= {0.0, 1.0}
    for value in volume.values():
        assert ".nii" not in str(value)
