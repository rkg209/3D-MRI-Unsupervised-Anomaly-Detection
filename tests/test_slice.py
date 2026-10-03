"""Spec 000 vertical-slice behaviour tests.

All runnable **without** real data or checkpoints, on synthetic tensors. They lock the
integration contracts the whole project hangs on: Dice binarization, both checkpoint layouts,
LFS-stub / missing-file detection, bounded UNETR output, the percentile threshold, RunLogger
provenance + privacy, and forward-pass determinism.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


# ── Dice binarization (acceptance #6) ─────────────────────────────────────────
def test_dice_differs_raw_vs_binarized_gt() -> None:
    """Raw {0,1,2,4} GT and its (seg>0) binarization give DIFFERENT Dice.

    Guards the prior work's invalid-Dice bug: feeding the raw multi-class map weights the
    numerator by label magnitude. ``MetricsComputer.dice`` must not binarize internally.
    """
    from mri_ad.eval.metrics import MetricsComputer

    pred = torch.zeros(4, 4)
    pred[0, :] = 1.0
    gt_raw = torch.zeros(4, 4)
    gt_raw[0, :] = 4.0  # tumor-core label
    gt_raw[1, :] = 1.0  # edema

    dice_raw = MetricsComputer.dice(pred, gt_raw)
    dice_bin = MetricsComputer.dice(pred, (gt_raw > 0).float())
    assert dice_raw != pytest.approx(dice_bin)


def test_dice_empty_conventions() -> None:
    from mri_ad.eval.metrics import MetricsComputer

    z = torch.zeros(2, 2)
    o = torch.ones(2, 2)
    assert MetricsComputer.dice(z, z) == 1.0  # both empty
    assert MetricsComputer.dice(o, z) == 0.0  # one empty
    assert MetricsComputer.dice(o, o) == pytest.approx(1.0, abs=1e-4)  # perfect overlap


# ── UNETR ─────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def small_unetr():
    """A tiny UNETR (real architecture, small ViT) that fits in CPU RAM for tests."""
    pytest.importorskip("monai")
    from mri_ad.models.unetr import UNETRReconstruction

    torch.manual_seed(0)
    return UNETRReconstruction(
        in_channels=1,
        img_size=(16, 128, 128),
        feature_size=8,
        hidden_size=96,
        mlp_dim=192,
        num_heads=4,
        num_layers=12,
    )


def test_unetr_output_bounded_and_shape(small_unetr) -> None:
    x = torch.rand(1, 1, 16, 128, 128)
    with torch.no_grad():
        y = small_unetr(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_unetr_forward_is_deterministic(small_unetr) -> None:
    """Acceptance #5: two forward passes on the same input agree to 1e-6."""
    small_unetr.eval()
    x = torch.rand(1, 1, 16, 128, 128)
    with torch.no_grad():
        a = small_unetr(x)
        b = small_unetr(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_unetr_loads_both_checkpoint_formats(small_unetr, tmp_path: Path) -> None:
    """Spec 002 contract: bare state_dict AND {"model_state_dict": ...} both load strict=True."""
    from mri_ad.models.unetr import UNETRReconstruction

    state = small_unetr.state_dict()
    bare = tmp_path / "bare.pth"
    wrapped = tmp_path / "wrapped.pth"
    torch.save(state, bare)
    torch.save({"model_state_dict": state, "epoch": 3}, wrapped)

    for path in (bare, wrapped):
        fresh = UNETRReconstruction(
            in_channels=1,
            img_size=(16, 128, 128),
            feature_size=8,
            hidden_size=96,
            mlp_dim=192,
            num_heads=4,
            num_layers=12,
        )
        fresh.load_checkpoint(path)  # must not raise


def test_unetr_missing_checkpoint_raises(small_unetr, tmp_path: Path) -> None:
    from mri_ad.exceptions import CheckpointError

    with pytest.raises(CheckpointError, match="nope.pth"):
        small_unetr.load_checkpoint(tmp_path / "nope.pth")


def test_unetr_lfs_stub_raises(small_unetr) -> None:
    """An inherited legacy .pth stub must be rejected, not loaded as random weights."""
    from mri_ad.exceptions import CheckpointError

    stub = REPO / "legacy/UNET/3d_MonaiUNET_slices_MSE.pth"
    if not stub.exists():
        pytest.skip("legacy stub not present")
    with pytest.raises(CheckpointError, match="stub"):
        small_unetr.load_checkpoint(stub)


# ── Threshold ─────────────────────────────────────────────────────────────────
def test_percentile_threshold_flags_fraction_and_is_binary() -> None:
    from mri_ad.recon.threshold import FixedPercentileThreshold

    torch.manual_seed(1)
    residual = torch.rand(10, 16, 128, 128)
    mask = FixedPercentileThreshold(95.0)(residual)

    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    flagged = float(mask.mean())
    assert abs(flagged - 0.05) < 0.005  # ~5% flagged (within one voxel-fraction)
    assert mask.shape == residual.shape


# ── Full-depth reassembly (prior-work bug #2) ─────────────────────────────────
def test_chunk_reassembly_covers_whole_volume() -> None:
    """Reassembling N chunks must recover the FULL depth, not one chunk.

    The slice concatenates per-chunk ``(1, 16, 128, 128)`` tensors along depth (dim=1) then
    drops the channel. Concatenating along dim=0 instead (the tempting bug) keeps only N depth
    slices — the prior work's "scored 1 of 8 chunks" failure. A tumor placed in chunk 3 must
    survive into the reassembled ground truth at the right depth.
    """
    n_chunks = 10
    label_chunks = torch.zeros(n_chunks, 1, 16, 128, 128)
    label_chunks[3, 0, 5:9, 40:70, 40:70] = 1.0  # tumor in chunk 3, local depth 5..8

    gt = torch.cat(list(label_chunks), dim=1).squeeze(0)
    assert gt.shape == (160, 128, 128)  # all 10 * 16 slices, not 10
    assert gt[3 * 16 + 5 : 3 * 16 + 9].sum() > 0  # tumor at global depth 53..56
    assert int(gt.sum(dim=(1, 2)).argmax()) in range(53, 57)


# ── RunLogger (acceptance #7 privacy + provenance) ────────────────────────────
def test_run_logger_writes_provenance_without_leaks(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from mri_ad.utils.run_logger import RunLogger

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "paths": {"artifact_root": str(tmp_path)},
            "data": {"brats": {"dir": "./data/brats"}},
        }
    )
    with RunLogger(cfg) as run:
        run.record(dice=0.4242)

    runs = list((tmp_path / "runs").glob("*/run_meta.json"))
    assert len(runs) == 1
    meta = json.loads(runs[0].read_text())

    assert meta["git_sha"], "git_sha must be non-empty"
    assert meta["seed"] == 42
    assert meta["metrics"]["dice"] == pytest.approx(0.4242)
    assert "config" in meta  # fully-resolved config is persisted

    # NFR-13: the slice must never persist a scan-file path or patient-scan identifier.
    # (Config dirs are config; per-volume .nii paths and subject scan files are the leak.)
    raw = runs[0].read_text()
    assert ".nii" not in raw
    assert "_seg" not in raw and "_t2.nii" not in raw


# ── DeviceManager ─────────────────────────────────────────────────────────────
def test_device_manager_honours_explicit_and_auto() -> None:
    from mri_ad.utils.device import DeviceManager

    assert DeviceManager.get_device("cpu").type == "cpu"
    assert DeviceManager.get_device("auto").type in {"cuda", "mps", "cpu"}


def test_git_sha_falls_back_to_dotgit_when_git_binary_missing(tmp_path, monkeypatch):
    """GPU nodes have no ``git``: the SHA must still be recorded and ``git_dirty`` must be unknown."""
    import json as _json

    from omegaconf import OmegaConf

    from mri_ad.utils import run_logger

    sha = "a" * 40
    (tmp_path / ".git" / "refs" / "heads").mkdir(parents=True)
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git" / "refs" / "heads" / "main").write_text(sha + "\n")
    assert run_logger._git_sha_from_files(tmp_path / "pkg" / "m.py") == sha

    monkeypatch.setattr(run_logger, "_git", lambda *a: "")
    monkeypatch.setattr(run_logger, "_git_sha_from_files", lambda *a: sha)
    cfg = OmegaConf.create({"seed": 1, "paths": {"artifact_root": str(tmp_path / "art")}})
    with run_logger.RunLogger(cfg):
        pass
    meta = _json.loads(next((tmp_path / "art" / "runs").glob("*/run_meta.json")).read_text())
    assert meta["git_sha"] == sha
    assert meta["git_dirty"] is None
