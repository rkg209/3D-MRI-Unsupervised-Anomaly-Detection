"""Spec 003 reconstruction & anomaly-map engine tests.

All runnable **without** real data or checkpoints. A ``_StubModel`` (near-identity + seeded
noise) exercises every engine code path at full ``(1,160,128,128)`` shape while staying fast and
deterministic — mirrors ``tests/test_slice.py``'s house convention of no real forward passes in
the default test run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("monai")

from omegaconf import OmegaConf  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from mri_ad.exceptions import ReconError  # noqa: E402
from mri_ad.models.base import AnomalyDetectionModel, ModelCard  # noqa: E402
from mri_ad.recon import (  # noqa: E402
    AbsoluteThreshold,
    FixedPercentileThreshold,
    OtsuThreshold,
    ReconResult,
    ReconstructionEngine,
    load_result,
    run_threshold_sweep,
    save_result,
    select_operating_point,
)
from mri_ad.utils.instantiate import instantiate_from_config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


class _StubModel(AnomalyDetectionModel):
    """Near-identity reconstruction with seeded noise — no real weights, fully deterministic."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._noise_seed = seed
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        generator = torch.Generator().manual_seed(self._noise_seed)
        noise = torch.randn(x.shape, generator=generator) * 0.01
        return x * 0.9 + noise

    def load_checkpoint(self, path: Path) -> None:
        """No-op — the stub carries no real weights."""

    @property
    def model_card(self) -> ModelCard:
        return ModelCard(
            name="stub",
            param_count=0,
            input_shape=(1, 1, 16, 128, 128),
            output_shape=(1, 1, 16, 128, 128),
            training_data="none",
            training_loss="none",
            output_activation=None,
            known_characteristics="test stub: x * 0.9 + seeded noise",
        )


def _cfg(tmp_path: Path, target: str = "mri_ad.recon.threshold.AbsoluteThreshold", params=None):
    return OmegaConf.create(
        {
            "device": "cpu",
            "recon": {
                "batch_size": 4,
                "save_results": True,
                "results_dir": str(tmp_path / "results"),
            },
            "threshold": {"target": target, "params": params or {"value": 0.05}},
        }
    )


def _make_brats_dir(tmp_path: Path, n_subjects: int = 1, depth: int = 32) -> tuple[Path, list[str]]:
    nib = pytest.importorskip("nibabel")
    d = tmp_path / "brats"
    d.mkdir()
    ids = []
    rng = np.random.default_rng(0)
    for i in range(n_subjects):
        vid = f"BraTS_{i}"
        subject_dir = d / vid
        subject_dir.mkdir()
        img = rng.random((64, 64, depth)).astype(np.float32)
        seg = rng.integers(0, 5, size=(64, 64, depth)).astype(np.float32)
        seg[seg == 3] = 0  # BraTS labels are {0,1,2,4}
        nib.save(nib.Nifti1Image(img, np.eye(4)), str(subject_dir / f"{vid}_t2.nii.gz"))
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(subject_dir / f"{vid}_seg.nii.gz"))
        ids.append(vid)
    return d, ids


# ── Acceptance #1/#2: shape, dtype, device, non-negativity, binary mask ───────────────────────
def test_engine_run_emits_four_tensors_correctly_shaped(tmp_path: Path) -> None:
    model = _StubModel()
    model.eval()
    engine = ReconstructionEngine(model, _cfg(tmp_path))
    torch.manual_seed(0)
    volume = torch.rand(1, 160, 128, 128, dtype=torch.float32)

    result = engine.run(volume, volume_id="vol-0")

    for name in ("original", "reconstruction", "residual", "anomaly_mask"):
        t = getattr(result, name)
        assert t.shape == (1, 160, 128, 128)
        assert t.dtype == torch.float32
        assert t.device.type == "cpu"

    assert bool((result.residual >= 0).all())
    assert set(result.anomaly_mask.unique().tolist()) <= {0.0, 1.0}


def test_engine_run_dataset_volume_reassembles_and_threads_ground_truth(tmp_path: Path) -> None:
    from mri_ad.data.datasets import BraTSDataset

    brats_dir, ids = _make_brats_dir(tmp_path, depth=32)
    cfg = _cfg(tmp_path)
    cfg = OmegaConf.merge(
        cfg,
        {
            "shape": {"height": 128, "width": 128},
            "data": {
                "preprocess": {"eps": 1.0e-8, "resize_mode": "trilinear", "chunk_depth": 16},
                "brats": {"dir": str(brats_dir), "modality": "t2"},
                "loader": {"cache_rate": 0.0, "batch_size": 4, "num_workers": 0},
            },
        },
    )
    dataset = BraTSDataset(cfg, ids)

    model = _StubModel()
    model.eval()
    engine = ReconstructionEngine(model, cfg)
    result = engine.run_dataset_volume(dataset, volume_index=0)

    assert result.volume_id == ids[0]
    assert result.original.shape == (1, 32, 128, 128)  # depth 32 already a multiple of 16
    assert result.ground_truth is not None
    assert result.ground_truth.shape == result.original.shape
    assert set(result.ground_truth.unique().tolist()) <= {0.0, 1.0}  # binarized (seg > 0)


# ── ReconError on malformed input / non-eval model ─────────────────────────────────────────────
def test_engine_raises_on_wrong_input_shape(tmp_path: Path) -> None:
    model = _StubModel()
    model.eval()
    engine = ReconstructionEngine(model, _cfg(tmp_path))
    with pytest.raises(ReconError):
        engine.run(torch.rand(1, 17, 128, 128), volume_id="bad-depth")
    with pytest.raises(ReconError):
        engine.run(torch.rand(1, 16, 64, 64), volume_id="bad-side")


def test_engine_raises_when_model_in_train_mode(tmp_path: Path) -> None:
    model = _StubModel()  # nn.Module defaults to training=True
    engine = ReconstructionEngine(model, _cfg(tmp_path))
    with pytest.raises(ReconError):
        engine.run(torch.rand(1, 16, 128, 128), volume_id="v")


# ── Acceptance #3: percentile flags exactly (100-p)% +/- 1, even with duplicate values ─────────
@pytest.mark.parametrize("p", [90.0, 95.0, 99.0])
def test_percentile_threshold_flags_exact_count_with_duplicates(p: float) -> None:
    torch.manual_seed(1)
    residual = torch.zeros(2, 16, 128, 128)  # mostly duplicate (zero) values
    n = residual.numel()
    n_spikes = 5000
    idx = torch.randperm(n)[:n_spikes]
    residual.view(-1)[idx] = torch.rand(n_spikes) + 1.0  # break ties with real spikes

    mask = FixedPercentileThreshold(p)(residual)
    expected_k = int(round(n * (100.0 - p) / 100.0))
    assert int(mask.sum().item()) == expected_k
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


# ── Acceptance #4: strategy swap is config-only ────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["percentile", "absolute", "otsu"])
def test_threshold_strategy_swap_is_config_only(name: str) -> None:
    node = OmegaConf.load(REPO / "configs" / "threshold" / f"{name}.yaml")
    strategy = instantiate_from_config(node)

    expected_type = {
        "percentile": FixedPercentileThreshold,
        "absolute": AbsoluteThreshold,
        "otsu": OtsuThreshold,
    }[name]
    assert isinstance(strategy, expected_type)

    torch.manual_seed(2)
    residual = torch.rand(4, 16, 128, 128)
    mask = strategy(residual)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_threshold_strategies_produce_different_masks() -> None:
    torch.manual_seed(3)
    residual = torch.rand(2, 16, 128, 128)
    masks = {
        name: instantiate_from_config(
            OmegaConf.load(REPO / "configs" / "threshold" / f"{name}.yaml")
        )(residual)
        for name in ("percentile", "absolute", "otsu")
    }
    assert not torch.equal(masks["percentile"], masks["absolute"])
    assert not torch.equal(masks["percentile"], masks["otsu"])


# ── Acceptance #5: sweep produces a Dice curve, argmax on val, refuses test split ──────────────
def test_threshold_sweep_selects_the_engineered_optimum() -> None:
    torch.manual_seed(4)
    shape = (1, 16, 32, 32)
    n = 1 * 16 * 32 * 32
    cube_size = int(round(n * 0.05))  # matches the p95 flagged-voxel count exactly

    flat_residual = torch.rand(n) * 0.3  # background strictly below the cube's value
    flat_residual[:cube_size] = 1.0  # the "tumor": unambiguously the highest residual voxels
    residual = flat_residual.view(shape)

    flat_gt = torch.zeros(n)
    flat_gt[:cube_size] = 1.0
    gt = flat_gt.view(shape)

    result = ReconResult(
        volume_id="synthetic-0",
        original=torch.zeros(shape),
        reconstruction=torch.zeros(shape),
        residual=residual,
        anomaly_mask=torch.zeros(shape),
        ground_truth=gt,
    )

    sweep_cfg = OmegaConf.create(
        {"percentiles": [90.0, 92.5, 95.0, 97.5, 99.0], "absolute": [0.05, 0.1, 0.15, 0.2]}
    )
    points = run_threshold_sweep([result], sweep_cfg, split="val")
    assert len(points) == 9  # one row per configured sweep point

    best = select_operating_point(points)
    assert best.strategy == "percentile"
    assert best.value == pytest.approx(95.0)
    assert best.dice_mean == pytest.approx(1.0, abs=1e-6)

    with pytest.raises(ReconError):
        run_threshold_sweep([result], sweep_cfg, split="test")  # leakage guard


# ── Acceptance #6: two models don't collide ─────────────────────────────────────────────────────
def test_save_result_two_run_ids_do_not_collide(tmp_path: Path) -> None:
    shape = (1, 16, 32, 32)
    result_a = ReconResult(
        volume_id="vol-x",
        original=torch.zeros(shape),
        reconstruction=torch.ones(shape),
        residual=torch.ones(shape),
        anomaly_mask=torch.zeros(shape),
        ground_truth=torch.zeros(shape),
    )
    result_b = ReconResult(
        volume_id="vol-x",
        original=torch.ones(shape) * 2,
        reconstruction=torch.zeros(shape),
        residual=torch.ones(shape) * 3,
        anomaly_mask=torch.ones(shape),
        ground_truth=torch.ones(shape),
    )

    root = tmp_path / "results"
    path_a = save_result(result_a, root, "run-a")
    path_b = save_result(result_b, root, "run-b")

    assert path_a != path_b
    assert path_a.is_file() and path_b.is_file()

    loaded_a = load_result(path_a)
    loaded_b = load_result(path_b)
    assert not torch.equal(loaded_a.residual, loaded_b.residual)


# ── Acceptance #7: bit-identical round-trip ─────────────────────────────────────────────────────
def test_save_load_round_trip_is_bit_identical(tmp_path: Path) -> None:
    shape = (1, 16, 64, 64)
    torch.manual_seed(5)
    result = ReconResult(
        volume_id="vol-rt",
        original=torch.rand(shape),
        reconstruction=torch.rand(shape),
        residual=torch.rand(shape),
        anomaly_mask=(torch.rand(shape) > 0.5).float(),
        ground_truth=(torch.rand(shape) > 0.5).float(),
    )
    path = save_result(result, tmp_path / "results", "run-rt")
    loaded = load_result(path)

    assert loaded.volume_id == result.volume_id
    for field in ("original", "reconstruction", "residual", "anomaly_mask", "ground_truth"):
        assert torch.equal(getattr(loaded, field), getattr(result, field))


# ── Optional real-module smoke test ─────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_engine_run_with_small_real_unetr(tmp_path: Path) -> None:
    from mri_ad.models.unetr import UNETRReconstruction

    torch.manual_seed(0)
    model = UNETRReconstruction(
        in_channels=1,
        img_size=(16, 128, 128),
        feature_size=8,
        hidden_size=96,
        mlp_dim=192,
        num_heads=4,
        num_layers=12,
    )
    model.eval()
    engine = ReconstructionEngine(model, _cfg(tmp_path))
    result = engine.run(torch.rand(1, 16, 128, 128), volume_id="unetr-smoke")
    assert result.original.shape == (1, 16, 128, 128)
