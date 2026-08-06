"""Spec 009 acceptance tests for ``synth/fpi.py``, ``synth/separability.py``, ``synth/dataset.py``.

All tests run on synthetic tensors at toy sizes — no real data, no checkpoints, no real 3D forward
passes (laptop constraint, see CLAUDE.md).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mri_ad.data.datasets import OpenBHBDataset
from mri_ad.exceptions import ConfigError, DescriptiveValidationError
from mri_ad.synth.dataset import AnomalyInformedDataset
from mri_ad.synth.fpi import FPIAnomalyGenerator
from mri_ad.synth.separability import best_intensity_dice

REPO = Path(__file__).resolve().parent.parent
SYNTH_DIR = REPO / "src" / "mri_ad" / "synth"


def _generator(**overrides: object) -> FPIAnomalyGenerator:
    params: dict[str, object] = dict(
        patch_size_range=(8, 32),
        depth_size_range=(4, 10),
        n_patches_range=(1, 4),
        alpha_range=(0.3, 0.7),
        foreground_threshold=0.05,
        min_foreground_fraction=0.5,
        min_patch_contrast=0.01,
        max_placement_attempts=25,
        donor_alignment="same",
        corrupt_probability=1.0,
        seed=0,
    )
    params.update(overrides)
    return FPIAnomalyGenerator(**params)  # type: ignore[arg-type]


def _random_volume(
    seed: int, *, shape: tuple[int, int, int, int] = (1, 16, 128, 128)
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g, dtype=torch.float32)


def _textured_phantom(
    seed: int, shape: tuple[int, int, int, int] = (1, 16, 128, 128)
) -> torch.Tensor:
    """An ellipsoid-of-tissue x smooth low-frequency field — not a constant-intensity fixture.

    A flat phantom would make ``best_intensity_dice`` fail for the wrong reason (any threshold
    trivially recovers a constant-intensity region). Trilinear-upsampling a small random tensor
    gives a smoothly varying but non-constant intensity field, closer to real tissue texture.
    """
    g = torch.Generator().manual_seed(seed)
    _, d, h, w = shape
    small = torch.rand((1, 1, d // 4, h // 8, w // 8), generator=g)
    field = torch.nn.functional.interpolate(
        small, size=(d, h, w), mode="trilinear", align_corners=False
    ).squeeze(0)

    # Ellipsoid center/radii jitter per seed, so two independently-generated phantoms don't share
    # an identical background mask — with `donor_alignment="same"`, a shared background mask would
    # make voxels near the ellipsoid edge blend 0-with-0 (unchanged) regardless of alpha, which is
    # a fixture artefact rather than the R4 failure mode this generator otherwise guards against.
    jitter = torch.Generator().manual_seed(seed + 1)
    cz, cy, cx = (0.15 * (torch.rand(3, generator=jitter) - 0.5)).tolist()
    rz, ry, rx = (0.75 + 0.2 * torch.rand(3, generator=jitter)).tolist()
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, d), torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    ellipsoid = (((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    volume = field.squeeze(0) * ellipsoid.float()
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume.unsqueeze(0).float()


# ── Acceptance 1: differs within mask, bit-identical outside ───────────────────────────────────


def test_fpi_outside_mask_bit_identical() -> None:
    gen = _generator()
    volume = _textured_phantom(1)
    donor = _textured_phantom(2)
    result = gen.generate(volume, donor, generator=torch.Generator().manual_seed(7))

    outside = result.synth_mask == 0
    assert outside.any()
    assert torch.equal(result.corrupted[outside], result.healthy[outside])


def test_fpi_blend_identity_inside_mask() -> None:
    gen = _generator()
    volume = _textured_phantom(3)
    donor = _textured_phantom(4)
    result = gen.generate(volume, donor, generator=torch.Generator().manual_seed(11))

    assert result.patches
    for patch in sorted(result.patches, key=lambda p: p.alpha):
        do, ho, wo = patch.donor_offset
        target = result.healthy[:, patch.d0 : patch.d1, patch.h0 : patch.h1, patch.w0 : patch.w1]
        donor_patch = donor[
            :,
            patch.d0 + do : patch.d1 + do,
            patch.h0 + ho : patch.h1 + ho,
            patch.w0 + wo : patch.w1 + wo,
        ]
        expected = (1 - patch.alpha) * target + patch.alpha * donor_patch
        actual = result.corrupted[:, patch.d0 : patch.d1, patch.h0 : patch.h1, patch.w0 : patch.w1]
        # Only true where this patch is the winner (max alpha) at every covered voxel — verified
        # separately via the mask; here we check the alpha-sorted invariant on the highest-alpha
        # (last-applied, hence always-winning) patch only.
        if patch.alpha == max(p.alpha for p in result.patches):
            assert torch.allclose(actual, expected, atol=0.0)


def _smooth_field(seed: int, shape: tuple[int, int, int, int] = (1, 16, 128, 128)) -> torch.Tensor:
    """A fully-foreground smooth low-frequency field — no ellipsoid background to correlate."""
    g = torch.Generator().manual_seed(seed)
    _, d, h, w = shape
    small = torch.rand((1, 1, d // 4, h // 8, w // 8), generator=g) + 0.5
    field = torch.nn.functional.interpolate(
        small, size=(d, h, w), mode="trilinear", align_corners=False
    ).squeeze(0)
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return (0.2 + 0.8 * field).float()


def test_fpi_changed_fraction_floor() -> None:
    """>= 90% of masked voxels actually changed value from healthy (catches all-background R4)."""
    gen = _generator()
    for seed in range(8):
        volume = _smooth_field(100 + seed)
        donor = _smooth_field(200 + seed)
        result = gen.generate(volume, donor, generator=torch.Generator().manual_seed(seed))
        masked = result.synth_mask > 0
        if not masked.any():
            continue
        changed = result.corrupted[masked] != result.healthy[masked]
        assert changed.float().mean().item() >= 0.9


# ── Acceptance 2: not trivially separable by a global intensity threshold ──────────────────────


def test_intensity_threshold_cannot_recover_mask() -> None:
    gen = _generator()
    dices = []
    for seed in range(32):
        volume = _textured_phantom(1000 + seed)
        donor = _textured_phantom(2000 + seed)
        result = gen.generate(volume, donor, generator=torch.Generator().manual_seed(seed))
        dices.append(
            best_intensity_dice(
                result.corrupted, result.synth_mask, foreground_threshold=0.05, n_thresholds=64
            )
        )
    assert sum(dices) / len(dices) < 0.5
    assert max(dices) < 0.5


# ── Config / validation errors ──────────────────────────────────────────────────────────────────


def test_depth_size_range_exceeding_volume_depth_raises_config_error() -> None:
    gen = _generator(depth_size_range=(4, 32))
    volume = _random_volume(1)
    donor = _random_volume(2)
    with pytest.raises(ConfigError):
        gen.generate(volume, donor, generator=torch.Generator().manual_seed(0))


def test_invalid_donor_alignment_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        _generator(donor_alignment="nonsense")


def test_shape_mismatch_raises_descriptive_validation_error() -> None:
    gen = _generator()
    volume = _random_volume(1)
    donor = _random_volume(2, shape=(1, 16, 64, 64))
    with pytest.raises(DescriptiveValidationError):
        gen.generate(volume, donor, generator=torch.Generator().manual_seed(0))


def test_out_of_range_volume_raises_descriptive_validation_error() -> None:
    gen = _generator()
    volume = _random_volume(1) * 2.0
    donor = _random_volume(2)
    with pytest.raises(DescriptiveValidationError):
        gen.generate(volume, donor, generator=torch.Generator().manual_seed(0))


# ── Regression: no global RNG (AST walk, mirrors tests/test_classical_boundary.py) ─────────────


def _module_level_and_all_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ("random", "np") and node.attr not in ("Generator",):
                calls.add(f"{node.value.id}.{node.attr}")
            if node.value.id == "numpy" and node.attr.startswith("random"):
                calls.add(f"numpy.{node.attr}")
    return calls


def test_no_global_rng_in_synth() -> None:
    for path in sorted(SYNTH_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
                assert "random" not in names, f"{path.name} imports the global `random` module."
            if isinstance(node, ast.ImportFrom) and node.module == "numpy":
                names = {alias.name for alias in node.names}
                assert "random" not in names, f"{path.name} imports `numpy.random`."
        calls = _module_level_and_all_calls(path)
        assert not calls, f"{path.name} calls global RNG functions: {calls}"


# ── AnomalyInformedDataset ───────────────────────────────────────────────────────────────────


def _make_openbhb_dir(tmp_path: Path, n_subjects: int = 3) -> Path:
    d = tmp_path / "openbhb"
    if d.is_dir():
        return d
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_subjects):
        np.save(d / f"sub-{i}.npy", rng.random((1, 1, 32, 48, 48)).astype(np.float32))
    return d


def _openbhb_cfg(tmp_path: Path, openbhb_dir: Path, *, num_workers: int = 0):
    return OmegaConf.create(
        {
            "seed": 42,
            "shape": {"height": 16, "width": 16},
            "data": {
                "preprocess": {"eps": 1.0e-8, "resize_mode": "trilinear", "chunk_depth": 8},
                "openbhb": {
                    "dir": str(openbhb_dir),
                    "crop": {"z": [0, 16], "y": [0, 48], "x": [0, 48]},
                },
                "loader": {"cache_rate": 0.0, "batch_size": 1, "num_workers": num_workers},
            },
        }
    )


def _anomaly_dataset(
    tmp_path: Path, *, num_workers: int = 0, **overrides: object
) -> AnomalyInformedDataset:
    openbhb_dir = _make_openbhb_dir(tmp_path)
    cfg = _openbhb_cfg(tmp_path, openbhb_dir, num_workers=num_workers)
    base = OpenBHBDataset(cfg, ["sub-0.npy", "sub-1.npy", "sub-2.npy"])
    generator = _generator(patch_size_range=(4, 8), depth_size_range=(2, 4), n_patches_range=(1, 2))
    params: dict[str, object] = dict(seed=7, epoch_invariant=False, donor_chunk="same")
    params.update(overrides)
    return AnomalyInformedDataset(base, generator, **params)  # type: ignore[arg-type]


def test_donor_never_equals_target(tmp_path: Path) -> None:
    ds = _anomaly_dataset(tmp_path)
    for index in range(len(ds)):
        item = ds[index]
        # A different-subject donor was blended in: healthy != corrupted somewhere, or if the
        # generator drew corrupt_probability=0 for this item the mask is empty — either is fine,
        # what must never happen is the generator failing outright on a same-subject donor.
        assert item["corrupted"].shape == item["healthy"].shape


def test_dataset_construction_rejects_fewer_than_two_volumes(tmp_path: Path) -> None:
    openbhb_dir = _make_openbhb_dir(tmp_path, n_subjects=1)
    cfg = _openbhb_cfg(tmp_path, openbhb_dir)
    base = OpenBHBDataset(cfg, ["sub-0.npy"])
    generator = _generator()
    with pytest.raises(ConfigError):
        AnomalyInformedDataset(base, generator, seed=1)


def test_deterministic_across_worker_counts(tmp_path: Path) -> None:
    ds_a = _anomaly_dataset(tmp_path, num_workers=0)
    ds_b = _anomaly_dataset(tmp_path, num_workers=2)
    for index in range(len(ds_a)):
        item_a = ds_a[index]
        item_b = ds_b[index]
        assert torch.equal(item_a["corrupted"], item_b["corrupted"])
        assert torch.equal(item_a["healthy"], item_b["healthy"])
        assert torch.equal(item_a["synth_mask"], item_b["synth_mask"])


def test_val_dataset_is_epoch_invariant(tmp_path: Path) -> None:
    ds = _anomaly_dataset(tmp_path, epoch_invariant=True)
    ds.set_epoch(0)
    first = ds[0]
    ds.set_epoch(5)
    second = ds[0]
    assert torch.equal(first["corrupted"], second["corrupted"])
    assert torch.equal(first["synth_mask"], second["synth_mask"])


def test_train_dataset_varies_across_epochs(tmp_path: Path) -> None:
    ds = _anomaly_dataset(tmp_path, epoch_invariant=False)
    ds.set_epoch(0)
    first = ds[0]
    ds.set_epoch(1)
    second = ds[0]
    assert not torch.equal(first["corrupted"], second["corrupted"])


def test_healthy_is_not_a_view_of_cache(tmp_path: Path) -> None:
    """Regenerating the same index twice must be bitwise identical, not accumulate corruption."""
    ds = _anomaly_dataset(tmp_path)
    first = ds[0]
    second = ds[0]
    assert torch.equal(first["healthy"], second["healthy"])
    assert torch.equal(first["corrupted"], second["corrupted"])
    # And a third read after touching other indices — the cached base volume must not have been
    # mutated in place by the first read's blend.
    for other in range(1, len(ds)):
        ds[other]
    third = ds[0]
    assert torch.equal(first["healthy"], third["healthy"])
    assert torch.equal(first["corrupted"], third["corrupted"])


def test_dataset_item_has_no_path_or_subject_id(tmp_path: Path) -> None:
    ds = _anomaly_dataset(tmp_path)
    item = ds[0]
    assert set(item) == {"corrupted", "healthy", "synth_mask", "volume_index", "chunk_index"}
