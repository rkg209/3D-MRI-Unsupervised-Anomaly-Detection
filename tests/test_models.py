"""Spec 002 model-registry & checkpoint-loading tests.

All runnable without real data or checkpoints, on synthetic tensors and small/downsized
architectures (mirroring the ``small_unetr`` fixture in ``test_slice.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("monai")


def _small_unet():
    from mri_ad.models.unet import UNetModel

    return UNetModel(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        num_res_units=1,
        norm="BATCH",
        dropout=0.0,
    )


def _small_attention_unet():
    from mri_ad.models.attention_unet import AttentionUNetModel

    return AttentionUNetModel(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        kernel_size=3,
    )


def _small_unetr():
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


def _small_diffusion():
    from mri_ad.models.diffusion import DiffusionADModel

    # t_noise=0, sampler="ddpm" -> exactly one reverse step (no DDIM strided schedule to build,
    # which keeps this factory decoupled from num_train_timesteps/num_inference_steps sizing —
    # tests/test_diffusion.py exercises the real DDIM path). A full-resolution (16,128,128) 3D
    # conv net call is expensive with no GPU; the shape/interface contract only needs the loop
    # to run >=1 time.
    return DiffusionADModel(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_res_blocks=(1, 1),
        channels=(4, 8),
        attention_levels=(False, False),
        norm_num_groups=4,
        num_head_channels=4,
        num_train_timesteps=2,
        t_noise=0,
        sampler="ddpm",
    )


def _small_msa_unetr():
    from mri_ad.models.msa_unetr import MultiScaleAttentionUNETR

    torch.manual_seed(0)
    return MultiScaleAttentionUNETR(
        in_channels=1,
        img_size=(16, 128, 128),
        feature_size=8,
        hidden_size=96,
        mlp_dim=192,
        num_heads=4,
        num_layers=12,
    )


MODEL_FACTORIES = {
    "unet": _small_unet,
    "attention_unet": _small_attention_unet,
    "unetr": _small_unetr,
    "diffusion": _small_diffusion,
    "msa_unetr": _small_msa_unetr,
}


# ── forward shape (acceptance test 2) ──────────────────────────────────────────
@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_forward_preserves_shape(name: str) -> None:
    model = MODEL_FACTORIES[name]()
    model.eval()
    x = torch.rand(2, 1, 16, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


# ── output bounds (acceptance test 3) ──────────────────────────────────────────
def test_unetr_output_bounded_to_unit_interval() -> None:
    model = _small_unetr()
    model.eval()
    x = torch.rand(1, 1, 16, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


@pytest.mark.parametrize("name", ["unet", "attention_unet"])
def test_unet_family_wraps_net_with_no_extra_output_activation(name: str) -> None:
    """UNet/AttUNet models apply no activation on top of the wrapped net's own output.

    (AttentionUnet's internal attention *gates* use a Sigmoid — that's not the output head; the
    interface guarantee is just that ``forward`` is ``self.net(x)``, nothing appended after it.)
    """
    model = MODEL_FACTORIES[name]()
    x = torch.rand(1, 1, 16, 128, 128)
    model.eval()
    with torch.no_grad():
        wrapped_out = model(x)
        net_out = model.net(x)
    assert torch.equal(wrapped_out, net_out)


# Models whose load_checkpoint targets the whole wrapper (D-5), not the inner `.net` — the
# on-disk keys are `net.*`, matching what CheckpointWriter actually saves.
WRAPPER_LEVEL_CHECKPOINT_MODELS = {"diffusion"}


# ── checkpoint round-trip, both layouts (acceptance tests 1 & 5) ──────────────
@pytest.mark.parametrize("name", ["unet", "attention_unet", "unetr", "diffusion"])
def test_checkpoint_round_trip_both_layouts(name: str, tmp_path: Path) -> None:
    model = MODEL_FACTORIES[name]()
    if name in WRAPPER_LEVEL_CHECKPOINT_MODELS:
        inner = model
    else:
        inner = model.net if hasattr(model, "net") else model
    state = inner.state_dict()

    bare = tmp_path / f"{name}_bare.pth"
    wrapped = tmp_path / f"{name}_wrapped.pth"
    torch.save(state, bare)
    torch.save({"model_state_dict": state, "epoch": 1}, wrapped)

    for path in (bare, wrapped):
        fresh = MODEL_FACTORIES[name]()
        fresh.load_checkpoint(path)  # must not raise
        if name in WRAPPER_LEVEL_CHECKPOINT_MODELS:
            fresh_inner = fresh
        else:
            fresh_inner = fresh.net if hasattr(fresh, "net") else fresh
        loaded_keys = set(fresh_inner.state_dict().keys())
        assert loaded_keys == set(state.keys())


# ── missing / stub checkpoints raise CheckpointError (acceptance test 4) ──────
@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_missing_checkpoint_raises(name: str, tmp_path: Path) -> None:
    from mri_ad.exceptions import CheckpointError

    model = MODEL_FACTORIES[name]()
    with pytest.raises(CheckpointError, match="nope.pth"):
        model.load_checkpoint(tmp_path / "nope.pth")


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_lfs_stub_checkpoint_raises(name: str, tmp_path: Path) -> None:
    from mri_ad.exceptions import CheckpointError

    stub = tmp_path / "stub.pth"
    stub.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 133\n")

    model = MODEL_FACTORIES[name]()
    with pytest.raises(CheckpointError, match="stub"):
        model.load_checkpoint(stub)


# ── model_card (acceptance test 7) ─────────────────────────────────────────────
@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_model_card_param_count_and_training_loss(name: str) -> None:
    model = MODEL_FACTORIES[name]()
    card = model.model_card
    assert card.param_count == sum(p.numel() for p in model.parameters())
    assert card.training_loss


# ── registry (acceptance test 6) ───────────────────────────────────────────────
def test_registry_rejects_incomplete_interface() -> None:
    from mri_ad.exceptions import ModelRegistrationError
    from mri_ad.models.base import AnomalyDetectionModel
    from mri_ad.models.registry import ModelRegistry

    class _Incomplete(AnomalyDetectionModel):
        def forward(self, x):  # noqa: ANN001, D102
            return x

        # load_checkpoint and model_card deliberately not implemented.

    registry = ModelRegistry()
    with pytest.raises(ModelRegistrationError):
        registry.register("incomplete", _Incomplete)


def test_registry_accepts_conforming_class_and_get_returns_interface_instance() -> None:
    from mri_ad.models.registry import ModelRegistry

    registry = ModelRegistry()
    registry.register("unetr", type(_small_unetr()))
    assert "unetr" in registry._classes  # noqa: SLF001


def test_build_default_registry_get_and_checkpoint_path() -> None:
    from mri_ad.models.base import AnomalyDetectionModel
    from mri_ad.models.registry import build_default_registry

    registry = build_default_registry()
    # Full-size UNETR (real configs/model/unetr.yaml params) — cheap to construct, not forwarded
    # here (that's `_small_unetr`'s job above); a full-size forward pass is a manual smoke check.
    model = registry.get("unetr")
    assert isinstance(model, AnomalyDetectionModel)
    assert registry.checkpoint_path("unetr").name == "unetr_mse_ssim_aug.pth"

    for name in ("unet", "attention_unet", "diffusion"):
        assert registry.checkpoint_path(name).suffix == ".pth"
