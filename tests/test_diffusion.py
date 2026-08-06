"""Spec 013 — AnoDDPM diffusion-paradigm tests.

All models here are **downsized** and all tensors are synthetic — no real data, no checkpoints,
no full-resolution forward passes (laptop constraint, see CLAUDE.md / the "no heavy local runs"
memory). Phase 3 (the real training run and t_noise sweep) is a GPU hand-off the user invokes
manually; nothing here launches it.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip("monai")

REPO = Path(__file__).resolve().parent.parent

_SMALL_KWARGS = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_res_blocks=(1, 1),
    channels=(4, 8),
    attention_levels=(False, False),
    norm_num_groups=4,
    num_head_channels=4,
    num_train_timesteps=10,
)


def _small_diffusion(**overrides):
    from mri_ad.models.diffusion import DiffusionADModel

    torch.manual_seed(0)
    return DiffusionADModel(**{**_SMALL_KWARGS, **overrides})


# ── registration (acceptance 1) ────────────────────────────────────────────────
def test_registers_from_yaml_alone(tmp_path: Path) -> None:
    from mri_ad.models.registry import ModelRegistry

    config_dir = tmp_path / "model"
    config_dir.mkdir()
    (config_dir / "diffusion.yaml").write_text((REPO / "configs/model/diffusion.yaml").read_text())

    registry = ModelRegistry()
    registry.add_from_yaml(config_dir / "diffusion.yaml")
    model = registry.get("diffusion")

    from mri_ad.models.base import AnomalyDetectionModel

    assert isinstance(model, AnomalyDetectionModel)


def test_matrix_and_paradigm_already_declare_the_diffusion_cell() -> None:
    """Acceptance 3: the cell is data, not code — no config edit needed once a checkpoint lands."""
    matrix_cfg = yaml.safe_load((REPO / "configs/matrix/arch_loss.yaml").read_text())
    cells = {(c["model"], c["loss"]): c for c in matrix_cfg["cells"]}
    assert cells[("diffusion", "ddpm")]["role"] == "paradigm"

    paradigm_cfg = yaml.safe_load((REPO / "configs/paradigm/default.yaml").read_text())
    columns = {c["key"]: c for c in paradigm_cfg["columns"]}
    assert columns["diffusion__ddpm"]["model"] == "diffusion"
    assert columns["diffusion__ddpm"]["loss"] == "ddpm"


# ── forward shape / bounds (acceptance test 1) ──────────────────────────────────
@pytest.mark.parametrize("sampler", ["ddim", "ddpm"])
def test_forward_preserves_shape(sampler: str) -> None:
    model = _small_diffusion(t_noise=6, sampler=sampler, num_inference_steps=4)
    model.eval()
    x = torch.rand(2, 1, 4, 8, 8)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_output_is_finite_and_in_unit_range() -> None:
    model = _small_diffusion(t_noise=6, sampler="ddim", num_inference_steps=4)
    model.eval()
    x = torch.rand(1, 1, 4, 8, 8)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all()
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


# ── reproducibility (D-6, NFR-1) ────────────────────────────────────────────────
def test_ddim_forward_is_bitwise_reproducible_across_calls() -> None:
    """eta=0 DDIM + a seeded initial noise draw -> the only stochastic op is pinned down."""
    model = _small_diffusion(t_noise=6, sampler="ddim", num_inference_steps=4, sample_seed=7)
    model.eval()
    x = torch.rand(1, 1, 4, 8, 8)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.equal(y1, y2)


def test_different_sample_seeds_produce_different_output() -> None:
    x = torch.rand(1, 1, 4, 8, 8)
    a = _small_diffusion(t_noise=6, sampler="ddim", num_inference_steps=4, sample_seed=1)
    b = _small_diffusion(t_noise=6, sampler="ddim", num_inference_steps=4, sample_seed=2)
    b.load_state_dict(a.state_dict())  # same weights, only the noise draw differs
    a.eval()
    b.eval()
    with torch.no_grad():
        ya = a(x)
        yb = b(x)
    assert not torch.equal(ya, yb)


def test_reverse_timesteps_always_start_at_exactly_t_noise() -> None:
    """A t_noise that falls between two strided DDIM steps must not silently shift the SNR."""
    model = _small_diffusion(t_noise=7, sampler="ddim", num_inference_steps=3)
    steps = model._reverse_timesteps(torch.device("cpu"))  # noqa: SLF001
    assert steps[0] == 7
    assert steps == sorted(steps, reverse=True)


# ── config validation ────────────────────────────────────────────────────────────
def test_channels_not_multiple_of_norm_num_groups_raises() -> None:
    with pytest.raises(ValueError, match="norm_num_groups"):
        _small_diffusion(channels=(4, 6), norm_num_groups=4)


def test_invalid_sampler_raises() -> None:
    with pytest.raises(ValueError, match="sampler"):
        _small_diffusion(sampler="ancestral")


# ── checkpoint (D-5) ─────────────────────────────────────────────────────────────
def test_own_checkpoint_round_trips_into_the_whole_wrapper(tmp_path: Path) -> None:
    model = _small_diffusion(t_noise=6)
    ckpt = tmp_path / "diffusion.pth"
    torch.save(model.state_dict(), ckpt)

    fresh = _small_diffusion(t_noise=6)
    fresh.load_checkpoint(ckpt)  # must not raise
    assert set(fresh.state_dict().keys()) == set(model.state_dict().keys())
    assert any(k.startswith("net.") for k in model.state_dict())


# ── model_card (documents the D-1 deviation) ──────────────────────────────────────
def test_model_card_documents_gaussian_not_simplex_noise() -> None:
    card = _small_diffusion().model_card
    assert "simplex" in card.known_characteristics.lower()
    assert "Spec 013" in card.training_data


# ── training objective (Phase 2, D-7) ───────────────────────────────────────────
def test_ddpm_step_returns_finite_scalar_with_live_grad_path() -> None:
    from torch import nn

    from mri_ad.train.objectives import ddpm_step

    model = _small_diffusion(t_noise=6)
    model.train()
    batch = {"image": torch.rand(2, 1, 4, 8, 8)}
    step = functools.partial(ddpm_step, criterion=nn.MSELoss())

    loss = step(model, batch, torch.device("cpu"))
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.net.parameters() if p.grad is not None]
    assert grad_norms and any(g > 0 for g in grad_norms)


def test_ddpm_step_reads_image_key_not_corrupted_healthy() -> None:
    from torch import nn

    from mri_ad.train.objectives import ddpm_step

    model = _small_diffusion(t_noise=6)
    step = functools.partial(ddpm_step, criterion=nn.MSELoss())
    with pytest.raises(KeyError):
        step(model, {"corrupted": torch.rand(1, 1, 4, 8, 8)}, torch.device("cpu"))


# ── from-scratch training wiring (D-8), no GPU ─────────────────────────────────
def test_from_scratch_config_resolves_and_builds_at_random_init(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(REPO))
    import scripts.run_train as run_train
    from omegaconf import OmegaConf

    from mri_ad.models import ModelRegistry
    from mri_ad.models.diffusion import DiffusionADModel

    registry = ModelRegistry()
    registry.register("diffusion", DiffusionADModel)
    registry._configs["diffusion"] = OmegaConf.create(  # noqa: SLF001
        {"params": dict(_SMALL_KWARGS), "checkpoint": str(tmp_path / "diffusion.pth")}
    )

    cfg = OmegaConf.create({"train": {"from_scratch": True, "save_as": "diffusion"}})
    model = run_train._build_and_warm_start_model(registry, cfg)  # noqa: SLF001
    assert isinstance(model, DiffusionADModel)
    # No checkpoint was loaded — the file at registry.checkpoint_path("diffusion") never had to
    # exist, unlike the fine-tune path's init_from/warm_start_from.
    assert not (tmp_path / "diffusion.pth").exists()


def test_ddpm_scratch_config_selects_val_loss_not_val_dice() -> None:
    cfg = yaml.safe_load((REPO / "configs/train/ddpm_scratch.yaml").read_text())
    assert cfg["from_scratch"] is True
    assert cfg["save_as"] == "diffusion"
    assert cfg["selection"]["metric"] == "val_loss"
    assert cfg["selection"]["mode"] == "min"


# ── boundary (Contract) ──────────────────────────────────────────────────────────
def test_no_downstream_special_casing() -> None:
    """No layer outside models/ may special-case diffusion by name (Spec 002/013 acceptance 2)."""
    import subprocess

    result = subprocess.run(
        ["grep", "-rl", "diffusion", "src/mri_ad/recon", "src/mri_ad/classical", "src/mri_ad/viz"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"diffusion referenced outside models/: {result.stdout}"


def test_eval_names_the_pre_existing_diffusion_available_stat() -> None:
    """eval/'s only structural hook for diffusion is the diffusion_available cross-cell stat.

    Not a stricter "diffusion appears nowhere else" grep — matrix.py/paradigm.py's own docstrings
    prose-describe that stat ("diffusion availability") without using the identifier, which would
    make a stricter check brittle against comment wording rather than the acceptance-2 contract.
    """
    for dirname in ("matrix.py", "paradigm.py"):
        text = (REPO / "src/mri_ad/eval" / dirname).read_text()
        assert "diffusion_available" in text
