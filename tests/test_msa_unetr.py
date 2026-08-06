"""Spec 012 (stretch) — multi-scale attention UNETR variant tests.

All models here are **downsized** (``feature_size=8, hidden_size=96, num_heads=4``) and all
tensors are synthetic — no real data, no checkpoints, no full-resolution forward passes (laptop
constraint, see CLAUDE.md and .claude/plans/012-multiscale-attention.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip("monai")

REPO = Path(__file__).resolve().parent.parent

_SMALL_KWARGS = dict(
    in_channels=1,
    img_size=(16, 128, 128),
    feature_size=8,
    hidden_size=96,
    mlp_dim=192,
    num_heads=4,
    num_layers=12,
)


def _small_unetr():
    from mri_ad.models.unetr import UNETRReconstruction

    torch.manual_seed(0)
    return UNETRReconstruction(**_SMALL_KWARGS)


def _small_msa_unetr(**overrides):
    from mri_ad.models.msa_unetr import MultiScaleAttentionUNETR

    torch.manual_seed(0)
    return MultiScaleAttentionUNETR(**{**_SMALL_KWARGS, **overrides})


# ── registration (acceptance 1) ────────────────────────────────────────────────
def test_registers_from_yaml_alone(tmp_path: Path) -> None:
    """The model registers with zero downstream code changes, from the YAML alone."""
    from mri_ad.models.registry import ModelRegistry

    config_dir = tmp_path / "model"
    config_dir.mkdir()
    (config_dir / "msa_unetr.yaml").write_text((REPO / "configs/model/msa_unetr.yaml").read_text())

    registry = ModelRegistry()
    registry.add_from_yaml(config_dir / "msa_unetr.yaml")
    model = registry.get("msa_unetr")

    from mri_ad.models.base import AnomalyDetectionModel

    assert isinstance(model, AnomalyDetectionModel)


def test_matrix_declares_msa_cell() -> None:
    """Acceptance 2: the row is declared, present, and honestly n/a before any number exists."""
    cfg = yaml.safe_load((REPO / "configs/matrix/arch_loss.yaml").read_text())
    cells = {(c["model"], c["loss"]): c for c in cfg["cells"]}
    cell = cells[("msa_unetr", "mse_ssim")]
    assert cell["role"] == "stretch"
    assert "gate closed" in cell["na_reason"] or "not trained" in cell["na_reason"]
    assert not (REPO / "artifacts/metrics/msa_unetr__mse_ssim/aggregate.json").exists()


# ── identity-at-init (the load-bearing test) ────────────────────────────────────
def test_identity_at_init_matches_plain_unetr() -> None:
    unetr = _small_unetr()
    msa = _small_msa_unetr()

    own_state = msa.state_dict()
    own_state.update(unetr.state_dict())
    msa.load_state_dict(own_state, strict=True)

    unetr.eval()
    msa.eval()
    x = torch.rand(1, 1, 16, 128, 128)
    with torch.no_grad():
        assert torch.equal(unetr(x), msa(x))


def test_gate_is_exactly_one_at_init() -> None:
    from mri_ad.models.attention_gate import MultiScaleAttentionGate

    torch.manual_seed(0)
    gate = MultiScaleAttentionGate(f_g=4, f_x=4, f_int=2, f_context=6)
    g = torch.randn(2, 4, 3, 5, 5)
    x = torch.randn(2, 4, 3, 5, 5)
    context = torch.randn(2, 6, 1, 2, 2)
    assert torch.equal(gate(g, x, context), x)


def test_gate_learns_away_from_identity() -> None:
    from mri_ad.models.attention_gate import MultiScaleAttentionGate

    torch.manual_seed(0)
    gate = MultiScaleAttentionGate(f_g=4, f_x=4, f_int=2)
    g = torch.randn(1, 4, 3, 5, 5)
    x = torch.randn(1, 4, 3, 5, 5)
    with torch.no_grad():
        gate.psi.conv.weight.add_(0.5)
    assert not torch.equal(gate(g, x), x)


def test_gate_context_argument_mismatch_raises() -> None:
    from mri_ad.models.attention_gate import MultiScaleAttentionGate

    with_context = MultiScaleAttentionGate(f_g=4, f_x=4, f_int=2, f_context=3)
    without_context = MultiScaleAttentionGate(f_g=4, f_x=4, f_int=2)
    g = torch.randn(1, 4, 3, 5, 5)
    x = torch.randn(1, 4, 3, 5, 5)
    context = torch.randn(1, 3, 1, 2, 2)

    with pytest.raises(ValueError, match="context"):
        with_context(g, x)  # declared f_context but no context passed
    with pytest.raises(ValueError, match="context"):
        without_context(g, x, context)  # no f_context but context passed


# ── forward shape / bounds (inherited via MODEL_FACTORIES, mirrored here directly) ─────────────
def test_forward_preserves_shape() -> None:
    model = _small_msa_unetr()
    model.eval()
    x = torch.rand(2, 1, 16, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_output_bounded_to_unit_interval() -> None:
    model = _small_msa_unetr()
    model.eval()
    x = torch.rand(1, 1, 16, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


# ── warm-start / checkpoint (D4, R3, R4) ────────────────────────────────────────
def test_load_from_unetr_leaves_gates_at_identity(tmp_path: Path) -> None:
    unetr = _small_unetr()
    ckpt = tmp_path / "unetr.pth"
    torch.save(unetr.state_dict(), ckpt)

    msa = _small_msa_unetr()
    msa.load_from_unetr(ckpt)

    unetr.eval()
    msa.eval()
    x = torch.rand(1, 1, 16, 128, 128)
    with torch.no_grad():
        assert torch.equal(unetr(x), msa(x))


def test_load_from_unetr_rejects_extra_or_missing_keys(tmp_path: Path) -> None:
    from mri_ad.exceptions import CheckpointError

    unetr = _small_unetr()
    state = unetr.state_dict()

    # A renamed key: something is missing AND something is unexpected.
    renamed = {("renamed." + k if k == "out.weight" else k): v for k, v in state.items()}
    renamed_ckpt = tmp_path / "renamed.pth"
    torch.save(renamed, renamed_ckpt)
    with pytest.raises(CheckpointError):
        _small_msa_unetr().load_from_unetr(renamed_ckpt)

    # A checkpoint that already includes gate keys (i.e. is an msa_unetr checkpoint).
    msa_ckpt_state = _small_msa_unetr().state_dict()
    msa_ckpt = tmp_path / "msa.pth"
    torch.save(msa_ckpt_state, msa_ckpt)
    with pytest.raises(CheckpointError):
        _small_msa_unetr().load_from_unetr(msa_ckpt)


def test_own_checkpoint_round_trips_strictly(tmp_path: Path) -> None:
    model = _small_msa_unetr()
    ckpt = tmp_path / "msa_unetr.pth"
    torch.save(model.state_dict(), ckpt)

    fresh = _small_msa_unetr()
    fresh.load_checkpoint(ckpt)  # must not raise
    assert set(fresh.state_dict().keys()) == set(model.state_dict().keys())


def test_missing_and_stub_checkpoints_raise(tmp_path: Path) -> None:
    from mri_ad.exceptions import CheckpointError

    model = _small_msa_unetr()
    with pytest.raises(CheckpointError, match="nope.pth"):
        model.load_checkpoint(tmp_path / "nope.pth")
    with pytest.raises(CheckpointError, match="nope.pth"):
        model.load_from_unetr(tmp_path / "nope.pth")

    stub = tmp_path / "stub.pth"
    stub.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 133\n")
    with pytest.raises(CheckpointError, match="stub"):
        model.load_checkpoint(stub)
    with pytest.raises(CheckpointError, match="stub"):
        model.load_from_unetr(stub)


# ── ablation flag (D5(b), the free multi-scale-context ablation) ───────────────
def test_ablation_flag_changes_parameter_count() -> None:
    with_context = _small_msa_unetr(multiscale_context=True)
    without_context = _small_msa_unetr(multiscale_context=False)
    n_with = sum(p.numel() for p in with_context.parameters())
    n_without = sum(p.numel() for p in without_context.parameters())
    assert n_with > n_without


def test_gated_scales_flag_drops_gate_modules() -> None:
    partial = _small_msa_unetr(gated_scales=(4,))
    assert partial.gate4 is not None
    assert partial.gate1 is None
    assert partial.gate2 is None
    assert partial.gate3 is None

    x = torch.rand(1, 1, 16, 128, 128)
    partial.eval()
    with torch.no_grad():
        y = partial(x)
    assert y.shape == x.shape


# ── config surface ───────────────────────────────────────────────────────────────
def test_params_match_unetr_yaml() -> None:
    unetr_cfg = yaml.safe_load((REPO / "configs/model/unetr.yaml").read_text())["params"]
    msa_cfg = yaml.safe_load((REPO / "configs/model/msa_unetr.yaml").read_text())["params"]
    for key, value in unetr_cfg.items():
        assert msa_cfg[key] == value, f"msa_unetr.yaml params.{key} diverges from unetr.yaml"


def test_no_batchnorm_anywhere() -> None:
    """R7: BatchNorm at batch size 2 creates a train/eval skew; the gate must use instance norm."""
    model = _small_msa_unetr()
    for module in model.modules():
        assert "BatchNorm" not in type(module).__name__


# ── boundary (the Contract) ─────────────────────────────────────────────────────
def test_no_downstream_special_casing() -> None:
    """Mechanizes the Contract: no layer outside models/ may special-case msa_unetr by name."""
    import subprocess

    result = subprocess.run(
        [
            "grep",
            "-rl",
            "msa_unetr",
            "src/mri_ad/recon",
            "src/mri_ad/eval",
            "src/mri_ad/classical",
            "src/mri_ad/viz",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"msa_unetr referenced outside models/: {result.stdout}"
