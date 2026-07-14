"""Scaffold contract tests.

These do not test ML behaviour — there is none yet. They test that the *contracts* the whole
project hangs on are in place and cannot silently drift: the shape invariant, the exception
hierarchy, the model interface, and the config tree. Real behaviour tests arrive spec by spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def test_volume_shape_invariant() -> None:
    from mri_ad import VOLUME_SHAPE

    assert VOLUME_SHAPE == (1, 16, 128, 128)


def test_exception_hierarchy_is_rooted() -> None:
    """Every package error must be catchable via the one base class."""
    from mri_ad import exceptions as ex

    leaves = [
        ex.DescriptiveValidationError,
        ex.SplitContractViolationError,
        ex.CheckpointError,
        ex.ModelRegistrationError,
        ex.ReconError,
        ex.EvalError,
        ex.ConfigError,
        ex.ArtifactError,
    ]
    for leaf in leaves:
        assert issubclass(leaf, ex.MRIAnomalyDetectionError)


def test_model_interface_cannot_be_instantiated() -> None:
    """AnomalyDetectionModel is an ABC — a partial implementation must not register."""
    from mri_ad.models.base import AnomalyDetectionModel

    with pytest.raises(TypeError):
        AnomalyDetectionModel()  # type: ignore[abstract]


def test_model_interface_requires_all_three_methods() -> None:
    from mri_ad.models.base import AnomalyDetectionModel

    for method in ("forward", "load_checkpoint", "model_card"):
        assert method in AnomalyDetectionModel.__abstractmethods__


@pytest.mark.parametrize(
    "rel",
    [
        "configs/config.yaml",
        "configs/data/default.yaml",
        "configs/model/unetr.yaml",
        "configs/model/unet.yaml",
        "configs/model/attention_unet.yaml",
        "configs/loss/mse_ssim.yaml",
        "configs/threshold/percentile.yaml",
        "configs/experiment/laptop.yaml",
        "configs/experiment/cluster.yaml",
    ],
)
def test_configs_parse(rel: str) -> None:
    """Every config must be valid YAML. No magic numbers live in code (FR-2)."""
    path = REPO / rel
    assert path.exists(), f"missing config: {rel}"
    assert yaml.safe_load(path.read_text()) is not None


def test_unetr_config_points_at_the_ported_class_not_monai() -> None:
    """Guards the Spec 002 deviation.

    The trained UNETR weights come from the custom `UNETR_Reconstruction`, whose output head
    and final Sigmoid differ from `monai.networks.nets.UNETR`. Pointing this config at MONAI's
    class would break state_dict loading — or worse, load partially and yield garbage.
    """
    cfg = yaml.safe_load((REPO / "configs/model/unetr.yaml").read_text())
    assert cfg["target"] == "mri_ad.models.unetr.UNETRReconstruction"
    assert "monai.networks.nets" not in cfg["target"]


def test_brats_labels_are_binarized_with_nearest_interpolation() -> None:
    """Guards against reintroducing the prior work's invalid-Dice bug.

    BraTS labels are {0,1,2,4}. Trilinear-interpolating them yields fractional 'labels' and a
    Dice weighted by label magnitude. Labels must be binarized and resized nearest-neighbour.
    """
    cfg = yaml.safe_load((REPO / "configs/data/default.yaml").read_text())
    assert cfg["brats"]["binarize_label"] is True
    assert cfg["brats"]["label_interpolation"] == "nearest"


def test_eval_chunking_is_deterministic() -> None:
    """A random crop on the eval path makes results irreproducible (violates NFR-1)."""
    cfg = yaml.safe_load((REPO / "configs/data/default.yaml").read_text())
    assert cfg["preprocess"]["chunking"] == "deterministic"


def test_lfs_stub_detection() -> None:
    """check_data must recognize a Git LFS pointer file as NOT being real weights."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from check_data import is_lfs_stub

    stub = REPO / "legacy/UNET/3d_MonaiUNET_slices_MSE.pth"
    if stub.exists():
        assert is_lfs_stub(stub), "the inherited .pth files ARE stubs; detection must catch them"
