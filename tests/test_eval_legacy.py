"""Spec 004 acceptance test 6: the legacy bug-compat pipeline.

The deterministic mechanics (preprocessing, metric formulas, single-subject accumulation, config
wiring) need no real model or data and are tested here on synthetic seeded tensors + a
``_StubModel``. The ±0.02-of-0.6255 assertion against real weights/BraTS is the one criterion not
verifiable on a laptop (R2) — see the skip-marked test at the bottom, which reads
``artifacts/metrics/<cell_id>/legacy_compat.json`` when present.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("monai")

from torch import Tensor, nn  # noqa: E402

from mri_ad.eval.legacy import (  # noqa: E402
    LEGACY_PUBLISHED_DICE,
    LegacyCompatConfig,
    LegacyCompatEvaluator,
    _legacy_crop_starts,
    legacy_preprocess,
)
from mri_ad.models.base import AnomalyDetectionModel, ModelCard  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


class _StubModel(AnomalyDetectionModel):
    """Near-identity reconstruction — no real weights, mirrors ``tests/test_recon.py``."""

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
            input_shape=(1, 1, 8, 32, 32),
            output_shape=(1, 1, 8, 32, 32),
            training_data="none",
            training_loss="none",
            output_activation=None,
            known_characteristics="test stub",
        )


_SMALL_CFG = LegacyCompatConfig(
    subject_count=10, chunk_index=1, chunk_depth=8, threshold=0.1, crop_size=(24, 48, 48), side=32
)


def _make_brats_dir(tmp_path: Path, n_subjects: int, native_shape: tuple[int, int, int]) -> Path:
    nib = pytest.importorskip("nibabel")
    d = tmp_path / "brats"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_subjects):
        vid = f"BraTS_{i}"
        subject_dir = d / vid
        subject_dir.mkdir()
        img = rng.random(native_shape).astype(np.float32)
        seg = rng.integers(0, 5, size=native_shape).astype(np.float32)
        seg[seg == 3] = 0  # BraTS labels are {0,1,2,4}
        nib.save(nib.Nifti1Image(img, np.eye(4)), str(subject_dir / f"{vid}_t2.nii.gz"))
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(subject_dir / f"{vid}_seg.nii.gz"))
    return d


# ── legacy_preprocess: crop bounds, untouched depth, fractional labels ─────────────────────────
def test_legacy_crop_starts_match_the_documented_55_35_formula() -> None:
    assert _legacy_crop_starts((160, 130, 170), 240, 240) == (55, 35)


def test_legacy_compat_config_default_subject_count_matches_the_notebook() -> None:
    # legacy/metric-uad.ipynb cell 14: `subject_count = 100`.
    assert LegacyCompatConfig().subject_count == 100


def test_legacy_preprocess_reproduces_the_prior_crop_resize_and_chunk_selection() -> None:
    cfg = LegacyCompatConfig(crop_size=(24, 48, 48), side=32, chunk_depth=8, chunk_index=1)
    rng = np.random.default_rng(1)
    t2 = rng.random((64, 64, 40)).astype(np.float32)
    # A constant marker on native depth-slices 8:16 — post-transpose these become depth axis
    # indices 8:16 too. If depth had been cropped (the unused start_d), this marker would land
    # somewhere other than the expected chunk (index 1 -> global depth 8:16).
    t2[:, :, 8:16] = 999.0
    seg = rng.integers(0, 5, size=(64, 64, 40)).astype(np.float32)
    seg[seg == 3] = 0

    image_chunk, label_chunk = legacy_preprocess(t2, seg, cfg)

    assert image_chunk.shape == (1, 1, 8, 32, 32)
    assert label_chunk.shape == (1, 1, 8, 32, 32)
    # Depth untouched: the marker (normalized to 1.0, since it's the global max) lands exactly
    # in the chunk selected by chunk_index * chunk_depth = 8, not shifted by a depth crop.
    assert bool((image_chunk > 0.9).all())
    # Bug #2/#3: the segmentation was trilinear-resized without binarizing -> fractional labels.
    uniques = set(label_chunk.unique().tolist())
    assert not uniques <= {0.0, 1.0, 2.0, 4.0}, "seg chunk should contain fractional values"


def test_legacy_preprocess_marker_outside_the_selected_chunk_does_not_leak_in() -> None:
    cfg = LegacyCompatConfig(crop_size=(24, 48, 48), side=32, chunk_depth=8, chunk_index=1)
    rng = np.random.default_rng(2)
    t2 = rng.random((64, 64, 40)).astype(np.float32) * 0.5  # keep background well below 1.0
    t2[:, :, 24:32] = 999.0  # chunk index 3, not the selected chunk (index 1)
    seg = np.zeros((64, 64, 40), dtype=np.float32)

    image_chunk, _ = legacy_preprocess(t2, seg, cfg)
    assert bool((image_chunk < 0.9).all())


# ── LegacyMetricsComputer already covered in tests/test_eval.py; evaluator wiring here ─────────
def test_legacy_evaluator_reports_only_the_last_subject(tmp_path: Path) -> None:
    brats_dir = _make_brats_dir(tmp_path, n_subjects=3, native_shape=(64, 64, 40))
    model = _StubModel()
    model.eval()
    evaluator = LegacyCompatEvaluator(model, _SMALL_CFG, torch.device("cpu"))

    result = evaluator.evaluate_directory(brats_dir)

    assert result.n_scores_accumulated == 1  # bug #1, made visible
    assert result.n_subjects_visited == 3
    assert result.last_volume_id == "BraTS_2"  # sort_subjects=True -> deterministic "last"

    # Confirm it really is the third subject's score, not an average of all three.
    t2 = np.asarray(
        pytest.importorskip("nibabel")
        .load(str(brats_dir / "BraTS_2" / "BraTS_2_t2.nii.gz"))
        .get_fdata(),
        dtype=np.float32,
    )
    seg = np.asarray(
        pytest.importorskip("nibabel")
        .load(str(brats_dir / "BraTS_2" / "BraTS_2_seg.nii.gz"))
        .get_fdata(),
        dtype=np.float32,
    )
    expected_dice, _ = evaluator.evaluate_subject(t2, seg)
    assert result.dice == pytest.approx(expected_dice)


def test_legacy_evaluator_raises_on_empty_directory(tmp_path: Path) -> None:
    from mri_ad.exceptions import DataError

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    model = _StubModel()
    model.eval()
    evaluator = LegacyCompatEvaluator(model, _SMALL_CFG, torch.device("cpu"))
    with pytest.raises(DataError):
        evaluator.evaluate_directory(empty_dir)


def test_legacy_evaluator_raises_when_model_left_in_train_mode() -> None:
    from mri_ad.exceptions import EvalError

    model = _StubModel()  # nn.Module defaults to training=True
    evaluator = LegacyCompatEvaluator(model, _SMALL_CFG, torch.device("cpu"))
    t2 = np.zeros((64, 64, 40), dtype=np.float32)
    seg = np.zeros((64, 64, 40), dtype=np.float32)
    with pytest.raises(EvalError):
        evaluator.evaluate_subject(t2, seg)


# ── Hydra config-group wiring ────────────────────────────────────────────────────────────────────
def test_legacy_compat_config_group_flips_the_mode_and_pins_the_absolute_threshold() -> None:
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        assert cfg.eval.legacy_compat is False

        cfg_legacy = compose(config_name="config", overrides=["eval=legacy_compat"])
        assert cfg_legacy.eval.legacy_compat is True
        assert cfg_legacy.threshold.target == "mri_ad.recon.threshold.AbsoluteThreshold"
        assert cfg_legacy.threshold.params.value == pytest.approx(0.1)


# ── Makefile <-> Hydra translation, and no env-var reads ────────────────────────────────────────
def test_makefile_translates_legacy_bug_compat_into_the_hydra_override() -> None:
    out_off = subprocess.run(
        ["make", "-n", "eval"], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert "eval=legacy_compat" not in out_off.stdout

    out_on = subprocess.run(
        ["make", "-n", "eval", "LEGACY_BUG_COMPAT=1"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "eval=legacy_compat" in out_on.stdout


def test_no_python_source_reads_the_legacy_bug_compat_environment_variable() -> None:
    """The Makefile translates LEGACY_BUG_COMPAT into a Hydra override at the shell boundary —
    no Python source may read it via ``os.environ``/``os.getenv`` (mentioning the name in a
    docstring, e.g. to document ``make eval LEGACY_BUG_COMPAT=1``, is fine)."""
    pattern = re.compile(r"os\.(?:environ(?:\.get)?|getenv)\([^)]*LEGACY_BUG_COMPAT")
    hits: dict[str, list[str]] = {}
    for base in (REPO / "src", REPO / "scripts"):
        for path in base.rglob("*.py"):
            text = path.read_text()
            matches = pattern.findall(text)
            if matches:
                hits[str(path.relative_to(REPO))] = matches
    assert not hits, f"Python source reads LEGACY_BUG_COMPAT via os.environ/getenv: {hits}"


# ── The one criterion not verifiable on a laptop (R2) ───────────────────────────────────────────
def test_legacy_compat_lands_within_tolerance_of_the_published_dice() -> None:
    # Namespaced by cell_id (Spec 005 D-A) since scripts/run_eval.py::_evaluate_legacy; the
    # default config cell is unetr__mse_ssim (configs/config.yaml: model=unetr, loss=mse_ssim).
    path = REPO / "artifacts" / "metrics" / "unetr__mse_ssim" / "legacy_compat.json"
    if not path.is_file():
        pytest.skip("legacy_compat.json not present — requires the cluster run (real weights).")
    import json

    payload = json.loads(path.read_text())
    assert abs(payload["dice"] - LEGACY_PUBLISHED_DICE) <= 0.02
