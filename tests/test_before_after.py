"""Spec 009 acceptance 4 tests for ``eval/before_after.py``.

Fully testable against fabricated ``aggregate.json``/``run_meta.json`` fixtures — no GPU, no
model, no real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mri_ad.eval.before_after import (
    CellRecord,
    assert_controlled,
    load_cell,
    render_csv,
    render_markdown,
)
from mri_ad.exceptions import ArtifactError

_CONFIG = {
    "threshold": {"target": "mri_ad.recon.threshold.FixedPercentileThreshold", "params": {"p": 99}},
    "data": {"preprocess": {"eps": 1e-8, "resize_mode": "trilinear"}, "eval_subset": None},
    "shape": {"channels": 1, "depth": 16, "height": 128, "width": 128},
    "model": {"params": {"hidden_size": 768, "num_layers": 12}},
}


def _write_cell(
    metrics_root: Path,
    runs_root: Path,
    *,
    model: str,
    loss: str,
    run_id: str,
    dice_mean: float = 0.5,
    split_hash: str = "hash-1",
    n_volumes: int = 10,
    split: str = "test",
    config_overrides: dict | None = None,
    published_dice: float = 0.6255,
) -> None:
    cell_dir = metrics_root / f"{model}__{loss}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "mode": "normal",
        "model": model,
        "loss": loss,
        "cell_id": f"{model}__{loss}",
        "split": split,
        "n_volumes": n_volumes,
        "split_hash": split_hash,
        "run_id": run_id,
        "results_run_id": f"{run_id}-recon",
        "headline": {"dice_mean": dice_mean, "dice_std": 0.05, "iou_mean": 0.35, "iou_std": 0.04},
        "context_only": {
            "psnr_db_mean": 20.0,
            "psnr_db_std": 1.0,
            "ssim_mean": 0.8,
            "ssim_std": 0.05,
        },
        "baseline_delta": {
            "published_dice": published_dice,
            "corrected_dice": dice_mean,
            "delta": dice_mean - published_dice,
        },
    }
    (cell_dir / "aggregate.json").write_text(json.dumps(aggregate))

    config = json.loads(json.dumps(_CONFIG))
    if config_overrides:
        for dotted, value in config_overrides.items():
            cur = config
            parts = dotted.split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps({"config": config, "metrics": {}}))


def _fixture_pair(tmp_path: Path, **overrides) -> tuple[Path, Path]:
    metrics_root = tmp_path / "metrics"
    runs_root = tmp_path / "runs"
    _write_cell(
        metrics_root, runs_root, model="unetr", loss="mse_ssim", run_id="run-before", dice_mean=0.50
    )
    after_overrides = dict(model="unetr_synth", loss="mse_ssim", run_id="run-after", dice_mean=0.60)
    after_overrides.update(overrides)
    _write_cell(metrics_root, runs_root, **after_overrides)
    return metrics_root, runs_root


# ── load_cell ─────────────────────────────────────────────────────────────────────────────────


def test_load_cell_reads_headline_never_published_dice(tmp_path: Path) -> None:
    metrics_root, runs_root = _fixture_pair(tmp_path)
    cell = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    assert isinstance(cell, CellRecord)
    assert cell.dice_mean == 0.50
    assert cell.run_id == "run-before"


def test_before_never_sourced_from_published_dice(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    runs_root = tmp_path / "runs"
    # published_dice (0.6255, the legacy number) deliberately differs from dice_mean.
    _write_cell(
        metrics_root,
        runs_root,
        model="unetr",
        loss="mse_ssim",
        run_id="run-1",
        dice_mean=0.50,
        published_dice=0.6255,
    )
    cell = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    assert cell.dice_mean == 0.50
    assert cell.dice_mean != 0.6255


def test_load_cell_missing_aggregate_raises_artifact_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        load_cell(tmp_path / "metrics", tmp_path / "runs", model="unetr", loss="mse_ssim")


def test_load_cell_missing_run_meta_raises_artifact_error(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    cell_dir = metrics_root / "unetr__mse_ssim"
    cell_dir.mkdir(parents=True)
    (cell_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "model": "unetr",
                "loss": "mse_ssim",
                "cell_id": "unetr__mse_ssim",
                "split": "test",
                "n_volumes": 5,
                "split_hash": "h",
                "run_id": "missing-run",
                "headline": {"dice_mean": 0.5, "dice_std": 0.0, "iou_mean": 0.3, "iou_std": 0.0},
            }
        )
    )
    with pytest.raises(ArtifactError):
        load_cell(metrics_root, tmp_path / "runs", model="unetr", loss="mse_ssim")


# ── assert_controlled ────────────────────────────────────────────────────────────────────────


def test_table_from_two_cells(tmp_path: Path) -> None:
    metrics_root, runs_root = _fixture_pair(tmp_path)
    before = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    after = load_cell(metrics_root, runs_root, model="unetr_synth", loss="mse_ssim")
    assert_controlled(before, after)  # must not raise
    markdown = render_markdown(before, after)
    assert "0.5000" in markdown
    assert "0.6000" in markdown
    assert "before" in markdown.lower()


def test_rejects_split_hash_mismatch(tmp_path: Path) -> None:
    metrics_root, runs_root = _fixture_pair(tmp_path, split_hash="hash-DIFFERENT")
    before = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    after = load_cell(metrics_root, runs_root, model="unetr_synth", loss="mse_ssim")
    with pytest.raises(ArtifactError, match="split_hash"):
        assert_controlled(before, after)


def test_rejects_threshold_drift(tmp_path: Path) -> None:
    metrics_root, runs_root = _fixture_pair(tmp_path, config_overrides={"threshold.params.p": 95})
    before = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    after = load_cell(metrics_root, runs_root, model="unetr_synth", loss="mse_ssim")
    with pytest.raises(ArtifactError, match="threshold"):
        assert_controlled(before, after)


def test_rejects_preprocess_drift(tmp_path: Path) -> None:
    metrics_root, runs_root = _fixture_pair(
        tmp_path, config_overrides={"data.preprocess.resize_mode": "nearest"}
    )
    before = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    after = load_cell(metrics_root, runs_root, model="unetr_synth", loss="mse_ssim")
    with pytest.raises(ArtifactError, match="data.preprocess"):
        assert_controlled(before, after)


def test_rejects_n_volumes_mismatch(tmp_path: Path) -> None:
    metrics_root, runs_root = _fixture_pair(tmp_path, n_volumes=99)
    before = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    after = load_cell(metrics_root, runs_root, model="unetr_synth", loss="mse_ssim")
    with pytest.raises(ArtifactError, match="n_volumes"):
        assert_controlled(before, after)


def test_rejects_model_params_drift(tmp_path: Path) -> None:
    metrics_root, runs_root = _fixture_pair(
        tmp_path, config_overrides={"model.params.hidden_size": 256}
    )
    before = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    after = load_cell(metrics_root, runs_root, model="unetr_synth", loss="mse_ssim")
    with pytest.raises(ArtifactError, match="model.params"):
        assert_controlled(before, after)


# ── render_markdown / render_csv ────────────────────────────────────────────────────────────────


def test_negative_delta_renders_as_regression(tmp_path: Path) -> None:
    metrics_root, runs_root = _fixture_pair(tmp_path, dice_mean=0.30)  # after < before (0.50)
    before = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    after = load_cell(metrics_root, runs_root, model="unetr_synth", loss="mse_ssim")
    markdown = render_markdown(before, after)
    assert "-0.2000" in markdown
    assert "REGRESSION" in markdown


def test_positive_delta_does_not_render_regression(tmp_path: Path) -> None:
    metrics_root, runs_root = _fixture_pair(tmp_path)  # after (0.60) > before (0.50)
    before = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    after = load_cell(metrics_root, runs_root, model="unetr_synth", loss="mse_ssim")
    markdown = render_markdown(before, after)
    assert "REGRESSION" not in markdown


def test_render_csv_writes_two_rows(tmp_path: Path) -> None:
    metrics_root, runs_root = _fixture_pair(tmp_path)
    before = load_cell(metrics_root, runs_root, model="unetr", loss="mse_ssim")
    after = load_cell(metrics_root, runs_root, model="unetr_synth", loss="mse_ssim")
    path = render_csv(before, after, tmp_path / "out" / "synth_before_after.csv")
    rows = path.read_text().strip().splitlines()
    assert len(rows) == 3  # header + before + after
    assert rows[1].startswith("before,")
    assert rows[2].startswith("after,")
