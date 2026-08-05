"""Spec 005 acceptance tests: the architecture x loss matrix generator.

All runnable without real data or checkpoints — synthetic ``aggregate.json`` fixtures written
under ``tmp_path`` via the real :class:`~mri_ad.eval.report.ReportGenerator`, never a real
forward pass (house convention, see ``tests/test_eval.py``).
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("monai")

from mri_ad.eval.matrix import (  # noqa: E402
    MatrixCell,
    compute_stats,
    load_cells,
    render_csv,
    render_markdown,
    spearman,
    write_stats,
)
from mri_ad.eval.report import CONTEXT_ONLY_NOTE, ReportGenerator  # noqa: E402
from mri_ad.exceptions import ArtifactError  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _aggregate(
    metrics_root: Path,
    *,
    model: str,
    loss: str,
    dice: float,
    iou: float,
    psnr: float,
    ssim: float,
    n_volumes: int = 2,
    run_id: str = "eval-run",
    results_run_id: str = "recon-run",
) -> None:
    """Write a real ``aggregate.json`` for ``f"{model}__{loss}"`` under ``metrics_root``."""
    cell_id = f"{model}__{loss}"
    gen = ReportGenerator(metrics_root / cell_id, metrics_root.parent / "figures" / cell_id)
    gen.write_aggregate(
        {
            "dice_mean": dice,
            "dice_std": 0.01,
            "iou_mean": iou,
            "iou_std": 0.01,
            "psnr_mean": psnr,
            "psnr_std": 0.5,
            "ssim_mean": ssim,
            "ssim_std": 0.01,
            "published_dice": 0.6255,
        },
        mode="normal",
        model=model,
        loss=loss,
        split="test",
        n_volumes=n_volumes,
        split_hash="hash-test",
        run_id=run_id,
        results_run_id=results_run_id,
    )


def _matrix_cfg(cells: list[dict[str, str]]) -> dict:
    return {"cells": cells}


def _full_cfg() -> dict:
    return _matrix_cfg(
        [
            {"model": "unetr", "loss": "mse", "role": "headline", "na_reason": "no checkpoint"},
            {"model": "unetr", "loss": "ssim", "role": "headline", "na_reason": "no checkpoint"},
            {
                "model": "unetr",
                "loss": "mse_ssim",
                "role": "headline",
                "na_reason": "not evaluated",
            },
            {
                "model": "unetr",
                "loss": "multiscale_mse",
                "role": "headline",
                "na_reason": "no checkpoint",
            },
            {
                "model": "unetr",
                "loss": "perceptual",
                "role": "headline",
                "na_reason": "no training code, no checkpoint",
            },
            {"model": "unet", "loss": "mse", "role": "reference", "na_reason": "not evaluated"},
            {
                "model": "attention_unet",
                "loss": "mse_ssim",
                "role": "reference",
                "na_reason": "not evaluated",
            },
            {"model": "diffusion", "loss": "ddpm", "role": "paradigm", "na_reason": "untrained"},
        ]
    )


# ── Acceptance 1: every declared cell renders a number or an explicit na_reason ────────────────
def test_every_declared_cell_renders_a_number_or_an_explicit_na_reason(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    _aggregate(
        metrics_root, model="unetr", loss="mse_ssim", dice=0.55, iou=0.4, psnr=22.0, ssim=0.8
    )
    cells = load_cells(_full_cfg(), metrics_root)
    assert len(cells) == 8
    for cell in cells:
        if cell.available:
            assert cell.render("dice", ".4f") == f"{cell.dice:.4f}"
        else:
            rendered = cell.render("dice", ".4f")
            assert rendered.startswith("n/a (")
            assert cell.na_reason


def test_no_cell_is_ever_blank_or_dropped_from_the_table(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    cells = load_cells(_full_cfg(), metrics_root)  # nothing evaluated -> all n/a
    assert len(cells) == 8
    for cell in cells:
        for field, fmt in (("dice", ".4f"), ("iou", ".4f"), ("psnr_db", ".2f"), ("ssim", ".4f")):
            rendered = cell.render(field, fmt)
            assert rendered.strip() != ""


# ── Acceptance 2: no hardcoding, every number traces to an artifact ────────────────────────────
def test_table_values_change_when_the_source_aggregate_json_changes(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    _aggregate(
        metrics_root, model="unetr", loss="mse_ssim", dice=0.55, iou=0.4, psnr=22.0, ssim=0.8
    )
    cells_before = load_cells(_full_cfg(), metrics_root)
    dice_before = next(c.dice for c in cells_before if c.cell_id == "unetr__mse_ssim")

    agg_path = metrics_root / "unetr__mse_ssim" / "aggregate.json"
    payload = json.loads(agg_path.read_text())
    payload["headline"]["dice_mean"] = 0.91
    agg_path.write_text(json.dumps(payload))

    cells_after = load_cells(_full_cfg(), metrics_root)
    dice_after = next(c.dice for c in cells_after if c.cell_id == "unetr__mse_ssim")

    assert dice_before == pytest.approx(0.55)
    assert dice_after == pytest.approx(0.91)


def test_matrix_generator_contains_no_hardcoded_metric_literals() -> None:
    path = REPO / "src" / "mri_ad" / "eval" / "matrix.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    allowlist = {0, 1, 2, 3, 0.5}  # 3 is spearman's minimum-sample-size guard (n < 3 -> None)
    offenders: list[object] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                continue
            if node.value not in allowlist:
                offenders.append(node.value)
    assert not offenders, f"unexpected numeric literals in matrix.py: {offenders}"


def test_every_available_cell_names_its_metrics_path_and_run_ids(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    _aggregate(
        metrics_root,
        model="unetr",
        loss="mse_ssim",
        dice=0.55,
        iou=0.4,
        psnr=22.0,
        ssim=0.8,
        run_id="eval-42",
        results_run_id="recon-7",
    )
    cells = load_cells(_full_cfg(), metrics_root)
    cell = next(c for c in cells if c.cell_id == "unetr__mse_ssim")
    assert cell.metrics_path is not None
    assert Path(cell.metrics_path).is_file()
    assert cell.eval_run_id == "eval-42"
    assert cell.recon_run_id == "recon-7"


def test_load_cells_raises_artifact_error_on_a_malformed_aggregate_json(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    cell_dir = metrics_root / "unetr__mse_ssim"
    cell_dir.mkdir(parents=True)
    (cell_dir / "aggregate.json").write_text("{not valid json")
    with pytest.raises(ArtifactError):
        load_cells(_full_cfg(), metrics_root)


def test_load_cells_raises_artifact_error_when_the_payload_cell_id_does_not_match_its_dir(
    tmp_path: Path,
) -> None:
    metrics_root = tmp_path / "metrics"
    # Written under unet__mse but the payload itself says unetr__mse_ssim — a stale/misplaced
    # file that must never be silently attributed to the directory it happens to sit in.
    _aggregate(metrics_root, model="unet", loss="mse", dice=0.5, iou=0.4, psnr=20.0, ssim=0.7)
    agg_path = metrics_root / "unet__mse" / "aggregate.json"
    payload = json.loads(agg_path.read_text())
    payload["cell_id"] = "unetr__mse_ssim"
    agg_path.write_text(json.dumps(payload))
    with pytest.raises(ArtifactError, match="cell_id"):
        load_cells(_full_cfg(), metrics_root)


def test_load_cells_raises_artifact_error_on_a_val_split_aggregate(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    _aggregate(
        metrics_root, model="unetr", loss="mse_ssim", dice=0.55, iou=0.4, psnr=22.0, ssim=0.8
    )
    agg_path = metrics_root / "unetr__mse_ssim" / "aggregate.json"
    payload = json.loads(agg_path.read_text())
    payload["split"] = "val"
    agg_path.write_text(json.dumps(payload))
    with pytest.raises(ArtifactError, match="split"):
        load_cells(_full_cfg(), metrics_root)


def test_load_cells_raises_artifact_error_on_a_duplicate_declared_cell(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    dup_cfg = _matrix_cfg(
        [
            {"model": "unetr", "loss": "mse", "role": "headline", "na_reason": "no checkpoint"},
            {"model": "unetr", "loss": "mse", "role": "headline", "na_reason": "no checkpoint"},
        ]
    )
    with pytest.raises(ArtifactError, match="unetr__mse"):
        load_cells(dup_cfg, metrics_root)


# ── Acceptance 3: PSNR/SSIM are labelled context-only in both output formats ───────────────────
def test_psnr_and_ssim_columns_are_labelled_context_only_in_csv_and_markdown(
    tmp_path: Path,
) -> None:
    metrics_root = tmp_path / "metrics"
    _aggregate(
        metrics_root, model="unetr", loss="mse_ssim", dice=0.55, iou=0.4, psnr=22.0, ssim=0.8
    )
    cells = load_cells(_full_cfg(), metrics_root)
    stats = compute_stats(cells)

    csv_path = render_csv(cells, tmp_path / "tables" / "arch_loss_matrix.csv")
    header = next(csv.reader(csv_path.open()))
    assert "psnr_db" in header and "ssim" in header

    md_path = render_markdown(cells, stats, tmp_path / "tables" / "arch_loss_matrix.md")
    text = md_path.read_text()
    assert "PSNR (dB) — context only" in text
    assert "SSIM — context only" in text
    assert CONTEXT_ONLY_NOTE in text


# ── Acceptance 4: Spearman rho, hand-rolled, ties, n<3 ──────────────────────────────────────────
def test_spearman_matches_a_known_hand_computed_value() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]  # perfectly monotonic -> rho == 1.0
    assert spearman(xs, ys) == pytest.approx(1.0)

    ys_inverse = [10.0, 8.0, 6.0, 4.0, 2.0]
    assert spearman(xs, ys_inverse) == pytest.approx(-1.0)


def test_spearman_handles_ties_with_average_ranks() -> None:
    xs = [1.0, 1.0, 2.0, 3.0]
    ys = [1.0, 2.0, 2.0, 3.0]
    # ranks(xs) = [1.5, 1.5, 3, 4]; ranks(ys) = [1, 2.5, 2.5, 4]
    rho = spearman(xs, ys)
    assert rho is not None
    assert -1.0 <= rho <= 1.0
    assert rho == pytest.approx(5 / 6, abs=1e-6)


def test_spearman_is_na_when_fewer_than_three_cells_are_available() -> None:
    assert spearman([1.0, 2.0], [2.0, 4.0]) is None
    assert spearman([], []) is None


def test_spearman_is_none_for_zero_variance_input() -> None:
    assert spearman([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None


def _cell(cell_id: str, *, dice: float | None, psnr_db: float | None) -> MatrixCell:
    model, loss = cell_id.split("__", 1)
    return MatrixCell(
        model=model,
        loss=loss,
        cell_id=cell_id,
        role="headline",
        dice=dice,
        iou=dice,
        psnr_db=psnr_db,
        ssim=0.5,
        na_reason=None if dice is not None else "not evaluated",
        metrics_path="synthetic",
        eval_run_id="e",
        recon_run_id="r",
        n_volumes=2,
    )


def test_compute_stats_pairs_psnr_and_dice_positionally_even_when_a_cell_is_missing_one() -> None:
    # Four "available" cells (dice is not None), but the third has no psnr_db — a ragged cell
    # that must be dropped from the correlation, not silently misaligned with a neighbour's psnr.
    cells = [
        _cell("unetr__mse", dice=0.40, psnr_db=25.0),
        _cell("unetr__ssim", dice=0.50, psnr_db=22.0),
        _cell("unetr__mse_ssim", dice=0.60, psnr_db=None),
        _cell("unetr__multiscale_mse", dice=0.30, psnr_db=28.0),
    ]
    stats = compute_stats(cells)
    assert stats.n_available == 4
    assert stats.n_pairs == 3  # the psnr_db=None cell is excluded from both series together
    assert stats.spearman_psnr_dice is not None


def test_stats_json_records_whether_the_diffusion_row_is_available(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    cells = load_cells(_full_cfg(), metrics_root)
    stats = compute_stats(cells)
    assert stats.diffusion_available is False

    _aggregate(metrics_root, model="diffusion", loss="ddpm", dice=0.3, iou=0.2, psnr=18.0, ssim=0.6)
    cells = load_cells(_full_cfg(), metrics_root)
    stats = compute_stats(cells)
    assert stats.diffusion_available is True

    json_path = write_stats(cells, stats, tmp_path / "tables" / "arch_loss_matrix.json")
    payload = json.loads(json_path.read_text())
    assert payload["stats"]["diffusion_available"] is True
    assert payload["stats"]["n_cells"] == 8


# ── Acceptance 5: best cell only from available cells ───────────────────────────────────────────
def test_best_cell_is_selected_only_from_available_cells(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    _aggregate(metrics_root, model="unetr", loss="mse", dice=0.40, iou=0.30, psnr=25.0, ssim=0.85)
    _aggregate(
        metrics_root,
        model="unetr",
        loss="mse_ssim",
        dice=0.62,
        iou=0.48,
        psnr=20.0,
        ssim=0.75,
    )
    cells = load_cells(_full_cfg(), metrics_root)
    stats = compute_stats(cells)
    assert stats.best_cell_id == "unetr__mse_ssim"
    assert stats.best_dice == pytest.approx(0.62)


def test_best_cell_is_none_when_no_cell_is_available(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    cells = load_cells(_full_cfg(), metrics_root)
    stats = compute_stats(cells)
    assert stats.best_cell_id is None
    assert stats.best_dice is None
    assert stats.n_available == 0
