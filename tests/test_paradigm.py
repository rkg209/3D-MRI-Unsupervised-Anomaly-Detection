"""Spec 007 acceptance tests: the paradigm comparison assembly + rendering.

All synthetic, all via ``tmp_path`` and the *real* writers (``classical.report`` /
``mri_ad.eval.report``) — mirrors ``tests/test_matrix.py``'s house convention. No real recon run,
no model, no GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("monai")

from mri_ad.classical.baseline import OofPrediction  # noqa: E402
from mri_ad.classical.metrics import GRANULARITY_NOTE as CLASSICAL_GRANULARITY_NOTE  # noqa: E402
from mri_ad.classical.metrics import (  # noqa: E402
    FoldMetrics,
    aggregate_folds,
)
from mri_ad.classical.report import ClassicalReportGenerator  # noqa: E402
from mri_ad.eval.paradigm import (  # noqa: E402
    GRANULARITY_NOTE,
    assert_same_samples,
    compute_stats,
    load_classical_column,
    load_dl_column,
    plot_curves,
    render_csv,
    render_markdown,
    write_stats,
)
from mri_ad.eval.report import ReportGenerator  # noqa: E402
from mri_ad.eval.slicelevel import SliceScore, write_slice_scores  # noqa: E402
from mri_ad.exceptions import ArtifactError  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


# ── fixtures: write real classical + DL artifacts under tmp_path ──────────────────────────────
def _write_classical(
    metrics_dir: Path,
    *,
    split_hash: str = "hash-abc",
    n_slices_per_volume: int = 4,
    n_volumes: int = 2,
) -> None:
    gen = ClassicalReportGenerator(metrics_dir, metrics_dir.parent / "classical_figures")
    folds = (
        FoldMetrics(
            fold=0,
            n_train=10,
            n_test=n_slices_per_volume * n_volumes,
            n_train_subjects=8,
            n_test_subjects=n_volumes,
            n_test_positive=n_volumes,
            positive_rate=0.25,
            roc_auc=0.7,
            pr_auc=0.5,
            pr_auc_prevalence_baseline=0.25,
            roc_auc_majority_baseline=0.5,
            scale_pos_weight=3.0,
        ),
    )
    metrics = aggregate_folds(
        folds,
        n_samples=n_slices_per_volume * n_volumes,
        n_subjects=n_volumes,
        n_features=4,
        n_positive=n_volumes,
        positive_rate=0.25,
        feature_names=["f0"],
    )
    gen.write_metrics_json(
        metrics,
        run_id="classical-run",
        seed=42,
        complete=True,
        subject_limit=None,
        split_info={"source": "brats_test", "split_hash": split_hash},
        features_info={"n_features": 1},
        counts_info={"n_samples": n_slices_per_volume * n_volumes},
        classifier_info={"name": "xgboost"},
    )

    rows = []
    for v in range(n_volumes):
        for s in range(n_slices_per_volume):
            label = 1 if s == 0 else 0  # first slice of each volume is tumor-bearing
            score = 0.9 if s == 0 else 0.1
            rows.append(
                OofPrediction(
                    granularity="slice-level",
                    fold=0,
                    volume_id=f"sub-{v}",
                    slice_index=s,
                    y_true=label,
                    y_score=score,
                )
            )
    gen.write_predictions(rows)


def _write_dl_cell(
    metrics_root: Path,
    *,
    cell_id: str,
    model: str,
    loss: str,
    split_hash: str = "hash-abc",
    n_slices_per_volume: int = 4,
    n_volumes: int = 2,
    dice: float = 0.55,
) -> None:
    metrics_dir = metrics_root / cell_id
    gen = ReportGenerator(metrics_dir, metrics_root.parent / "figures" / cell_id)
    gen.write_aggregate(
        {
            "dice_mean": dice,
            "dice_std": 0.01,
            "iou_mean": dice - 0.1,
            "iou_std": 0.01,
            "psnr_mean": 20.0,
            "psnr_std": 0.5,
            "ssim_mean": 0.7,
            "ssim_std": 0.01,
            "published_dice": 0.6255,
        },
        mode="normal",
        model=model,
        loss=loss,
        split="test",
        n_volumes=n_volumes,
        split_hash=split_hash,
        run_id="eval-run",
        results_run_id="recon-run",
    )

    rows = []
    for v in range(n_volumes):
        for s in range(n_slices_per_volume):
            label = 1 if s == 0 else 0
            flagged_fraction = 0.2 if s == 0 else 0.01
            rows.append(
                SliceScore(
                    volume_id=f"sub-{v}",
                    slice_index=s,
                    flagged_voxels=int(flagged_fraction * 100),
                    n_voxels=100,
                    flagged_fraction=flagged_fraction,
                    foreground_fraction=0.5,
                    label=label,
                )
            )
    write_slice_scores(rows, metrics_dir / "slice_scores.csv")


def _paradigm_cfg() -> dict:
    return {
        "score_field": "flagged_fraction",
        "slice_threshold": None,
        "threshold_grid": [0.01, 0.05, 0.1],
        "columns": [
            {
                "key": "classical",
                "paradigm": "Classical",
                "localizes": False,
                "na_reason": "not run",
            },
            {
                "key": "unetr__mse_ssim",
                "model": "unetr",
                "loss": "mse_ssim",
                "paradigm": "UNETR",
                "localizes": True,
                "na_reason": "not evaluated",
            },
            {
                "key": "diffusion__ddpm",
                "model": "diffusion",
                "loss": "ddpm",
                "paradigm": "Diffusion (AnoDDPM)",
                "localizes": True,
                "na_reason": "untrained",
            },
        ],
    }


def _assemble(tmp_path: Path, *, with_dl: bool = True, with_diffusion: bool = False):
    classical_dir = tmp_path / "classical" / "metrics"
    metrics_root = tmp_path / "metrics"
    _write_classical(classical_dir)
    if with_dl:
        _write_dl_cell(metrics_root, cell_id="unetr__mse_ssim", model="unetr", loss="mse_ssim")
    if with_diffusion:
        _write_dl_cell(
            metrics_root, cell_id="diffusion__ddpm", model="diffusion", loss="ddpm", dice=0.3
        )

    cfg = _paradigm_cfg()
    columns = []
    samples = {}
    split_hashes = {}
    for col_cfg in cfg["columns"]:
        if col_cfg["key"] == "classical":
            column, rows = load_classical_column(cfg, classical_dir)
        else:
            column, rows = load_dl_column(cfg, col_cfg, metrics_root)
        columns.append(column)
        samples[column.key] = rows
        split_hashes[column.key] = column.split_hash
    return columns, samples, split_hashes


# ── Acceptance 1: same-sample assertion ────────────────────────────────────────────────────────
def test_mismatched_split_hashes_are_rejected(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical" / "metrics"
    metrics_root = tmp_path / "metrics"
    _write_classical(classical_dir, split_hash="hash-A")
    _write_dl_cell(
        metrics_root, cell_id="unetr__mse_ssim", model="unetr", loss="mse_ssim", split_hash="hash-B"
    )

    cfg = _paradigm_cfg()
    columns = []
    samples = {}
    split_hashes = {}
    for col_cfg in cfg["columns"][:2]:
        if col_cfg["key"] == "classical":
            column, rows = load_classical_column(cfg, classical_dir)
        else:
            column, rows = load_dl_column(cfg, col_cfg, metrics_root)
        columns.append(column)
        samples[column.key] = rows
        split_hashes[column.key] = column.split_hash

    with pytest.raises(ArtifactError, match="Split-hash mismatch"):
        assert_same_samples(samples, split_hashes)


def test_dl_and_classical_slice_keysets_must_match_exactly(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical" / "metrics"
    metrics_root = tmp_path / "metrics"
    _write_classical(classical_dir, n_volumes=2)
    _write_dl_cell(
        metrics_root, cell_id="unetr__mse_ssim", model="unetr", loss="mse_ssim", n_volumes=3
    )

    cfg = _paradigm_cfg()
    _, rows_c = load_classical_column(cfg, classical_dir)
    _, rows_d = load_dl_column(cfg, cfg["columns"][1], metrics_root)
    samples = {"classical": rows_c, "unetr__mse_ssim": rows_d}
    split_hashes = {"classical": "hash-abc", "unetr__mse_ssim": "hash-abc"}

    with pytest.raises(ArtifactError, match="Sample-set mismatch"):
        assert_same_samples(samples, split_hashes)


def test_subset_on_one_side_only_is_caught_and_named(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical" / "metrics"
    metrics_root = tmp_path / "metrics"
    _write_classical(classical_dir, n_volumes=5)  # full pool
    _write_dl_cell(
        metrics_root, cell_id="unetr__mse_ssim", model="unetr", loss="mse_ssim", n_volumes=2
    )  # data.eval_subset=2 applied, classical.subject_limit not applied

    cfg = _paradigm_cfg()
    _, rows_c = load_classical_column(cfg, classical_dir)
    _, rows_d = load_dl_column(cfg, cfg["columns"][1], metrics_root)
    samples = {"classical": rows_c, "unetr__mse_ssim": rows_d}
    split_hashes = {"classical": "hash-abc", "unetr__mse_ssim": "hash-abc"}

    with pytest.raises(ArtifactError, match="eval_subset"):
        assert_same_samples(samples, split_hashes)


def test_slice_labels_agree_with_classical_slice_labels(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical" / "metrics"
    metrics_root = tmp_path / "metrics"
    _write_classical(classical_dir)
    _write_dl_cell(metrics_root, cell_id="unetr__mse_ssim", model="unetr", loss="mse_ssim")

    cfg = _paradigm_cfg()
    _, rows_c = load_classical_column(cfg, classical_dir)
    _, rows_d = load_dl_column(cfg, cfg["columns"][1], metrics_root)
    samples = {"classical": rows_c, "unetr__mse_ssim": rows_d}
    split_hashes = {"classical": "hash-abc", "unetr__mse_ssim": "hash-abc"}
    assert_same_samples(samples, split_hashes)  # both fixtures label slice 0 as tumor -> agree


def test_label_disagreement_is_caught(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical" / "metrics"
    metrics_root = tmp_path / "metrics"
    _write_classical(classical_dir)
    _write_dl_cell(metrics_root, cell_id="unetr__mse_ssim", model="unetr", loss="mse_ssim")

    cfg = _paradigm_cfg()
    _, rows_c = load_classical_column(cfg, classical_dir)
    _, rows_d = load_dl_column(cfg, cfg["columns"][1], metrics_root)
    rows_d[("sub-0", 0)] = 0  # flip the label on the DL side only
    samples = {"classical": rows_c, "unetr__mse_ssim": rows_d}
    split_hashes = {"classical": "hash-abc", "unetr__mse_ssim": "hash-abc"}

    with pytest.raises(ArtifactError, match="Label disagreement"):
        assert_same_samples(samples, split_hashes)


# ── Acceptance 2: the headline table ────────────────────────────────────────────────────────────
def test_table_has_auc_for_every_available_column(tmp_path: Path) -> None:
    columns, samples, split_hashes = _assemble(tmp_path, with_dl=True)
    assert_same_samples(samples, split_hashes)
    stats = compute_stats(columns)
    path = render_markdown(columns, stats, tmp_path / "tables" / "paradigm_comparison.md")
    text = path.read_text()
    assert "Slice-level ROC-AUC" in text
    assert "Slice-level PR-AUC" in text


def test_classical_dice_cell_reads_no_localization(tmp_path: Path) -> None:
    columns, samples, split_hashes = _assemble(tmp_path, with_dl=True)
    classical = next(c for c in columns if c.key == "classical")
    assert classical.render("dice", ".4f") == "n/a (no localization)"
    assert classical.render("iou", ".4f") == "n/a (no localization)"


def test_absent_diffusion_column_reads_na_untrained(tmp_path: Path) -> None:
    columns, _, _ = _assemble(tmp_path, with_dl=True, with_diffusion=False)
    diffusion = next(c for c in columns if c.key == "diffusion__ddpm")
    assert not diffusion.available
    assert diffusion.render("roc_auc", ".4f") == "n/a (untrained)"


def test_malformed_aggregate_raises_rather_than_rendering_na(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    _write_dl_cell(metrics_root, cell_id="unetr__mse_ssim", model="unetr", loss="mse_ssim")
    agg_path = metrics_root / "unetr__mse_ssim" / "aggregate.json"
    payload = json.loads(agg_path.read_text())
    payload["split"] = "val"  # corrupt: a val-split result masquerading as test
    agg_path.write_text(json.dumps(payload))

    cfg = _paradigm_cfg()
    with pytest.raises(ArtifactError, match="split"):
        load_dl_column(cfg, cfg["columns"][1], metrics_root)


def test_half_present_column_is_malformed_not_na(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    _write_dl_cell(metrics_root, cell_id="unetr__mse_ssim", model="unetr", loss="mse_ssim")
    (metrics_root / "unetr__mse_ssim" / "slice_scores.csv").unlink()  # only aggregate.json remains

    cfg = _paradigm_cfg()
    with pytest.raises(ArtifactError, match="half-present"):
        load_dl_column(cfg, cfg["columns"][1], metrics_root)


# ── Acceptance 3: overlaid curves ──────────────────────────────────────────────────────────────
def test_plot_curves_writes_one_figure_with_all_available_models(tmp_path: Path) -> None:
    columns, _, _ = _assemble(tmp_path, with_dl=True)
    path = plot_curves(columns, tmp_path / "figures" / "paradigm_curves.png")
    assert path.is_file()
    assert path.stat().st_size > 0


def test_pr_panel_carries_the_prevalence_baseline(tmp_path: Path) -> None:
    columns, _, _ = _assemble(tmp_path, with_dl=True)
    classical = next(c for c in columns if c.key == "classical")
    assert classical.curves is not None
    assert classical.curves.prevalence == pytest.approx(0.25)


# ── Acceptance 4: the granularity caveat + the pooled-vs-per-fold note ─────────────────────────
def test_granularity_note_matches_classical_verbatim() -> None:
    assert GRANULARITY_NOTE == CLASSICAL_GRANULARITY_NOTE


def test_markdown_contains_the_granularity_note_verbatim(tmp_path: Path) -> None:
    columns, _, _ = _assemble(tmp_path, with_dl=True)
    stats = compute_stats(columns)
    text = render_markdown(columns, stats, tmp_path / "tables" / "x.md").read_text()
    assert GRANULARITY_NOTE in text


def test_markdown_states_pooled_oof_differs_from_per_fold_mean(tmp_path: Path) -> None:
    columns, _, _ = _assemble(tmp_path, with_dl=True)
    stats = compute_stats(columns)
    text = render_markdown(columns, stats, tmp_path / "tables" / "x.md").read_text()
    assert "per-fold mean" in text.lower()


# ── Acceptance 5: winners/losers are named mechanically ────────────────────────────────────────
def test_stats_name_the_best_column_even_when_it_is_classical(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical" / "metrics"
    metrics_root = tmp_path / "metrics"
    _write_classical(classical_dir)  # classical roc_auc from curves will be high (clean separation)
    _write_dl_cell(
        metrics_root, cell_id="unetr__mse_ssim", model="unetr", loss="mse_ssim", dice=0.1
    )

    cfg = _paradigm_cfg()
    columns = []
    for col_cfg in cfg["columns"][:2]:
        if col_cfg["key"] == "classical":
            column, _ = load_classical_column(cfg, classical_dir)
        else:
            column, _ = load_dl_column(cfg, col_cfg, metrics_root)
        columns.append(column)

    stats = compute_stats(columns)
    text = render_markdown(columns, stats, tmp_path / "tables" / "x.md").read_text()
    assert stats.best_roc_auc_key == "classical"
    assert "classical" in text.lower()


def test_diffusion_losing_to_unetr_is_rendered_not_dropped(tmp_path: Path) -> None:
    columns, _, _ = _assemble(tmp_path, with_dl=True, with_diffusion=True)
    stats = compute_stats(columns)
    diffusion = next(c for c in columns if c.key == "diffusion__ddpm")
    assert diffusion.available
    path = render_markdown(columns, stats, tmp_path / "tables" / "x.md")
    text = path.read_text()
    assert "Diffusion (AnoDDPM)" in text
    assert stats.best_dice_key != "diffusion__ddpm"  # unetr's dice (0.55) beats diffusion's (0.3)


# ── Acceptance 6: byte-identical regeneration ──────────────────────────────────────────────────
def test_paradigm_table_regenerates_byte_identically_from_saved_artifacts(tmp_path: Path) -> None:
    columns, samples, split_hashes = _assemble(tmp_path, with_dl=True)
    assert_same_samples(samples, split_hashes)
    stats = compute_stats(columns)

    md1 = render_markdown(columns, stats, tmp_path / "a.md").read_text()
    csv1 = render_csv(columns, tmp_path / "a.csv").read_text()
    json1 = write_stats(columns, stats, tmp_path / "a.json").read_text()

    columns2, samples2, split_hashes2 = _assemble(tmp_path, with_dl=True)
    assert_same_samples(samples2, split_hashes2)
    stats2 = compute_stats(columns2)
    md2 = render_markdown(columns2, stats2, tmp_path / "b.md").read_text()
    csv2 = render_csv(columns2, tmp_path / "b.csv").read_text()
    json2 = write_stats(columns2, stats2, tmp_path / "b.json").read_text()

    assert md1 == md2
    assert csv1 == csv2
    assert json1 == json2


# ── Boundary ────────────────────────────────────────────────────────────────────────────────────
def test_paradigm_module_has_no_ml_imports() -> None:
    import ast

    path = REPO / "src" / "mri_ad" / "eval" / "paradigm.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    forbidden = {"torch", "monai", "nibabel", "torchmetrics", "sklearn", "scipy", "skimage"}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    hit = {m for m in modules if any(m == f or m.startswith(f + ".") for f in forbidden)}
    assert not hit


def test_render_csv_header_has_all_expected_fields(tmp_path: Path) -> None:
    columns, _, _ = _assemble(tmp_path, with_dl=True)
    path = render_csv(columns, tmp_path / "tables" / "x.csv")
    header = next(csv.reader(path.open()))
    assert "roc_auc" in header and "dice" in header and "na_reason" in header
