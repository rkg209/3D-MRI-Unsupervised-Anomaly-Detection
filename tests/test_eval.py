"""Spec 004 evaluation-harness tests.

All runnable **without** real data or checkpoints — synthetic seeded tensors and saved
``ReconResult`` fixtures, never a real forward pass (house convention, see ``tests/test_recon.py``
and ``tests/test_slice.py``).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

pytest.importorskip("monai")

from omegaconf import OmegaConf  # noqa: E402

from mri_ad.eval.evaluator import AggregateEvaluator, VolumeEvaluator  # noqa: E402
from mri_ad.eval.loader import iter_results, resolve_results_dir, result_paths  # noqa: E402
from mri_ad.eval.metrics import (  # noqa: E402
    AggregateMetrics,
    LegacyMetricsComputer,
    MetricsComputer,
    VolumeMetrics,
    aggregate,
)
from mri_ad.eval.report import CONTEXT_ONLY_NOTE, PER_VOLUME_FIELDS, ReportGenerator  # noqa: E402
from mri_ad.exceptions import ArtifactError, EvalError  # noqa: E402
from mri_ad.recon.io import save_result  # noqa: E402
from mri_ad.recon.types import ReconResult  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _result(
    volume_id: str,
    shape: tuple[int, ...] = (1, 16, 32, 32),
    *,
    ground_truth: torch.Tensor | None = None,
    dice_like: float | None = None,
) -> ReconResult:
    """Build a synthetic ``ReconResult`` whose Dice is controllable via ``dice_like``.

    ``dice_like=1.0`` -> mask == ground truth exactly. ``dice_like=0.0`` -> mask is the exact
    complement of a nonempty ground truth (Dice 0). ``dice_like is None`` -> random.
    """
    if ground_truth is None:
        ground_truth = torch.zeros(shape)
        ground_truth.view(-1)[: ground_truth.numel() // 2] = 1.0

    if dice_like == 1.0:
        mask = ground_truth.clone()
    elif dice_like == 0.0:
        mask = 1.0 - ground_truth
    else:
        torch.manual_seed(hash(volume_id) % (2**31))
        mask = (torch.rand(shape) > 0.5).float()

    original = torch.rand(shape)
    reconstruction = torch.rand(shape)
    residual = (original - reconstruction).abs()
    return ReconResult(
        volume_id=volume_id,
        original=original,
        reconstruction=reconstruction,
        residual=residual,
        anomaly_mask=mask,
        ground_truth=ground_truth,
    )


# ── Acceptance #2: closed-form Dice/IoU, empty-mask conventions, monotonic consistency ─────────
@pytest.mark.parametrize(
    ("pred", "gt", "expected_dice", "expected_iou"),
    [
        # 3 of 4 voxels overlap: pred={1,1,1,0}, gt={1,1,0,1} -> inter=2, dice=2*2/(3+3)=2/3,
        # iou=2/(3+3-2)=0.5
        ([1, 1, 1, 0], [1, 1, 0, 1], 2 / 3, 0.5),
        # perfect overlap
        ([1, 1, 0, 0], [1, 1, 0, 0], 1.0, 1.0),
        # no overlap, both nonempty
        ([1, 1, 0, 0], [0, 0, 1, 1], 0.0, 0.0),
    ],
)
def test_dice_and_iou_match_closed_form_values_on_hand_built_tensors(
    pred: list[float], gt: list[float], expected_dice: float, expected_iou: float
) -> None:
    p = torch.tensor(pred, dtype=torch.float32)
    g = torch.tensor(gt, dtype=torch.float32)
    assert MetricsComputer.dice(p, g) == pytest.approx(expected_dice, abs=1e-4)
    assert MetricsComputer.iou(p, g) == pytest.approx(expected_iou, abs=1e-4)


def test_dice_and_iou_empty_mask_conventions() -> None:
    z = torch.zeros(4)
    o = torch.ones(4)
    assert MetricsComputer.dice(z, z) == 1.0
    assert MetricsComputer.iou(z, z) == 1.0
    assert MetricsComputer.dice(o, z) == 0.0
    assert MetricsComputer.iou(o, z) == 0.0


def test_iou_and_dice_are_monotonically_consistent() -> None:
    torch.manual_seed(0)
    for _ in range(20):
        pred = (torch.rand(64) > 0.5).float()
        gt = (torch.rand(64) > 0.5).float()
        assert MetricsComputer.iou(pred, gt) <= MetricsComputer.dice(pred, gt) + 1e-6


# ── PSNR / SSIM units (context-only metrics) ────────────────────────────────────────────────────
def test_psnr_is_finite_and_capped_for_identical_volumes() -> None:
    from mri_ad.eval.metrics import PSNR_CEILING_DB

    vol = torch.rand(1, 16, 32, 32)
    value = MetricsComputer.psnr(vol, vol)
    assert value == pytest.approx(PSNR_CEILING_DB, abs=1e-6)
    assert value < float("inf")


def test_psnr_decreases_monotonically_with_noise() -> None:
    torch.manual_seed(0)
    orig = torch.rand(1, 16, 32, 32)
    p_low = MetricsComputer.psnr(orig + torch.randn_like(orig) * 0.01, orig)
    p_high = MetricsComputer.psnr(orig + torch.randn_like(orig) * 0.5, orig)
    assert p_low > p_high


def test_ssim_is_one_for_identical_volumes() -> None:
    torch.manual_seed(0)
    vol = torch.rand(1, 16, 32, 32)
    assert MetricsComputer.ssim(vol, vol) == pytest.approx(1.0, abs=1e-3)


def test_ssim_raises_eval_error_when_a_spatial_dim_is_below_the_window_size() -> None:
    small = torch.rand(1, 8, 32, 32)  # depth 8 < win_size 11
    with pytest.raises(EvalError, match="11"):
        MetricsComputer.ssim(small, small)


# ── aggregate() ──────────────────────────────────────────────────────────────────────────────────
def test_aggregate_computes_mean_and_population_std() -> None:
    volumes = [
        VolumeMetrics(volume_id="a", dice=1.0, iou=1.0, psnr=10.0, ssim=0.5),
        VolumeMetrics(volume_id="b", dice=0.0, iou=0.0, psnr=20.0, ssim=0.7),
        VolumeMetrics(volume_id="c", dice=1.0, iou=1.0, psnr=30.0, ssim=0.9),
    ]
    agg = aggregate(volumes)
    assert isinstance(agg, AggregateMetrics)
    assert agg.n_volumes == 3
    assert agg.dice_mean == pytest.approx(2 / 3)
    assert agg.psnr_mean == pytest.approx(20.0)


def test_aggregate_raises_on_empty_sequence() -> None:
    with pytest.raises(EvalError):
        aggregate([])


# ── Legacy metrics: differ from the corrected ones on purpose ──────────────────────────────────
def test_legacy_dice_has_no_eps_and_does_not_binarize_the_ground_truth() -> None:
    pred = torch.zeros(4)
    pred[0] = 1.0
    gt_raw = torch.zeros(4)
    gt_raw[0] = 4.0  # tumor-core label, unbinarized
    gt_raw[1] = 1.0  # edema

    legacy = LegacyMetricsComputer.dice_bugcompat(pred, gt_raw)
    corrected = MetricsComputer.dice(pred, (gt_raw > 0).float())
    assert legacy != pytest.approx(corrected)


def test_legacy_iou_binarizes_neither_input_and_adds_1e_6() -> None:
    pred = torch.tensor([0.0, 0.0])  # both empty after "binarization" that never happens
    gt_raw = torch.tensor([0.0, 0.0])
    # both zero -> denominator is the fixed 1e-6, not the corrected 1.0-by-convention path
    assert LegacyMetricsComputer.iou_bugcompat(pred, gt_raw) == 0.0


def test_legacy_dice_zero_denominator_is_nan_not_one() -> None:
    z = torch.zeros(4)
    assert LegacyMetricsComputer.dice_bugcompat(z, z) != LegacyMetricsComputer.dice_bugcompat(z, z)


# ── Acceptance #1 (part): ground truth must already be binarized before Dice ───────────────────
def test_ground_truth_is_binarized_before_dice_and_raw_labels_are_rejected() -> None:
    shape = (1, 16, 32, 32)
    pred = torch.zeros(shape)
    gt_raw = torch.zeros(shape)
    gt_raw.view(-1)[:10] = 4.0  # unbinarized BraTS label {0,1,2,4}
    result = ReconResult(
        volume_id="raw-labels",
        original=torch.zeros(shape),
        reconstruction=torch.zeros(shape),
        residual=torch.zeros(shape),
        anomaly_mask=pred,
        ground_truth=gt_raw,
    )
    with pytest.raises(EvalError):
        VolumeEvaluator().evaluate(result)


def test_labels_are_resized_nearest_neighbour_and_stay_binary() -> None:
    from mri_ad.data.transforms import build_transforms

    cfg = OmegaConf.create(
        {
            "shape": {"height": 128, "width": 128},
            "data": {"preprocess": {"eps": 1.0e-8, "resize_mode": "trilinear"}},
        }
    )
    torch.manual_seed(0)
    seg = torch.randint(0, 5, (32, 240, 240)).float()
    seg[seg == 3] = 0  # BraTS labels are {0,1,2,4}
    image = torch.rand(32, 240, 240)

    transforms = build_transforms(cfg, dataset="brats")
    out = transforms({"image": image, "label": seg})
    uniques = set(out["label"].unique().tolist())
    assert uniques <= {0.0, 1.0}  # trilinear would produce fractional values


def test_every_depth_chunk_is_scored_not_just_chunk_four() -> None:
    # Tumor planted only in the region corresponding to chunk 7 of 10 (depth 112:128 of 160).
    shape = (1, 160, 32, 32)
    gt = torch.zeros(shape)
    gt[:, 112:128, :, :] = 1.0
    mask = gt.clone()  # perfect prediction, but only visible if all chunks are scored
    result = ReconResult(
        volume_id="chunk7",
        original=torch.zeros(shape),
        reconstruction=torch.zeros(shape),
        residual=torch.zeros(shape),
        anomaly_mask=mask,
        ground_truth=gt,
    )
    metrics = VolumeEvaluator().evaluate(result)
    assert metrics.dice > 0.99


def test_all_subjects_are_accumulated_not_just_the_last(tmp_path: Path) -> None:
    shape = (1, 16, 16, 16)
    results = [
        _result("vol-0", shape, dice_like=1.0),
        _result("vol-1", shape, dice_like=0.0),
        _result("vol-2", shape, dice_like=1.0),
    ]
    root = tmp_path / "results"
    for r in results:
        save_result(r, root, "run-a")

    agg = AggregateEvaluator().evaluate_split(root / "run-a")
    assert agg.n_volumes == 3
    assert agg.dice_mean == pytest.approx(2 / 3, abs=1e-3)


# ── Acceptance #3: evaluate_split refuses anything but a Path ──────────────────────────────────
class _NotAPath:
    """Stands in for a model instance without importing torch.nn."""


def test_evaluate_split_raises_type_error_when_handed_a_model_instead_of_a_path() -> None:
    for bad in (_NotAPath(), object()):
        with pytest.raises(TypeError, match="never re-runs forward"):
            AggregateEvaluator().evaluate_split(bad)  # type: ignore[arg-type]


# ── loader.py units ──────────────────────────────────────────────────────────────────────────────
def test_resolve_results_dir_picks_the_most_recently_modified_run(tmp_path: Path) -> None:
    import time

    root = tmp_path / "results"
    older = root / "run-old"
    newer = root / "run-new"
    older.mkdir(parents=True)
    (older / "vol.pt").write_bytes(b"x")
    time.sleep(0.01)
    newer.mkdir(parents=True)
    (newer / "vol.pt").write_bytes(b"x")

    assert resolve_results_dir(root) == newer


def test_resolve_results_dir_honours_explicit_run_id(tmp_path: Path) -> None:
    root = tmp_path / "results"
    (root / "run-a").mkdir(parents=True)
    (root / "run-a" / "vol.pt").write_bytes(b"x")
    assert resolve_results_dir(root, run_id="run-a") == root / "run-a"


def test_resolve_results_dir_raises_artifact_error_when_root_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        resolve_results_dir(tmp_path / "nope")


def test_resolve_results_dir_raises_artifact_error_when_no_run_has_results(tmp_path: Path) -> None:
    root = tmp_path / "results"
    (root / "run-empty").mkdir(parents=True)
    with pytest.raises(ArtifactError):
        resolve_results_dir(root)


def test_read_manifest_returns_none_when_absent_and_the_dict_when_present(tmp_path: Path) -> None:
    from mri_ad.eval.loader import read_manifest

    root = tmp_path / "run-a"
    root.mkdir()
    assert read_manifest(root) is None

    (root / "manifest.json").write_text('{"model": "unetr", "split": "test", "n_volumes": 3}')
    manifest = read_manifest(root)
    assert manifest == {"model": "unetr", "split": "test", "n_volumes": 3}


def test_iter_results_streams_and_result_paths_lists_every_pt_file(tmp_path: Path) -> None:
    shape = (1, 16, 16, 16)
    root = tmp_path / "results"
    for i in range(3):
        save_result(_result(f"vol-{i}", shape), root, "run-a")

    paths = result_paths(root / "run-a")
    assert len(paths) == 3

    loaded_ids = {r.volume_id for r in iter_results(root / "run-a")}
    assert loaded_ids == {"vol-0", "vol-1", "vol-2"}


def test_evaluate_split_raises_eval_error_when_a_result_has_no_ground_truth(tmp_path: Path) -> None:
    shape = (1, 16, 16, 16)
    result = ReconResult(
        volume_id="no-gt",
        original=torch.zeros(shape),
        reconstruction=torch.zeros(shape),
        residual=torch.zeros(shape),
        anomaly_mask=torch.zeros(shape),
        ground_truth=None,
    )
    root = tmp_path / "results"
    save_result(result, root, "run-a")
    with pytest.raises(EvalError):
        AggregateEvaluator().evaluate_split(root / "run-a")


def test_labels_dir_override_takes_precedence_over_embedded_ground_truth(tmp_path: Path) -> None:
    shape = (1, 16, 16, 16)
    embedded_gt = torch.zeros(shape)
    result = ReconResult(
        volume_id="vol-override",
        original=torch.zeros(shape),
        reconstruction=torch.zeros(shape),
        residual=torch.zeros(shape),
        anomaly_mask=torch.ones(shape),
        ground_truth=embedded_gt,
    )
    root = tmp_path / "results"
    save_result(result, root, "run-a")

    override_gt = torch.ones(shape)  # matches the all-ones prediction -> Dice 1.0
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    torch.save(override_gt, labels_dir / "vol-override.pt")

    agg = AggregateEvaluator().evaluate_split(root / "run-a", labels_dir=labels_dir)
    assert agg.dice_mean == pytest.approx(1.0, abs=1e-4)


# ── Acceptance #5: per-volume CSV + aggregate JSON, mean/std, context-only labelling ───────────
def _volume_rows() -> list[dict[str, object]]:
    return [
        {
            "volume_id": "vol-0",
            "dice": 1.0,
            "iou": 1.0,
            "psnr_db_context_only": 10.0,
            "ssim_context_only": 0.5,
        },
        {
            "volume_id": "vol-1",
            "dice": 0.0,
            "iou": 0.0,
            "psnr_db_context_only": 20.0,
            "ssim_context_only": 0.7,
        },
    ]


def _aggregate_payload() -> dict[str, object]:
    return {
        "dice_mean": 0.5,
        "dice_std": 0.5,
        "iou_mean": 0.5,
        "iou_std": 0.5,
        "psnr_mean": 15.0,
        "psnr_std": 5.0,
        "ssim_mean": 0.6,
        "ssim_std": 0.1,
        "published_dice": 0.6255,
    }


def test_per_volume_csv_and_aggregate_json_are_written_with_mean_and_std(tmp_path: Path) -> None:
    gen = ReportGenerator(tmp_path / "metrics", tmp_path / "figures")
    csv_path = gen.write_per_volume(_volume_rows())
    json_path = gen.write_aggregate(
        _aggregate_payload(),
        mode="normal",
        model="unetr",
        loss="mse_ssim",
        split="test",
        n_volumes=2,
        split_hash="hash-test",
    )

    assert csv_path.is_file() and json_path.is_file()
    rows = gen.read_per_volume()
    assert len(rows) == 2
    assert list(rows[0].keys()) == list(PER_VOLUME_FIELDS)

    agg = gen.read_aggregate()
    assert agg["headline"]["dice_mean"] == pytest.approx(0.5)
    assert agg["n_volumes"] == 2


def test_psnr_and_ssim_are_labelled_context_only_everywhere_they_appear(tmp_path: Path) -> None:
    gen = ReportGenerator(tmp_path / "metrics", tmp_path / "figures")
    gen.write_per_volume(_volume_rows())
    gen.write_aggregate(
        _aggregate_payload(),
        mode="normal",
        model="unetr",
        loss="mse_ssim",
        split="test",
        n_volumes=2,
        split_hash="hash-test",
    )

    header = list(csv.DictReader((tmp_path / "metrics" / "per_volume.csv").open()).fieldnames)
    assert "psnr_db_context_only" in header
    assert "ssim_context_only" in header

    agg = gen.read_aggregate()
    assert "psnr_db_mean" in agg["context_only"]
    assert "ssim_mean" in agg["context_only"]
    assert "psnr_db_mean" not in agg["headline"]
    assert "ssim_mean" not in agg["headline"]

    summary_path = gen.write_summary_markdown()
    assert CONTEXT_ONLY_NOTE in summary_path.read_text()


def test_aggregate_json_round_trips_through_the_report_reader(tmp_path: Path) -> None:
    gen = ReportGenerator(tmp_path / "metrics", tmp_path / "figures")
    gen.write_aggregate(
        _aggregate_payload(),
        mode="normal",
        model="unetr",
        loss="mse_ssim",
        split="test",
        n_volumes=2,
        split_hash="hash-test",
        run_id="run-a",
    )
    agg = gen.read_aggregate()
    assert agg["run_id"] == "run-a"
    assert agg["mode"] == "normal"


def test_write_aggregate_raises_artifact_error_on_missing_key(tmp_path: Path) -> None:
    gen = ReportGenerator(tmp_path / "metrics", tmp_path / "figures")
    incomplete = {k: v for k, v in _aggregate_payload().items() if k != "published_dice"}
    with pytest.raises(ArtifactError):
        gen.write_aggregate(
            incomplete,
            mode="normal",
            model="unetr",
            loss="mse_ssim",
            split="test",
            n_volumes=2,
            split_hash="hash-test",
        )


# ── Acceptance #7: aggregate.json records the delta from the published number ──────────────────
def test_aggregate_json_records_the_delta_from_the_published_number(tmp_path: Path) -> None:
    gen = ReportGenerator(tmp_path / "metrics", tmp_path / "figures")
    gen.write_aggregate(
        _aggregate_payload(),
        mode="normal",
        model="unetr",
        loss="mse_ssim",
        split="test",
        n_volumes=2,
        split_hash="hash-test",
    )
    agg = gen.read_aggregate()
    delta = agg["baseline_delta"]
    assert delta["published_dice"] == pytest.approx(0.6255)
    assert delta["corrected_dice"] == pytest.approx(0.5)
    assert delta["delta"] == pytest.approx(0.5 - 0.6255)

    summary = gen.write_summary_markdown().read_text()
    assert "0.6255" in summary


def test_plot_dice_distribution_writes_a_png(tmp_path: Path) -> None:
    gen = ReportGenerator(tmp_path / "metrics", tmp_path / "figures")
    path = gen.plot_dice_distribution(_volume_rows())
    assert path.is_file()
    assert path.suffix == ".png"
