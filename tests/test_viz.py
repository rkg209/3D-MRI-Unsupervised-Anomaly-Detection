"""Spec 010 visualization tests: acceptance 1, 3, 4, 5 (plus config/generator hygiene).

All runnable **without** real data, checkpoints, or a 3D forward pass — synthetic seeded tensors
at ``(1, 16, 32, 32)``, modelled on ``tests/test_eval.py``'s ``_result`` builder. The last ``pad``
depth slices are set exactly to zero so every test exercises the zero-padded-volume case.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("monai")

from mri_ad.eval.loader import load_by_volume_id  # noqa: E402
from mri_ad.exceptions import ArtifactError, VizError  # noqa: E402
from mri_ad.recon.io import save_result  # noqa: E402
from mri_ad.recon.types import ReconResult  # noqa: E402
from mri_ad.viz.colormaps import hot_rgb, normalize_slice  # noqa: E402
from mri_ad.viz.depth import DepthSpec, select_depths  # noqa: E402
from mri_ad.viz.frames import FrameRenderer, frames_digest, iter_frames  # noqa: E402
from mri_ad.viz.panels import PanelSpec, RenderSpec, build_panels  # noqa: E402
from mri_ad.viz.video import VideoSpec, write_video  # noqa: E402
from mri_ad.viz.viewer import VolumeViewer  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _result(
    volume_id: str = "vol-0",
    depth: int = 16,
    hw: int = 32,
    pad: int = 3,
    *,
    with_ground_truth: bool = True,
) -> ReconResult:
    """Synthetic ``ReconResult``; the last ``pad`` depth slices are exactly zero (padded volume)."""
    torch.manual_seed(hash(volume_id) % (2**31))
    shape = (1, depth, hw, hw)
    original = torch.rand(shape)
    reconstruction = torch.rand(shape)
    if pad > 0:
        original[:, depth - pad :] = 0.0
        reconstruction[:, depth - pad :] = 0.0
    residual = (original - reconstruction).abs()
    mask = (residual > 0.3).float()
    gt = None
    if with_ground_truth:
        gt = torch.zeros(shape)
        gt[:, depth // 2, hw // 4 : hw // 2, hw // 4 : hw // 2] = 1.0
    return ReconResult(
        volume_id=volume_id,
        original=original,
        reconstruction=reconstruction,
        residual=residual,
        anomaly_mask=mask,
        ground_truth=gt,
    )


def _render_spec(**overrides: object) -> RenderSpec:
    panels = (
        PanelSpec(key="original", title="original", cmap="gray"),
        PanelSpec(key="reconstruction", title="reconstruction", cmap="gray"),
        PanelSpec(key="residual", title="residual", cmap="hot"),
        PanelSpec(key="anomaly_mask", title="anomaly mask", cmap="gray"),
        PanelSpec(key="ground_truth", title="ground truth", cmap="gray", overlay="original"),
    )
    defaults: dict[str, object] = {
        "panels": panels,
        "figsize": (6.0, 1.28),  # 600x128 px @ dpi 100 -- both even (R2)
        "dpi": 100,
        "caption": "saved reconstruction · no live inference",
        "annotate_volume_id": False,
        "residual_scale": "per_volume",
        "overlay_color": (255, 0, 0),
        "overlay_alpha": 0.45,
        "eps": 1e-8,
    }
    defaults.update(overrides)
    return RenderSpec(**defaults)  # type: ignore[arg-type]


# ── Acceptance #1: five panels, one slider ──────────────────────────────────────────────────────
def test_build_panels_returns_five_titled_rgb_panels() -> None:
    result = _result()
    spec = _render_spec()
    panels = build_panels(
        result, None, depth=2, spec=spec, residual_vmax=float(result.residual.max())
    )
    assert len(panels) == 5
    assert [t for t, _ in panels] == [p.title for p in spec.panels]
    for _, rgb in panels:
        assert rgb.dtype == np.uint8
        assert rgb.ndim == 3
        assert rgb.shape[-1] == 3


def test_all_panels_are_driven_by_one_depth_index() -> None:
    depth, hw = 16, 32
    original = torch.zeros(1, depth, hw, hw)
    reconstruction = torch.zeros(1, depth, hw, hw)
    residual = torch.zeros(1, depth, hw, hw)
    mask = torch.zeros(1, depth, hw, hw)
    gt = torch.zeros(1, depth, hw, hw)
    for d in range(depth):
        original[0, d] = (d + 1) / depth
        reconstruction[0, d] = (depth - d) / depth
        residual[0, d] = abs(original[0, d, 0, 0].item() - reconstruction[0, d, 0, 0].item())
        mask[0, d, : (d % hw) + 1, :] = 1.0
        gt[0, d, :, : (d % hw) + 1] = 1.0
    result = ReconResult("vol", original, reconstruction, residual, mask, gt)
    spec = _render_spec()
    residual_vmax = float(residual.max())  # mirrors FrameRenderer's per_volume scaling (D-D)

    panels_a = build_panels(result, None, depth=5, spec=spec, residual_vmax=residual_vmax)
    panels_b = build_panels(result, None, depth=6, spec=spec, residual_vmax=residual_vmax)
    for (title_a, rgb_a), (_, rgb_b) in zip(panels_a, panels_b, strict=True):
        assert not np.array_equal(rgb_a, rgb_b), f"panel {title_a!r} did not change with depth"


def test_ground_truth_panel_degrades_when_ground_truth_is_none() -> None:
    result = _result(with_ground_truth=False)
    spec = _render_spec()
    panels = build_panels(
        result, None, depth=1, spec=spec, residual_vmax=float(result.residual.max())
    )
    title, rgb = panels[-1]
    assert title == "ground truth (unavailable)"
    assert np.array_equal(rgb, np.zeros_like(rgb))


def test_frame_shape_equals_figsize_times_dpi() -> None:
    result = _result()
    spec = _render_spec()
    with FrameRenderer(spec) as renderer:
        frame = renderer.render(result, None, depth=1)
    expected_w = int(spec.figsize[0] * spec.dpi)
    expected_h = int(spec.figsize[1] * spec.dpi)
    assert frame.shape == (expected_h, expected_w, 3)
    assert frame.dtype == np.uint8


# ── Acceptance #3: reproducible ─────────────────────────────────────────────────────────────────
def test_frames_are_bitwise_reproducible_across_two_exporters() -> None:
    result = _result()
    spec = _render_spec()
    depths = [1, 2, 5]
    frames_a = list(iter_frames(result, None, depths, spec))
    frames_b = list(iter_frames(result, None, depths, spec))
    assert len(frames_a) == len(frames_b) == len(depths)
    for a, b in zip(frames_a, frames_b, strict=True):
        assert np.array_equal(a, b)
    assert frames_digest(frames_a) == frames_digest(frames_b)


def test_frames_digest_is_stable_across_processes() -> None:
    code = (
        "import torch\n"
        "from mri_ad.recon.types import ReconResult\n"
        "from mri_ad.viz.frames import iter_frames, frames_digest\n"
        "from mri_ad.viz.panels import PanelSpec, RenderSpec\n"
        "torch.manual_seed(0)\n"
        "shape = (1, 16, 32, 32)\n"
        "original = torch.rand(shape)\n"
        "reconstruction = torch.rand(shape)\n"
        "residual = (original - reconstruction).abs()\n"
        "mask = (residual > 0.3).float()\n"
        "result = ReconResult('v', original, reconstruction, residual, mask, None)\n"
        "panels = (PanelSpec('original','original','gray'), "
        "PanelSpec('reconstruction','reconstruction','gray'), "
        "PanelSpec('residual','residual','hot'), "
        "PanelSpec('anomaly_mask','anomaly mask','gray'), "
        "PanelSpec('ground_truth','ground truth','gray','original'))\n"
        "spec = RenderSpec(panels=panels, figsize=(6.0,1.28), dpi=100, "
        "caption='c', annotate_volume_id=False)\n"
        "print(frames_digest(iter_frames(result, None, [1, 2, 5], spec)))\n"
    )
    outs = []
    for _ in range(2):
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
        )
        outs.append(out.stdout.strip())
    assert outs[0] == outs[1]
    assert len(outs[0]) == 64  # sha256 hex digest


def test_export_sidecar_digest_matches_recomputed_frames(tmp_path: Path) -> None:
    result = _result()
    spec = _render_spec()
    depth_spec = DepthSpec(policy="all")
    viewer = VolumeViewer(result, spec=spec, depth=depth_spec)
    out_path = tmp_path / "demo.gif"
    viewer.export_video(out_path, fps=4)

    sidecar = json.loads(Path(str(out_path) + ".frames.json").read_text())
    recomputed = frames_digest(iter_frames(result, None, viewer.depths, spec))
    assert sidecar["sha256"] == recomputed
    assert sidecar["n_frames"] == len(viewer.depths)


def test_exported_video_reads_back_with_the_expected_frame_count(tmp_path: Path) -> None:
    import imageio

    result = _result()
    spec = _render_spec()
    depth_spec = DepthSpec(policy="nonzero")
    viewer = VolumeViewer(result, spec=spec, depth=depth_spec)
    out_path = tmp_path / "demo.gif"
    viewer.export_video(out_path, fps=4)

    reader = imageio.get_reader(str(out_path))
    n_read = reader.get_length()
    if n_read in (float("inf"), None):
        n_read = sum(1 for _ in reader)
    assert n_read == len(viewer.depths)


# ── Acceptance #4: all-zero slice ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "a",
    [
        np.zeros((8, 8), dtype=np.float32),
        np.full((8, 8), 3.0, dtype=np.float32),
        np.pad(np.ones((1, 1), dtype=np.float32), ((0, 7), (0, 7))),
    ],
)
def test_normalize_all_zero_slice_returns_zeros_without_dividing_by_zero(a: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(divide="raise", invalid="raise"):
            out = normalize_slice(a)
    if np.allclose(a, a.flat[0]):
        assert np.array_equal(out, np.zeros_like(a))
    assert np.isfinite(out).all()


def test_render_spec_rejects_an_invalid_residual_scale() -> None:
    with pytest.raises(VizError):
        _render_spec(residual_scale="per-volume")  # typo: hyphen instead of underscore


def test_build_panels_refuses_per_volume_scaling_without_a_supplied_vmax() -> None:
    result = _result()
    spec = _render_spec(residual_scale="per_volume")
    with pytest.raises(VizError, match="residual_vmax"):
        build_panels(result, None, depth=1, spec=spec)  # residual_vmax defaults to None


def test_build_panels_refuses_to_render_a_nan_slice() -> None:
    result = _result()
    result.original[0, 1, 0, 0] = float("nan")
    spec = _render_spec()
    with pytest.raises(VizError, match="NaN"):
        build_panels(result, None, depth=1, spec=spec, residual_vmax=float(result.residual.max()))


def test_hot_rgb_matches_the_legacy_recipe_on_a_non_degenerate_slice() -> None:
    from matplotlib import cm

    rng = np.random.default_rng(0)
    a = rng.random((8, 8)).astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    normed = (a - lo) / (hi - lo)
    expected = (cm.hot(normed)[..., :3] * 255).astype(np.uint8)
    got = hot_rgb(a)
    assert np.array_equal(got, expected)


def test_every_depth_of_a_padded_volume_renders_finite_uint8() -> None:
    result = _result(pad=3)
    spec = _render_spec()
    depths = select_depths(result.original, result.ground_truth, DepthSpec(policy="all"))
    residual_vmax = float(result.residual.max())
    for d in depths:
        for _, rgb in build_panels(result, None, d, spec, residual_vmax=residual_vmax):
            assert rgb.dtype == np.uint8
            assert np.isfinite(rgb).all()


def test_nonzero_policy_drops_the_zero_padded_slices() -> None:
    result = _result(depth=16, pad=3)
    depths = select_depths(result.original, None, DepthSpec(policy="nonzero"))
    assert max(depths) < 16 - 3


def test_a_volume_of_only_empty_slices_still_exports_a_non_empty_video(tmp_path: Path) -> None:
    depth, hw = 16, 8
    original = torch.zeros(1, depth, hw, hw)
    reconstruction = torch.zeros(1, depth, hw, hw)
    residual = torch.zeros(1, depth, hw, hw)
    mask = torch.zeros(1, depth, hw, hw)
    result = ReconResult("empty-vol", original, reconstruction, residual, mask, None)
    spec = _render_spec()
    viewer = VolumeViewer(result, spec=spec, depth=DepthSpec(policy="nonzero"))
    assert len(viewer.depths) >= 2  # the all-empty fallback chain (D-C)
    out_path = tmp_path / "demo.gif"
    viewer.export_video(out_path, fps=4)
    assert out_path.is_file() and out_path.stat().st_size > 0


# ── Acceptance #5: no identifiers ───────────────────────────────────────────────────────────────
def test_frames_are_identical_for_two_volumes_differing_only_in_volume_id() -> None:
    torch.manual_seed(0)
    shape = (1, 16, 32, 32)
    original = torch.rand(shape)
    reconstruction = torch.rand(shape)
    residual = (original - reconstruction).abs()
    mask = (residual > 0.3).float()
    r1 = ReconResult("BraTS20_Training_001", original, reconstruction, residual, mask, None)
    r2 = ReconResult("/data/private/patient_x", original, reconstruction, residual, mask, None)
    spec = _render_spec(annotate_volume_id=False)
    with FrameRenderer(spec) as renderer:
        frame1 = renderer.render(r1, None, depth=2)
    with FrameRenderer(spec) as renderer:
        frame2 = renderer.render(r2, None, depth=2)
    assert np.array_equal(frame1, frame2)


def test_caption_contains_no_path_separator() -> None:
    spec = _render_spec()
    assert "/" not in spec.caption
    assert "\\" not in spec.caption


def test_render_signatures_accept_no_path_argument() -> None:
    for func in (build_panels, FrameRenderer.render):
        for param in inspect.signature(func).parameters.values():
            assert param.annotation is not Path, f"{func} accepts a Path argument: {param.name}"


def test_missing_volume_error_reports_a_count_not_a_listing_of_ids(tmp_path: Path) -> None:
    root = tmp_path / "results" / "run-a"
    save_result(_result("vol-a"), tmp_path / "results", "run-a")
    save_result(_result("vol-b"), tmp_path / "results", "run-a")
    with pytest.raises(ArtifactError) as excinfo:
        load_by_volume_id(root, "vol-missing")
    msg = str(excinfo.value)
    assert "2 volume" in msg
    assert "vol-a" not in msg
    assert "vol-b" not in msg


# ── extras split ─────────────────────────────────────────────────────────────────────────────────
def test_show_interactive_raises_vizerror_when_ipywidgets_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    result = _result()
    viewer = VolumeViewer(result, spec=_render_spec())
    with pytest.raises(VizError, match=r"\.\[viz\]"):
        viewer.show_interactive()


# ── ffmpeg-gated ─────────────────────────────────────────────────────────────────────────────────
def _have_ffmpeg() -> bool:
    try:
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:
        return False


_FFMPEG = _have_ffmpeg()


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not available in this environment")
def test_mp4_export_when_ffmpeg_is_available(tmp_path: Path) -> None:
    result = _result()
    spec = _render_spec()
    viewer = VolumeViewer(result, spec=spec, depth=DepthSpec(policy="all"))
    out_path = tmp_path / "demo.mp4"
    viewer.export_video(out_path, fps=4)
    assert out_path.is_file() and out_path.stat().st_size > 0


# ── config / generator hygiene ──────────────────────────────────────────────────────────────────
def test_viz_config_group_parses_and_round_trips_into_the_specs() -> None:
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        render = RenderSpec.from_cfg(cfg.viz)
        assert len(render.panels) == 5
        depth = DepthSpec(
            policy=cfg.viz.depth.policy,
            window=cfg.viz.depth.window,
            min_intensity=cfg.viz.depth.min_intensity,
            explicit=cfg.viz.depth.explicit,
            max_frames=cfg.viz.depth.max_frames,
        )
        assert depth.policy == "nonzero"
        video = VideoSpec(
            codec=cfg.viz.video.codec,
            quality=cfg.viz.video.quality,
            pixelformat=cfg.viz.video.pixelformat,
            macro_block_size=cfg.viz.video.macro_block_size,
        )
        assert video.codec == "libx264"


def test_render_spec_rejects_odd_pixel_dimensions() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "panels": [{"key": "original", "title": "original", "cmap": "gray"}],
            "dpi": 100,
            "figsize": [15.01, 3.2],
            "caption": "c",
            "annotate_volume_id": False,
            "normalize": {"residual_scale": "per_volume", "eps": 1e-8},
            "overlay": {"color": [255, 0, 0], "alpha": 0.45},
        }
    )
    with pytest.raises(VizError):
        RenderSpec.from_cfg(cfg)


def test_explicit_depth_range_overrides_the_auto_trim() -> None:
    result = _result(depth=16, pad=3)
    depths = select_depths(result.original, None, DepthSpec(policy="explicit", explicit=(2, 4)))
    assert depths == [2, 3, 4]


def test_iter_frames_is_a_generator_and_does_not_materialize_the_volume() -> None:
    result = _result()
    spec = _render_spec()
    gen = iter_frames(result, None, [1, 2, 3], spec)
    assert inspect.isgenerator(gen)


def test_write_video_raises_vizerror_for_an_unsupported_suffix(tmp_path: Path) -> None:
    with pytest.raises(VizError):
        write_video(iter([]), tmp_path / "demo.avi", 4, VideoSpec())
