"""Spec 010 acceptance test 2: ``make demo`` performs no forward pass and needs no GPU.

Three independent layers, mirroring ``tests/test_eval_boundary.py``'s technique: static (AST over
module-level imports), a subprocess ``sys.modules`` check, and a runtime check that patches
``torch.cuda.is_available`` to explode.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("monai")

REPO = Path(__file__).resolve().parent.parent
VIZ_DIR = REPO / "src" / "mri_ad" / "viz"
RUN_DEMO = REPO / "scripts" / "run_demo.py"
FORBIDDEN = {"mri_ad.models", "mri_ad.recon.engine", "monai", "ipywidgets", "IPython"}


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _viz_source_files() -> list[Path]:
    return sorted(VIZ_DIR.glob("*.py")) + [RUN_DEMO]


# ── static: AST over module-level imports ───────────────────────────────────────────────────────
def test_viz_modules_import_no_model_or_notebook_dependencies() -> None:
    violations: dict[str, set[str]] = {}
    for path in _viz_source_files():
        modules = _module_level_imports(path)
        hit = {m for m in modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)}
        if hit:
            violations[path.name] = hit
    assert not violations, f"viz/* imports forbidden modules at module scope: {violations}"


def test_no_viz_module_sets_a_global_matplotlib_backend() -> None:
    violations: dict[str, list[int]] = {}
    for path in _viz_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        lines = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "use"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "matplotlib"
            ):
                lines.append(node.lineno)
        if lines:
            violations[path.name] = lines
    assert not violations, f"viz/* calls matplotlib.use(): {violations}"


# ── runtime: run_demo.py in a subprocess over a synthetic results dir ──────────────────────────
def _make_results(tmp_path: Path, run_id: str = "run-a", n: int = 2) -> Path:
    from mri_ad.recon.io import save_result
    from mri_ad.recon.types import ReconResult

    shape = (1, 16, 16, 16)
    root = tmp_path / "results"
    for i in range(n):
        gt = torch.zeros(shape)
        gt.view(-1)[: gt.numel() // 2] = 1.0
        result = ReconResult(
            volume_id=f"vol-{i}",
            original=torch.rand(shape),
            reconstruction=torch.rand(shape),
            residual=torch.rand(shape),
            anomaly_mask=(torch.rand(shape) > 0.5).float(),
            ground_truth=gt,
        )
        save_result(result, root, run_id)
    return root


def test_run_demo_exports_from_a_saved_result_in_a_subprocess(tmp_path: Path) -> None:
    root = _make_results(tmp_path, run_id="run-a")
    out = subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            f"paths.artifact_root={tmp_path}",
            f"recon.results_dir={root}",
            "viz.results_run_id=run-a",
            f"viz.output_path={tmp_path / 'demo.gif'}",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    demo_path = tmp_path / "demo.gif"
    assert demo_path.is_file() and demo_path.stat().st_size > 0


def test_run_demo_fails_readably_when_no_results_exist(tmp_path: Path) -> None:
    out = subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            f"paths.artifact_root={tmp_path}",
            f"recon.results_dir={tmp_path / 'results'}",
            f"viz.output_path={tmp_path / 'demo.gif'}",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    assert "IndexError" not in out.stderr
    assert "ArtifactError" in out.stderr or "make recon" in out.stderr


def test_importing_mri_ad_viz_never_imports_the_notebook_extra() -> None:
    """``ipywidgets``/``IPython`` are the one leak this module controls directly.

    ``monai``/``mri_ad.models``/``mri_ad.recon.engine`` are *not* asserted absent here: importing
    ``mri_ad.viz.exporter`` pulls in ``mri_ad.eval.loader`` -> ``mri_ad.recon.io``, and Python
    always runs ``mri_ad.recon``'s ``__init__.py`` (which eagerly imports ``recon.engine`` ->
    ``models`` -> ``monai``) before a submodule import completes — a pre-existing fact of this
    codebase (confirmed present on ``main`` before Spec 010), not something this spec introduces.
    Acceptance test 2's actual guarantee — no model is *constructed* or *called* — is covered by
    the runtime guards below, matching ``tests/test_eval_boundary.py``'s established technique.
    """
    code = (
        "import sys\n"
        "import mri_ad.viz\n"
        "leaked = ['ipywidgets', 'IPython']\n"
        "hit = [m for m in leaked if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(hit))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "LEAKED:", out.stdout


def test_demo_export_completes_even_if_every_model_call_would_explode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Layer 1 of acceptance 2: the nn.Module dispatcher itself, not just ``forward()``."""
    import torch.nn as nn

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("a model was called during a demo export")

    monkeypatch.setattr(nn.Module, "__call__", _boom)
    monkeypatch.setattr(nn.Module, "_call_impl", _boom)

    root = _make_results(tmp_path, run_id="run-a")
    from mri_ad.eval.loader import resolve_results_dir
    from mri_ad.viz.depth import DepthSpec
    from mri_ad.viz.exporter import DemoExporter
    from mri_ad.viz.panels import PanelSpec, RenderSpec
    from mri_ad.viz.video import VideoSpec

    results_dir = resolve_results_dir(root, run_id="run-a")
    panels = (
        PanelSpec("original", "original", "gray"),
        PanelSpec("reconstruction", "reconstruction", "gray"),
        PanelSpec("residual", "residual", "hot"),
        PanelSpec("anomaly_mask", "anomaly mask", "gray"),
        PanelSpec("ground_truth", "ground truth", "gray", "original"),
    )
    render = RenderSpec(panels=panels, figsize=(6.0, 1.28), dpi=100, caption="c")
    exporter = DemoExporter(render=render, depth=DepthSpec(policy="all"), video=VideoSpec())
    export = exporter.export(results_dir, None, tmp_path / "demo.gif")
    assert export.n_frames > 0


def test_demo_export_never_calls_build_default_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Layer 2 of acceptance 2: not even a model object is constructed."""
    import mri_ad.models as models_module

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("build_default_registry was called during a demo export")

    monkeypatch.setattr(models_module, "build_default_registry", _boom)

    root = _make_results(tmp_path, run_id="run-a")
    from mri_ad.eval.loader import resolve_results_dir
    from mri_ad.viz.depth import DepthSpec
    from mri_ad.viz.exporter import DemoExporter
    from mri_ad.viz.panels import PanelSpec, RenderSpec
    from mri_ad.viz.video import VideoSpec

    results_dir = resolve_results_dir(root, run_id="run-a")
    panels = (
        PanelSpec("original", "original", "gray"),
        PanelSpec("reconstruction", "reconstruction", "gray"),
        PanelSpec("residual", "residual", "hot"),
        PanelSpec("anomaly_mask", "anomaly mask", "gray"),
        PanelSpec("ground_truth", "ground truth", "gray", "original"),
    )
    render = RenderSpec(panels=panels, figsize=(6.0, 1.28), dpi=100, caption="c")
    exporter = DemoExporter(render=render, depth=DepthSpec(policy="all"), video=VideoSpec())
    export = exporter.export(results_dir, None, tmp_path / "demo.gif")
    assert export.n_frames > 0


def test_export_succeeds_when_cuda_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: object, **kwargs: object) -> bool:
        raise AssertionError("torch.cuda.is_available was called during a demo export")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)

    from mri_ad.recon.types import ReconResult
    from mri_ad.viz.depth import DepthSpec
    from mri_ad.viz.viewer import VolumeViewer

    shape = (1, 16, 16, 16)
    gt = torch.zeros(shape)
    gt.view(-1)[: gt.numel() // 2] = 1.0
    result = ReconResult(
        volume_id="vol-0",
        original=torch.rand(shape),
        reconstruction=torch.rand(shape),
        residual=torch.rand(shape),
        anomaly_mask=(torch.rand(shape) > 0.5).float(),
        ground_truth=gt,
    )
    viewer = VolumeViewer(result, depth=DepthSpec(policy="all"))
    out_path = tmp_path / "demo.gif"
    viewer.export_video(out_path, fps=4)
    assert out_path.is_file() and out_path.stat().st_size > 0
