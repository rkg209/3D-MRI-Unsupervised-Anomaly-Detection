"""Spec 004 acceptance tests 4 and 8: the eval/report boundary.

Acceptance 4 — normal-mode ``make eval`` never calls ``forward()``, never constructs a model, and
never even imports ``mri_ad.models``/``mri_ad.recon.engine`` at module scope. Three independent
layers, mirroring ``tests/test_model_boundary.py``'s technique.

Acceptance 8 — ``make report`` has zero ML imports and finishes well under its 2-minute budget.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("monai")

from omegaconf import OmegaConf  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FORBIDDEN = {"mri_ad.models", "mri_ad.recon.engine"}
FORBIDDEN_ML_MODULES = {"torch", "monai", "nibabel", "torchmetrics", "sklearn", "scipy", "skimage"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _module_level_imports(path: Path) -> set[str]:
    """Only import statements at the *top level* of the module (not inside a function body)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _cfg(tmp_path: Path, results_dir: Path, run_id: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 42,
            "deterministic": True,
            "device": "cpu",
            "model": {"name": "stub"},
            "loss": {"name": "mse"},
            "recon": {"results_dir": str(results_dir)},
            "eval": {
                "legacy_compat": False,
                "results_run_id": run_id,
                "metrics_dir": str(tmp_path / "metrics"),
                "figures_dir": str(tmp_path / "figures"),
                "labels_dir": None,
                "legacy": {"published_dice": 0.6255, "published_iou": 0.4551},
            },
        }
    )


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
            anomaly_mask=gt.clone(),
            ground_truth=gt,
        )
        save_result(result, root, run_id)
    return root


def _write_manifest(
    results_dir: Path,
    run_id: str,
    *,
    model: str,
    split: str,
    loss: str | None = "mse",
    split_hash: str | None = "hash-test",
) -> None:
    import json

    from mri_ad.eval.loader import MANIFEST_FILENAME

    payload = {"model": model, "split": split, "n_volumes": 2}
    if loss is not None:
        payload["loss"] = loss
    if split_hash is not None:
        payload["split_hash"] = split_hash
    (results_dir / run_id / MANIFEST_FILENAME).write_text(json.dumps(payload, indent=2))


# ── manifest provenance guard: refuse a val-split or wrong-model results directory ─────────────
def test_evaluate_normal_refuses_a_val_split_manifest(tmp_path: Path) -> None:
    from scripts.run_eval import _evaluate_normal

    from mri_ad.exceptions import ArtifactError

    root = _make_results(tmp_path, run_id="run-a")
    _write_manifest(root, "run-a", model="stub", split="val")
    cfg = _cfg(tmp_path, root, "run-a")
    with pytest.raises(ArtifactError, match="val"):
        _evaluate_normal(cfg, run_id="test")


def test_evaluate_normal_refuses_a_wrong_model_manifest(tmp_path: Path) -> None:
    from scripts.run_eval import _evaluate_normal

    from mri_ad.exceptions import ArtifactError

    root = _make_results(tmp_path, run_id="run-a")
    _write_manifest(root, "run-a", model="unet", split="test")
    cfg = _cfg(tmp_path, root, "run-a")  # cfg.model.name == "stub"
    with pytest.raises(ArtifactError, match="unet"):
        _evaluate_normal(cfg, run_id="test")


def test_evaluate_normal_refuses_a_manifest_with_no_loss(tmp_path: Path) -> None:
    from scripts.run_eval import _evaluate_normal

    from mri_ad.exceptions import ArtifactError

    root = _make_results(tmp_path, run_id="run-a")
    _write_manifest(root, "run-a", model="stub", split="test", loss=None)  # pre-Spec-005 shape
    cfg = _cfg(tmp_path, root, "run-a")
    with pytest.raises(ArtifactError, match="loss"):
        _evaluate_normal(cfg, run_id="test")


def test_evaluate_normal_refuses_a_wrong_loss_manifest(tmp_path: Path) -> None:
    from scripts.run_eval import _evaluate_normal

    from mri_ad.exceptions import ArtifactError

    root = _make_results(tmp_path, run_id="run-a")
    _write_manifest(root, "run-a", model="stub", split="test", loss="ssim")
    cfg = _cfg(tmp_path, root, "run-a")  # cfg.loss.name == "mse"
    with pytest.raises(ArtifactError, match="ssim"):
        _evaluate_normal(cfg, run_id="test")


def test_evaluate_normal_accepts_a_matching_manifest(tmp_path: Path) -> None:
    from scripts.run_eval import _evaluate_normal

    root = _make_results(tmp_path, run_id="run-a")
    _write_manifest(root, "run-a", model="stub", split="test")
    cfg = _cfg(tmp_path, root, "run-a")
    summary = _evaluate_normal(cfg, run_id="test")
    assert summary["n_volumes"] == 2


def test_evaluate_normal_refuses_a_manifest_with_no_split_hash(tmp_path: Path) -> None:
    from scripts.run_eval import _evaluate_normal

    from mri_ad.exceptions import ArtifactError

    root = _make_results(tmp_path, run_id="run-a")
    _write_manifest(root, "run-a", model="stub", split="test", split_hash=None)  # pre-007 shape
    cfg = _cfg(tmp_path, root, "run-a")
    with pytest.raises(ArtifactError, match="split_hash"):
        _evaluate_normal(cfg, run_id="test")


def test_evaluate_normal_refuses_a_results_dir_with_no_manifest_at_all(tmp_path: Path) -> None:
    from scripts.run_eval import _evaluate_normal

    from mri_ad.exceptions import ArtifactError

    _make_results(tmp_path, run_id="run-a")  # no manifest written
    cfg = _cfg(tmp_path, tmp_path / "results", "run-a")
    with pytest.raises(ArtifactError, match="manifest"):
        _evaluate_normal(cfg, run_id="test")


# ── Acceptance #4, layer 1: runtime — patch the nn.Module dispatcher, not forward() ────────────
def test_evaluate_normal_completes_even_if_every_model_call_would_explode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch.nn as nn
    from scripts.run_eval import _evaluate_normal

    def _boom(*args, **kwargs):
        raise AssertionError("a model was called during normal-mode eval")

    monkeypatch.setattr(nn.Module, "__call__", _boom)
    monkeypatch.setattr(nn.Module, "_call_impl", _boom)

    root = _make_results(tmp_path, run_id="run-a")
    _write_manifest(root, "run-a", model="stub", split="test")
    cfg = _cfg(tmp_path, tmp_path / "results", "run-a")
    summary = _evaluate_normal(cfg, run_id="test")
    assert summary["n_volumes"] == 2


# ── Acceptance #4, layer 2: construction — not even a model object is built ────────────────────
def test_evaluate_normal_never_calls_build_default_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mri_ad.models as models_module

    def _boom(*args, **kwargs):
        raise AssertionError("build_default_registry was called during normal-mode eval")

    monkeypatch.setattr(models_module, "build_default_registry", _boom)

    from scripts.run_eval import _evaluate_normal

    root = _make_results(tmp_path, run_id="run-a")
    _write_manifest(root, "run-a", model="stub", split="test")
    cfg = _cfg(tmp_path, tmp_path / "results", "run-a")
    summary = _evaluate_normal(cfg, run_id="test")
    assert summary["n_volumes"] == 2


# ── Acceptance #4, layer 3: static — AST walk of module-level imports ──────────────────────────
def test_run_eval_module_level_imports_never_resolve_under_models_or_recon_engine() -> None:
    path = REPO / "scripts" / "run_eval.py"
    modules = _module_level_imports(path)
    hit = {m for m in modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)}
    assert not hit, f"scripts/run_eval.py imports forbidden modules at module scope: {hit}"


def test_eval_package_source_never_imports_models_or_recon_engine_except_legacy() -> None:
    eval_dir = REPO / "src" / "mri_ad" / "eval"
    violations: dict[str, set[str]] = {}
    for path in sorted(eval_dir.glob("*.py")):
        if path.name == "legacy.py":
            continue  # the single allowlisted exception (Spec 004 plan)
        modules = _imported_modules(path)
        hit = {m for m in modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)}
        if hit:
            violations[path.name] = hit
    assert not violations, f"eval/ imports forbidden modules: {violations}"


# ── Acceptance #8: make report has zero ML imports, and finishes fast ──────────────────────────
def test_importing_eval_report_leaves_no_ml_module_in_sys_modules() -> None:
    code = (
        "import sys\n"
        "import mri_ad.eval.report\n"
        f"leaked = {sorted(FORBIDDEN_ML_MODULES)!r}\n"
        "hit = [m for m in leaked if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(hit))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert "LEAKED:" not in out.stdout or out.stdout.strip() == "LEAKED:", out.stdout


def test_execing_run_report_as_a_module_leaves_no_ml_module_in_sys_modules() -> None:
    code = (
        "import runpy, sys\n"
        "runpy.run_path('scripts/run_report.py', run_name='not_main')\n"
        f"leaked = {sorted(FORBIDDEN_ML_MODULES)!r}\n"
        "hit = [m for m in leaked if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(hit))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "LEAKED:", out.stdout


# ── Spec 005: mri_ad.eval.matrix / run_arch_loss_matrix.py stay in the report-safe subgraph ────
def test_importing_eval_matrix_leaves_no_ml_module_in_sys_modules() -> None:
    code = (
        "import sys\n"
        "import mri_ad.eval.matrix\n"
        f"leaked = {sorted(FORBIDDEN_ML_MODULES)!r}\n"
        "hit = [m for m in leaked if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(hit))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "LEAKED:", out.stdout


def test_execing_run_arch_loss_matrix_as_a_module_leaves_no_ml_module_in_sys_modules() -> None:
    code = (
        "import runpy, sys\n"
        "runpy.run_path('scripts/run_arch_loss_matrix.py', run_name='not_main')\n"
        f"leaked = {sorted(FORBIDDEN_ML_MODULES)!r}\n"
        "hit = [m for m in leaked if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(hit))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "LEAKED:", out.stdout


# ── Spec 007: mri_ad.eval.{curves,paradigm} / run_paradigm_comparison.py stay report-safe ──────
def test_importing_eval_curves_leaves_no_ml_module_in_sys_modules() -> None:
    code = (
        "import sys\n"
        "import mri_ad.eval.curves\n"
        f"leaked = {sorted(FORBIDDEN_ML_MODULES)!r}\n"
        "hit = [m for m in leaked if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(hit))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "LEAKED:", out.stdout


def test_importing_eval_paradigm_leaves_no_ml_module_in_sys_modules() -> None:
    code = (
        "import sys\n"
        "import mri_ad.eval.paradigm\n"
        f"leaked = {sorted(FORBIDDEN_ML_MODULES)!r}\n"
        "hit = [m for m in leaked if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(hit))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "LEAKED:", out.stdout


def test_execing_run_paradigm_comparison_as_a_module_leaves_no_ml_module_in_sys_modules() -> None:
    code = (
        "import runpy, sys\n"
        "runpy.run_path('scripts/run_paradigm_comparison.py', run_name='not_main')\n"
        f"leaked = {sorted(FORBIDDEN_ML_MODULES)!r}\n"
        "hit = [m for m in leaked if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(hit))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "LEAKED:", out.stdout


def test_report_orchestrator_skips_missing_inputs_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``make report`` on a totally empty ``artifacts/`` tree must print skip-reasons, not raise."""
    from omegaconf import OmegaConf
    from scripts.run_report import main as run_report_main

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "deterministic": True,
            "model": {"name": "unetr"},
            "loss": {"name": "mse_ssim"},
            "eval": {
                "metrics_dir": str(tmp_path / "metrics"),
                "figures_dir": str(tmp_path / "figures"),
            },
            "matrix": {
                "metrics_root": str(tmp_path / "metrics"),
                "tables_dir": str(tmp_path / "tables"),
                "basename": "arch_loss_matrix",
                "cells": [],
            },
            "paradigm": {
                "classical_metrics_dir": str(tmp_path / "classical" / "metrics"),
                "metrics_root": str(tmp_path / "metrics"),
                "tables_dir": str(tmp_path / "tables"),
                "figures_dir": str(tmp_path / "figures"),
                "basename": "paradigm_comparison",
                "score_field": "flagged_fraction",
                "slice_threshold": None,
                "threshold_grid": [0.1],
                "columns": [
                    {
                        "key": "classical",
                        "paradigm": "Classical",
                        "localizes": False,
                        "na_reason": "not run",
                    }
                ],
            },
        }
    )

    class _NullRun:
        run_id = "test"

        def record(self, **kwargs):
            pass

    monkeypatch.setattr("scripts.run_report.RunLogger", lambda cfg: _NullRunContext())

    class _NullRunContext:
        def __enter__(self):
            return _NullRun()

        def __exit__(self, *args):
            return False

    run_report_main.__wrapped__(cfg)  # bypass hydra.main's CLI arg parsing
    out = capsys.readouterr().out
    assert "[skip: per-cell]" in out
    assert "[skip: arch x loss matrix]" not in out  # empty cells list is not an error
    assert "[paradigm]" in out or "[skip: paradigm comparison]" in out


def test_report_generation_finishes_well_under_the_two_minute_budget(tmp_path: Path) -> None:
    from mri_ad.eval.report import ReportGenerator

    rows = [
        {
            "volume_id": f"vol-{i}",
            "dice": 0.5,
            "iou": 0.4,
            "psnr_db_context_only": 20.0,
            "ssim_context_only": 0.6,
        }
        for i in range(250)
    ]
    aggregate = {
        "dice_mean": 0.5,
        "dice_std": 0.1,
        "iou_mean": 0.4,
        "iou_std": 0.1,
        "psnr_mean": 20.0,
        "psnr_std": 2.0,
        "ssim_mean": 0.6,
        "ssim_std": 0.05,
        "published_dice": 0.6255,
    }

    generator = ReportGenerator(tmp_path / "metrics", tmp_path / "figures")
    start = time.monotonic()
    generator.write_per_volume(rows)
    generator.write_aggregate(
        aggregate,
        mode="normal",
        model="unetr",
        loss="mse_ssim",
        split="test",
        n_volumes=250,
        split_hash="hash-test",
    )
    generator.plot_dice_distribution(rows)
    generator.write_summary_markdown()
    elapsed = time.monotonic() - start

    assert elapsed < 120.0
