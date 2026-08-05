"""Spec 006 acceptance 7 + R6: import-graph boundaries for ``classical/``.

Mirrors ``tests/test_model_boundary.py``. ``classical/`` must never import ``models/``,
``recon/``, or ``eval/`` — that keeps the classical baseline a genuinely independent control, not
something quietly coupled to the reconstruction pipeline it is being compared against. Separately,
``xgboost`` must never be required just to ``import mri_ad.classical`` — it fails to import on
machines missing ``libomp`` (R6), and only ``classifier.py`` may touch it, lazily.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLASSICAL_DIR = REPO / "src" / "mri_ad" / "classical"
RUN_CLASSICAL = REPO / "scripts" / "run_classical.py"
FORBIDDEN = {"mri_ad.models", "mri_ad.recon", "mri_ad.eval"}
RUN_CLASSICAL_ALLOWED_PREFIXES = (
    "hydra",
    "omegaconf",
    "mri_ad.classical",
    "mri_ad.data.split",
    "mri_ad.exceptions",
    "mri_ad.utils.run_logger",
    "mri_ad.utils.seed",
    "pathlib",
    "__future__",
)


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
    """Only imports at module scope (not inside a function) — mirrors run_classical.py's rule."""
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_classical_never_imports_downstream_layers() -> None:
    violations: dict[str, set[str]] = {}
    for path in sorted(CLASSICAL_DIR.rglob("*.py")):
        modules = _imported_modules(path)
        hit = {m for m in modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)}
        if hit:
            violations[path.name] = hit
    assert not violations, f"classical/ imports downstream layers: {violations}"


def test_only_classifier_module_imports_xgboost() -> None:
    violations = []
    for path in sorted(CLASSICAL_DIR.rglob("*.py")):
        if path.name == "classifier.py":
            continue
        modules = _imported_modules(path)
        if any(m == "xgboost" or m.startswith("xgboost.") for m in modules):
            violations.append(path.name)
    assert not violations, f"only classifier.py may import xgboost, found in: {violations}"


def test_classifier_module_imports_xgboost_lazily_not_at_module_scope() -> None:
    top_level = _module_level_imports(CLASSICAL_DIR / "classifier.py")
    assert "xgboost" not in top_level


def test_run_classical_module_level_imports_are_restricted() -> None:
    modules = _module_level_imports(RUN_CLASSICAL)
    violations = {
        m
        for m in modules
        if not any(m == p or m.startswith(p + ".") for p in RUN_CLASSICAL_ALLOWED_PREFIXES)
    }
    assert not violations, f"run_classical.py has disallowed module-level imports: {violations}"


def test_importing_classical_does_not_require_xgboost() -> None:
    """Poison `xgboost` in a subprocess's sys.modules; `import mri_ad.classical` must survive."""
    code = (
        "import sys\n"
        "sys.modules['xgboost'] = None\n"  # None in sys.modules -> ImportError if re-imported
        "import mri_ad.classical\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
